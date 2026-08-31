"""harbor ``BaseEnvironment`` adapter for xrlenv.

We subclass :class:`harbor.environments.docker.docker.DockerEnvironment`
rather than :class:`harbor.BaseEnvironment` directly, so the heavy
lifting harbor already does (docker compose orchestration, mounts,
env-var resolution, agent/verifier dir wiring, network policy,
keep-containers semantics, the ``preflight`` daemon check) is
inherited. We add:

- A constructor that accepts xrlenv-specific kwargs
  (``xrlenv_task_key``, ``xrlenv_group_id``, ``xrlenv_resources``,
  ``xrlenv_image_pin_mode``) without breaking harbor's signature.
  Today's LocalDocker mode records them on the instance; cluster
  mode (next slice) feeds them to the scheduler at the per-task
  invocation point.

- A subclass hook (``_xrlenv_route_command``) the cluster-mode
  follow-on overrides to redirect harbor's ``docker``/``docker
  compose`` invocations through xrlenv's gRPC stack to a
  scheduler-chosen node. LocalDocker mode is a pass-through:
  harbor's CLI calls run against whatever ``DOCKER_HOST`` resolves
  to (the local daemon by default), unchanged.

The architectural shape mirrors :class:`xrlenv.compat.docker_client.
XrlenvDockerClient`: subclass the upstream class, swap behavior at
specific seams, leave everything else inherited. The trade-off
documented over there applies here too — for arbitrary upstream
harness behavior we don't have to enumerate, inheritance keeps the
contract intact for free.

Why this lives in ``xrlenv_plugins/harbor/`` rather than
``xrlenv/compat/``:

- ``xrlenv.compat.docker_client`` adapts the *universal* Python
  Docker SDK; one shim serves every consumer that ever uses Docker
  via Python.
- This module adapts *one* RL framework's ``BaseEnvironment``
  Protocol; the next RL framework needs its own adapter (different
  Protocol, different methods). Per-framework plug-ins live under
  ``xrlenv_plugins/`` so xrlenv core doesn't accumulate
  framework-specific maintenance debt as the ecosystem grows.

See the package README for the canonical pattern other frameworks'
plug-ins should follow.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import shlex
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from xrlenv.client.client import Client
    from xrlenv.client.container_session import ClusterContainerSession

    from harbor import EnvironmentConfig, TrialPaths
    from harbor.models.task.config import NetworkPolicy

try:
    from harbor.environments.base import ExecResult
    from harbor.environments.capabilities import EnvironmentCapabilities
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.trial.paths import EnvironmentPaths
except ImportError as exc:  # pragma: no cover — surface a clear hint.
    raise ImportError(
        "harbor is not installed — xrlenv's harbor plug-in requires "
        "``harbor>=0.5``. Install via ``pip install 'xrlenv[terminal-bench-2]'`` "
        "or ``pip install 'harbor>=0.5'`` directly.",
    ) from exc

# ``network_mode`` / ``NetworkMode`` exist only in newer harbor; harbor 0.8.x
# expressed the same internet contract through ``EnvironmentConfig.allow_internet``
# (a plain ``bool`` defaulting to True). Import defensively so the plug-in loads
# on BOTH — keying a hard import off ``NetworkMode`` would re-raise the misleading
# "harbor is not installed" on every 0.8.x task. ``_network_mode_for_task``
# branches on whichever field the installed harbor actually provides.
try:
    from harbor.models.task.config import NetworkMode
except ImportError:  # pragma: no cover — older harbor without the network_mode model
    NetworkMode = None  # type: ignore[assignment, misc]

from xrlenv.compat.metadata import (
    LABEL_ARTIFACT_PATH,
    LABEL_DISPLAYED_NAME,
    current_rollout_metadata,
    metadata_to_labels,
)
from xrlenv.errors import XRLEnvError
from xrlenv_plugins.harbor import compose as _hc

LOGGER = logging.getLogger(__name__)

# Whole-stack footprint (scheduler reserve) main gets when the task declares no
# cpu/mem — matches the raw-container default budget so a compose main is reserved
# like a single-acquire container would be. Sidecars add on top (Q2).
_DEFAULT_MAIN_CPU = 2.0
_DEFAULT_MAIN_MEM_BYTES = 4 * 1024**3


class XrlenvHarborEnvironment(DockerEnvironment):
    """harbor ``BaseEnvironment`` whose container ops can be routed
    through xrlenv when the consumer opts in.

    Constructor accepts every kwarg ``DockerEnvironment`` does, plus:

    - ``xrlenv_task_key`` (str | None): anti-affinity grouping key.
      Cluster mode uses it to keep concurrent rollouts of the same
      task on different nodes when ``max_runs_per_task`` is set.
    - ``xrlenv_group_id`` (str | None): cancellation cohort. Cluster
      mode propagates a ``cancel_group`` to all containers sharing
      this id, so a stuck-cohort kill takes them all down together.
    - ``xrlenv_resources`` (ResourceSpec | None): scheduler input.
      Recorded on the instance for cluster-mode consumption; the
      cluster-mode follow-on feeds it to the bin-packer when
      picking a node. LocalDocker today doesn't apply it locally —
      harbor's own ``EnvironmentConfig`` (``cpus`` / ``memory_mb``)
      drives the local cgroup, so applying ours on top would
      double-cap. If the consumer wants tighter local limits, set
      them via harbor's ``EnvironmentConfig`` directly.
    - ``xrlenv_image_pin_mode``: spec-19 audit input. Recorded on
      the instance for observability; cluster mode threads it into
      the catalog's pin-resolver decision.
    - ``xrlenv_cap_add`` (list[str] | None): Linux capabilities to add
      to the task container (e.g. ``["NET_ADMIN", "SYS_ADMIN"]``).
    - ``xrlenv_devices`` (list[str] | None): host devices to expose
      (e.g. ``["/dev/loop0:/dev/loop0"]``).
    - ``xrlenv_privileged`` (bool): run the container ``--privileged``.

      These three are applied **only in cluster mode** (forwarded to
      ``acquire_container``); the control plane's ``KwargsPolicy`` is the
      gate — ``cap_add`` is allowed by default, ``privileged`` needs the
      operator's ``allow_privileged`` opt-in in ``nodes.yaml``. LocalDocker
      records but does not apply them (harbor's compose path owns the local
      daemon). Infra/sysadmin tasks that need ``NET_ADMIN`` (iptables),
      ``SYS_ADMIN`` (ip netns), or ``SYS_PTRACE`` (debuggers) set
      ``xrlenv_cap_add`` to run under the cluster without a privileged host.

    LocalDocker (today): the kwargs are recorded on the instance for
    observability; everything else flows through harbor's inherited
    docker-compose path against the local daemon. The class is a
    drop-in for harbor.DockerEnvironment with no behavior change.

    Cluster (next slice): override ``_xrlenv_route_command`` to
    redirect harbor's docker/docker-compose subprocess invocations
    through xrlenv's gRPC stack to a scheduler-chosen node. The
    seam is in place; the routing implementation lands when the
    cluster-mode ContainerControl ships.
    """

    _XRLENV_KWARGS = (
        "xrlenv_task_key", "xrlenv_group_id", "xrlenv_resources",
        "xrlenv_image_pin_mode", "xrlenv_owner_id", "xrlenv_project_id",
        "xrlenv_run_id",
        "xrlenv_cap_add", "xrlenv_devices", "xrlenv_privileged",
        "xrlenv_cpu_pinning",
        "xrlenv_cpu_multiplier", "xrlenv_mem_multiplier",
        "xrlenv_image_template",
    )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        keep_containers: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Pop xrlenv-only kwargs before forwarding to harbor.
        # DockerEnvironment.__init__ rejects unknowns at the
        # super().__init__() chain.
        self._xrlenv_kwargs: dict[str, Any] = {
            k: kwargs.pop(k) for k in self._XRLENV_KWARGS if k in kwargs
        }
        # The process-global XRLENV_HARBOR_IMAGE_TEMPLATE env var was removed in
        # favour of the scoped `xrlenv_image_template` kwarg (a per-run handoff
        # via EnvironmentConfig(kwargs=...)). Fail loud if a caller still relies
        # on the old env var with no kwarg override — silently ignoring it would
        # resolve the WRONG image (fall through to docker_image / hb__<name>).
        if (
            os.environ.get("XRLENV_HARBOR_IMAGE_TEMPLATE")
            and "xrlenv_image_template" not in self._xrlenv_kwargs
        ):
            raise RuntimeError(
                "XRLENV_HARBOR_IMAGE_TEMPLATE is no longer read (removed in the "
                "harbor image-template refactor) and was being silently ignored "
                "— which resolves the wrong image. Pass it as the "
                "`xrlenv_image_template` kwarg on EnvironmentConfig(kwargs=...) "
                "instead, or unset the env var."
            )
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            keep_containers,
            *args,
            **kwargs,
        )

    @property
    def xrlenv_kwargs(self) -> dict[str, Any]:
        """The xrlenv-specific kwargs this instance was constructed
        with. Read-only snapshot; useful for scheduler integrations
        that want to inspect routing intent post-construction."""
        return dict(self._xrlenv_kwargs)

    def _xrlenv_route_command(
        self, command: list[str], *, kind: str,
    ) -> list[str]:
        """Hook for cluster-mode routing.

        Called by the cluster-mode follow-on (next slice) at every
        ``docker``/``docker compose`` invocation site. Default
        implementation is a pass-through — LocalDocker mode runs
        the command unchanged against the local daemon.

        Cluster-mode override translates ``["docker", "compose",
        "-f", path, "up", "-d", ...]`` into a gRPC dispatch to the
        chosen node's node-agent, which executes the same command
        against its own local daemon. Same observable behavior, but
        on the chosen node.

        Args:
            command: docker/docker-compose argv as harbor would
              shell out.
            kind: ``"docker"`` or ``"docker-compose"`` — caller
              hint so the cluster-mode routing can pick the right
              wire shape.

        Returns:
            The (possibly rewritten) argv to actually invoke. The
            LocalDocker default returns ``command`` unchanged.
        """
        del kind  # cluster-mode override uses this; LocalDocker doesn't
        return command


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.C.1 — Cluster-routed harbor Environment.
# ──────────────────────────────────────────────────────────────────────────────


def _keepalive_argv(task_env: Mapping[str, Any] | dict, systemd_init: bool) -> tuple[list[str] | None, list[str]]:
    """(entrypoint, command) the container is started with. Default: the image's own entrypoint +
    ``sleep infinity`` (or ``/sbin/init`` under ``XRLENV_SYSTEMD_INIT``). With the task marker
    ``XRLENV_KEEPALIVE_ENTRYPOINT`` truthy, override the image's ENTRYPOINT so the keep-alive is
    exec'd directly: ``entrypoint=["sleep"], command=["infinity"]``."""
    if systemd_init:
        return None, ["/sbin/init"]
    flag = str(task_env.get("XRLENV_KEEPALIVE_ENTRYPOINT", "")).strip().lower() in ("1", "true", "yes", "on")
    if flag:
        return ["sleep"], ["infinity"]
    return None, ["sleep", "infinity"]


def _sanitize_image_tag(name: str) -> str:
    """Mirror harbor's ``_sanitize_docker_image_name`` for the
    ``hb__<environment_name>`` tag the build script writes on each
    node. Lowercase + replace anything outside ``[a-z0-9._-]`` with
    ``-``."""
    return re.sub(r"[^a-z0-9._-]", "-", name.lower())


def _sanitize_container_name(name: str) -> str:
    """Container names are stricter than image tags — no leading ``-``,
    only ``[a-zA-Z0-9_.-]``. Mirror harbor's
    ``_sanitize_docker_compose_project_name`` shape."""
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", name)
    return sanitized.lstrip("-_.") or "harbor-cluster-session"


_DEFAULT_GRPC_PORT = 50051
_DEFAULT_EXEC_TIMEOUT_S = 1800.0
# Sysbox file transfers (exec `tar|base64`) are I/O, not long-running commands,
# so they must NOT inherit the 30-min agent-exec default. The upload path already
# sizes its own timeout; the download path (log-salvage / artifact pull, run at
# TEARDOWN) previously reused _DEFAULT_EXEC_TIMEOUT_S — so on a degraded/wedged
# sysbox node a post-timeout `docker exec` to tar the logs blocked the client for
# ~1860 s (timeout + the transport's exec deadline margin), hanging the trial's
# finalization for half an hour and freezing the whole sweep (2026-07-08 cap=8
# incident: tw_526185). A teardown transfer must give up in minutes; a legit
# small log/artifact tar completes in seconds.
_SYSBOX_DOWNLOAD_TIMEOUT_S = 180.0

# Fail-fast admission-queue budget for a single acquire attempt. Harbor wraps
# environment setup (this acquire + the agent install that follows it) in a hard
# ``wait_for`` (its ``_AGENT_SETUP_TIMEOUT_SEC``, 360 s default). Under high
# request concurrency against a capacity-CAPPED runtime — e.g. the per-node
# sysbox cap (nodes.yaml ``max_concurrent_by_runtime``) — the admission queue can
# legitimately hold an acquire longer than that window: with cap=4 the last of N
# queued sysbox tasks waits ~(N-4)/4 x task-time for a slot. If the acquire
# out-waits harbor's window, harbor CANCELS it → a non-retriable ``TimeoutError``
# fails the trial. That's the mismatch that forces callers to hand-lower their
# concurrency — which they should not have to know about.
#
# Instead we bound the queue wait BELOW harbor's setup window so an at-cap acquire
# returns ``CapacityExhausted`` (which the trial queue is configured to retry) with
# headroom left for the post-acquire image-pull / container-start / agent-upload.
# Because the acquire is harbor's FIRST setup step, the retry re-attempts ONLY the
# acquire — no container/agent setup is wasted — and the task still runs exactly
# once (the attempt that finally gets a slot), so a waited-then-passed trial is a
# clean pass in the tally, not a re-roll. This lets a consumer request any
# concurrency while xrlenv transparently paces the capped runtime.
#
# Env-tunable for operators who raise harbor's setup timeout via
# ``agent_setup_timeout_multiplier`` (keep this comfortably below the resulting
# window so CapacityExhausted still beats the cancel).
_ACQUIRE_QUEUE_TIMEOUT_S = float(
    os.environ.get("XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S", "") or 240.0
)

# Max base64 chars embedded in a single exec command for the sysbox exec-based
# upload path. Well under ARG_MAX (~2 MiB) with margin for the surrounding
# script + env; harbor's solution/tests uploads are KB, so they fit in one exec.
_SYSBOX_XFER_CHUNK = 256 * 1024


def _client_from_env() -> Client:
    """Construct an ``xrlenv.Client`` from the same env-var protocol
    the docker-py drop-in uses (``XRLENV_GRPC_HOST`` / ``_PORT`` /
    ``_CONSUMER_TOKEN`` / ``_GRPC_SECURE``). Raises ``RuntimeError``
    with a clear hint if ``XRLENV_GRPC_HOST`` is unset — cluster
    mode is opt-in by env, so a missing host means the operator
    forgot to set it before launching the harness."""
    host = os.environ.get("XRLENV_GRPC_HOST")
    if not host:
        raise RuntimeError(
            "XrlenvHarborEnvironmentCluster requires XRLENV_GRPC_HOST "
            "to point at the control plane (e.g. ``XRLENV_GRPC_HOST="
            "127.0.0.1`` for a local ``xrlenv up`` daemon). Set "
            "XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN before launching "
            "the harness, or pick the local-mode "
            "``xrlenv_plugins.harbor:XrlenvHarborEnvironment`` "
            "import_path instead.",
        )

    port_raw = os.environ.get("XRLENV_GRPC_PORT")
    port = _DEFAULT_GRPC_PORT
    if port_raw:
        try:
            port = int(port_raw)
        except ValueError:
            LOGGER.warning(
                "XRLENV_GRPC_PORT=%r is not an int; falling back to %d",
                port_raw, _DEFAULT_GRPC_PORT,
            )

    token = os.environ.get("XRLENV_CONSUMER_TOKEN")
    secure_raw = os.environ.get("XRLENV_GRPC_SECURE", "")
    secure = secure_raw.strip().lower() in ("true", "1", "yes", "on")

    from xrlenv.client.client import Client as _Client  # avoid cycle

    return _Client.grpc(host=host, port=port, token=token, secure=secure)


def _relax_modes(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    """Tar ``filter`` that makes an entry world read/exec (dirs world rwx) and
    root-owned — used ONLY when uploading into a **sysbox** container.

    Under sysbox-runc the container's rootfs is userns-idmapped, but docker's
    ``put_archive`` does NOT apply that id-shift to the files it extracts: they
    land owned by an unmapped uid (shown as ``65534``/nobody), and container-root
    then can't ``chmod`` them (no ``CAP_FOWNER`` over an unmapped owner). So
    harbor's own ``chmod +x /solution/solve.sh`` silently EPERMs and the oracle
    exec dies with ``126 Permission denied``. The mode BITS, however, are honored
    by extraction independent of ownership — so if the tar entry already carries
    the exec bit, root can run the (65534-owned) script fine. We set the bits up
    front here so no in-container chmod is needed. (Verified on the sysbox node:
    a 0755 tar entry lands ``-rwxr-xr-x 65534 65534`` and executes.)
    """
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o777 if ti.isdir() else (ti.mode | 0o755)
    return ti


def _tar_one_file(
    source: Path, arcname: str, *, world_accessible: bool = False,
) -> bytes:
    """Wrap one local file in a tarball whose single entry is named
    ``arcname``. The cluster's ``put_archive`` extracts entries
    relative to the target_dir, so ``arcname`` becomes the leaf
    inside the container. ``world_accessible`` (sysbox path) relaxes the
    entry's modes so an un-chmod-able uploaded file is still runnable — see
    :func:`_relax_modes`."""
    buf = io.BytesIO()
    filt = _relax_modes if world_accessible else None
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(str(source), arcname=arcname, filter=filt)
    return buf.getvalue()


def _tar_dir_contents(source_dir: Path, *, world_accessible: bool = False) -> bytes:
    """Wrap a local directory's *contents* (not the directory itself)
    in a tarball. Mirrors harbor's ``docker compose cp src/. main:dst``
    semantics: dst gets the children of src, not a nested ``src``
    sub-dir. ``world_accessible`` (sysbox path) relaxes entry modes — see
    :func:`_relax_modes`."""
    buf = io.BytesIO()
    filt = _relax_modes if world_accessible else None
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for child in sorted(Path(source_dir).iterdir()):
            tf.add(str(child), arcname=child.name, filter=filt)
    return buf.getvalue()


def _untar_one_file(tarball: bytes, target: Path) -> None:
    """Extract a single-entry tarball into ``target``. The cluster's
    ``get_archive(/path/to/file)`` returns a tar with one entry named
    ``file`` — we read it and write to ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if not members:
            raise RuntimeError(
                "get_archive returned no file entries — did the source "
                "path exist inside the container?",
            )
        if len(members) > 1:
            raise RuntimeError(
                "download_file expected one tar entry, got "
                f"{len(members)}: {[m.name for m in members]}",
            )
        member = members[0]
        extracted = tf.extractfile(member)
        if extracted is None:
            raise RuntimeError(
                f"tar member {member.name!r} could not be read",
            )
        target.write_bytes(extracted.read())


def _untar_dir_contents(tarball: bytes, target_dir: Path) -> None:
    """Extract a multi-entry tarball under ``target_dir``. The
    cluster's ``get_archive(/some/dir)`` returns a tar whose entries
    are rooted at ``dir`` — we strip that one leading component so
    ``target_dir`` mirrors harbor's ``docker compose cp main:src/. dst``
    semantics (children of src land directly under dst)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r") as tf:
        members = tf.getmembers()
        if not members:
            return
        # Docker's get_archive emits entries like ``dirname/...``; the
        # first path component is the source dir's basename. Strip it.
        root = members[0].name.split("/", 1)[0]
        for m in members:
            parts = m.name.split("/", 1)
            if parts[0] == root:
                m.name = parts[1] if len(parts) > 1 else "."
            if m.name == "" or m.name == ".":
                continue
            tf.extract(m, path=str(target_dir), filter="data")


class XrlenvHarborEnvironmentCluster(XrlenvHarborEnvironment):
    """harbor ``BaseEnvironment`` whose container ops route through
    the xrlenv cluster instead of local ``docker compose`` + ``docker
    cp``.

    harbor users opt in by setting ``environment.import_path:
    xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`` in their
    ``job.yaml``, exactly the way they'd pick ``e2b``, ``modal``, or
    ``daytona``. The local-mode :class:`XrlenvHarborEnvironment` is
    untouched — operators choose between the two by ``import_path``,
    no flag, no auto-detect.

    **Cluster opt-in via env:**

    - ``XRLENV_GRPC_HOST`` (string, required) — control-plane host.
    - ``XRLENV_GRPC_PORT`` (int, default 50051).
    - ``XRLENV_CONSUMER_TOKEN`` (string) — bearer token from
      ``xrlenv tokens issue consumer``. Required when the
      control plane runs with auth.
    - ``XRLENV_GRPC_SECURE`` (``"true"`` / ``"1"`` / ``"yes"`` /
      ``"on"``; default false) — TLS channel.

    **Image distribution (P1.7.C.1 staged):**

    Images are pre-built on each node via the per-benchmark build
    flow (e.g. ``xrlenv_plugins/benchmarks/terminal_bench_2_1/
    build_cache.py`` + ``build_plan_gen.py``). The image tag the cluster
    looks up is either ``task_env_config.docker_image`` (when the
    upstream task ships a prebuilt) or ``hb__<environment_name>``
    (the harbor convention). If the chosen node doesn't have the
    image, ``acquire_container`` fails fast with ``ImageNotFound``
    pointing at the build script.

    Real build-on-acquire (``HarborImageBuilder`` + acquire→build
    →re-acquire fallback) is **P1.7.C.2** — out of scope for this
    slice.

    **Single-service only (P1.7.C.1 staged):**

    Multi-service compose tasks (a few harbor tasks attach a ``db`` /
    ``redis`` helper) are out of scope for this slice. The four
    overridden methods (``start`` / ``stop`` / ``exec`` /
    ``{up,down}load_*``) all assume a single ``main`` service. The
    cluster ``acquire_container`` returns one container; multi-
    service support requires either multiple acquires + a private
    network, or compose-on-the-node-side, both deferred to a
    follow-on slice.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        keep_containers: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            keep_containers,
            *args,
            **kwargs,
        )
        self._xrlenv_client: Client | None = None
        self._xrlenv_session: ClusterContainerSession | None = None
        # Set in start() from the task's container_runtime marker. When the task
        # runs under sysbox, file transfer routes through the exec-based path
        # (_sysbox_upload_tarball / _sysbox_download_tarball) instead of docker
        # put_archive/get_archive, which re-touch the idmapped /etc/resolv.conf
        # bind and 500 with ``openat etc/resolv.conf: file exists``. Uploaded
        # files also carry world r-x in the tar (see _relax_modes).
        self._sysbox_upload = False
        self._xfer_seq = 0  # unique-ifies the in-container staging file

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        """Override DockerEnvironment's capabilities for cluster mode.

        The legacy ``is_mounted`` / ``can_disable_internet`` boolean
        properties are silently shadowed by ``DockerEnvironment``'s
        own ``capabilities`` override via MRO — so we have to
        override the new API directly. Without this, harbor's
        ``Verifier.verify`` reads ``capabilities.mounted=True`` from
        the parent and skips the post-test ``download_dir`` call,
        leaving the consumer's ``trial/verifier/`` empty and
        producing ``RewardFileNotFoundError``.

        - ``mounted=False``: cluster never bind-mounts host paths;
          consumer host ≠ node host. Forces harbor to use the
          post-trial download_dir branch.
        - ``disable_internet=True``: the cluster honors a task's
          ``network_mode="no-network"`` by acquiring its container with
          ``--network none`` (loopback-only — localhost still works,
          external internet blocked). The raw acquire path applies
          ``network_mode`` directly to ``docker run`` (see
          ``node/raw_container.py``), so this is real enforcement, not
          a silent downgrade. ``start`` (below) sets it from the task's
          resolved ``network_mode`` (see ``_network_mode_for_task``); the
          default ``public`` leaves the bridge (internet on) untouched.
        - ``docker_compose``: **True for a multi-service task**, else
          False. A task whose ``docker-compose.yaml`` declares more
          than one service routes through ``acquire_compose_project``
          (compose-on-the-node); a single-service / compose-less task
          keeps the raw ``acquire_container`` path. Computed from the
          task compose via ``_multi_service_compose`` (§4b).
        - ``disable_internet=True``: honored on the single-container
          path via post-install ``apply_egress``. **Caveat for a
          multi-service task:** ``apply_egress`` is not yet supported
          for compose projects (it would restrict only ``main`` and
          leave sidecars open, so it raises) — an *offline* compose
          task therefore fails loud at post-install egress until
          project-network egress lands. None of the onboarded compose
          tasks are offline; the capability stays truthful for the
          single-container path.
        - ``windows=False``: cluster's acquire path is Linux-only.
        - ``gpus=False``: GPU scheduling isn't yet plumbed through
          the cluster acquire path.
        """
        can_egress = self._can_enforce_egress()
        return EnvironmentCapabilities(
            gpus=False,
            # Gated on _can_enforce_egress (like the allowlist flags): harbor's
            # NO_NETWORK validation gates on disable_internet, so advertising it
            # only when we can actually seal egress makes harbor fail-closed-
            # REJECT an offline compose / sysbox / privileged task at validation
            # rather than accept it and then run open (the compose start() path
            # cannot apply a startup baseline). A 0.8-era ``allow_internet=false``
            # task is mapped by harbor's TaskConfig deprecation shim to a
            # NO_NETWORK baseline, which our start()/_apply_baseline_network_policy
            # then enforces on the single-container runc path.
            disable_internet=can_egress,
            windows=False,
            mounted=False,
            docker_compose=self._multi_service_compose() is not None,
            # harbor 0.20 native per-phase network policy: harbor's Trial drives
            # ``set_network_policy()`` at each phase boundary and we enforce it
            # via the spec-07 ``apply_egress`` iptables primitive
            # (:meth:`_apply_network_policy`). This SUPERSEDES the old
            # open-setup→post-install ``apply_egress`` hack — harbor owns the
            # *when*, we own only the *enforcement*. Gated by
            # :meth:`_can_enforce_egress` so harbor fail-closed-REJECTS an
            # offline compose / sysbox / privileged task at trial validation
            # rather than under-enforcing it mid-run. CIDR/IPv4 only — hostname,
            # wildcard, and IPv6 allowlisting stay ``False`` so harbor rejects
            # such a task instead of silently leaving it open (no DNS-name
            # allowlist primitive in ``apply_egress`` yet).
            dynamic_network_policy=can_egress,
            network_allowlist=can_egress,
            network_allowlist_ipv4_addresses=can_egress,
            network_allowlist_ipv4_cidrs=can_egress,
        )

    def _can_enforce_egress(self) -> bool:
        """True iff the spec-07 ``apply_egress`` primitive can enforce a network
        policy for THIS task.

        ``apply_egress`` installs iptables in the container's netns via
        ``nsenter`` and is only a trusted boundary when the workload cannot flush
        those rules. So it is refused for a **compose project** (it would restrict
        only ``main`` and leave sidecars open — see
        :meth:`ClusterContainerSession.apply_egress`), and for any
        **egress-escapable** footprint: a non-``runc`` runtime whose inner root
        owns its netns (sysbox), ``--privileged``, or ``CAP_NET_ADMIN`` (mirrors
        :func:`xrlenv.backends.egress.container_can_escape_egress`).

        Gates the ``dynamic_network_policy`` / ``network_allowlist`` capabilities
        so harbor **rejects** an offline task it can't seal at validation time,
        instead of us silently under-enforcing.
        """
        if self._multi_service_compose() is not None:
            return False
        task_env = getattr(self.task_env_config, "env", None) or {}
        runtime = str(task_env.get("XRLENV_CONTAINER_RUNTIME", "")).strip() or "runc"
        if runtime != "runc":
            return False
        xk = self._xrlenv_kwargs
        if bool(xk.get("xrlenv_privileged", False)):
            return False
        cap_add = xk.get("xrlenv_cap_add") or []
        return not any(
            str(c).strip().upper().removeprefix("CAP_") == "NET_ADMIN"
            for c in cap_add
        )

    def _network_mode_for_task(self) -> str | None:
        """The ``--network`` mode for this task's container, derived from the
        task's internet contract — version-tolerant across harbor releases.

        Newer harbor exposes ``EnvironmentConfig.network_mode`` (a
        ``NetworkMode`` enum, default ``public``) and leaves the legacy
        ``allow_internet`` flag deprecated/``None``. There, read
        ``network_mode``: ``no-network`` → ``"none"`` (loopback-only — external
        internet blocked, localhost intact); ``public`` (default) and
        ``allowlist`` → ``None`` (default bridge, internet on). Keying off the
        deprecated ``allow_internet`` instead would read ``None`` and silently
        treat every internet-on task as offline.

        harbor 0.8.x has no ``network_mode`` model; ``allow_internet`` (a plain
        ``bool``, default ``True``) is the live field. There, fall back to
        ``"none"`` only when it is explicitly disabled.

        Either way the raw acquire path applies the result straight to
        ``docker run`` (``node/raw_container.py``) — the real enforcement behind
        the ``capabilities.disable_internet=True`` advertised above.

        ``allowlist`` is *not* yet enforced as an allowlist here — raw
        ``docker run`` has no host-allowlist primitive, so it falls back to the
        full bridge rather than refusing the task. Tightening it to a real
        egress allowlist is tracked separately; until then it errs open, never
        silently closed.
        """
        cfg = self.task_env_config
        network_mode = getattr(cfg, "network_mode", None)
        if NetworkMode is not None and network_mode is not None:
            # Newer harbor: network_mode is the authoritative field.
            return "none" if network_mode == NetworkMode.NO_NETWORK else None
        # harbor 0.8.x: no network_mode model — allow_internet (default True) is
        # the live field. Block external internet only when it's disabled.
        return "none" if not getattr(cfg, "allow_internet", True) else None

    def task_internet_disabled(self) -> bool:
        """True if this task's contract forbids open internet (offline task).

        The version-tolerant offline decision, reused from
        :meth:`_network_mode_for_task` (``"none"`` ⇔ offline) so the
        network/allow_internet reading lives in exactly one place. Since the
        cluster env always acquires OPEN now (install needs network), the
        consumer's Trial calls this *after* install to decide whether to
        restrict the container's egress via :meth:`apply_egress`. ``True`` →
        restrict; ``False`` → leave the bridge open.
        """
        return self._network_mode_for_task() == "none"

    def _resolve_image_ref(self) -> str:
        """Pick the image ref the node will acquire, by precedence:

        1. ``xrlenv_image_template`` (kwarg), if set — a ``str.format``
           template with ``{task_id}`` (the task directory name, e.g. ``88`` for
           ``Harbor-Dataset/88``) and ``{environment_name}`` fields. This is how a
           benchmark whose images live in a private registry under a *derived*
           name — e.g. seta-env, pushed as
           ``<host>:5011/seta-env/{task_id}:main`` — points the cluster at them
           without a per-task ``docker_image`` and without subclassing this class.
           Passed per-run by the sweep driver via ``EnvironmentConfig(kwargs=...)``
           (a scoped handoff — NOT a process-global env var).
        2. ``task_env_config.docker_image`` — an upstream-published prebuilt
           (e.g. terminal-bench-2's ``alexgshaw/<task>:<rev>``), or a private-
           registry ref written by a repin step (LHTB's ``build_cache --stage repin``).
        3. ``hb__<environment_name>`` — the locally-built harbor convention.
        """
        template = self._xrlenv_kwargs.get("xrlenv_image_template")
        if template:
            return str(template).format(
                task_id=Path(self.environment_dir).parent.name,
                environment_name=self.environment_name,
            )
        if self.task_env_config.docker_image:
            # expandvars so a task.toml can carry a HOST-AGNOSTIC private-registry ref
            # like "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}/lhtb/
            # <task>:main", resolved from the consumer's .env at acquire — so the cache
            # never bakes a registry host and survives a CP/registry IP change (GUIDELINE
            # §5.3.1). No-op for a literal docker.io ref.
            ref = os.path.expandvars(str(self.task_env_config.docker_image))
            if "${" in ref:
                raise XRLEnvError(
                    f"unresolved registry placeholder in docker_image {ref!r} — export "
                    "XRLENV_PRIVATE_REGISTRY_HOST / XRLENV_PRIVATE_REGISTRY_PORT (source "
                    ".env) before the run.",
                )
            return ref
        return _sanitize_image_tag(f"hb__{self.environment_name}")

    # ── Multi-service compose (§4b) ──────────────────────────────────────────

    def _task_compose_path(self) -> Path | None:
        """The task's own ``docker-compose.yaml`` (or ``.yml``) in the
        environment dir, or ``None`` — mirrors harbor's
        ``_environment_docker_compose_path`` discovery. Tolerates a not-yet-wired
        ``environment_dir`` (``capabilities`` may be probed before harbor's
        ``__init__`` has run) → ``None`` (compose-less)."""
        env_dir = getattr(self, "environment_dir", None)
        if not env_dir:
            return None
        for name in ("docker-compose.yaml", "docker-compose.yml"):
            path = Path(env_dir) / name
            if path.is_file():
                return path
        return None

    def _multi_service_compose(self) -> dict[str, Any] | None:
        """The parsed task compose **iff** it declares more than one service
        (the predicate that routes a task onto the compose-project path), else
        ``None`` → the raw ``acquire_container`` path. Read each call (cheap);
        harbor invokes ``capabilities`` + ``start`` a handful of times per trial."""
        path = self._task_compose_path()
        if path is None:
            return None
        try:
            doc = _hc.load_compose(path.read_text(errors="replace"))
        except OSError:
            return None
        return doc if _hc.is_multi_service(doc) else None

    def _image_namespace_tag(self, main_ref: str | None = None) -> tuple[str | None, str]:
        """``(namespace, tag)`` for sub-dir sidecar refs, by precedence:

        1. an explicit ``xrlenv_image_template`` kwarg — split on the
           literal ``{task_id}`` (``<prefix>/{task_id}:<tag>`` → ``namespace=<prefix>``);
        2. **derived from the resolved main image ref** (``main_ref``) — a *repinned*
           registry ref like ``<host:port>/lhtb/<task>:main`` yields namespace
           ``<host:port>/lhtb``. This lets a multi-service compose task's sidecars resolve
           **without** the template — which is why chess-mate runs in the ordinary sweep.
           (The template also overrides the *main* image, so it can't be set for a whole
           sweep; deriving from the already-repinned main ref sidesteps that entirely.)
           Non-compose tasks never reach this — they have no build sidecars.

        Returns ``(None, "main")`` when neither yields a private-registry namespace;
        :func:`compose.assemble_project` then fails loud only if the task actually has
        sub-dir build services. The template split is on the placeholder, never the host's
        ``:``, so a registry ``host:port`` prefix stays unambiguous."""
        template = self._xrlenv_kwargs.get("xrlenv_image_template")
        if template and "{task_id}" in str(template):
            prefix, _, suffix = str(template).partition("{task_id}")
            namespace = prefix.rstrip("/") or None
            tag = "main"
            if suffix.startswith(":"):
                candidate = suffix[1:]
                # The corpus uses ':main'. Ignore a path separator or an unresolved
                # placeholder (e.g. '{environment_name}') → fall back to main.
                if candidate and "/" not in candidate and "{" not in candidate:
                    tag = candidate
            return namespace, tag
        return _hc.registry_namespace_and_tag(main_ref)

    _COMPOSE_INCOMPATIBLE_MARKERS = (
        "XRLENV_SYSTEMD_INIT",
        "XRLENV_INNER_DOCKERD",
        "XRLENV_INSTALL_DOCKERD",
    )

    def _reject_compose_incompatible_markers(self, task_id: str) -> None:
        """A multi-service compose task runs under **runc** (the CP policy gate
        makes it safe without sysbox, plan §3.5), so the single-container substrate
        markers are **not** honored on the compose path: a non-``runc``
        ``XRLENV_CONTAINER_RUNTIME`` (sysbox) and the ``XRLENV_SYSTEMD_INIT`` /
        ``XRLENV_INNER_DOCKERD`` / ``XRLENV_INSTALL_DOCKERD`` companions. A task that
        sets both a multi-service compose AND one of these is a contradiction —
        **reject it loudly** rather than silently drop the marker (which would run
        the task in a substrate it wasn't authored for)."""
        task_env = getattr(self.task_env_config, "env", None) or {}

        def _flag(key: str) -> bool:
            return str(task_env.get(key, "")).strip().lower() in (
                "1", "true", "yes", "on",
            )

        bad: list[str] = []
        runtime = str(task_env.get("XRLENV_CONTAINER_RUNTIME", "")).strip() or None
        if runtime and runtime != "runc":
            bad.append(f"XRLENV_CONTAINER_RUNTIME={runtime}")
        bad.extend(key for key in self._COMPOSE_INCOMPATIBLE_MARKERS if _flag(key))
        if bad:
            raise XRLEnvError(
                f"multi-service compose task {task_id!r} also sets substrate "
                f"marker(s) {bad} — a compose project runs under runc and cannot "
                f"honor them. Remove the marker(s), or make the task single-service "
                f"if it genuinely needs that substrate.",
            )

    # ── Shared acquire helpers (single + compose paths) ──────────────────────

    def _acquire_labels(self) -> dict[str, str]:
        """The docker labels for a cluster acquire (raw or compose). Default-
        populates ``xrlenv.rollout.artifact_path`` (per-trial dir) +
        ``xrlenv.rollout.displayed_name`` (harbor trial id) so the admin
        ``/rollouts/raw`` view shows a recognisable per-trial row without the
        operator wrapping each trial in ``rollout_metadata(...)``; an explicit
        ``rollout_metadata`` block still takes precedence (read last)."""
        labels: dict[str, str] = {
            "harbor.session_id": self.session_id,
            "harbor.environment_name": self.environment_name,
            LABEL_ARTIFACT_PATH: str(self.trial_paths.trial_dir),
            LABEL_DISPLAYED_NAME: self.session_id,
        }
        labels.update(metadata_to_labels(current_rollout_metadata()))
        return labels

    def _effective_cpu_mem_limits(self) -> tuple[float | None, int | None]:
        """The task's effective ``(cpu_limit_cores, mem_limit_bytes)`` — harbor's
        canonical ``_effective_cpus`` / ``_effective_memory_mb`` (already merging
        ``override_*`` + gated by the enforcement policy) scaled by the optional
        per-task ``xrlenv_cpu_multiplier`` / ``xrlenv_mem_multiplier`` (default 1.0).
        ``None`` where the task declares nothing → the caller omits the cap (raw
        path) or reserves the default main footprint (compose path)."""
        xk = self._xrlenv_kwargs
        cpus = self._effective_cpus
        mem_mb = self._effective_memory_mb
        cpu_mult = float(xk.get("xrlenv_cpu_multiplier", 1.0) or 1.0)
        mem_mult = float(xk.get("xrlenv_mem_multiplier", 1.0) or 1.0)
        cpu_limit = float(cpus) * cpu_mult if cpus else None
        mem_limit_bytes = int(mem_mb * mem_mult) * 1024 * 1024 if mem_mb else None
        return cpu_limit, mem_limit_bytes

    async def _setup_cluster_log_dirs(self) -> None:
        """Create + world-write the in-container ``/logs/{agent,verifier,artifacts}``
        dirs harbor relies on. Local mode gets these from bind mounts; cluster mode
        (``mounted=False``) has none, so without this harbor's first agent write
        exits 1 and the trial fails on the post-run ``download_dir(/logs/verifier)``.
        These are empty write-targets during the agent run — same visibility as
        harbor's local mode. Runs against ``main`` on both the raw and compose
        paths."""
        log_dirs = " ".join(str(d) for d in (
            EnvironmentPaths.agent_dir,
            EnvironmentPaths.verifier_dir,
            EnvironmentPaths.artifacts_dir,
        ))
        setup_result = await self.exec(
            f"mkdir -p {log_dirs} && chmod 777 {log_dirs}",
            user="root",
        )
        if setup_result.return_code != 0:
            raise RuntimeError(
                f"cluster harbor start: log-dir setup failed "
                f"(exit={setup_result.return_code}): "
                f"stderr={setup_result.stderr!r}",
            )

    async def _start_compose_project(self, doc: dict[str, Any]) -> None:
        """Bring up a multi-service compose PROJECT on the cluster (§4b).

        Assembles the image-ref-only compose (main-synthesis + build/local-tag
        rewrite + per-sidecar caps), computes the whole-stack footprint, and
        acquires via ``acquire_compose_project``. The returned
        ``ClusterComposeSession`` targets ``main``, so ``exec`` / file transfer /
        ``stop`` are inherited unchanged."""
        task_id = Path(self.environment_dir).parent.name
        # Compose runs under runc — reject any single-container substrate marker.
        self._reject_compose_incompatible_markers(task_id)

        main_ref = self._resolve_image_ref()
        # Derive the sidecar namespace from the (repinned) main ref so the compose
        # resolves without an image-template kwarg — chess-mate runs in the sweep.
        namespace, tag = self._image_namespace_tag(main_ref)
        try:
            rewritten, images = _hc.assemble_project(
                doc, task_id=task_id, main_ref=main_ref,
                namespace=namespace, tag=tag,
            )
        except ValueError as exc:
            # e.g. sub-dir build services with no {task_id} template — fail loud.
            raise XRLEnvError(str(exc)) from exc

        # Whole-stack footprint (scheduler reserve): main's declared cpu/mem (or the
        # default budget when undeclared) + the sidecar aggregate. Cap main in the
        # doc too — cluster mode IS harbor's resources-override for main (there's no
        # override compose file on the node), so a declared main cpu/mem is enforced.
        cpu_limit, mem_limit_bytes = self._effective_cpu_mem_limits()
        main_svc = rewritten["services"]["main"]
        if cpu_limit is not None:
            main_svc["cpus"] = cpu_limit
        if mem_limit_bytes is not None:
            main_svc["mem_limit"] = int(mem_limit_bytes)
        sidecar_cpu, sidecar_mem_mb = _hc.sidecar_footprint(rewritten)
        footprint_cpu = (
            cpu_limit if cpu_limit is not None else _DEFAULT_MAIN_CPU
        ) + sidecar_cpu
        footprint_mem_bytes = (
            mem_limit_bytes if mem_limit_bytes is not None else _DEFAULT_MAIN_MEM_BYTES
        ) + sidecar_mem_mb * 1024 * 1024

        compose_yaml = yaml.safe_dump(rewritten, sort_keys=False)
        client = _client_from_env()
        self._xrlenv_client = client
        try:
            self._xrlenv_session = await client.acquire_compose_project(
                compose_yaml=compose_yaml,
                images=images,
                footprint_cpu=footprint_cpu,
                footprint_mem_bytes=int(footprint_mem_bytes),
                main_service="main",
                labels=self._acquire_labels(),
                task_key=self.environment_name,
                # Fail-fast below harbor's setup wait so an at-cap acquire surfaces a
                # retriable CapacityExhausted (see _ACQUIRE_QUEUE_TIMEOUT_S).
                queue_timeout_s=_ACQUIRE_QUEUE_TIMEOUT_S,
            )
        except Exception:
            await client.close()
            self._xrlenv_client = None
            raise
        # Same in-container log-dir setup as the single path (targets main).
        await self._setup_cluster_log_dirs()

    async def start(self, force_build: bool) -> None:
        """Acquire a remote raw container scoped to a fresh
        rollout. Pre-built images on the node are required this
        slice; ``force_build=True`` is currently a no-op with a log
        line — real build-on-acquire ships in P1.7.C.2."""
        if force_build:
            self.logger.info(
                "force_build=True ignored in cluster mode — pre-build "
                "via scripts/build-task-images.sh on each node, or "
                "wait for build-on-acquire (P1.7.C.2).",
            )

        # §4b — a multi-service task routes to the compose-project path; a
        # single-service / compose-less task keeps the raw acquire below,
        # byte-for-byte.
        compose_doc = self._multi_service_compose()
        if compose_doc is not None:
            await self._start_compose_project(compose_doc)
            return

        image_ref = self._resolve_image_ref()
        client = _client_from_env()
        self._xrlenv_client = client

        # Build labels for the cluster acquire. We default-populate
        # ``xrlenv.rollout.artifact_path`` (the per-trial dir on the
        # consumer's filesystem — harbor's verifier/agent/artifacts
        # land there post-trial) and ``xrlenv.rollout.displayed_name``
        # (the harbor trial id, e.g. "fix-git__vAcCM84") so the admin
        # ``/rollouts/raw`` view shows a recognisable per-trial row
        # without the operator having to wrap each trial in
        # ``with xrlenv.rollout_metadata(...):``.
        #
        # An explicit ``rollout_metadata(...)`` block takes precedence
        # — read from the contextvar last so the operator can override
        # either field per-trial when needed (matches the docker-py
        # drop-in's contextvar precedence).
        labels = self._acquire_labels()

        # Elevated container capabilities, opt-in via environment.kwargs
        # (xrlenv_cap_add / xrlenv_devices / xrlenv_privileged). Forwarded
        # verbatim to acquire_container; the control plane's KwargsPolicy is
        # the gate. cap_add (NET_ADMIN/SYS_ADMIN/SYS_PTRACE) is allowed by
        # default — no nodes.yaml change needed — so infra tasks (iptables,
        # ip netns, debuggers) recover here; privileged stays default-deny
        # unless the operator sets allow_privileged. Absent kwargs → the
        # acquire_container defaults (cap_add/devices=None, privileged=False),
        # i.e. no behavior change for tasks that don't request them.
        xk = self._xrlenv_kwargs
        cap_add = xk.get("xrlenv_cap_add")
        devices = xk.get("xrlenv_devices")
        privileged = bool(xk.get("xrlenv_privileged", False))

        # Honor the task's declared CPU/memory. Delegate to harbor's canonical
        # ``_effective_*`` accessors — the same ones docker.py / e2b / modal /
        # daytona use — so we inherit harbor's resolution for free:
        # ``override_cpus`` / ``override_memory_mb`` are already merged into
        # ``task_env_config`` at init, and the ``*_enforcement_policy`` gates the
        # value (``ignore`` → ``None``). ``None`` → omit the kwarg → the node
        # applies its safe default cap, so a task that declares nothing is
        # unchanged. Without this the cluster ran every task at the node default
        # (2 CPU / 4 GiB) regardless of task.toml — silently under-provisioning
        # the memory-hungry tasks (e.g. those declaring 8-16 GiB).
        # ``acquire_container`` has no disk/GPU param, so storage_mb/gpus aren't
        # forwarded: no per-container disk cap is applied today (that
        # over-provisions, never starves) and the verified corpus requests 0 GPUs.
        #
        # Optional per-task resource multipliers (default 1.0 = unchanged) are
        # applied inside ``_effective_cpu_mem_limits`` on top of the task's
        # *effective* cpu/mem (i.e. after any harbor override_cpus/override_memory_mb),
        # so ``--cpus-multiplier 2`` doubles whatever each task declares — a headroom
        # / contention ablation that preserves the corpus's *relative* sizing (unlike
        # a flat override_cpus that flattens every task to one value). For a
        # cpuset-pinned task the node reserves ``ceil(cpu_limit)`` cores, so a
        # multiplier widens ``nproc`` too.
        cpu_limit, mem_limit_bytes = self._effective_cpu_mem_limits()

        # CPU is applied as a CFS ``--cpus`` quota (burstable across cores) —
        # exactly how harbor runs the container. cpuset PINNING (confining the
        # container to a right-sized set of cores) stays OFF by default: harbor
        # never sets it, and blanket-pinning can hurt latency-sensitive tasks.
        #
        # It is opt-in **per task** because of a harbor flaw on large hosts:
        # harbor applies only the CFS quota + a hard memory cap and never sets
        # cpuset, so ``nproc`` / CPU affinity inside the container reports the
        # *host* core count, not the task's declared ``cpus``. On a big node
        # (e.g. 192 cores) oracles that scale to ``nproc`` — ``make -j$(nproc)``,
        # ninja's auto ``-j``, OpenBLAS/OMP thread pools — then fan out to
        # ~host-count workers and blow past the declared memory limit (OOM:
        # install-windows-3.11's QEMU build SIGKILL'd cc1). Pinning sizes the
        # affinity mask to ``ceil(cpus)`` so ``nproc`` == the task budget again,
        # while the CFS quota + memory cap are still applied (limits respected).
        #
        # Two per-task opt-in channels (harbor has no per-task ``kwargs``):
        #   * job-level ``environment.kwargs: {xrlenv_cpu_pinning: true}`` —
        #     applies to every task in the job (blunt), and
        #   * per-task ``[environment.env] XRLENV_CPU_PINNING = "1"`` in the
        #     task's own ``task.toml`` — the surgical channel the patched-cache
        #     pipeline uses to mark only the nproc-scaling oracles.
        runtime_limits = None
        task_env = getattr(self.task_env_config, "env", None) or {}
        env_marker = str(task_env.get("XRLENV_CPU_PINNING", "")).strip().lower() in (
            "1", "true", "yes", "on",
        )
        if bool(xk.get("xrlenv_cpu_pinning", False)) or env_marker:
            from xrlenv.backends.base import RuntimeLimits  # avoid cycle

            runtime_limits = RuntimeLimits(cpu_pinning=True)

        # Per-task container-runtime routing (sysbox). Read ONLY from the task's
        # own ``[environment.env] XRLENV_CONTAINER_RUNTIME`` — deliberately no
        # job-level kwarg channel, so routing is case-by-case in the corpus and
        # never a global default (an accidental cluster-wide sysbox switch would
        # hard-fail every task on a cluster with no sysbox node). ``None`` (the
        # unmarked default) → acquire on the node default runtime (runc), no
        # change. When set, the control plane's KwargsPolicy gates it
        # (``allowed_runtimes`` must include it) and the scheduler pins the
        # acquire to a node advertising that runtime (else BackendCapabilityMissing,
        # fail-loud). The marker is written case-by-case by
        # ``benchmarks/terminalworld/build_cache.py --stage sysbox``.
        container_runtime = (
            str(task_env.get("XRLENV_CONTAINER_RUNTIME", "")).strip() or None
        )
        # Companion marker: for a DinD task whose image ships a docker daemon
        # but starts nothing (harbor's DooD host-socket mount was the shortcut
        # we refuse), bring up a nested dockerd after acquire so solve.sh's
        # ``docker`` talks to its own unprivileged daemon — the faithful sysbox
        # substrate. Only meaningful alongside a sysbox runtime.
        inner_dockerd = str(
            task_env.get("XRLENV_INNER_DOCKERD", ""),
        ).strip().lower() in ("1", "true", "yes", "on")
        # Companion marker: bring the nested dockerd up with the LEGACY image
        # store (containerd-snapshotter off) so pushes produce docker schema2
        # manifests, not OCI. TerminalWorld DinD tasks were authored against an
        # older docker (schema2 default); a modern nested dockerd's containerd
        # store pushes OCI manifests, which schema2-only tools in the task (e.g.
        # tw_709166's `dockdiver`) 404 on. Faithful to the task's target env.
        dockerd_legacy_store = str(
            task_env.get("XRLENV_DOCKERD_LEGACY_STORE", ""),
        ).strip().lower() in ("1", "true", "yes", "on")
        # Companion marker: install the docker engine before starting it, for a
        # CLI-only DooD image (docker-ce-cli, no daemon — it relied on the host
        # socket). apt-installs docker-ce from the image's configured repo.
        install_dockerd = str(
            task_env.get("XRLENV_INSTALL_DOCKERD", ""),
        ).strip().lower() in ("1", "true", "yes", "on")
        # Companion marker: boot the container with systemd as PID 1
        # (``command=[/sbin/init]``) instead of ``sleep infinity``, so a task
        # that expects a real init (``systemctl start …``, boot-enabled services)
        # runs faithfully. Sysbox provides an unprivileged systemd PID 1 (that is
        # its purpose); the image must ship systemd (e.g. the CentOS-7 tasks). We
        # wait for the boot to settle before harbor runs solve.sh.
        systemd_init = str(
            task_env.get("XRLENV_SYSTEMD_INIT", ""),
        ).strip().lower() in ("1", "true", "yes", "on")
        # Companion marker: the image's ENTRYPOINT is an interactive shell/REPL
        # (SWE-bench Pro: ``ENTRYPOINT ["/bin/bash"]``), which would swallow the
        # keep-alive ``sleep infinity`` as a script name and exit at once — every
        # later exec then 409s on a stopped container. Start such images with
        # ``entrypoint=["sleep"], command=["infinity"]`` (upstream's own evaluator
        # overrides the entrypoint the same way). Never combined with systemd init.
        keepalive_entrypoint, keepalive_command = _keepalive_argv(task_env, systemd_init)
        # Sysbox file transfer routes through the exec-based path (tar+base64 via
        # exec) instead of docker put_archive/get_archive, which 500 on the
        # idmapped /etc/resolv.conf. Also relaxes uploaded tar modes to world r-x.
        self._sysbox_upload = bool(container_runtime) and container_runtime != "runc"

        try:
            self._xrlenv_session = await client.acquire_container(
                image=image_ref,
                command=keepalive_command,
                entrypoint=keepalive_entrypoint,
                name=_sanitize_container_name(self.session_id),
                labels=labels,
                task_key=self.environment_name,
                # Fail-fast below harbor's setup ``wait_for`` so an at-cap acquire
                # surfaces a retriable ``CapacityExhausted`` instead of being
                # cancelled into a non-retriable timeout. See _ACQUIRE_QUEUE_TIMEOUT_S.
                queue_timeout_s=_ACQUIRE_QUEUE_TIMEOUT_S,
                cap_add=cap_add,
                devices=devices,
                privileged=privileged,
                container_runtime=container_runtime,
                cpu_limit=cpu_limit,
                mem_limit_bytes=mem_limit_bytes,
                runtime_limits=runtime_limits,
                # Always acquire OPEN. The agent's install/bootstrap phase
                # (clone / npm) needs network, so we can't acquire offline
                # tasks with --network none anymore. An offline task's egress
                # is instead restricted AFTER install by the consumer's Trial
                # via apply_egress() (open-setup→tighten). The env can't do it
                # at acquire — install (harbor's _setup_agent) hasn't run yet.
                # disable_internet=True stays truthful: the env CAN disable,
                # now post-install via apply_egress rather than at acquire.
                network_mode=None,
            )
        except Exception:
            await client.close()
            self._xrlenv_client = None
            raise

        # Create + chmod the in-container log dirs harbor relies on.
        # Local mode gets these for free from docker-compose bind
        # mounts (the host paths auto-create). Cluster mode has no
        # mounts (``is_mounted=False``), so the three dirs are absent
        # in a typical tb2 base image; harbor's first agent ``bash -c
        # "... > /logs/agent/oracle.txt 2>&1"`` then exits 1 with
        # "no such file or directory" and the trial fails on the
        # post-run ``download_dir(/logs/verifier)``.
        #
        # **Integrity note.** The three dirs are *write-targets*,
        # empty during agent run. The agent's solve.sh writes to
        # ``/logs/agent/`` (its own log) and reads nothing from
        # ``/logs/verifier`` or ``/logs/artifacts``. The verifier's
        # actual test scripts go to ``/tests`` (separate path),
        # uploaded by harbor's ``Verifier`` AFTER the agent
        # completes — the agent never sees them. This matches
        # harbor's local-mode setup exactly (the bind-mounted
        # ``/logs/verifier`` is also visible-but-empty to the
        # agent there); we're not opening any new visibility.
        await self._setup_cluster_log_dirs()

        # systemd substrate: wait for the boot to settle (services started)
        # BEFORE harbor runs solve.sh, so ``systemctl start …`` finds a live
        # systemd. (The log-dir mkdir above already works via ``docker exec``
        # while systemd boots in parallel.)
        if systemd_init:
            await self._wait_for_systemd()

        # DinD substrate: start a nested dockerd inside the (sysbox) container
        # and wait for its socket BEFORE harbor runs the agent's solve.sh, so a
        # ``docker pull`` / ``docker run`` in the task talks to its own daemon.
        if inner_dockerd:
            await self._start_inner_dockerd(
                legacy_store=dockerd_legacy_store,
                install_dockerd=install_dockerd,
            )

        # harbor 0.20 native network policy: enforce a non-public startup
        # baseline now (no-op for the common public baseline → sealed later at
        # the agent-phase boundary by harbor via _apply_network_policy).
        await self._apply_baseline_network_policy()

    async def _wait_for_systemd(self, timeout_s: int = 60) -> None:
        """Wait for systemd (PID 1) to finish booting inside the container.

        ``systemctl is-system-running`` reports ``initializing`` / ``starting``
        during boot and ``running`` / ``degraded`` once units have been
        processed. We poll until it leaves the starting states (``degraded`` is
        fine — some units legitimately fail in a container) or ``timeout_s``
        elapses, so a task's ``systemctl start …`` isn't racing the boot.
        """
        script = (
            f"for i in $(seq 1 {timeout_s}); do "
            "state=$(systemctl is-system-running 2>/dev/null); "
            'case "$state" in '
            "running|degraded|maintenance) "
            'echo "xrlenv: systemd $state after ${i}s"; exit 0 ;; '
            "esac; sleep 1; done; "
            "echo \"xrlenv: systemd still '$state' after ${timeout_s}s\" >&2; exit 0"
        )
        result = await self.exec(script, user="root", timeout_sec=timeout_s + 15)
        self.logger.info(
            "systemd bring-up: %s", (result.stdout or result.stderr or "").strip(),
        )

    async def _start_inner_dockerd(
        self, timeout_s: int = 90, *, legacy_store: bool = False,
        install_dockerd: bool = False,
    ) -> None:
        """Start a docker daemon *inside* the container and wait for its socket.

        The faithful substrate for TerminalWorld's DooD (Docker-out-of-Docker)
        tasks: the task's own compose mounts the host ``/var/run/docker.sock``
        — insecure on a shared multi-tenant cluster, and the cluster acquire
        path drops host mounts anyway. Under ``sysbox-runc`` the container can
        run its OWN unprivileged dockerd (that is sysbox's purpose), which is
        how a real TerminalWorld VM provides docker (its init/systemd starts
        it). We never edit ``solve.sh`` — we only bring the daemon up the way
        the VM's boot would, then let the unchanged task run.

        The task image must ship a docker daemon (``docker.io`` / ``docker-ce``).
        A CLI-only DooD image (``docker-ce-cli`` only, relying on the host socket
        by design) has no ``dockerd`` to launch: without ``install_dockerd`` this
        fails loud with a clear message rather than an opaque ``Cannot connect to
        the Docker daemon``. ``install_dockerd`` (see ``XRLENV_INSTALL_DOCKERD``)
        installs the daemon first — ``apt-get install docker-ce containerd.io``
        from the image's already-configured docker repo (falling back to
        ``docker.io``), needs in-container network. dockerd is launched detached
        (it survives the exec by reparenting to the container's PID 1 — ``sleep
        infinity`` — kept alive because sysbox acquires skip tini); we then poll
        ``docker info`` until the daemon answers or ``timeout_s`` elapses.

        The socket is ``chmod 666``'d once the daemon is up so a task whose image
        drops to a **non-root** user (``USER …``) can still reach the nested
        daemon — dockerd creates ``/var/run/docker.sock`` root-owned ``0660``, so
        without this a non-root ``solve.sh`` gets ``permission denied ... docker
        API``. Safe in a single-tenant task container.

        ``legacy_store`` disables the containerd image store
        (``containerd-snapshotter: false``) so pushes produce docker schema2
        manifests instead of OCI — for tasks whose tooling only understands
        schema2 (see ``XRLENV_DOCKERD_LEGACY_STORE``). Written to
        ``/etc/docker/daemon.json`` before the daemon starts.
        """
        daemon_json = (
            "mkdir -p /etc/docker && "
            "printf '{\"features\":{\"containerd-snapshotter\":false}}' "
            "> /etc/docker/daemon.json\n"
            if legacy_store else ""
        )
        if install_dockerd:
            no_dockerd = (
                "  echo 'xrlenv: installing docker engine (CLI-only image)…' >&2\n"
                "  export DEBIAN_FRONTEND=noninteractive\n"
                "  apt-get update >/var/log/xrlenv-dockerd-install.log 2>&1\n"
                "  apt-get install -y docker-ce containerd.io "
                ">>/var/log/xrlenv-dockerd-install.log 2>&1 || "
                "apt-get install -y docker.io "
                ">>/var/log/xrlenv-dockerd-install.log 2>&1\n"
                "  command -v dockerd >/dev/null 2>&1 || "
                "{ echo 'xrlenv: docker engine install failed; log tail:' >&2; "
                "tail -n 30 /var/log/xrlenv-dockerd-install.log >&2; exit 3; }\n"
            )
        else:
            no_dockerd = (
                "  echo 'xrlenv: image has no dockerd (CLI-only DooD image?) — "
                "set XRLENV_INSTALL_DOCKERD to install it' >&2\n  exit 3\n"
            )
        script = f"""set -e
if ! command -v dockerd >/dev/null 2>&1; then
{no_dockerd}fi
if ! docker info >/dev/null 2>&1; then
  {daemon_json}  setsid nohup dockerd >/var/log/xrlenv-dockerd.log 2>&1 </dev/null &
fi
for i in $(seq 1 {timeout_s}); do
  if docker info >/dev/null 2>&1; then
    chmod 666 /var/run/docker.sock 2>/dev/null || true
    echo "xrlenv: nested dockerd ready after ${{i}}s"
    exit 0
  fi
  sleep 1
done
echo 'xrlenv: nested dockerd did not become ready; log tail:' >&2
tail -n 40 /var/log/xrlenv-dockerd.log >&2 || true
exit 1
"""
        # A CLI-only image needs an apt install first (slow); give it headroom.
        exec_timeout = (300 if install_dockerd else timeout_s + 30)
        result = await self.exec(script, user="root", timeout_sec=exec_timeout)
        if result.return_code != 0:
            raise RuntimeError(
                "cluster harbor start: nested dockerd bring-up failed "
                f"(exit={result.return_code}): stdout={result.stdout!r} "
                f"stderr={result.stderr!r}. The task image must ship a docker "
                f"daemon (docker.io/docker-ce) and run under sysbox-runc.",
            )
        self.logger.info(
            "nested dockerd ready: %s", (result.stdout or "").strip(),
        )

    async def stop(self, delete: bool) -> None:
        """Tear down the remote container + close the cluster client.
        Idempotent: a second ``stop()`` call is a no-op so harbor's
        cleanup paths can call it without checking state.

        ``delete`` is always honoured because ``acquire_container``'s
        contract is "session-scoped, cluster destroys on session end".
        ``keep_containers=True`` is logged as a no-op since the
        session model doesn't expose a "stop but keep" branch."""
        if self._xrlenv_session is None:
            return
        if self._keep_containers:
            self.logger.warning(
                "keep_containers=True is not yet honoured in cluster "
                "mode — the session model destroys the container on "
                "stop. Use the local-mode XrlenvHarborEnvironment if "
                "you need keep-containers semantics.",
            )
        del delete  # see docstring — destroy is unconditional this slice
        try:
            await self._xrlenv_session.destroy()
        except Exception as exc:
            self.logger.warning(
                "ClusterContainerSession.destroy failed: %r", exc,
            )
        finally:
            self._xrlenv_session = None
            if self._xrlenv_client is not None:
                try:
                    await self._xrlenv_client.close()
                except Exception as exc:
                    self.logger.warning(
                        "Cluster Client.close failed: %r", exc,
                    )
                self._xrlenv_client = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run ``command`` inside the remote container via the
        cluster's streaming-exec wire. Streaming (vs batched) is
        load-bearing — terminal-bench-2 tasks run for 1-2 hours;
        batched would risk idle TCP drops along the consumer↔control-
        plane↔node path, plus buffer hundreds of MB of stdout in node
        memory. Streaming flushes per chunk and keeps every hop
        provably alive."""
        if self._xrlenv_session is None:
            raise RuntimeError(
                "XrlenvHarborEnvironmentCluster.exec called before "
                "start() — no active cluster session.",
            )

        resolved_user = self._resolve_user(user)
        merged_env = self._merge_env(env)
        effective_cwd = cwd or self.task_env_config.workdir
        timeout_s = (
            float(timeout_sec) if timeout_sec else _DEFAULT_EXEC_TIMEOUT_S
        )

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        exit_code = -1

        async for chunk in self._xrlenv_session.exec_stream(
            ["bash", "-c", command],
            timeout_s=timeout_s,
            cwd=effective_cwd,
            env=merged_env,
            user=str(resolved_user) if resolved_user is not None else None,
        ):
            if chunk.stdout:
                stdout_chunks.append(chunk.stdout)
            if chunk.stderr:
                stderr_chunks.append(chunk.stderr)
            if chunk.done:
                exit_code = chunk.exit_code

        stdout = b"".join(stdout_chunks).decode(errors="replace")
        stderr = (
            b"".join(stderr_chunks).decode(errors="replace")
            if stderr_chunks else None
        )
        return ExecResult(
            stdout=stdout or None,
            stderr=stderr,
            return_code=exit_code,
        )

    async def apply_egress(
        self,
        cidrs: list[str],
        *,
        ports: tuple[int, ...] | None = None,
        dns_resolver: str | None = None,
    ) -> None:
        """Restrict this container's egress to ``cidrs`` (spec-07 mechanism).

        Thin passthrough to the cluster session's ``apply_egress`` — the env
        owns the session, so the consumer's Trial drives the *policy* (when to
        restrict + which endpoints) without reaching into a private attribute.
        ``cidrs`` is a list of ``"a.b.c.d/32"`` (or wider) strings; an **empty**
        list means block all external egress (loopback + metadata-DROP remain),
        for a task whose agent needs no external endpoint. ``ports`` optionally
        narrows every cidr to those destination ports. The node installs the
        iptables program in the container's netns; the workload (no
        CAP_NET_ADMIN) can't undo it. Idempotent; fail-closed node-side.
        """
        if self._xrlenv_session is None:
            raise RuntimeError(
                "XrlenvHarborEnvironmentCluster.apply_egress called before "
                "start() — no active cluster session.",
            )
        # Imported here (not at module top) so the plug-in still imports on a
        # control-plane / SDK build whose xrlenv predates the egress module.
        from xrlenv.backends.egress import EgressAllowlist, EgressRule

        allowlist = EgressAllowlist(
            rules=tuple(EgressRule(cidr=c, ports=ports) for c in cidrs),
        )
        await self._xrlenv_session.apply_egress(
            allowlist, dns_resolver=dns_resolver,
        )

    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        """Enforce a harbor per-phase ``NetworkPolicy`` (harbor 0.20 native seam).

        harbor's ``Trial`` calls ``set_network_policy`` — which routes here — at
        each phase boundary: e.g. a ``PUBLIC`` baseline while the agent installs,
        then ``NO_NETWORK`` / ``ALLOWLIST`` for ``agent.run()``, then a restore to
        the baseline afterwards. This is the native replacement for the old
        open-setup→post-install ``apply_egress`` hack (and the sweep-side seal
        wrapper it once needed): harbor decides *when* to switch; we own only the
        *enforcement*, via the spec-07 ``apply_egress`` primitive.

        Mode → egress program (:mod:`xrlenv.backends.egress`):

        - ``PUBLIC``     → ``apply_egress(["0.0.0.0/0"])`` — allow all external.
          The compiled program DROPs cloud-metadata **before** the ``0.0.0.0/0``
          ACCEPT, so even a re-opened container can't reach IMDS (a deliberate,
          stricter deviation from a bare bridge). This is how a post-agent-phase
          *restore-to-PUBLIC-baseline* re-opens egress with the CIDR primitive.
        - ``NO_NETWORK`` → ``apply_egress([])`` — block all external egress
          (loopback + the mandatory metadata DROP remain).
        - ``ALLOWLIST``  → ``apply_egress(allowed_hosts)`` — permit only the
          listed v4 IPs/CIDRs (e.g. the LLM proxy's ``<node-ip>/32``).
          Hostname / wildcard / IPv6 entries are rejected upstream by
          ``validate_network_policy_support`` (see :meth:`_can_enforce_egress`
          and the ``network_allowlist_*`` capabilities), so ``allowed_hosts``
          here is always v4 IP/CIDR.
        """
        from harbor.models.task.config import NetworkMode as _NetworkMode

        mode = network_policy.network_mode
        if mode == _NetworkMode.PUBLIC:
            cidrs = ["0.0.0.0/0"]
        elif mode == _NetworkMode.NO_NETWORK:
            cidrs = []
        elif mode == _NetworkMode.ALLOWLIST:
            cidrs = list(network_policy.allowed_hosts)
        else:  # pragma: no cover — harbor's NetworkMode is a closed enum
            raise XRLEnvError(f"unsupported harbor network_mode {mode!r}")
        await self.apply_egress(cidrs)

    async def _apply_baseline_network_policy(self) -> None:
        """Enforce a **non-public** startup baseline (the task's ``[environment]``
        policy) right after acquire, before harbor's install phase runs.

        We always ACQUIRE open (``network_mode=None``) so harbor's ``_setup_agent``
        has network. harbor sets ``self._network_policy`` to the resolved baseline
        at construction and assumes the env *starts in it*; ``set_network_policy``
        then no-ops when a later phase equals the baseline. So if the baseline is
        non-public we must apply it now — otherwise a task whose agent-phase policy
        equals a non-public baseline would never be sealed. The common offline
        shape (``[environment] public`` → ``[agent] no-network``) is a no-op here
        and gets sealed later by harbor at the agent-phase boundary.
        """
        from harbor.models.task.config import NetworkMode as _NetworkMode

        # Read the underlying attr (not the ``network_policy`` property) with a
        # default: harbor's __init__ always sets ``_network_policy``, but unit
        # tests that build the env via ``__new__`` skip it — treat "unset" as the
        # public default (no baseline to enforce).
        baseline = getattr(self, "_network_policy", None)
        if baseline is None or baseline.network_mode == _NetworkMode.PUBLIC:
            return  # acquired open == public baseline; nothing to enforce
        if not self._can_enforce_egress():  # pragma: no cover — capability-gated
            raise XRLEnvError(
                f"baseline network policy {baseline.network_mode!r} requested but "
                "egress cannot be enforced for this task "
                "(compose / sysbox / privileged) — harbor should have rejected it "
                "at validation via capabilities.",
            )
        await self._apply_network_policy(baseline)

    async def _ensure_dir(self, path: str) -> None:
        """``mkdir -p`` ``path`` inside the container as root.

        Docker's ``put_archive`` requires the target dir to already
        exist (404 ``"Could not find the file"`` otherwise). harbor's
        local mode hides this via ``docker compose cp`` which auto-
        creates the dst dir; cluster mode has to mkdir explicitly.
        Mirrors the apple_container backend's ``_upload_tar`` step."""
        if self._xrlenv_session is None:
            raise RuntimeError(
                "_ensure_dir called before start() — no cluster session.",
            )
        # Retry the (idempotent) mkdir on a transient exec abort. Under load
        # the first in-container exec right after acquire occasionally returns
        # a negative ``exit_code`` with empty stderr — the exec stream was cut
        # before delivering a real code, not mkdir failing. A single un-retried
        # attempt turned that hiccup into a whole-trial AddTestsDirError
        # (observed on random tasks: make-doom-for-mips, nginx-request-logging).
        # ``mkdir -p`` is idempotent, so retrying is always safe.
        last_exit = 0
        last_stderr = b""
        for attempt in range(3):
            result = await self._xrlenv_session.exec(
                ["mkdir", "-p", path],
                timeout_s=10.0,
                user="root",
            )
            if result.exit_code == 0:
                return
            last_exit, last_stderr = result.exit_code, result.stderr
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"mkdir -p {path!r} failed after 3 attempts "
            f"(exit={last_exit}): stderr={last_stderr!r}",
        )

    async def _sysbox_upload_tarball(
        self, target_dir: str, tarball: bytes,
    ) -> None:
        """Extract ``tarball`` under ``target_dir`` inside the container via
        ``exec`` + ``base64 -d | tar -x`` — NOT docker ``put_archive``.

        docker's archive-PUT re-processes the container's rootfs bind mounts and
        500s on the idmapped ``/etc/resolv.conf`` under sysbox
        (``openat etc/resolv.conf: file exists``); a plain ``tar -x -C
        <target_dir>`` running as a normal in-container process only writes to
        ``target_dir`` and never touches ``/etc``. It also runs as container-root
        (mapped), so extracted files are root-owned + chmod-able (the
        put_archive path left them 65534-owned; see _relax_modes).

        The base64 payload is embedded in the exec command line (exec has no
        stdin). Small tars (harbor's solution/tests dirs are KB) go in one exec;
        larger ones are chunk-appended to a staging file to stay under ARG_MAX.
        """
        if self._xrlenv_session is None:
            raise RuntimeError("_sysbox_upload_tarball before start()")
        b64 = base64.b64encode(tarball).decode("ascii")
        q = shlex.quote
        tdir = q(target_dir)
        timeout = max(60.0, len(b64) / (512 * 1024))  # ~generous
        if len(b64) <= _SYSBOX_XFER_CHUNK:
            script = (
                f"set -e; mkdir -p {tdir}; "
                f"printf %s {q(b64)} | base64 -d | tar -x -C {tdir}"
            )
            await self._sysbox_exec_checked(script, timeout_s=timeout,
                                            what="upload")
            return
        # Chunked: stage the base64 in a unique file, then decode+extract.
        self._xfer_seq += 1
        staging = f"/tmp/.xrlenv-xfer.{os.getpid()}.{self._xfer_seq}.b64"
        sq = q(staging)
        await self._sysbox_exec_checked(
            f"set -e; mkdir -p {tdir}; : > {sq}", timeout_s=30.0, what="upload-init",
        )
        for i in range(0, len(b64), _SYSBOX_XFER_CHUNK):
            chunk = b64[i:i + _SYSBOX_XFER_CHUNK]
            await self._sysbox_exec_checked(
                f"printf %s {q(chunk)} >> {sq}", timeout_s=60.0, what="upload-chunk",
            )
        await self._sysbox_exec_checked(
            f"set -e; base64 -d {sq} | tar -x -C {tdir}; rm -f {sq}",
            timeout_s=timeout, what="upload-extract",
        )

    async def _sysbox_download_tarball(self, parent: str, name: str) -> bytes:
        """Return a ``get_archive``-compatible tarball of ``<parent>/<name>``
        via ``exec`` ``tar -c -C <parent> <name> | base64`` — NOT docker
        ``get_archive`` (same sysbox ``/etc/resolv.conf`` 500 as the upload
        path). ``tar -c -C <parent> <name>`` roots entries at ``<name>/…`` — the
        exact shape docker ``get_archive`` emits — so the existing
        ``_untar_*`` helpers consume it unchanged."""
        if self._xrlenv_session is None:
            raise RuntimeError("_sysbox_download_tarball before start()")
        q = shlex.quote
        script = f"cd {q(parent)} && tar -cf - {q(name)} | base64"
        result = await self._xrlenv_session.exec(
            ["bash", "-c", script], timeout_s=_SYSBOX_DOWNLOAD_TIMEOUT_S,
            user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"sysbox download of {parent}/{name} failed "
                f"(exit={result.exit_code}): stderr={result.stderr!r}",
            )
        # Strip base64 line-wrapping (GNU base64 wraps at 76 cols by default).
        raw = bytes(result.stdout or b"")
        return base64.b64decode(b"".join(raw.split()))

    async def _sysbox_exec_checked(
        self, script: str, *, timeout_s: float, what: str,
    ) -> None:
        """Run ``bash -c script`` as root; raise on a non-zero exit."""
        assert self._xrlenv_session is not None
        result = await self._xrlenv_session.exec(
            ["bash", "-c", script], timeout_s=timeout_s, user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"sysbox {what} exec failed (exit={result.exit_code}): "
                f"stderr={result.stderr!r}",
            )

    async def upload_file(
        self, source_path: Path | str, target_path: str,
    ) -> None:
        if self._xrlenv_session is None:
            raise RuntimeError(
                "upload_file called before start() — no cluster session.",
            )
        source = Path(source_path)
        target = Path(target_path)
        tarball = _tar_one_file(
            source, arcname=target.name, world_accessible=self._sysbox_upload,
        )
        if self._sysbox_upload:
            await self._sysbox_upload_tarball(str(target.parent), tarball)
            return
        # docker put_archive extracts the tarball under ``target_dir``;
        # to land bytes at ``/foo/bar.txt`` we tar one entry named
        # ``bar.txt`` and put it under ``/foo``. Make sure ``/foo``
        # exists first — Docker's PUT /containers/.../archive returns
        # 404 if the path doesn't.
        await self._ensure_dir(str(target.parent))
        await self._xrlenv_session.put_archive(
            target_dir=str(target.parent),
            tarball=tarball,
        )

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str,
    ) -> None:
        if self._xrlenv_session is None:
            raise RuntimeError(
                "upload_dir called before start() — no cluster session.",
            )
        source = Path(source_dir)
        if not source.is_dir():
            raise NotADirectoryError(
                f"upload_dir source is not a directory: {source}",
            )
        # Match harbor's local-mode ``docker compose cp src/. main:dst``
        # semantics: tarball children of source land directly under
        # target_dir (no nested basename sub-dir).
        tarball = _tar_dir_contents(source, world_accessible=self._sysbox_upload)
        if self._sysbox_upload:
            await self._sysbox_upload_tarball(target_dir, tarball)
            return
        await self._ensure_dir(target_dir)
        await self._xrlenv_session.put_archive(
            target_dir=target_dir,
            tarball=tarball,
        )

    async def download_file(
        self, source_path: str, target_path: Path | str,
    ) -> None:
        if self._xrlenv_session is None:
            raise RuntimeError(
                "download_file called before start() — no cluster session.",
            )
        if self._sysbox_upload:
            src = Path(source_path)
            tarball = await self._sysbox_download_tarball(
                str(src.parent), src.name,
            )
        else:
            tarball = await self._xrlenv_session.get_archive(source_path)
        _untar_one_file(tarball, Path(target_path))

    async def download_dir(
        self, source_dir: str, target_dir: Path | str,
    ) -> None:
        if self._xrlenv_session is None:
            raise RuntimeError(
                "download_dir called before start() — no cluster session.",
            )
        if self._sysbox_upload:
            src = Path(source_dir)
            tarball = await self._sysbox_download_tarball(
                str(src.parent), src.name,
            )
        else:
            tarball = await self._xrlenv_session.get_archive(source_dir)
        _untar_dir_contents(tarball, Path(target_dir))

    async def _chown_to_host_user(
        self, path: str, recursive: bool = False,
    ) -> None:
        """No-op in cluster mode. Bind-mount host UID alignment is
        moot when the container is on a remote node and outputs come
        back via ``get_archive``. The local FS at the consumer
        already gets the consumer's UID/GID at extraction time."""
        del path, recursive  # see docstring
