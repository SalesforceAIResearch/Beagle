"""pier ``BaseEnvironment`` adapter for xrlenv.

pier (https://github.com/datacurve-ai/pier) is a harbor fork; this module is the
direct analog of ``xrlenv_plugins.harbor.environment`` retargeted at pier's classes
(pier reimplements the harness in-tree and does NOT import harbor at runtime, so we
subclass **pier's** ``DockerEnvironment``, not harbor's).

We subclass :class:`pier.environments.docker.docker.DockerEnvironment`
rather than :class:`pier.environments.base.BaseEnvironment` directly, so the heavy
lifting pier already does (docker compose orchestration, mounts,
env-var resolution, agent/verifier dir wiring, network policy,
keep-containers semantics, the ``preflight`` daemon check) is
inherited. We add:

- A constructor that accepts xrlenv-specific kwargs
  (``xrlenv_task_key``, ``xrlenv_group_id``, ``xrlenv_resources``,
  ``xrlenv_image_pin_mode``) without breaking pier's signature.
  Today's LocalDocker mode records them on the instance; cluster
  mode feeds them to the scheduler at the per-task invocation point.

- A subclass hook (``_xrlenv_route_command``) the cluster-mode
  follow-on overrides to redirect pier's ``docker``/``docker
  compose`` invocations through xrlenv's gRPC stack to a
  scheduler-chosen node. LocalDocker mode is a pass-through:
  pier's CLI calls run against whatever ``DOCKER_HOST`` resolves
  to (the local daemon by default), unchanged.

pier is selected the same way its built-in environments are — via
``EnvironmentConfig.import_path`` (pier ships this escape hatch first-class):
``environment.import_path: xrlenv_plugins.pier:XrlenvPierEnvironmentCluster``.

The architectural shape mirrors :class:`xrlenv.compat.docker_client.
XrlenvDockerClient`: subclass the upstream class, swap behavior at
specific seams, leave everything else inherited. The trade-off
documented over there applies here too — for arbitrary upstream
harness behavior we don't have to enumerate, inheritance keeps the
contract intact for free.

Why this lives in ``xrlenv_plugins/pier/`` rather than
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
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from xrlenv.client.client import Client
    from xrlenv.client.container_session import ClusterContainerSession

    from pier.models.task.config import EnvironmentConfig
    from pier.models.trial.config import ServiceVolumeConfig
    from pier.models.trial.paths import TrialPaths

try:
    from pier.environments.agent_setup import (
        EGRESS_PROXY_PORT,
        EGRESS_PROXY_SERVICE,
        new_proxy_token,
        proxy_environment,
        proxy_policy_env,
        squid_bootstrap_command,
    )
    from pier.environments.base import ExecResult
    from pier.environments.capabilities import EnvironmentCapabilities
    from pier.environments.docker.docker import DockerEnvironment
    from pier.models.agent.network import NetworkAllowlist
    from pier.models.trial.paths import EnvironmentPaths
except ImportError as exc:  # pragma: no cover — surface a clear hint.
    raise ImportError(
        "datacurve-pier is not installed — xrlenv's pier plug-in requires "
        "``datacurve-pier==0.3.0``. Install via ``pip install 'xrlenv[deep-swe]'`` "
        "or ``pip install 'datacurve-pier==0.3.0'`` directly.",
    ) from exc

from xrlenv.compat.metadata import (
    LABEL_ARTIFACT_PATH,
    LABEL_DISPLAYED_NAME,
    current_rollout_metadata,
    metadata_to_labels,
)
from xrlenv.errors import XRLEnvError
from xrlenv_plugins.pier import compose as _hc

# pier (like harbor 0.8.x) expresses the container's internet contract through
# ``EnvironmentConfig.allow_internet`` (a plain ``bool`` defaulting to True) — it has
# no ``NetworkMode`` enum. We keep the ``NetworkMode`` symbol as ``None`` (typed
# ``Any`` so the ``is not None`` guard in ``_network_mode_for_task`` — shared verbatim
# with the harbor plug-in — is not flagged as statically dead); it simply falls
# through to the ``allow_internet`` branch.
NetworkMode: Any = None

LOGGER = logging.getLogger(__name__)

# Whole-stack footprint (scheduler reserve) main gets when the task declares no
# cpu/mem — matches the raw-container default budget so a compose main is reserved
# like a single-acquire container would be. Sidecars add on top (Q2).
_DEFAULT_MAIN_CPU = 2.0
_DEFAULT_MAIN_MEM_BYTES = 4 * 1024**3


class XrlenvPierEnvironment(DockerEnvironment):
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
    )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        keep_containers: bool = False,
        mounts_json: list[ServiceVolumeConfig] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Pop xrlenv-only kwargs before forwarding to harbor.
        # DockerEnvironment.__init__ rejects unknowns at the
        # super().__init__() chain.
        self._xrlenv_kwargs: dict[str, Any] = {
            k: kwargs.pop(k) for k in self._XRLENV_KWARGS if k in kwargs
        }
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            keep_containers,
            mounts_json,
            *args,
            **kwargs,
        )

    @staticmethod
    def type() -> str:  # type: ignore[override]  # base narrows to EnvironmentType; a str is explicitly allowed for third-party envs
        """pier's ``BaseEnvironment`` declares ``type()`` abstract as ``-> str``
        precisely so a third-party env like ours can return an arbitrary
        identifier (its docstring says so); the concrete ``DockerEnvironment`` we
        subclass narrows the return to the ``EnvironmentType`` enum, so mypy sees
        our ``str`` as an override mismatch — hence the ignore. harbor keyed the
        analog off the enum; pier requires us to supply one, so we return a stable
        label distinguishing an xrlenv-routed environment in logs."""
        return "xrlenv-cluster"

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
    return sanitized.lstrip("-_.") or "pier-cluster-session"


def _parse_dockerfile_from_ref(text: str) -> str | None:
    """The image ref of the **first** ``FROM`` in a Dockerfile, or ``None``.

    Used to resolve the base image of a separate-verifier container from its
    ``tests/Dockerfile`` (§2.5). Handles the forms the corpus uses:

    - ``FROM public.ecr.aws/…:tag`` → the ref;
    - ``FROM --platform=linux/amd64 <ref> AS base`` → ``<ref>`` (flags + ``AS``
      stage name dropped);
    - quoted refs (``FROM "ref"``) → unquoted;
    - an **unresolved** ``ARG``/``${…}`` ref (e.g. ``FROM $BASE``) → ``None`` (the
      caller then falls back to the parent task's top-level ``docker_image``);
    - no ``FROM`` (or only comments) → ``None``.

    Pure function so the parse is unit-testable in isolation.
    """
    for line in text.splitlines():
        m = re.match(r"\s*FROM\s+(.*)", line, re.IGNORECASE)
        if not m:
            continue
        # Drop ``--platform=…`` (and any other ``--flag``) tokens; stop at ``AS``.
        toks: list[str] = []
        for tok in m.group(1).split():
            if tok.startswith("--"):
                continue
            if tok.upper() == "AS":
                break
            toks.append(tok)
        if not toks:
            return None
        ref = toks[0].strip().strip('"').strip("'")
        if not ref or "$" in ref:  # unresolved ARG / ${..} → let caller fall back
            return None
        return ref
    return None


# ── Egress-proxy synthesis (§4b — the network-allowlist capability) ────────────
#
# pier air-gaps an offline task and lets its *installed agent* reach only an
# allowlist of domains, via a Squid proxy sidecar (``pier.environments.agent_setup.
# write_docker_proxy_compose``): ``main`` on an ``internal: true`` network with the
# proxy as its ONLY route out, the proxy on both that + the default (internet)
# bridge, the allowlist → squid ``dstdomain``. pier BUILDS the proxy image locally;
# the cluster can't build on-node and must not push to the (prod-shared) private
# registry, so we run the proxy from a **mirror-pullable ``ubuntu:24.04``** and
# install squid at container start — the squid policy itself is pier's, verbatim.

# Mirror-pullable proxy base (docker.io → the FSx pull-through mirror). No build,
# no push.
_EGRESS_PROXY_IMAGE = "ubuntu:24.04"

# Compose-up window for the egress path: the proxy apt-installs squid at container
# start, so ``main``'s ``depends_on: service_healthy`` waits on that + squid boot.
_EGRESS_UP_TIMEOUT_S = 300.0


def _is_ipv4_literal(host: str) -> bool:
    """True iff ``host`` is a bare IPv4 literal (e.g. ``internal-ip``). Such a host is
    directly iptables-sealable via :meth:`apply_egress`, so it needs no Squid domain-proxy —
    see :meth:`XrlenvPierEnvironmentCluster._egress_domains`. Pure (stdlib, no DNS)."""
    import ipaddress
    try:
        ipaddress.IPv4Address(host)
    except ValueError:
        return False
    return True


def _egress_proxy_runtime_command() -> str:
    """The proxy container's command: apt-install squid at start (no build), then
    run pier's ``squid_bootstrap_command`` (writes the allowlist squid.conf from
    ``$ALLOWLIST_DOMAINS`` + ``$PROXY_TOKEN`` and execs squid). The bootstrap body
    is pier's verbatim (minus its shebang), so the squid policy — auth, ``dstdomain``
    allowlist, ``deny all`` — is byte-identical to pier's."""
    body = squid_bootstrap_command()
    lines = body.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]  # drop the shebang — we run under an explicit bash -lc
    install = (
        "set -eu; export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update && apt-get install -y --no-install-recommends "
        "apache2-utils ca-certificates squid && rm -rf /var/lib/apt/lists/*; "
    )
    cmd = install + "\n".join(lines)
    # An allowlisted service can listen on a non-standard HIGH port — notably the LLM gateway the
    # agent calls during the RUN phase (e.g. ``http://<node>:18088``). pier's ``Safe_ports`` permits
    # only 80/443, so ``http_access deny !Safe_ports`` denies such a request (ERR_ACCESS_DENIED)
    # BEFORE the domain-allowlist rule runs. Widen ``Safe_ports`` to the high range; the
    # ``allowed_domains`` ACL stays the real gate (only allowlisted hosts are ever reachable, on any
    # port). Targeted string-swap of pier's verbatim line — a no-op if upstream changes it.
    cmd = cmd.replace(
        "acl Safe_ports port 80 443\n", "acl Safe_ports port 80 443\nacl Safe_ports port 1025-65535\n"
    )
    # Same for CONNECT: an HTTP client that TUNNELS (undici/Node fetch CONNECTs even to an http://
    # origin) hits ``http_access deny CONNECT !SSL_ports`` (only 443) when the allowlisted service is
    # on a high port (the gateway on :18088), and hangs. Widen ``SSL_ports`` to the high range too;
    # the ``allowed_domains`` ACL still gates which hosts are reachable. (curl forwards absolute-URI
    # and doesn't need this — but Node's fetch does.)
    cmd = cmd.replace(
        "acl SSL_ports port 443\n", "acl SSL_ports port 443\nacl SSL_ports port 1025-65535\n"
    )
    return cmd


def build_egress_proxy_compose(
    *, main_ref: str, main_command: list[str], domains: list[str], token: str,
) -> tuple[dict[str, Any], list[str]]:
    """Synthesize the cluster egress-proxy compose (§4b) + its ensure-present image
    list — a faithful port of pier's ``write_docker_proxy_compose`` shape, but with
    the proxy running from ``ubuntu:24.04`` + a runtime squid-install command
    instead of a locally-built image. Returns ``(compose_dict, images)``. Pure."""
    compose: dict[str, Any] = {
        "services": {
            _hc.MAIN_SERVICE: {
                "image": main_ref,
                "command": list(main_command),
                # internal-only: main has NO direct egress; the proxy is its
                # sole route out.
                "networks": ["pier-egress-internal"],
                "depends_on": {
                    EGRESS_PROXY_SERVICE: {"condition": "service_healthy"},
                },
            },
            EGRESS_PROXY_SERVICE: {
                "image": _EGRESS_PROXY_IMAGE,
                # The node brings this up with the real ``docker compose`` CLI, which INTERPOLATES
                # ``$VAR`` across the compose doc against the NODE's host env before the container
                # runs. ``$PROXY_TOKEN`` / ``$ALLOWLIST_DOMAINS`` are unset there, so an un-escaped
                # command would have them blanked to empty — Squid would then ``htpasswd`` an EMPTY
                # password and build an EMPTY allowlist, so every authenticated client (git sending
                # the real ``agent:<token>``) gets a 407. Escape ``$`` → ``$$`` so compose emits a
                # literal ``$`` and the CONTAINER's bash expands them from ``environment:`` below at
                # runtime. (Only the runtime-expanded command is escaped; ``environment:`` values are
                # literals compose passes through untouched.)
                "command": ["bash", "-lc", _egress_proxy_runtime_command().replace("$", "$$")],
                "environment": proxy_policy_env(
                    NetworkAllowlist(domains=list(domains)), token,
                ),
                "healthcheck": {
                    "test": ["CMD-SHELL", "bash -lc '</dev/tcp/127.0.0.1/8080'"],
                    "interval": "2s",
                    "timeout": "2s",
                    "retries": 60,
                },
                # both nets: reachable from main (internal) AND the internet (default).
                "networks": ["pier-egress-internal", "default"],
            },
        },
        "networks": {"pier-egress-internal": {"internal": True}},
    }
    return compose, [main_ref, _EGRESS_PROXY_IMAGE]


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
    os.environ.get("XRLENV_PIER_ACQUIRE_QUEUE_TIMEOUT_S", "") or 240.0
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
            "XrlenvPierEnvironmentCluster requires XRLENV_GRPC_HOST "
            "to point at the control plane (e.g. ``XRLENV_GRPC_HOST="
            "127.0.0.1`` for a local ``xrlenv up`` daemon). Set "
            "XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN before launching "
            "the harness, or pick the local-mode "
            "``xrlenv_plugins.harbor:XrlenvPierEnvironment`` "
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


class XrlenvPierEnvironmentCluster(XrlenvPierEnvironment):
    """harbor ``BaseEnvironment`` whose container ops route through
    the xrlenv cluster instead of local ``docker compose`` + ``docker
    cp``.

    harbor users opt in by setting ``environment.import_path:
    xrlenv_plugins.harbor:XrlenvPierEnvironmentCluster`` in their
    ``job.yaml``, exactly the way they'd pick ``e2b``, ``modal``, or
    ``daytona``. The local-mode :class:`XrlenvPierEnvironment` is
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
    script (e.g. ``examples/benchmarks-onboarding/terminal-bench-2/
    scripts/build-task-images.sh``). The image tag the cluster
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
        mounts_json: list[ServiceVolumeConfig] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Cluster mode does NOT build an agent-preinstalled image, so we must NOT
        # let pier think the agent is baked in. pier passes ``agent.install_spec()``
        # into the environment; an installed agent's ``setup()`` treats a matching
        # ``environment.agent_install_spec`` as *already preinstalled* and SKIPS its
        # runtime ``install()`` — which would leave the agent binary absent on the
        # cluster container. Drop it here (and advertise ``preinstall_agents=False``
        # in ``capabilities``) so pier installs the agent at runtime via our
        # ``exec``. LocalDocker (the base class) keeps it — there pier's compose path
        # can genuinely build a preinstalled image. Moot for the OracleAgent (no
        # install spec); required for the installed-agent / egress path.
        kwargs.pop("agent_install_spec", None)
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            keep_containers,
            mounts_json,
            *args,
            **kwargs,
        )
        # Belt-and-suspenders: if a future pier passes it positionally or the base
        # sets a default, force it off on the instance too.
        self.agent_install_spec = None
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
        # NB: ``self._egress_proxy_env`` is already initialized to ``{}`` by pier's
        # DockerEnvironment.__init__, and pier's inherited ``agent_process_env``
        # already injects it into agent commands — so we don't redeclare it or
        # override that method; ``_start_egress_compose_project`` just sets it.

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
        - ``preinstall_agents=False``: the cluster does not build an
          agent-preinstalled image, so pier must run each installed
          agent's runtime ``install()`` via our ``exec`` (paired with
          clearing ``agent_install_spec`` in ``__init__``). Advertising
          True would make pier skip install and leave the binary absent.
        - ``filtered_egress=True``: an offline task carrying an agent
          ``network_allowlist`` runs behind the Squid egress-proxy sidecar
          (§4b) — ``main`` on an ``internal: true`` network with the proxy
          its only route out, allowlist → squid ``dstdomain``, and the
          proxy env injected into agent commands only (``agent_process_env``).
          The OracleAgent path supplies no allowlist, so the proxy is never
          synthesized for the oracle gate (it acquires normally).
        """
        return EnvironmentCapabilities(
            gpus=False,
            disable_internet=True,
            filtered_egress=True,
            preinstall_agents=False,
            windows=False,
            mounted=False,
            docker_compose=self._multi_service_compose() is not None,
        )

    def _egress_domains(self) -> list[str] | None:
        """The agent network-allowlist domains **iff** this task should run behind
        the Squid egress proxy — i.e. it's offline (``allow_internet=False``) AND a
        non-empty ``network_allowlist`` was supplied (the installed-agent path).
        Mirrors pier's ``_prepare_egress_proxy_compose`` predicate. ``None`` → no
        proxy (the oracle path, and any online task).

        **Open-install refinement** (``notes/pier-open-install-egress.md``): if EVERY
        allowlisted host is a bare **v4 IP literal**, the Squid *domain* proxy buys
        nothing — its only job would be to pass a fixed IP through. Skipping it routes
        the task onto the single-container **OPEN** path (``acquire_container``,
        ``network_mode=None``), so the trusted install phase runs with a DIRECT route
        (no per-request proxy tax — the ~9× install slowdown that blew the 360s setup
        window), and the consumer seals the agent's **run** phase to those IPs via
        :meth:`apply_egress` (spec-07 iptables — a hard, un-undoable IP allowlist).
        Same effective egress restriction, no Squid. The proxy stays only when the
        allowlist carries a **hostname** (needs DNS-aware domain filtering)."""
        allowlist = getattr(self, "network_allowlist", None)
        domains = list(getattr(allowlist, "domains", []) or []) if allowlist else []
        if not (domains and self.task_internet_disabled()):
            return None
        if all(_is_ipv4_literal(d) for d in domains):
            return None  # iptables-sealable → single-container open-install (see docstring)
        return domains


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

    def _is_verifier_session(self) -> bool:
        """True when this env is pier's **separate verifier** container.

        pier's ``Trial._verify_with_separate_environment`` builds the verifier env's
        ``session_id`` via ``_separate_verifier_session_id`` — always containing the
        literal ``__verifier__`` marker (only dropped in the rare >63-char overflow
        case, where the id is truncated to ``<prefix>__<sha1>``). It also passes
        ``agent_install_spec=None`` and ``network_allowlist=None`` for the verifier,
        so the marker plus "no install spec" is the reliable signal. We route the
        verifier onto a special image-resolution + ``/tests``-upload path (§2.5)."""
        return "__verifier__" in (self.session_id or "")

    def _verifier_base_image(self) -> str | None:
        """Resolve the base image for a **separate verifier** container.

        DeepSWE ships ``[verifier.environment]`` **without** a ``docker_image``, so
        ``resolve_effective_verifier_env_config`` returns that block and
        ``task_env_config.docker_image`` is ``None`` — a plain precedence would fall
        to ``hb__<env>`` (nonexistent). The verifier ``tests/Dockerfile`` is
        ``FROM <the task's ECR base>`` + a few ``COPY``s, and pier does not build it
        in separate mode; we run that base image directly and upload the tests
        ourselves. Resolve the base ref by, in order:

        1. the ``FROM`` of ``<environment_dir>/Dockerfile`` (the verifier build
           context is the task's ``tests/`` dir — the authoritative "what image does
           grading run on"), skipping an unresolved ``$``-ARG form;
        2. the **parent task's** top-level ``[environment] docker_image`` (read the
           sibling ``task.toml`` at ``<environment_dir>/../task.toml``).

        Returns ``None`` if neither resolves (caller then falls through)."""
        env_dir = Path(self.environment_dir)
        # (1) tests/Dockerfile FROM
        dockerfile = env_dir / "Dockerfile"
        if dockerfile.is_file():
            try:
                ref = _parse_dockerfile_from_ref(
                    dockerfile.read_text(errors="replace"),
                )
                if ref:
                    return ref
            except OSError:
                pass
        # (2) parent task.toml top-level [environment] docker_image
        task_toml = env_dir.parent / "task.toml"
        if task_toml.is_file():
            try:
                doc = tomllib.loads(task_toml.read_text())
                image = (doc.get("environment") or {}).get("docker_image")
                if image:
                    return str(image)
            except (OSError, tomllib.TOMLDecodeError):
                pass
        return None

    def _resolve_image_ref(self) -> str:
        """Pick the image ref the node will acquire, by precedence:

        1. ``XRLENV_PIER_IMAGE_TEMPLATE`` (env), if set — a ``str.format``
           template with ``{task_id}`` (the task directory name) and
           ``{environment_name}`` fields. This is how a benchmark whose images live
           in a private registry under a *derived* name points the cluster at them
           without a per-task ``docker_image`` and without subclassing this class.
        2. ``task_env_config.docker_image`` — an upstream-published prebuilt
           (e.g. terminal-bench-2's ``alexgshaw/<task>:<rev>``, or DeepSWE's ECR
           ref on the *agent* env).
        3. **Separate-verifier fallback** — when this is the verifier container and
           the resolved config carries no ``docker_image`` (DeepSWE's case), resolve
           the base image from ``tests/Dockerfile`` ``FROM`` / the parent task's
           top-level ``docker_image`` (see :meth:`_verifier_base_image`).
        4. ``hb__<environment_name>`` — the locally-built harbor convention.
        """
        template = os.environ.get("XRLENV_PIER_IMAGE_TEMPLATE")
        if template:
            return template.format(
                task_id=Path(self.environment_dir).parent.name,
                environment_name=self.environment_name,
            )
        if self.task_env_config.docker_image:
            return str(self.task_env_config.docker_image)
        # Separate-verifier: the verifier's task_env_config has no docker_image, so
        # resolve the base image from the tests build context / parent task.toml
        # rather than acquiring a nonexistent ``hb__<env>`` (see §2.5).
        if self._is_verifier_session():
            verifier_image = self._verifier_base_image()
            if verifier_image:
                return verifier_image
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

    def _image_namespace_tag(self) -> tuple[str | None, str]:
        """Parse ``(namespace, tag)`` for sub-dir sidecar refs out of
        ``XRLENV_PIER_IMAGE_TEMPLATE`` by splitting on the literal ``{task_id}``
        placeholder (``<prefix>/{task_id}:<tag>`` → ``namespace=<prefix>``,
        ``tag=<tag>``). Returns ``(None, "main")`` when the template is absent or
        lacks ``{task_id}`` — :func:`compose.assemble_project` then fails loud only
        if the task actually has sub-dir build services (a task with none needs no
        namespace). The split is on the placeholder, never on the host's ``:``, so a
        registry ``host:port`` prefix is unambiguous."""
        template = os.environ.get("XRLENV_PIER_IMAGE_TEMPLATE")
        if not template or "{task_id}" not in template:
            return None, "main"
        prefix, _, suffix = template.partition("{task_id}")
        namespace = prefix.rstrip("/") or None
        tag = "main"
        if suffix.startswith(":"):
            candidate = suffix[1:]
            # The corpus uses ':main'. Ignore anything with a path separator or an
            # unresolved placeholder (e.g. '{environment_name}') → fall back to main.
            if candidate and "/" not in candidate and "{" not in candidate:
                tag = candidate
        return namespace, tag

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
            "pier.session_id": self.session_id,
            "pier.environment_name": self.environment_name,
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

    async def _maybe_upload_verifier_tests(self) -> None:
        """Upload the tests build context to ``/tests`` for a **separate verifier**
        container (§2.5). No-op for the agent container.

        pier's separate-verifier path constructs ``Verifier(skip_tests_upload=True)``
        (hardcoded, no env hook) on the assumption the grader is baked into the
        verifier image. DeepSWE's verifier image is just the prebuilt ECR base
        (pier does not build ``tests/Dockerfile`` in separate mode), so nothing
        populates ``/tests`` and ``test.sh`` is absent → verify fails. We reproduce
        the ``tests/Dockerfile`` COPY by uploading the build context (the task's
        ``tests/`` dir == ``self.environment_dir``) to ``/tests``, excluding the
        ``Dockerfile`` itself (it isn't a test artifact). ``upload_dir`` is inherited
        (routed through the cluster ``put_archive``). The in-container ``/tests`` is
        created by ``put_archive``'s parent-dir contract; ``mkdir -p`` first to be
        safe on images without it."""
        if not self._is_verifier_session():
            return
        env_dir = Path(self.environment_dir)
        if not env_dir.is_dir():
            self.logger.warning(
                "verifier tests upload: environment_dir %s is not a dir; skipping",
                env_dir,
            )
            return
        tests_dir = str(EnvironmentPaths.tests_dir)
        mk = await self.exec(f"mkdir -p {tests_dir} && chmod 777 {tests_dir}", user="root")
        if mk.return_code != 0:
            raise RuntimeError(
                f"cluster verifier start: /tests mkdir failed "
                f"(exit={mk.return_code}): stderr={mk.stderr!r}",
            )
        # upload_dir copies the *contents* of source into target (harbor/pier
        # ``compose cp src/. dst`` semantics). We upload the tests dir contents; the
        # verifier ``Dockerfile`` is harmless but not a test artifact, so we stage a
        # filtered copy without it to keep ``/tests`` byte-identical to the COPY.
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory(prefix="xrlenv-pier-tests-") as staging:
            staged = Path(staging) / "tests"
            shutil.copytree(
                env_dir, staged,
                ignore=shutil.ignore_patterns("Dockerfile", "docker-compose.yaml",
                                              "docker-compose.yml"),
            )
            await self.upload_dir(source_dir=staged, target_dir=tests_dir)
        # DEBUG, not INFO: this fires once per task (every verifier setup) with a
        # constant target, so at sweep scale it floods the log — hundreds of
        # identical lines that bury the actual errors. The failure path
        # (mkdir/upload) raises with detail, so no diagnostic is lost by demoting
        # the success line.
        self.logger.debug(
            "cluster verifier: uploaded tests build context -> %s", tests_dir,
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
        namespace, tag = self._image_namespace_tag()
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
        # Separate-verifier /tests upload (§2.5) — no-op unless this is a verifier
        # session. A multi-service verifier task routes here.
        await self._maybe_upload_verifier_tests()

    async def _start_egress_compose_project(self, domains: list[str]) -> None:
        """Bring up the Squid egress-proxy compose (§4b): ``main`` (the task image,
        offline on an ``internal`` network) + the proxy sidecar (its only route out,
        squid allowlist = ``domains``). Records ``self._egress_proxy_env`` so
        ``agent_process_env`` routes the agent's commands through the proxy."""
        main_ref = self._resolve_image_ref()
        token = new_proxy_token()
        compose, images = build_egress_proxy_compose(
            main_ref=main_ref,
            main_command=list(_hc.MAIN_KEEPALIVE),
            domains=domains,
            token=token,
        )
        # Whole-stack footprint (scheduler reserve): main's declared cpu/mem (or the
        # default budget) + a small proxy reserve. Cap main in the doc too — cluster
        # mode IS pier's resources-override for main.
        cpu_limit, mem_limit_bytes = self._effective_cpu_mem_limits()
        main_svc = compose["services"][_hc.MAIN_SERVICE]
        if cpu_limit is not None:
            main_svc["cpus"] = cpu_limit
        if mem_limit_bytes is not None:
            main_svc["mem_limit"] = int(mem_limit_bytes)
        footprint_cpu = (
            cpu_limit if cpu_limit is not None else _DEFAULT_MAIN_CPU
        ) + _hc.DEFAULT_SIDECAR_CPU
        footprint_mem_bytes = (
            mem_limit_bytes if mem_limit_bytes is not None else _DEFAULT_MAIN_MEM_BYTES
        ) + _hc.DEFAULT_SIDECAR_MEM_MB * 1024 * 1024

        compose_yaml = yaml.safe_dump(compose, sort_keys=False)
        client = _client_from_env()
        self._xrlenv_client = client
        try:
            self._xrlenv_session = await client.acquire_compose_project(
                compose_yaml=compose_yaml,
                images=images,
                footprint_cpu=footprint_cpu,
                footprint_mem_bytes=int(footprint_mem_bytes),
                main_service=_hc.MAIN_SERVICE,
                labels=self._acquire_labels(),
                task_key=self.environment_name,
                queue_timeout_s=_ACQUIRE_QUEUE_TIMEOUT_S,
                # Generous compose-up window: the proxy installs squid at container
                # start (no prebuilt image), so ``depends_on: service_healthy`` waits
                # on the apt-install + squid boot before main is up.
                up_timeout_s=_EGRESS_UP_TIMEOUT_S,
            )
        except Exception:
            await client.close()
            self._xrlenv_client = None
            raise
        # Agent commands (pier's installed-agent _exec) get the proxy env via
        # agent_process_env; verifier/task exec bypass it.
        self._egress_proxy_env = proxy_environment(
            token, EGRESS_PROXY_SERVICE, EGRESS_PROXY_PORT,
        )
        await self._setup_cluster_log_dirs()
        await self._maybe_upload_verifier_tests()

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

        # §4b — an offline task carrying an agent network-allowlist runs behind the
        # Squid egress proxy (single-service task + proxy sidecar). Takes precedence
        # over the raw acquire; never fires for the OracleAgent (no allowlist).
        egress_domains = self._egress_domains()
        if egress_domains is not None:
            await self._start_egress_compose_project(egress_domains)
            return

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
        # Sysbox file transfer routes through the exec-based path (tar+base64 via
        # exec) instead of docker put_archive/get_archive, which 500 on the
        # idmapped /etc/resolv.conf. Also relaxes uploaded tar modes to world r-x.
        self._sysbox_upload = bool(container_runtime) and container_runtime != "runc"

        try:
            self._xrlenv_session = await client.acquire_container(
                image=image_ref,
                command=(["/sbin/init"] if systemd_init else ["sleep", "infinity"]),
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

        # Separate-verifier: pier hardcodes ``Verifier(skip_tests_upload=True)`` on
        # the separate path (it assumes the tests are baked into the verifier
        # image), but DeepSWE's verifier image is the plain prebuilt ECR base — the
        # grader lives only in the unbuilt ``tests/Dockerfile`` COPY layer. So we
        # reproduce that COPY by uploading the tests build context to ``/tests``
        # ourselves (§2.5). No-op for the agent container.
        await self._maybe_upload_verifier_tests()

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
                "stop. Use the local-mode XrlenvPierEnvironment if "
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
                "XrlenvPierEnvironmentCluster.exec called before "
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
                "XrlenvPierEnvironmentCluster.apply_egress called before "
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
