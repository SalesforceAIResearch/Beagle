"""Docker sandbox backend (spec 01 §"Docker adapter").

One container per sandbox. Lifecycle is explicit (we don't use ``--rm``); cgroup
limits come from :class:`ResourceSpec`; the stub talks to the node agent over a
per-sandbox unix socket bind-mounted into ``/run/xrlenv/stub.sock``. Snapshot,
``port_forward``, and streaming primitives are stubbed for Slice 1 — they will
land in subsequent slices.

The driver wraps the synchronous ``docker-py`` client with ``asyncio.to_thread``
because docker-py is blocking. We accept the thread-pool overhead in exchange
for not maintaining an HTTP/Unix-socket client by hand.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import docker
from pydantic import BaseModel, ConfigDict

from xrlenv.backends.base import (
    ExecChunk,
    ImageInUse,
    ImageRecord,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxBackend,
    SandboxCapabilities,
    SandboxHandle,
    ServiceSpec,
    SnapshotID,
    TemplateRef,
)

if TYPE_CHECKING:
    from docker.models.containers import Container


# ──────────────────────────────────────────────────────────────────────────────
# Capabilities (spec 01 — Docker is a container runtime; snapshots are
# best-effort docker commit; live-state is not captured).
# ──────────────────────────────────────────────────────────────────────────────

DOCKER_CAPABILITIES = SandboxCapabilities(
    supports_snapshot=True,
    supports_chainable_snapshot=False,
    live_state_captured=False,
    supports_port_forward=True,
    supports_gpu=False,
    isolation_class="container",
    fast_create_p50_ms=300,
)


StubTransport = Literal["uds", "tcp"]


def _default_stub_transport() -> StubTransport:
    """Pick the right stub transport for this host.

    - **Linux**: uds. Container creates ``/run/xrlenv/stub.sock`` under a
      bind-mounted directory; the host opens the file directly. This is the
      spec-01 default.
    - **macOS / Windows**: tcp. Docker-Desktop's host↔VM filesystem bridge
      surfaces uds files as plain files but does not route socket connections
      across the boundary, so we fall back to a published TCP port on
      ``127.0.0.1``. This trade-off is local-laptop-only; cloud nodes (Linux)
      use UDS as the spec mandates.
    """
    return "uds" if platform.system().lower() == "linux" else "tcp"


class DockerBackendConfig(BaseModel):
    """Per-node configuration for the Docker driver."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    runs_root: Path
    """Host directory under which per-sandbox uds dirs are created."""

    xrlenv_pkg_path: Path
    """Host path of the ``xrlenv`` package (mounted read-only into the container
    so the stub can run as ``python3 -m xrlenv.sandbox_stub`` without baking the
    package into every template image)."""

    xrlenv_plugins_path: Path | None = None
    """Host path of the ``xrlenv_plugins`` package, mounted alongside
    ``xrlenv`` so plug-in adapter modules
    (``xrlenv_plugins.benchmarks.<name>.adapter``) import natively
    inside the sandbox without any pip step. Auto-defaulted in
    :func:`xrlenv.control.runtime` and :class:`xrlenv.node.cli` to the
    sibling of :attr:`xrlenv_pkg_path` when present; pass ``None`` to
    skip the mount (e.g. tests that don't exercise plug-ins)."""

    extra_plugin_roots: tuple[Path, ...] = ()
    """D22 — additional plug-in roots to bind-mount into every sandbox.

    Each entry is a directory whose immediate child is named
    ``xrlenv_plugins`` (i.e. the parent of an external
    ``xrlenv_plugins/<category>/<name>/`` namespace-package
    contribution). The runtime mounts each entry read-only at
    ``/opt/xrlenv-extras/<idx>`` inside the sandbox and prepends each
    mount target to ``PYTHONPATH``. PEP-420 namespace-package semantics
    (``xrlenv_plugins/`` and ``xrlenv_plugins/benchmarks/`` have no
    ``__init__.py``) merge contributions across PYTHONPATH entries, so
    an external pip-package's adapter at
    ``site-packages/xrlenv_plugins/benchmarks/foo/adapter.py``
    imports natively from inside the sandbox once
    ``site-packages/`` is on this list.

    Populated at node boot from the operator's
    :envvar:`XRLENV_TEMPLATE_DIRS` env var + ``xrlenv.benchmarks``
    entry-points; see
    :func:`xrlenv.control.template_discovery.find_external_template_dir_manifests`
    and :func:`xrlenv.control.template_discovery.find_entry_point_manifest_files`.
    Defaults to ``()`` — backwards compatible with the pre-D22 mount
    set."""

    stub_startup_timeout_s: float = 30.0
    """How long to wait for the in-container stub uds to appear after create."""

    stub_transport: StubTransport = "uds"
    """Wire transport for the stub. Use :func:`_default_stub_transport` to auto-
    select; :func:`build_local_runtime` does that for you."""

    stub_tcp_container_port: int = 49100
    """Container-side TCP port the stub binds when ``stub_transport='tcp'``.
    Mapped to an ephemeral host port via Docker's ``-p 0:<port>``."""


#: Per-request HTTP read timeout (seconds) applied to the docker-py
#: client constructed by :class:`DockerBackend` when no explicit client
#: is supplied. Picked to match
#: :py:attr:`xrlenv.node.image_cache.ImageCacheConfig.default_pull_timeout_s`
#: (which lives in a separate module to avoid a backends → node import
#: cycle) so all three layers of the cold-pull deadline stack share the
#: same ceiling:
#:
#: 1. control-plane → node gRPC wire (``acquire_container``, 600 s)
#: 2. node-side image cache pull deadline (``ensure_present``, 600 s)
#: 3. docker-py HTTP socket — *this constant* (600 s)
#:
#: docker-py's own default is 60 s (``docker.constants.DEFAULT_TIMEOUT_SECONDS``)
#: which fires inside the daemon's ``POST /containers/create`` whenever
#: the underlying image isn't already present locally and the daemon
#: triggers an implicit pull: the urllib3 ``UnixHTTPConnectionPool``
#: raises ``ReadTimeout`` while the daemon is still streaming layers.
#: SWE-bench Pro images are 5-15 GB and routinely exceed 60 s on a cold
#: cache, so the pre-fix node-agent surfaced
#: ``gRPC error UNKNOWN: ReadTimeout`` even though the pull itself would
#: succeed given enough wall time.
#:
#: **Cross-file invariant**: if you bump
#: ``ImageCacheConfig.default_pull_timeout_s``, bump this constant too.
#: The grep-for-invariant comment on that field references this one.
DOCKER_CLIENT_HTTP_TIMEOUT_S: float = 600.0

#: Minimum interval between containerd content-store GC passes
#: (``ctr content prune``). The prune takes the containerd metadata lock
#: and walks the whole content store; running it after *every* pull turned
#: into a daemon-wide lock storm at high pull concurrency (concurrent pulls
#: each firing a prune → ``docker ps`` / ``docker system df`` starved →
#: command timeouts). Debouncing to one prune per interval decouples GC
#: cadence from pull concurrency. Operator-tunable.
_CONTENT_GC_MIN_INTERVAL_S: float = float(
    os.environ.get("XRLENV_CONTENT_GC_MIN_INTERVAL_S", "60"),
)


class DockerBackend(SandboxBackend):
    name = "docker"
    capabilities = DOCKER_CAPABILITIES

    def __init__(
        self,
        config: DockerBackendConfig,
        client: docker.DockerClient | None = None,
    ) -> None:
        self._config = config
        # See :data:`DOCKER_CLIENT_HTTP_TIMEOUT_S` for the why; caller-
        # supplied clients pass through unchanged so test fixtures that
        # inject their own mock aren't affected.
        self._client = (
            client
            if client is not None
            else docker.from_env(timeout=DOCKER_CLIENT_HTTP_TIMEOUT_S)
        )
        config.runs_root.mkdir(parents=True, exist_ok=True)
        # Debounce + single-flight state for the containerd content-store
        # GC (see ``_gc_containerd_content``). Without this, a burst of
        # concurrent pulls each fired a ``ctr content prune`` and the
        # contended containerd metadata lock starved ``docker ps`` /
        # ``docker system df``.
        self._last_content_gc_monotonic: float = 0.0
        self._content_gc_running: bool = False
        # Cache of ``docker info → DockerRootDir`` (the storage volume).
        # It's stable for the daemon's lifetime, so we resolve it once and
        # reuse it: calling ``docker info`` on every disk check is slow
        # under load and, on a transient failure, silently fell back to
        # ``/`` — which on a node whose data-root is a separate EBS volume
        # mis-measures free disk against the small root fs and makes the
        # cache evictor thrash. ``None`` = not yet resolved.
        self._docker_root_dir: str | None = None

    @property
    def docker_client(self) -> docker.DockerClient:
        """Public accessor for the underlying docker-py client.

        Used by ``RawContainerManager`` (P1.7.A.1) to talk to the
        daemon directly for case-2/3 raw containers — bypasses the
        EnvAdapter / stub-runtime path that ``create()`` uses.
        Other callers should prefer the typed surface (``create``,
        ``destroy``, ``put_archive``, ``image_exists``, …) so the
        backend abstraction stays portable to phase-2 CubeSandbox.
        """
        return self._client

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def create(
        self,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
    ) -> SandboxHandle:
        if network_policy == "egress-allowlist":
            raise NotImplementedError("egress-allowlist lands in phase 1 (spec 07)")

        sb_id = uuid.uuid4().hex
        transport = self._config.stub_transport
        host_run_dir = self._config.runs_root / sb_id
        stub_uds_path = host_run_dir / "stub.sock"
        # Only the uds transport needs a per-sandbox host directory (it gets
        # bind-mounted into the container at /run/xrlenv). For tcp transport
        # there is no host-side artifact to manage.
        #
        # The directory is mode 0o777 because the host node-agent runs as the
        # xrlenv system user (uid ~990 on AL2023) but the in-sandbox stub
        # process runs as the Dockerfile's USER sandbox (uid 1000). Without a
        # mode that lets uid 1000 create stub.sock inside the bind-mount, the
        # stub crashes at startup with `bind() PermissionError [Errno 13]`.
        # macOS Docker Desktop hides this because its file-sharing layer
        # remaps uids; Linux native does not. Wide-open mode is acceptable
        # here because (a) the dir is per-sandbox under a uuid4-named path,
        # so external callers can't enumerate it, and (b) the parent
        # ``runs_root`` (default ``/var/lib/xrlenv/runs``) is mode 0o755
        # owned by ``xrlenv:xrlenv``, gating list/cd to the system user.
        if transport == "uds":
            host_run_dir.mkdir(parents=True, exist_ok=True)
            host_run_dir.chmod(0o777)
        # D22 — assemble the volume + PYTHONPATH layering for external
        # plug-in roots. Each entry mounts at /opt/xrlenv-extras/<idx>
        # (positional index, not basename-derived, so two same-basename
        # roots never collide). The PYTHONPATH lists the in-tree mount
        # first then each extras prefix in order — matches the discovery
        # order and gives Python a deterministic first-wins resolution
        # if two roots ever ship the same <category>/<name>.
        extras_volumes: dict[str, dict[str, str]] = {}
        extras_pythonpath: list[str] = []
        for idx, root in enumerate(self._config.extra_plugin_roots):
            target = f"/opt/xrlenv-extras/{idx}"
            extras_volumes[str(root)] = {"bind": target, "mode": "ro"}
            extras_pythonpath.append(target)
        pythonpath = ":".join(["/opt/xrlenv-pkg", *extras_pythonpath])
        run_kwargs: dict[str, Any] = {
            "image": template.image,
            "name": f"xrlenv-sb-{sb_id}",
            "detach": True,
            "init": True,  # tini as PID 1 (spec 01)
            "auto_remove": False,
            "labels": {
                "xrlenv.sandbox_id": sb_id,
                "xrlenv.template": template.name,
            },
            "volumes": {
                # Mount the xrlenv package so the stub can `import xrlenv`.
                # We mount one level above so PYTHONPATH=/opt/xrlenv-pkg works.
                str(self._config.xrlenv_pkg_path): {
                    "bind": "/opt/xrlenv-pkg/xrlenv",
                    "mode": "ro",
                },
                # Same trick for xrlenv_plugins/ so plug-in adapter modules
                # (e.g. ``xrlenv_plugins.benchmarks.<your_plugin>.adapter``)
                # import natively inside the sandbox. The mount is
                # optional — pass ``xrlenv_plugins_path=None`` to skip
                # when no plug-ins are in play.
                **(
                    {
                        str(self._config.xrlenv_plugins_path): {
                            "bind": "/opt/xrlenv-pkg/xrlenv_plugins",
                            "mode": "ro",
                        },
                    }
                    if self._config.xrlenv_plugins_path is not None
                    else {}
                ),
                **extras_volumes,
            },
            "environment": {
                "PYTHONPATH": pythonpath,
            },
            "mem_limit": resources.mem_limit_bytes,
            "cpu_period": 100_000,
            "cpu_quota": int(resources.cpu_limit * 100_000),
            "network_mode": (
                "none" if (network_policy == "none" and transport != "tcp") else "bridge"
            ),
        }

        if transport == "uds":
            run_kwargs["volumes"][str(host_run_dir)] = {"bind": "/run/xrlenv", "mode": "rw"}
            run_kwargs["environment"]["XRLENV_STUB_UDS"] = "/run/xrlenv/stub.sock"
            run_kwargs["command"] = [
                # ``python3`` is universal — every Debian-derived base
                # ships it (``python:3.X-slim`` has python+python3+python3.X
                # symlinks; ``apt install python3`` on bare ubuntu/debian
                # only ships ``python3``, no bare ``python``). Using the
                # universal name keeps the stub-runtime layer free of a
                # ``python-is-python3`` install just to satisfy this CMD.
                "python3", "-m", "xrlenv.sandbox_stub",
                "--uds", "/run/xrlenv/stub.sock",
            ]
        else:
            container_port = self._config.stub_tcp_container_port
            run_kwargs["ports"] = {f"{container_port}/tcp": ("127.0.0.1", None)}
            run_kwargs["command"] = [
                # ``python3`` (not ``python``) — see uds-branch comment.
                "python3", "-m", "xrlenv.sandbox_stub",
                "--bind-host", "0.0.0.0",
                "--bind-port", str(container_port),
            ]

        container: Container = await asyncio.to_thread(
            self._client.containers.run, **run_kwargs
        )

        try:
            endpoint = await self._await_stub_ready(
                container=container,
                transport=transport,
                stub_uds_path=stub_uds_path,
                container_port=self._config.stub_tcp_container_port,
                timeout_s=self._config.stub_startup_timeout_s,
            )
        except TimeoutError as exc:
            # Capture the failed container's stdout/stderr + exit metadata
            # *before* force-remove — otherwise the operator gets a generic
            # "stub did not become reachable" with zero context, since the
            # container is gone by the time `docker logs <name>` runs.
            tail = await asyncio.to_thread(_collect_failure_diagnostics, container)
            await asyncio.to_thread(_force_remove, container)
            shutil.rmtree(host_run_dir, ignore_errors=True)
            raise TimeoutError(
                f"sandbox stub ({transport}) did not become reachable within "
                f"{self._config.stub_startup_timeout_s}s "
                f"(template={template.name}, image={template.image}, "
                f"sandbox_id={sb_id}). {tail}"
            ) from exc

        return SandboxHandle(
            id=sb_id,
            backend=self.name,
            backend_ref=container.id,
            stub_endpoint=endpoint,
        )

    async def destroy(self, sb: SandboxHandle) -> None:
        try:
            container = await asyncio.to_thread(self._client.containers.get, sb.backend_ref)
        except Exception:
            # Already gone — destroy is idempotent (spec 01).
            self._cleanup_run_dir(sb)
            return

        await asyncio.to_thread(_force_remove, container)
        self._cleanup_run_dir(sb)

    async def list_owned_sandboxes(self) -> list[SandboxHandle]:
        """List every container labeled ``xrlenv.sandbox_id=*`` on this host.

        Spec 09 GC layer 2: the node-agent calls this on startup to find
        sandboxes left behind by a previous process (OOM kill, systemd
        restart, panic). Containers without the ``xrlenv.sandbox_id``
        label are foreign and never returned. The returned handles carry
        empty ``stub_endpoint`` because the original bind path / TCP
        port isn't recoverable for orphans — :py:meth:`destroy` only
        needs ``backend_ref`` (the container id).
        """
        containers = await asyncio.to_thread(
            self._client.containers.list,
            all=True,
            filters={"label": "xrlenv.sandbox_id"},
        )
        out: list[SandboxHandle] = []
        for container in containers:
            sb_id = container.labels.get("xrlenv.sandbox_id")
            if not sb_id:
                continue
            out.append(
                SandboxHandle(
                    id=sb_id,
                    backend=self.name,
                    backend_ref=container.id,
                    stub_endpoint="",
                )
            )
        return out

    # ── Image cache surface (spec 15) ────────────────────────────────────────

    async def list_images(
        self, *, include_shared_size: bool = False,
    ) -> list[ImageRecord]:
        """List Docker images present locally that the cache manager tracks.

        Each :class:`ImageRecord` carries ``size_bytes`` (the full image
        footprint including shared layers, as ``docker images`` reports).
        When ``include_shared_size`` is set, it also carries
        ``shared_size_bytes`` (the portion of that footprint belonging to
        layers also present in another tagged image on the same node,
        sourced from Docker's ``GET /system/df`` endpoint) so ``xrlenv
        build calibrate`` can derive ``unique = size - shared``.

        ``include_shared_size`` defaults to ``False`` because ``system
        df`` walks the entire layer graph and is slow on a node with a
        large catalog — it was the call that timed out the ``/images``
        view and starved the eviction loop under heavy pulls. The hot
        path (eviction, adaptive-control stats) never needs it; only the
        calibrate path turns it on. When ``df`` is requested but fails
        (older daemon, transient socket hiccup), ``shared_size_bytes``
        stays ``None`` and callers fall back to ``size_bytes``.
        """
        images = await asyncio.to_thread(self._client.images.list)
        # Per-image ``SharedSize`` from ``docker system df`` — only when
        # asked, since the endpoint walks the layer graph once per cached
        # image (the slow path). Off by default keeps this a cheap
        # ``images.list``.
        shared_by_id: dict[str, int] = {}
        if include_shared_size:
            try:
                df = await asyncio.to_thread(self._client.df)
                for entry in (df or {}).get("Images") or ():
                    img_id = entry.get("Id")
                    shared = entry.get("SharedSize")
                    if isinstance(img_id, str) and isinstance(shared, int) \
                            and shared >= 0:
                        shared_by_id[img_id] = shared
            except Exception:
                # Older daemons may not surface SharedSize, or the df call
                # may transiently fail; degrade to the legacy size-only
                # path. Callers see ``shared_size_bytes=None`` and revert
                # to the docker-images-reported ``size_bytes`` for budget
                # accounting (the pre-calibrate behavior).
                shared_by_id = {}
        out: list[ImageRecord] = []
        for img in images:
            tags = list(img.tags or [])
            digest = img.attrs.get("RepoDigests")
            digest_str: str | None = None
            if digest:
                # RepoDigests entries look like "name@sha256:...".
                digest_str = str(digest[0])
            size = int(img.attrs.get("Size") or 0)
            shared_size = shared_by_id.get(str(img.id)) \
                if shared_by_id else None
            # Surface ``Config.Labels`` so the spec-15 ownership classifier
            # can read ``org.xrlenv.owned`` / ``org.xrlenv.role``. Docker
            # exposes labels under ``attrs["Config"]["Labels"]`` for tagged
            # images; for some pulled images the same dict is mirrored at
            # ``attrs["ContainerConfig"]["Labels"]``. Prefer Config; fall
            # back to ContainerConfig; treat missing as empty.
            raw_labels = (
                (img.attrs.get("Config") or {}).get("Labels")
                or (img.attrs.get("ContainerConfig") or {}).get("Labels")
                or {}
            )
            labels = {str(k): str(v) for k, v in raw_labels.items() if v is not None}
            if not tags:
                # Untagged (dangling) images surface under their id so the
                # cache manager can still see + evict them.
                out.append(
                    ImageRecord(
                        name=img.id, size_bytes=size,
                        shared_size_bytes=shared_size,
                        digest=digest_str, labels=labels,
                    ),
                )
                continue
            for tag in tags:
                out.append(
                    ImageRecord(
                        name=tag, size_bytes=size,
                        shared_size_bytes=shared_size,
                        digest=digest_str, labels=labels,
                    ),
                )
        return out

    async def image_exists(self, image: str) -> bool:
        """Return ``True`` iff ``image`` resolves locally via the
        Docker daemon, accepting either tag form
        (``xrlenv/hello-shell:0.1``) or digest form
        (``xrlenv/hello-shell@sha256:...``). Wraps
        ``client.images.get(ref)`` and translates ``ImageNotFound``
        into ``False`` so the image cache doesn't trigger a pull
        for a locally-built image whose digest pin came from the
        local content-addressed Id rather than ``RepoDigests``."""

        def _exists() -> bool:
            try:
                self._client.images.get(image)
            except docker.errors.ImageNotFound:
                return False
            return True

        return await asyncio.to_thread(_exists)

    async def pull_image(self, image: str, *, timeout_s: float = 600.0) -> None:
        """Pull ``image`` via the local Docker daemon. No-op if already present."""

        def _do_pull() -> None:
            self._client.images.pull(image)

        try:
            await asyncio.wait_for(asyncio.to_thread(_do_pull), timeout=timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"docker pull {image!r} did not finish within {timeout_s:g}s"
            ) from exc
        # GC containerd's content store after each pull. Containerd
        # retains pulled layer blobs even after Docker unpacks them
        # into its data-root; on nodes where the local disk is small
        # (e.g. 97 GB HyperPod root) this accumulates and fills /.
        await self._gc_containerd_content()

    async def _gc_containerd_content(self) -> None:
        """Prune unreferenced blobs from containerd's content store.

        After Docker unpacks pulled layers, containerd retains the
        compressed layer tarballs; on small disks this accumulates and
        fills the volume. The prune is a containerd GC — it takes the
        metadata lock and walks the whole content store — so it is
        **debounced + single-flight**: at most one prune per
        ``_CONTENT_GC_MIN_INTERVAL_S``, and a burst of concurrently
        finishing pulls coalesces to one pass instead of a per-pull lock
        storm that starves ``docker ps`` / ``docker system df``.
        Best-effort: failures are logged and swallowed.
        """
        now = time.monotonic()
        if (
            self._content_gc_running
            or now - self._last_content_gc_monotonic < _CONTENT_GC_MIN_INTERVAL_S
        ):
            return
        self._content_gc_running = True

        def _prune() -> None:
            try:
                subprocess.run(
                    ["ctr", "-n", "moby", "content", "prune", "references"],
                    capture_output=True, timeout=30,
                )
            except FileNotFoundError:
                pass  # ctr not installed
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "containerd content prune failed (non-fatal): %s", exc,
                )

        try:
            await asyncio.to_thread(_prune)
        finally:
            # Stamp completion time (not start) so the interval measures
            # quiet time between prunes, and clear the single-flight flag.
            self._last_content_gc_monotonic = time.monotonic()
            self._content_gc_running = False

    async def remove_image(self, image: str, *, force: bool = False) -> None:
        def _do_remove() -> None:
            with suppress(docker.errors.ImageNotFound):
                try:
                    self._client.images.remove(image=image, force=force)
                except docker.errors.APIError as exc:
                    # A 409 Conflict with force=False means the image is still
                    # referenced — most often by a container the platform did
                    # not create (a node-level sidecar, e.g. the EFA monitoring
                    # exporter). Docker refusing is the *safe* outcome: we must
                    # not untag a live non-xrlenv container's image. Surface it
                    # as a typed signal so eviction skips quietly instead of
                    # logging a stack trace every sweep.
                    if getattr(exc, "status_code", None) == 409:
                        raise ImageInUse(image) from exc
                    raise

        await asyncio.to_thread(_do_remove)

    def resolve_image_digest(self, image_ref: str) -> str | None:
        """Spec 19 §"Image and asset supply chain": return the local
        digest for ``image_ref`` (``"sha256:abcd..."``) so the catalog
        can pin ``image:tag`` -> ``image@sha256:...`` at register time.

        Returns ``None`` when the image isn't present locally + can't be
        pulled — the catalog logs a warning and registers unpinned in
        that case rather than wedging registration on a network fetch
        every operator now has to wait through.

        **Local-build trap.** Recent Docker (buildx as default builder)
        pre-populates ``RepoDigests`` for locally-built images with an
        entry whose digest is just the image's content ``Id`` (the
        config hash). That digest is NOT registry-resolvable — no
        registry has it. If we returned it, the catalog would pin
        ``image:tag`` → ``image@sha256:<id>`` and any node that later
        tries ``ensure_present(image@sha256:<id>)`` falls through to a
        doomed ``docker pull``, dying with ``pull access denied``
        because no registry hosts the local-only build. So we
        explicitly skip RepoDigests entries whose digest matches the
        image's local ``Id`` — they're buildx's per-host artefact, not
        a real cross-host pin.
        """
        try:
            img = self._client.images.get(image_ref)
        except docker.errors.ImageNotFound:
            try:
                img = self._client.images.pull(image_ref)
                if isinstance(img, list):  # docker-py returns list when no tag
                    img = img[0]
            except Exception:
                return None
        local_id = img.attrs.get("Id")  # ``sha256:<config-hash>``
        digests = img.attrs.get("RepoDigests") or []
        for entry in digests:
            if "@sha256:" not in entry:
                continue
            digest = entry.split("@", 1)[1]
            # Skip buildx's local-only RepoDigests (digest == Id).
            # Real registry-resolved digests are the manifest hash,
            # which differs from the config-hash Id.
            if digest == local_id:
                continue
            return str(digest)
        # No registry-resolvable digest. Don't fall back to the local
        # content-addressed ``Id`` here — pinning to ``Id`` produces a
        # ``<repo>@sha256:<id>`` reference that no registry can
        # resolve, so any later ``ensure_present`` triggers a doomed
        # ``docker pull``. The catalog's "unpinned + warn" branch is
        # the right outcome for locally-built images that haven't been
        # pushed to a registry. Production deployments should push
        # images so they have ``RepoDigests``.
        return None

    async def free_disk_bytes(self) -> int:
        """Return free bytes on the docker storage driver's disk pool.

        ``docker info`` exposes the storage path; we shell-call ``shutil.disk_usage``
        against it. When the path lookup fails (e.g. remote daemon over TCP
        with no local mount), we fall back to ``shutil.disk_usage('/')`` so
        the cache manager always gets a finite number to compare against.
        """
        return (await asyncio.to_thread(self._disk_usage)).free

    async def total_disk_bytes(self) -> int:
        """Return the total size of the docker storage driver's disk pool.

        Same lookup as :py:meth:`free_disk_bytes` (docker info →
        DockerRootDir → ``shutil.disk_usage``), so the reported pool is
        the actual filesystem the daemon writes into rather than the
        node's root filesystem. Reported for telemetry / heartbeat only —
        the adaptive eviction model sizes its headroom from the largest
        cached image, not from this total.
        """
        return (await asyncio.to_thread(self._disk_usage)).total

    def _disk_usage(self) -> Any:
        docker_root = self._resolve_docker_root_dir()
        try:
            return shutil.disk_usage(docker_root)
        except OSError:
            return shutil.disk_usage("/")

    def disk_monitor_path(self) -> str | None:
        """Filesystem path on the docker data-root volume for I/O-saturation
        sampling (:class:`xrlenv.node.disk_io.DiskIoSampler`), or ``None``
        until ``DockerRootDir`` has been resolved.

        Returning ``None`` while ``docker info`` hasn't succeeded yet keeps
        the sampler from binding to the small root filesystem (the ``"/"``
        fallback) by mistake — it retries on a later tick and binds to the
        real data-root device (e.g. the EBS mount at ``/opt/sagemaker``)
        once the daemon is up."""
        if self._docker_root_dir is not None:
            return self._docker_root_dir
        # Populate the cache if the daemon is up; stays None on failure.
        self._resolve_docker_root_dir()
        return self._docker_root_dir

    def _resolve_docker_root_dir(self) -> str:
        """Return Docker's storage root (``DockerRootDir``), resolved once
        and cached.

        ``DockerRootDir`` is stable for the daemon's lifetime, so we ask
        ``docker info`` only until it succeeds, then reuse the value. This
        avoids two failure modes that mis-measured free disk on a node
        whose data-root sits on a separate volume (e.g. an EBS mount at
        ``/opt/sagemaker``):

        * ``docker info`` is slow under heavy pull load (one call per disk
          check, and the cache evictor checks often); and
        * a transient ``info`` failure used to fall back to ``/`` (the
          small root fs), making the evictor see almost no free space and
          thrash — evicting cold images the build immediately re-pulls.

        Falls back to ``/`` only while ``info`` has *never* succeeded
        (daemon still starting), logging once so the misreport is visible.
        """
        if self._docker_root_dir is not None:
            return self._docker_root_dir
        try:
            info = self._client.info()
            root = info.get("DockerRootDir")
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "docker info failed while resolving DockerRootDir (%s); "
                "measuring free disk against '/' until it succeeds — this "
                "misreports on a node whose data-root is a separate volume",
                exc,
            )
            return "/"
        if not root:
            return "/"
        self._docker_root_dir = str(root)
        return self._docker_root_dir

    # ── Action primitives ────────────────────────────────────────────────────

    def exec(
        self,
        sb: SandboxHandle,
        cmd: list[str],
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[ExecChunk]:
        if stdin is not None:
            raise NotImplementedError("stdin via exec lands in a later slice")
        return _docker_exec_iter(
            client=self._client,
            backend_ref=sb.backend_ref,
            cmd=cmd,
            env=env,
            timeout_s=timeout_s,
        )

    async def read_file(self, sb: SandboxHandle, path: str) -> bytes:
        container = await asyncio.to_thread(self._client.containers.get, sb.backend_ref)
        # docker-py's get_archive returns a tar stream containing one entry.
        # ``bits`` is a LAZY socket-reading generator: iterating it (inside
        # ``_extract_single_file``) does blocking reads, so it must run in a
        # thread, not on the event loop — otherwise a large file freezes the
        # heartbeat the same way the raw ``get_archive`` node-lost bug did.
        bits, _stat = await asyncio.to_thread(container.get_archive, path)
        return await asyncio.to_thread(
            _extract_single_file, bits, Path(path).name,
        )

    async def write_file(self, sb: SandboxHandle, path: str, data: bytes) -> None:
        container = await asyncio.to_thread(self._client.containers.get, sb.backend_ref)
        archive = _make_single_file_tar(Path(path).name, data)
        ok = await asyncio.to_thread(
            container.put_archive, str(Path(path).parent), archive
        )
        if not ok:
            raise OSError(f"docker put_archive failed for {path}")

    async def put_archive(
        self,
        sb: SandboxHandle,
        target_dir: str,
        tarball: bytes,
        *,
        clean_target: bool = False,
    ) -> None:
        """Extract a tar archive into ``target_dir`` inside the container.

        When ``clean_target=True`` we first run ``rm -rf <target_dir>``
        as root via ``docker exec --user root`` and check the exit
        status — if the wipe fails we raise rather than silently
        proceeding, so agent-created residue under the target path
        cannot survive into the verifier phase (audit H1, 2026-04-29).
        Then we create ``target_dir`` (also as root) and call Docker's
        ``put_archive``. Docker auto-detects gzip; the caller can ship
        either plain or gzipped tar.
        """
        container = await asyncio.to_thread(self._client.containers.get, sb.backend_ref)
        api = self._client.api

        if clean_target:
            await self._exec_as_root(
                api,
                container.id,
                ["rm", "-rf", target_dir],
                step="wipe",
                target_dir=target_dir,
            )

        # ``put_archive`` requires the target directory to exist; create
        # it as root to handle paths under /opt, /tests etc. that the
        # image's default user wouldn't normally have permission to
        # mkdir under. We check the exit status to fail-fast on
        # exec-time errors (e.g. mkdir refused on a read-only mount).
        await self._exec_as_root(
            api,
            container.id,
            ["mkdir", "-p", target_dir],
            step="mkdir",
            target_dir=target_dir,
        )

        ok = await asyncio.to_thread(
            container.put_archive, target_dir, tarball
        )
        if not ok:
            raise OSError(
                f"docker put_archive failed for target_dir={target_dir} "
                f"(payload={len(tarball)} bytes)"
            )

    @staticmethod
    async def _exec_as_root(
        api: Any,
        container_id: str,
        cmd: list[str],
        *,
        step: str,
        target_dir: str,
    ) -> None:
        """Run ``cmd`` as root in the container, fail-closed on non-zero.

        Used by ``put_archive`` for the wipe + mkdir pre-extraction
        steps (D12 stage 1). The wipe must be checked because a
        silent failure would let agent residue contaminate the
        verifier phase.
        """
        handle = await asyncio.to_thread(
            api.exec_create,
            container_id,
            cmd,
            user="root",
            stdout=True,
            stderr=True,
            tty=False,
        )
        # ``stream=False`` returns the combined stdout+stderr bytes
        # synchronously after the exec finishes; we don't need it
        # except to confirm completion before inspecting exit code.
        await asyncio.to_thread(api.exec_start, handle["Id"], stream=False)
        info = await asyncio.to_thread(api.exec_inspect, handle["Id"])
        exit_code = int(info.get("ExitCode") or 0)
        if exit_code != 0:
            raise OSError(
                f"docker put_archive {step} step failed (exit={exit_code}) "
                f"for target_dir={target_dir} cmd={cmd!r}"
            )

    # ── Streaming files ──────────────────────────────────────────────────────

    def read_file_stream(self, sb: SandboxHandle, path: str) -> AsyncIterator[bytes]:
        raise NotImplementedError("file streaming lands in Slice 2 (spec 01)")

    async def write_file_stream(
        self,
        sb: SandboxHandle,
        path: str,
        src: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError("file streaming lands in Slice 2 (spec 01)")

    # ── Long-lived in-sandbox processes ──────────────────────────────────────

    async def spawn_service(self, sb: SandboxHandle, spec: ServiceSpec) -> Any:
        raise NotImplementedError("services land in Slice 3 (spec 01)")

    async def spawn_services(
        self,
        sb: SandboxHandle,
        specs: list[ServiceSpec],
    ) -> list[Any]:
        raise NotImplementedError("services land in Slice 3 (spec 01)")

    # ── Network exposure ─────────────────────────────────────────────────────

    async def port_forward(self, sb: SandboxHandle, internal_port: int) -> str:
        raise NotImplementedError("port_forward lands in Slice 3 (spec 07)")

    # ── Snapshot / restore ───────────────────────────────────────────────────

    async def snapshot(self, sb: SandboxHandle) -> SnapshotID:
        raise NotImplementedError("snapshot lands in phase 2 (spec 01)")

    async def restore(self, snapshot: SnapshotID) -> SandboxHandle:
        raise NotImplementedError("restore lands in phase 2 (spec 01)")

    # ── Observation primitive ────────────────────────────────────────────────

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        container = await asyncio.to_thread(self._client.containers.get, sb.backend_ref)
        raw = await asyncio.to_thread(container.stats, stream=False)
        return _parse_stats(raw)

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _await_stub_ready(
        self,
        *,
        container: Container,
        transport: StubTransport,
        stub_uds_path: Path,
        container_port: int,
        timeout_s: float,
    ) -> str:
        """Wait until the stub is reachable and return its endpoint URI.

        For ``uds`` we poll for the socket file; for ``tcp`` we resolve the
        published host port via ``docker inspect`` and TCP-connect to it. We
        actively probe (not just file-exists) to avoid the macOS race where
        the bind-mount file appears before the listener is accepting.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if transport == "uds":
                if stub_uds_path.exists():
                    return f"unix://{stub_uds_path}"
            else:
                host_port = await self._published_host_port(container, container_port)
                if host_port is not None and await _tcp_probe("127.0.0.1", host_port):
                    return f"tcp://127.0.0.1:{host_port}"
            await asyncio.sleep(0.1)
        raise TimeoutError(
            f"sandbox stub ({transport}) did not become reachable within {timeout_s}s"
        )

    async def _published_host_port(
        self, container: Container, container_port: int
    ) -> int | None:
        await asyncio.to_thread(container.reload)
        ports = container.attrs.get("NetworkSettings", {}).get("Ports") or {}
        bindings = ports.get(f"{container_port}/tcp") or []
        for b in bindings:
            host_port = b.get("HostPort")
            if host_port:
                return int(host_port)
        return None

    def _cleanup_run_dir(self, sb: SandboxHandle) -> None:
        if sb.stub_endpoint.startswith("unix://"):
            host_run_dir = Path(sb.stub_endpoint.removeprefix("unix://")).parent
            shutil.rmtree(host_run_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _force_remove(container: Container) -> None:
    # Container may already be gone; destroy is idempotent (spec 01).
    with suppress(Exception):
        container.remove(force=True)


def _collect_failure_diagnostics(container: Container) -> str:
    """Best-effort collection of a failed-to-start container's logs + exit
    state. Called from the stub-startup-timeout path *before*
    :func:`_force_remove` wipes the container — gives the operator the
    actual reason in the raised exception (which then lands in
    ``coordinator.log`` and ``journalctl -u xrlenv-node``) instead of a
    bare "stub did not become reachable" with no context.

    Returns a single-line summary (status + exit code + tail of logs).
    Tolerant of every Docker SDK failure mode — an unreachable daemon
    or container-already-gone returns a degraded but still-readable
    string rather than masking the original timeout.
    """
    parts: list[str] = []
    try:
        container.reload()
        state = container.attrs.get("State", {}) or {}
        status = state.get("Status") or "unknown"
        exit_code = state.get("ExitCode")
        oci_error = state.get("Error") or ""
        parts.append(f"container.status={status}")
        if exit_code is not None:
            parts.append(f"exit_code={exit_code}")
        if oci_error:
            parts.append(f"oci_error={oci_error[:200]}")
    except Exception as exc:
        parts.append(f"reload_failed={type(exc).__name__}")

    try:
        # ``tail`` keeps the message bounded; the stub's startup is short
        # so 30 lines covers the import + bind path and any traceback.
        raw = container.logs(stdout=True, stderr=True, tail=30)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        text = text.strip().replace("\n", " | ")
        if text:
            parts.append(f"stdout/stderr={text[:1500]}")
        else:
            parts.append("stdout/stderr=<empty>")
    except Exception as exc:
        parts.append(f"logs_failed={type(exc).__name__}")
    return " ".join(parts)


async def _tcp_probe(host: str, port: int) -> bool:
    """Probe via a real HTTP/1.1 ``GET /healthz`` over a single connection.

    Just opening + closing a TCP socket leaves aiohttp's server-side keep-alive
    state in a confused half-open shape; the next legitimate request from the
    SDK then sees ``Server disconnected``. Sending a complete HTTP request and
    closing the socket cleanly avoids the race.
    """
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except (OSError, ConnectionError):
        return False
    try:
        request = (
            f"GET /healthz HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()
        writer.write(request)
        await writer.drain()
        # Read the status line to confirm the server actually responded.
        status_line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        ok = status_line.startswith(b"HTTP/1.1 200")
    except (OSError, TimeoutError):
        ok = False
    finally:
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()
    return ok


async def _docker_exec_iter(
    *,
    client: docker.DockerClient,
    backend_ref: str,
    cmd: list[str],
    env: dict[str, str] | None,
    timeout_s: float | None,
) -> AsyncIterator[ExecChunk]:
    container = await asyncio.to_thread(client.containers.get, backend_ref)
    api = client.api
    exec_handle = await asyncio.to_thread(
        api.exec_create,
        container.id,
        cmd,
        environment=env or {},
        stdout=True,
        stderr=True,
        tty=False,
    )
    exec_id: str = exec_handle["Id"]
    stream = await asyncio.to_thread(api.exec_start, exec_id, stream=True, demux=True)

    deadline = (time.monotonic() + timeout_s) if timeout_s else None

    def _next_chunk() -> tuple[bytes | None, bytes | None] | None:
        try:
            chunk: tuple[bytes | None, bytes | None] = next(stream)
            return chunk
        except StopIteration:
            return None

    while True:
        if deadline is not None and time.monotonic() > deadline:
            yield ExecChunk(stream="exit", exit_code=124)  # SIGKILL convention
            return
        chunk = await asyncio.to_thread(_next_chunk)
        if chunk is None:
            break
        stdout_bytes, stderr_bytes = chunk
        if stdout_bytes:
            yield ExecChunk(stream="stdout", data=stdout_bytes)
        if stderr_bytes:
            yield ExecChunk(stream="stderr", data=stderr_bytes)

    info = await asyncio.to_thread(api.exec_inspect, exec_id)
    yield ExecChunk(stream="exit", exit_code=int(info.get("ExitCode") or 0))


def _make_single_file_tar(name: str, data: bytes) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _extract_single_file(bits: Any, name: str) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    for chunk in bits:
        buf.write(chunk)
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        member = tf.getmember(name)
        f = tf.extractfile(member)
        if f is None:
            raise OSError(f"could not extract {name} from docker archive")
        return f.read()


def _parse_stats(raw: dict[str, Any]) -> ResourceUsage:
    """Best-effort cgroup stats → ResourceUsage."""
    cpu = raw.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
    rss = raw.get("memory_stats", {}).get("usage", 0)
    networks = raw.get("networks") or {}
    rx = sum(n.get("rx_bytes", 0) for n in networks.values())
    tx = sum(n.get("tx_bytes", 0) for n in networks.values())
    return ResourceUsage(
        cpu_seconds=cpu / 1e9,
        rss_bytes=int(rss),
        disk_bytes=0,  # disk usage requires `docker system df` semantics — Slice 3
        rx_bytes=int(rx),
        tx_bytes=int(tx),
    )
