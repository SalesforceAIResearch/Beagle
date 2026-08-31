"""SandboxBackend interface and supporting data shapes (spec 01).

Every sandbox runtime — Docker now, CubeSandbox/Firecracker later — implements
the same async interface. Returned handles are opaque outside the issuing
backend (the control plane and node agent shuffle them around but never look
inside). Capability flags drive the scheduler's placement decisions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ──────────────────────────────────────────────────────────────────────────────
# Network policies (spec 07)
# ──────────────────────────────────────────────────────────────────────────────

NetworkPolicy = Literal["none", "open", "egress-allowlist"]


# ──────────────────────────────────────────────────────────────────────────────
# Mounts, services, resource specs
# ──────────────────────────────────────────────────────────────────────────────


class MountSpec(BaseModel):
    """A host-path bind-mount made visible inside the sandbox at create time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host_path: str
    sandbox_path: str
    readonly: bool = True


class ServiceSpec(BaseModel):
    """A long-lived in-sandbox process declared either via the manifest's
    ``services:`` block or spawned dynamically post-create.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    cmd: list[str]
    port: int | None = None
    env: dict[str, str] | None = None
    health_check: list[str] | None = None
    startup_timeout_s: float = 30.0
    wait_until_ready: bool = True
    depends_on: list[str] = Field(default_factory=list)
    restart: Literal["never", "on_failure", "always"] = "never"


class CpuIsolation(StrEnum):
    """How a container's CPU request must be satisfied
    (cluster-resource-isolation-plan P6). A **scheduling-relevant** policy, so it
    lives on :class:`ResourceSpec` — NOT on :class:`RuntimeLimits`, which stays
    scheduling-neutral.

    - ``OFF`` — CFS ``--cpus`` quota only, burstable across cores (today's default;
      how harbor runs). No pinning.
    - ``BEST_EFFORT`` — pin to ``ceil(cpu_limit)`` dedicated cores **if the node has
      free pinned capacity**, else fall back to CFS quota. **Scheduling-neutral** —
      no placement constraint. This is the compat target of the legacy
      ``RuntimeLimits.cpu_pinning=True``.
    - ``REQUIRED`` — pin or **fail**: the scheduler places only on an
      ``isolation_capable`` node with free pinned capacity, and node-side ledger
      exhaustion is a hard error, never a silent quota degrade. **Scheduling-
      relevant.** (P6 turns on the placement predicate in a later sequenced step —
      see plan §8.8. Until then ``REQUIRED`` is treated exactly like ``BEST_EFFORT``
      on the node, so this enum is behavior-neutral to introduce.)
    """

    OFF = "off"
    BEST_EFFORT = "best_effort"
    REQUIRED = "required"

    @property
    def pins(self) -> bool:
        """True when the node should attempt cpuset pinning (``BEST_EFFORT`` or
        ``REQUIRED``); ``OFF`` is quota-only. Lets the node key on "does this pin?"
        without re-deriving the mode."""
        return self is not CpuIsolation.OFF


class ResourceSpec(BaseModel):
    """Hard cgroup / hypervisor limits baked in at sandbox creation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_request: float
    cpu_limit: float
    mem_request_bytes: int
    mem_limit_bytes: int
    disk_request_bytes: int
    gpu_required: bool = False
    mounts: tuple[MountSpec, ...] = ()
    cpu_isolation: CpuIsolation = CpuIsolation.OFF
    """P6 CPU-isolation policy (scheduling-relevant). Default ``OFF`` = CFS quota
    only. See :class:`CpuIsolation`. Derived once from the harness request +
    the legacy ``RuntimeLimits.cpu_pinning`` alias via
    :func:`effective_cpu_isolation`."""


class RuntimeLimits(BaseModel):
    """Container-shape limits that do **not** affect scheduling (P0b,
    cluster-resource-isolation-plan).

    Kept distinct from :class:`ResourceSpec` (which the scheduler /
    capacity estimator consume): pids / shm / tmpfs / read-only-rootfs
    are applied at container creation only. Every field is optional —
    an unset field means "docker default, do not constrain".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pids_limit: int | None = None
    shm_size_bytes: int | None = None
    tmpfs: dict[str, str] = Field(default_factory=dict)
    readonly_rootfs: bool = False
    cpu_pinning: bool = False
    """**Compatibility alias (P6) — prefer ``ResourceSpec.cpu_isolation``.**
    ``True`` maps to ``CpuIsolation.BEST_EFFORT`` via
    :func:`effective_cpu_isolation` (pin to ``ceil(cpu_limit)`` dedicated cores if
    the node has free capacity, else CFS quota — the historical opt-in cpuset
    behavior). ``False`` → CFS ``--cpus`` quota only, exactly how harbor runs.
    Kept for back-compat while callers migrate to ``cpu_isolation``; it never
    expresses ``REQUIRED`` (hard isolation must be requested explicitly)."""

    def is_empty(self) -> bool:
        """True when the harness specified no runtime limit at all."""
        return (
            self.pids_limit is None
            and self.shm_size_bytes is None
            and not self.tmpfs
            and not self.readonly_rootfs
            and not self.cpu_pinning
        )


def resolve_cpu_isolation(explicit: CpuIsolation, *, cpu_pinning: bool) -> CpuIsolation:
    """The derive-once primitive (P6): an explicit mode wins; else the legacy
    ``cpu_pinning=True`` compat alias → ``BEST_EFFORT``; else ``OFF``. Use this at
    the control-plane ingress, where only the raw parts (a requested mode + the
    legacy pinning bool) are available rather than a full :class:`ResourceSpec`."""
    if explicit is not CpuIsolation.OFF:
        return explicit
    return CpuIsolation.BEST_EFFORT if cpu_pinning else CpuIsolation.OFF


def effective_cpu_isolation(
    resources: ResourceSpec | None,
    runtime_limits: RuntimeLimits | None,
) -> CpuIsolation:
    """Derive the single effective CPU-isolation mode from a full
    :class:`ResourceSpec` + :class:`RuntimeLimits` (thin wrapper over
    :func:`resolve_cpu_isolation`).

    Precedence: an explicit ``resources.cpu_isolation`` (≠ ``OFF``) wins; otherwise
    the legacy ``runtime_limits.cpu_pinning=True`` alias → ``BEST_EFFORT``; otherwise
    ``OFF``. Callers derive this **once** (control plane, before placement) and thread
    the result to scheduler / wire / node / admin — never re-derive per surface
    (Risk 1)."""
    return resolve_cpu_isolation(
        resources.cpu_isolation if resources is not None else CpuIsolation.OFF,
        cpu_pinning=bool(runtime_limits is not None and runtime_limits.cpu_pinning),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Capabilities
# ──────────────────────────────────────────────────────────────────────────────


class SandboxCapabilities(BaseModel):
    """Capability flags advertised by each backend (spec 01)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    supports_snapshot: bool
    supports_chainable_snapshot: bool
    live_state_captured: bool
    supports_port_forward: bool
    supports_gpu: bool
    isolation_class: Literal["container", "microvm", "none"]
    fast_create_p50_ms: int


# ──────────────────────────────────────────────────────────────────────────────
# Opaque IDs and handles
# ──────────────────────────────────────────────────────────────────────────────


class TemplateRef(BaseModel):
    """Resolved template handle the backend uses to instantiate a sandbox."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    image: str  # e.g. "xrlenv/hello-shell:0.1"
    digest: str | None = None  # pinned per spec 00 invariant 4


class SandboxHandle(BaseModel):
    """Opaque-to-callers identifier of a running sandbox.

    Only the issuing backend dereferences ``backend_ref``. The control plane
    treats the handle as a string-keyed token; ``id`` is platform-stable
    (matches ``sandboxes.id`` in StateStore).

    ``stub_endpoint`` is a URI: ``unix:///path/on/host/stub.sock`` for the
    spec-default Linux UDS transport, or ``tcp://127.0.0.1:<port>`` for the
    macOS Docker-Desktop fallback (uds-over-bind-mount does not work across
    Docker-Desktop's host↔VM boundary).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    backend: str
    backend_ref: str
    stub_endpoint: str


class SnapshotID(BaseModel):
    """Backend-opaque snapshot identifier (spec 01 / spec 18)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    ref: str


class ExecChunk(BaseModel):
    """One chunk of streamed exec output (spec 01)."""

    model_config = ConfigDict(extra="forbid")

    stream: Literal["stdout", "stderr", "exit"]
    data: bytes = b""
    exit_code: int | None = None


class ResourceUsage(BaseModel):
    """Cgroup / hypervisor counter snapshot used by the capacity estimator."""

    model_config = ConfigDict(extra="forbid")

    cpu_seconds: float
    rss_bytes: int
    disk_bytes: int
    rx_bytes: int
    tx_bytes: int


class ImageInUse(Exception):
    """Raised by ``remove_image`` when the backend refuses to delete an
    image because it is still referenced — typically by a container the
    platform did not create (a node-level sidecar). Distinct from a real
    failure: eviction treats it as "skip, held externally" rather than an
    error, and never force-deletes (that would untag a live container's
    image)."""


class ImageRecord(BaseModel):
    """One image present in a backend's local cache (spec 15)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """Tag-or-digest reference, e.g. ``xrlenv/hello-shell:0.1`` or
    ``xrlenv/hello-shell@sha256:...``.
    """
    size_bytes: int
    """On-disk size as reported by the backend (Docker: image size, includes
    shared layers — eviction LRU is approximate at the layer-share level).
    """
    shared_size_bytes: int | None = None
    """Bytes belonging to layers this image shares with one or more other
    images cached on the same node — as reported by Docker's ``system df``
    API (``SharedSize``). ``None`` when the backend doesn't expose layer
    sharing (e.g. older Docker daemons, the in-process LocalBackend, the
    in-memory test backend). Together with :attr:`size_bytes` this lets
    callers compute the **incremental cost** of caching this image on a
    node where its base layers are already present:

        unique_bytes = size_bytes - (shared_size_bytes or 0)

    ``xrlenv build calibrate`` writes ``unique_bytes`` to the plan YAML's
    ``placement.size_hint_bytes`` when available, so the FFD bin-packer
    no longer over-reserves by the shared-layer footprint of every entry.
    """
    digest: str | None = None
    """Content-addressed digest when available (None when the backend stores
    only tags, e.g. an image built locally without a registry push).
    """
    last_used_at: float | None = None
    """Monotonic-clock seconds when this image was last referenced by a
    sandbox. ``None`` for images we discovered at startup but have not
    seen used since the cache manager came up.
    """
    labels: dict[str, str] = Field(default_factory=dict)
    """Docker image labels (``Config.Labels``) — used by the spec-15
    ownership classifier to distinguish xrlenv-built images from
    operator-foreign ones. Empty dict for images surfaced by name
    only (digests or backends that don't expose labels).
    """


class ExecResult(BaseModel):
    """Result of a one-shot in-sandbox command (Slice 3.5 RunInSandbox).

    Used by :py:meth:`NodeTransport.run_in_sandbox` so the coordinator's
    init-script step has a uniform return shape across in-process
    (NodeAgent) and remote (RemoteNodeTransport) backends.
    """

    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# The interface
# ──────────────────────────────────────────────────────────────────────────────


class SandboxBackend(ABC):
    """Runtime-agnostic interface to one sandbox runtime (spec 01).

    All methods are async. Streaming methods return async iterators. Backends
    that do not implement a capability advertise it via :class:`SandboxCapabilities`
    and raise :class:`NotImplementedError` at call time so the scheduler's
    template-vs-capability check is the single load-bearing guard.
    """

    name: str
    capabilities: SandboxCapabilities

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    async def create(
        self,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
    ) -> SandboxHandle:
        """Instantiate a sandbox from ``template`` with the given resource caps
        and network policy. Returns the handle to use for all subsequent calls.
        """

    @abstractmethod
    async def destroy(self, sb: SandboxHandle) -> None:
        """Terminate the sandbox and release its resources. Idempotent."""

    # ── Action primitives ────────────────────────────────────────────────────

    @abstractmethod
    def exec(
        self,
        sb: SandboxHandle,
        cmd: list[str],
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[ExecChunk]:
        """Run a one-shot command; stream stdout/stderr; finish with an
        ``ExitChunk``-shaped chunk.
        """

    @abstractmethod
    async def read_file(self, sb: SandboxHandle, path: str) -> bytes:
        """Read a small file (≤16 MB) from inside the sandbox."""

    @abstractmethod
    async def write_file(self, sb: SandboxHandle, path: str, data: bytes) -> None:
        """Write a small file (≤16 MB) inside the sandbox; creates parent dirs."""

    @abstractmethod
    async def put_archive(
        self,
        sb: SandboxHandle,
        target_dir: str,
        tarball: bytes,
        *,
        clean_target: bool = False,
    ) -> None:
        """Extract a tar(.gz) archive into ``target_dir`` inside the sandbox.

        Used by the platform's verifier-asset injection (D12 stage 1):
        the resolver tarballs per-task grader assets host-side, the
        coordinator ships the bytes to the node, and the node extracts
        them into the running sandbox at reward time. The grader
        files therefore do not exist in the container during the
        agent's ``step()`` loop — closes audit H1's timing-isolation
        half.

        ``clean_target=True`` requests a root-backed ``rm -rf
        <target_dir>`` before extraction. Implementations MUST run
        the wipe with verifier authority (``docker exec --user root``
        for the Docker backend) and MUST raise on non-zero exit so
        agent-created residue cannot survive into the verifier phase.
        Then create ``target_dir`` (also as root) and extract.
        Implementations should accept either plain or gzipped tar;
        Docker's ``put_archive`` handles both transparently.
        """

    # ── Streaming files (large payloads) ─────────────────────────────────────

    @abstractmethod
    def read_file_stream(self, sb: SandboxHandle, path: str) -> AsyncIterator[bytes]:
        """Read a file as a chunked async byte stream (1 MB chunks default)."""

    @abstractmethod
    async def write_file_stream(
        self,
        sb: SandboxHandle,
        path: str,
        src: AsyncIterator[bytes],
    ) -> None:
        """Stream-write a file with atomic write-to-temp + rename semantics."""

    # ── Long-lived in-sandbox processes ──────────────────────────────────────

    @abstractmethod
    async def spawn_service(self, sb: SandboxHandle, spec: ServiceSpec) -> object:
        """Start one long-lived service inside the sandbox."""

    @abstractmethod
    async def spawn_services(
        self,
        sb: SandboxHandle,
        specs: list[ServiceSpec],
    ) -> list[object]:
        """Topologically launch a service set with shared port discovery."""

    # ── Network exposure ─────────────────────────────────────────────────────

    @abstractmethod
    async def port_forward(self, sb: SandboxHandle, internal_port: int) -> str:
        """Expose an internal sandbox port at a node-reachable URL."""

    # ── Snapshot / restore (capability-gated) ────────────────────────────────

    @abstractmethod
    async def snapshot(self, sb: SandboxHandle) -> SnapshotID: ...

    @abstractmethod
    async def restore(self, snapshot: SnapshotID) -> SandboxHandle: ...

    # ── Observation primitive ────────────────────────────────────────────────

    @abstractmethod
    async def stats(self, sb: SandboxHandle) -> ResourceUsage: ...

    # ── GC layer-2 introspection (spec 09 §"Garbage collection") ────────────

    async def list_owned_sandboxes(self) -> list[SandboxHandle]:
        """Return every still-alive sandbox this backend owns on the host.

        Used by :py:meth:`xrlenv.node.agent.NodeAgent.gc_orphans` at startup
        to find containers/microVMs left behind by a previous node-agent
        process (e.g. the daemon was OOM-killed or systemd-restarted while
        sandboxes were running). The returned :class:`SandboxHandle` carries
        empty ``stub_endpoint`` because the host bind-mount path / TCP port
        is not recoverable for orphans — only ``id`` and ``backend_ref`` are
        needed to invoke :py:meth:`destroy`.

        Default implementation returns ``[]`` so capability is opt-in;
        backends that can enumerate their own state (Docker via label
        filter, Cube via VM list) override it.
        """
        return []

    # ── Image-cache surface (spec 15 §"Image Cache Manager") ────────────────

    async def list_images(
        self, *, include_shared_size: bool = False,
    ) -> list[ImageRecord]:
        """Return every image cached locally that the cache manager tracks.

        Returns ``[]`` by default; backends override (Docker via
        ``client.images.list()``, Cube via the microVM image catalog).
        Used by :class:`xrlenv.node.image_cache.ImageCacheManager` to
        build its working-set view at startup and on every report tick.

        ``include_shared_size`` (default ``False``) requests per-image
        layer-sharing data (``ImageRecord.shared_size_bytes``). It can be
        far more expensive than the plain listing (Docker walks the whole
        layer graph via ``system df``), so the hot path — eviction, the
        adaptive-control stats, the live ``/images`` view — leaves it off;
        only ``xrlenv build calibrate`` (unique-size accounting) sets it.
        """
        return []

    async def image_exists(self, image: str) -> bool:
        """Return ``True`` if ``image`` is resolvable locally (by tag or
        digest reference) without contacting a registry. The Docker
        backend wraps ``client.images.get(ref)`` so both
        ``xrlenv/hello-shell:0.1`` and
        ``xrlenv/hello-shell@sha256:...`` answer correctly for
        locally-built images that have no ``RepoDigests``. Used by
        :class:`xrlenv.node.image_cache.ImageCacheManager` to short-
        circuit ``ensure_present`` before attempting a pull. Default
        falls back to ``list_images()`` for backends that don't expose
        a single-ref existence check.
        """
        images = await self.list_images()
        return any(img.name == image for img in images)

    async def pull_image(self, image: str, *, timeout_s: float = 600.0) -> None:
        """Pull ``image`` into the local cache. No-op if already present.

        Default raises :class:`NotImplementedError` so backends without a
        meaningful image-pull step (e.g. function-call mode) fail loud
        rather than silently swallow warmup directives.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pull_image"
        )

    async def remove_image(self, image: str, *, force: bool = False) -> None:
        """Evict ``image`` from the local cache. No-op if not present.

        ``force`` is for the operator-driven ``xrlenv images unpin --evict``
        path; the cache manager normally calls this only when the image is
        not in-use and not pinned, so force=False is the safe default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement remove_image"
        )

    async def free_disk_bytes(self) -> int:
        """Return free bytes on the disk pool the image cache evicts against.

        Used by the cache manager to decide whether ``ensure_present`` must
        evict before pulling. Backends that do not occupy a meaningful disk
        budget (function-call) return :py:obj:`sys.maxsize` so the cache
        manager treats them as never under pressure.
        """
        import sys
        return sys.maxsize

    async def total_disk_bytes(self) -> int:
        """Return the total size of the disk pool the image cache evicts against.

        Reported for telemetry / heartbeat only — the adaptive eviction
        model sizes its headroom from the largest cached image and the
        live :py:meth:`free_disk_bytes`, not from this total. Backends
        without a meaningful disk pool (function-call) return
        :py:obj:`sys.maxsize` as an "unknown total" sentinel; their
        :py:meth:`free_disk_bytes` also returns ``sys.maxsize`` so the
        cache treats them as never under pressure.
        """
        import sys
        return sys.maxsize
