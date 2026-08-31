"""NodeAgent — the per-host orchestrator-side surface (spec 04).

In Slice 1 the agent runs in the same Python process as the control plane and
exposes its surface as plain async methods. The control plane treats the agent
like any other transport: in Slice 3 we'll wrap the same surface in the
spec-21 bidi gRPC stream and the control plane code will not care which is in
play. Keeping the surface pure-Python here means the proto wire format can
evolve without churning the orchestration logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from xrlenv.api.constants import DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES
from xrlenv.backends.base import (
    ExecResult,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    RuntimeLimits,
    SandboxBackend,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.backends.egress import EgressAllowlist
from xrlenv.errors import XRLEnvError
from xrlenv.node.health import NodeHealthSnapshot
from xrlenv.node.hw_probe import HardwareInfo, probe_hardware
from xrlenv.node.image_cache import (
    EvictOutcome,
    ImageCacheManager,
    ImageQueryResult,
    NodeImageReport,
)
from xrlenv.node.raw_compose import ComposeProjectRecord
from xrlenv.node.raw_container import RawContainerManager, RawContainerRecord
from xrlenv.node.stub_client import StubClient
from xrlenv.node.trajectory_reader import FetchRangeKind, LocalTrajectoryReader
from xrlenv.observability.tracing import get_tracer
from xrlenv.types import Trajectory

LOGGER = logging.getLogger(__name__)


class NodeAgentConfig(BaseModel):
    """Construction config for one :class:`NodeAgent`."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    node_id: str
    backends: SkipValidation[dict[str, SandboxBackend]]
    """Map of backend name → backend instance. Phase 0 has just ``docker``.

    SkipValidation: pydantic can't isinstance-check ABC instances against the
    SandboxBackend abstract base reliably (test fakes don't always subclass),
    and the dict carries opaque component references rather than wire data.
    """

    raw_destroy_concurrency: int = 4
    """Issue #18 fix #4 — cap on concurrent ``docker rm -f`` calls for
    raw containers on this node. Under heavy parallel destroys the
    docker daemon serialises overlay-fs teardowns and individual
    removes stretch to 35-90 s, which (a) blew the coordinator's
    historical 30 s destroy ceiling, and (b) held disk layers alive
    while teardown queued, amplifying disk pressure. ``4`` is the
    measured sweet spot. ``0`` disables the cap (legacy unbounded
    behaviour — kept for tests that exercise the pre-cap path)."""

    raw_create_concurrency: int = 4
    """Issue #18 — cap on concurrent ``docker run`` (container create)
    calls for raw containers on this node, symmetric with
    ``raw_destroy_concurrency``. A burst of acquires (SWE-bench Pro at
    ``--num-workers=64`` on a 2-node cluster) fired dozens of
    simultaneous ``containers.run`` calls at one docker daemon already
    saturated extracting multi-GB image layers; each create then
    stretched past the node's 600 s docker HTTP-client ceiling and
    failed the acquire outright. ``4`` mirrors the destroy cap. ``0``
    disables the cap (unbounded — kept for tests)."""

    raw_sysbox_create_concurrency: int = 1
    """A separate, tighter create cap for sysbox (non-``runc``) acquires.
    sysbox-runc's pre-register with ``sysbox-fs`` is far slower than a
    plain runc create, so the general ``raw_create_concurrency`` (4) still
    lets concurrent sysbox creates overwhelm sysbox-fs — surfacing as a
    transient ``pre-register with sysbox-fs … DeadlineExceeded`` 500 that
    fails the acquire. Serialising sysbox creates (default ``1``) stops it
    at the source (approach C — prevention); the node-side create
    retry-with-backoff recovers any transient that still slips through
    (approach A). ``0`` disables the sysbox-specific gate (sysbox creates
    fall back to the general create cap)."""

    raw_sysbox_destroy_concurrency: int = 1
    """Symmetric tighter cap for sysbox (non-``runc``) DESTROYS. A sysbox
    teardown unmounts the container's sysbox-fs FUSE layers (``fusermount3``);
    concurrent unmounts under high churn wedge sysbox-fs the same way concurrent
    creates overwhelm its register step — the wedged ``docker rm`` then hangs in
    D-state and LEAKS the container, holding a cap slot and dragging the whole
    sysbox layer (2026-07-08 conc-32 sweep leaked 4 sysbox containers). The
    general ``raw_destroy_concurrency`` (4) is too loose; serialise sysbox
    destroys (default ``1``). ``0`` disables the sysbox gate (falls back to the
    general destroy cap)."""

    raw_archive_concurrency: int = 4
    """Node-lost guardrail — cap on concurrent bulk container⇄node
    transfers (``get_archive`` / ``put_archive``) for raw containers on
    this node. This is the multi-tenant blast-radius bound: with 10+
    users each able to submit the heaviest job (EvoClaw copies the whole
    ``/testbed`` — hundreds of MB — out of every eval container), an
    unbounded fan-out of large tar streams pins the thread pool away
    from create/exec, balloons node RAM, and saturates the docker
    daemon's tar IO. Paired with the chunk-streaming ``get_archive``
    read (a single copy no longer freezes the heartbeat regardless of
    size), capping concurrent transfers is what keeps the node alive
    under the workload that took it ``lost``. ``4`` mirrors the
    create/destroy caps. ``0`` disables the cap (unbounded — tests)."""

    raw_max_get_archive_relay_bytes: int = DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES
    """Plane-split guardrail — the max bytes a single raw-container
    ``get_archive`` may relay back through the control plane. The control
    plane is a metadata/orchestration channel, not a bulk-data pipe (spec
    00 invariant 6). A transfer whose streamed size exceeds this is
    refused at the node (``ArchiveTooLarge``), failing THAT one transfer
    cleanly without touching the rollout — so no tenant can push a whole
    container filesystem (EvoClaw's ``docker cp {c}:/testbed .``) through
    the CP and starve it. Default 128 MiB is far above every legitimate
    small read (reward/verifier files, patches, logs) yet blocks
    whole-repo copies. ``0`` disables the cap (unbounded — tests).
    Operators tune via ``XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES``. Bulk
    artifact capture belongs to the artifact-export primitive
    (notes/artifact-export-primitive-proposal.md)."""

    stub_request_timeout_s: float = 3600.0
    """Per-stub-call HTTP timeout (layer 3 of the timeout model — see
    ``docs/integration/timeouts.md``). Default 1 h is the safety-net
    cap for sandboxes whose coordinator did NOT inject a per-sandbox
    override; in normal operation D17 stage 1 (P1.1) tightens this to
    ``max(init,setup,step,teardown) + 60 s`` per sandbox via the
    platform-private ``_xrlenv_http_timeout_s`` key on
    ``init_params`` (consumed by :py:meth:`NodeAgent.env_setup`).
    The real upper bounds remain the rollout's ``hard_s`` (deadline-
    watcher) and the adapter's per-step subprocess timeout
    (``init_params['step_timeout_s']``, populated by the resolver
    from each task.toml's ``[agent].timeout_sec``). D17 stage 2
    (deferred to P1.2 with D16) tracks the proper per-call
    plumb-through that will let each call carry its own cap rather
    than one cap per sandbox."""


class _SandboxRecord(BaseModel):
    """Per-sandbox bookkeeping held in-memory by the agent."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    handle: SandboxHandle
    template: str
    backend: str
    image: str | None = None
    """Image ref the sandbox depends on; used by the image cache manager
    to bump/decrement its refcount across the create/destroy lifecycle.
    None when the cache manager isn't wired (Slice 1 parity).
    """
    created_at: float = Field(default_factory=time.monotonic)
    stub: SkipValidation[StubClient | None] = None
    stub_request_timeout_s_override: float | None = None
    """D17 stage 1 (P1.1) per-sandbox HTTP cap. The coordinator
    populates this via the platform-private ``_xrlenv_http_timeout_s``
    key in ``env_setup``'s ``init_params`` (derived from the manifest's
    max inner timeout + 60 s buffer). Absent override, ``_stub_for``
    falls back to :attr:`NodeAgentConfig.stub_request_timeout_s` (the
    1 h safety net default). Full per-call plumb-through lands with
    D16 in P1.2."""




class NodeAgent:
    """Per-host data-plane controller.

    Owns the backend drivers, the in-flight sandbox table, and a per-sandbox
    stub client cache. The control plane drives the agent through the methods
    on this class.
    """

    def __init__(
        self,
        config: NodeAgentConfig,
        *,
        image_cache: ImageCacheManager | None = None,
        trajectory_reader: LocalTrajectoryReader | None = None,
    ) -> None:
        self._config = config
        self._sandboxes: dict[str, _SandboxRecord] = {}
        self._lock = asyncio.Lock()
        self._hw: HardwareInfo | None = None
        # P1.6.g — H3 lazy lifecycle: when ``BuildImagesCommand``
        # arrives, the node remembers (image_ref → BuilderRef +
        # kwargs) here so the image cache can call back into the
        # benchmark builder later if the ref isn't yet present at
        # rollout-create time. Process-lifetime memory only;
        # cross-restart recovery is operator re-apply (acceptable
        # — the build coordinator already has the source of truth
        # in state.db).
        self._lazy_builders: dict[
            str, tuple[Any, dict[str, str]],
        ] = {}
        # Sub-slice 2 — per-image_ref source-build registry. Owned by
        # the agent (not the wire link) so the image cache's
        # builder_lookup hook can consult it from any code path,
        # including post-restart ensure_present after eviction.
        # Lazy-constructed on first access via ``source_builder()`` —
        # nodes that never receive a build don't allocate the cache
        # root or load the registry.
        self._source_builder: Any | None = None
        if image_cache is not None:
            # Wire the image cache's builder lookup at construction
            # time so ensure_present can dispatch lazily.
            image_cache._builder_lookup = self._lookup_image_producer
        self._image_cache = image_cache
        self._trajectory_reader = trajectory_reader

        # P1.7.A.1 — raw-container manager per docker backend. Only
        # docker backends support raw containers in phase 1
        # (CubeSandbox in phase 2 will need a parallel manager).
        # Built eagerly in __init__ so the manager is ready as soon
        # as the dispatcher routes the first AcquireContainerCommand.
        # Guard on dict-shape: a few legacy tests pass ``backends=
        # ["fake"]`` (list of strings) abusing the SkipValidation
        # escape hatch — those tests don't exercise raw containers,
        # so we silently skip manager construction in that shape.
        self._raw_managers: dict[str, RawContainerManager] = {}
        if isinstance(config.backends, dict):
            for backend_name, backend in config.backends.items():
                client = getattr(backend, "docker_client", None)
                if client is not None:
                    self._raw_managers[backend_name] = RawContainerManager(
                        docker_client=client,
                        # P1.7.B.2: pass the same ImageCacheManager case-1
                        # sandboxes use, so ensure_image_present=True (the
                        # new default) routes pulls/builds through the
                        # existing cache + LRU eviction layer.
                        image_cache=image_cache,
                        destroy_concurrency=config.raw_destroy_concurrency,
                        create_concurrency=config.raw_create_concurrency,
                        sysbox_create_concurrency=(
                            config.raw_sysbox_create_concurrency
                        ),
                        sysbox_destroy_concurrency=(
                            config.raw_sysbox_destroy_concurrency
                        ),
                        archive_concurrency=config.raw_archive_concurrency,
                        max_get_archive_relay_bytes=(
                            config.raw_max_get_archive_relay_bytes
                        ),
                    )

    # ── Inventory & probes ───────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        return self._config.node_id

    @property
    def image_cache(self) -> ImageCacheManager | None:
        return self._image_cache

    def supported_backends(self) -> list[str]:
        return sorted(self._config.backends)

    def supported_runtimes(self) -> list[str]:
        """OCI runtimes docker has registered on this node (§5.3).

        Advertised on ``NodeHello`` so the scheduler filters candidate
        nodes when an acquire requests a non-default ``container_runtime``.
        Sourced from ``docker info`` via the docker raw-container manager;
        falls back to ``["runc"]`` when no docker manager is wired (e.g. a
        fake-backend test node) so the normal path always advertises runc.
        """
        mgr = self.raw_container_manager("docker")
        if mgr is None:
            return ["runc"]
        runtimes, _default = mgr.registered_runtimes()
        return sorted(runtimes)

    def probe_docker_runtimes_ready(self) -> bool:
        """Attempt to enumerate this node's docker runtimes, returning whether
        docker answered.

        The node link gates its first ``NodeHello`` on this so a node whose agent
        starts seconds after a docker restart doesn't advertise a conservative
        ``{'runc'}`` runtime set for the whole connection (the redeploy race —
        ``supported_runtimes`` is sent once at hello and, on a persistent
        connection, never re-advertised). Triggers the probe as a side effect
        (``registered_runtimes`` caches on a successful ``docker info``). Returns
        ``True`` when no docker manager is wired (a fake-backend test node — there
        is nothing to wait for)."""
        mgr = self.raw_container_manager("docker")
        if mgr is None:
            return True
        mgr.registered_runtimes()  # attempt the probe (caches on success)
        return mgr.runtimes_probed()

    def default_runtime(self) -> str:
        """The docker daemon's default runtime (§5.3, §9).

        Security-relevant: if this is anything but ``runc`` the
        ``allowed_runtimes`` opt-in is silently bypassed, so the control
        plane re-verifies it on every (re)connect. ``"runc"`` when no
        docker manager is wired.
        """
        mgr = self.raw_container_manager("docker")
        if mgr is None:
            return "runc"
        _runtimes, default = mgr.registered_runtimes()
        return default

    def isolation_capable(self) -> bool:
        """P6 (§8.6) — whether this node can enforce the shared-parent cpuset
        isolation scheme, advertised on ``NodeHello``.

        Delegates to the docker raw-container manager's §8.5 self-test (cgroup
        v2 + a proven parent→child cpuset inheritance). ``False`` when no docker
        manager is wired (a fake-backend test node) — the safe default; a node
        never claims isolation it hasn't proven. (Step 2a: the manager reports
        ``False`` until the self-test lands in the next P6 step.)"""
        mgr = self.raw_container_manager("docker")
        if mgr is None:
            return False
        return mgr.isolation_capable()

    def pinned_cpu_capacity(self) -> tuple[int, int]:
        """P6 (§8.6, R6) — ``(pinned_cpus_free, pinned_cpus_total)`` for the
        heartbeat's live pinned-CPU accounting (reporting only).

        Sourced from the docker manager's core ledger. ``(0, 0)`` when no docker
        manager is wired — the ``unknown`` sentinel the control plane leaves out
        of any pinned-capacity decision."""
        mgr = self.raw_container_manager("docker")
        if mgr is None:
            return (0, 0)
        return mgr.pinned_cpu_capacity()

    async def refresh_pinned_cpu_gap(self) -> None:
        """P6 step-4a — refresh the docker manager's cached legacy-unpinned-runc
        count so the next ``pinned_cpu_capacity()`` read reflects the drained
        gap. Called on the heartbeat cadence (the async path can await the docker
        list ``pinned_cpu_capacity`` itself must not). No-op without a manager."""
        mgr = self.raw_container_manager("docker")
        if mgr is not None:
            await mgr.refresh_legacy_gap()

    def raw_container_manager(
        self, backend: str = "docker",
    ) -> RawContainerManager | None:
        """The raw-container manager for ``backend`` (the disk guard's
        offender source). ``None`` when no docker-backed manager is
        wired (e.g. a fake-backend test node)."""
        mgr = self._raw_managers.get(backend)
        if mgr is None and self._raw_managers:
            mgr = next(iter(self._raw_managers.values()))
        return mgr

    def health_snapshot(self) -> NodeHealthSnapshot | None:
        """Stage-1 — per-node docker-operation health for the heartbeat,
        from the raw-container manager. ``None`` when no raw manager is
        wired (a backend with no docker client). See
        ``notes/admission-stage-1-observability.md``.
        """
        mgr = self._raw_managers.get("docker")
        if mgr is None and self._raw_managers:
            mgr = next(iter(self._raw_managers.values()))
        return mgr.health_snapshot() if mgr is not None else None

    # ── P1.6.g — H3 lazy image lifecycle ─────────────────────────────────────

    def register_lazy_image_builders(
        self,
        mapping: dict[str, tuple[Any, dict[str, str]]],
    ) -> None:
        """Record (image_ref → (BuilderRef, kwargs)) so the image cache's
        ``ensure_present`` can later call the right benchmark builder
        when the ref isn't locally tagged AND isn't registry-pullable.
        Called by the spec-21 ``BuildImagesCommand`` handler before
        any synchronous build dispatch — the registration survives
        even when the operator's plan is "register only, lazy build."
        """
        self._lazy_builders.update(mapping)

    def register_scratch_source(
        self, image_ref: str, source: Any, *, durable_to: str | None = None,
    ) -> None:
        """Register a content-addressed scratch ref → build source (spec 06
        ``image_build``). ``ImageCacheManager.ensure_present`` on this ref then
        builds-and-pushes it to the scratch registry embedded in the ref
        (via the source builder's ``build_and_push``). When ``durable_to`` is
        set, the built image is also copied there so it survives scratch GC.
        Called by the coordinator for scratch_build rollouts on the in-process
        transport. ``source`` is a ``GitSource | TarballSource``."""
        self.source_builder().register_scratch_source(
            image_ref, source, durable_to=durable_to,
        )

    def source_builder(self) -> Any:
        """Lazy-construct + return the per-node ``GitSourceBuilder``.

        Single instance per agent so the active-builds registry +
        persistent source-spec registry are unified across the
        wire dispatch path (``BuildImageCommand``) and the image
        cache's ``ensure_present`` hook (build-on-acquire).
        """
        if self._source_builder is None:
            from xrlenv.node.source_builder import GitSourceBuilder
            self._source_builder = GitSourceBuilder()
        return self._source_builder

    def _lookup_image_producer(
        self, image_ref: str,
    ) -> Callable[[str, float], Awaitable[None]] | None:
        """Image-cache builder hook. Returns an async producer that:

        1. Checks the legacy benchmark builder registry
           (``register_lazy_image_builders``); when matched, dispatches
           to the registered :class:`BenchmarkImageBuilder`.
        2. Falls through to the per-image_ref source-spec registry
           (sub-slice 2 build-on-acquire); when matched, re-runs
           the source-build through the agent's ``GitSourceBuilder``.

        Returns ``None`` if neither matches — the cache then falls
        through to ``backend.pull_image`` (registry-pullable refs
        keep working unchanged).
        """
        record = self._lazy_builders.get(image_ref)
        if record is not None:
            ref, kwargs = record

            async def _produce_legacy(_ref: str, timeout_s: float) -> None:
                from xrlenv.control.image_builder import (
                    ImageBuilderDecl,
                    load_image_builder,
                )

                decl = ImageBuilderDecl.model_validate({
                    "module": ref.module, "class": ref.class_name,
                })
                builder = load_image_builder(decl)
                result = await asyncio.wait_for(
                    builder.build(
                        image_ref=_ref, kwargs=dict(kwargs), force=False,
                    ),
                    timeout=timeout_s,
                )
                if result.status != "done":
                    raise RuntimeError(
                        f"benchmark builder failed for {_ref}: "
                        f"{result.error or 'unknown'}",
                    )

            return _produce_legacy

        # Sub-slice 2: per-image_ref source-spec registry. Lazy-
        # construct the source_builder on first lookup so nodes
        # that never received a source-build don't pay the cache-
        # root + registry-load cost.
        if self._source_builder is None:
            # Tarball/git registry might still exist on disk from a
            # prior process — construct the builder so it can load.
            from xrlenv.node.source_builder import GitSourceBuilder
            self._source_builder = GitSourceBuilder()
        return self._source_builder.lookup_producer(image_ref)

    def hardware(self) -> HardwareInfo:
        """Hardware profile advertised to the control plane in ``NodeHello``.

        ``disk_bytes`` is measured against the **sandbox data-root** (the
        volume the container runtime writes into), not ``/``. The capacity
        estimator sizes the sandbox-writable pool from this number, so
        probing ``/`` on a node whose data-root is a separate volume caps
        the node at a fraction of its real capacity — a phantom disk bound
        that reads as "pool at capacity" while cpu/mem sit idle.

        The result is cached only once the data-root is known. While the
        daemon is still starting :py:meth:`_sandbox_data_root` returns
        ``None``; we fall back to ``/`` for this call but do NOT cache it,
        so the next hello (each redial re-sends one) re-probes and picks up
        the real root. The control plane also reconciles ``disk_bytes``
        from the heartbeat's ``total_disk_bytes``, which closes the window
        for a node that connected before its daemon was up.
        """
        if self._hw is not None:
            return self._hw
        data_root = self._sandbox_data_root()
        hw = probe_hardware(data_root if data_root is not None else "/")
        if data_root is None:
            LOGGER.warning(
                "node %s: sandbox data-root not resolved yet (backend daemon "
                "still starting?) — advertising disk_bytes measured against "
                "'/' (%.1f GiB) for this hello. NOT cached: the next hello "
                "re-probes, and the control plane reconciles disk_bytes from "
                "the heartbeat.",
                self.node_id, hw.disk_bytes / 1024 ** 3,
            )
            return hw
        self._hw = hw
        return hw

    def _sandbox_data_root(self) -> str | None:
        """Path on the volume sandboxes write into, or ``None`` if unknown.

        Asks the backends for their resolved data-root (docker's
        ``DockerRootDir``, via ``disk_monitor_path``), preferring ``docker``
        — the only backend that carries raw containers in phase 1. Returns
        ``None`` rather than guessing ``/``: the caller distinguishes
        "unknown" from a genuine single-volume host so it doesn't cache a
        wrong reading. Defensive throughout — a test double without the
        hook, a non-dict ``backends`` (legacy list-of-strings fixtures), or
        a raising probe all collapse to ``None``.

        Cost: ``disk_monitor_path`` resolves ``DockerRootDir`` via one
        ``docker info`` and caches it for the daemon's lifetime, so this is
        at most one daemon round-trip per node process — and usually zero,
        because the image cache's disk sampling has already populated that
        cache by the time a redial re-sends hello. The uncached case is the
        first hello at process start, when the node is idle and ``info`` is
        fast. Later hellos on a busy node hit the cache and never touch the
        daemon.
        """
        backends = self._config.backends
        if not isinstance(backends, dict):
            return None
        ordered = [b for name, b in backends.items() if name == "docker"]
        ordered += [b for name, b in backends.items() if name != "docker"]
        for backend in ordered:
            probe = getattr(backend, "disk_monitor_path", None)
            if probe is None:
                continue
            try:
                path = probe()
            except Exception:
                LOGGER.debug(
                    "node %s: data-root probe raised on a backend; trying "
                    "the next one", self.node_id, exc_info=True,
                )
                continue
            if path:
                return str(path)
        return None

    def disk_state(self) -> tuple[int, int]:
        """Issue #14 — last-known ``(free_bytes, total_bytes)`` for the
        node's disk pool. The in-process transport caches the most
        recent value the image-cache backend reported (refreshed on
        every ``ImageCacheManager.report()`` and on every sweep tick).
        Returns ``(0, 0)`` when no image cache is wired (stripped test
        fixtures), which the placement gate treats as "unknown /
        healthy" so legacy fixtures don't accidentally trip the gate.
        """
        cache = self._image_cache
        if cache is None:
            return 0, 0
        return (
            cache.last_free_disk_bytes,
            cache.last_total_disk_bytes,
        )

    def seconds_since_last_command_timeout(self) -> float | None:
        """Issue #18 (Ask #2) — always ``None`` for the in-process
        transport: same-process method calls have no wire reply-
        timeout to record. The scheduler's node-health gate is a
        no-op for laptop / single-process deployments."""
        return None

    # ── Sandbox lifecycle ────────────────────────────────────────────────────

    async def create_sandbox(
        self,
        *,
        rollout_id: str,
        backend: str,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
        stub_request_timeout_s: float | None = None,
    ) -> SandboxHandle:
        with get_tracer().start_as_current_span(
            "xrlenv.node.create_sandbox",
            attributes={
                "rollout_id": rollout_id,
                "backend": backend,
                "image": template.image,
                "node_id": self.node_id,
            },
        ):
            return await self._create_sandbox_impl(
                rollout_id=rollout_id,
                backend=backend,
                template=template,
                resources=resources,
                network_policy=network_policy,
                stub_request_timeout_s=stub_request_timeout_s,
            )

    async def _create_sandbox_impl(
        self,
        *,
        rollout_id: str,
        backend: str,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
        stub_request_timeout_s: float | None,
    ) -> SandboxHandle:
        if backend not in self._config.backends:
            raise ValueError(
                f"node {self.node_id} does not have backend {backend!r} "
                f"(available: {self.supported_backends()})"
            )

        driver = self._config.backends[backend]

        # Spec 15 ensure-present: pull the template image (and evict cold to
        # make room) *before* asking the backend to create the sandbox, so
        # the backend's create call never blocks on a fresh registry pull
        # while holding scheduler/admission state. The cache manager is
        # optional — when not wired, the backend's own create-time pull is
        # the safety net (the existing slice-1 path).
        if self._image_cache is not None:
            await self._image_cache.ensure_present(template.image)
            self._image_cache.acquire(template.image)
        try:
            handle = await driver.create(template, resources, network_policy)
        except BaseException:
            if self._image_cache is not None:
                self._image_cache.release(template.image)
            raise

        record = _SandboxRecord(
            handle=handle,
            template=template.name,
            backend=backend,
            image=template.image,
            # A5 / D17 stage 1: stage the coordinator-derived HTTP
            # cap on the record BEFORE any stub-touching call so
            # init_cmd's StubClient is built with the manifest-
            # derived cap, not the 1 h default. Audit response: this
            # closes H2 — the prior path injected the cap through
            # env_setup's init_params, which run_in_sandbox(init_cmd)
            # bypassed by triggering ``_stub_for`` first.
            stub_request_timeout_s_override=stub_request_timeout_s,
        )
        async with self._lock:
            self._sandboxes[handle.id] = record
        LOGGER.info(
            "node=%s rollout=%s sandbox=%s created (template=%s, backend=%s)",
            self.node_id,
            rollout_id,
            handle.id,
            template.name,
            backend,
        )
        return handle

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        async with self._lock:
            record = self._sandboxes.pop(sb.id, None)
        if record is not None and record.stub is not None:
            await record.stub.close()

        driver = self._config.backends.get(sb.backend)
        if driver is None:
            LOGGER.warning("destroy_sandbox: backend %s not registered", sb.backend)
            return
        try:
            await driver.destroy(sb)
        finally:
            # Spec 15 release: drop the image-cache refcount even when the
            # backend's destroy raises, otherwise the image stays "in_use"
            # forever and the LRU sweep can't evict it.
            if (
                self._image_cache is not None
                and record is not None
                and record.image is not None
            ):
                self._image_cache.release(record.image)
        LOGGER.info("node=%s sandbox=%s destroyed", self.node_id, sb.id)

    async def fetch_trajectory(
        self,
        rollout_id: str,
        *,
        range_kind: FetchRangeKind = "whole",
        step_start: int = 0,
        step_end: int | None = None,
    ) -> Trajectory:
        """Spec 17 §"Fetch": serve a sealed trajectory body to the
        control-plane viewer.

        Reads the local platform-jsonl run dir via
        :class:`LocalTrajectoryReader` and slices according to ``range_kind``.
        Raises :class:`FileNotFoundError` when the run dir is absent — the
        node-side gRPC dispatch maps that into a ``CommandReply`` with
        ``status=FAILED`` so the cache + viewer surface ``ReplayUnavailable``.
        """
        if self._trajectory_reader is None:
            raise RuntimeError(
                f"node {self.node_id} has no trajectory reader configured; "
                "wire LocalRuntime / xrlenv-node serve with a runs_root"
            )
        return await asyncio.to_thread(
            self._trajectory_reader.read_range,
            rollout_id,
            range_kind=range_kind,
            step_start=step_start,
            step_end=step_end,
        )

    async def query_image(self, image: str) -> ImageQueryResult:
        """A1 / D18+D19 (P1.2) — answer "do you have this image?" so
        the scheduler can do image-affinity placement and the
        coordinator can do pre-flight existence checks.

        Delegates to the wired :class:`ImageCacheManager` when
        present; otherwise falls back to a backend-direct check
        (no last-used / digest metadata in that case). The
        latter path keeps tests + LocalRuntime configurations
        without an explicit image cache wired honest.
        """
        if self._image_cache is not None:
            return await self._image_cache.query(image)
        # No cache wired (test fixtures, minimal LocalRuntime). Ask
        # the first backend that supports the lookup; phase-0 nodes
        # have one backend, so this is unambiguous.
        for driver in self._config.backends.values():
            try:
                present = await driver.image_exists(image)
            except Exception:
                LOGGER.debug(
                    "query_image: backend.image_exists raised for %s; "
                    "treating as absent",
                    image, exc_info=True,
                )
                continue
            return ImageQueryResult(present=present)
        return ImageQueryResult(present=False)

    async def report_images(
        self, *, include_shared_size: bool = False,
    ) -> NodeImageReport:
        """B7.6 (P1.2.c) — full per-node image cache snapshot.

        Asked on demand by the admin ``/images`` route via the
        spec-21 ``ReportImagesCommand``. Delegates to the wired
        :class:`ImageCacheManager` when present; falls back to an
        empty report when the agent was constructed without a cache
        (test fixtures, minimal LocalRuntime configurations) so
        callers don't crash on a missing optional dependency.

        Operator diagnostic: logs a warning whenever the empty-fallback
        path runs in production (gRPC-attached node), so a stale
        node-side daemon that predates the cache-wiring fix surfaces
        in the node logs instead of silently rendering "Cache is
        empty / 0.00 GiB" in the admin UI.
        """
        if self._image_cache is not None:
            return await self._image_cache.report(
                include_shared_size=include_shared_size,
            )
        LOGGER.warning(
            "node=%s report_images: no ImageCacheManager wired; returning "
            "empty report. If you see this on a gRPC-attached node, the "
            "xrlenv-node daemon is running pre-fix code — pull the latest "
            "and restart `xrlenv-node serve`.",
            self._config.node_id,
        )
        return NodeImageReport()

    async def evict_image(
        self, *, image_ref: str, force: bool = False,
        timeout_s: float = 30.0,
    ) -> EvictOutcome:
        """Operator-driven node-cache eviction (``xrlenv images evict``).

        The in-process :class:`NodeTransport` counterpart to the
        gRPC node link's ``_exec_evict_image`` — delegates to the wired
        :class:`ImageCacheManager.evict_ref`. Falls back to a ``failed``
        outcome when no cache is wired (test fixtures / minimal
        LocalRuntime), mirroring the gRPC path. ``timeout_s`` is part of
        the transport signature but unused in-process (no wire wait)."""
        if self._image_cache is None:
            return EvictOutcome(
                status="failed",
                detail="no ImageCacheManager wired on this node",
            )
        return await self._image_cache.evict_ref(image_ref, force=force)

    async def list_sandbox_ids(self, *, backend: str | None = None) -> list[str]:
        """A3 / D15 (P1.1) — return the IDs of sandboxes the agent
        currently tracks in its in-memory ``_sandboxes`` table.

        Used by the spec-09 GC layer 3 reconciler in
        :class:`~xrlenv.control.gc_reconciler.GCReconciler` as the
        node side of the reverse query — the control plane diffs
        the returned set against ``state.list_sandboxes()`` to
        spot orphans in either direction.

        ``backend`` filter is reserved for phase-2 mixed-backend
        hosts; phase-0 nodes always have one backend and may pass
        ``None`` to return all.

        Async to match :class:`~xrlenv.control.node_transport.NodeTransport`'s
        Protocol signature (the gRPC sibling :py:meth:`RemoteNodeTransport.list_sandbox_ids`
        is genuinely async); the body itself is a sync dict read.
        """
        async with self._lock:
            if backend is None:
                return list(self._sandboxes)
            return [
                sid for sid, rec in self._sandboxes.items()
                if rec.backend == backend
            ]

    async def gc_orphans(self) -> list[str]:
        """Spec 09 GC layer 2: destroy sandboxes the agent doesn't know about.

        Walks every registered backend's :py:meth:`list_owned_sandboxes` and
        diffs the result against the in-memory ``_sandboxes`` table. Anything
        the backend reports but the agent's table doesn't track is presumed
        orphaned by a previous node-agent process and destroyed. Returns the
        list of sandbox ids that were reaped (empty list when the host is
        clean).

        Called from the node-CLI's startup path *before* the gRPC link is
        opened so the control plane never schedules new work onto a host
        with stale containers competing for the cgroup budget.
        """
        reaped: list[str] = []
        async with self._lock:
            known_ids = set(self._sandboxes.keys())
        for backend_name, driver in self._config.backends.items():
            try:
                live = await driver.list_owned_sandboxes()
            except Exception:
                LOGGER.exception(
                    "gc_orphans: backend=%s list_owned_sandboxes failed; skipping",
                    backend_name,
                )
                continue
            for handle in live:
                if handle.id in known_ids:
                    continue
                LOGGER.warning(
                    "gc_orphans: node=%s backend=%s reaping orphan sandbox=%s "
                    "(backend_ref=%s)",
                    self.node_id, backend_name, handle.id, handle.backend_ref,
                )
                try:
                    await driver.destroy(handle)
                    reaped.append(handle.id)
                except Exception:
                    LOGGER.exception(
                        "gc_orphans: failed to destroy orphan sandbox=%s",
                        handle.id,
                    )
        return reaped

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        driver = self._config.backends[sb.backend]
        return await driver.stats(sb)

    async def put_archive(
        self,
        sb: SandboxHandle,
        target_dir: str,
        tarball: bytes,
        *,
        clean_target: bool = False,
    ) -> None:
        """D12 stage 1: extract tar(.gz) bytes into ``target_dir`` inside
        the sandbox. Delegates to the backend's ``put_archive``; bypasses
        the in-sandbox stub because verifier asset injection is a
        platform action, not an agent action.

        ``clean_target=True`` requests a root-backed pre-wipe of
        ``target_dir`` — see :py:meth:`SandboxBackend.put_archive` for
        the full contract.
        """
        driver = self._config.backends[sb.backend]
        await driver.put_archive(sb, target_dir, tarball, clean_target=clean_target)

    async def run_in_sandbox(
        self,
        sb: SandboxHandle,
        cmd: list[str],
        *,
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Slice 3.5: spec-21 RunInSandbox dispatched in-process via the stub.

        The remote (gRPC) transport's :py:meth:`run_in_sandbox` ships the
        same call as a ``RunInSandboxCommand`` to the node's outbound link;
        keeping the surface uniform across both transports lets the
        coordinator route ``manifest.init_cmd`` through ``NodeTransport``
        regardless of which transport is in play.
        """
        client = await self._stub_for(sb)
        reply = await client.commands(
            cmd, timeout_s=timeout_s, cwd=cwd, env=env
        )
        return ExecResult(
            exit_code=int(reply.get("exit_code") or 0),
            stdout=(reply.get("stdout") or "").encode("utf-8"),
            stderr=(reply.get("stderr") or "").encode("utf-8"),
            timed_out=bool(reply.get("timed_out") or False),
        )

    # ── Raw container session (P1.7.A.1, spec-21 raw-container family) ──────
    #
    # Case 2/3 evaluation harnesses (swebench, harbor, OSWorld) talk to docker
    # directly — they don't ship the in-sandbox stub layer. These methods
    # delegate to ``RawContainerManager`` which wraps the docker daemon
    # without going through ``_stub_for(sb)``. Per-rollout ownership is
    # enforced by the manager's ``container_id ↔ rollout_id`` map.

    async def acquire_container(
        self,
        *,
        rollout_id: str,
        backend: str,
        image: str,
        command: list[str] | None = None,
        entrypoint: list[str] | None = None,
        user: str | None = None,
        cap_add: list[str] | None = None,
        devices: list[str] | None = None,
        privileged: bool = False,
        network_mode: str | None = None,
        binds: list[str] | None = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        resources: ResourceSpec | None = None,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawContainerRecord:
        # Issue #12 — pull / build deadline override forwarded to the
        # raw-container manager, which threads it into
        # ``ImageCacheManager.ensure_present(deadline_s=...)``. ``None``
        # falls back to the manager's default (currently 600 s).
        # P1 — ``resources`` is the effective ResourceSpec; the manager
        # turns its CPU/memory limits into docker cgroup kwargs.
        # P0b — ``runtime_limits`` carries the container-shape limits.
        return await self._require_raw_manager(backend).acquire(
            rollout_id=rollout_id,
            image=image,
            command=command,
            entrypoint=entrypoint,
            user=user,
            cap_add=cap_add,
            devices=devices,
            privileged=privileged,
            network_mode=network_mode,
            binds=binds,
            name=name,
            labels=labels,
            environment=environment,
            ensure_image_present=ensure_image_present,
            userns_mode=userns_mode,
            ensure_image_deadline_s=acquire_timeout_s,
            resources=resources,
            runtime_limits=runtime_limits,
            container_runtime=container_runtime,
        )

    async def container_exec(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        backend: str = "docker",
    ) -> dict[str, Any]:
        return await self._require_raw_manager(backend).exec(
            rollout_id=rollout_id,
            container_id=container_id,
            cmd=cmd,
            timeout_s=timeout_s,
            cwd=cwd,
            env=env,
            user=user,
        )

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
        backend: str = "docker",
    ) -> None:
        await self._require_raw_manager(backend).apply_egress(
            rollout_id=rollout_id,
            container_id=container_id,
            allowlist=allowlist,
            dns_resolver=dns_resolver,
        )

    async def destroy_container(
        self, *, rollout_id: str, container_id: str,
        force: bool = True, backend: str = "docker",
    ) -> None:
        await self._require_raw_manager(backend).destroy(
            rollout_id=rollout_id,
            container_id=container_id,
            force=force,
        )

    async def acquire_compose_project(
        self,
        *,
        rollout_id: str,
        project_name: str,
        compose_yaml: str,
        images: list[str] | None = None,
        main_service: str = "main",
        up_timeout_s: float | None = None,
        backend: str = "docker",
    ) -> ComposeProjectRecord:
        """P1.7.C.2 — bring up a multi-service compose project and register each
        member ↔ rollout_id (so the container-scoped exec/archive path addresses
        ``main`` unchanged). Delegates to the raw-container manager."""
        return await self._require_raw_manager(backend).acquire_compose_project(
            rollout_id=rollout_id,
            project_name=project_name,
            compose_yaml=compose_yaml,
            images=images or [],
            main_service=main_service or "main",
            up_timeout_s=up_timeout_s,
        )

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str,
        force: bool = True, backend: str = "docker",
    ) -> None:
        """P1.7.C.2 — ``docker compose down`` the whole project + deregister every
        member. Reaps the entire stack, never a lone sidecar."""
        await self._require_raw_manager(backend).destroy_compose_project(
            rollout_id=rollout_id,
            project_name=project_name,
            force=force,
        )

    async def container_put_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        target_dir: str,
        tarball: bytes,
        backend: str = "docker",
    ) -> None:
        await self._require_raw_manager(backend).put_archive(
            rollout_id=rollout_id,
            container_id=container_id,
            target_dir=target_dir,
            tarball=tarball,
        )

    async def container_get_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
        backend: str = "docker",
    ) -> bytes:
        return await self._require_raw_manager(backend).get_archive(
            rollout_id=rollout_id,
            container_id=container_id,
            source_path=source_path,
        )

    def container_get_archive_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
        backend: str = "docker",
    ) -> Any:
        """Streaming get_archive — returns the manager's async
        generator directly (caller iterates with ``async for``). Each
        yielded item is a ``bytes`` chunk of the tar. This is the wire
        dispatch path (``grpc_link``): the tar is read one chunk at a
        time off the event loop so a large ``/testbed`` copy can never
        starve the heartbeat + mark the node lost."""
        return self._require_raw_manager(backend).get_archive_stream(
            rollout_id=rollout_id,
            container_id=container_id,
            source_path=source_path,
        )

    async def list_raw_container_ids(
        self, *, backend: str = "docker",
    ) -> list[str]:
        """Reverse query for the raw-GC reconciler. Returns the
        container_ids docker reports under the
        ``xrlenv.session_kind=raw`` label — independent of the
        manager's in-memory map; the reconciler diffs both to
        find orphans on either side."""
        return await self._require_raw_manager(backend).list_on_docker()

    async def list_raw_containers_info(
        self, *, backend: str = "docker",
    ) -> list[tuple[str, str, str]]:
        """P1.7.C.2 — like :meth:`list_raw_container_ids` but with correlation
        labels: ``(container_id, rollout_id, compose_project)`` per raw container.
        Lets the reconciler recognise + route compose-project mains."""
        return await self._require_raw_manager(backend).list_on_docker_info()

    async def list_managed_container_info(
        self, *, backend: str = "docker",
    ) -> list[tuple[str, str, str, str]]:
        """Audit H11 — EVERY xrlenv-managed container (any ``xrlenv.rollout_id``) WITH labels:
        ``(container_id, rollout_id, compose_project, session_kind)`` — including compose
        SIDECARS the raw-only listing omits. readopt-on-connect uses it to quarantine a node
        with a sidecar-only compose survivor. Name matches the ``NodeTransport`` protocol so the
        in-process runtime can use the agent directly as its transport."""
        return await self._require_raw_manager(backend).list_all_managed_on_docker_info()

    async def force_destroy_raw_container(
        self, *, container_id: str, backend: str = "docker",
    ) -> None:
        """Privileged docker rm -f for the reconciler's node-only
        orphan path. Bypasses the per-rollout ownership check —
        only reachable via the spec-21
        ``ForceDestroyContainerCommand`` (not exposed on
        ``rollout_control.proto``)."""
        await self._require_raw_manager(backend).force_destroy(
            container_id=container_id,
        )

    def container_exec_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        backend: str = "docker",
    ) -> Any:
        """Streaming exec — returns the manager's async generator
        directly. Caller iterates with ``async for``. Each yielded
        chunk dict has the ``ContainerExecChunk`` shape; the
        terminator has ``done=True``."""
        return self._require_raw_manager(backend).exec_stream(
            rollout_id=rollout_id,
            container_id=container_id,
            cmd=cmd,
            timeout_s=timeout_s,
            cwd=cwd,
            env=env,
            user=user,
        )

    def _require_raw_manager(self, backend: str) -> RawContainerManager:
        """Resolve the raw-container manager for ``backend``.

        Falls back to ``"docker"`` when ``backend`` is empty (the
        wire field defaults to "" before the consumer-facing layer
        applies the DEFAULT_BACKEND policy). Raises if no docker
        backend was registered on this node — the dispatcher
        surfaces this as a clean error reply.
        """
        key = backend or "docker"
        mgr = self._raw_managers.get(key)
        if mgr is None:
            raise XRLEnvError(
                f"node {self.node_id!r} has no raw-container "
                f"manager for backend {key!r}. Phase-1 only the "
                f"docker backend supports raw containers.",
            )
        return mgr

    # ── EnvAdapter driver (spec 14) ──────────────────────────────────────────

    async def env_setup(
        self,
        sb: SandboxHandle,
        *,
        adapter_module: str,
        adapter_class: str,
        init_params: dict[str, Any],
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        client = await self._stub_for(sb)
        return await client.env_setup(
            adapter_module=adapter_module,
            adapter_class=adapter_class,
            init_params=init_params,
            sandbox_id=sb.id,
            request_timeout_s=request_timeout_s,
        )

    async def env_step(
        self,
        sb: SandboxHandle,
        action: Any,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        with get_tracer().start_as_current_span(
            "xrlenv.node.env_step",
            attributes={
                "sandbox_id": sb.id,
                "node_id": self.node_id,
            },
        ):
            client = await self._stub_for(sb)
            return await client.env_step(
                action, request_timeout_s=request_timeout_s,
            )

    async def env_teardown(
        self,
        sb: SandboxHandle,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        client = await self._stub_for(sb)
        try:
            return await client.env_teardown(request_timeout_s=request_timeout_s)
        finally:
            # Keep the stub client around — destroy_sandbox is the canonical
            # cleanup point so partial-teardown errors do not orphan the
            # connection.
            pass

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _set_stub_request_timeout(
        self, sb: SandboxHandle, cap_s: float,
    ) -> None:
        """Stage the per-sandbox HTTP cap.

        Primary call site is :py:meth:`create_sandbox` via the
        ``stub_request_timeout_s`` kwarg, which sets the cap before
        any stub-touching operation. Kept as a public-ish helper for
        the test suite + opportunistic late-stage updates (the latter
        will close the existing stub so the next ``_stub_for`` rebuilds
        with the new cap)."""
        old_stub: StubClient | None = None
        async with self._lock:
            record = self._sandboxes.get(sb.id)
            if record is None:
                record = _SandboxRecord(
                    handle=sb, template="<unknown>", backend=sb.backend,
                )
                self._sandboxes[sb.id] = record
            if record.stub_request_timeout_s_override == cap_s:
                return
            record.stub_request_timeout_s_override = cap_s
            if record.stub is not None:
                # A late-stage cap change must close the existing stub
                # so the next ``_stub_for`` rebuilds with the new cap.
                # Closing outside the lock to avoid blocking ``_stub_for``
                # callers on aiohttp shutdown I/O.
                old_stub = record.stub
                record.stub = None
        if old_stub is not None:
            with suppress(Exception):
                await old_stub.close()

    async def _stub_for(self, sb: SandboxHandle) -> StubClient:
        async with self._lock:
            record = self._sandboxes.get(sb.id)
            if record is None:
                # The sandbox may have been created by a backend call we didn't
                # observe in this process (e.g. test fixture); rebuild a record
                # opportunistically rather than fail.
                record = _SandboxRecord(handle=sb, template="<unknown>", backend=sb.backend)
                self._sandboxes[sb.id] = record
            if record.stub is None:
                if sb.stub_endpoint.startswith("unix://") and not Path(
                    sb.stub_endpoint.removeprefix("unix://")
                ).exists():
                    raise FileNotFoundError(
                        f"sandbox {sb.id} stub uds {sb.stub_endpoint} not present"
                    )
                cap = (
                    record.stub_request_timeout_s_override
                    if record.stub_request_timeout_s_override is not None
                    else self._config.stub_request_timeout_s
                )
                record.stub = StubClient(
                    sb.stub_endpoint,
                    request_timeout_s=cap,
                )
            return record.stub

    def _gen_id(self) -> str:
        return uuid.uuid4().hex
