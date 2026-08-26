"""Build coordinator (P1.6.b).

Owns the ``xrlenv build apply`` lifecycle: validate plan → hash → look
up per-benchmark builders + size hints → plan placements → persist
snapshot → dispatch per-node jobs → consume per-image results → mark
plan completed.

LocalRuntime (this slice) wires the coordinator against an
:class:`InProcessNodeBuilder`. P1.6.c plugs the same coordinator into a
gRPC-backed dispatch implementation so distributed clusters share the
control flow.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from xrlenv.control.build_plan import (
    BuildEntry,
    BuildPlan,
    GitSource,
    LocalSource,
    RegistrySource,
    TarballSource,
    compute_plan_id,
)
from xrlenv.control.image_builder import load_image_builder
from xrlenv.control.image_planner import (
    ImageToPlace,
    InsufficientCapacity,
    NodeBudget,
    NodeId,
    PlacementResult,
    PlanAssignment,
    expected_on_disk_bytes,
    plan_opportunistic_placements,
    plan_placements,
)
from xrlenv.control.node_builder import (
    BuilderRef,
    BuildJob,
    NodeBuilder,
)
from xrlenv.control.state import (
    BuildAssignmentRecord,
    BuildAssignmentStatus,
    BuildPlanStatus,
    StateStore,
)
from xrlenv.control.template_catalog import TemplateCatalog, TemplateManifest
from xrlenv.errors import ManifestInvalid
from xrlenv.observability.tracing import get_tracer

# Async callable the coordinator uses to dispatch a single registry-source
# image to a node's local image cache. Returns ``("ok", None)`` on success,
# ``("failed", "<message>")`` on a node-side error, or ``("timeout",
# "<message>")`` on a control-plane WIRE timeout — the node may still complete
# the pull, so the coordinator records it ``registered`` (lazy) rather than
# ``failed``.
EnsurePresentFn = Callable[
    [str, str, float],  # node_id, image_ref, timeout_s
    Awaitable[tuple[str, str | None]],
]

# Async callable the coordinator uses to dispatch a single source-build
# entry to a node. Source can be GitSource or TarballSource (the latter
# ships in a follow-on; today's BuildImageFn impls reject TarballSource
# with an operator-friendly error). Returns ``("ok", None)`` on success
# or ``("failed", "<message>")`` on a node-side build failure. Build
# wall-clock is bounded by ``timeout_s``; longer builds get a clear
# timeout error.
BuildImageFn = Callable[
    [str, str, "GitSource | TarballSource", float, dict[str, str], bool],
    # node_id, image_ref, source, timeout_s, labels, skip_if_present
    Awaitable[tuple[str, str | None]],
]

# Async callable the coordinator uses to dispatch a single source-build-AND-push
# entry (``xrlenv build push``) to a node. Same args as BuildImageFn minus
# skip_if_present — the push path always registry-HEAD-skips (build-once
# fleet-wide, resumable) — and it additionally returns the pushed
# ``<repo>@sha256:...`` digest (``None`` on failure / no push). The node builds
# image_ref then pushes it to the registry the ref encodes; the coordinator
# records the digest into ``BuildOutcome.digests``.
BuildPushFn = Callable[
    [str, str, "GitSource | TarballSource", float, dict[str, str]],
    # node_id, image_ref, source, timeout_s, labels
    Awaitable[tuple[str, str | None, str | None]],
]

# Async callable returning a node's current ``(free_bytes, total_bytes)`` disk
# state, or ``None`` when unknown (no live transport / no heartbeat yet). Lets
# the coordinator pace build dispatch so it doesn't drive a node below its
# eviction reserve — which would otherwise force the node to evict images the
# build just produced (or, worse, hit ENOSPC mid-build).
FreeDiskFn = Callable[[str], Awaitable["tuple[int, int] | None"]]

# Synthetic benchmark tag stored in ``BuildAssignmentRecord.benchmark``
# for per-image-ref plan rows. Distinguishes them from legacy
# benchmark-driven rows in admin queries without requiring a schema
# change.
PER_IMAGE_REF_BENCHMARK_TAG = "<per-image-ref>"

# Default deadline for ``ensure_present`` calls; matches the legacy
# eager-prefetch path (``raw_container_service._eager_prefetch``).
DEFAULT_ENSURE_PRESENT_TIMEOUT_S = 600.0

# Default deadline for source-build dispatch. Builds typically take
# longer than registry pulls — git clone + ``docker build`` on a
# medium context often runs 2-15 minutes — so the default is generous.
# Operators can override per-entry once that schema field lands; for
# now a single coordinator-side default covers everything.
DEFAULT_BUILD_IMAGE_TIMEOUT_S = 1800.0

# Max concurrent per-image dispatches (ensure_present / build_image)
# across all nodes.  Without this, ``asyncio.gather`` over a 500-entry
# plan fires 500 simultaneous docker-pull or docker-build commands,
# overwhelming the Docker daemon and Docker-Hub rate limits on the
# receiving nodes.  The semaphore bounds in-flight RPCs; queued
# coroutines yield the event loop so the admin server stays responsive.
#
# Also bounds peak containerd content-store usage on nodes where the
# local disk is small (containerd retains pulled layer blobs until
# GC'd; see DockerBackend._gc_containerd_content). Override via
# ``XRLENV_BUILD_CONCURRENCY`` env var.
DEFAULT_BUILD_CONCURRENCY = int(os.environ.get("XRLENV_BUILD_CONCURRENCY", "32"))

# Disk-aware dispatch pacing. Before materializing an image on a node, the
# coordinator waits until the node has at least HEADROOM_FACTOR x the largest
# planned image's size free. This keeps a heavy ``build apply`` from driving a
# node below its eviction reserve — which is what made a node oscillate between
# ENOSPC-risk during the build and an over-evicted (too-empty) cache after.
# The wait is bounded; on timeout the dispatch proceeds and lets node-side
# eviction cope (graceful degrade rather than a stalled plan). All three are
# env-tunable; the factor wants to be roughly the node's eviction reserve so
# dispatch backs off before eviction has to fight the build.
DISPATCH_DISK_HEADROOM_FACTOR = float(
    os.environ.get("XRLENV_BUILD_DISK_HEADROOM_FACTOR", "3.0"),
)
DISPATCH_DISK_POLL_S = float(os.environ.get("XRLENV_BUILD_DISK_POLL_S", "5.0"))
DISPATCH_DISK_WAIT_TIMEOUT_S = float(
    os.environ.get("XRLENV_BUILD_DISK_WAIT_TIMEOUT_S", "300.0"),
)

LOGGER = logging.getLogger(__name__)


class NodeBudgetProvider(Protocol):
    """Runtime hands the coordinator the per-node disk budget snapshot.

    LocalRuntime computes this in-process; DistributedRuntime queries
    each node's heartbeat (P1.6.c). Returned bytes are *available for
    image cache*, i.e. capacity - reserved_runtime - buffer.
    """

    async def get_budgets(
        self, *, reserved_runtime_gb: int, buffer_gb: int,
        cap_per_node_gb: int | None,
    ) -> list[NodeBudget]: ...


class ClusterInventoryProvider(Protocol):
    """Runtime hands the coordinator a snapshot of which nodes currently
    have which image refs cached. Used by ``--fill-missing`` apply mode
    to find entries that need rebuilding without re-dispatching the
    ones already present.

    Returns ``{image_ref: set[node_id]}`` for every image visible across
    the connected fleet. Empty dict when no nodes are connected or all
    reports fail.
    """

    async def get_inventory(self) -> dict[str, set[str]]: ...


def _shard_for_push(
    images: Sequence[ImageToPlace], nodes: Sequence[NodeBudget],
) -> PlacementResult:
    """Size-balanced (LPT-greedy) shard of build+push work across the connected
    nodes, WITHOUT the FFD fit constraint ``plan_placements`` imposes.

    ``xrlenv build push`` images are TRANSIENT — built, pushed to the registry,
    then evictable (nothing pins them on-node) — so a node's shard need not fit
    its disk all at once; the dispatch loop's disk-aware pacing plus the
    image-cache LRU (which reclaims already-pushed images) keep real disk
    bounded. That is what lets a push plan far larger than total cluster disk
    (the 1376-image SETA plan the Slurm build host handled) fan out natively.
    Every image is assigned to exactly one node; replication is not meaningful
    for a push-once-to-the-shared-registry campaign.
    """
    load: dict[NodeId, int] = {n.node_id: 0 for n in nodes}
    by_node: dict[NodeId, list[PlanAssignment]] = {n.node_id: [] for n in nodes}
    # Longest-processing-time first: the biggest builds go to the least-loaded
    # node so build-bytes stay balanced across the fleet.
    for img in sorted(images, key=lambda i: i.size_bytes, reverse=True):
        nid = min(load, key=lambda k: load[k])
        by_node[nid].append(PlanAssignment(
            image_ref=img.image_ref, node_id=nid,
            benchmark=PER_IMAGE_REF_BENCHMARK_TAG, size_bytes=img.size_bytes,
        ))
        load[nid] += img.size_bytes
    return PlacementResult(
        assignments=tuple(a for rows in by_node.values() for a in rows),
        assignments_by_node={
            nid: tuple(rows) for nid, rows in by_node.items() if rows
        },
    )


class BuildOutcome(BaseModel):
    """Coordinator's return value from ``apply``."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    status: str
    """One of ``in_flight`` (dry-run never reaches this), ``completed``,
    ``partial_failure``, ``rejected_in_flight``, or ``no_op_already_completed``."""
    placement: PlacementResult | None = None
    """Always populated. ``None`` only when the plan was rejected before
    placement (e.g. another plan in flight)."""
    successes: int = 0
    failures: int = 0
    deferred: int = 0
    """P1.6.g (F2=2) — number of images registered as ``status=registered``
    that didn't fit the budget at apply time. Lazy-built on first
    ``ensure_present`` via the H3 builder hook. ``0`` in eager mode."""
    error_summary: list[str] = []
    digests: dict[str, str] = {}
    """``xrlenv build push`` only — image_ref → the pushed ``<repo>@sha256:...``
    for every entry the fleet built and pushed. Empty for local-tag builds
    (``build apply``). The CLI emits these as a registry-source pin plan so the
    fleet pulls digest-pinned images (invariant 4)."""


class BuildCoordinator:
    """Apply build plans against a runtime.

    Constructed once per runtime (LocalRuntime / DistributedRuntime in
    P1.6.c). The state store + catalog references are shared with the
    rollout coordinator so admin queries against either path see the
    same persisted snapshot.
    """

    def __init__(
        self,
        *,
        catalog: TemplateCatalog,
        state: StateStore,
        node_builder: NodeBuilder,
        budget_provider: NodeBudgetProvider,
        ensure_present_fn: EnsurePresentFn | None = None,
        build_image_fn: BuildImageFn | None = None,
        build_push_fn: BuildPushFn | None = None,
        inventory_provider: ClusterInventoryProvider | None = None,
        free_disk_fn: FreeDiskFn | None = None,
    ) -> None:
        self._catalog = catalog
        self._state = state
        self._node_builder = node_builder
        self._budget_provider = budget_provider
        self._inventory_provider = inventory_provider
        """Optional cluster-inventory hook. When wired (DistributedRuntime
        provides one; LocalRuntime can pass ``None`` since single-host
        plans don't need it), enables the ``fill_missing`` apply mode
        that targets only entries absent from every connected node."""
        self._ensure_present_fn = ensure_present_fn
        """Dispatch hook for registry-source per-image-ref entries. The
        runtime wires an async callable that lowers
        ``(node_id, image_ref)`` to the node-side image-cache
        ``ensure_present`` path. ``None`` disables registry dispatch —
        applying an entries-shaped plan with registry entries in that
        case raises a clear ``ManifestInvalid``."""
        self._build_image_fn = build_image_fn
        """Dispatch hook for source-build per-image-ref entries
        (``GitSource``, ``TarballSource``). The runtime wires an async
        callable that lowers
        ``(node_id, image_ref, source, timeout_s, labels)`` to the
        node-side ``GitSourceBuilder`` / ``TarballSourceBuilder``.
        ``None`` disables source-build dispatch — applying a plan with
        git/tarball entries in that case raises a clear
        ``ManifestInvalid``."""
        self._build_push_fn = build_push_fn
        """Dispatch hook for source-build-AND-push entries (``xrlenv build
        push``). Parallel to ``build_image_fn`` but the node pushes the built
        image to the registry the ref encodes and returns the digest. ``None``
        disables push mode — ``apply(push=True)`` with git/tarball entries then
        raises a clear ``ManifestInvalid``."""
        self._free_disk_fn = free_disk_fn
        """Optional live per-node disk probe. When wired (DistributedRuntime
        passes ``transport.disk_state``), build dispatch paces itself against
        each node's free disk so a heavy plan doesn't overrun the node's
        eviction reserve. ``None`` (LocalRuntime / tests) disables the gate —
        single-host applies don't need it."""

    def _dispatch_watermark_bytes(
        self, assignments: Sequence[Any], entry_by_ref: dict[str, Any],
    ) -> int:
        """Free-disk watermark below which build dispatch to a node pauses.

        ``max(planned image size) x HEADROOM_FACTOR`` — adaptive to the
        workload's image sizes (no fixed fraction of disk). Returns ``0`` when
        no size hints are known (uncalibrated plan) or no disk probe is wired,
        which makes the gate a no-op rather than blocking blindly.
        """
        if self._free_disk_fn is None:
            return 0

        def _size_hint(entry: Any) -> int:
            # size_hint_bytes lives on the nested EntryPlacement, not on
            # BuildEntry directly. Defensive: tolerate a missing placement /
            # hint so the watermark degrades to 0 (gate off) rather than
            # raising and killing the whole apply.
            placement = getattr(entry, "placement", None)
            return int(getattr(placement, "size_hint_bytes", 0) or 0)

        max_image = max(
            (
                _size_hint(entry_by_ref[a.image_ref])
                for a in assignments if a.image_ref in entry_by_ref
            ),
            default=0,
        )
        return int(max_image * DISPATCH_DISK_HEADROOM_FACTOR)

    async def _await_disk_headroom(
        self, node_id: str, watermark_bytes: int,
    ) -> None:
        """Block until ``node_id`` reports ``watermark_bytes`` free, bounded.

        No-op when the gate is disabled (no probe, or watermark ``0``). On
        timeout it proceeds anyway and lets node-side eviction reclaim — a
        single slow/full node must not wedge the whole plan. Reads the probe
        (heartbeat-cached ``disk_state``) — no per-poll daemon round-trip.
        """
        if self._free_disk_fn is None or watermark_bytes <= 0:
            return
        deadline = time.monotonic() + DISPATCH_DISK_WAIT_TIMEOUT_S
        announced = False
        while True:
            state = await self._free_disk_fn(node_id)
            free = state[0] if state else None
            if free is None or free >= watermark_bytes:
                return
            if time.monotonic() >= deadline:
                LOGGER.warning(
                    "build dispatch to %s: disk-headroom wait timed out "
                    "(free=%dB < watermark=%dB); proceeding — node eviction "
                    "will reclaim", node_id, free, watermark_bytes,
                )
                return
            if not announced:
                LOGGER.info(
                    "build dispatch to %s paused: free=%dB < watermark=%dB; "
                    "waiting for node eviction to free space",
                    node_id, free, watermark_bytes,
                )
                announced = True
            await asyncio.sleep(DISPATCH_DISK_POLL_S)

    async def apply(
        self,
        plan: BuildPlan,
        *,
        dry_run: bool = False,
        force: bool = False,
        eager: bool = False,
        fill_missing: bool = False,
        bypass_in_flight_check: bool = False,
        applied_by: str = "local",
        skip_if_present: bool = False,
        concurrency: int | None = None,
        push: bool = False,
    ) -> BuildOutcome:
        with get_tracer().start_as_current_span(
            "xrlenv.coordinator.build_apply",
            attributes={
                "dry_run": dry_run,
                "force": force,
                "eager": eager,
                "fill_missing": fill_missing,
                "bypass_in_flight_check": bypass_in_flight_check,
                "skip_if_present": skip_if_present,
                "applied_by": applied_by,
                "concurrency": concurrency if concurrency is not None else -1,
                "push": push,
            },
        ):
            return await self._apply_impl(
                plan,
                dry_run=dry_run,
                force=force,
                eager=eager,
                fill_missing=fill_missing,
                bypass_in_flight_check=bypass_in_flight_check,
                applied_by=applied_by,
                skip_if_present=skip_if_present,
                concurrency=concurrency,
                push=push,
            )

    async def _apply_impl(
        self,
        plan: BuildPlan,
        *,
        dry_run: bool,
        force: bool,
        eager: bool,
        fill_missing: bool = False,
        bypass_in_flight_check: bool = False,
        applied_by: str,
        skip_if_present: bool,
        concurrency: int | None = None,
        push: bool = False,
    ) -> BuildOutcome:
        """Apply ``plan`` against the runtime.

        ``dry_run=True`` produces the placement without dispatching;
        useful for ``xrlenv build apply --dry-run`` to print what would
        happen. ``force=True`` propagates to each builder's ``build()``
        call so already-tagged images get rebuilt; idempotency layer 1
        (the plan_id check) still applies — re-applying the same plan
        is still a content-addressed no-op when its prior status was
        ``completed``.

        ``skip_if_present=True`` (per-image-ref plans only) makes the
        per-(node, image_ref) source builder short-circuit when the
        image is already tagged on the chosen node — used by the
        operator-driven ``xrlenv build apply --skip-if-present`` flow
        for warm-cluster re-applies (post-calibrate, partial-failure
        recovery). Default ``False`` preserves the existing "always
        rebuild" behavior so existing operator scripts don't silently
        change semantics. ``force`` overrides ``skip_if_present``:
        ``force=True`` always dispatches even if present.

        ``eager=False`` (default — P1.6.g, F2=2) opportunistic mode:
        registers all assignments as ``status="registered"``, then
        pre-dispatches the bin-packer-fitted subset for synchronous
        build. Overflow rows stay ``registered`` and lazy-build on
        first ``ensure_present`` (matches H3 rotate-via-eviction
        model). The plan transitions to ``completed`` once all
        pre-dispatched rows reach terminal status — deferred rows
        don't block completion.

        ``eager=True`` preserves the original P1.6.b/c behavior:
        bin-packer raises :class:`InsufficientCapacity` if anything
        doesn't fit; every assignment goes ``pending → building →
        done|failed`` synchronously.
        """
        # P1.7.C.2 — per-image-ref shape branches off here. Legacy
        # benchmark-driven dispatch continues below.
        if plan.is_per_image_ref():
            # ``local`` sources are build-host-only: they docker-build a path in
            # place and assume it exists on the building host. The cluster
            # build-apply path ships sources to nodes that may not share that
            # path, so reject them here with a clear remediation rather than
            # silently dropping them at the per-source partition. Express them as
            # git/tarball + ``xrlenv build push``, or build on a shared-fs host.
            local_refs = [
                e.image_ref for e in plan.entries
                if isinstance(e.context_source, LocalSource)
            ]
            if local_refs:
                raise ManifestInvalid(
                    "build plan rejected: 'local' context sources are "
                    "build-host-only and aren't supported on the cluster "
                    "build-apply/push paths (they ship sources to nodes that may "
                    "not share the path). Either express them as git/tarball "
                    "sources and run ``xrlenv build push`` to build+push them "
                    "across the fleet, or build them with "
                    "deploy/registry/build_and_push_images.py on a shared-fs build host "
                    "and apply a registry-source plan. Offending entries: "
                    + ", ".join(local_refs[:10])
                    + (" ..." if len(local_refs) > 10 else ""),
                )
            return await self._apply_per_image_ref(
                plan, dry_run=dry_run, force=force, eager=eager,
                fill_missing=fill_missing,
                bypass_in_flight_check=bypass_in_flight_check,
                applied_by=applied_by,
                skip_if_present=skip_if_present and not force,
                concurrency=concurrency,
                push=push,
            )

        # 1. Validate every benchmark referenced has an image_builder.
        per_benchmark_builder: dict[str, BuilderRef] = {}
        per_benchmark_size: dict[str, int] = {}
        for spec in plan.benchmarks:
            manifest = self._lookup_manifest(spec.name)
            if manifest.image_builder is None:
                raise ManifestInvalid(
                    f"benchmark {spec.name!r} doesn't declare an "
                    f"image_builder block; plug-ins must ship a "
                    f"BenchmarkImageBuilder to be xrlenv-build-apply-compatible",
                )
            per_benchmark_builder[spec.name] = BuilderRef(
                module=manifest.image_builder.module,
                class_name=manifest.image_builder.class_name,
            )
            per_benchmark_size[spec.name] = self._read_size_hint(
                manifest.image_builder,
            )

        # 2. Enumerate the concrete image refs each benchmark would
        # build, then construct ImageToPlace rows.
        images: list[ImageToPlace] = []
        per_benchmark_kwargs: dict[str, dict[str, str]] = {}
        for spec in plan.benchmarks:
            ref = per_benchmark_builder[spec.name]
            from xrlenv.control.image_builder import ImageBuilderDecl

            decl = ImageBuilderDecl.model_validate({
                "module": ref.module, "class": ref.class_name,
            })
            builder = load_image_builder(decl)
            try:
                refs = builder.enumerate_image_refs(
                    selection=spec.selection.to_kwargs(),
                )
            except Exception as exc:
                raise ManifestInvalid(
                    f"benchmark {spec.name!r}: enumerate_image_refs failed: {exc}",
                ) from exc
            r = spec.replication or plan.replication
            for image_ref in refs:
                images.append(ImageToPlace(
                    image_ref=image_ref,
                    size_bytes=per_benchmark_size[spec.name],
                    replication=r,
                    benchmark=spec.name,
                ))
            if spec.build_path is not None:
                per_benchmark_kwargs[spec.name] = {"build_path": spec.build_path}

        # 3. Ask the runtime for the per-node budget snapshot.
        nodes = await self._budget_provider.get_budgets(
            reserved_runtime_gb=plan.budget.reserved_runtime_gb,
            buffer_gb=plan.budget.buffer_gb,
            cap_per_node_gb=plan.budget.cap_per_node_gb,
        )

        # 4. Plan placements. Two modes:
        #
        # - eager=True: legacy P1.6.b/c semantics. Bin-packer raises
        #   InsufficientCapacity on overflow; every image is
        #   dispatched synchronously.
        # - eager=False (default — P1.6.g, F2=2): opportunistic FFD
        #   places what fits, records the rest as deferred. Deferred
        #   rows persist as ``status="registered"`` and lazy-build via
        #   the ``ensure_present`` hook on first rollout.
        deferred: tuple[Any, ...] = ()
        if eager:
            try:
                placement = plan_placements(images, nodes)
            except InsufficientCapacity:
                raise  # surface to the CLI
        else:
            opp = plan_opportunistic_placements(images, nodes)
            placement = opp.placed
            deferred = opp.deferred

        # 5. Compute plan_id; check existing state.
        plan_id = compute_plan_id(plan)
        plan_json = plan.model_dump_json(exclude_none=True)

        if dry_run:
            return BuildOutcome(
                plan_id=plan_id, status="dry_run",
                placement=placement,
                deferred=len(deferred),
            )

        # Idempotency layers 1 + 2 — see notes/phase-1-to-do.md
        # "Slice P1.6 — control-plane-driven image builds":
        #
        # - in_flight  → reject (concurrency control: fork E (1)).
        # - completed  → no-op UNLESS force=True (M1: --force escape
        #   hatch must actually escape, mirroring the CLI/docs claim
        #   that it rebuilds after upstream Dockerfile bumps).
        # - partial_failure → residual-only retry by default (M2):
        #   load existing assignments and dispatch JUST the failed
        #   rows; preserve done rows + their timestamps. force=True
        #   re-dispatches everything.
        existing = self._state.get_build_plan(plan_id)
        residual_only = False
        if existing is not None:
            if existing.status == "in_flight" and not bypass_in_flight_check:
                return BuildOutcome(
                    plan_id=plan_id, status="rejected_in_flight",
                    placement=placement,
                )
            if existing.status == "completed" and not force:
                return BuildOutcome(
                    plan_id=plan_id, status="no_op_already_completed",
                    placement=placement,
                )
            if existing.status == "partial_failure" and not force:
                residual_only = True

        # 6. Persist the plan record + assignments. residual_only
        # preserves prior done rows (M2); fresh applies upsert all rows
        # to ``pending``. force=True against an existing plan also goes
        # through the fresh-apply path so the operator's intent is
        # honored.
        self._state.record_build_plan(
            plan_id=plan_id, applied_by=applied_by, plan_json=plan_json,
            name=plan.name,
        )
        # Move plan back to in_flight on retry/force so the in-flight
        # rejection branch above protects against a concurrent apply.
        if existing is not None:
            self._state.update_build_plan_status(plan_id, "in_flight")

        if residual_only:
            existing_rows = {
                (r.node_id, r.image_ref): r
                for r in self._state.list_assignments(plan_id)
            }
            dispatch_assignments = [
                a for a in placement.assignments
                if existing_rows.get((a.node_id, a.image_ref)) is None
                or existing_rows[(a.node_id, a.image_ref)].status == "failed"
            ]
            # Audit P1.6-M2a (2026-05-05): when the live-budget probe
            # returns different free-disk numbers between the first
            # apply and the retry, FFD can replan a previously-failed
            # image onto a *different* node. The new ``(node_id,
            # image_ref)`` pair has no existing row, so the
            # ``update_assignment_status(... building)`` call inside
            # the per-node task would raise ``KeyError`` on the first
            # tick and leave the plan stuck in ``in_flight``. Materialize
            # those rows up front. Old failed rows (for the same
            # image_ref at a now-vacated node) are intentionally
            # preserved in the snapshot so the operator can see the
            # placement history; a phase-2 follow-on may add a
            # ``superseded`` status to clean up the rollup.
            for a in dispatch_assignments:
                if (a.node_id, a.image_ref) not in existing_rows:
                    self._state.record_assignment(BuildAssignmentRecord(
                        plan_id=plan_id,
                        node_id=a.node_id,
                        image_ref=a.image_ref,
                        benchmark=a.benchmark,
                        status="pending",
                    ))
            # Pre-existing done rows count as successes for the
            # final outcome rollup so re-applying a partial failure
            # that recovers the last brick reads as `completed`.
            preexisting_successes = sum(
                1 for r in existing_rows.values() if r.status == "done"
            )
        else:
            # P1.6.g — fresh-apply persistence:
            # - Eager mode: every assignment written as ``pending``;
            #   bin-packer guaranteed full placement so dispatch covers
            #   every row.
            # - Opportunistic mode (default): placed rows go ``pending``
            #   (about to be synchronously built); deferred rows go
            #   ``registered`` (lazy-build via ensure_present on first
            #   rollout, F2=2). Deferred rows still record their
            #   preferred-home node so image-affinity routing has a
            #   default target (F5=2).
            #
            # Force re-apply against an existing plan: purge prior
            # assignments before recording the new placement. Without
            # this, force re-applies that the FFD bin-packer happens
            # to lay out on a different (node_id, image_ref) than
            # the prior run leave stale rows behind — record_assignment
            # is an upsert keyed on (plan_id, node_id, image_ref), so
            # new rows insert cleanly but old rows on now-vacated
            # nodes are never deleted, and the admin /builds view
            # then shows total > len(plan.benchmarks * selection).
            # The residual_only branch above does NOT take this path
            # by design — partial_failure retries deliberately
            # preserve done rows.
            if existing is not None:
                self._state.delete_assignments(plan_id)
            initial_status: BuildAssignmentStatus = "pending"
            for assignment in placement.assignments:
                self._state.record_assignment(BuildAssignmentRecord(
                    plan_id=plan_id,
                    node_id=assignment.node_id,
                    image_ref=assignment.image_ref,
                    benchmark=assignment.benchmark,
                    status=initial_status,
                ))
            for d in deferred:
                self._state.record_assignment(BuildAssignmentRecord(
                    plan_id=plan_id,
                    node_id=d.preferred_home,
                    image_ref=d.image_ref,
                    benchmark=d.benchmark,
                    status="registered",
                ))
            dispatch_assignments = list(placement.assignments)
            preexisting_successes = 0

        # 7. Dispatch per-node jobs IN PARALLEL (M3). One async task
        # per node; results flow back into a shared queue + the state
        # store updater drains it. asyncio.gather waits for every node
        # to finish.
        import asyncio

        # Bucket the dispatch list by node so each task knows its
        # slice. residual_only may have dropped a node's rows entirely.
        per_node: dict[str, list[Any]] = {}
        for a in dispatch_assignments:
            per_node.setdefault(a.node_id, []).append(a)

        # Audit P1.6.g-H1 fix (2026-05-05): every deferred (registered)
        # row needs its builder mapping registered on the nodes a
        # rollout might land on so a later ``ensure_present`` on the
        # deferred ref invokes the benchmark builder instead of falling
        # through to ``backend.pull_image``.
        #
        # Audit P1.6.g-H2 fix (2026-05-05): the original H1 fix only
        # registered the mapping on the row's ``preferred_home`` node.
        # On multi-node clusters the scheduler can land the rollout
        # elsewhere (it doesn't read the build snapshot), and that
        # node would have no mapping. Broadcast the lazy_registrations
        # to every node in the budget snapshot — the cost is trivial
        # (~200 B per ref times deferred count, well under 1 MiB at
        # the 500-instance scale) and it makes the lazy path correct
        # regardless of placement. Locality is recovered by the
        # companion change in :class:`Scheduler` that scores each
        # row's preferred_home as a soft affinity bonus.
        per_node_lazy: dict[str, list[Any]] = {}
        from xrlenv.control.image_planner import (
            PlanAssignment as _PlanAssignment,
        )
        deferred_assignments = [
            _PlanAssignment(
                image_ref=d.image_ref,
                benchmark=d.benchmark,
                node_id=d.preferred_home,
                size_bytes=int(d.size_bytes),
            )
            for d in deferred
        ]
        if deferred_assignments:
            for budget in nodes:
                per_node_lazy[budget.node_id] = list(deferred_assignments)
        # Every broadcast target must receive a BuildJob even if it
        # has zero synchronous placements; otherwise the registration
        # never reaches it.
        for nid in per_node_lazy:
            per_node.setdefault(nid, [])

        successes = preexisting_successes
        failures: list[str] = []
        results_lock = asyncio.Lock()

        async def _run_one_node(
            node_id: str, assignments: list[Any],
        ) -> None:
            nonlocal successes
            job = BuildJob(
                node_id=node_id,
                assignments=tuple(assignments),
                builder_per_benchmark=per_benchmark_builder,
                build_kwargs_per_benchmark=per_benchmark_kwargs,
                force=force,
                lazy_registrations=tuple(per_node_lazy.get(node_id, ())),
            )
            for a in assignments:
                self._state.update_assignment_status(
                    plan_id=plan_id, node_id=node_id,
                    image_ref=a.image_ref, status="building",
                )
            async for result in self._node_builder.execute(job):
                # State-store updates are cheap; the lock guards the
                # nonlocal counters but most work is independent.
                async with results_lock:
                    self._state.update_assignment_status(
                        plan_id=plan_id, node_id=node_id,
                        image_ref=result.image_ref,
                        status=result.status,
                        error=result.error,
                    )
                    if result.status == "done":
                        successes += 1
                    else:
                        failures.append(
                            f"{node_id}/{result.image_ref}: "
                            f"{result.error or 'unknown'}",
                        )

        if per_node:
            await asyncio.gather(*(
                _run_one_node(nid, rows) for nid, rows in per_node.items()
            ))

        # 8. Final plan status. CAS-style flip from ``in_flight`` so
        # an operator cancel that already moved the plan to
        # ``cancelled`` doesn't get clobbered here. Without this,
        # a cancel that arrives between the dispatch loop and the
        # terminal-status update reverts to ``partial_failure`` /
        # ``completed`` on the next assignment finish.
        terminal: BuildPlanStatus = (
            "completed" if not failures else "partial_failure"
        )
        flipped = self._state.try_update_build_plan_status(
            plan_id, expected_current="in_flight", new_status=terminal,
        )
        # If the flip didn't take, the plan is in some other status
        # (most commonly ``cancelled``). Return the OUTCOME we
        # computed locally, but with the actual persisted status so
        # the CLI / admin report stays consistent with state.db.
        actual_status: BuildPlanStatus = terminal
        if not flipped:
            current = self._state.get_build_plan(plan_id)
            if current is not None:
                actual_status = current.status
                LOGGER.info(
                    "build apply for %s: skipping terminal flip to "
                    "%r — plan is already %r (operator cancelled?)",
                    plan_id, terminal, actual_status,
                )
        return BuildOutcome(
            plan_id=plan_id, status=actual_status,
            placement=placement,
            successes=successes, failures=len(failures),
            deferred=len(deferred),
            error_summary=failures[:20],  # cap so the CLI output stays readable
        )

    async def _apply_per_image_ref(
        self,
        plan: BuildPlan,
        *,
        dry_run: bool,
        force: bool,
        eager: bool,
        fill_missing: bool = False,
        bypass_in_flight_check: bool = False,
        applied_by: str,
        skip_if_present: bool = False,
        concurrency: int | None = None,
        push: bool = False,
    ) -> BuildOutcome:
        """Apply an entries-shaped plan.

        Branches per entry on ``context_source.type``:

        - ``registry`` — dispatch via ``ensure_present_fn`` (pull
          from the registry on the chosen node).
        - ``git`` — dispatch via ``build_image_fn`` (clone + build
          on the chosen node).
        - ``tarball`` — dispatch via ``build_image_fn`` (untar +
          build on the chosen node). The CLI's
          :func:`resolve_tarball_sources` populated each
          ``TarballSource.content_b64`` field at apply time so the
          dispatcher receives wire-ready bytes.

        ``eager=False`` (default — matches the legacy benchmark-driven
        path) opportunistic mode: FFD places what fits, overflow
        entries are recorded as ``status="registered"`` against their
        preferred-home node. **Registry-source** deferred entries
        require no extra wiring — the runtime's ``ensure_present``
        path on first rollout pulls from Docker Hub on demand, and
        the node-side image-cache LRU evicts cold images when disk
        pressure rises. This is the "build-then-evict on the fly"
        operator UX. **Git/tarball** deferred entries are rejected
        upfront (``ManifestInvalid``) — lazy build for non-registry
        per-image-ref sources needs per-node source-spec broadcast
        infrastructure that the legacy benchmark path has but per-
        image-ref doesn't carry yet; pass ``eager=True`` if you
        really need git/tarball entries that don't fit, or use the
        opportunistic-friendly registry shape.

        ``eager=True`` preserves the original semantics: bin-packer
        raises :class:`InsufficientCapacity` if anything doesn't fit;
        every assignment dispatches synchronously.
        """
        # 1. Source-type gate. Tarball entries that haven't been
        # operator-side-resolved (no content_b64) reject loudly here
        # — their dispatch needs the bytes inline. Plans coming
        # through ``cmd_build_apply`` are always pre-resolved; this
        # gate is defensive in case a programmatic caller skipped
        # the helper.
        registry_entries: list[BuildEntry] = []
        build_entries: list[BuildEntry] = []  # git + tarball
        unresolved_tarballs: list[str] = []
        for e in plan.entries:
            if isinstance(e.context_source, RegistrySource):
                registry_entries.append(e)
            elif isinstance(e.context_source, GitSource):
                build_entries.append(e)
            elif isinstance(e.context_source, TarballSource):
                if e.context_source.content_b64 is None:
                    unresolved_tarballs.append(e.image_ref)
                else:
                    build_entries.append(e)
        if unresolved_tarballs:
            raise ManifestInvalid(
                "build plan rejected: tarball entries reached the "
                "coordinator without their bytes resolved. The CLI's "
                "``resolve_tarball_sources`` helper must run before "
                "``coordinator.apply``. Offending entries: "
                + ", ".join(unresolved_tarballs[:10])
                + (" ..." if len(unresolved_tarballs) > 10 else ""),
            )

        # 2. Dispatchers required by the entry mix must be wired.
        if registry_entries and self._ensure_present_fn is None:
            raise ManifestInvalid(
                "per-image-ref plan with registry entries requires "
                "ensure_present_fn on BuildCoordinator; the active "
                "runtime did not wire one (LocalRuntime / "
                "DistributedRuntime needs an upgrade).",
            )
        if build_entries and not push and self._build_image_fn is None:
            raise ManifestInvalid(
                "per-image-ref plan with git/tarball-source entries "
                "requires build_image_fn on BuildCoordinator; the "
                "active runtime did not wire one (source-build "
                "dispatch is unavailable).",
            )
        if build_entries and push and self._build_push_fn is None:
            raise ManifestInvalid(
                "per-image-ref plan applied with push=True requires "
                "build_push_fn on BuildCoordinator; the active runtime did "
                "not wire one (source-build-and-push dispatch is "
                "unavailable).",
            )

        # 3. Lower entries to ImageToPlace rows. Size + replication
        # come straight from the entry; no benchmark indirection.
        # Build the image_ref → entry lookup once — both the
        # deferred-source check (step 5) and the dispatch loop
        # (step 8) reuse it.
        entry_by_ref_for_check: dict[str, BuildEntry] = {
            e.image_ref: e for e in plan.entries
        }
        images: list[ImageToPlace] = [
            ImageToPlace(
                image_ref=e.image_ref,
                # Reserve the expected ON-DISK footprint, not the plan's
                # (compressed) registry-probe hint — otherwise FFD packs every
                # image into the apparent free space, defers nothing, and the
                # build ENOSPCs. cluster-reported hints pass through unchanged.
                size_bytes=expected_on_disk_bytes(
                    e.placement.size_hint_bytes, e.placement.size_hint_source,
                ),
                replication=e.placement.preferred_home_count,
                benchmark=PER_IMAGE_REF_BENCHMARK_TAG,
            )
            for e in plan.entries
        ]

        # 4. Per-node budget snapshot.
        nodes = await self._budget_provider.get_budgets(
            reserved_runtime_gb=plan.budget.reserved_runtime_gb,
            buffer_gb=plan.budget.buffer_gb,
            cap_per_node_gb=plan.budget.cap_per_node_gb,
        )

        # 4.5. Pin-budget enforcement (F4 — hard reject at apply time).
        # Per-node conservative bound: every entry with ``pinned: true``
        # counts toward every node's projected pinned total because
        # FFD hasn't run yet and any pinned entry could land on any
        # node. If the projected pinned bytes exceed a node's
        # available bytes, the cluster has no room for non-pinned
        # rollouts on that node — silent over-pinning bites weeks
        # later when unrelated work hits the threshold, so we reject
        # upfront instead. The check is intentionally conservative:
        # if it passes, FFD can definitely find placement; if it
        # fails, FFD MIGHT still find one but the plan is too risky
        # for the operator to ship without explicitly relaxing the
        # budget or unpinning entries.
        pinned_total_bytes = sum(
            e.placement.size_hint_bytes for e in plan.entries if e.pinned
        )
        if pinned_total_bytes > 0 and nodes:
            offending: list[tuple[str, int, int]] = []
            for node in nodes:
                if pinned_total_bytes > node.available_bytes:
                    offending.append((
                        node.node_id,
                        pinned_total_bytes,
                        node.available_bytes,
                    ))
            if offending:
                lines = [
                    f"node {nid}: pinned total {pinned / 1024**3:.1f} GB "
                    f"> available {avail / 1024**3:.1f} GB "
                    f"(over by {(pinned - avail) / 1024**3:.1f} GB)"
                    for nid, pinned, avail in offending
                ]
                raise ManifestInvalid(
                    "build plan rejected: pin-budget over-commit. The "
                    "sum of size_hint_bytes for entries with "
                    "``pinned: true`` exceeds at least one node's "
                    "available image-cache budget; FFD might land "
                    "every pinned entry on the same node, leaving no "
                    "room for non-pinned rollouts. Either drop "
                    "``pinned`` on lower-priority entries, raise the "
                    "budget (``budget.reserved_runtime_gb`` or "
                    "``cap_per_node_gb``), or shrink the entries' "
                    "size_hint_bytes (recalibrate after a real "
                    "cluster build via ``xrlenv build calibrate``).\n"
                    + "\n".join(f"  - {line}" for line in lines),
                )

        # 4.6. ``--fill-missing`` short-circuit. Bypass FFD entirely:
        # ask the inventory provider which entries are absent from the
        # connected fleet, place ONLY those, and re-anchor the
        # already-present entries' assignment rows to whichever node
        # actually has them. The operator's intent is "make the cluster
        # match the plan; don't redo what already matches" — see
        # ``api_build_apply`` docs for the operator-facing rationale.
        # Works regardless of the plan's existing status (completed /
        # partial_failure / cancelled / superseded).
        if push and fill_missing:
            raise ManifestInvalid(
                "build plan rejected: push mode does not combine with "
                "--fill-missing. The push path already registry-HEAD-skips "
                "already-pushed refs (build-once fleet-wide, resumable), so "
                "fill_missing's node-inventory optimization is redundant. "
                "Apply with push and WITHOUT --fill-missing.",
            )
        if fill_missing:
            if self._inventory_provider is None:
                raise ManifestInvalid(
                    "build plan rejected: --fill-missing requires a "
                    "ClusterInventoryProvider on the BuildCoordinator. "
                    "DistributedRuntime wires one; LocalRuntime currently "
                    "doesn't (single-host plans don't need it). Apply "
                    "without --fill-missing, OR run against a cluster "
                    "with --connect-host.",
                )
            return await self._apply_per_image_ref_fill_missing(
                plan, dry_run=dry_run, applied_by=applied_by,
                nodes=nodes, images=images,
                entry_by_ref=entry_by_ref_for_check,
                skip_if_present=skip_if_present,
                concurrency=concurrency,
            )

        # 5. Place. Two modes:
        #
        # - eager=True: legacy P1.6.b/c semantics. Bin-packer raises
        #   InsufficientCapacity on overflow.
        # - eager=False (default): opportunistic FFD places what fits;
        #   overflow rows go to ``deferred`` and lazy-build at runtime
        #   via the existing acquire-time ``ensure_present`` path
        #   (works out-of-the-box for registry-source entries; non-
        #   registry deferred entries reject below).
        deferred: tuple[Any, ...] = ()
        if push:
            # Build-push: images are transient (pushed to the registry, then
            # evictable), so shard the whole plan size-balanced across every
            # connected node WITHOUT the FFD fit constraint — a plan far larger
            # than total cluster disk still fans out. No deferred entries;
            # dispatch-time disk-pacing + the image-cache LRU bound real disk.
            if not nodes:
                raise ManifestInvalid(
                    "build push rejected: no nodes are connected to build and "
                    "push on. Start the fleet's node agents first.",
                )
            placement = _shard_for_push(images, nodes)
        elif eager:
            placement = plan_placements(images, nodes)
        else:
            opp = plan_opportunistic_placements(images, nodes)
            placement = opp.placed
            deferred = opp.deferred
            # Non-registry sources need per-node source-spec broadcast
            # before lazy-build at acquire time works (the legacy
            # benchmark path's ``lazy_registrations`` mechanism).
            # The per-image-ref dispatcher doesn't carry that wiring
            # yet, so reject git/tarball deferred entries with a clear
            # remediation path: shrink the plan, add nodes, or
            # ``--eager`` to surface as InsufficientCapacity.
            non_registry_deferred = [
                d for d in deferred
                if not isinstance(
                    entry_by_ref_for_check[d.image_ref].context_source,
                    RegistrySource,
                )
            ]
            if non_registry_deferred:
                lines = [
                    f"  - {d.image_ref}"
                    for d in non_registry_deferred[:10]
                ]
                more = (
                    f"  ... and {len(non_registry_deferred) - 10} more"
                    if len(non_registry_deferred) > 10 else ""
                )
                raise ManifestInvalid(
                    "build plan rejected: opportunistic per-image-ref "
                    "apply doesn't support deferred git/tarball entries "
                    "yet — lazy-build for non-registry sources needs "
                    "per-node source-spec broadcast that the per-image-"
                    "ref dispatcher doesn't carry today. The following "
                    "git/tarball entries didn't fit the cluster budget:"
                    "\n" + "\n".join(lines) + ("\n" + more if more else "")
                    + "\n\nOptions: (1) shrink the plan / connect more "
                    "nodes so every git/tarball entry fits; "
                    "(2) re-apply with ``--eager`` to surface the same "
                    "rejection as InsufficientCapacity; (3) keep the "
                    "non-registry entries pinned/required and move the "
                    "rest to a separate plan.",
                )

        # 6. Compute plan_id; honor in_flight + completed-no-op idempotency.
        plan_id = compute_plan_id(plan)
        plan_json = plan.model_dump_json(exclude_none=True)

        if dry_run:
            return BuildOutcome(
                plan_id=plan_id, status="dry_run",
                placement=placement, deferred=len(deferred),
            )

        existing = self._state.get_build_plan(plan_id)
        if existing is not None:
            if existing.status == "in_flight" and not bypass_in_flight_check:
                return BuildOutcome(
                    plan_id=plan_id, status="rejected_in_flight",
                    placement=placement,
                )
            if existing.status == "completed" and not force:
                return BuildOutcome(
                    plan_id=plan_id, status="no_op_already_completed",
                    placement=placement,
                )

        # 7. Persist plan + assignment rows.
        self._state.record_build_plan(
            plan_id=plan_id, applied_by=applied_by, plan_json=plan_json,
            name=plan.name,
        )
        if existing is not None:
            self._state.update_build_plan_status(plan_id, "in_flight")
            # Purge prior assignments before recording the new
            # placement. Without this, force re-applies that the FFD
            # bin-packer happens to lay out on a different
            # (node_id, image_ref) than the prior run leave stale
            # rows in build_plan_assignments — record_assignment is
            # an upsert keyed on (plan_id, node_id, image_ref), so
            # the new rows insert cleanly but the old rows on
            # now-vacated nodes are never deleted. The admin /builds
            # view then shows total > len(plan.entries) (e.g. 13/13
            # for an 8-entry canonical plan that's been re-applied
            # 3-4 times across topology changes).
            self._state.delete_assignments(plan_id)

        for assignment in placement.assignments:
            self._state.record_assignment(BuildAssignmentRecord(
                plan_id=plan_id,
                node_id=assignment.node_id,
                image_ref=assignment.image_ref,
                benchmark=PER_IMAGE_REF_BENCHMARK_TAG,
                status="pending",
            ))
        # Deferred (overflow) rows persist as ``registered``. The
        # runtime acquire path's ``ensure_present`` pulls the image on
        # demand when a rollout lands on a node that doesn't have it
        # cached; the node's LRU evictor reclaims disk for the new
        # pull if the cache is tight. ``preferred_home`` is recorded
        # so the scheduler's image-affinity bonus routes rollouts to
        # the most-free node FFD picked as the fallback target — but
        # this is a soft hint, not a hard placement lock.
        for d in deferred:
            self._state.record_assignment(BuildAssignmentRecord(
                plan_id=plan_id,
                node_id=d.preferred_home,
                image_ref=d.image_ref,
                benchmark=PER_IMAGE_REF_BENCHMARK_TAG,
                status="registered",
            ))

        # 8. Dispatch per-(node, image) calls in parallel, bounded by a
        # concurrency semaphore so we don't fire hundreds of docker
        # pulls/builds simultaneously and overwhelm Docker daemons +
        # registry rate limits.
        import asyncio

        entry_by_ref = entry_by_ref_for_check

        successes = 0
        failures: list[str] = []
        digests: dict[str, str] = {}  # image_ref -> pushed digest (push mode only)
        results_lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency or DEFAULT_BUILD_CONCURRENCY)
        watermark_bytes = self._dispatch_watermark_bytes(
            placement.assignments, entry_by_ref,
        )

        async def _dispatch_one(node_id: str, image_ref: str) -> None:
            nonlocal successes
            async with sem:
                # Disk-aware pacing: wait for the node to have room before
                # adding another image, so the build can't drive it below its
                # eviction reserve (no-op when the gate is unwired).
                await self._await_disk_headroom(node_id, watermark_bytes)
                self._state.update_assignment_status(
                    plan_id=plan_id, node_id=node_id,
                    image_ref=image_ref, status="building",
                )
                entry = entry_by_ref[image_ref]
                repo_digest: str | None = None
                try:
                    if isinstance(entry.context_source, RegistrySource):
                        assert self._ensure_present_fn is not None
                        status, error = await self._ensure_present_fn(
                            node_id, image_ref,
                            DEFAULT_ENSURE_PRESENT_TIMEOUT_S,
                        )
                    elif isinstance(
                        entry.context_source, (GitSource, TarballSource),
                    ):
                        if push:
                            # Build AND push to the registry image_ref encodes,
                            # capturing the pushed digest for the pin plan.
                            assert self._build_push_fn is not None
                            status, error, repo_digest = (
                                await self._build_push_fn(
                                    node_id, image_ref, entry.context_source,
                                    DEFAULT_BUILD_IMAGE_TIMEOUT_S,
                                    dict(entry.labels),
                                )
                            )
                        else:
                            assert self._build_image_fn is not None
                            status, error = await self._build_image_fn(
                                node_id, image_ref, entry.context_source,
                                DEFAULT_BUILD_IMAGE_TIMEOUT_S,
                                dict(entry.labels),
                                skip_if_present,
                            )
                    else:
                        status = "failed"
                        error = (
                            f"unsupported context_source type: "
                            f"{type(entry.context_source).__name__}"
                        )
                except Exception as exc:
                    LOGGER.exception(
                        "per-image-ref dispatch raised for node=%s image=%s",
                        node_id, image_ref,
                    )
                    status, error = "failed", (
                        f"dispatch raised: {type(exc).__name__}: {exc}"
                    )
                # A wire timeout ("timeout") means the control plane gave up
                # waiting for the reply, but the node may still be completing
                # the pull — record it as ``registered`` (lazy, verified /
                # pulled on first acquire), NOT ``failed``. Otherwise a
                # timed-out-but-pulled image is mislabelled failed and the
                # failure count overstates real failures (the 2026-06 incident).
                terminal: BuildAssignmentStatus
                if status == "ok":
                    terminal = "done"
                elif status == "timeout":
                    terminal = "registered"
                else:
                    terminal = "failed"
                async with results_lock:
                    self._state.update_assignment_status(
                        plan_id=plan_id, node_id=node_id,
                        image_ref=image_ref, status=terminal,
                        error=error if terminal == "failed" else None,
                    )
                    if terminal == "done":
                        successes += 1
                        if repo_digest:
                            digests[image_ref] = repo_digest
                    elif terminal == "failed":
                        failures.append(
                            f"{node_id}/{image_ref}: {error or 'unknown'}",
                        )
                    # ``registered`` (wire timeout) is neither — it surfaces as
                    # deferred=N in the poll and resolves on acquire.

        await asyncio.gather(*(
            _dispatch_one(a.node_id, a.image_ref)
            for a in placement.assignments
        ))

        terminal_plan: BuildPlanStatus = (
            "completed" if not failures else "partial_failure"
        )
        # CAS-style flip — see the matching block in the legacy
        # benchmark-driven apply() above for the rationale. An
        # operator cancel that moves the plan to ``cancelled`` while
        # this dispatch loop is mid-flight must NOT be overwritten
        # by our terminal-status update.
        flipped = self._state.try_update_build_plan_status(
            plan_id, expected_current="in_flight", new_status=terminal_plan,
        )
        actual_plan_status: BuildPlanStatus = terminal_plan
        if not flipped:
            current = self._state.get_build_plan(plan_id)
            if current is not None:
                actual_plan_status = current.status
                LOGGER.info(
                    "per-image-ref apply for %s: skipping terminal "
                    "flip to %r — plan is already %r (operator "
                    "cancelled?)",
                    plan_id, terminal_plan, actual_plan_status,
                )
        return BuildOutcome(
            plan_id=plan_id, status=actual_plan_status,
            placement=placement,
            successes=successes, failures=len(failures),
            deferred=len(deferred),
            error_summary=failures[:20],
            digests=digests,
        )

    async def _apply_per_image_ref_fill_missing(
        self,
        plan: BuildPlan,
        *,
        dry_run: bool,
        applied_by: str,
        nodes: list[NodeBudget],
        images: list[ImageToPlace],
        entry_by_ref: dict[str, BuildEntry],
        skip_if_present: bool,
        concurrency: int | None = None,
    ) -> BuildOutcome:
        """``--fill-missing`` apply: bring the cluster into the plan's
        intended state without re-doing work that's already correct.

        Steps:

        1. Pull the cluster's current image inventory via
           :attr:`_inventory_provider`. Yields
           ``{image_ref: set[node_id_where_present]}``.

        2. Partition ``plan.entries`` into "already-present somewhere"
           and "absent from every node." Present entries get an
           assignment row anchored at one of the nodes that has them
           (the first one alphabetically, for determinism), with
           ``status=done``. Stale rows pointing at other nodes for the
           same image_ref are purged via ``delete_assignments`` then
           re-recorded — the operator's intent is "rows reflect
           reality."

        3. Missing entries get FFD-placed against current free disk
           (opportunistic: places what fits, the rest become
           ``status=registered`` and lazy-pull at acquire time). Those
           placed rows get dispatched via :attr:`_ensure_present_fn`
           (registry) or :attr:`_build_image_fn` (git/tarball).

        4. Plan flips to ``completed`` once dispatch finishes (with
           the same "deferred rows don't block completion" semantics
           as the standard opportunistic path); ``partial_failure``
           if any newly-dispatched pull failed.
        """
        plan_id = compute_plan_id(plan)
        plan_json = plan.model_dump_json(exclude_none=True)

        # 1. Inventory snapshot.
        assert self._inventory_provider is not None
        inventory = await self._inventory_provider.get_inventory()

        # 2. Partition.
        present_pairs: list[tuple[str, str]] = []   # (image_ref, node_id)
        missing_entries: list[BuildEntry] = []
        for entry in plan.entries:
            cached_on = inventory.get(entry.image_ref) or set()
            if cached_on:
                # Pick deterministically — alphabetical is stable
                # across re-runs so re-anchoring doesn't churn rows
                # between equivalent nodes.
                chosen = sorted(cached_on)[0]
                present_pairs.append((entry.image_ref, chosen))
            else:
                missing_entries.append(entry)

        # 3. FFD-place ONLY the missing entries (opportunistic).
        missing_images = [
            img for img in images
            if img.image_ref in {e.image_ref for e in missing_entries}
        ]
        deferred: tuple[Any, ...] = ()
        placement = PlacementResult(assignments=())
        if missing_images and nodes:
            opp = plan_opportunistic_placements(missing_images, nodes)
            placement = opp.placed
            deferred = opp.deferred
            non_registry_deferred = [
                d for d in deferred
                if not isinstance(
                    entry_by_ref[d.image_ref].context_source,
                    RegistrySource,
                )
            ]
            if non_registry_deferred:
                raise ManifestInvalid(
                    "build plan rejected: --fill-missing computed "
                    "git/tarball entries that don't fit current "
                    "cluster disk; lazy build for non-registry sources "
                    "isn't wired for per-image-ref. Either shrink the "
                    "plan / connect more nodes so they fit, OR drop "
                    "--fill-missing and retry with --force (which "
                    "purges + re-dispatches everything).",
                )

        if dry_run:
            return BuildOutcome(
                plan_id=plan_id, status="dry_run",
                placement=placement, deferred=len(deferred),
            )

        # 4. Persist plan + assignment rows. Delete-and-replace: the
        # operator's intent is "rows reflect reality"; stale rows
        # pointing at the wrong node get cleared so the image-affinity
        # scheduler doesn't keep routing rollouts there.
        existing = self._state.get_build_plan(plan_id)
        self._state.record_build_plan(
            plan_id=plan_id, applied_by=applied_by, plan_json=plan_json,
            name=plan.name,
        )
        self._state.update_build_plan_status(plan_id, "in_flight")
        self._state.delete_assignments(plan_id)
        # Re-anchored "already present" entries: one row per entry,
        # status=done, pointing at the node that actually has it.
        for image_ref, node_id in present_pairs:
            self._state.record_assignment(BuildAssignmentRecord(
                plan_id=plan_id,
                node_id=node_id,
                image_ref=image_ref,
                benchmark=PER_IMAGE_REF_BENCHMARK_TAG,
                status="done",
            ))
        # Missing-and-placed entries: pending → building → done|failed
        # via the dispatch loop below.
        for assignment in placement.assignments:
            self._state.record_assignment(BuildAssignmentRecord(
                plan_id=plan_id,
                node_id=assignment.node_id,
                image_ref=assignment.image_ref,
                benchmark=PER_IMAGE_REF_BENCHMARK_TAG,
                status="pending",
            ))
        # Missing-and-deferred entries (didn't fit the budget):
        # status=registered, lazy-pulled at acquire time. Only
        # registry-source entries reach here (non-registry deferred
        # rejects above).
        for d in deferred:
            self._state.record_assignment(BuildAssignmentRecord(
                plan_id=plan_id,
                node_id=d.preferred_home,
                image_ref=d.image_ref,
                benchmark=PER_IMAGE_REF_BENCHMARK_TAG,
                status="registered",
            ))

        # 5. Dispatch only the newly-placed (pending) rows. Re-anchored
        # done rows need no work — the image is already there.
        import asyncio

        successes = len(present_pairs)
        failures: list[str] = []
        results_lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency or DEFAULT_BUILD_CONCURRENCY)
        watermark_bytes = self._dispatch_watermark_bytes(
            placement.assignments, entry_by_ref,
        )

        async def _dispatch_one(node_id: str, image_ref: str) -> None:
            nonlocal successes
            async with sem:
                # Disk-aware pacing: wait for the node to have room before
                # adding another image, so the build can't drive it below its
                # eviction reserve (no-op when the gate is unwired).
                await self._await_disk_headroom(node_id, watermark_bytes)
                self._state.update_assignment_status(
                    plan_id=plan_id, node_id=node_id,
                    image_ref=image_ref, status="building",
                )
                entry = entry_by_ref[image_ref]
                try:
                    if isinstance(entry.context_source, RegistrySource):
                        assert self._ensure_present_fn is not None
                        status, error = await self._ensure_present_fn(
                            node_id, image_ref,
                            DEFAULT_ENSURE_PRESENT_TIMEOUT_S,
                        )
                    elif isinstance(
                        entry.context_source, (GitSource, TarballSource),
                    ):
                        assert self._build_image_fn is not None
                        status, error = await self._build_image_fn(
                            node_id, image_ref, entry.context_source,
                            DEFAULT_BUILD_IMAGE_TIMEOUT_S,
                            dict(entry.labels),
                            skip_if_present,
                        )
                    else:
                        status = "failed"
                        error = (
                            f"unsupported context_source: "
                            f"{type(entry.context_source).__name__}"
                        )
                except Exception as exc:
                    LOGGER.exception(
                        "fill-missing dispatch raised for node=%s image=%s",
                        node_id, image_ref,
                    )
                    status, error = "failed", (
                        f"dispatch raised: {type(exc).__name__}: {exc}"
                    )
                # A wire timeout ("timeout") means the control plane gave up
                # waiting for the reply, but the node may still be completing
                # the pull — record it as ``registered`` (lazy, verified /
                # pulled on first acquire), NOT ``failed``. Otherwise a
                # timed-out-but-pulled image is mislabelled failed and the
                # failure count overstates real failures (the 2026-06 incident).
                terminal: BuildAssignmentStatus
                if status == "ok":
                    terminal = "done"
                elif status == "timeout":
                    terminal = "registered"
                else:
                    terminal = "failed"
                async with results_lock:
                    self._state.update_assignment_status(
                        plan_id=plan_id, node_id=node_id,
                        image_ref=image_ref, status=terminal,
                        error=error if terminal == "failed" else None,
                    )
                    if terminal == "done":
                        successes += 1
                    elif terminal == "failed":
                        failures.append(
                            f"{node_id}/{image_ref}: {error or 'unknown'}",
                        )
                    # ``registered`` (wire timeout) is neither — it surfaces as
                    # deferred=N in the poll and resolves on acquire.

        if placement.assignments:
            await asyncio.gather(*(
                _dispatch_one(a.node_id, a.image_ref)
                for a in placement.assignments
            ))

        terminal_plan: BuildPlanStatus = (
            "completed" if not failures else "partial_failure"
        )
        # CAS-style status flip — see the legacy benchmark-driven path
        # for the rationale (operator cancel must not be overwritten).
        actual_plan_status = terminal_plan
        flipped = self._state.try_update_build_plan_status(
            plan_id, expected_current="in_flight",
            new_status=terminal_plan,
        )
        if not flipped:
            current = self._state.get_build_plan(plan_id)
            if current is not None:
                actual_plan_status = current.status
                LOGGER.info(
                    "fill-missing apply for %s: skipping terminal "
                    "flip to %r — plan is already %r",
                    plan_id, terminal_plan, actual_plan_status,
                )

        # ``existing`` referenced to silence the linter; it's also
        # useful diagnostically for future audit branches.
        _ = existing
        return BuildOutcome(
            plan_id=plan_id, status=actual_plan_status,
            placement=placement,
            successes=successes, failures=len(failures),
            deferred=len(deferred),
            error_summary=failures[:20],
        )

    def _lookup_manifest(self, name: str) -> TemplateManifest:
        try:
            return self._catalog.get(name)
        except KeyError as exc:
            raise ManifestInvalid(
                f"benchmark {name!r} is not registered in the template "
                f"catalog; install the plug-in or add its manifest to "
                f"XRLENV_TEMPLATE_DIRS",
            ) from exc

    @staticmethod
    def _read_size_hint(decl: object) -> int:
        """Resolve the builder class's static ``IMAGE_SIZE_HINT_BYTES``
        without instantiating it (pure metadata lookup)."""
        # Import the module + read the class attribute. Cheaper than
        # constructing the builder; the builder construction may need
        # the harbor cache or other side-state we don't have here.
        import importlib

        module_name = getattr(decl, "module", None)
        class_name = getattr(decl, "class_name", None)
        if not isinstance(module_name, str) or not isinstance(class_name, str):
            raise ManifestInvalid(
                "image_builder decl must carry string module + class_name",
            )
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ManifestInvalid(
                f"image_builder {module_name}.{class_name} cannot be loaded: {exc}",
            ) from exc
        hint = getattr(cls, "IMAGE_SIZE_HINT_BYTES", None)
        if not isinstance(hint, int) or hint <= 0:
            raise ManifestInvalid(
                f"image_builder {module_name}.{class_name}.IMAGE_SIZE_HINT_BYTES "
                f"must be a positive int (the bin-packer needs it pre-build)",
            )
        return hint


__all__ = [
    "BuildCoordinator",
    "BuildOutcome",
    "NodeBudgetProvider",
]
