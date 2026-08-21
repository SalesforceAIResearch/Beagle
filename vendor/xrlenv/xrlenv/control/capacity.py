"""Static capacity estimator (spec 10).

Computes per-(node, template) ``max_concurrent`` plus the binding constraint
so the scheduler can do capacity-aware placement and the admin panel can
explain why a given cell is small.

Multi-pool disk accounting (spec 10 §"Disk is multi-pool", Slice 6): the
node's disk is split into two pools — ``sandbox_writable`` (per-sandbox
COW + scratch) and ``image_cache`` (the docker image layer store managed
by :class:`xrlenv.node.image_cache.ImageCacheManager`). The capacity
estimator only counts sandbox creation against ``sandbox_writable`` so
a hot pin set in ``image_cache`` doesn't starve sandbox placement.

Online refinement (EMA-of-p95 effective request) lands when the node-agent
ships per-template usage telemetry; the API slot is reserved here so callers
can wire it without changing the surface later.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from xrlenv.backends.base import ResourceSpec, ResourceUsage
from xrlenv.control.defaults import DEFAULT_BACKEND
from xrlenv.control.template_catalog import TemplateManifest
from xrlenv.node.hw_probe import HardwareInfo

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Backend overhead (spec 10) — static defaults; phase-1 tunes from telemetry.
# ──────────────────────────────────────────────────────────────────────────────


BindingConstraint = Literal[
    "cpu",
    "mem",
    "disk:sandbox_writable",
    "gpu",
    "backend_missing",
]


class BackendOverhead(BaseModel):
    """Per-runtime constants the estimator subtracts from each placement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_per_sandbox: float
    mem_bytes_per_sandbox: int


DEFAULT_BACKEND_OVERHEAD: dict[str, BackendOverhead] = {
    "docker": BackendOverhead(cpu_per_sandbox=0.05, mem_bytes_per_sandbox=50 * 1024 * 1024),
    "cubesandbox": BackendOverhead(cpu_per_sandbox=0.10, mem_bytes_per_sandbox=128 * 1024 * 1024),
    "local-process-debug": BackendOverhead(cpu_per_sandbox=0.0, mem_bytes_per_sandbox=0),
}


class HeadroomConfig(BaseModel):
    """Reserved fraction of each axis kept free for the OS + node agent.

    ``disk_reserved_bytes`` is the OS-level reserve (kept above whichever
    pool the disk lives on). ``image_cache_pool_fraction`` carves out
    the share of the node's disk that belongs to the image cache pool;
    the remainder is the sandbox-writable pool the capacity estimator
    counts against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_fraction: float = 0.10
    mem_fraction: float = 0.15
    disk_reserved_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB
    image_cache_pool_fraction: float = 0.5
    """Fraction of the node's usable disk reserved for cached images.

    Default 50% mirrors a typical SWE-bench / OSWorld profile where image
    layers and per-sandbox writable disk consume comparable budgets. The
    image-cache manager evicts within its own pool; the capacity estimator
    sizes ``disk:sandbox_writable`` against the remaining fraction so a
    pin-heavy operator doesn't accidentally starve sandbox creation.
    Operators tune this when their templates skew toward one side.
    """


class NodeProfile(BaseModel):
    """The estimator's view of a node — hardware + which backends it advertises."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    hardware: HardwareInfo
    backends: tuple[str, ...]


class CapacityCell(BaseModel):
    """One cell in the (node, template) capacity matrix (spec 10 / spec 13)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    template: str
    max_concurrent: int
    binding_constraint: BindingConstraint
    cpu_cap: int
    mem_cap: int
    disk_cap: int


# ──────────────────────────────────────────────────────────────────────────────
# Estimator interface
# ──────────────────────────────────────────────────────────────────────────────


class CapacityEstimator(Protocol):
    """Protocol consumed by the scheduler + admin panel."""

    def capacity(
        self,
        node: NodeProfile,
        manifest: TemplateManifest,
        *,
        backend: str = DEFAULT_BACKEND,
    ) -> CapacityCell: ...

    def fits(
        self,
        node: NodeProfile,
        running: list[tuple[str, ResourceSpec]],
        candidate: TemplateManifest,
        *,
        task_key: str | None = None,
        task_count_on_node: int = 0,
        max_runs_per_task: int = 4,
        backend: str = DEFAULT_BACKEND,
    ) -> bool: ...

    def matrix(
        self,
        nodes: list[NodeProfile],
        manifests: list[TemplateManifest],
        *,
        backend: str = DEFAULT_BACKEND,
    ) -> list[CapacityCell]: ...

    def slack_after_placement(
        self,
        node: NodeProfile,
        candidate: TemplateManifest,
        running: list[tuple[str, ResourceSpec]],
        *,
        backend: str = DEFAULT_BACKEND,
    ) -> float:
        """A7 / D13 (P1.1) — return the minimum-axis remaining slack
        fraction (0.0 - 1.0) after hypothetically placing ``candidate``
        on ``node`` already carrying ``running`` sandboxes. Used by
        ``Scheduler.place`` as the load-vector placement score.

        Higher = more headroom across all relevant axes; the
        scheduler picks the node with the highest score so the cluster
        stays balanced across CPU / mem / disk rather than just
        slots-of-this-template-remaining (the pre-D13 metric).

        Axes the candidate opts out of (request <= 0) are excluded
        from the min so they don't artificially dominate.
        """
        ...

    def report_usage(
        self, node_id: str, template_id: str, peak: ResourceUsage
    ) -> None: ...


# ──────────────────────────────────────────────────────────────────────────────
# Static implementation
# ──────────────────────────────────────────────────────────────────────────────


class StaticCapacityEstimator:
    """Computes capacity from declared template profiles + node hardware.

    Independent caps are taken along three axes (cpu, mem, sandbox-writable
    disk) and the minimum wins. The binding label is reported so an operator
    can immediately see *why* a node is under-utilized.
    """

    def __init__(
        self,
        *,
        headroom: HeadroomConfig | None = None,
        backend_overhead: dict[str, BackendOverhead] | None = None,
    ) -> None:
        self._headroom = headroom or HeadroomConfig()
        self._overhead = backend_overhead or DEFAULT_BACKEND_OVERHEAD

    # ── Capacity for an empty node ───────────────────────────────────────────

    def capacity(
        self,
        node: NodeProfile,
        manifest: TemplateManifest,
        *,
        backend: str = DEFAULT_BACKEND,
    ) -> CapacityCell:
        if backend not in node.backends:
            return CapacityCell(
                node_id=node.node_id,
                template=manifest.name,
                max_concurrent=0,
                binding_constraint="backend_missing",
                cpu_cap=0,
                mem_cap=0,
                disk_cap=0,
            )
        if manifest.resources.gpu_required and not node.hardware.has_gpu:
            return CapacityCell(
                node_id=node.node_id,
                template=manifest.name,
                max_concurrent=0,
                binding_constraint="gpu",
                cpu_cap=0,
                mem_cap=0,
                disk_cap=0,
            )

        overhead = self._overhead.get(
            backend,
            BackendOverhead(cpu_per_sandbox=0.0, mem_bytes_per_sandbox=0),
        )
        cpu_cap = self._cpu_cap(node, manifest, overhead, running=[])
        mem_cap = self._mem_cap(node, manifest, overhead, running=[])
        disk_cap = self._disk_cap(node, manifest)
        max_concurrent = max(0, min(cpu_cap, mem_cap, disk_cap))
        binding = _binding_label(cpu_cap, mem_cap, disk_cap)

        return CapacityCell(
            node_id=node.node_id,
            template=manifest.name,
            max_concurrent=max_concurrent,
            binding_constraint=binding,
            cpu_cap=cpu_cap,
            mem_cap=mem_cap,
            disk_cap=disk_cap,
        )

    # ── "Does one more fit?" used by the scheduler ───────────────────────────

    def fits(
        self,
        node: NodeProfile,
        running: list[tuple[str, ResourceSpec]],
        candidate: TemplateManifest,
        *,
        task_key: str | None = None,
        task_count_on_node: int = 0,
        max_runs_per_task: int = 4,
        backend: str = DEFAULT_BACKEND,
    ) -> bool:
        # Fast-path the capability filter so we don't do arithmetic for nothing.
        if backend not in node.backends:
            return False
        if candidate.resources.gpu_required and not node.hardware.has_gpu:
            return False

        # Per-task fairness cap (spec 02 — group anti-affinity / GRPO).
        if task_key is not None and task_count_on_node >= max_runs_per_task:
            return False

        overhead = self._overhead.get(
            backend,
            BackendOverhead(cpu_per_sandbox=0.0, mem_bytes_per_sandbox=0),
        )
        # Subtract resources currently committed by other rollouts on
        # this node. ``running`` carries each sandbox's effective
        # resources (post-resolver overlay for Pattern A) so a heavy
        # per-task instance gets correctly counted.
        cpu_cap = self._cpu_cap(node, candidate, overhead, running)
        mem_cap = self._mem_cap(node, candidate, overhead, running)
        disk_cap = self._disk_cap_remaining(node, candidate, running)
        return min(cpu_cap, mem_cap, disk_cap) >= 1

    def matrix(
        self,
        nodes: list[NodeProfile],
        manifests: list[TemplateManifest],
        *,
        backend: str = DEFAULT_BACKEND,
    ) -> list[CapacityCell]:
        return [self.capacity(n, m, backend=backend) for n in nodes for m in manifests]

    def slack_after_placement(
        self,
        node: NodeProfile,
        candidate: TemplateManifest,
        running: list[tuple[str, ResourceSpec]],
        *,
        backend: str = DEFAULT_BACKEND,
    ) -> float:
        """A7 / D13 (P1.1) — load-vector placement score.

        Computes the post-placement slack fraction on each axis
        (CPU / mem / sandbox-writable disk) and returns the minimum.
        Equivalent to "what's the most-saturated axis after I place
        this candidate, expressed as 1 - utilization?" — higher is
        better, the scheduler picks the maximum.

        Axes the candidate opts out of (request <= 0) are excluded
        from the min so a CPU-only or mem-only template doesn't
        artificially see disk slack as a dominating factor.

        Returns ``0.0`` when the candidate doesn't fit on any
        non-opted-out axis (i.e. would push utilization above the
        headroom-discounted cap); the caller's ``fits`` check should
        already have filtered that case out, but the floor here means
        the return is always in ``[0.0, 1.0]``.
        """
        if backend not in node.backends:
            return 0.0
        overhead = self._overhead.get(
            backend,
            BackendOverhead(cpu_per_sandbox=0.0, mem_bytes_per_sandbox=0),
        )
        slacks: list[float] = []

        # CPU axis
        if candidate.resources.cpu_request > 0:
            cpu_total = node.hardware.vcpus * (1 - self._headroom.cpu_fraction)
            running_cpu = self._running_cpu_load(running, overhead)
            cpu_after = (
                running_cpu
                + candidate.resources.cpu_request
                + overhead.cpu_per_sandbox
            )
            if cpu_total > 0:
                slacks.append(max(0.0, (cpu_total - cpu_after) / cpu_total))

        # Memory axis
        if candidate.resources.mem_request_bytes > 0:
            mem_total = node.hardware.mem_bytes * (1 - self._headroom.mem_fraction)
            running_mem = self._running_mem_load(running, overhead)
            mem_after = (
                running_mem
                + candidate.resources.mem_request_bytes
                + overhead.mem_bytes_per_sandbox
            )
            if mem_total > 0:
                slacks.append(max(0.0, (mem_total - mem_after) / mem_total))

        # Sandbox-writable disk axis (spec-10 multi-pool)
        if candidate.resources.disk_request_bytes > 0:
            disk_total = self._sandbox_writable_pool_bytes(node)
            running_disk = sum(r.disk_request_bytes for _, r in running)
            disk_after = running_disk + candidate.resources.disk_request_bytes
            if disk_total > 0:
                slacks.append(max(0.0, (disk_total - disk_after) / disk_total))

        # If every axis was opted out (rare but valid — pure-Python
        # adapters that don't request anything), there's no signal to
        # rank by; return 1.0 so node_id tiebreak decides.
        if not slacks:
            return 1.0
        return min(slacks)

    def report_usage(
        self, node_id: str, template_id: str, peak: ResourceUsage
    ) -> None:
        # Online refinement lands when the node-agent ships per-template usage
        # telemetry (spec 10 §"Phase 0 — online refinement"). The slot is here
        # so the scheduler / runtime wiring doesn't have to change later.
        LOGGER.debug(
            "capacity.report_usage node=%s template=%s peak=%r (online refinement TBD)",
            node_id,
            template_id,
            peak,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _cpu_cap(
        self,
        node: NodeProfile,
        candidate: TemplateManifest,
        overhead: BackendOverhead,
        running: list[tuple[str, ResourceSpec]],
    ) -> int:
        # Templates that declare cpu_request <= 0 are opting out of cpu
        # accounting (e.g. trust-mode pure-python adapters); treat as
        # effectively unbounded so they don't dominate disk/mem.
        if candidate.resources.cpu_request <= 0:
            return 1_000_000
        per_sandbox = candidate.resources.cpu_request + overhead.cpu_per_sandbox
        running_load = self._running_cpu_load(running, overhead)
        usable = node.hardware.vcpus * (1 - self._headroom.cpu_fraction) - running_load
        return max(0, math.floor(usable / per_sandbox))

    def _mem_cap(
        self,
        node: NodeProfile,
        candidate: TemplateManifest,
        overhead: BackendOverhead,
        running: list[tuple[str, ResourceSpec]],
    ) -> int:
        if candidate.resources.mem_request_bytes <= 0:
            return 1_000_000
        per_sandbox = candidate.resources.mem_request_bytes + overhead.mem_bytes_per_sandbox
        running_load = self._running_mem_load(running, overhead)
        usable = node.hardware.mem_bytes * (1 - self._headroom.mem_fraction) - running_load
        return max(0, math.floor(usable / per_sandbox))

    def _sandbox_writable_pool_bytes(self, node: NodeProfile) -> int:
        """Bytes available to the sandbox-writable disk pool.

        ``hardware.disk_bytes`` is the node's total disk; subtract the OS
        reserve and the image-cache pool's share to get the sandbox-
        writable budget. Spec 10 §"Disk is multi-pool" / spec 15.
        """
        usable_after_os = max(
            0, node.hardware.disk_bytes - self._headroom.disk_reserved_bytes
        )
        image_pool = int(usable_after_os * self._headroom.image_cache_pool_fraction)
        return max(0, usable_after_os - image_pool)

    def _disk_cap(self, node: NodeProfile, candidate: TemplateManifest) -> int:
        per_sandbox = candidate.resources.disk_request_bytes
        if per_sandbox <= 0:
            return 1_000_000
        return max(0, math.floor(self._sandbox_writable_pool_bytes(node) / per_sandbox))

    def _disk_cap_remaining(
        self,
        node: NodeProfile,
        candidate: TemplateManifest,
        running: list[tuple[str, ResourceSpec]],
    ) -> int:
        per_sandbox = candidate.resources.disk_request_bytes
        if per_sandbox <= 0:
            return 1_000_000
        in_use = sum(r.disk_request_bytes for _, r in running)
        usable = self._sandbox_writable_pool_bytes(node) - in_use
        return max(0, math.floor(usable / per_sandbox))

    def _running_cpu_load(
        self,
        running: list[tuple[str, ResourceSpec]],
        overhead: BackendOverhead,
    ) -> float:
        # ``overhead`` is the candidate placement's backend overhead.
        # We charge running sandboxes the same overhead because the
        # cluster is single-backend in phase 0; phase 1 mixed
        # docker+cubesandbox clusters will need per-sandbox backend
        # tracking via SandboxRecord.backend.
        if not running:
            return 0.0
        return sum(
            r.cpu_request + overhead.cpu_per_sandbox for _, r in running
        )

    def _running_mem_load(
        self,
        running: list[tuple[str, ResourceSpec]],
        overhead: BackendOverhead,
    ) -> int:
        if not running:
            return 0
        return sum(
            r.mem_request_bytes + overhead.mem_bytes_per_sandbox for _, r in running
        )


def _binding_label(cpu_cap: int, mem_cap: int, disk_cap: int) -> BindingConstraint:
    smallest = min(cpu_cap, mem_cap, disk_cap)
    if smallest == cpu_cap:
        return "cpu"
    if smallest == mem_cap:
        return "mem"
    return "disk:sandbox_writable"


# ──────────────────────────────────────────────────────────────────────────────
# Stage-3 — health-derived adaptive admission (AIMD).
# See notes/admission-stage-3-aimd-controller.md.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NodeHealthInput:
    """The slice of a node's Stage-1 health snapshot the AIMD
    controller reads. The Stage-3 control loop builds one per connected
    node from the heartbeat-mirrored ``NodeHealthStats``."""

    create_p95_ms: float
    docker_error_count: int
    docker_timeout_count: int


class AimdConfig(BaseModel):
    """Operator knobs for the health-derived admission controller."""

    model_config = ConfigDict(extra="forbid")

    initial_limit: int = 16
    """Slow-start seed — a freshly-connected node's admission limit
    before any health data is observed (D3). Operator-tunable; the
    right start point is cluster-shape-dependent."""

    p95_bad_threshold_ms: float = 60_000.0
    """A node whose ``docker run`` p95 latency exceeds this has a *bad*
    tick → multiplicative decrease (D4). Operator-tunable."""

    max_limit: int = 64
    """Runaway-guardrail: additive-increase never grows a node's limit
    past this. NOT a resource calculation — the real bound is the
    health signal; this only stops drift during a long quiet stretch."""

    decrease_factor: float = 0.5
    """Multiplicative-decrease factor applied on a bad tick."""

    increase_step: int = 1
    """Additive-increase applied on a good tick where the node is
    saturating its current limit."""

    floor: int = 1
    """A node's limit never contracts below this — one in-flight
    acquire keeps the node making progress."""


class HealthAimdController:
    """Per-node AIMD admission controller (Stage 3, pillar P1).

    Pure: :meth:`step` takes the current per-node health + load and
    runs one AIMD round; :meth:`limit_for` reads a node's limit back.
    No I/O, no async — the Stage-3 control loop owns the cadence and
    the data-gathering.

    AIMD: a *bad* health tick halves a node's limit (contract hard); a
    *good* tick grows it by one. Error faults are edge-triggered (only a
    NEW docker error/timeout since the prior tick is bad — not a still-
    populated 120 s window; see :meth:`_is_bad`), so a single transient
    fault costs one halving, not one-per-tick-until-it-ages-out. Growth
    has two regimes: a node saturating its current limit
    (``node_load == limit``) earns demonstrated growth up to
    ``max_limit``; a node whose live load is at/below the seed converges
    its limit back to the seed on clean ticks — recovering UP if a
    transient fault contracted it below (so it can't stay pinned at the
    floor) and decaying DOWN if a past burst pushed it above (so an idle
    node resets to the default instead of holding a stale-high limit). A
    node genuinely carrying load above the seed keeps its earned
    headroom. The limit is
    deliberately NOT derived from ``StaticCapacityEstimator`` — for raw
    containers per-task cost is unknowable upfront, so the health
    signal is the real bound and ``max_limit`` is only a guardrail.
    """

    def __init__(self, config: AimdConfig | None = None) -> None:
        self._config = config or AimdConfig()
        self._limits: dict[str, int] = {}
        # node_id -> monotonic ts of the last multiplicative decrease,
        # surfaced on the admin "Cluster health" page.
        self._last_contraction: dict[str, float] = {}
        # node_id -> the docker error / timeout COUNT observed on the
        # previous tick. The Stage-1 health snapshot reports a *windowed*
        # count (errors in the last ~120 s), but the controller ticks far
        # more often (~15 s), so a single error sits in the window across
        # ~8 ticks. Halving on every tick where the window is non-empty
        # collapses 16→1 from one error and pins the node at the floor
        # for the whole window (and indefinitely under a steady error
        # trickle). We instead edge-trigger: a tick is "bad" only when the
        # count *grew* since last tick — i.e. a genuinely NEW error landed.
        self._last_err_count: dict[str, int] = {}
        self._last_timeout_count: dict[str, int] = {}

    @property
    def config(self) -> AimdConfig:
        return self._config

    def limit_for(self, node_id: str) -> int:
        """A node's current admission limit; an unseen node gets the
        slow-start seed."""
        return self._limits.get(node_id, self._config.initial_limit)

    def last_contraction_at(self, node_id: str) -> float | None:
        """Monotonic timestamp of ``node_id``'s last contraction, or
        ``None`` if it has never contracted."""
        return self._last_contraction.get(node_id)

    def _is_bad(self, node_id: str, health: NodeHealthInput | None) -> bool | None:
        """``True`` = bad tick, ``False`` = good, ``None`` = unknown
        (no Stage-1 health from this node — a pre-Stage-1 node-agent).

        Errors are *edge-triggered*: only a count that grew since the
        previous tick marks the tick bad (a NEW error landed). A
        still-non-empty window from an old error is not a fresh fault —
        otherwise one error would halve the limit on every tick until it
        ages out of the ~120 s window (see ``__init__``). The windowed
        count can also shrink (eviction) or reset (node reconnect); both
        clamp to a non-negative delta so they never read as new faults.
        ``create_p95`` latency stays level-triggered — it is already a
        smooth instantaneous saturation signal, not a cumulative count.
        """
        if health is None:
            return None
        prev_err = self._last_err_count.get(node_id, 0)
        prev_timeout = self._last_timeout_count.get(node_id, 0)
        self._last_err_count[node_id] = health.docker_error_count
        self._last_timeout_count[node_id] = health.docker_timeout_count
        new_errors = max(0, health.docker_error_count - prev_err)
        new_timeouts = max(0, health.docker_timeout_count - prev_timeout)
        if new_errors > 0 or new_timeouts > 0:
            return True
        return health.create_p95_ms > self._config.p95_bad_threshold_ms

    def step(
        self,
        *,
        health: dict[str, NodeHealthInput | None],
        load: dict[str, int],
    ) -> None:
        """Run one AIMD round over every currently-connected node.

        ``load`` is the authoritative connected-node set — one entry
        per node, its current running-container count. ``health``
        carries the latest Stage-1 snapshot per node (``None`` when the
        node has not reported Stage-1 health).
        """
        cfg = self._config
        for node_id, node_load in load.items():
            limit = self._limits.get(node_id, cfg.initial_limit)
            bad = self._is_bad(node_id, health.get(node_id))
            if bad is True:
                contracted = max(cfg.floor, int(limit * cfg.decrease_factor))
                if contracted < limit:
                    self._last_contraction[node_id] = time.monotonic()
                limit = contracted
            elif bad is False:
                if node_load == limit:
                    # Good health AND the node is exactly at its limit —
                    # it earned demonstrated growth above the seed.
                    limit = min(cfg.max_limit, limit + cfg.increase_step)
                elif node_load < cfg.initial_limit and limit != cfg.initial_limit:
                    # Converge to the slow-start seed (the default) from
                    # EITHER side, one step per clean tick, whenever the
                    # node's actual load is at/below the seed — i.e. the
                    # seed alone would comfortably cover it, so the current
                    # limit is unjustified:
                    #   * limit ABOVE the seed → decay DOWN. A node pushed
                    #     high under a past burst and now idle has no live
                    #     evidence it can still take that many; reset to the
                    #     default and let it re-earn headroom via
                    #     demonstrated load (and avoid a stale-high limit
                    #     admitting a sudden flood). This is the "idle node
                    #     stuck at 23" case.
                    #   * limit BELOW the seed → recover UP. A node halved
                    #     to the floor by a transient fault would otherwise
                    #     only regrow when ``node_load == limit`` exactly —
                    #     never met once it has drained to idle — so it'd
                    #     stay pinned at 1 long after the daemon recovered.
                    # A node genuinely carrying load above the seed
                    # (``node_load >= initial_limit``) keeps its elevated
                    # limit — that load justifies it; we don't decay into
                    # its working set. A node still OVER its limit
                    # post-contraction holds and drains.
                    step = cfg.increase_step
                    if limit > cfg.initial_limit:
                        limit = max(cfg.initial_limit, limit - step)
                    else:
                        limit = min(cfg.initial_limit, limit + step)
            # bad is None (no health data) → hold.
            self._limits[node_id] = limit
        # Drop state for nodes that are no longer connected.
        for stale in set(self._limits) - set(load):
            self._limits.pop(stale, None)
            self._last_contraction.pop(stale, None)
            self._last_err_count.pop(stale, None)
            self._last_timeout_count.pop(stale, None)


__all__ = [
    "DEFAULT_BACKEND_OVERHEAD",
    "AimdConfig",
    "BackendOverhead",
    "BindingConstraint",
    "CapacityCell",
    "CapacityEstimator",
    "HeadroomConfig",
    "HealthAimdController",
    "NodeHealthInput",
    "NodeProfile",
    "StaticCapacityEstimator",
]
