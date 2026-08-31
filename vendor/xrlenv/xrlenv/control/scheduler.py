"""Capacity-aware scheduler (spec 03 + spec 10).

The scheduler asks the :class:`CapacityEstimator` whether the candidate
template fits on each eligible node, then picks the node whose remaining
capacity stays most balanced after the placement (largest minimum axis
across cpu / mem / sandbox-writable disk). Ties break by node_id for
deterministic placements in tests and admin tooling.

Per spec 00 invariant 2, capacity is treated as released only when the node
confirms destroy — the scheduler reads ``StateStore.list_sandboxes()`` and
counts only sandboxes whose ``status`` is not ``destroyed``.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from xrlenv.backends.base import CpuIsolation, ResourceSpec
from xrlenv.control.capacity import (
    CapacityEstimator,
    HealthAimdController,
    NodeProfile,
    StaticCapacityEstimator,
)
from xrlenv.control.defaults import DEFAULT_BACKEND
from xrlenv.control.node_transport import NodeTransport
from xrlenv.control.state import StateStore
from xrlenv.control.template_catalog import TemplateCatalog, TemplateManifest
from xrlenv.disk_policy import (
    DISK_PRESSURE_FREE_BYTES_FLOOR as DISK_PRESSURE_FREE_BYTES_FLOOR,
)
from xrlenv.disk_policy import (
    DISK_PRESSURE_FREE_FRACTION_FLOOR as DISK_PRESSURE_FREE_FRACTION_FLOOR,
)
from xrlenv.disk_policy import disk_admit_free_floor_bytes
from xrlenv.errors import BackendCapabilityMissing, CapacityExhausted
from xrlenv.observability.tracing import get_tracer

if TYPE_CHECKING:
    from xrlenv.control.state import SandboxRecord


# Issue #14 — disk-pressure gate. Any node whose last-reported free disk
# is at/below the admit floor refuses placement. The floor
# (``max(absolute, 5% of total)``) lives in ``xrlenv.disk_policy`` (imported
# above) as the SINGLE source of truth shared with the node image cache —
# the two planes must not hardcode disagreeing disk thresholds (that
# produced the P1 disk-exclusion deadlock; see notes/audit.md).


# Issue #18 (Ask #2) — node-health timeout gate. A node that missed a
# command reply within this trailing window is excluded from
# placement. The gate is biased toward over-excluding: the cost of a
# false exclusion is one cooldown window of reduced capacity (the
# admission queue absorbs it), while the cost of a false *non*-
# exclusion is a consumer waiting out a full acquire ceiling (600 s)
# on a wedged node — see the 89/30 acquire-timeout skew in issue #18,
# where the scheduler kept routing to a node that had already missed
# multiple replies. 120 s is long enough to ride out a transient
# blip's worth of clustered timeouts without flapping, short enough
# that a recovered node rejoins quickly.
NODE_TIMEOUT_COOLDOWN_S = 120.0


def _is_command_timeout_unhealthy(node: NodeTransport) -> bool:
    # Tolerate transports that pre-date the protocol extension or
    # return something unreadable (MagicMock fixtures, in-process
    # transport returning None). Anything we can't read cleanly is
    # treated as healthy so the gate stays opt-in.
    probe = getattr(node, "seconds_since_last_command_timeout", None)
    if probe is None:
        return False
    try:
        elapsed = probe()
    except Exception:
        return False
    if elapsed is None:
        return False  # node has never timed out
    # Strict numeric-instance check — mirrors the tuple check in
    # ``_is_disk_pressured``. A MagicMock fixture returns a MagicMock
    # here (whose ``__float__`` defaults to ``1.0``), which would
    # otherwise falsely trip the gate. Anything that isn't a real
    # ``int`` / ``float`` classifies as "unknown / healthy".
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        return False
    # Negative elapsed shouldn't happen (monotonic clock) but guard
    # anyway: a garbage value classifies as healthy, not unhealthy.
    return 0.0 <= float(elapsed) < NODE_TIMEOUT_COOLDOWN_S


def _node_supported_runtimes(node: NodeTransport) -> list[str]:
    """§5.3 — the node's advertised OCI runtimes, defensively.

    Node stand-ins / test doubles that pre-date ``supported_runtimes()``
    are treated as advertising only ``["runc"]`` — so a sysbox request
    simply won't match them (never a crash), and the normal runc path is
    unaffected.
    """
    probe = getattr(node, "supported_runtimes", None)
    if probe is None:
        return ["runc"]
    try:
        runtimes = list(probe() or [])
    except Exception:
        return ["runc"]
    return runtimes or ["runc"]


def _is_disk_pressured(node: NodeTransport) -> bool:
    # Tolerate transports that pre-date ``disk_state()`` or return
    # something other than a (free, total) pair (e.g. MagicMock fixtures
    # in legacy unit tests). Anything we can't read cleanly is treated
    # as "unknown / healthy" so the gate stays opt-in for transports
    # that opt out of the protocol extension.
    probe = getattr(node, "disk_state", None)
    if probe is None:
        return False
    try:
        sample = probe()
    except Exception:
        return False
    # Strict tuple check so MagicMock-style fixtures (which return
    # arbitrary-typed mocks even when subscripted) classify as
    # "unknown" rather than tripping the gate on garbage values.
    if not isinstance(sample, tuple) or len(sample) != 2:
        return False
    try:
        free, total = int(sample[0]), int(sample[1])
    except (TypeError, ValueError):
        return False
    # ``(0, 0)`` is the documented "unknown" sentinel — a freshly-
    # connected node that hasn't sent its first heartbeat. Treat as
    # healthy so the gate doesn't blackhole the cluster on bring-up.
    if free == 0 and total == 0:
        return False
    return free <= disk_admit_free_floor_bytes(total)


def _pinnable_cores(resources: ResourceSpec | None) -> int:
    """P6 — whole cores a *pinning* rollout (``BEST_EFFORT`` or ``REQUIRED``)
    will try to reserve on its node: ``ceil(cpu_limit)``, floored at 1. ``0``
    for ``OFF``. Mirrors the node's :func:`_allocate_cpuset` pin count.

    The scheduler uses this both for the hard ``REQUIRED`` placement predicate +
    reservation (§8.12) and the soft ``BEST_EFFORT`` capable-node score nudge;
    ``is_required`` (checked at the call site) selects which. A ``REQUIRED``
    request with a non-positive ``cpu_limit`` still needs one dedicated core."""
    if resources is None or not resources.cpu_isolation.pins:
        return 0
    return max(1, math.ceil(resources.cpu_limit))


def _node_isolation_capable(node: NodeTransport) -> bool:
    """§8.6 — whether ``node`` advertised P6 isolation capability, defensively.

    Strict ``is True`` so a MagicMock / pre-P6 transport (whose
    ``isolation_capable()`` returns a truthy mock or is absent) classifies as
    NON-capable — a REQUIRED rollout is never placed on a node that can't prove
    it enforces isolation (fail-safe, matching the ``_is_disk_pressured``
    tolerance discipline)."""
    probe = getattr(node, "isolation_capable", None)
    if probe is None:
        return False
    try:
        return probe() is True
    except Exception:
        return False


def _node_pinned_cpu_state(node: NodeTransport) -> tuple[int, int]:
    """§8.6 R6 — the node's last-reported ``(pinned_cpus_free, pinned_cpus_total)``,
    defensively. ``(0, 0)`` (the "unknown" sentinel) for a transport that
    pre-dates the field or returns a non-(int,int) value (e.g. a MagicMock
    fixture) — so a REQUIRED placement, which needs ``free >= need >= 1``, is
    refused on a node whose pinnable capacity can't be read cleanly."""
    probe = getattr(node, "pinned_cpu_state", None)
    if probe is None:
        return (0, 0)
    try:
        sample = probe()
    except Exception:
        return (0, 0)
    if not isinstance(sample, tuple) or len(sample) != 2:
        return (0, 0)
    try:
        return (int(sample[0]), int(sample[1]))
    except (TypeError, ValueError):
        return (0, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Score function (A1 / D18 — P1.2)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlacementFeatures:
    """Per-node features the scoring function consumes when ``Scheduler.place``
    walks candidates. Future features (network slack, etc.) get added here as
    new fields with sensible defaults; the score-function interface stays
    backward-compatible."""

    node_id: str
    manifest: TemplateManifest
    resource_slack: float
    """Min-axis remaining-slack fraction across the requested resource axes
    (cpu / mem / disk) after hypothetically placing the candidate. Bottleneck-
    aware: a saturated CPU axis can't be compensated by empty disk. Range
    ``[0, 1]``."""
    image_present: bool
    """``True`` if the candidate's image is on this node (per the bidi
    ``QueryImage`` snapshot the admission queue captured pre-place)."""
    is_planner_preferred_home: bool = False
    """Audit P1.6.g-H2 (2026-05-05): ``True`` if this node is the build
    coordinator's recorded ``preferred_home`` for a deferred (status=
    ``registered``) row matching the candidate's image AND no
    candidate has the image yet. Soft signal: the cluster-wide gate
    is enforced at :meth:`Scheduler.place` time so every candidate in
    the same call sees a consistent answer (P1.6.g-M3). Once any
    node materializes the image the flag is forced to ``False`` for
    all candidates and the existing ``image_present`` channel takes
    over. The default score function interprets this as a half-
    strength image-affinity bonus so first-rollout placements honor
    the planner's spread plan without overruling a saturated
    preferred home."""
    best_effort_pinnable: bool = False
    """P6 — ``True`` when this is a ``BEST_EFFORT``-cpu-isolation rollout AND this
    node is isolation-capable with enough free pinnable cores (minus in-flight
    ``REQUIRED`` reservations) to *actually pin* it. A SOFT signal: the default
    score function (:func:`weighted_sum_score`) adds a small
    :data:`_BEST_EFFORT_ISOLATION_BONUS` so best_effort work lands where it gets
    dedicated cores instead of degrading to CFS quota — subordinate to resource
    slack, so a saturated capable node still loses to an empty incapable one.
    :meth:`Scheduler.place` computes it and hands it to the score function, so a
    CUSTOM ``score_fn`` OWNS this policy (it decides whether / how to weight the
    signal, on its own score scale). ``False`` for ``OFF`` / ``REQUIRED``
    (``REQUIRED`` uses the hard placement predicate, never this soft signal), a
    non-capable node, or a capable node without enough free cores."""


ScoreFn = Callable[[PlacementFeatures], float]
"""Type alias for the scoring callback. Takes per-node features, returns a
float (higher = better candidate). Operators plug in custom scoring via
``Scheduler(... score_fn=...)``."""


#: A1 / D18 (P1.2) — default weights baked into
#: :func:`weighted_sum_score`. The scoring formula is
#: ``score = w_R * R + w_I * I`` where both ``R`` (resource slack) and
#: ``I`` (image presence) are in ``[0, 1]`` and weights sum to 1.
#:
#: Defaults are resource-biased (2/3 vs 1/3) on the explicit
#: justification that capacity-to-actually-run is fundamentally
#: heavier than one-time-cost-amortized-over-rollout: a saturated
#: node either fails the ``fits()`` capacity gate or runs the
#: rollout slowly throughout its life, while a cold-image node only
#: pays a one-time pull/build cost at start. Different categories of
#: cost; the resource term should weigh more.
DEFAULT_RESOURCE_WEIGHT: float = 2.0 / 3.0
DEFAULT_IMAGE_AFFINITY_WEIGHT: float = 1.0 / 3.0

#: P6 — the DEFAULT score function's weight for the ``best_effort_pinnable``
#: feature: a small additive bonus toward an isolation-capable node with free
#: pinnable cores, for a ``BEST_EFFORT`` rollout — so it lands where it will
#: actually pin instead of degrading to CFS quota. Deliberately SMALL relative to
#: the default scorer's ``[0, 1]`` scale (subordinate to the resource-slack term,
#: whose weight is ~0.67): a capable-but-saturated node must not beat an
#: incapable-but-empty one (best_effort runs fine unpinned on the empty node; it
#: would run *worse* pinned on the saturated one). Below the preferred-home
#: half-bonus (~0.17), so image affinity still wins. This is DEFAULT-scorer
#: policy, NOT a scheduler-imposed nudge — a custom ``score_fn`` receives the
#: ``best_effort_pinnable`` feature and owns its own weighting on its own scale.
_BEST_EFFORT_ISOLATION_BONUS: float = 0.1


def weighted_sum_score(
    *,
    resource_weight: float = DEFAULT_RESOURCE_WEIGHT,
    image_affinity_weight: float = DEFAULT_IMAGE_AFFINITY_WEIGHT,
) -> ScoreFn:
    """Build a :data:`ScoreFn` that computes ``w_R * R + w_I * I`` (+ a small P6
    isolation bonus).

    Both inputs are normalised to ``[0, 1]``; weights sum to ``1.0``, so the
    ``w_R * R + w_I * I`` base is in ``[0, 1]``. Validation (range + sum) runs at
    factory-call time so misconfiguration is caught at boot, not silently at
    placement time. P6: a ``best_effort_pinnable`` candidate additionally takes a
    small :data:`_BEST_EFFORT_ISOLATION_BONUS`, which can push the score slightly
    above ``1.0`` — harmless, since scores are only ever compared relatively.

    Operators override defaults by re-invoking the factory:

    .. code-block:: python

        sched = Scheduler(
            ...,
            score_fn=weighted_sum_score(
                resource_weight=0.4, image_affinity_weight=0.6,
            ),
        )

    See ``docs/technical/scheduling.md`` for the calibration story
    (when to bias toward affinity vs slack).
    """
    if not (0.0 <= resource_weight <= 1.0):
        raise ValueError(
            f"resource_weight must be in [0, 1]; got {resource_weight!r}"
        )
    if not (0.0 <= image_affinity_weight <= 1.0):
        raise ValueError(
            f"image_affinity_weight must be in [0, 1]; got "
            f"{image_affinity_weight!r}"
        )
    if not math.isclose(
        resource_weight + image_affinity_weight, 1.0, abs_tol=1e-9,
    ):
        raise ValueError(
            f"resource_weight + image_affinity_weight must sum to 1.0; "
            f"got {resource_weight} + {image_affinity_weight} = "
            f"{resource_weight + image_affinity_weight}"
        )

    def _score(features: PlacementFeatures) -> float:
        # Audit P1.6.g-H2 (2026-05-05): preferred_home is a soft
        # bonus equal to half the image-affinity weight. The
        # cluster-wide gate ("only when no node has the image yet")
        # lives in :meth:`Scheduler.place`, which sets
        # ``is_planner_preferred_home=False`` on every candidate
        # once any node has the image (P1.6.g-M3 fix). The score
        # function therefore only needs the per-candidate gate
        # below: a node that has the image takes the full bonus;
        # otherwise, if it carried the preferred_home flag through
        # from the scheduler's gate, take the half bonus. Stays
        # inside the [0, 1] range.
        affinity = (
            1.0 if features.image_present
            else (0.5 if features.is_planner_preferred_home else 0.0)
        )
        # P6 — soft best_effort capable-node preference: a small additive bonus
        # for a candidate that can actually pin the best_effort rollout (the
        # scheduler set this feature; see PlacementFeatures.best_effort_pinnable).
        # Subordinate to the resource-slack term. A custom score_fn that ignores
        # the feature simply won't nudge — it owns the policy.
        isolation = (
            _BEST_EFFORT_ISOLATION_BONUS if features.best_effort_pinnable else 0.0
        )
        return (
            resource_weight * features.resource_slack
            + image_affinity_weight * affinity
            + isolation
        )

    return _score


#: Module-level default score function — exported so callers can name
#: it explicitly when they want to defer to the platform's default
#: while still passing ``score_fn=...`` (e.g., from a config-driven
#: factory that may or may not override scoring).
DEFAULT_SCORE_FN: ScoreFn = weighted_sum_score()


class Placement(BaseModel):
    """Result of a successful scheduling decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    node: SkipValidation[NodeTransport]
    """SkipValidation: a NodeTransport is a runtime component reference (either
    in-process NodeAgent or a gRPC-backed RemoteNodeTransport), not wire data.
    """
    backend: str = DEFAULT_BACKEND
    score: float
    """Placement score in ``[0, 1]`` — the weighted sum
    ``w_R * R + w_I * I`` from the ``Scheduler.place`` formula
    (A7/D13 + A1/D18, P1.1+P1.2). Higher = better candidate.
    Compared as a float; the scheduler picks the maximum, ties
    broken by ``node_id`` ascending.

    Synthetic ``Placement(score=1)`` / ``score=1.0`` constructions
    in tests are fine — the field is informational outside
    ``Scheduler.place``; pydantic coerces ``int`` to ``float``."""

    reservation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Opaque token returned by :py:meth:`Scheduler.place`. The coordinator
    passes it back to :py:meth:`Scheduler.commit_placement` once the placed
    sandbox is recorded in state, or to :py:meth:`Scheduler.release_placement`
    if create-sandbox fails. Used so concurrent ``place()`` calls see each
    other's pending placements before any state mutation has happened —
    closes the race where N concurrent calls all read ``state.list_sandboxes()``
    too early and ignore the in-flight peers' anti-affinity contribution.

    Defaults to a fresh uuid so tests can construct ``Placement(...)``
    without thinking about reservation lifecycle. Real ``Scheduler.place``
    overrides it with a token that's already entered into ``_pending``.
    """


@dataclass
class _PendingPlacement:
    """In-flight reservation tracked by :class:`Scheduler` until the
    coordinator finishes recording its sandbox in state. Counts toward
    every concurrent :py:meth:`Scheduler.place` call's view of cluster
    load so anti-affinity (``max_runs_per_task``) and capacity decisions
    don't see stale state.

    ``effective_resources`` is the post-resolver overlay snapshot
    captured at placement time so concurrent placements see the *real*
    per-task resource cost of in-flight reservations rather than the
    outer manifest's defaults (Slice 9b — Pattern A with heterogeneous
    per-task resources)."""

    node_id: str
    template_name: str
    task_key: str | None
    effective_resources: ResourceSpec
    container_runtime: str | None = None
    """OCI runtime of this in-flight placement (§ per-node runtime cap). Counted
    with running sessions so a burst of concurrent ``place()`` calls can't
    over-place past a runtime cap before any becomes a running session."""
    pinned_cores_reserved: int = 0
    """P6 step-4b (§8.12) — whole pinnable cores this in-flight REQUIRED-isolation
    placement will consume on its node. Summed per-node under ``_pending_lock`` so
    a burst of concurrent REQUIRED ``place()`` calls can't over-commit a node's
    pinnable pool before any becomes a running (heartbeat-reflected) pin. ``0`` for
    OFF / BEST_EFFORT placements (not gated)."""


@dataclass(frozen=True)
class RawSessionLoad:
    """Capacity-accounting view of one active raw-container session.

    Raw containers don't appear in ``state.list_sandboxes()`` — they live
    in ``RawContainerCoordinator._sessions`` as in-memory rows. Without a
    second load source, :py:meth:`Scheduler._gather_cluster_load` would
    be blind to them and the capacity gate would over-place. The runtime
    wires ``RawContainerCoordinator.iter_load_entries`` in via
    :py:meth:`Scheduler.set_raw_session_provider`; the scheduler folds
    the result on top of the StateStore-managed load for every placement
    decision.

    Field shape mirrors the per-row tuple ``_gather_cluster_load``
    consumes from ``SandboxRecord`` (template_name + effective_resources +
    task_key match) so the two sources merge cleanly."""

    node_id: str
    template_name: str
    effective_resources: ResourceSpec
    task_key: str | None
    container_runtime: str | None = None
    """OCI runtime of this running raw session (§ per-node runtime cap). Lets the
    scheduler count concurrent sysbox containers per node without reshaping the
    resource-load tuple. ``None`` for the ordinary runc path."""


RawSessionProvider = Callable[[], list[RawSessionLoad]]


def _empty_raw_session_provider() -> list[RawSessionLoad]:
    return []


class Scheduler:
    """Capacity-aware single-rollout scheduler (Slice 2).

    Slice 2.5+ adds the admission-queue draining loop on top; this class only
    answers "given a manifest and the current cluster state, where should
    this rollout land?". The admission queue calls :py:meth:`place` and
    handles the ``CapacityExhausted`` case by enqueueing.
    """

    def __init__(
        self,
        nodes: list[NodeTransport],
        *,
        catalog: TemplateCatalog,
        state: StateStore,
        estimator: CapacityEstimator | None = None,
        max_runs_per_task: int = 4,
        allow_empty: bool = False,
        image_aware_placement: bool = True,
        score_fn: ScoreFn | None = None,
        raw_session_provider: RawSessionProvider | None = None,
        aimd_controller: HealthAimdController | None = None,
        runtime_caps: dict[str, dict[str, int]] | None = None,
    ) -> None:
        if not nodes and not allow_empty:
            raise ValueError(
                "Scheduler requires at least one node (pass allow_empty=True "
                "for distributed bring-up where nodes arrive lazily)"
            )
        self._nodes = list(nodes)
        self._catalog = catalog
        self._state = state
        self._estimator = estimator or StaticCapacityEstimator()
        self._max_runs_per_task = max_runs_per_task
        self._image_aware = image_aware_placement
        # ``None`` defers to the module-level default; passing the
        # default explicitly is also valid (some config-driven
        # factories always pass a ``score_fn`` even when no override
        # is intended). Operators with custom scoring use
        # ``weighted_sum_score(resource_weight=..., image_affinity_weight=...)``
        # for tuned defaults, or pass any other ``ScoreFn``-shaped
        # callable for fully custom logic.
        self._score_fn: ScoreFn = score_fn or DEFAULT_SCORE_FN
        # Threading lock + pending-placement registry. ``place`` does no
        # I/O while holding the lock — it's just arithmetic over
        # state-snapshot + pending-snapshot — so contention is bounded
        # by computation only. Both ``commit_placement`` and
        # ``release_placement`` acquire the same lock so concurrent
        # placements see each other's reservations until they're
        # superseded by state records.
        self._pending_lock = threading.Lock()
        self._pending: dict[str, _PendingPlacement] = {}
        # Second load source — raw-container sessions live in
        # ``RawContainerCoordinator._sessions``, not ``state.list_sandboxes()``.
        # Default to an empty provider so the Scheduler is usable standalone
        # (single-rollout SDK, tests). Production wiring in ``runtime.py`` /
        # ``distributed_runtime.py`` calls ``set_raw_session_provider`` once
        # the RawContainerCoordinator exists.
        self._raw_session_provider: RawSessionProvider = (
            raw_session_provider or _empty_raw_session_provider
        )
        # Stage-3 (P1) — health-derived adaptive admission. ``None``
        # (the default, and when ``adaptive_admission`` is off) makes
        # the per-node AIMD-limit filter in ``place`` a no-op.
        self._aimd = aimd_controller
        # Per-node per-runtime concurrency caps (sysbox-fs wedge prevention —
        # notes/design-per-node-runtime-concurrency-cap.md). ``{node_id:
        # {runtime: max_concurrent}}`` from nodes.yaml. Empty / missing ⇒
        # unlimited ⇒ the placement gate is a no-op for that node+runtime, so
        # every uncapped runtime (runc, None) is unchanged.
        self._runtime_caps: dict[str, dict[str, int]] = dict(runtime_caps or {})

    def node_load_snapshot(self) -> dict[str, int]:
        """Stage-3 — per-node running-container count, for the AIMD
        control loop. A coarse point-in-time read (in-flight pending
        reservations are not folded in); the loop only needs a load
        signal, not the placement-grade count ``place`` uses."""
        load = self._gather_cluster_load(task_key=None)
        return {nid: len(running) for nid, (running, _tc) in load.items()}

    def set_raw_session_provider(self, provider: RawSessionProvider) -> None:
        """Wire a raw-container load source into ``_gather_cluster_load``.

        Called once at runtime construction, after both ``Scheduler`` and
        ``RawContainerCoordinator`` have been built. Without this, the
        scheduler's capacity gate can't see active raw containers and
        will over-place them — silent over-subscription that breaks the
        operator's parallelism contract.

        Idempotent: a follow-up call replaces the provider (useful for
        tests that swap the provider mid-flight).
        """
        self._raw_session_provider = provider

    def set_runtime_caps(self, caps: dict[str, dict[str, int]]) -> None:
        """Install per-node per-runtime concurrency caps
        (``{node_id: {runtime: max_concurrent}}``) from ``nodes.yaml``.

        Called once at distributed-runtime construction, after the inventory
        is loaded. Replaces the current map (empty by default ⇒ no caps ⇒ the
        placement gate is a no-op). See
        notes/design-per-node-runtime-concurrency-cap.md.
        """
        self._runtime_caps = dict(caps)

    def add_node(self, node: NodeTransport) -> None:
        self._nodes.append(node)

    def remove_node(self, node_id: str) -> None:
        self._nodes = [n for n in self._nodes if n.node_id != node_id]

    @property
    def nodes(self) -> list[NodeTransport]:
        return list(self._nodes)

    @property
    def estimator(self) -> CapacityEstimator:
        return self._estimator

    @property
    def image_aware_placement(self) -> bool:
        return self._image_aware

    # ── Placement ────────────────────────────────────────────────────────────

    def place(
        self,
        manifest: TemplateManifest,
        *,
        task_key: str | None = None,
        backend: str | None = None,
        container_runtime: str | None = None,
        image_present: dict[str, bool] | None = None,
        preferred_home_node: str | None = None,
        reserve: ResourceSpec | None = None,
        exclude_node_ids: frozenset[str] | None = None,
    ) -> Placement:
        """Pick a node and reserve capacity for one placement.

        ``reserve`` (fleet reservation, phase 1 — spec 03/10): when set, the
        placement gates feasibility, scores, and reserves against this
        **footprint** ``ResourceSpec`` instead of ``manifest.resources``.
        This is the fleet-opening acquire: the whole fleet's peak cpu+mem is
        reserved on one node at once so the later companions can't be starved.
        ``reserve is None`` (the default, every non-fleet placement) is the
        legacy behavior byte-for-byte — the container's own
        ``manifest.resources`` is reserved. The manifest's ``name`` / ``image``
        (used for error messages + image affinity) are untouched either way;
        only the reserved resource amount changes. See the atomic-handoff
        section of ``notes/fleet-reservation-r1-load-accounting.md``.
        """
        with get_tracer().start_as_current_span(
            "xrlenv.scheduler.place",
            attributes={
                "template": manifest.name,
                "image": manifest.image or "",
                "backend": backend or DEFAULT_BACKEND,
                "node_count": len(self._nodes),
                "fleet_footprint_reserve": reserve is not None,
            },
        ):
            return self._place_impl(
                manifest,
                task_key=task_key,
                backend=backend,
                container_runtime=container_runtime,
                image_present=image_present,
                preferred_home_node=preferred_home_node,
                reserve=reserve,
                exclude_node_ids=exclude_node_ids,
            )

    def _place_impl(
        self,
        manifest: TemplateManifest,
        *,
        task_key: str | None,
        backend: str | None,
        container_runtime: str | None = None,
        image_present: dict[str, bool] | None,
        preferred_home_node: str | None,
        reserve: ResourceSpec | None = None,
        exclude_node_ids: frozenset[str] | None = None,
    ) -> Placement:
        # ``backend`` is per-rollout user policy. The plug-in author
        # doesn't know which sandbox runtime the operator wants; the
        # consumer supplies it via the run-config (or per-rollout
        # kwarg). Falls back to the platform default when the caller
        # didn't supply one.
        effective_backend = backend or DEFAULT_BACKEND
        # Filter to nodes whose backend list advertises the requested backend.
        # Pure-capability check first so we can raise a precise error if no
        # node could ever support this template (vs "all eligible nodes are
        # full right now").
        backend_capable = [
            n for n in self._nodes if effective_backend in n.supported_backends()
        ]
        if not backend_capable:
            raise BackendCapabilityMissing(
                f"no node supports backend {effective_backend!r} for template "
                f"{manifest.name!r}"
            )

        # §5.3 — runtime filter. Only narrows when a non-default runtime is
        # requested, so the normal (runc / None) path is unchanged and every
        # node stays eligible. A missing ``supported_runtimes`` on a node
        # stand-in defaults to ["runc"] (getattr guard), so a sysbox request
        # simply won't match it — never a crash. Runs BEFORE the disk /
        # capacity gates so we fail with a precise "no node supports runtime"
        # rather than an opaque capacity error, and before any reservation.
        if container_runtime and container_runtime != "runc":
            runtime_capable = [
                n for n in backend_capable
                if container_runtime in _node_supported_runtimes(n)
            ]
            if not runtime_capable:
                raise BackendCapabilityMissing(
                    f"no node supports runtime {container_runtime!r} for "
                    f"template {manifest.name!r} (backend "
                    f"{effective_backend!r}). Install/advertise the runtime "
                    f"on a node pool (§5.3), or acquire without "
                    f"container_runtime."
                )
            backend_capable = runtime_capable

        # D-AR-2026-07-07-B — re-admit exclusion. The coordinator re-admits an
        # acquire that failed at create time with a node-saturation 5xx /
        # DeadlineExceeded, passing the id(s) of the node(s) that just failed so
        # placement steers to a sibling. This runs AFTER the capability checks
        # (backend + runtime) so we can tell the two failure classes apart:
        #   - pool empty BEFORE exclusion  → BackendCapabilityMissing (above)
        #   - pool non-empty, emptied only BY exclusion → CapacityExhausted here
        #     (the excluded nodes *can* serve this template; they're just hot).
        # The coordinator handles the all-capable-excluded case by relaxing the
        # exclusion to ∅ before it ever reaches here (decision 3), so this gate
        # firing means "some capable nodes remain, they were just all excluded
        # this attempt" — queue and retry, don't hard-fail the template.
        if exclude_node_ids:
            unexcluded = [
                n for n in backend_capable if n.node_id not in exclude_node_ids
            ]
            if not unexcluded:
                raise CapacityExhausted(
                    f"all backend-capable nodes for template {manifest.name!r} "
                    f"are excluded this admission attempt (recent create-time "
                    f"saturation on {sorted(exclude_node_ids)!r}); the admission "
                    f"queue will retry as a node recovers"
                )
            backend_capable = unexcluded

        # Issue #14 — disk-pressure gate (defense in depth on top of
        # the issue #13 sweep). If the node's last-reported free disk
        # is below the critical floor, refuse to place; let the sweep
        # / operator clear the pressure first. ``(0, 0)`` from
        # ``disk_state()`` is the documented "unknown" sentinel
        # (just-connected node, transport without disk probe) and is
        # treated as healthy so freshly-attached nodes aren't
        # accidentally blackholed before their first heartbeat.
        backend_capable = [
            n for n in backend_capable if not _is_disk_pressured(n)
        ]
        if not backend_capable:
            raise CapacityExhausted(
                f"all backend-capable nodes are under disk pressure for "
                f"template {manifest.name!r} — image-cache sweep cannot "
                f"reclaim enough disk; check operator pin set, container "
                f"overlay growth, or external disk consumers"
            )

        # Issue #18 (Ask #2) — node-health timeout gate. Exclude any
        # node that missed a command reply within the cooldown window;
        # it's wedged or overloaded and routing fresh acquires there
        # just burns a full acquire ceiling per attempt. When this
        # filters every candidate, ``CapacityExhausted`` propagates to
        # the admission queue, which holds the request until a node's
        # cooldown elapses — strictly better than a synchronous error
        # or a doomed placement. Self-healing: a node rejoins
        # automatically once its last timeout ages past the window.
        backend_capable = [
            n for n in backend_capable if not _is_command_timeout_unhealthy(n)
        ]
        if not backend_capable:
            raise CapacityExhausted(
                f"all backend-capable nodes for template {manifest.name!r} "
                f"missed a command reply within the last "
                f"{NODE_TIMEOUT_COOLDOWN_S:.0f}s — every candidate is in "
                f"the node-health cooldown; the admission queue will "
                f"retry once a node recovers"
            )

        # Fleet reservation (phase 1): a fleet-opening placement reserves
        # the whole declared FOOTPRINT, not the lead container's own small
        # request. Substituting ``manifest.resources`` with the footprint
        # here routes the entire feasibility gate (``estimator.fits``), the
        # slack score, AND the ``_pending`` reservation through the footprint
        # — a single substitution rather than a parallel fleet-aware code
        # path. ``name`` / ``image`` are preserved (error messages + image
        # affinity unchanged). ``reserve is None`` leaves the manifest exactly
        # as passed — the legacy per-container path, byte-for-byte.
        if reserve is not None:
            manifest = manifest.model_copy(update={"resources": reserve})

        # ``manifest.resources`` is the post-overlay snapshot for
        # Pattern A — the coordinator hands the scheduler the
        # already-resolved manifest. We charge this exact ResourceSpec
        # to the in-flight reservation + the placement so concurrent
        # ``place()`` calls observe the right per-task cost.
        candidate_resources = manifest.resources

        # P6 step-4b (§8.12) — whole pinnable cores a REQUIRED-isolation rollout
        # needs (0 for OFF / BEST_EFFORT). Drives the hard placement predicate +
        # the reservation. ``pin_need`` is the same count for BEST_EFFORT too —
        # used only for the SOFT capable-node score nudge (no hard filter / no
        # reservation) below. ``is_required`` selects hard vs soft.
        is_required = candidate_resources.cpu_isolation is CpuIsolation.REQUIRED
        pin_need = _pinnable_cores(candidate_resources)
        need_pinned = pin_need if is_required else 0

        # Audit P1.6.g-M3 (2026-05-05): the preferred_home bonus is
        # gated on ``image_present=False`` *everywhere* — once any
        # candidate has the image already, the warm cache should
        # beat the soft routing hint regardless of slack delta,
        # otherwise a high-slack preferred_home can outscore a
        # loaded-but-warm node and trigger a duplicate lazy build.
        # The gate fires at this scope (per-call) rather than per-
        # candidate so every node in the loop below sees a consistent
        # "is preferred_home routing in play this call?" answer.
        any_image_present = (
            self._image_aware
            and image_present is not None
            and any(image_present.values())
        )

        # Take the lock for the whole "read load → decide → reserve"
        # sequence. The body does no I/O while holding the lock, so the
        # critical section is bounded by arithmetic over state +
        # pending snapshots. Holding it across the decision means
        # concurrent ``place()`` calls observe each other's reservations
        # in ``_load_with_pending`` and respect each other's
        # anti-affinity contribution before any sandbox has been
        # recorded in state.
        with self._pending_lock:
            cluster_load = self._load_with_pending(task_key=task_key)

            # Stage-3 (P1) — adaptive-admission filter. Exclude any node
            # already at or above its health-derived AIMD limit; the
            # overflow then queues (Stage 2's admission queue) rather
            # than melting the node's docker daemon. No-op when no
            # controller is wired (``adaptive_admission`` off).
            if self._aimd is not None:
                backend_capable = [
                    n for n in backend_capable
                    if len(cluster_load[n.node_id][0])
                    < self._aimd.limit_for(n.node_id)
                ]
                if not backend_capable:
                    raise CapacityExhausted(
                        f"all backend-capable nodes for template "
                        f"{manifest.name!r} are at their health-derived "
                        f"adaptive admission limit; the admission queue "
                        f"will retry as nodes drain or recover"
                    )

            # Per-node per-runtime concurrency cap (sysbox-fs wedge prevention —
            # notes/design-per-node-runtime-concurrency-cap.md). Exclude nodes
            # already at their cap for the REQUESTED runtime, counting running +
            # in-flight (``_pending``) sessions of that runtime. No-op for an
            # uncapped runtime (``runc`` / ``None`` / no nodes.yaml entry), so
            # the ordinary path is byte-for-byte unchanged. When the cap empties
            # an otherwise-non-empty pool, ``CapacityExhausted`` propagates to
            # the admission queue, which HOLDS the request until a container of
            # this runtime is destroyed and frees a slot (a node-confirmed
            # destroy kicks the queue) — overflow queues, it doesn't fail.
            if container_runtime:
                rt_counts = self._runtime_count_with_pending(container_runtime)
                under_cap = [
                    n for n in backend_capable
                    if not self._runtime_at_cap(
                        n.node_id, container_runtime, rt_counts,
                    )
                ]
                if not under_cap:
                    raise CapacityExhausted(
                        f"all backend-capable nodes for template "
                        f"{manifest.name!r} are at their per-node "
                        f"{container_runtime!r} concurrency cap; the admission "
                        f"queue will hold the request until a "
                        f"{container_runtime!r} container is destroyed and frees "
                        f"a slot"
                    )
                backend_capable = under_cap

            # P6 step-4b (§8.12) — REQUIRED cpu-isolation placement predicate.
            # A rollout that REQUIRES isolation may only land on an
            # isolation-capable node with enough FREE pinnable cores, counting
            # in-flight REQUIRED reservations (``_pending``) the node hasn't yet
            # reflected in a heartbeat — so a burst of concurrent REQUIRED
            # placements can't each read the same free cores and over-commit the
            # pinnable pool. OFF / BEST_EFFORT (``need_pinned == 0``) skip this
            # entirely: best_effort degrades to CFS quota on the node (mechanism,
            # not policy). A node reporting the ``(0, 0)`` unknown sentinel, or a
            # capable node whose free pool (minus pending) can't cover the pin,
            # is filtered; if that empties an otherwise-non-empty pool,
            # ``CapacityExhausted`` propagates to the admission queue, which HOLDS
            # the required request until a capable node frees pinnable cores (a
            # node-confirmed destroy kicks the queue). Legacy-gap nodes report
            # ``pinned_cpus_free=0`` (step-4a fold), so they're refused here too.
            #
            # ``pending_pinned`` (per-node in-flight REQUIRED reservations) is
            # read ONCE here, under the lock, and reused by the soft BEST_EFFORT
            # score nudge in the loop below.
            pending_pinned = (
                self._pending_pinned_cores_by_node() if pin_need > 0 else {}
            )
            if is_required and need_pinned > 0:
                isolation_ok = [
                    n for n in backend_capable
                    if _node_isolation_capable(n)
                    and _node_pinned_cpu_state(n)[0]
                    - pending_pinned.get(n.node_id, 0) >= need_pinned
                ]
                if not isolation_ok:
                    raise CapacityExhausted(
                        f"no isolation-capable node has {need_pinned} free "
                        f"pinnable core(s) for template {manifest.name!r} "
                        f"(cpu_isolation=required); the admission queue will "
                        f"retry as a capable node frees pinnable cores"
                    )
                backend_capable = isolation_ok

            best: Placement | None = None
            best_node: NodeTransport | None = None
            for node in backend_capable:
                running, task_count = cluster_load[node.node_id]
                profile = NodeProfile(
                    node_id=node.node_id,
                    hardware=node.hardware(),
                    backends=tuple(node.supported_backends()),
                )
                if not self._estimator.fits(
                    profile,
                    running=running,
                    candidate=manifest,
                    task_key=task_key,
                    task_count_on_node=task_count,
                    max_runs_per_task=self._max_runs_per_task,
                    backend=effective_backend,
                ):
                    continue

                # A7 / D13 score: load-vector cost. The estimator
                # returns the minimum-axis remaining slack fraction
                # (0.0 - 1.0) after hypothetically placing the
                # candidate; we scale to int (0 - 1000) so
                # ``Placement.score``'s int annotation stays load-
                # bearing for tests that construct synthetic
                # placements. Pre-D13 this was
                # ``max_concurrent - same_template_count``, which only
                # measured slots-of-this-template-remaining and
                # under-balanced heterogeneous Pattern-A clusters.
                # A7/D13 + A1/D18 score: delegate to the configured
                # ``score_fn``. The default (:data:`DEFAULT_SCORE_FN`,
                # built from :func:`weighted_sum_score`) returns the
                # weighted sum ``w_R * R + w_I * I``; operators
                # plugging in custom logic see the same per-node
                # ``PlacementFeatures`` interface. ``image_present``
                # is collapsed to a bool here so the feature struct
                # stays simple — affinity-off / no-snapshot both map
                # to ``False``, and the score function decides what
                # ``False`` means.
                slack = self._estimator.slack_after_placement(
                    profile, manifest, running, backend=effective_backend,
                )
                image_present_bool = (
                    self._image_aware
                    and image_present is not None
                    and image_present.get(node.node_id, False)
                )
                # Audit P1.6.g-H2 (2026-05-05): preferred_home is also
                # gated by image_aware_placement — operators who turn
                # affinity off don't expect the scheduler to read the
                # build snapshot for them either.
                #
                # Audit P1.6.g-M3 (2026-05-05): also globally suppressed
                # when any candidate already has the image, so the
                # warm-cache node always wins regardless of slack
                # delta (see ``any_image_present`` computed above).
                is_preferred_home = (
                    self._image_aware
                    and preferred_home_node is not None
                    and preferred_home_node == node.node_id
                    and not any_image_present
                )
                # P6 — soft BEST_EFFORT capable-node preference, as a FEATURE the
                # score function owns (not a scheduler-imposed bonus): True for a
                # best_effort rollout (NOT required — that used the hard predicate
                # above) on an isolation-capable node with enough free pinnable
                # cores (minus in-flight required reservations) to actually pin it.
                # The default score_fn adds a small bonus; a custom score_fn
                # decides its own weighting. See PlacementFeatures.
                best_effort_pinnable = (
                    pin_need > 0 and not is_required
                    and _node_isolation_capable(node)
                    and _node_pinned_cpu_state(node)[0]
                    - pending_pinned.get(node.node_id, 0) >= pin_need
                )
                features = PlacementFeatures(
                    node_id=node.node_id,
                    manifest=manifest,
                    resource_slack=slack,
                    image_present=image_present_bool,
                    is_planner_preferred_home=is_preferred_home,
                    best_effort_pinnable=best_effort_pinnable,
                )
                total_score = self._score_fn(features)
                if (
                    best is None
                    or total_score > best.score
                    or (total_score == best.score and best_node is not None
                        and node.node_id < best_node.node_id)
                ):
                    best = Placement(
                        node=node,
                        backend=effective_backend,
                        score=total_score,
                        # Reservation_id is rewritten below once we've
                        # confirmed ``best`` won't be replaced by a later
                        # iteration. Pydantic's ``frozen=True`` makes the
                        # final placement immutable from this point.
                        reservation_id="<pending>",
                    )
                    best_node = node

            if best is None or best_node is None:
                raise CapacityExhausted(
                    f"no node has capacity for template {manifest.name!r} "
                    f"(task_key={task_key!r})"
                )

            # Allocate the reservation token atomically with the
            # decision so subsequent in-flight ``place()`` calls see
            # this placement's pending contribution.
            reservation_id = uuid.uuid4().hex
            self._pending[reservation_id] = _PendingPlacement(
                node_id=best_node.node_id,
                template_name=manifest.name,
                task_key=task_key,
                effective_resources=candidate_resources,
                container_runtime=container_runtime,
                pinned_cores_reserved=need_pinned,
            )
            return Placement(
                node=best_node,
                backend=effective_backend,
                score=best.score,
                reservation_id=reservation_id,
            )

    def capable_node_ids(
        self,
        *,
        backend: str | None = None,
        container_runtime: str | None = None,
    ) -> frozenset[str]:
        """Node ids that *could* ever serve this (backend, runtime) pair.

        Pure-capability set — the same backend + runtime filter
        :py:meth:`_place_impl` applies before any disk / health / capacity
        gate, with none of those transient gates applied. Used by the raw
        coordinator's re-admit loop (D-AR-2026-07-07-B): when the set of
        create-time-failed nodes covers *every* capable node, there is no
        sibling to steer to, so the coordinator relaxes the exclusion to ∅ and
        lets the request queue on the (shared) pool rather than hard-failing.
        A missing ``supported_runtimes`` on a node defaults to ``["runc"]``, so
        a non-default runtime simply won't match it (never a crash).
        """
        effective_backend = backend or DEFAULT_BACKEND
        capable = [
            n for n in self._nodes
            if effective_backend in n.supported_backends()
        ]
        if container_runtime and container_runtime != "runc":
            capable = [
                n for n in capable
                if container_runtime in _node_supported_runtimes(n)
            ]
        return frozenset(n.node_id for n in capable)

    def commit_placement(self, placement: Placement) -> None:
        """Drop a pending reservation once its sandbox is recorded in state.

        After this returns the placement is "covered" by
        ``state.list_sandboxes()`` and counting it in pending too would
        double up the contribution. Idempotent: a reservation_id that
        was never reserved (or already committed/released) is a no-op,
        so the coordinator can call this without tracking commit state
        separately.
        """
        with self._pending_lock:
            self._pending.pop(placement.reservation_id, None)

    def release_placement(self, placement: Placement) -> None:
        """Drop a pending reservation when the placement is abandoned.

        Called by the coordinator on the ``create_sandbox`` /
        ``insert_sandbox`` failure path, before the placement ever
        reaches ``state.list_sandboxes()``. Same effect as
        :py:meth:`commit_placement` (decrement pending) but the
        semantic difference is worth a separate name — abandonment is
        a recoverable error in the caller, while commit is the happy
        path.
        """
        with self._pending_lock:
            self._pending.pop(placement.reservation_id, None)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _runtime_at_cap(
        self, node_id: str, runtime: str, counts: dict[str, int],
    ) -> bool:
        """Whether ``node_id`` is at/over its cap for ``runtime`` right now.

        ``None`` cap (no nodes.yaml entry for this node+runtime) ⇒ unlimited ⇒
        never at cap, so an uncapped runtime is a no-op for the placement gate.
        """
        cap = self._runtime_caps.get(node_id, {}).get(runtime)
        if cap is None:
            return False
        return counts.get(node_id, 0) >= cap

    def _runtime_count_with_pending(self, runtime: str) -> dict[str, int]:
        """Per-node count of running + in-flight raw sessions of ``runtime``.

        Running sessions come from the raw-session load provider; in-flight ones
        from ``_pending`` — so a burst of concurrent :py:meth:`place` calls can't
        over-place past a runtime cap before any becomes a running session (the
        same race ``_pending`` already solves for cpu/mem). MUST be called under
        ``_pending_lock`` (it reads ``_pending``); the placement gate does.
        Managed StateStore sandboxes are not folded in — sysbox (the runtime this
        guards) is a raw-container-only runtime, so a managed sandbox never
        carries it and counting them would add noise without changing the count.
        """
        counts: dict[str, int] = defaultdict(int)
        for raw in self._raw_session_provider():
            if raw.container_runtime == runtime:
                counts[raw.node_id] += 1
        for pending in self._pending.values():
            if pending.container_runtime == runtime:
                counts[pending.node_id] += 1
        return dict(counts)

    def _pending_pinned_cores_by_node(self) -> dict[str, int]:
        """P6 step-4b (§8.12) — per-node sum of in-flight REQUIRED pinned-core
        reservations (``_pending``). Subtracted from the node's last-reported
        ``pinned_cpus_free`` in the placement predicate so concurrent REQUIRED
        ``place()`` calls can't each see the same free cores and over-commit a
        node's pinnable pool before any becomes a running (heartbeat-reflected)
        pin. MUST be called under ``_pending_lock`` (it reads ``_pending``)."""
        counts: dict[str, int] = defaultdict(int)
        for pending in self._pending.values():
            if pending.pinned_cores_reserved:
                counts[pending.node_id] += pending.pinned_cores_reserved
        return dict(counts)

    def _load_with_pending(
        self, *, task_key: str | None
    ) -> dict[str, tuple[list[tuple[str, ResourceSpec]], int]]:
        """Like :py:meth:`_gather_cluster_load` but folds in-flight
        reservations from :attr:`_pending` on top of the state snapshot.

        Caller must hold ``self._pending_lock``. ``_gather_cluster_load``
        is left as the state-only view for tests / inspection that
        don't need pending semantics.

        Returns ``{node_id: ([(template_name, effective_resources)],
        task_count)}``. Each entry is one running (or pending) sandbox's
        per-task resource snapshot — for Pattern A this is the
        resolver overlay rather than the outer manifest's defaults.
        """
        load = self._gather_cluster_load(task_key=task_key)
        # ``load`` keys are the currently registered nodes; fold pending
        # reservations into matching node entries. A reservation that
        # references a node which has since gone away is silently
        # dropped — the placement's release path will eventually clean
        # it up, and meanwhile we don't want to attribute load to a
        # node that isn't in the scheduler.
        for pending in self._pending.values():
            if pending.node_id not in load:
                continue
            running, task_count = load[pending.node_id]
            running = [*running, (pending.template_name, pending.effective_resources)]
            if task_key is not None and pending.task_key == task_key:
                task_count += 1
            load[pending.node_id] = (running, task_count)
        return load

    def _gather_cluster_load(
        self, *, task_key: str | None
    ) -> dict[str, tuple[list[tuple[str, ResourceSpec]], int]]:
        """Return ``{node_id: ([(template_name, effective_resources)],
        task_count)}``.

        Sandboxes with ``status='destroyed'`` are excluded — capacity is
        released only on node-confirmed destroy (invariant 2). Each
        running sandbox contributes one entry; ``effective_resources``
        comes from the sandbox record's snapshot when present (Pattern
        A overlay), falling back to the catalog manifest's resources
        when ``effective_resources_json`` is ``None`` (Simple / Pattern
        B / pre-9b rows).
        """
        per_node_running: dict[str, list[tuple[str, ResourceSpec]]] = defaultdict(list)
        per_node_task_count: dict[str, int] = defaultdict(int)
        sandboxes = self._state.list_sandboxes()
        rollout_task_keys: dict[str, str | None] = {}
        if task_key is not None:
            for r in self._state.list_rollouts():
                rollout_task_keys[r.rollout_id] = r.task_key

        # Cache catalog manifests so the fallback lookup doesn't
        # re-query on every sandbox.
        catalog_manifests = {m.name: m for m in self._catalog.list()}

        for sb in sandboxes:
            if sb.status == "destroyed":
                continue
            resources = self._effective_resources_for(sb, catalog_manifests)
            per_node_running[sb.node_id].append((sb.template, resources))
            if (
                task_key is not None
                and sb.rollout_id is not None
                and rollout_task_keys.get(sb.rollout_id) == task_key
            ):
                per_node_task_count[sb.node_id] += 1

        # Second load source: active raw-container sessions. Raw containers
        # don't pass through ``state.insert_sandbox`` (they have their own
        # ``RawRolloutRecord`` flow), so the StateStore loop above wouldn't
        # see them. Fold them in here so a node hosting N raw containers
        # looks identical to one hosting N managed sandboxes from the
        # capacity gate's perspective. ``_pending`` already covers the
        # in-flight window between ``place()`` and the session being
        # registered (see ``RawContainerCoordinator.acquire`` for the
        # commit-after-register ordering).
        for raw in self._raw_session_provider():
            per_node_running[raw.node_id].append(
                (raw.template_name, raw.effective_resources)
            )
            if task_key is not None and raw.task_key == task_key:
                per_node_task_count[raw.node_id] += 1

        result: dict[str, tuple[list[tuple[str, ResourceSpec]], int]] = {}
        for node in self._nodes:
            running = list(per_node_running.get(node.node_id, ()))
            task_count = per_node_task_count.get(node.node_id, 0)
            result[node.node_id] = (running, task_count)
        return result

    @staticmethod
    def _effective_resources_for(
        sb: SandboxRecord,
        catalog_manifests: dict[str, TemplateManifest],
    ) -> ResourceSpec:
        # Prefer the per-sandbox snapshot — that's the post-resolver
        # overlay value the coordinator wrote at placement time.
        if sb.effective_resources_json:
            return ResourceSpec.model_validate_json(sb.effective_resources_json)
        # Fallback for pre-9b rows or Simple / Pattern B sandboxes:
        # use the outer manifest's resources from the catalog. If the
        # template was unregistered between insert and now, charge a
        # zero-resource sandbox — the alternative is crashing the
        # placement loop.
        m = catalog_manifests.get(sb.template)
        if m is not None:
            return m.resources
        return ResourceSpec(
            cpu_request=0.0, cpu_limit=0.0,
            mem_request_bytes=0, mem_limit_bytes=0,
            disk_request_bytes=0,
        )


__all__ = ["CapacityExhausted", "Placement", "Scheduler"]
