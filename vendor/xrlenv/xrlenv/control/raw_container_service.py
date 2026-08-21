"""P1.7.A.1 + P1.7.B.2 — Control-plane fan-out for raw-container sessions.

Sits between the consumer-facing ``RolloutControlServicer`` (which
serves the AcquireContainer / ContainerExec / DestroyContainer
RPCs) and the per-node ``NodeTransport`` (which talks the spec-21
wire to the chosen node). Tracks ``rollout_id → (node, container_id,
container_name, image)`` so subsequent ContainerExec / Destroy can
route to the right node.

State is **in-memory only** for the per-rollout session map.
Case 2/3 evaluations are short-lived (seconds-to-minutes per
instance), harnesses handle their own retry, and durable
session state via StateStore integration is deferred: a
control-plane restart mid-evaluation surfaces as a connection
error to the harness, which re-acquires.

P1.7.B.2 (2026-05-07): the previous first-available ``_pick_node``
stub is replaced by the standard placement flow — same
``Scheduler.place(image_present=..., preferred_home_node=...)``
case-1 uses, with the per-placement ``query_image`` fan-out from
:func:`xrlenv.control.image_presence.query_image_presence`. Raw
acquires now benefit from image-affinity scoring + (when an
operator has run ``xrlenv images plan``) FFD-recorded
``preferred_home`` rows from StateStore. See
``notes/p1-7-b-2-image-affinity-plan.md`` for the design doc.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import logging
import math
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from xrlenv.backends.base import CpuIsolation, ResourceSpec, RuntimeLimits
from xrlenv.backends.egress import EgressAllowlist
from xrlenv.control.admission import DEFAULT_QUEUE_TIMEOUT_S

if TYPE_CHECKING:
    from xrlenv.control.admission import AdmissionQueue
    from xrlenv.control.kwargs_policy import KwargsPolicy
    from xrlenv.control.state import RawRolloutRecord, RawRolloutStatus
from xrlenv.compat.metadata import (
    LABEL_ARTIFACT_PATH,
    LABEL_DISPLAYED_NAME,
    LABEL_FLEET_CPU_REQUEST,
    LABEL_FLEET_ID,
    LABEL_FLEET_MEM_REQUEST,
    LABEL_GROUP_ID,
)
from xrlenv.control.compose_policy import vet_compose_project
from xrlenv.control.compose_prepare import (
    prepare_compose,
    subnet_claims,
    subnets_overlap,
)
from xrlenv.control.image_planner import (
    ImageToPlace,
    NodeBudget,
    NodeId,
    plan_opportunistic_placements,
)
from xrlenv.control.image_presence import query_image_presence
from xrlenv.control.node_transport import NodeTransport
from xrlenv.control.registry_resolver import RegistryDigestResolver
from xrlenv.control.scheduler import RawSessionLoad
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.errors import (
    AuthDenied,
    BackendCapabilityMissing,
    CapacityExhausted,
    FleetOverBudget,
    ImageMissingOnNode,
    ImagePullFailed,
    MountDenied,
    NodeCommandTimeout,
    NodeLost,
    PinCapacityExhausted,
    XRLEnvError,
)
from xrlenv.types import TerminateRawGroupReport

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImagePlanResult:
    """One row of ``plan_image_distribution``'s result.

    ``status`` is one of ``"placed"`` (FFD found a preferred_home with
    enough budget), ``"deferred"`` (no node had room; lazy build on
    first acquire), or ``"failed"`` (eager prefetch crashed).
    ``preferred_home_node`` is empty when status=="deferred" with
    no node available.
    """

    image_ref: str
    preferred_home_node: str
    status: str
    error: str | None = None


# Default resource budget for synthetic raw-container manifests. Modest
# enough to fit a node with reasonable headroom; the consumer's harness
# is what actually drives container memory / CPU usage. The scheduler
# uses this only for capacity gating; oversized values would falsely
# reject capable nodes.
#
# P3(a) integer-core admission (cluster-resource-isolation-plan): raw
# grading containers are cpuset-pinned by default (P2) — each consumes
# ``ceil(cpu_limit)`` *whole* host cores from the node core ledger. The
# scheduler packs by ``cpu_request``, so ``cpu_request`` is set to
# ``ceil(cpu_limit)`` (an integer count of cores) — never the fractional
# request — so the estimator reserves exactly what the ledger will pin
# and can't admit more pinned containers than a node has cores. For the
# default 2.0-CPU container this is 2 == cpu_limit; the rounding only
# bites on a fractional harness override (see _effective_raw_resources).
_DEFAULT_RAW_RESOURCES = ResourceSpec(
    cpu_request=2.0,    # P3(a): ceil(cpu_limit) — integer-core admission
    cpu_limit=2.0,
    mem_request_bytes=512 * 1024 * 1024,    # 512 MiB
    mem_limit_bytes=4 * 1024 * 1024 * 1024,   # 4 GiB
    disk_request_bytes=2 * 1024 * 1024 * 1024,  # 2 GiB
)


@dataclass(frozen=True)
class RawContainerSession:
    """In-memory record of one acquired raw container.

    Held by ``RawContainerCoordinator._sessions`` keyed by
    ``rollout_id``. ``node`` is the live :class:`NodeTransport`
    handle the coordinator dispatches subsequent
    exec/destroy commands through; if the node disconnects,
    subsequent calls fail with the gRPC connection error the
    transport surfaces (no automatic re-routing in P1.7.A.1).

    ``task_key`` is stored so :py:meth:`RawContainerCoordinator.iter_load_entries`
    can emit it to the scheduler — the scheduler's ``max_runs_per_task``
    accounting needs to know which raw containers belong to which task
    on each node, the same way it inspects ``RolloutRecord.task_key``
    for managed sandboxes.
    """

    rollout_id: str
    node: NodeTransport
    node_id: str
    container_id: str
    container_name: str
    image: str
    created_at: _dt.datetime
    task_key: str | None = None
    fleet_id: str | None = None
    """Fleet-reservation identity (phase 1, opt-in). ``None`` (default)
    means this container is *not* part of a fleet — the entire fleet code
    path is gated on ``fleet_id is not None``, so a default session behaves
    exactly as before. When set (parsed from the ``xrlenv.fleet_id`` label),
    this session is a **member** of the fleet reservation keyed by this id:
    :py:meth:`RawContainerCoordinator.iter_load_entries` suppresses its own
    cpu/mem charge (the fleet's footprint entry stands in for it) and emits
    only its disk footprint. ``fleet_id`` is a THIRD identity axis, distinct
    from ``task_key`` (fairness) and the container's own identity — see spec
    00 glossary + ``notes/fleet-reservation-r1-load-accounting.md``."""
    effective_resources: ResourceSpec = _DEFAULT_RAW_RESOURCES
    """P0a — the effective ResourceSpec this session was placed with
    (the harness CPU/memory request merged with the raw-container
    default, or the default itself). Stored so
    :py:meth:`RawContainerCoordinator.iter_load_entries` charges the
    scheduler's steady-state load accounting the *same* footprint the
    placement decision used — a harness asking for more than the
    default must not be under-counted for subsequent placements."""
    queue_wait_s: float = 0.0
    """Issue #18 (Ask #1) — seconds this acquire spent parked in the
    admission queue before a node was assigned. ``0.0`` on the fast
    path (capacity available immediately) and whenever the coordinator
    runs without an admission queue wired. Surfaced back to the
    consumer so they can right-size their concurrency."""
    container_runtime: str | None = None
    """OCI runtime this session runs under (``sysbox-runc`` / ``runc`` / None).
    Fed to the scheduler's per-node runtime-concurrency cap via
    :py:meth:`RawContainerCoordinator.iter_load_entries` so it can count
    concurrent sysbox containers per node (notes/design-per-node-runtime-
    concurrency-cap.md). ``None`` is the ordinary runc path."""
    deadline_at: float = 0.0
    """Issue #18 — wall-clock epoch-seconds after which this session
    is force-reaped. A raw container otherwise lives until the
    consumer explicitly destroys it; if the consumer process dies
    mid-rollout the container, its capacity reservation, and its
    ``raw_rollouts`` row would leak forever. The raw-GC reconciler
    sweeps sessions past this deadline. ``0.0`` means "no deadline"
    (test-constructed sessions only — ``acquire`` always sets a real
    value: consumer-supplied ``session_deadline_s`` or the
    coordinator's default cap)."""
    compose_project_name: str | None = None
    """P1.7.C.2 — in-memory-only marker (no ``state.py`` migration): when set,
    this ``main`` session backs a multi-service compose PROJECT, not a lone
    container. ``effective_resources`` carries the whole-stack footprint (so
    ``iter_load_entries`` charges the project once via the normal session loop),
    and teardown routes to the node's strict ``destroy_compose_project`` (down the
    WHOLE project) instead of a single ``destroy_container``. The project's member
    ids + subnet claims live in :class:`RawContainerCoordinator._compose_projects`
    (a session can't hold them). ``None`` = an ordinary single-container session,
    byte-for-byte unchanged."""


@dataclass(frozen=True)
class _ComposeProjectRecord:
    """P1.7.C.2 — the project-level metadata a :class:`RawContainerSession` can't
    hold. Held in :class:`RawContainerCoordinator._compose_projects` keyed by the
    project's ``rollout_id`` (1:1 with its ``main`` session). ``node_id`` +
    ``project_name`` are what a teardown needs to down the whole project;
    ``service_container_ids`` are the members (main + sidecars); ``subnet_claims``
    feed the node-exclusive anti-affinity (§5 / 3b). The whole-stack footprint
    lives on the session's ``effective_resources`` (the capacity charge), not here.
    """

    project_name: str
    node_id: str
    service_container_ids: dict[str, str]
    subnet_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawComposeAcquireResult:
    """Wire-friendly result of ``acquire_compose_project`` returned to the consumer
    via the ``AcquireComposeProject`` RPC (the compose analog of
    :class:`~xrlenv.control.service.RawAcquireResult`). ``main_container_id`` is the
    full docker id the consumer uses for exec/archive against the ``main`` service;
    ``service_container_ids`` is the whole-project member map."""

    rollout_id: str
    node_id: str
    main_container_id: str
    main_container_name: str
    project_name: str
    service_container_ids: dict[str, str]
    queue_wait_s: float = 0.0


def _sanitize_compose_project_name(name: str) -> str:
    """Docker Compose project names must be lowercase and match
    ``[a-z0-9][a-z0-9_-]*``. Lowercase, replace anything else with ``-``, and
    prefix a digit-leading name so the result is always valid."""
    slug = re.sub(r"[^a-z0-9_-]", "-", name.lower())
    if not slug or not re.match(r"[a-z0-9]", slug):
        slug = f"x{slug}"
    return slug


def _compose_main_image(compose: dict[str, Any], main_service: str) -> str | None:
    """The ``main`` service's declared ``image:`` (the tag to digest-resolve), or
    ``None`` when absent."""
    services = compose.get("services")
    if isinstance(services, dict):
        svc = services.get(main_service)
        if isinstance(svc, dict) and svc.get("image"):
            return str(svc["image"])
    return None


@dataclass
class FleetReservation:
    """One open fleet reservation (phase 1, opt-in — spec 03/10/21).

    A *fleet* is a multi-container task the consumer declares up front so
    the control plane reserves its whole peak footprint on a **single node**
    at once, rather than admitting each container greedily and starving the
    heavier companions. Held in-memory in
    :class:`RawContainerCoordinator._fleets`, keyed by ``fleet_id``; one row
    per open fleet.

    **This table is the ONLY place that knows "what does the fleet reserve"**
    (``footprint``); ``RawContainerSession.fleet_id`` is the only place that
    knows "is this container in a fleet". :py:meth:`iter_load_entries` reads
    both. See ``notes/fleet-reservation-r1-load-accounting.md``.

    Core is generic: it never learns *why* a fleet has its footprint — the
    consumer declares the peak via labels and core reserves it verbatim.
    """

    fleet_id: str
    """Reservation identity — the THIRD identity axis (distinct from
    ``task_key`` fairness and a container's own identity). Parsed from the
    ``xrlenv.fleet_id`` label."""
    node_id: str
    """Single-node pin. The whole footprint is reserved here; companions
    land here too (MVP: no cross-node splitting, no relocation)."""
    footprint: ResourceSpec
    """The declared PEAK — ``cpu_request`` + ``mem_request_bytes`` only in
    v1 (``disk_request_bytes`` is structurally 0 and never read on the fleet
    path; disk stays per-container via member disk-only load entries). This
    is the single load entry the whole fleet contributes for cpu/mem, no
    matter how many members are up."""
    members: dict[str, ResourceSpec] = field(default_factory=dict)
    """Live members — ``rollout_id → its own effective ResourceSpec``. Serves
    two roles: **ref count** (the reservation releases only when this empties
    on node-confirmed destroy of the LAST member — invariant 2; releasing on
    the first member's destroy would free capacity a surviving member still
    holds) and **budget** (the ``FleetOverBudget`` check sums members' own
    cpu/mem against the footprint). A companion's slot is reserved here under
    the lock at admission — before its container exists — so concurrent
    companions of one fleet can't both pass the budget check. Not read by
    ``iter_load_entries`` (which keys suppression on ``session.fleet_id``)."""
    opened_ts: float = 0.0
    last_acquire_ts: float = 0.0
    """Refreshed on every companion acquire; drives the
    ``fleet_reservation_ttl_s`` reclaim of a fleet whose consumer crashed
    without destroying cleanly (later slice)."""
    task_key: str | None = None
    """Emitted on the fleet's single load entry so the fleet counts as
    **one** unit toward ``(node, task_key)`` ``max_runs_per_task`` fairness —
    members emit no task_key, so they add nothing (spec 03)."""
    owner: str = "default"
    container_runtime: str | None = None
    """§5.3 — the OCI runtime the fleet runs under (e.g. ``sysbox-runc``).
    Set from the opener's request and validated against the pinned node.
    Companions SKIP ``Scheduler.place`` (drawn from this reservation), so a
    scheduler-only runtime filter can't cover them; instead every member
    must match this value and the pinned node must advertise it, enforced at
    companion-acquire time. ``None`` = the ordinary runc fleet, unchanged."""


class _SchedulerProtocol(Protocol):
    """Minimal Scheduler surface this coordinator depends on.

    The actual ``xrlenv.control.scheduler.Scheduler`` satisfies this
    structurally. Declared as a Protocol (rather than depending on
    Scheduler directly) so tests can pass a fake without importing
    the full scheduler stack.
    """

    @property
    def nodes(self) -> list[NodeTransport]: ...

    @property
    def image_aware_placement(self) -> bool: ...

    def place(
        self,
        manifest: TemplateManifest,
        *,
        task_key: str | None = None,
        backend: str | None = None,
        image_present: dict[str, bool] | None = None,
        preferred_home_node: str | None = None,
        reserve: ResourceSpec | None = None,
        exclude_node_ids: frozenset[str] | None = None,
    ) -> Any: ...

    def capable_node_ids(
        self,
        *,
        backend: str | None = None,
        container_runtime: str | None = None,
    ) -> frozenset[str]:
        """D-AR-2026-07-07-B — pure-capability node set for the re-admit
        relax decision (see :func:`_is_retriable_acquire_saturation`)."""
        ...

    def commit_placement(self, placement: Any) -> None: ...

    def release_placement(self, placement: Any) -> None: ...


class _StateStoreProtocol(Protocol):
    """Minimal StateStore surface RawContainerCoordinator depends on.

    Real :class:`xrlenv.control.state.SqliteStateStore` and the
    in-memory variant both expose these. Older test doubles may
    only declare ``find_registered_preferred_home``; the
    persistence path (``record_build_plan`` /
    ``record_assignment``) is reached only by ``plan_image_distribution``,
    which raises a clear error when those attributes are absent.
    """

    def find_registered_preferred_home(
        self, image: str,
    ) -> str | None: ...

    def record_build_plan(
        self, *, plan_id: str, applied_by: str, plan_json: str,
    ) -> Any: ...

    def record_assignment(self, record: Any) -> None: ...

    # Fleet reservation (phase 1). Optional on older test doubles — the
    # coordinator reaches these via ``getattr`` best-effort, so a store
    # lacking them degrades to in-memory-only fleet state.
    def record_fleet_reservation(self, record: Any) -> None: ...

    def touch_fleet_reservation(
        self, fleet_id: str, *, last_acquire_ts: float,
    ) -> None: ...

    def delete_fleet_reservation(self, fleet_id: str) -> None: ...

    def list_fleet_reservations(self) -> list[Any]: ...

    # Compose projects (P1.7.C.2) — same best-effort ``getattr`` treatment.
    def record_compose_project(self, record: Any) -> None: ...

    def delete_compose_project(self, rollout_id: str) -> None: ...

    def list_compose_projects(self) -> list[Any]: ...


# Fleet reservation (phase 1) — how long a persisted reservation row with NO
# live members may sit before the raw-GC reconciler reclaims it. Guards against
# a leaked row (a fleet whose consumer crashed without a clean last-member
# destroy) and, after a control-plane restart, a fleet whose pinned node never
# reconnects. Measured from ``last_acquire_ts`` and only applied once the
# reconciler is past its re-adoption grace, so a briefly-disconnected node's
# live fleet is never reclaimed out from under it. Env-tunable.
FLEET_RESERVATION_TTL_DEFAULT_S: float = float(
    os.environ.get("XRLENV_FLEET_RESERVATION_TTL_S", "600"),
)


# Issue #18 — default cap on a raw session's wall-clock lifetime
# (4 h). A raw container otherwise lives until the consumer calls
# ``destroy``; if the consumer process dies mid-rollout, nothing
# reaps the container, its capacity reservation, or its
# ``raw_rollouts`` row. 4 h sits well above any single SWE-bench /
# terminal-bench grading task, so it's a leak backstop rather than a
# limit consumers normally hit. Consumers with genuinely longer work
# raise it per-acquire via ``session_deadline_s``.
RAW_SESSION_DEADLINE_DEFAULT_S: float = 4 * 60 * 60.0

# Consumer-liveness reaper. How long a raw session may go with NO liveness
# signal — neither a session-scoped RPC (implicit) nor an explicit heartbeat —
# before the raw-GC reconciler reaps it. Distinct from the 4 h deadline above:
# the deadline caps even a healthy long-running job; this reaps a session whose
# *consumer went away*. Only applies to sessions that (a) heartbeated at least
# once and (b) have no RPC in flight (see ``liveness_reap_candidates``), so a
# non-heartbeating consumer or one blocked in a long exec is never touched.
# TTL is ~3-4x the SDK heartbeat interval so a couple of dropped beats don't
# false-reap. Env-tunable.
RAW_LIVENESS_TTL_DEFAULT_S: float = float(
    os.environ.get("XRLENV_RAW_LIVENESS_TTL_S", "120"),
)


# Synthetic env-adapter / reward decls. The scheduler reads
# ``manifest.env_adapter`` and ``manifest.reward`` only via
# ``manifest.name`` for logging / metrics labelling — the actual
# rollout doesn't run an EnvAdapter or a reward command (raw
# acquires are case 2/3, harness-driven). These placeholders
# satisfy the schema so the synthetic manifest validates.
_RAW_ENV_ADAPTER = EnvAdapterDecl(
    module="xrlenv.envs.noop",
    class_name="NoopEnvAdapter",
)
_RAW_REWARD_CONTRACT = RewardContract(
    mode="external_final",
    cmd=("/bin/true",),
)


def _acquire_cancel_reason(
    exc: BaseException, *, queue_wait_s: float, placed: bool,
) -> str:
    """Factual ``error`` reason for a cleanly-cancelled raw acquire.

    Called only when the acquire was cancelled AND torn down without
    error (status ``cancelled``). A bare ``asyncio.CancelledError``
    stringifies to ``""`` — the old code recorded the useless
    ``"CancelledError: "``. We record the teardown facts (how long it
    waited, whether it had been placed) and the canceller's own message
    if one was set (``task.cancel("...")`` / a deadline-watcher reason).

    We deliberately do NOT speculate about *who* issued the cancel: a
    ``CancelledError`` carries no origin, and the outcome is identical
    whoever asked — what we can state truthfully is that it was cancelled
    and unwound cleanly. The origin, when it matters, lives in the
    canceller's own message (surfaced here) or the client's logs.
    """
    where = "after placement" if placed else "while queued (never placed)"
    base = (
        f"acquire cancelled {where} after {queue_wait_s:.1f}s wait; "
        "unwound cleanly"
    )
    detail = str(exc).strip()
    return f"{base} — {detail}" if detail else base


def _acquire_capacity_rejected_reason(exc: BaseException) -> str:
    """Factual ``error`` reason for an acquire the scheduler declined to
    place within ``queue_timeout_s`` (:class:`CapacityExhausted`).

    Sealed as ``capacity_rejected`` (NOT ``failed``): the pool was at
    capacity the whole wait, so the rollout's work never ran and nothing
    errored — this is backpressure/pacing. We keep the original
    ``CapacityExhausted`` text (it names the ``queue_timeout_s`` that
    expired) and append the two operator levers, stated *generically*: a
    control-plane primitive can't name a specific consumer's env var
    (three-plane split), so the harbor-specific knob
    (``XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S``) lives in that plug-in's
    docs + the admin hint, not here.
    """
    detail = str(exc).strip()
    return (
        "capacity_rejected: the control plane declined to place this acquire "
        "within queue_timeout_s (pool at capacity the whole wait). This is "
        "backpressure, not a rollout failure — the attempt never ran. To wait "
        "longer for a slot raise queue_timeout_s on the acquire; otherwise have "
        "the caller retry (a fresh acquire re-queues from scratch)."
        + (f" [{detail}]" if detail else "")
    )


def _parse_metadata_labels(
    labels: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    """Pull ``xrlenv.rollout.artifact_path`` /
    ``xrlenv.rollout.displayed_name`` off an outgoing labels dict.

    The drop-in's ``create_container`` override (cluster mode)
    injects these from the consumer's ``xrlenv.rollout_metadata(...)``
    contextvar. The coordinator parses them off ``labels`` here
    and persists them to the typed columns on
    ``RawRolloutRecord``. Returns ``(artifact_path, displayed_name)``;
    either value may be ``None``.
    """
    if not labels:
        return None, None
    return (
        labels.get(LABEL_ARTIFACT_PATH),
        labels.get(LABEL_DISPLAYED_NAME),
    )


def _parse_group_id_label(labels: dict[str, str] | None) -> str | None:
    """Pull the ``xrlenv.group_id`` docker label off an outgoing
    labels dict for persistence on ``RawRolloutRecord.group_id``.

    Unlike the metadata labels, ``xrlenv.group_id`` is purely
    operator-supplied (no xrlenv-side emitter). Returns ``None``
    when the label is absent.
    """
    if not labels:
        return None
    return labels.get(LABEL_GROUP_ID)


def _parse_fleet_labels(
    labels: dict[str, str] | None,
) -> tuple[str | None, ResourceSpec | None]:
    """Parse the generic fleet-declaration labels off an outgoing labels
    dict — the consumer→CP hop (spec 21 §Fleet acquire fields).

    Returns ``(fleet_id, footprint)``:

    - ``(None, None)`` — no ``xrlenv.fleet_id`` label ⇒ **not a fleet
      acquire**. This is the common path; the caller takes the unchanged
      per-container flow. Zero cost for every non-fleet acquire.
    - ``(fleet_id, footprint)`` — a **fleet-opening** acquire: both
      ``xrlenv.fleet_id`` *and* the two footprint labels
      (``xrlenv.fleet_cpu_request`` + ``xrlenv.fleet_mem_request``) present.
      ``footprint`` is a :class:`ResourceSpec` carrying **cpu + mem only**
      (v1 — disk is per-container, never in the footprint; see
      ``notes/fleet-reservation-r1-load-accounting.md``).
    - ``(fleet_id, None)`` — a **companion** acquire: ``xrlenv.fleet_id``
      present but no footprint labels ⇒ this container joins an
      already-open fleet; the footprint was declared by the fleet-opening
      acquire, not re-declared here.

    The two footprint labels are the *peak task-level* numbers the consumer
    declares, NOT any single container's own ``--cpus`` / ``--memory`` (those
    still flow via ``cpu_limit`` / ``mem_limit_bytes`` as before). Core never
    interprets *why* the footprint has its value — it reserves it verbatim.

    A malformed footprint label (non-numeric, or fleet_id present with only
    one of the two footprint labels) raises :class:`XRLEnvError`
    (category=user) — a declaration bug should fail loud at acquire, not
    silently degrade to per-container admission and re-introduce starvation.
    """
    if not labels:
        return None, None
    fleet_id = labels.get(LABEL_FLEET_ID)
    if fleet_id is None:
        return None, None
    cpu_raw = labels.get(LABEL_FLEET_CPU_REQUEST)
    mem_raw = labels.get(LABEL_FLEET_MEM_REQUEST)
    if cpu_raw is None and mem_raw is None:
        # Companion: fleet_id only, footprint declared by the opener.
        return fleet_id, None
    if cpu_raw is None or mem_raw is None:
        raise XRLEnvError(
            f"fleet declaration for {fleet_id!r} is incomplete: a "
            f"fleet-opening acquire must set BOTH {LABEL_FLEET_CPU_REQUEST} "
            f"and {LABEL_FLEET_MEM_REQUEST} (got cpu={cpu_raw!r}, "
            f"mem={mem_raw!r}). A companion sets neither.",
        )
    try:
        cpu_request = float(cpu_raw)
        mem_request_bytes = int(mem_raw)
    except (TypeError, ValueError) as exc:
        raise XRLEnvError(
            f"fleet footprint for {fleet_id!r} is malformed: "
            f"{LABEL_FLEET_CPU_REQUEST}={cpu_raw!r} (want a float), "
            f"{LABEL_FLEET_MEM_REQUEST}={mem_raw!r} (want an int, bytes). "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if cpu_request <= 0 or mem_request_bytes <= 0:
        raise XRLEnvError(
            f"fleet footprint for {fleet_id!r} must be positive: got "
            f"cpu_request={cpu_request}, mem_request_bytes={mem_request_bytes}.",
        )
    # v1: footprint is cpu + mem only. cpu_limit mirrors cpu_request and
    # mem_limit mirrors mem_request purely to satisfy the ResourceSpec schema
    # — the scheduler's load accounting reads only the two *_request fields.
    # disk_request_bytes=0 is structural (see the disk decision in the R1
    # note); disk reaches the scheduler through per-member disk-only entries.
    footprint = ResourceSpec(
        cpu_request=cpu_request,
        cpu_limit=cpu_request,
        mem_request_bytes=mem_request_bytes,
        mem_limit_bytes=mem_request_bytes,
        disk_request_bytes=0,
    )
    return fleet_id, footprint


def _fleet_member_disk_only(spec: ResourceSpec) -> ResourceSpec:
    """A :class:`ResourceSpec` charging ONLY this container's disk (cpu +
    mem zeroed).

    Used for a fleet member in :py:meth:`RawContainerCoordinator.iter_load_entries`:
    its cpu + mem are covered exactly once by the fleet's footprint entry, so
    charging them again per-member would double-count. Its **disk**, however,
    stays per-container — the one documented exception to member suppression
    (spec 10 §Fleet reservation accounting). Zeroing cpu/mem while keeping the
    real ``disk_request_bytes`` is what prevents a silent disk under-count.
    """
    return ResourceSpec(
        cpu_request=0.0,
        cpu_limit=0.0,
        mem_request_bytes=0,
        mem_limit_bytes=0,
        disk_request_bytes=spec.disk_request_bytes,
    )


def _effective_raw_resources(
    *,
    cpu_limit: float | None = None,
    mem_limit_bytes: int | None = None,
    cpu_isolation: CpuIsolation = CpuIsolation.OFF,
) -> ResourceSpec:
    """Derive the effective ``ResourceSpec`` for a raw acquire (P0a).

    Starts from :data:`_DEFAULT_RAW_RESOURCES` and applies the
    harness's CPU/memory request (extracted by the docker-py drop-in
    from ``host_config``) as an override. A harness asking for *less*
    yields a smaller effective spec (frees scheduler capacity); asking
    for *more* yields a larger one (the scheduler then places only on a
    node that can satisfy it). Either way the spec is derived **once**,
    here, and the same object feeds capacity + placement.

    P3(a) integer-core admission: a raw container is cpuset-pinned (P2)
    to ``ceil(cpu_limit)`` *whole* cores. ``cpu_request`` is therefore
    set to ``ceil(cpu_limit)`` — the integer core count the node ledger
    will actually reserve — so the estimator (which packs by
    ``cpu_request``) can't admit more pinned containers than a node has
    cores. ``cpu_limit`` keeps the exact (possibly fractional) value for
    the runtime CFS quota. ``mem_request == mem_limit`` for memory.

    P6: the ingress-derived ``cpu_isolation`` mode is stamped here (the same
    derive-once point), so it rides ``ResourceSpec.cpu_isolation`` to placement
    and the node command; the node then reads it (see raw_container.py).
    """
    overrides: dict[str, float | int | CpuIsolation] = {}
    if cpu_limit is not None and cpu_limit > 0:
        # cpu_request is the integer core count (matches the cpuset
        # the node pins); cpu_limit stays exact for the CFS quota.
        overrides["cpu_request"] = float(math.ceil(cpu_limit))
        overrides["cpu_limit"] = cpu_limit
    if mem_limit_bytes is not None and mem_limit_bytes > 0:
        overrides["mem_request_bytes"] = mem_limit_bytes
        overrides["mem_limit_bytes"] = mem_limit_bytes
    if cpu_isolation is not CpuIsolation.OFF:
        overrides["cpu_isolation"] = cpu_isolation
    if not overrides:
        return _DEFAULT_RAW_RESOURCES
    return _DEFAULT_RAW_RESOURCES.model_copy(update=overrides)


def _synthetic_manifest_for_raw(
    image: str, resources: ResourceSpec = _DEFAULT_RAW_RESOURCES,
) -> TemplateManifest:
    """Build the minimum-viable ``TemplateManifest`` the scheduler
    reads when scoring raw-container acquires.

    The scheduler reads (in :func:`xrlenv.control.scheduler.Scheduler.place`):
    ``manifest.image``, ``manifest.resources`` (capacity gate +
    score), ``manifest.name`` (logging / metrics labels). Other
    fields aren't consulted at placement; the schema requires them
    so we provide dataclass defaults.

    ``resources`` is the effective ``ResourceSpec`` (P0a) — the
    raw-container default, or a harness-overridden spec from
    :func:`_effective_raw_resources`.

    The resulting manifest is **never registered in the catalog**
    and never drives a real rollout — it exists for the duration
    of one placement decision.
    """
    return TemplateManifest(
        name=f"raw-container/{image}",
        version="raw",
        digest="raw-synthetic-no-pin",
        image=image,
        resources=resources,
        env_adapter=_RAW_ENV_ADAPTER,
        reward=_RAW_REWARD_CONTRACT,
        # registry_digest is the default; the scheduler doesn't
        # read pin_mode (the catalog does, and we bypass it).
    )


# ── D-AR-2026-07-07-B — control-plane re-admit on node-saturation create fault ─
#
# The node-local create-cap+retry (raw_container.py ``_create_with_retry``)
# absorbs a *transient* create burst on one node. It cannot help when a single
# node is *sustainedly* overloaded — the sysbox-fs FUSE wedges, every create
# on it returns ``pre-register with sysbox-fs: DeadlineExceeded``, and the node
# stays hot for minutes. The right recovery is to steer the acquire to a
# *sibling* node. The AdmissionQueue only re-queues on a *proactive*
# ``CapacityExhausted`` (the scheduler declined to place); it never sees a
# *reactive* create-time saturation 5xx, because that surfaces from
# ``node.acquire_container`` AFTER placement. This re-admit loop closes that
# gap: on a retriable create-time saturation fault it releases the placement,
# excludes the failed node, and re-admits.
_CP_REQUEUE_MAX: int = 3
"""Max re-admits after the first attempt (so up to 1 + 3 = 4 create attempts
across distinct nodes). A backstop against a whole-fleet saturation storm
looping forever; the total wall-clock cap below is the primary bound."""
_CP_REQUEUE_TOTAL_CAP_S: float = 180.0
"""Total wall-clock cap on the *re-admit phase* — measured from the FIRST
create-time saturation failure, NOT from the start of acquire (so a normal
first-attempt queue wait keeps its full ``queue_timeout_s``, 24 h default).
Once engaged it clamps BOTH each attempt's admission-wait ``timeout_s`` and its
node-wire ``acquire_timeout_s`` to the remaining budget, so a queued ``_Waiter``
carrying the failed-node exclusion cannot outlast the cap on the surviving
pool. For the target case the node's own ~31 s create-retry makes each attempt
fast, so the cap rarely binds; it caps the pathological all-timeout /
long-queue-wait case."""

_MIN_CREATE_DEADLINE_S: float = 180.0
"""Floor for the node-wire ``acquire_timeout_s`` (the container CREATE deadline)
on a re-admit attempt. The re-admit cap above bounds how long we keep RE-TRYING
placement — but once a node is chosen the create is *committed* and must not be
starved. A long sysbox queue-wait can leave only a few tens of seconds of cap,
and a sysbox container create under concurrent load routinely needs more than
that (2026-07-08 conc-32: tw_650591 / tw_709166 hit ``NodeCommandTimeout`` at
25-30 s of leftover cap mid-create, then failed the trial). So the clamp may
shrink the admission-WAIT to the remaining cap, but the committed create keeps at
least this floor — still far under harbor's 600 s env-build ceiling. Never raised
ABOVE an explicit caller ``acquire_timeout_s`` (a caller asking for a tight create
deadline is honoured); it only undoes the re-admit budget's over-reduction."""

# Substrings that mark a create failure as *node saturation* (worth re-admitting
# on a sibling), reconstructed CP-side from the node reply as
# ``node <id>: remote command <kind>: node-side docker create failed …: <text>``.
# Deliberately NARROW and aligned with the node's own
# ``_is_retryable_create_error``: docker 5xx (docker-py APIError renders any 5xx
# as "<code> Server Error"), gRPC/context DeadlineExceeded, the sysbox-fs
# pre-register stall, and generic "timed out"/"timeout" text. A 4xx
# (name-conflict 409 / no-such-image 404) or a policy denial would fail
# identically on any node — TERMINAL, not listed here — so it propagates.
_SATURATION_MARKERS: tuple[str, ...] = (
    "server error",       # docker-py APIError for any 5xx ("500 Server Error …")
    "deadlineexceeded",
    "deadline exceeded",
    "sysbox-fs",
    "timed out",
    "timeout",
)

# Error types that are ALWAYS terminal for a re-admit regardless of message — a
# retry on a sibling node reproduces them, so never re-admit. ``CapacityExhausted``
# is here because it comes from the admission/scheduler PLACE step (the queue
# already waited ``queue_timeout_s`` / the pool is genuinely full); re-admitting
# with an exclusion cannot help. EXCEPTION: its subclass
# ``PinCapacityExhausted`` (P6 §8.7 node-side REQUIRED pin-or-fail) IS
# node-specific and re-admittable — the classifier below checks it BEFORE this
# tuple so the ``CapacityExhausted`` base membership doesn't swallow it.
_TERMINAL_ACQUIRE_ERRORS: tuple[type[BaseException], ...] = (
    BackendCapabilityMissing,
    ImageMissingOnNode,
    ImagePullFailed,
    CapacityExhausted,
    MountDenied,
    AuthDenied,
    FleetOverBudget,
)


def _is_retriable_acquire_saturation(exc: BaseException) -> bool:
    """True when ``exc`` is a create-time *node-saturation* fault worth
    re-admitting on a sibling node (D-AR-2026-07-07-B).

    - A wire-level :class:`NodeCommandTimeout` is the canonical node-overload /
      wedged-agent signal → retry elsewhere.
    - A specific terminal subclass (:data:`_TERMINAL_ACQUIRE_ERRORS`) never
      re-admits — it would reproduce on any node.
    - A generic remote failure (reconstructed as a bare :class:`XRLEnvError`) is
      retriable only when its message matches a :data:`_SATURATION_MARKERS`
      substring. Everything else — an unknown ``XRLEnvError``, a 4xx
      name-conflict, a non-``XRLEnvError`` — propagates unretried.

    Note the terminal-subclass check runs BEFORE the message match so a
    ``CapacityExhausted("queue_timeout_s … expired")`` (whose text contains
    "timeout") is correctly classified terminal, not retriable.

    P6 step-4c follow-up: node-side REQUIRED pin-or-fail
    (:class:`PinCapacityExhausted`) is NODE-SPECIFIC (a stale-heartbeat / ledger
    race), not global pool exhaustion — re-admit it on a sibling capable node.
    Checked BEFORE ``_TERMINAL_ACQUIRE_ERRORS`` because it subclasses
    ``CapacityExhausted`` (a terminal type). In-process the exception IS a
    ``PinCapacityExhausted``; over the gRPC wire it is reconstructed as a bare
    ``XRLEnvError`` carrying the node's ``error_kind`` (the class name
    "PinCapacityExhausted") in its message — match both.
    """
    if isinstance(exc, NodeCommandTimeout):
        return True
    if isinstance(exc, PinCapacityExhausted) or (
        isinstance(exc, XRLEnvError)
        and "pincapacityexhausted" in str(exc).lower()
    ):
        return True
    if isinstance(exc, _TERMINAL_ACQUIRE_ERRORS):
        return False
    if isinstance(exc, XRLEnvError):
        msg = str(exc).lower()
        return any(marker in msg for marker in _SATURATION_MARKERS)
    return False


class RawContainerCoordinator:
    """Coordinates the lifecycle of raw-container sessions.

    Public surface mirrors the consumer-facing RPCs:

    - :meth:`acquire` — assigns a fresh ``rollout_id``, picks a
      node, sends ``AcquireContainerCommand`` to the node, records
      the session, returns the session object.
    - :meth:`exec` — looks up the session by ``rollout_id``,
      validates the caller's ``container_id`` matches the session's
      (defends against stale handles), dispatches
      ``ContainerExecCommand`` to the session's node.
    - :meth:`destroy` — looks up, dispatches
      ``DestroyContainerCommand``, drops the in-memory session
      regardless of the wire-level outcome (the node-side
      manager is also idempotent on missing-container).
    """

    def __init__(
        self,
        *,
        scheduler: _SchedulerProtocol,
        state: _StateStoreProtocol | None = None,
        kwargs_policy: KwargsPolicy | None = None,
        admission: AdmissionQueue | None = None,
        session_deadline_default_s: float = RAW_SESSION_DEADLINE_DEFAULT_S,
        liveness_ttl_s: float = RAW_LIVENESS_TTL_DEFAULT_S,
        digest_resolver: RegistryDigestResolver | None = None,
        raw_reconnect_capable: bool = True,
    ) -> None:
        self._scheduler = scheduler
        # audit H11 — restart-safety gate. A distributed deployment that opts OUT of the periodic
        # raw-GC reconciler (``gc_reconcile_interval_s=None``) has NO way to inventory a
        # reconnecting node's surviving raw/compose containers, so it CANNOT support restart-safe
        # raw sessions: a node-agent restart would readmit uncharged survivors. Such a deployment
        # is gym/step-only. When wired False (by ``distributed_runtime`` in that config), raw +
        # compose acquires fail loud rather than silently accrue un-reconcilable load. Defaults
        # True so in-process / test coordinators (which drive the reconciler manually or don't
        # exercise reconnect) are unaffected.
        self._raw_reconnect_capable = raw_reconnect_capable
        # Freshness model (Part 2): when wired, every acquire's image ref
        # is resolved registry tag -> content digest before placement /
        # dispatch, so a node materializes exactly the bytes the registry
        # currently serves under that tag (the mutable-tag staleness fix).
        # ``None`` keeps the legacy pass-through (tests, registry-less
        # standalone use) — acquires use the ref verbatim.
        self._digest_resolver = digest_resolver
        # Issue #18 — system default cap on a raw session's wall-clock
        # lifetime, used when the consumer doesn't pass an explicit
        # ``session_deadline_s``. The raw-GC reconciler force-reaps
        # any session past its deadline so an abandoned container
        # (consumer process killed mid-rollout, never called destroy)
        # can't leak its container + capacity + ``raw_rollouts`` row
        # indefinitely. Generous by default — sized well above any
        # single grading task — so it's a safety net, not a guillotine.
        self._session_deadline_default_s = session_deadline_default_s
        # Optional: when present, ``find_registered_preferred_home``
        # contributes the planner's soft-routing bonus (P1.6.g-H2).
        # When absent (older test fixtures), placement still works
        # via image_present + slack scoring alone.
        self._state = state
        # Issue #18 fix #1: when wired, raw acquires queue on
        # ``CapacityExhausted`` instead of erroring synchronously.
        # The historical raw path called ``scheduler.place(...)``
        # directly; under heavy parallel load + disk pressure that
        # surfaced ``RESOURCE_EXHAUSTED`` to consumers and turned a
        # transient saturation into a cascading failure. With
        # admission wired, consumers see a longer wall time under
        # tight capacity instead of an error. ``None`` keeps the
        # legacy direct-place path for tests and pre-admission
        # standalone use.
        self._admission = admission
        # Issue #6: cluster-wide docker-kwarg policy. Authoritative
        # enforcement point — re-validates kwargs the drop-in
        # fast-failed against DEFAULT_POLICY, plus operator-specific
        # tweaks (denied_caps, allowed_devices extensions, Level-2
        # opt-ins) that the drop-in can't see. Defaults to
        # DEFAULT_POLICY when the operator hasn't shipped a
        # nodes.yaml policy section (existing single-node / laptop
        # deployments behave exactly as before).
        from xrlenv.control.kwargs_policy import DEFAULT_POLICY
        self._kwargs_policy = kwargs_policy or DEFAULT_POLICY
        self._sessions: dict[str, RawContainerSession] = {}
        # P1.7.C.2 — multi-service compose projects, keyed by ``rollout_id`` (1:1
        # with the ``main`` session that carries ``compose_project_name``). Holds
        # the member ids + subnet claims a session can't; empty for every
        # non-compose deployment. Populated by ``acquire_compose_project`` and
        # dropped on confirmed ``destroy_compose_project`` teardown.
        self._compose_projects: dict[str, _ComposeProjectRecord] = {}
        # P1.7.C.2 / 3b — IN-FLIGHT compose subnet claims (rollout_id ->
        # (node_id, claims)) from the moment a project is placed on a node until
        # it either registers in ``_compose_projects`` (success) or the acquire
        # fails. Folded into the subnet-conflict set so two concurrent same-subnet
        # acquires don't both snapshot an empty committed table and collide on one
        # node (the audit's in-flight race). Empty for non-compose / DNS-only.
        self._pending_subnet_claims: dict[str, tuple[str, tuple[str, ...]]] = {}
        # Per-subnet placement lock (keyed by the sorted claim set) so two
        # concurrent acquires that pin the SAME subnet serialize their whole
        # exclude→place/admit→reserve-pending section — closing the in-flight race
        # on BOTH the direct and admission paths (a same-subnet acquire can't be
        # admitted onto a node the in-flight sibling just took). Same-subnet
        # projects can't co-locate anyway, so serializing their placement (not
        # their execution — the lock releases before the node ``up``) is free.
        # Different-subnet / DNS-only acquires never contend. Bounded by the
        # number of distinct pinned subnets in the corpus (tiny).
        self._subnet_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        # Fleet reservations (phase 1, opt-in), keyed by ``fleet_id``. Empty
        # for every non-fleet deployment — the whole fleet path is gated on a
        # session/label carrying a ``fleet_id``, so tb2.1 / SWE-bench never
        # populate this and ``iter_load_entries`` stays byte-for-byte. One
        # ``FleetReservation`` per open fleet; the admission flow (later slice)
        # creates and drops rows, the reconcile path (R7) rebuilds them from
        # labels after a control-plane restart.
        self._fleets: dict[str, FleetReservation] = {}
        # fleet_ids whose OPENING acquire is in flight — the placement has been
        # reserved (``_pending`` holds the footprint) but the ``FleetReservation``
        # is not created until the lead container is up. Guards against a second
        # concurrent opener for the same fleet_id double-reserving the footprint
        # (F2). Removed on the opener's success (reservation created) or failure.
        self._fleet_opening: set[str] = set()
        # Issue #18 (audit M1 round 2): rollout_ids whose ``acquire``
        # is in flight — the row has been written ``acquiring`` but no
        # session exists yet (request parked in the admission queue,
        # or mid wire-level ``acquire_container``). The raw-GC SQLite
        # reconciler unions this with ``_sessions`` to decide whether
        # an ``acquiring`` row is a genuine ghost: a fixed stale-age
        # window can't tell a legitimately-queued acquire (consumer
        # raised ``queue_timeout_s`` to 7200s) from a post-restart
        # leftover. Tracking liveness directly removes the time proxy.
        # In-memory by design: after a control-plane restart this set
        # is empty — which is exactly correct, since a restart kills
        # every in-flight ``acquire`` coroutine, so every leftover
        # ``acquiring`` row genuinely IS a ghost.
        self._acquiring_ids: set[str] = set()
        # Issue #18 fix #2: ids that have entered the destroy wire-
        # call but haven't yet been popped from ``_sessions``. Same
        # session row keeps charging ``iter_load_entries`` (capacity
        # accounting) AND ``Scheduler.place`` (image affinity) for
        # the whole teardown, but admin + tests can see the rollout
        # is in the destroy phase.
        #
        # Per-rollout single-flight OWNER map (audit M14): value is an ``asyncio.Future``
        # the owner resolves with the teardown's terminal OUTCOME — ``None`` (node-confirmed /
        # success) or the wire exception object (for a duplicate caller to JOIN and apply its
        # own contract, instead of the old optimistic "return success while the owner may still
        # fail"). Membership (``rid in self._destroying``) still means "a destroy is in flight".
        self._destroying: dict[str, asyncio.Future[BaseException | None]] = {}
        # Consumer-liveness reaper state (bare dicts/sets, like the sets above —
        # never mutate the frozen session). ``_last_seen_at`` is bumped by every
        # session-scoped RPC (implicit heartbeat) AND every explicit heartbeat;
        # ``_inflight_rpcs`` counts open session-scoped RPCs (a session with one
        # in flight is never liveness-reaped — the consumer is connected and
        # waiting); ``_heartbeated`` is the set whose consumer has sent >=1
        # explicit heartbeat (gates fast-reap so non-heartbeating clients fall
        # back to the deadline). All keyed by rollout_id, popped on destroy.
        self._last_seen_at: dict[str, float] = {}
        self._inflight_rpcs: dict[str, int] = {}
        self._heartbeated: set[str] = set()
        self._liveness_ttl_s = liveness_ttl_s
        self._lock = asyncio.Lock()

    def _require_raw_reconnect_capable(self, what: str) -> None:
        """Fail loud (audit H11) if this deployment can't restart-safely carry raw/compose
        sessions — i.e. the periodic raw-GC reconciler is disabled
        (``gc_reconcile_interval_s=None``), so a reconnecting node's surviving containers can't
        be inventoried + re-adopted and would readmit uncharged load. Such a deployment is
        gym/step-only; reject the acquire rather than silently accrue un-reconcilable state."""
        if not self._raw_reconnect_capable:
            raise XRLEnvError(
                f"{what} acquire rejected: this deployment has the periodic raw-GC reconciler "
                "disabled (gc_reconcile_interval_s=None), so it cannot restart-safely inventory "
                "reconnecting-node survivors and is NOT capable of distributed raw/compose "
                "sessions (gym/step only). Set gc_reconcile_interval_s to enable raw/compose.",
            )

    async def acquire(
        self,
        *,
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
        task_key: str | None = None,
        request_id: str | None = None,
        owner_id: str = "default",
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        session_deadline_s: float | None = None,
        cpu_limit: float | None = None,
        mem_limit_bytes: int | None = None,
        cpu_isolation: CpuIsolation = CpuIsolation.OFF,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawContainerSession:
        """Acquire a raw container against the cluster.

        Placement flow (P1.7.B.2):

        1. Per-placement image-presence fan-out via
           :func:`xrlenv.control.image_presence.query_image_presence`.
           Returns a fresh ``{node_id: bool}`` map; the scheduler
           uses it as the image-affinity input.
        2. Optional preferred-home lookup via the StateStore
           (populated by ``xrlenv build apply`` or the new
           ``xrlenv images plan`` operator CLI).
        3. ``Scheduler.place(...)`` scores nodes by
           ``weighted_sum_score(R=2/3, I=1/3)`` and returns the
           winner. Same algorithm case-1 uses.
        4. Pre-flight ``query_image`` on the winner (D19 mirror) —
           clean error if scheduler's snapshot was stale.
        5. ``AcquireContainerCommand`` to the winner; node runs
           ``ImageCacheManager.ensure_present(image)`` to pull /
           build / no-op as appropriate.

        Args:
            image: image ref to acquire (registry tag or per-node-
                local tag). Required.
            task_key: optional anti-affinity key; the scheduler
                rejects nodes already running ``max_runs_per_task``
                rollouts with this key. Useful when the consumer
                wants to spread parallel acquires for the same
                logical task across nodes.
            ensure_image_present: defaults to True (the user's
                UX). False reverts to the strict legacy contract
                ("no surprise pulls during evaluation"); the
                node-side manager raises XRLEnvError if the image
                isn't present locally. Surfaced as an SDK kwarg
                for deterministic-eval consumers.
            command / name / labels / environment: passed through
                to docker.containers.run on the chosen node.
        """
        self._require_raw_reconnect_capable("raw container")
        # Issue #6: authoritative policy enforcement. The control plane
        # is the SOLE authoritative validator for Level 1 / Level 2
        # docker-kwarg policy (the drop-in only fast-fails Level 3,
        # which has no policy override). Audit M1: previously the
        # drop-in pre-checked Level 1/2 against DEFAULT_POLICY, which
        # made operator opt-ins like ``allow_host_network: true`` and
        # operator extensions like ``allowed_devices: [..., /dev/sda]``
        # unreachable — the drop-in would reject before the cluster
        # policy was ever consulted. Now the drop-in forwards; this
        # check is the only gate.
        from xrlenv.control.kwargs_policy import (
            KwargsPolicyViolation,
            validate_kwargs,
        )
        _rejections = validate_kwargs(
            devices=devices,
            cap_add=cap_add,
            privileged=privileged if privileged else None,
            network_mode=network_mode,
            binds=binds,
            runtime=container_runtime,
            policy=self._kwargs_policy,
        )
        if _rejections:
            raise KwargsPolicyViolation(_rejections)

        # Freshness model (Part 2): resolve a registry tag -> content
        # digest BEFORE any record/placement, so the audit row, the
        # session, image-affinity, and the node's ensure_present all key
        # on the exact bytes the registry currently serves under that tag
        # (the mutable-tag staleness fix). Pass-through when no resolver
        # is wired, or the ref is already digest-pinned / not
        # registry-qualified. A transient registry outage falls back to
        # the last-known-good digest within the stale window; past that,
        # RegistryResolveError fails the acquire before any state is
        # written (operator-chosen, no acquiring-ghost left behind).
        if self._digest_resolver is not None:
            image = await self._digest_resolver.resolve(image)

        # Mint rollout_id up front: we want a durable record of the
        # acquire attempt even if scheduler.place / pre-flight /
        # node.acquire_container raises. The audit trail is more
        # useful than a missing row.
        rollout_id = uuid.uuid4().hex
        artifact_path, displayed_name = _parse_metadata_labels(labels)
        group_id = _parse_group_id_label(labels)
        # Fleet reservation (phase 1, opt-in): parse the generic fleet labels.
        # ``fleet_id is None`` (no ``xrlenv.fleet_id`` label) is the common
        # path — every branch below is gated on it, so a non-fleet acquire is
        # byte-for-byte unchanged. A malformed footprint raises here (user
        # error, before any state is written).
        fleet_id, fleet_footprint = _parse_fleet_labels(labels)
        # Write the ``acquiring`` record before any wire activity so
        # admin /rollouts surfaces the in-flight attempt + persists
        # an audit trail on placement failures.
        #
        # This is INTENTIONALLY ahead of the admission queue (below), so the
        # admission cap / AIMD limit throttle *execution* (container create,
        # CPU/mem) but not the state-store write itself: a submission burst
        # writes one row per acquire here regardless. That's an accepted
        # trade-off — the audit record for a pre-admission failure is worth
        # more than back-pressuring the write — because the runaway-WAL risk
        # it used to feed is now bounded by the periodic WAL checkpointer
        # (xrlenv/control/wal_checkpointer.py); a burst's rows are tens of MB,
        # folded into the main DB every checkpoint tick. See
        # [[wal-runaway-cp-stall]] for the incident this reasoning comes from.
        self._record_acquiring(
            rollout_id=rollout_id,
            image=image,
            artifact_path=artifact_path,
            displayed_name=displayed_name,
            task_key=task_key,
            group_id=group_id,
            fleet_id=fleet_id,
            owner_id=owner_id,
        )
        # Audit M1 (round 2): mark the acquire in-flight the instant
        # the row exists. The ``finally`` below guarantees removal on
        # every exit path (success hands off to ``_sessions``;
        # error / ``CancelledError`` discards), so the reconciler
        # never sees a stale entry.
        async with self._lock:
            self._acquiring_ids.add(rollout_id)

        # ``placement`` is bound only after ``self._scheduler.place(...)``
        # returns; the leak-fix lifecycle (commit on success / release on
        # failure) is scoped to the inner ``try`` block below so that a
        # failure *before* ``place()`` doesn't try to release a placement
        # that was never reserved. See the leak-fix block below for the
        # design rationale.
        placement = None
        queue_wait_s = 0.0
        # Fleet role, resolved inside the try by ``_classify_fleet_acquire``.
        # Stays ``"non_fleet"`` for the default path AND if classification
        # raises (duplicate opener / companion-before-opener) — so the
        # ``except`` runs fleet cleanup only for a role that actually reserved
        # something (an opener that marked opening, a companion that took a
        # slot). An ``overflow`` companion reserves nothing (normal placement),
        # so it too is excluded from fleet cleanup.
        fleet_role = "non_fleet"
        reservation: FleetReservation | None = None
        try:
            # P0a — derive the effective ResourceSpec from the
            # harness's CPU/memory request (the drop-in extracted it
            # from host_config). Derived once; the same object feeds
            # the admission queue, the scheduler, and capacity.
            effective_resources = _effective_raw_resources(
                cpu_limit=cpu_limit, mem_limit_bytes=mem_limit_bytes,
                cpu_isolation=cpu_isolation,
            )
            synthetic_manifest = _synthetic_manifest_for_raw(
                image, effective_resources,
            )
            # Classify the fleet role + reserve the member slot / mark opening
            # atomically (F2). Non-fleet → ``"non_fleet"`` and no-op. A
            # companion's FleetOverBudget check + slot reservation happen here,
            # before any node command. An opener reserves the FOOTPRINT (below);
            # a companion skips placement entirely and targets the pinned node.
            fleet_role, reservation = await self._classify_fleet_acquire(
                rollout_id=rollout_id,
                fleet_id=fleet_id,
                fleet_footprint=fleet_footprint,
                effective_resources=effective_resources,
            )
            # An opener reserves the whole declared footprint on one node; a
            # non-fleet / companion / overflow acquire reserves nothing extra.
            fleet_reserve = fleet_footprint if fleet_role == "opener" else None
            # The container is a fleet MEMBER (suppressed in accounting, covered
            # by the footprint) only as an opener or an in-budget companion. An
            # ``overflow`` companion is admitted as an ordinary container (normal
            # placement, charged its OWN resources), so it carries no fleet_id in
            # accounting — else iter_load_entries would wrongly suppress it.
            effective_fleet_id = (
                fleet_id if fleet_role in ("opener", "companion") else None
            )
            # Issue #18 fix #1: route through the admission queue when
            # wired. The queue does its own ``query_image_presence`` +
            # ``_lookup_preferred_home`` internally so the call shape
            # mirrors what managed sandboxes use. On
            # ``CapacityExhausted`` the queue holds the request up to
            # ``queue_timeout_s`` and surfaces a clean timeout only if
            # that bound is reached. Without admission wired (older
            # single-node tests), the direct ``scheduler.place(...)``
            # path below preserves legacy synchronous behaviour.
            # D-AR-2026-07-07-B — control-plane re-admit loop. Fleet role was
            # classified ONCE above; ONLY the place+acquire attempt repeats. On a
            # retriable create-time saturation fault (node overloaded — sysbox-fs
            # pre-register DeadlineExceeded / docker 5xx / wire timeout) we release
            # the placement, exclude the failed node, and re-admit on a sibling.
            # See _is_retriable_acquire_saturation + the module-scope constants.
            _failed_node_ids: set[str] = set()
            _readmit_attempt = 0
            _readmit_deadline: float | None = None  # armed on first saturation
            _last_saturation_exc: BaseException | None = None
            # Bound before the loop so the except handler can reference it even
            # if a rare fault (e.g. the admission-internal image-presence query)
            # raises BEFORE placement resolved a node. The "<unknown>" sentinel
            # is filtered out of _failed_node_ids below (no node to exclude).
            node_id = "<unknown>"
            # Pure-capability set for the relax decision (computed LAZILY on the
            # first saturation failure, so the happy path never pays for it and
            # never depends on it): when the failed set covers EVERY node that
            # could serve this (backend, runtime) there is no sibling to steer to
            # — relax the exclusion to the empty set and let the request queue on
            # the shared pool rather than hard-fail.
            _capable_node_ids: frozenset[str] | None = None
            while True:
                # Effective exclusion this attempt (relax when it covers all
                # capable). Only reached with a non-empty exclusion AFTER a
                # saturation failure, so the capability probe stays off the
                # happy path.
                _readmit_exclude: frozenset[str] = frozenset(_failed_node_ids)
                if _readmit_exclude:
                    if _capable_node_ids is None:
                        _capable_node_ids = self._scheduler.capable_node_ids(
                            backend="docker",
                            container_runtime=container_runtime,
                        )
                    if _readmit_exclude >= _capable_node_ids:
                        _readmit_exclude = frozenset()
                # Per-attempt budgets. The re-admit cap engages only AFTER the
                # first saturation failure, so a normal first-attempt queue wait
                # keeps its full queue_timeout_s (24 h default). Once engaged it
                # clamps BOTH the admission-wait timeout AND the node-wire
                # acquire_timeout to the remaining cap, so a queued _Waiter on the
                # surviving pool cannot outlast the cap.
                _attempt_queue_timeout = max(0.0, queue_timeout_s - queue_wait_s)
                _attempt_acquire_timeout = acquire_timeout_s
                if _readmit_deadline is not None:
                    _remaining_cap = _readmit_deadline - time.monotonic()
                    if _remaining_cap <= 0:
                        assert _last_saturation_exc is not None
                        raise _last_saturation_exc
                    _attempt_queue_timeout = min(
                        _attempt_queue_timeout, _remaining_cap,
                    )
                    _attempt_acquire_timeout = (
                        _remaining_cap
                        if _attempt_acquire_timeout is None
                        else min(_attempt_acquire_timeout, _remaining_cap)
                    )
                    # Don't let the re-admit budget starve a COMMITTED create: the
                    # cap bounds placement re-tries (the admission-WAIT clamp on
                    # _attempt_queue_timeout above), but the node-wire create needs
                    # adequate time once a node is chosen. Floor it — capped at the
                    # caller's explicit acquire_timeout_s so a deliberately tight
                    # create deadline is still honoured. See _MIN_CREATE_DEADLINE_S.
                    _create_floor = (
                        _MIN_CREATE_DEADLINE_S
                        if acquire_timeout_s is None
                        else min(_MIN_CREATE_DEADLINE_S, acquire_timeout_s)
                    )
                    _attempt_acquire_timeout = max(
                        _attempt_acquire_timeout, _create_floor,
                    )
                try:
                    if fleet_role == "companion":
                        # Companion of an open fleet (F3): SKIP placement entirely —
                        # the fleet already reserved its footprint on one node when it
                        # opened, and the slot fit within that footprint was checked in
                        # ``_classify_fleet_acquire``. Target the pinned node directly;
                        # no ``place()``, no ``_pending`` reservation, no
                        # ``commit_placement`` (``placement`` stays ``None``).
                        assert reservation is not None  # role == companion ⇒ set
                        # §5.3 — companions skip Scheduler.place, so the per-node
                        # runtime filter can't cover them. Enforce runtime
                        # consistency here: (a) every member must match the fleet's
                        # runtime, and (b) the pinned node must advertise it. Fail
                        # loud rather than silently run a companion under the wrong
                        # runtime on a node that may not even support it.
                        _want_rt = container_runtime or None
                        _fleet_rt = reservation.container_runtime or None
                        if _want_rt != _fleet_rt:
                            raise XRLEnvError(
                                f"fleet companion requested container_runtime="
                                f"{_want_rt!r} but fleet {fleet_id!r} was opened with "
                                f"{_fleet_rt!r}; every member of a fleet must run under "
                                f"the same runtime.",
                            )
                        node = self._node_by_id(reservation.node_id)
                        node_id = reservation.node_id
                        if _fleet_rt and _fleet_rt != "runc":
                            _node_rts = list(
                                getattr(node, "supported_runtimes", lambda: ["runc"])()
                                or [],
                            ) or ["runc"]
                            if _fleet_rt not in _node_rts:
                                raise XRLEnvError(
                                    f"fleet {fleet_id!r} pinned node {node_id!r} no "
                                    f"longer advertises runtime {_fleet_rt!r} "
                                    f"(advertised: {_node_rts}); cannot place companion.",
                                )
                        LOGGER.info(
                            "raw-container.coordinator.fleet-companion image=%s "
                            "fleet=%s node=%s runtime=%s (drawn from reservation, "
                            "no re-place)",
                            image, fleet_id, node_id, _fleet_rt or "runc",
                        )
                    elif self._admission is not None:
                        # Stage 2 (notes/admission-stage-2-queue-clocks.md):
                        # ``queue_timeout_s`` defaults to DEFAULT_QUEUE_TIMEOUT_S
                        # (24 h) — waiting in the queue is not a failure and
                        # must not be timed out under normal use; a consumer
                        # that wants fail-fast passes a small value explicitly.
                        # ``acquire_timeout_s`` remains the *pull-side* deadline
                        # on the chosen node — an orthogonal axis, and (like the
                        # session deadline) it starts only after admission, so
                        # queue-wait never erodes a run-time budget.
                        queue_wait_start = time.monotonic()
                        acquire_kwargs: dict[str, Any] = {
                            "manifest": synthetic_manifest,
                            "task_key": task_key,
                            "request_id": request_id,
                            "owner_id": owner_id,
                            "backend": "docker",
                            # D-AR-2026-07-07-B — per-attempt admission-wait budget:
                            # min(remaining queue_timeout_s, remaining re-admit cap).
                            # On attempt 0 this is just ``queue_timeout_s`` (the cap is
                            # not engaged until the first saturation failure).
                            "timeout_s": _attempt_queue_timeout,
                        }
                        # §5.3 — thread the OCI runtime so the admission queue's
                        # placement (and every drain retry) filters to nodes that
                        # advertise it. Omitted when None so the non-sysbox call
                        # shape is byte-for-byte the legacy signature.
                        if container_runtime:
                            acquire_kwargs["container_runtime"] = container_runtime
                        # Fleet opener ONLY: admit + queue against the whole footprint,
                        # not the lead's own request. Passed only when opening a fleet
                        # so the non-fleet / companion call is byte-for-byte the legacy
                        # signature (no ``reserve`` kwarg).
                        if fleet_reserve is not None:
                            acquire_kwargs["reserve"] = fleet_reserve
                        # D-AR-2026-07-07-B — steer away from nodes that just failed a
                        # create with node saturation. Stored on the queued ``_Waiter``
                        # too, so a parked re-admit can't drain back onto the hot node.
                        if _readmit_exclude:
                            acquire_kwargs["exclude_node_ids"] = _readmit_exclude
                        placement = await self._admission.acquire(**acquire_kwargs)
                        queue_wait_s += time.monotonic() - queue_wait_start
                        node = placement.node
                        node_id = getattr(node, "node_id", "<unknown>")
                        # Log at WARN when the consumer's request actually
                        # had to wait — operator-actionable signal that
                        # they're over-requesting concurrency. The 1 s
                        # threshold filters fast-path admissions (where
                        # ``queue_wait_s`` is bounded by the scheduler's
                        # sync ``place`` call) from genuine queue stalls.
                        if queue_wait_s >= 1.0:
                            LOGGER.warning(
                                "raw-container.coordinator.admit-queued "
                                "image=%s node=%s queue_wait_s=%.1f "
                                "(consumer concurrency exceeds cluster capacity; "
                                "reduce --num-workers / consider a larger cluster)",
                                image, node_id, queue_wait_s,
                            )
                        else:
                            LOGGER.info(
                                "raw-container.coordinator.admit image=%s node=%s "
                                "score=%.3f queue_wait_s=%.3f",
                                image, node_id, placement.score, queue_wait_s,
                            )
                    else:
                        image_present = await query_image_presence(
                            self._scheduler, image, backend="docker",
                        )
                        preferred_home = self._lookup_preferred_home(image)
                        place_kwargs: dict[str, Any] = {
                            "task_key": task_key,
                            "backend": "docker",
                            "image_present": image_present,
                            "preferred_home_node": preferred_home,
                        }
                        # §5.3 — runtime-aware placement on the no-admission
                        # fallback path too (older single-node tests). Omitted when
                        # None to keep the legacy call shape.
                        if container_runtime:
                            place_kwargs["container_runtime"] = container_runtime
                        # Fleet opener ONLY (see admission path) — non-fleet / companion
                        # keeps the legacy call signature verbatim.
                        if fleet_reserve is not None:
                            place_kwargs["reserve"] = fleet_reserve
                        # D-AR-2026-07-07-B — same failed-node exclusion on the direct
                        # (no-admission) placement path.
                        if _readmit_exclude:
                            place_kwargs["exclude_node_ids"] = _readmit_exclude
                        placement = self._scheduler.place(
                            synthetic_manifest, **place_kwargs,
                        )
                        node = placement.node
                        node_id = getattr(node, "node_id", "<unknown>")
                        LOGGER.info(
                            "raw-container.coordinator.place image=%s node=%s "
                            "score=%.3f image_present=%s preferred_home=%s",
                            image, node_id, placement.score,
                            image_present, preferred_home,
                        )
                    # Stamp the chosen node onto the record now, before the
                    # wire-level acquire so admin shows it during the
                    # potentially-slow image pull.
                    self._update_record(rollout_id, node_id=node_id)

                    # D19 mirror: pre-flight query_image on the winner. If
                    # ensure_image_present=True the node will pull anyway,
                    # but this gives the operator a clean diagnostic
                    # ("scheduler picked X based on a stale snapshot") rather
                    # than an opaque pull-then-timeout.
                    try:
                        present_check = await node.query_image(image)
                    except Exception:
                        LOGGER.warning(
                            "raw-container.coordinator.preflight: query_image "
                            "failed on node=%s for %s; proceeding with acquire "
                            "(ensure_present will pull/build if needed)",
                            node_id, image, exc_info=True,
                        )
                    else:
                        if not present_check.present and not ensure_image_present:
                            raise XRLEnvError(
                                f"raw-container acquire: scheduler picked node "
                                f"{node_id!r} based on image-presence snapshot, "
                                f"but pre-flight query_image says {image!r} is no "
                                f"longer there (likely LRU eviction between "
                                f"snapshot and acquire). ensure_image_present is "
                                f"False so we won't pull. Retry to re-place, or "
                                f"pass ensure_image_present=True to fall through "
                                f"to ensure_present on the chosen node.",
                            )

                    record = await node.acquire_container(
                        rollout_id=rollout_id,
                        backend="docker",
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
                        # D-AR-2026-07-07-B — node-wire deadline clamped to the
                        # remaining re-admit cap once engaged (min with acquire_timeout_s),
                        # so a run of all-timeout attempts can't exceed the total cap.
                        acquire_timeout_s=_attempt_acquire_timeout,
                        # P1 — the effective ResourceSpec, stamped onto
                        # AcquireContainerCommand so the node applies cgroup
                        # limits. Same object used for placement + load
                        # accounting, so the container is capped at exactly
                        # what the scheduler reserved.
                        resources=effective_resources,
                        # P0b — container-shape RuntimeLimits (pids / shm /
                        # tmpfs / read-only). Scheduling-neutral; the node
                        # applies only the limits the harness specified.
                        runtime_limits=runtime_limits,
                        # §5.1 — OCI runtime selector; the node verifies it is
                        # registered before ``containers.run`` and fails loud.
                        container_runtime=container_runtime,
                    )
                except BaseException as exc:
                    # A fleet COMPANION is pinned to the fleet's node (no place),
                    # so there is no sibling to re-admit it to — propagate. And a
                    # non-saturation fault (terminal 4xx / policy / capability, or
                    # a non-XRLEnvError) would reproduce on any node — propagate to
                    # the outer handler, which runs the placement-release + fleet
                    # cleanup + status update exactly as before.
                    if fleet_role == "companion" or not (
                        _is_retriable_acquire_saturation(exc)
                    ):
                        raise
                    # Retriable create-time saturation on a placed attempt.
                    # Release THIS attempt's placement so its _pending footprint
                    # doesn't leak, then set placement=None so the outer handler
                    # won't double-release. (An opener keeps its _fleet_opening
                    # marker across re-admits — only the final failure, via the
                    # outer except, clears it; do NOT touch it here.)
                    if placement is not None:
                        self._scheduler.release_placement(placement)
                        placement = None
                    if node_id and node_id != "<unknown>":
                        _failed_node_ids.add(node_id)
                    _last_saturation_exc = exc
                    _readmit_attempt += 1
                    if _readmit_deadline is None:
                        _readmit_deadline = (
                            time.monotonic() + _CP_REQUEUE_TOTAL_CAP_S
                        )
                    if _readmit_attempt > _CP_REQUEUE_MAX:
                        raise
                    LOGGER.warning(
                        "raw-container.coordinator.re-admit image=%s "
                        "failed_node=%s attempt=%d/%d excluded=%s reason=%s",
                        image, node_id, _readmit_attempt, _CP_REQUEUE_MAX,
                        sorted(_failed_node_ids), f"{type(exc).__name__}: {exc}",
                    )
                    continue
                break
            # Acquire succeeded; build the session and hand the
            # rollout_id off from ``_acquiring_ids`` to ``_sessions``
            # atomically under one lock — a concurrent reconciler
            # sweep must never observe the rollout in neither set.
            # Issue #18 — wall-clock reap deadline. Consumer-supplied
            # ``session_deadline_s`` (>0) wins; otherwise the
            # coordinator's default cap. The raw-GC reconciler
            # force-destroys the session once ``deadline_at`` passes.
            deadline_at = time.time() + (
                session_deadline_s
                if session_deadline_s is not None and session_deadline_s > 0
                else self._session_deadline_default_s
            )
            session = RawContainerSession(
                rollout_id=rollout_id,
                node=node,
                node_id=node_id,
                container_id=record.container_id,
                container_name=record.container_name,
                image=image,
                created_at=_dt.datetime.now(_dt.UTC),
                task_key=task_key,
                # Fleet member (opener or in-budget companion) when set —
                # iter_load_entries then suppresses this session's cpu/mem (the
                # fleet footprint stands in) and charges only its disk. ``None``
                # for a non-fleet OR an ``overflow`` acquire (charged in full).
                fleet_id=effective_fleet_id,
                # P0a — persist the spec used for placement so
                # iter_load_entries charges the same footprint.
                effective_resources=effective_resources,
                queue_wait_s=queue_wait_s,
                deadline_at=deadline_at,
                # Per-node runtime-concurrency cap: iter_load_entries surfaces
                # this so the scheduler counts concurrent sysbox containers.
                container_runtime=container_runtime,
            )
            opened_reservation: FleetReservation | None = None
            async with self._lock:
                if fleet_role == "opener":
                    # role == opener ⇒ both were present (guaranteed by
                    # _classify_fleet_acquire); assert narrows for the checker.
                    assert fleet_id is not None
                    assert fleet_footprint is not None
                    # Create the reservation now the lead is up — the atomic
                    # handoff (R1 note): ``_pending`` has covered the footprint
                    # since ``place(reserve=)``; creating the reservation before
                    # ``commit_placement`` (below) means the footprint is covered
                    # by BOTH for a brief instant (safe over-count) and never has
                    # a gap. The lead is the reservation's first member.
                    opened_reservation = FleetReservation(
                        fleet_id=fleet_id,
                        node_id=node_id,
                        footprint=fleet_footprint,  # opener ⇒ not None
                        members={rollout_id: effective_resources},
                        opened_ts=time.time(),
                        last_acquire_ts=time.time(),
                        task_key=task_key,
                        owner=owner_id,
                        # §5.3 — pin the fleet's runtime from the opener so
                        # every companion is validated against it.
                        container_runtime=container_runtime,
                    )
                    self._fleets[fleet_id] = opened_reservation
                    self._fleet_opening.discard(fleet_id)
                # (companion's member slot was reserved in _classify_fleet_acquire)
                self._sessions[rollout_id] = session
                self._acquiring_ids.discard(rollout_id)
                # Seed the liveness clock at acquire (an implicit first touch).
                self._last_seen_at[rollout_id] = time.time()
            # Persist the tiny reservation row OUTSIDE the lock (best-effort) —
            # an opener writes it (restart recovers the CP-only footprint from
            # it); a companion refreshes last_acquire_ts to keep the TTL grace
            # window honest across a restart. Deleted on last-member teardown.
            if opened_reservation is not None:
                self._persist_fleet_reservation(opened_reservation)
            elif fleet_role == "companion" and fleet_id is not None:
                self._touch_fleet_reservation(fleet_id, time.time())
        except BaseException as exc:
            # Audit M1 (round 3): catch ``BaseException``, not just
            # ``Exception`` — mirrors ``destroy()``. ``asyncio``
            # cancellation (``CancelledError``, a ``BaseException``
            # subclass) is a real exit path for an acquire whose
            # consumer gave up or whose deadline-watcher fired; it
            # must run the SAME cleanup as an ordinary failure or it
            # leaks a ``_pending`` scheduler reservation (the capacity
            # gate would eventually reject every placement) and leaves
            # the persisted row stuck ``acquiring``.
            #
            # Fleet cleanup (F1) — undo any reservation this acquire took,
            # BEFORE the placement release below. An **opener** may have
            # created the ``FleetReservation`` (if it failed after the lead
            # came up) and marked the fleet opening; a **companion** reserved
            # a member slot in ``_classify_fleet_acquire``. Both must be
            # rolled back or the footprint / slot leaks. ``fleet_role`` is
            # still ``"non_fleet"`` when classification itself raised (nothing
            # was reserved), so this no-ops there. ``release_placement``
            # (below) drops the opener's ``_pending`` footprint; here we only
            # touch the fleet table.
            if fleet_role in ("opener", "companion"):
                async with self._lock:
                    if fleet_id is not None:
                        self._fleet_opening.discard(fleet_id)
                    # Removes the lead (opener, if the reservation got created)
                    # or the companion's slot; drops the reservation if it was
                    # this member's last. Idempotent when nothing was added.
                    self._release_fleet_member_locked(rollout_id)
            # Leak fix: release the ``_pending`` reservation if
            # ``place()`` had already minted one. Failures *before*
            # ``place()`` returned leave ``placement`` as ``None`` and
            # skip the release. ``release_placement`` and
            # ``_update_record`` are synchronous — safe to run while
            # unwinding a cancellation (no ``await`` to re-raise). The
            # release is ALSO the teardown step whose outcome decides the
            # status below, so we capture whether it succeeded.
            cleanup_error: BaseException | None = None
            if placement is not None:
                try:
                    self._scheduler.release_placement(placement)
                except Exception as ce:
                    cleanup_error = ce
                    LOGGER.exception(
                        "raw-container.acquire: releasing the scheduler "
                        "reservation failed while unwinding rollout=%s",
                        rollout_id,
                    )
            # Status reflects the OUTCOME of the cancellation, not its
            # origin. A ``CancelledError`` means a cancel was requested —
            # by whoever (consumer give-up, deadline-watcher, shutdown,
            # batch teardown); we can't tell from here and it doesn't
            # matter. If we then unwound cleanly, the cancel SUCCEEDED →
            # ``cancelled``. If the teardown itself errored (a leaked
            # reservation), the cancel FAILED → ``failed``. A
            # non-cancellation exception is a plain ``failed``. (Either
            # way the reason is recorded — the old code's bare
            # ``"CancelledError: "`` was the 172-row complaint.)
            if isinstance(exc, asyncio.CancelledError) and cleanup_error is None:
                status = "cancelled"
                error = _acquire_cancel_reason(
                    exc, queue_wait_s=queue_wait_s,
                    placed=placement is not None,
                )
            elif isinstance(exc, asyncio.CancelledError):
                status = "failed"
                error = (
                    "acquire cancellation failed to release the scheduler "
                    f"reservation (slot may be leaked): "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            elif isinstance(exc, CapacityExhausted) and cleanup_error is None:
                # Pure backpressure: the scheduler declined to place within
                # queue_timeout_s. Distinct from ``failed`` so a retried-then-
                # succeeded task doesn't read as a failure (spec 13). A
                # cleanup_error (leaked reservation) would be a real fault →
                # falls through to ``failed`` below.
                status = "capacity_rejected"
                error = _acquire_capacity_rejected_reason(exc)
            else:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            self._update_record(
                rollout_id,
                status=status,
                error=error,
                finished_at=time.time(),
            )
            raise
        finally:
            # Guarantee the in-flight marker is cleared on every exit.
            # On success the handoff above already removed it
            # (``discard`` is idempotent); on error / cancellation the
            # ``except`` ran first — this is belt-and-braces so a row
            # can't stay permanently protected from reconciliation.
            async with self._lock:
                self._acquiring_ids.discard(rollout_id)

        # Leak fix: the session was registered in ``_sessions`` inside
        # the ``try`` above — BEFORE this ``commit_placement`` drops the
        # ``_pending`` reservation — so the scheduler's load provider
        # (``iter_load_entries``) is already returning this session when
        # a concurrent ``place()`` runs in the brief window where both
        # are visible — that's a one-rollout over-count, which mirrors
        # the managed-template ordering (see ``coordinator.py`` around
        # the ``insert_sandbox`` → ``commit_placement`` pair). The
        # inverse ordering (commit then register) would create an
        # under-count window and re-introduce the over-placement bug
        # under load.
        #
        # A fleet **companion** skipped ``place()`` (it drew from the fleet's
        # existing reservation), so there is no ``_pending`` to commit —
        # ``placement`` is ``None``. An **opener** committing here drops the
        # ``_pending`` footprint now that its ``FleetReservation`` covers the
        # load (created above, strictly before this commit — the R1 handoff).
        if placement is not None:
            self._scheduler.commit_placement(placement)
        self._update_record(
            rollout_id,
            status="running",
            container_id=record.container_id,
            container_name=record.container_name,
            # Audit P2: persist the resolved deadline + effective
            # ResourceSpec so a control-plane restart can re-adopt this
            # session with its ORIGINAL semantics instead of resetting
            # to the default deadline / raw footprint.
            deadline_at=deadline_at,
            effective_resources_json=effective_resources.model_dump_json(),
            # Audit H11: persist the container runtime so a CP restart restores it on
            # re-adoption — else the per-node runtime concurrency cap would count a surviving
            # sysbox-runc container as absent and could over-place past the daemon-protection cap.
            container_runtime=container_runtime,
            # Persist the FINAL fleet association (not the raw label): an
            # ``overflow`` container was admitted as an ordinary container, so
            # its row must NOT carry a fleet_id — else the restart rebuild would
            # wrongly re-adopt it as a suppressed fleet member. ``effective_fleet_id``
            # is None for overflow / non-fleet, set for opener / companion.
            fleet_id=effective_fleet_id,
        )
        LOGGER.info(
            "raw-container.coordinator.acquire rollout=%s node=%s "
            "container=%s image=%s",
            rollout_id, session.node_id, session.container_id[:12], image,
        )
        return session

    # ── P1.7.C.2 — multi-service compose project acquire ────────────────────

    def _subnet_placement_guard(self, claims: tuple[str, ...]) -> Any:
        """An async context manager that serializes placement for acquires sharing
        the same pinned subnet set. Empty claims (DNS-only) → a no-op context (no
        serialization). ``setdefault`` is atomic under asyncio's single thread."""
        if not claims:
            return contextlib.nullcontext()
        key = tuple(sorted(claims))
        return self._subnet_locks.setdefault(key, asyncio.Lock())

    def _subnet_conflict_nodes_locked(
        self, claims: tuple[str, ...],
    ) -> frozenset[str]:
        """Nodes that must be excluded for a project claiming ``claims`` — those
        running a **committed** compose project OR an **in-flight** placement whose
        subnet claim overlaps. Caller holds ``self._lock``. Empty claims → empty
        set (DNS-only → no exclusion)."""
        if not claims:
            return frozenset()
        conflict: set[str] = set()
        for rec in self._compose_projects.values():
            if any(
                subnets_overlap(c, sc) for c in claims for sc in rec.subnet_claims
            ):
                conflict.add(rec.node_id)
        for node_id, pending in self._pending_subnet_claims.values():
            if any(subnets_overlap(c, sc) for c in claims for sc in pending):
                conflict.add(node_id)
        return frozenset(conflict)

    async def acquire_compose_project(
        self,
        *,
        compose_yaml: str,
        images: list[str],
        footprint: ResourceSpec,
        main_service: str = "main",
        project_name: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        request_id: str | None = None,
        labels: dict[str, str] | None = None,
        owner_id: str = "default",
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        session_deadline_s: float | None = None,
        up_timeout_s: float | None = None,
    ) -> RawComposeAcquireResult:
        """Bring up a multi-service compose project on a scheduler-chosen node.

        Mirrors :meth:`acquire`'s lifecycle in the order `acquire_container` uses
        (vet → digest → acquiring row → place → node command → session/commit →
        running), but as its own method (the compose flow diverges too much to
        branch). The whole stack is one scheduling unit: the ``main`` session
        carries ``effective_resources=footprint`` (so ``iter_load_entries`` charges
        it once), the sidecars are never CP sessions, and teardown routes to
        :meth:`destroy_compose_project`. See ``notes/multi-service-compose-step3-
        plan.md`` §2.
        """
        self._require_raw_reconnect_capable("compose project")
        main_service = main_service or "main"
        # 1. Parse + vet (before any row) — a policy violation raises here,
        #    leaving no acquiring-ghost behind.
        parsed = yaml.safe_load(compose_yaml)
        if not isinstance(parsed, dict):
            raise XRLEnvError(
                "acquire_compose_project: compose_yaml did not parse to a mapping.",
            )
        vet_compose_project(parsed, policy=self._kwargs_policy)

        # 2. Digest-resolve the main image BEFORE any record/placement (freshness),
        #    then thread the resolved ref through the compose + images list (§2.3).
        original_main = _compose_main_image(parsed, main_service)
        resolved_main = original_main
        if self._digest_resolver is not None and original_main:
            resolved_main = await self._digest_resolver.resolve(original_main)
        placement_image = resolved_main or f"compose/{main_service}"

        # 3. Mint rollout_id + prepare (stamp labels + pin) + write the acquiring
        #    row + mark in-flight, all before wire activity.
        rollout_id = uuid.uuid4().hex
        project = _sanitize_compose_project_name(
            project_name or f"xrlenv-{rollout_id[:16]}",
        )
        prepared = prepare_compose(
            parsed, images, rollout_id=rollout_id, project_name=project,
            main_service=main_service, resolved_main_ref=resolved_main,
        )
        artifact_path, displayed_name = _parse_metadata_labels(labels)
        self._record_acquiring(
            rollout_id=rollout_id, image=placement_image,
            artifact_path=artifact_path, displayed_name=displayed_name,
            task_key=task_key, group_id=group_id, owner_id=owner_id,
        )
        async with self._lock:
            self._acquiring_ids.add(rollout_id)

        # §5 / 3b — subnet anti-affinity. A static-IP project pins a subnet its
        # solve.sh hard-codes; docker refuses two overlapping networks on one
        # daemon, so it must not land on a node already running (or in-flight
        # placing) a project whose subnet claim overlaps this one. Empty claims
        # (service-DNS-only tasks) → no exclusion → unbounded concurrency.
        claims = subnet_claims(parsed)

        placement = None
        node = None
        node_id = "<unknown>"
        record = None
        node_up_attempted = False   # H10: True once the wire `up` may have created containers
        queue_wait_s = 0.0
        try:
            # 4. Place the whole stack (reserve=footprint). The exclude→place/admit
            #    →reserve-pending section runs under the per-subnet guard so two
            #    same-subnet acquires serialize: the second sees the first's
            #    pending claim and is steered off its node — on BOTH the admission
            #    and direct paths. DNS-only (no claims) → no-op guard, no contention.
            manifest = _synthetic_manifest_for_raw(placement_image, footprint)
            async with self._subnet_placement_guard(claims):
                if self._admission is not None:
                    async with self._lock:
                        subnet_exclude = self._subnet_conflict_nodes_locked(claims)
                    q_start = time.monotonic()
                    placement = await self._admission.acquire(
                        manifest=manifest, task_key=task_key,
                        request_id=request_id, owner_id=owner_id, backend="docker",
                        timeout_s=queue_timeout_s, reserve=footprint,
                        exclude_node_ids=subnet_exclude or None,
                    )
                    queue_wait_s = time.monotonic() - q_start
                    node = placement.node
                    node_id = getattr(node, "node_id", "<unknown>")
                    if claims:
                        async with self._lock:
                            self._pending_subnet_claims[rollout_id] = (
                                node_id, claims,
                            )
                else:
                    # No admission queue (single-node tests): mirror acquire()'s
                    # direct path — score by main-image affinity + preferred-home
                    # so the project lands near a node that already has the main
                    # image cached (§2.2). exclude→place→reserve under the main
                    # lock too (place() is synchronous).
                    image_present = await query_image_presence(
                        self._scheduler, placement_image, backend="docker",
                    )
                    preferred_home = self._lookup_preferred_home(placement_image)
                    async with self._lock:
                        subnet_exclude = self._subnet_conflict_nodes_locked(claims)
                        placement = self._scheduler.place(
                            manifest, task_key=task_key, backend="docker",
                            image_present=image_present,
                            preferred_home_node=preferred_home,
                            reserve=footprint,
                            exclude_node_ids=subnet_exclude or None,
                        )
                        node = placement.node
                        node_id = getattr(node, "node_id", "<unknown>")
                        if claims:
                            self._pending_subnet_claims[rollout_id] = (
                                node_id, claims,
                            )
            self._update_record(rollout_id, node_id=node_id)

            # 5. Issue the node command with the labeled + digest-pinned compose.
            #    From here the node's `docker compose up` MAY create containers, so a
            #    subsequent failure/cancel can leave a live stack (audit H10) — capacity
            #    can't be released until teardown is node-confirmed.
            node_up_attempted = True
            record = await node.acquire_compose_project(
                rollout_id=rollout_id,
                project_name=project,
                compose_yaml=yaml.safe_dump(prepared.compose, sort_keys=False),
                images=list(prepared.images),
                main_service=main_service,
                up_timeout_s=up_timeout_s,
            )

            # 6. Hand _acquiring → steady-state: the main session (footprint), the
            #    project record, commit the pending placement, seal row running.
            deadline_at = time.time() + (
                session_deadline_s
                if session_deadline_s is not None and session_deadline_s > 0
                else self._session_deadline_default_s
            )
            session = RawContainerSession(
                rollout_id=rollout_id, node=node, node_id=node_id,
                container_id=record.main_container_id,
                container_name=record.main_container_name,
                image=placement_image,
                created_at=_dt.datetime.now(_dt.UTC),
                task_key=task_key,
                effective_resources=footprint,
                queue_wait_s=queue_wait_s,
                deadline_at=deadline_at,
                compose_project_name=project,
            )
            service_ids = dict(record.service_container_ids)
            async with self._lock:
                self._sessions[rollout_id] = session
                self._compose_projects[rollout_id] = _ComposeProjectRecord(
                    project_name=project, node_id=node_id,
                    service_container_ids=service_ids,
                    subnet_claims=claims,
                )
                # The claim is now committed on the project record; drop the
                # in-flight reservation (its purpose is served).
                self._pending_subnet_claims.pop(rollout_id, None)
                self._acquiring_ids.discard(rollout_id)
                self._last_seen_at[rollout_id] = time.time()
            if placement is not None:
                self._scheduler.commit_placement(placement)
            self._update_record(
                rollout_id, status="running",
                container_id=record.main_container_id,
                container_name=record.main_container_name,
                deadline_at=deadline_at,
                effective_resources_json=footprint.model_dump_json(),
            )
            # Persist the tiny project row so a CP restart recovers the footprint
            # + subnet claims (the node can't re-derive them). Deleted on
            # confirmed teardown / node loss.
            self._persist_compose_project(
                rollout_id=rollout_id, project_name=project, node_id=node_id,
                footprint=footprint, subnet_claims_v=claims, owner=owner_id,
            )
            LOGGER.info(
                "raw-container.coordinator.acquire-compose rollout=%s node=%s "
                "project=%s services=%d main=%s",
                rollout_id, node_id, project, len(service_ids),
                record.main_container_id[:12],
            )
            return RawComposeAcquireResult(
                rollout_id=rollout_id, node_id=node_id,
                main_container_id=record.main_container_id,
                main_container_name=record.main_container_name,
                project_name=project, service_container_ids=service_ids,
                queue_wait_s=queue_wait_s,
            )
        except BaseException as exc:
            # Drop the in-flight marker + subnet reservation (frees the node for a same-subnet
            # acquire waiting behind this one).
            async with self._lock:
                self._acquiring_ids.discard(rollout_id)
                self._pending_subnet_claims.pop(rollout_id, None)
            # audit H10 — CAPACITY-SAFE cleanup. Once the wire `up` was attempted the node's
            # `docker compose up` MAY have created live containers, so the SAME rule as destroy
            # applies: capacity cannot be released until absence is node-CONFIRMED. Attempt a
            # whole-project down; only release the placement reservation + seal terminal if it
            # CONFIRMS teardown. If teardown can't be confirmed, the stack may be live → RETAIN a
            # GC-reclaimable capacity charge (register a compose session + project + COMMIT the
            # reservation so iter_load_entries keeps charging it and the raw-GC deadline/liveness
            # reaper retries the down + releases) — never release capacity into a live stack.
            if not node_up_attempted:
                # Pre-`up` failure (CapacityExhausted decline, placement/prepare error) — nothing
                # could have been created on the node → safe to release + seal, no teardown needed.
                if placement is not None:
                    self._scheduler.release_placement(placement)
                if isinstance(exc, CapacityExhausted):
                    status: RawRolloutStatus = "capacity_rejected"
                    error = _acquire_capacity_rejected_reason(exc)
                else:
                    status = "failed"
                    error = f"acquire_compose_project failed: {type(exc).__name__}: {exc}"
                self._update_record(
                    rollout_id, status=status, error=error, finished_at=time.time(),
                )
                raise
            # The wire was attempted → the stack may be live. Try a node-confirmed down.
            teardown_confirmed = False
            if node is not None:
                try:
                    await node.destroy_compose_project(
                        rollout_id=rollout_id, project_name=project, force=True,
                    )
                    teardown_confirmed = True
                except BaseException:
                    LOGGER.critical(
                        "raw-container.coordinator.acquire-compose rollout=%s project=%s node=%s "
                        "FAILED and its teardown could NOT be confirmed — the stack may be LIVE; "
                        "RETAINING a capacity charge for the raw-GC reaper to reclaim (audit H10)",
                        rollout_id, project, node_id,
                    )
            if teardown_confirmed:
                if placement is not None:
                    self._scheduler.release_placement(placement)
                self._update_record(
                    rollout_id, status="failed",
                    error=f"acquire_compose_project failed: {type(exc).__name__}: {exc}",
                    finished_at=time.time(),
                )
                raise
            # Teardown unconfirmed → retain a GC-reclaimable charge. Register a compose session +
            # project (container ids from the record when the wire returned; empty when it was
            # cancelled mid-up — whole-project down routes by project NAME regardless) and COMMIT
            # the reservation so it becomes a durable session charge. Row stays non-terminal so
            # the liveness/deadline reaper reclaims it (the consumer got the exception and won't
            # heartbeat → liveness reaps it), then node-confirmed teardown frees capacity.
            deadline_at = time.time() + self._session_deadline_default_s
            service_ids = (
                dict(record.service_container_ids) if record is not None else {}
            )
            # ``node_up_attempted`` is set immediately AFTER ``node`` is assigned from the
            # placement, so reaching here guarantees a node (narrows the Optional for mypy).
            assert node is not None
            async with self._lock:
                self._sessions[rollout_id] = RawContainerSession(
                    rollout_id=rollout_id, node=node, node_id=node_id,
                    container_id=(record.main_container_id if record is not None else ""),
                    container_name=(record.main_container_name if record is not None else ""),
                    image=placement_image,
                    created_at=_dt.datetime.now(_dt.UTC),
                    task_key=task_key, effective_resources=footprint,
                    deadline_at=deadline_at, compose_project_name=project,
                )
                self._compose_projects[rollout_id] = _ComposeProjectRecord(
                    project_name=project, node_id=node_id,
                    service_container_ids=service_ids, subnet_claims=claims,
                )
                self._last_seen_at[rollout_id] = time.time()
            if placement is not None:
                self._scheduler.commit_placement(placement)
            self._update_record(
                rollout_id, status="running",
                error=(
                    f"acquire_compose_project failed with an UNCONFIRMED teardown "
                    f"({type(exc).__name__}: {exc}); stack may be live — capacity retained for "
                    f"raw-GC reaper"
                ),
                container_id=(record.main_container_id if record is not None else None),
                deadline_at=deadline_at,
                effective_resources_json=footprint.model_dump_json(),
            )
            self._persist_compose_project(
                rollout_id=rollout_id, project_name=project, node_id=node_id,
                footprint=footprint, subnet_claims_v=claims, owner=owner_id,
            )
            raise

    # ── P1.7.B.2 W6+W7+W8: ref-list FFD bin-packing ─────────────────────────

    async def plan_image_distribution(
        self,
        *,
        rows: list[ImageToPlace],
        eager_prefetch: bool = False,
    ) -> list[ImagePlanResult]:
        """Run FFD bin-packing across the cluster's per-node free-disk
        budget; persist per-ref ``preferred_home`` rows in StateStore
        so subsequent raw-container acquires steer via
        ``Scheduler.place(preferred_home_node=...)``.

        Optional ``eager_prefetch`` dispatches ``EnsurePresentCommand``
        to each preferred_home node so images arrive before the first
        acquire — useful for batch sweeps where the first ``N``
        acquires shouldn't pay a serial cold-pull penalty.

        Invoked by the operator CLI (``xrlenv images plan``); the
        consumer-facing SDK does NOT expose this method (would
        violate the docker-py drop-in contract; the audience's
        harness must contain only the one ``xrlenv.from_env()``
        line).
        """
        if self._state is None:
            raise XRLEnvError(
                "plan_image_distribution requires a StateStore "
                "(LocalRuntime + DistributedRuntime both wire one).",
            )
        if not rows:
            return []

        # 1. Snapshot per-node free disk via report_images.
        node_budgets: list[NodeBudget] = []
        for node in self._scheduler.nodes:
            if "docker" not in node.supported_backends():
                continue
            try:
                report = await node.report_images()
                node_budgets.append(NodeBudget(
                    node_id=NodeId(node.node_id),
                    available_bytes=int(report.free_disk_bytes),
                ))
            except Exception:
                LOGGER.exception(
                    "plan_image_distribution: report_images failed "
                    "on node=%s; excluding from budget snapshot",
                    getattr(node, "node_id", "<unknown>"),
                )

        if not node_budgets:
            raise XRLEnvError(
                "plan_image_distribution: no docker-capable nodes "
                "with reachable report_images. Connect at least one "
                "node-agent and retry.",
            )

        # 2. Run FFD planner. Opportunistic mode: rows that don't
        #    fit become ``deferred`` with a preferred_home (the
        #    most-free node at consideration time), so they still
        #    influence routing — the lazy ``ensure_present`` on
        #    first acquire materialises them.
        plan_result = plan_opportunistic_placements(
            images=rows, nodes=node_budgets,
        )

        # 3. Persist rows so ``find_registered_preferred_home``
        #    returns them for subsequent raw acquires.
        plan_id = f"raw-image-plan/{uuid.uuid4().hex}"
        self._state.record_build_plan(
            plan_id=plan_id,
            applied_by="xrlenv-images-plan",
            plan_json="{}",  # synthetic; not consumed by case-1 dispatch
        )

        # Import lazily so the unit tests' StateStore protocol
        # doesn't have to declare BuildAssignmentRecord.
        from xrlenv.control.state import BuildAssignmentRecord

        results: list[ImagePlanResult] = []
        for assignment in plan_result.placed.assignments:
            self._state.record_assignment(BuildAssignmentRecord(
                plan_id=plan_id,
                node_id=assignment.node_id,
                image_ref=assignment.image_ref,
                benchmark="raw-image-plan",
                # status="registered" so find_registered_preferred_home
                # returns this node_id when raw acquires query for the
                # image.
                status="registered",
            ))
            results.append(ImagePlanResult(
                image_ref=assignment.image_ref,
                preferred_home_node=assignment.node_id,
                status="placed",
                error=None,
            ))
        for deferred in plan_result.deferred:
            home = deferred.preferred_home or ""
            if home:
                self._state.record_assignment(BuildAssignmentRecord(
                    plan_id=plan_id,
                    node_id=home,
                    image_ref=deferred.image_ref,
                    benchmark="raw-image-plan",
                    status="registered",
                ))
            results.append(ImagePlanResult(
                image_ref=deferred.image_ref,
                preferred_home_node=home,
                status="deferred",
                error=None,
            ))

        # 4. Optional eager prefetch.
        if eager_prefetch:
            await self._eager_prefetch(plan_result, results)

        LOGGER.info(
            "plan_image_distribution: plan_id=%s placed=%d deferred=%d "
            "eager_prefetch=%s",
            plan_id, len(plan_result.placed.assignments),
            len(plan_result.deferred), eager_prefetch,
        )
        return results

    async def _eager_prefetch(
        self,
        plan_result: Any,
        results: list[ImagePlanResult],
    ) -> None:
        """Dispatch ``EnsurePresentCommand`` to each preferred_home
        per assignment. Concurrent + best-effort: a per-node failure
        marks the corresponding result row as failed but doesn't
        abort the sweep."""
        node_by_id = {n.node_id: n for n in self._scheduler.nodes}
        result_by_image = {r.image_ref: r for r in results}

        async def _prefetch(node: Any, image_ref: str) -> None:
            try:
                # ``RemoteNodeTransport.ensure_present`` returns
                # ``(status, error)``; we surface the error path
                # via the exception channel.
                ensure = getattr(node, "ensure_present", None)
                if ensure is None:
                    return  # local-transport fallback no-ops
                result = await ensure(image_ref, timeout_s=600.0)
                if isinstance(result, tuple) and result[0] != "ok":
                    raise XRLEnvError(
                        f"ensure_present returned status={result[0]!r}: "
                        f"{result[1]}",
                    )
            except Exception as exc:
                LOGGER.exception(
                    "eager_prefetch: ensure_present failed "
                    "node=%s image=%s",
                    node.node_id, image_ref,
                )
                row = result_by_image.get(image_ref)
                if row is not None:
                    # Mutating tuple-ish dataclass; rebuild.
                    idx = results.index(row)
                    results[idx] = ImagePlanResult(
                        image_ref=row.image_ref,
                        preferred_home_node=row.preferred_home_node,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )

        tasks: list[Any] = []
        for assignment in plan_result.placed.assignments:
            node = node_by_id.get(assignment.node_id)
            if node is not None:
                tasks.append(_prefetch(node, assignment.image_ref))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=False)

    # ── P1.7.B.3 record persistence ─────────────────────────────────────────

    def session_owner(self, rollout_id: str) -> str | None:
        """Owner of a raw (case-2/3) session by id, or ``None`` if unknown.

        Used by the gRPC servicer to enforce that container RPCs (Exec,
        Destroy, Put/GetArchive, …) act only on the caller's own sessions
        (audit M2). Reads the persisted row; falls back to ``None`` when no
        StateStore is wired or the id is unknown."""
        if self._state is None:
            return None
        get_raw = getattr(self._state, "get_raw_rollout", None)
        if get_raw is None:
            return None
        record = get_raw(rollout_id)
        return record.owner_id if record is not None else None

    def _record_acquiring(
        self,
        *,
        rollout_id: str,
        image: str,
        artifact_path: str | None,
        displayed_name: str | None,
        task_key: str | None = None,
        group_id: str | None = None,
        fleet_id: str | None = None,
        owner_id: str = "default",
    ) -> None:
        """Write the initial ``acquiring`` row.

        Best-effort: when no StateStore is wired (older test fixtures
        that built the coordinator with ``state=None``) OR when the
        wired StateStore stub doesn't carry the new
        ``record_raw_rollout`` method (some test doubles), silently
        skip persistence. The in-memory ``_sessions`` map remains
        the live coordinator state regardless.
        """
        if self._state is None:
            return
        record_fn = getattr(self._state, "record_raw_rollout", None)
        if record_fn is None:
            return
        from xrlenv.control.state import RawRolloutRecord
        record_fn(RawRolloutRecord(
            rollout_id=rollout_id,
            status="acquiring",
            image=image,
            artifact_path=artifact_path,
            displayed_name=displayed_name,
            task_key=task_key,
            group_id=group_id,
            fleet_id=fleet_id,
            owner_id=owner_id,
            created_at=time.time(),
        ))

    def _update_record(self, rollout_id: str, **fields: Any) -> None:
        """Update the persisted record. Best-effort: missing
        StateStore = no-op; missing method on a test stub = no-op;
        missing-row = log + ignore (avoids cascading failures when
        the coordinator was constructed without a state reference,
        e.g. in some test fixtures)."""
        if self._state is None:
            return
        update_fn = getattr(self._state, "update_raw_rollout", None)
        if update_fn is None:
            return
        try:
            update_fn(rollout_id, **fields)
        except KeyError:
            LOGGER.warning(
                "raw-container.coordinator: update_raw_rollout(%s) "
                "found no row; ignoring (was the record_acquiring "
                "step skipped?)",
                rollout_id,
            )
        except Exception:
            LOGGER.exception(
                "raw-container.coordinator: update_raw_rollout(%s) "
                "failed; continuing",
                rollout_id,
            )

    def _lookup_preferred_home(self, image: str) -> str | None:
        """Mirror of ``AdmissionQueue._lookup_preferred_home``.

        Returns the planner-recorded ``preferred_home`` for the
        image, or ``None`` if no row matches or the StateStore
        doesn't expose the lookup (older test fixtures). Quietly
        absorbs unexpected backend errors so a state-store hiccup
        doesn't wedge an acquire.
        """
        if self._state is None:
            return None
        finder = getattr(self._state, "find_registered_preferred_home", None)
        if finder is None:
            return None
        try:
            result = finder(image)
            return result if isinstance(result, str) else None
        except Exception:
            LOGGER.exception(
                "raw-container.coordinator: find_registered_preferred_home "
                "raised for image=%s; ignoring",
                image,
            )
            return None

    async def exec(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(rollout_id)
        if container_id != session.container_id:
            raise XRLEnvError(
                f"raw-container exec: container_id "
                f"{container_id[:12]!r} does not match the session's "
                f"container ({session.container_id[:12]!r}). The "
                f"caller may be holding a stale handle from a prior "
                f"acquire / destroy cycle.",
            )
        with self._session_rpc(rollout_id):
            return await session.node.container_exec(
                rollout_id=rollout_id,
                container_id=session.container_id,
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
    ) -> None:
        session = self._require_session(rollout_id)
        if container_id != session.container_id:
            raise XRLEnvError(
                f"raw-container apply_egress: container_id "
                f"{container_id[:12]!r} does not match the session's container "
                f"({session.container_id[:12]!r}). The caller may be holding a "
                f"stale handle from a prior acquire / destroy cycle.",
            )
        with self._session_rpc(rollout_id):
            await session.node.apply_egress(
                rollout_id=rollout_id,
                container_id=session.container_id,
                allowlist=allowlist,
                dns_resolver=dns_resolver,
            )

    async def put_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        target_dir: str,
        tarball: bytes,
    ) -> None:
        session = self._require_session(rollout_id)
        if container_id != session.container_id:
            raise XRLEnvError(
                f"raw-container put_archive: container_id "
                f"{container_id[:12]!r} does not match session's "
                f"({session.container_id[:12]!r}).",
            )
        with self._session_rpc(rollout_id):
            await session.node.container_put_archive(
                rollout_id=rollout_id,
                container_id=session.container_id,
                target_dir=target_dir,
                tarball=tarball,
            )

    async def exec_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming-exec coordinator path. Resolves the session
        on the first iteration; once resolved, the session's
        node owns the stream — if the node disconnects mid-stream
        the underlying transport raises ControlPlaneLost via the
        synthetic FAILED reply pushed by ``RemoteNodeTransport.close``.
        """
        session = self._require_session(rollout_id)
        if container_id != session.container_id:
            raise XRLEnvError(
                f"raw-container exec_stream: container_id "
                f"{container_id[:12]!r} does not match session's "
                f"({session.container_id[:12]!r}).",
            )
        with self._session_rpc(rollout_id):
            async for chunk in session.node.container_exec_stream(
                rollout_id=rollout_id,
                container_id=session.container_id,
                cmd=cmd,
                timeout_s=timeout_s,
                cwd=cwd,
                env=env,
                user=user,
            ):
                yield chunk

    async def get_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
    ) -> bytes:
        session = self._require_session(rollout_id)
        if container_id != session.container_id:
            raise XRLEnvError(
                f"raw-container get_archive: container_id "
                f"{container_id[:12]!r} does not match session's "
                f"({session.container_id[:12]!r}).",
            )
        with self._session_rpc(rollout_id):
            return await session.node.container_get_archive(
                rollout_id=rollout_id,
                container_id=session.container_id,
                source_path=source_path,
            )

    async def destroy(
        self,
        *,
        rollout_id: str,
        container_id: str,
        force: bool = True,
        reason: str | None = None,
    ) -> None:
        """Tear down a raw session.

        ``reason`` (issue #18) — when set, a NODE-CONFIRMED teardown seals the ``raw_rollouts``
        row as ``reaped`` with ``error=reason`` instead of the normal ``released``. Used by the
        raw-GC deadline / liveness reaper so an operator can tell a force-reaped session apart
        from a consumer-initiated destroy, without it counting as a workload failure.

        Per-rollout SINGLE-FLIGHT (audit M14): a second destroy for a rollout already tearing
        down returns idempotently.

        A teardown that RAISES or TIMES OUT is NOT node-confirmed, so it RETAINS the session +
        its capacity charge and does NOT seal the row (invariant 2, audit H8) — the raw-GC
        reconciler / a retry re-attempts, and only a confirmed destroy seals + frees. A consumer
        timeout is swallowed (work done); a hard failure / reaper timeout re-raises.
        """
        session = self._require_session(rollout_id)
        if container_id != session.container_id:
            raise XRLEnvError(
                f"raw-container destroy: container_id "
                f"{container_id[:12]!r} does not match the session's "
                f"container ({session.container_id[:12]!r}).",
            )
        # P1.7.C.2 — a compose PROJECT session tears down the WHOLE project via
        # the node's strict ``destroy_compose_project`` (a failed down RETAINS the
        # session + capacity — invariant 2), not the single-container path below.
        # The reap path (deadline/liveness) calls ``destroy(container_id=main)``,
        # so routing here covers both consumer and reaper teardown.
        if session.compose_project_name is not None:
            await self._destroy_compose_session(session, force=force, reason=reason)
            return
        # Issue #18 fix #2 / audit H8: mark the session ``destroying`` BEFORE the
        # wire-level destroy. The session stays in ``_sessions`` and keeps charging
        # ``iter_load_entries`` until the node CONFIRMS teardown. Invariant 2 is strict —
        # capacity is released ONLY on node-confirmed destroy — so a FAILED or TIMED-OUT
        # destroy (the node did not ack) RETAINS the session + its capacity charge and defers
        # teardown to the raw-GC reconciler, exactly like the compose path
        # (:meth:`_destroy_compose_session`). Only a clean, node-confirmed destroy (below)
        # seals the row + frees capacity + wakes admission. The 300 s wire ceiling
        # (``grpc_endpoint.py``) + fix #4's ``destroy_concurrency=4`` size the call to the
        # worst-case overlay-fs teardown latency.
        async with self._lock:
            existing = self._destroying.get(rollout_id)
            if existing is None:
                # We own the single-flight destroy: register a future joiners await.
                self._destroying[rollout_id] = asyncio.get_running_loop().create_future()
        if existing is not None:
            # Single-flight JOIN (audit M14): a destroy for this rollout is already in flight.
            # A second caller (consumer double-destroy, or a reaper racing the consumer) must
            # NOT issue a second wire call, double-seal the row, or double-kick admission. Await
            # the OWNER's terminal outcome and apply THIS caller's contract to it, instead of the
            # old optimistic "return success" that could report done while the owner later fails.
            LOGGER.debug(
                "raw-container.destroy rollout=%s already in flight — joining owner", rollout_id,
            )
            # SHIELD the shared owner-future (audit M14): a bare ``await existing`` would let a
            # cancelled joiner cancel the SHARED future, so the owner's ``set_result`` no-ops
            # (future already done) and every OTHER joiner gets ``CancelledError``. Shielding
            # cancels only this joiner's wait; the shared future survives for owner + siblings.
            outcome = await asyncio.shield(existing)
            if isinstance(outcome, NodeCommandTimeout):
                # A slow node teardown: a consumer swallows it (work done); a reaper re-raises.
                if reason is not None:
                    raise outcome
            elif isinstance(outcome, BaseException):
                raise outcome  # hard failure — the container may still be alive
            return
        try:
            await session.node.destroy_container(
                rollout_id=rollout_id,
                container_id=session.container_id,
                force=force,
            )
        except NodeCommandTimeout as exc:
            # Not node-confirmed. The node is briefly wedged / I/O-saturated and the node-side
            # ``container.remove`` is almost always still in progress. RETAIN the session +
            # its capacity charge (invariant 2) — do NOT seal the row, drop the session, or
            # wake admission. The raw-GC reconciler confirms absence (coordinator-only orphan
            # → :meth:`seal_orphan` frees capacity) or the deadline/liveness sweep re-attempts.
            # A CONSUMER destroy (``reason is None``) still returns success — the rollout's
            # work is done and a slow teardown must not false-fail it; a REAPER destroy
            # (``reason`` set) re-raises so the reaper records it. (Was: seal ``released`` +
            # drop the session + kick admission here — an invariant-2 violation, audit H8.)
            # Resolve the single-flight future with the timeout so a JOINER applies its own
            # contract (audit M14); no session/capacity change (invariant 2 — retained). But
            # CONVERGE on node loss first: if a racing ``handle_node_lost`` already tore the
            # session down mid-wire-call, raise ``NodeLost`` to match every joiner (audit M14)
            # instead of re-applying the timeout contract (re-raise / consumer false-success).
            if await self._resolve_or_converge_on_node_loss(rollout_id, exc):
                raise NodeLost(
                    f"rollout {rollout_id} node-lost during destroy; teardown superseded",
                ) from exc
            LOGGER.warning(
                "raw-container.destroy rollout=%s timed out (%s); session RETAINED "
                "(capacity held — invariant 2), teardown deferred to the raw-GC reconciler",
                rollout_id, exc,
            )
            if reason is not None:
                raise
            return
        except BaseException as exc:
            # Not node-confirmed → the container may still be alive. RETAIN the session +
            # capacity (invariant 2) and re-raise; the raw-GC reconciler / a consumer retry /
            # the deadline+liveness sweep re-attempts teardown, and only a confirmed destroy
            # frees capacity. Mirrors :meth:`_destroy_compose_session`. (Was: seal ``failed``
            # + drop the session + kick admission here — an invariant-2 violation, audit H8.)
            # Resolve the single-flight future with the failure so JOINERS re-raise it too
            # (audit M14); no session/capacity change (invariant 2 — retained). CONVERGE on node
            # loss first (audit M14): a racing ``handle_node_lost`` that tore the session down
            # mid-wire-call already resolved joiners with ``NodeLost`` — raise it here too rather
            # than re-raising this owner's own (now moot) failure.
            if await self._resolve_or_converge_on_node_loss(rollout_id, exc):
                raise NodeLost(
                    f"rollout {rollout_id} node-lost during destroy; teardown superseded",
                ) from exc
            LOGGER.warning(
                "raw-container.destroy rollout=%s FAILED (%r); session RETAINED "
                "(capacity held — invariant 2), teardown deferred to the raw-GC reconciler",
                rollout_id, exc,
            )
            raise
        # ── node-confirmed teardown: finalize atomically w.r.t. cancellation (audit M14) ──
        # The wire destroy is node-confirmed; finalization (seal the row + free capacity + wake
        # admission + resolve the single-flight future) MUST complete even if THIS coroutine is
        # cancelled here — else the row seals terminal while the session/capacity stay charged
        # and ``_destroying`` stays set (which GC + ``seal_orphan`` then skip forever: wedged).
        # ``shield`` runs finalization to completion; a pending CancelledError still propagates.
        await asyncio.shield(self._finalize_confirmed_destroy(rollout_id, reason=reason))

    async def terminate_raw_group(
        self,
        group_id: str,
        reason: str = "group_terminated",
        *,
        owner_id: str | None = None,
    ) -> TerminateRawGroupReport:
        """Destroy every still-running raw container carrying ``group_id`` — the raw-container
        analogue of :meth:`RolloutCoordinator.cancel_group`. A consumer aborting a run (e.g.
        Ctrl-C) calls this so its containers are torn down actively (a node-confirmed destroy
        frees capacity immediately) instead of lingering until the raw-liveness reaper fires.

        Best-effort + idempotent: a row already terminal (or one that never got a container) is
        reported under ``already_terminal``, not re-destroyed; a per-container teardown failure is
        logged and the sweep continues. ``owner_id`` (when set) scopes the sweep to that tenant —
        the ``WHERE owner_id`` filter means a consumer can't terminate another tenant's group by
        guessing its id (mirrors ``cancel_group``).

        Group membership + owner live on the ``raw_rollouts`` row (a live session carries
        neither), so the sweep lists from ``state``; ``reason`` seals a torn-down row as
        ``reaped`` (via :meth:`destroy`) so it reads as a group teardown, not a workload failure.
        """
        terminated: list[str] = []
        already_terminal: list[str] = []
        # Group membership lives on the ``raw_rollouts`` row, so we sweep from state. The raw-listing
        # methods aren't on the narrow ``_StateStoreProtocol`` (some stores omit them), so reach them
        # via ``getattr`` like ``record_raw_rollout`` / ``get_raw_rollout`` do — a store without it
        # (e.g. attached/local mode) simply has nothing to sweep.
        list_raw = getattr(self._state, "list_raw_rollouts", None)
        if list_raw is None:
            return TerminateRawGroupReport(
                group_id=group_id, terminated=(), already_terminal=(),
            )
        for row in list_raw(group_id=group_id, owner_id=owner_id):
            # Non-terminal statuses are ``acquiring`` (no container yet) and ``running``; only a
            # ``running`` row has a container to tear down.
            if row.status != "running" or not row.container_id:
                already_terminal.append(row.rollout_id)
                continue
            try:
                await self.destroy(
                    rollout_id=row.rollout_id,
                    container_id=row.container_id,
                    force=True,
                    reason=reason,
                )
                terminated.append(row.rollout_id)
            except Exception:
                LOGGER.exception(
                    "terminate_raw_group: rollout=%s teardown failed; leaving to the raw-GC "
                    "reconciler (capacity stays charged until node-confirmed)",
                    row.rollout_id,
                )
        return TerminateRawGroupReport(
            group_id=group_id,
            terminated=tuple(terminated),
            already_terminal=tuple(already_terminal),
        )

    def _resolve_destroying_locked(
        self, rollout_id: str, outcome: BaseException | None,
    ) -> None:
        """Resolve + pop the single-flight destroy future (audit M14). Caller holds ``_lock``.

        ``outcome`` is the teardown's terminal result a JOINER awaits: ``None`` (node-confirmed
        / success) or the wire exception object (so the joiner applies its own contract). Stored
        as the future's RESULT (never ``set_exception``) so an un-awaited future never triggers
        asyncio's "exception was never retrieved" warning. Exactly-once + no-op if absent."""
        fut = self._destroying.pop(rollout_id, None)
        if fut is not None and not fut.done():
            fut.set_result(outcome)

    async def _resolve_destroying(
        self, rollout_id: str, outcome: BaseException | None,
    ) -> None:
        async with self._lock:
            self._resolve_destroying_locked(rollout_id, outcome)

    async def _resolve_or_converge_on_node_loss(
        self, rollout_id: str, exc: BaseException,
    ) -> bool:
        """Resolve the single-flight destroy future for a FAILED/TIMED-OUT wire owner, but
        CONVERGE on node loss first (audit M14).

        A racing ``handle_node_lost`` can pop the session + resolve the future with ``NodeLost``
        WHILE this owner's wire ``destroy_container`` is in flight; the wire call then surfaces
        its own timeout/failure. Without this check the owner would re-apply its own contract
        (re-raise the failure, or return a false success for a consumer timeout) while every
        joiner already saw ``NodeLost`` — the exact divergence M14 closes for the success path.

        Under a single lock: if the session is GONE, node loss superseded this teardown → keep
        the future's ``NodeLost`` outcome (resolve defensively in case the pop raced ahead of the
        resolve) and return ``True`` so the caller raises ``NodeLost`` to converge. Otherwise the
        session is still ours → resolve the future with ``exc`` (joiners apply their own contract)
        and return ``False`` so the caller applies its normal timeout/failure policy."""
        async with self._lock:
            if rollout_id not in self._sessions:
                if rollout_id in self._destroying:
                    self._resolve_destroying_locked(
                        rollout_id,
                        NodeLost(
                            f"rollout {rollout_id} node-lost during in-flight destroy",
                        ),
                    )
                return True
            self._resolve_destroying_locked(rollout_id, exc)
            return False

    async def _finalize_confirmed_destroy(
        self, rollout_id: str, *, reason: str | None,
    ) -> None:
        """Finalize a node-confirmed single-container destroy (audit M14/H8): seal the row, drop
        the session + fleet membership + liveness state, resolve the single-flight future, wake
        admission. Exactly-once + cancellation-safe (invoked under ``asyncio.shield``).

        GENERATION-SAFE (audit M14): the whole seal+pop runs under a single lock. If the session
        was already gone when we go to finalize, ``handle_node_lost`` tore it down (node loss)
        DURING our wire call — the ONLY path that can remove an in-flight-destroy's session
        (single-flight blocks a second owner; ``seal_orphan`` skips an in-flight rollout). So the
        owner RAISES ``NodeLost`` to CONVERGE on the same terminal outcome joiners already saw
        (audit M14), instead of returning a stale success — never re-sealing the terminal row or
        firing a second admission kick."""
        finalized = False
        async with self._lock:
            if rollout_id not in self._sessions:
                # node-lost mid-destroy → converge with joiners; don't double-seal / double-kick.
                self._resolve_destroying_locked(rollout_id, None)
                raise NodeLost(
                    f"rollout {rollout_id} node-lost during destroy; teardown superseded",
                )
            else:
                if reason is None:
                    self._update_record(
                        rollout_id, status="released", finished_at=time.time(),
                    )
                else:
                    # Clean teardown from a non-consumer reason (raw-GC deadline / liveness
                    # reap) — seal ``reaped``, NOT ``failed``: the rollout's work did not error;
                    # the platform reclaimed an over-budget/abandoned session. ``reason`` is
                    # recorded so an operator can tell a reap from a clean ``released`` and reaps
                    # don't inflate the failure rate.
                    self._update_record(
                        rollout_id, status="reaped", error=reason, finished_at=time.time(),
                    )
                self._sessions.pop(rollout_id, None)
                # Fleet reservation (phase 1): drop this rollout from its fleet (if any); the
                # reservation is released only when its LAST member is gone (invariant 2).
                self._release_fleet_member_locked(rollout_id)
                self._last_seen_at.pop(rollout_id, None)
                self._inflight_rpcs.pop(rollout_id, None)
                self._heartbeated.discard(rollout_id)
                # Resolve joiners with success LAST, once state is consistent.
                self._resolve_destroying_locked(rollout_id, None)
                finalized = True
        # Issue #18 fix #1: wake the admission queue so a waiter parked on CapacityExhausted can
        # re-place against the freshly-released capacity — only when WE actually freed it.
        if finalized and self._admission is not None:
            self._admission.kick()
        LOGGER.info(
            "raw-container.coordinator.destroy rollout=%s ok=%s%s", rollout_id, finalized,
            "" if finalized else " (already finalized elsewhere — no re-seal)",
        )

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str | None = None,
    ) -> None:
        """P1.7.C.2 — consumer-initiated teardown of a multi-service compose
        project (the ``DestroyComposeProject`` RPC entry point). Validates the
        rollout is an active compose project, then routes to the strict
        project-down path. A stale / non-compose rollout fails loud."""
        session = self._sessions.get(rollout_id)
        if session is None or session.compose_project_name is None:
            raise XRLEnvError(
                f"destroy_compose_project: rollout {rollout_id!r} is not an "
                f"active compose project on this control plane (already destroyed?).",
            )
        if project_name and session.compose_project_name != project_name:
            raise XRLEnvError(
                f"destroy_compose_project: rollout {rollout_id!r} owns project "
                f"{session.compose_project_name!r}, not {project_name!r}.",
            )
        await self._destroy_compose_session(session, force=True, reason=None)

    def is_compose_project(self, rollout_id: str) -> bool:
        """True iff ``rollout_id`` is an active multi-service compose project (audit H10).

        The raw-GC reconciler uses this to route a coordinator-only orphan whose visible
        ``session_kind=raw`` MAIN container is gone: a compose project's ``session_kind=compose``
        sidecars are off the raw diff, so main-absence is NOT node confirmation the whole project
        is down — it must go through node-confirmed whole-project teardown, not a bare seal."""
        session = self._sessions.get(rollout_id)
        return session is not None and session.compose_project_name is not None

    async def destroy_compose_orphan(
        self, *, rollout_id: str, reason: str | None = None,
    ) -> None:
        """H10 — reconcile a coordinator-only orphan that is a COMPOSE PROJECT (its main
        container vanished from the node's raw list).

        Main-absence does NOT confirm the project's ``session_kind=compose`` sidecars are gone
        (they're excluded from the raw-GC inventory), so a bare :meth:`seal_orphan` here would
        release the whole project's AGGREGATE capacity while sidecars may still run — an
        invariant-2 violation. Route to the strict, node-confirmed whole-project down instead:
        capacity is freed ONLY on a confirmed ``docker compose down`` (a failed down RETAINS the
        session + aggregate capacity and re-raises, so the reconciler retries next sweep). No-op
        if the session already finalized (generation-safe)."""
        session = self._sessions.get(rollout_id)
        if session is None:
            return  # already finalized by a concurrent destroy (generation-safe)
        if session.compose_project_name is None:
            raise XRLEnvError(
                f"destroy_compose_orphan: rollout {rollout_id!r} is not a compose project.",
            )
        await self._destroy_compose_session(session, force=True, reason=reason)

    async def _destroy_compose_session(
        self,
        session: RawContainerSession,
        *,
        force: bool,
        reason: str | None,
    ) -> None:
        """Strict, whole-project teardown of a compose session (§3).

        Unlike single-container :meth:`destroy` (whose ``finally`` unconditionally
        drops the session — the orphan reconciler reaps any leftover), a compose
        down is **node-confirmed**: on a failed / timed-out down the node keeps the
        project running, so we **retain** the session + ``_compose_projects`` row
        (capacity held — invariant 2) and re-raise, letting a retry / the raw-GC
        reaper re-attempt. Only a confirmed down seals the row + frees capacity."""
        rollout_id = session.rollout_id
        project = self._compose_projects.get(rollout_id)
        project_name = (
            project.project_name if project is not None
            else session.compose_project_name
        ) or ""
        async with self._lock:
            existing = self._destroying.get(rollout_id)
            # M14: re-check the session is still live UNDER the lock. ``destroy_compose_orphan``
            # snapshots the session BEFORE acquiring single-flight; a concurrent finalization can
            # pop it in between, and issuing a wire down for an already-gone project would be a
            # second down. Only claim ownership when the session is still present.
            session_live = rollout_id in self._sessions
            if existing is None and session_live:
                self._destroying[rollout_id] = asyncio.get_running_loop().create_future()
        if existing is not None:
            # Single-flight JOIN (audit M14): compose teardown is STRICT — a joiner re-raises
            # ANY owner failure (no consumer-swallow), so a duplicate never reports a project
            # destroyed while the owner's down actually failed.
            LOGGER.debug(
                "raw-container.compose-destroy rollout=%s already in flight — joining owner",
                rollout_id,
            )
            # SHIELD the shared owner-future (audit M14) — a cancelled joiner must not cancel it
            # out from under the owner + sibling joiners.
            outcome = await asyncio.shield(existing)
            if isinstance(outcome, BaseException):
                raise outcome
            return
        if not session_live:
            # Already finalized by a concurrent path (stale orphan snapshot, audit M14) — no
            # session left to tear down, so DON'T issue a second wire down.
            LOGGER.debug(
                "raw-container.compose-destroy rollout=%s already finalized — skipping stale down",
                rollout_id,
            )
            return
        try:
            await session.node.destroy_compose_project(
                rollout_id=rollout_id, project_name=project_name, force=force,
            )
        except BaseException as exc:
            # Not node-confirmed → the project may still be up. Retain everything; a subsequent
            # consumer retry or the raw-GC reaper re-attempts. Resolve the single-flight future
            # with the failure so joiners re-raise it too (capacity held — invariant 2). But
            # CONVERGE on node loss first (audit M14): a racing ``handle_node_lost`` can pop the
            # session + resolve the future with ``NodeLost`` while our wire down is in flight —
            # raise ``NodeLost`` to match every joiner instead of re-raising this (now moot) wire
            # error. Mirrors the raw destroy path.
            if await self._resolve_or_converge_on_node_loss(rollout_id, exc):
                raise NodeLost(
                    f"rollout {rollout_id} node-lost during compose destroy; teardown superseded",
                ) from exc
            LOGGER.warning(
                "raw-container.coordinator.compose-destroy rollout=%s project=%s "
                "FAILED — session retained for retry (capacity held): %r",
                rollout_id, project_name, exc,
            )
            raise
        # ── node-confirmed down: finalize atomically w.r.t. cancellation (audit M14) ──
        await asyncio.shield(
            self._finalize_confirmed_compose_destroy(
                rollout_id, project_name, reason=reason,
            )
        )

    async def _finalize_confirmed_compose_destroy(
        self, rollout_id: str, project_name: str, *, reason: str | None,
    ) -> None:
        """Finalize a node-confirmed whole-project down (audit M14): seal the row, drop the
        session + project record + subnet claims + liveness state, resolve the single-flight
        future, drop the persisted row, wake admission. Exactly-once + cancellation-safe
        (invoked under ``asyncio.shield``).

        GENERATION-SAFE (audit M14): if the session was already gone when we finalize,
        ``handle_node_lost`` tore the project down (node loss) during our wire down — the owner
        RAISES ``NodeLost`` to converge with joiners instead of returning a stale success, and
        never re-seals / re-kicks."""
        finalized = False
        async with self._lock:
            if rollout_id not in self._sessions:
                self._resolve_destroying_locked(rollout_id, None)
                raise NodeLost(
                    f"compose rollout {rollout_id} node-lost during destroy; "
                    f"teardown superseded",
                )
            else:
                if reason is None:
                    self._update_record(
                        rollout_id, status="released", finished_at=time.time(),
                    )
                else:
                    self._update_record(
                        rollout_id, status="reaped", error=reason, finished_at=time.time(),
                    )
                self._sessions.pop(rollout_id, None)
                self._compose_projects.pop(rollout_id, None)
                self._pending_subnet_claims.pop(rollout_id, None)
                self._last_seen_at.pop(rollout_id, None)
                self._inflight_rpcs.pop(rollout_id, None)
                self._heartbeated.discard(rollout_id)
                self._resolve_destroying_locked(rollout_id, None)
                finalized = True
        if finalized:
            # Confirmed teardown → drop the persisted row + wake admission (only when WE freed).
            self._delete_compose_project_row(rollout_id)
            if self._admission is not None:
                self._admission.kick()
        LOGGER.info(
            "raw-container.coordinator.compose-destroy rollout=%s project=%s ok=%s",
            rollout_id, project_name, finalized,
        )

    async def seal_orphan(
        self,
        *,
        rollout_id: str,
        container_id: str,
        reason: str | None = None,
    ) -> None:
        """Seal a *coordinator-only orphan* — a session whose docker
        container the node no longer reports (already gone: a node-
        autonomous disk-pressure reap, an OOM kill, or an external
        ``docker rm``).

        Unlike :meth:`destroy`, this performs **no wire-level destroy**.
        The container is already gone on the node (that is what makes it
        a coordinator-only orphan — the raw-GC diff found it in the
        coordinator's sessions but NOT in the node's container list), so
        a destroy RPC would only race. Worse, once the node has dropped
        its own record the RPC fails with a benign ``container '...' not
        registered on this node`` error, and :meth:`destroy`'s
        raised-teardown branch would then seal the row ``failed`` with a
        generic ``destroy_container raised: ...`` message — burying the
        node's real reap cause (audit P3; a live disk-guard smoke hit
        exactly this: a disk-reaped rollout sealed ``failed`` with a
        confusing "was it already destroyed?" error instead of ``reaped``
        with the disk-pressure cause).

        Seals ``reaped`` with ``error=reason`` when the node reported a
        real reap cause, else ``released`` (container vanished for some
        other reason — OOM / external ``docker rm``). Drops the in-memory
        session so its capacity charge is freed and wakes the admission
        queue, symmetric with :meth:`destroy`.
        """
        async with self._lock:
            if rollout_id in self._destroying:
                # Single-flight (audit M14): a ``destroy`` is already tearing this rollout down
                # and owns finalization. Don't let the reconciler's confirmed-absence path
                # ALSO seal the row + pop the session + kick admission — that would
                # double-finalize. The in-flight destroy will seal it exactly once.
                LOGGER.debug(
                    "raw-container.seal_orphan rollout=%s skipped — destroy in flight",
                    rollout_id,
                )
                return
            session = self._sessions.get(rollout_id)
            mismatch = (
                session is not None
                and container_id != session.container_id
            )
        if session is None:
            # Generation-safe (audit M14): between the reconciler's ``list_sessions()`` snapshot
            # and now, a concurrent ``destroy`` already finalized this rollout (sealed the row +
            # popped the session). Re-sealing here would DOUBLE-FINALIZE — rewrite the terminal
            # row (``released`` → ``reaped``) and fire a second admission kick off a stale
            # snapshot. The session is already gone, so there is nothing to seal: no-op.
            LOGGER.debug(
                "raw-container.seal_orphan rollout=%s skipped — session already finalized "
                "(stale orphan snapshot)",
                rollout_id,
            )
            return
        if mismatch:
            # Defensive: coordinator state is internally consistent, so
            # this shouldn't fire. If it does, don't seal the wrong row —
            # surface it so the caller's fallback drops the stale session.
            assert session is not None
            raise XRLEnvError(
                f"raw-container seal_orphan: container_id "
                f"{container_id[:12]!r} does not match the session's "
                f"container ({session.container_id[:12]!r}).",
            )
        if reason is not None:
            self._update_record(
                rollout_id,
                status="reaped",
                error=reason,
                finished_at=time.time(),
            )
        else:
            self._update_record(
                rollout_id,
                status="released",
                finished_at=time.time(),
            )
        async with self._lock:
            self._sessions.pop(rollout_id, None)
            self._resolve_destroying_locked(rollout_id, None)
            # Fleet reservation: release this rollout's fleet membership (if
            # any); reservation dropped when its last member is gone.
            self._release_fleet_member_locked(rollout_id)
            self._last_seen_at.pop(rollout_id, None)
            self._inflight_rpcs.pop(rollout_id, None)
            self._heartbeated.discard(rollout_id)
        if self._admission is not None:
            self._admission.kick()
        LOGGER.info(
            "raw-container.coordinator.seal_orphan rollout=%s status=%s"
            "%s",
            rollout_id,
            "reaped" if reason is not None else "released",
            f" reason={reason!r}" if reason is not None else "",
        )

    async def drop_orphan_session(self, rollout_id: str, container_id: str) -> None:
        """Fallback drop for the raw-GC reconciler when :meth:`seal_orphan` raised (a defensive
        container-id mismatch, or a state-store hiccup): GENERATION-SAFELY drop the in-memory
        session — only if it STILL carries ``container_id`` (don't drop a newer generation) —
        WITH proper fleet / liveness release + admission kick, instead of a bare ``_sessions``
        pop that leaks fleet membership and skips admission (audit Low)."""
        dropped = False
        async with self._lock:
            s = self._sessions.get(rollout_id)
            if s is not None and s.container_id == container_id:
                self._sessions.pop(rollout_id, None)
                self._compose_projects.pop(rollout_id, None)
                self._pending_subnet_claims.pop(rollout_id, None)
                self._release_fleet_member_locked(rollout_id)
                self._last_seen_at.pop(rollout_id, None)
                self._inflight_rpcs.pop(rollout_id, None)
                self._heartbeated.discard(rollout_id)
                self._resolve_destroying_locked(rollout_id, None)
                dropped = True
        if dropped and self._admission is not None:
            self._admission.kick()

    # ── Introspection ──────────────────────────────────────────────────────

    def list_sessions(self) -> list[RawContainerSession]:
        """Snapshot of in-memory sessions. Useful for the admin
        panel + GC integration (P1.7.A.2)."""
        return list(self._sessions.values())

    async def readopt(
        self, row: RawRolloutRecord, node: NodeTransport,
        *, is_current: Callable[[], bool] | None = None,
    ) -> bool:
        """Re-adopt a persisted raw session after a control-plane restart.

        A CP restart wipes ``_sessions``, but the durable
        ``raw_rollouts`` row and the node-side container both survive.
        Without re-adoption the raw-GC reconciler sees the still-running
        container as a node-only orphan and force-destroys it, and the
        startup SQLite sweep seals the row ``failed/lost-on-restart`` —
        killing live work on every restart (the ``lost-on-restart``
        failures). This rebuilds the in-memory session from the row +
        the reconnected node transport so the rollout survives and the
        consumer can resume exec/destroy against it.

        Idempotent and safe: returns ``False`` (no-op) if the rollout
        already has a session or an in-flight acquire, or if the row
        lacks the ``container_id`` / ``node_id`` needed to route to it.
        The row's status stays ``running`` — re-adoption doesn't change
        the lifecycle, only restores the in-memory handle.
        """
        if not row.container_id or not row.node_id:
            return False
        rid = row.rollout_id
        now = time.time()
        # Audit P2: restore the ORIGINAL deadline + effective ResourceSpec
        # persisted at acquire, so a re-adopted session keeps its custom
        # semantics. Fall back to the default deadline / raw footprint for
        # rows written before P2 (deadline_at / effective_resources_json
        # NULL) — matching the pre-P2 behaviour, so nothing regresses.
        row_deadline = getattr(row, "deadline_at", None)
        deadline_at = (
            row_deadline
            if row_deadline is not None
            else row.created_at + self._session_deadline_default_s
        )
        effective_resources = _DEFAULT_RAW_RESOURCES
        resources_json = getattr(row, "effective_resources_json", None)
        if resources_json:
            try:
                effective_resources = ResourceSpec.model_validate_json(
                    resources_json,
                )
            except Exception:
                LOGGER.warning(
                    "raw-container.coordinator.readopt: could not parse "
                    "effective_resources_json for rollout=%s; using the "
                    "default raw footprint", rid,
                )
        async with self._lock:
            existing = self._sessions.get(rid)
            if existing is not None:
                # Already have a session for this rollout. If it routes through a STALE
                # transport — a node-agent reconnected under the same node_id BEFORE the old
                # stream's teardown ran, so readopt-on-connect runs against the new transport
                # while the old session lingers (its own ``_on_disconnected`` then no-ops
                # because the registry already points at the replacement) — RE-ROUTE it to the
                # current transport (audit H11 ownership transfer). The container lives on the
                # same physical node and the new stream is now the authoritative connection, so
                # transferring is correct and, crucially, avoids a deadlock where the survivor is
                # never re-adoptable. If it is already on this transport this is an idempotent
                # re-adopt. Either way the survivor IS accounted → return True (successful
                # transfer) so readopt-on-connect does not fail closed forever.
                if existing.node is not node:
                    # audit H11 — gate the TRANSFER on this transport being CURRENT. A DELAYED
                    # OLD readoption task (its stream already superseded) must NOT steal the
                    # session back from the replacement: it would then fail its final registry
                    # check and its transport-scoped rollback would DELETE the session the
                    # replacement legitimately owns. Only the current transport may take over;
                    # a stale caller returns False (its pass fails closed, and since it never
                    # acquired the session its rollback touches nothing it doesn't own).
                    if is_current is not None and not is_current():
                        LOGGER.info(
                            "raw-container.coordinator.readopt rollout=%s — stale transport is "
                            "no longer current; NOT transferring (H11)", rid,
                        )
                        return False
                    self._sessions[rid] = dataclass_replace(
                        existing, node=node, node_id=row.node_id,
                    )
                    self._last_seen_at[rid] = now
                    LOGGER.info(
                        "raw-container.coordinator.readopt rollout=%s node=%s — re-routed "
                        "stale-generation session to the current transport (H11)",
                        rid, row.node_id,
                    )
                return True
            if rid in self._acquiring_ids:
                # An in-flight acquire owns this rollout; don't clobber it. Fail closed so
                # readopt-on-connect retries once the acquire has registered its session.
                return False
            self._sessions[rid] = RawContainerSession(
                rollout_id=rid,
                node=node,
                node_id=row.node_id,
                container_id=row.container_id,
                container_name=row.container_name or "",
                image=row.image,
                created_at=_dt.datetime.fromtimestamp(
                    row.created_at, _dt.UTC,
                ),
                task_key=row.task_key,
                # Fleet reservation (phase 1): restore the container's fleet
                # membership from its persisted label so the fleet-rebuild pass
                # (rebuild_fleets_from_state) can re-attach this node-confirmed-
                # alive container to its reservation. ``None`` for non-fleet.
                fleet_id=getattr(row, "fleet_id", None),
                effective_resources=effective_resources,
                deadline_at=deadline_at,
                # Audit H11: restore the container runtime so a re-adopted sysbox-runc container
                # still counts against the per-node runtime concurrency cap (else the scheduler
                # would treat it as absent and could over-place past the daemon-protection cap).
                container_runtime=getattr(row, "container_runtime", None),
            )
            # Fresh liveness grace — the consumer needs time to reconnect
            # to the restarted control plane before the liveness reaper
            # would consider this session abandoned.
            self._last_seen_at[rid] = now
        LOGGER.info(
            "raw-container.coordinator.readopt rollout=%s node=%s "
            "container=%s — re-adopted after control-plane restart",
            rid, row.node_id, row.container_id[:12],
        )
        return True

    async def readopt_compose_project(
        self, compose_row: Any, main_container_id: str, node: NodeTransport,
        *, is_current: Callable[[], bool] | None = None,
    ) -> bool:
        """P1.7.C.2 — re-adopt a compose PROJECT after a control-plane restart.

        The compose analog of :meth:`readopt`: a CP restart wipes ``_sessions`` +
        ``_compose_projects``, but the persisted project row + the node-side
        containers survive. Rebuilds the ``main`` session (with the
        ``compose_project_name`` marker + the footprint from the row, so
        ``iter_load_entries`` charges the project again) AND the
        ``_compose_projects`` metadata (project name + subnet claims from the
        row), so teardown routes to a whole-project ``down`` — not a
        single-container destroy that would leak the sidecars. Called by the
        raw-GC reconciler for a node container whose ``xrlenv.compose_project``
        label matches a persisted row (node = ground truth). Idempotent."""
        import json

        rid = compose_row.rollout_id
        footprint = _DEFAULT_RAW_RESOURCES
        try:
            footprint = ResourceSpec.model_validate_json(compose_row.footprint_json)
        except Exception:
            LOGGER.warning(
                "readopt-compose: bad footprint_json for rollout=%s; default", rid,
            )
        try:
            claims = tuple(json.loads(compose_row.subnet_claims_json or "[]"))
        except Exception:
            claims = ()
        now = time.time()
        async with self._lock:
            existing = self._sessions.get(rid)
            if existing is not None:
                # Stale-generation re-route (audit H11) — mirrors :meth:`readopt`. A node-agent
                # reconnect under the same node_id can leave the compose main's session routed
                # through the old transport; re-route it (and refresh the project's node binding)
                # to the current transport so the whole-project survivor stays adoptable rather
                # than deadlocking readopt-on-connect. Idempotent when already on this transport.
                if existing.node is not node:
                    # audit H11 — gate the TRANSFER on currency (mirrors :meth:`readopt`): a stale
                    # old readoption task must NOT steal the project back from the replacement and
                    # then delete it via its failed-pass rollback.
                    if is_current is not None and not is_current():
                        LOGGER.info(
                            "raw-container.coordinator.readopt-compose rollout=%s — stale "
                            "transport no longer current; NOT transferring (H11)", rid,
                        )
                        return False
                    self._sessions[rid] = dataclass_replace(
                        existing, node=node, node_id=compose_row.node_id,
                    )
                    proj = self._compose_projects.get(rid)
                    if proj is not None:
                        self._compose_projects[rid] = dataclass_replace(
                            proj, node_id=compose_row.node_id,
                        )
                    self._last_seen_at[rid] = now
                    LOGGER.info(
                        "raw-container.coordinator.readopt-compose rollout=%s project=%s "
                        "node=%s — re-routed stale-generation session to current transport (H11)",
                        rid, compose_row.project_name, compose_row.node_id,
                    )
                return True
            if rid in self._acquiring_ids:
                return False
            self._sessions[rid] = RawContainerSession(
                rollout_id=rid, node=node, node_id=compose_row.node_id,
                container_id=main_container_id, container_name="",
                image=f"compose:{compose_row.project_name}",
                created_at=_dt.datetime.fromtimestamp(
                    compose_row.created_ts, _dt.UTC,
                ),
                effective_resources=footprint,
                deadline_at=compose_row.created_ts + self._session_deadline_default_s,
                compose_project_name=compose_row.project_name,
            )
            # Members are re-derived at teardown (``docker compose down`` works by
            # project name), so recording just ``main`` here is sufficient.
            self._compose_projects[rid] = _ComposeProjectRecord(
                project_name=compose_row.project_name,
                node_id=compose_row.node_id,
                service_container_ids={"main": main_container_id},
                subnet_claims=claims,
            )
            self._last_seen_at[rid] = now
        LOGGER.info(
            "raw-container.coordinator.readopt-compose rollout=%s project=%s "
            "node=%s — re-adopted after control-plane restart",
            rid, compose_row.project_name, compose_row.node_id,
        )
        return True

    async def handle_node_lost(
        self, node_id: str, *, transport: NodeTransport | None = None,
    ) -> int:
        """Seal every in-flight raw session on a lost node as
        ``failed``/``node_lost``. Returns the number sealed.

        STREAM-GENERATION-SAFE (audit H11): when ``transport`` is given (the stream-close
        path), only sessions whose ``session.node IS that transport`` are sealed — so a stale
        old-stream close that fires AFTER a reconnected replacement re-adopted the node's
        sessions can NOT seal the replacement's live sessions. The watchdog path passes no
        ``transport`` (it lost the whole node id) and falls back to matching by ``node_id``.

        Symmetric with ``Coordinator.handle_node_lost`` for gym/step
        rollouts — which, pre-fix, was the ONLY node-loss seal, so a
        lost node's raw container sessions lingered in ``_sessions``
        forever: never ghosts (so the GC SQLite sweep, which keys off
        ``list_sessions``, skipped them) AND on a node the docker-diff
        sweep can't reach (it's gone from the registry). The result was
        ``running`` rows that never go terminal, inflating every
        operator view and the admin "active containers" count — exactly
        the decoupled-count symptom from the 2026-06-09 analysis (an
        operator restarting a worker node orphaned all its raw rows).

        A lost node is unreachable by definition, so its containers
        can't be destroyed through it; dropping the in-memory session
        also frees its scheduler load (``iter_load_entries`` reads
        ``_sessions``). In-flight *acquires* (``_acquiring_ids``) are
        intentionally left alone — their own coroutine fails against the
        gone node and runs its cleanup. Idempotent — a second call for
        the same node finds no sessions and no-ops.
        """
        async with self._lock:
            lost_ids = [
                rid for rid, s in self._sessions.items()
                if (s.node is transport if transport is not None
                    else s.node_id == node_id)
            ]
            compose_lost = [
                rid for rid in lost_ids if rid in self._compose_projects
            ]
            for rid in lost_ids:
                self._sessions.pop(rid, None)
                # P1.7.C.2 — a compose project is pinned to one node too, so a
                # lost node takes its whole project. Drop the project metadata
                # alongside the session so no stale ``_compose_projects`` row
                # lingers (3b/3c read it for subnet anti-affinity + reconcile;
                # a leftover could falsely block placement). No-op for non-compose.
                self._compose_projects.pop(rid, None)
                self._pending_subnet_claims.pop(rid, None)
                # Fleet reservation: a fleet is pinned to one node, so a lost
                # node takes its whole fleet with it — releasing each member
                # drops the reservation once the last one is gone (the
                # footprint returns to the — now absent — node's freed pool).
                self._release_fleet_member_locked(rid)
                self._last_seen_at.pop(rid, None)
                self._inflight_rpcs.pop(rid, None)
                # audit Low — discard heartbeat membership too (the finalize/seal paths do), so a
                # lost node leaves NO trace in ``_heartbeated``; else the id lingers and could
                # falsely mark a (hypothetically UUID-reused) rollout as already heartbeated.
                self._heartbeated.discard(rid)
                # Resolve any in-flight destroy's single-flight future with the NODE-LOSS
                # outcome (audit M14): a joiner then sees the SAME terminal decision the racing
                # wire owner does (node gone → teardown couldn't be node-confirmed), instead of a
                # false "success" while the owner separately raises its own hard failure.
                self._resolve_destroying_locked(
                    rid, NodeLost(f"node {node_id} lost during in-flight destroy"),
                )
        # A lost node takes its compose projects with it — drop their persisted
        # rows so a CP restart doesn't rebuild a project on a node that's gone.
        for rid in compose_lost:
            self._delete_compose_project_row(rid)
        now = time.time()
        for rid in lost_ids:
            self._update_record(
                rid,
                status="failed",
                error=(
                    f"node_lost: node {node_id} went away; raw session "
                    "sealed (cannot destroy a container through a lost "
                    "node)"
                ),
                finished_at=now,
            )
        if lost_ids:
            LOGGER.warning(
                "raw-container.coordinator.handle_node_lost node=%s "
                "sealed %d in-flight session(s)",
                node_id, len(lost_ids),
            )
        return len(lost_ids)

    def list_acquiring_ids(self) -> set[str]:
        """Issue #18 (audit M1) — snapshot of rollout_ids whose
        ``acquire`` is in flight (row written ``acquiring``, no
        session yet — parked in the admission queue or mid wire-level
        acquire). The raw-GC SQLite reconciler unions this with
        ``list_sessions`` so a legitimately-queued acquire is never
        swept as a ghost, regardless of how large the consumer set
        ``queue_timeout_s`` / ``acquire_timeout_s``."""
        return set(self._acquiring_ids)

    def is_destroying(self, rollout_id: str) -> bool:
        """Issue #18 fix #2 — returns ``True`` while ``destroy``
        is in flight for this rollout_id. Used by the admin panel
        and tests to distinguish a healthy ``running`` session
        from one mid-teardown (both appear in ``list_sessions``)."""
        return rollout_id in self._destroying

    # ── Consumer-liveness reaper ────────────────────────────────────────────

    @contextlib.contextmanager
    def _session_rpc(self, rollout_id: str) -> Iterator[None]:
        """Mark a session-scoped RPC in flight for the body's duration.

        Bumps the liveness clock on entry and exit and holds an in-flight
        count, so the reaper never destroys a session with an open RPC — the
        consumer is connected and waiting (the long-blocking-exec case). The
        count drains to 0 so it can't leak.
        """
        self._last_seen_at[rollout_id] = time.time()
        self._inflight_rpcs[rollout_id] = self._inflight_rpcs.get(rollout_id, 0) + 1
        try:
            yield
        finally:
            remaining = self._inflight_rpcs.get(rollout_id, 0) - 1
            if remaining > 0:
                self._inflight_rpcs[rollout_id] = remaining
            else:
                self._inflight_rpcs.pop(rollout_id, None)
            self._last_seen_at[rollout_id] = time.time()

    def mark_heartbeat(self, rollout_ids: list[str]) -> int:
        """Record explicit heartbeats for the given raw sessions.

        Bumps the liveness clock and flags each as heartbeated — which is what
        *enables* fast liveness-reaping (a session that never heartbeats falls
        back to the deadline cap). Ignores ids that aren't live raw sessions
        (gym/step rollouts, or already gone). Returns the count recognised.
        """
        now = time.time()
        recognised = 0
        for rollout_id in rollout_ids:
            if rollout_id in self._sessions:
                self._last_seen_at[rollout_id] = now
                self._heartbeated.add(rollout_id)
                recognised += 1
        return recognised

    def liveness_reap_candidates(
        self, now: float | None = None,
    ) -> list[RawContainerSession]:
        """Raw sessions whose consumer appears to have gone away.

        A session qualifies only when ALL hold: it heartbeated at least once
        (so we know its consumer runs the keepalive — others are left to the
        deadline cap), it has no session-scoped RPC in flight, and its liveness
        clock is staler than ``liveness_ttl_s``. Pure query — the reconciler
        does the (paced) destroying.
        """
        clock = time.time() if now is None else now
        ttl = self._liveness_ttl_s
        out: list[RawContainerSession] = []
        for rollout_id, session in self._sessions.items():
            if rollout_id not in self._heartbeated:
                continue
            if self._inflight_rpcs.get(rollout_id, 0) > 0:
                continue
            if clock - self._last_seen_at.get(rollout_id, clock) > ttl:
                out.append(session)
        return out

    def iter_load_entries(self) -> list[RawSessionLoad]:
        """Return one :class:`RawSessionLoad` per currently-active
        raw-container session, for the scheduler's capacity gate.

        Wired in at runtime construction via
        :py:meth:`Scheduler.set_raw_session_provider`. Without this
        wiring, the scheduler would be blind to raw-container load
        (raw containers don't pass through ``state.list_sandboxes()``)
        and would silently over-place — see the leak-fix block in
        :py:meth:`acquire` for the commit/release lifecycle that keeps
        ``_pending`` honest during the brief in-flight window.

        Resource footprint comes from the effective ``ResourceSpec``
        stored on the session at acquire time — the same spec
        ``acquire`` passed to ``Scheduler.place`` — so the post-acquire
        load entry charges exactly what the placement decision did,
        including any harness CPU/memory override (P0a).

        **Fleet reservation (phase 1, opt-in — R1).** When ``self._fleets``
        is non-empty this becomes a two-loop *list transformation* — the
        scheduler's summation (``_gather_cluster_load`` / ``capacity``) is
        untouched; only the shape of the list changes:

        1. **Non-fleet sessions** (``fleet_id is None``) — one entry each,
           byte-for-byte the legacy behaviour.
        2. **Fleet members** (``fleet_id`` set + a live reservation) — their
           cpu + mem are covered exactly ONCE by their fleet's footprint entry
           (loop below), so per-member they emit only a **disk-only** entry.
           This is the one documented exception to member suppression: disk
           stays per-container so it isn't under-counted (spec 10). A member
           whose reservation has already been dropped (the brief last-member
           release window) falls through to the non-fleet branch — a
           conservative over-count, never a gap.
        3. **Open fleets** — exactly one footprint entry per reservation,
           carrying the DECLARED peak cpu + mem once (and the reservation's
           ``task_key``, so the fleet counts as one fairness unit).

        With ``self._fleets`` empty and every session ``fleet_id is None``,
        loop (1) is the current code verbatim and loops (2)/(3) add nothing —
        the returned list is identical to the pre-fleet function (the golden
        backward-compat guarantee). See
        ``notes/fleet-reservation-r1-load-accounting.md``.
        """
        entries: list[RawSessionLoad] = []
        # Loops (1)+(2): per-session entries. A session is a fleet member only
        # when it carries a ``fleet_id`` AND that fleet still has a live
        # reservation; otherwise it charges its full footprint as before.
        for s in self._sessions.values():
            # Suppress a member's cpu/mem (covered once by the fleet footprint, loop 3) ONLY when
            # it is a VALIDATED member of a live reservation ON THE SAME NODE (audit H11). A
            # session that merely carries the fleet_id label but isn't in the reservation's
            # membership — or is on a different node than the reservation — must charge its FULL
            # footprint, else its cpu/mem goes uncounted on its node and enables over-placement.
            _res = self._fleets.get(s.fleet_id) if s.fleet_id is not None else None
            if (
                _res is not None
                and s.rollout_id in _res.members
                and s.node_id == _res.node_id
            ):
                entries.append(RawSessionLoad(
                    node_id=s.node_id,
                    template_name=f"raw-fleet-member/{s.image}",
                    effective_resources=_fleet_member_disk_only(
                        s.effective_resources,
                    ),
                    # No task_key: the fleet counts fairness once on its
                    # footprint entry (loop 3), members add nothing.
                    task_key=None,
                    # A fleet member IS a real running container — count it
                    # toward the per-node runtime cap (footprint entry, loop 3,
                    # is NOT a container and stays runtime-None).
                    container_runtime=s.container_runtime,
                ))
            else:
                # Non-fleet session (the common path — identical to the
                # pre-fleet code), OR a fleet-labeled session that isn't a
                # validated same-node member: charge the full container. Conservative.
                entries.append(RawSessionLoad(
                    node_id=s.node_id,
                    template_name=f"raw-container/{s.image}",
                    effective_resources=s.effective_resources,
                    task_key=s.task_key,
                    container_runtime=s.container_runtime,
                ))
        # Loop (3): one footprint entry per open fleet — the fleet's whole
        # cpu+mem load, counted once regardless of member count.
        for f in self._fleets.values():
            entries.append(RawSessionLoad(
                node_id=f.node_id,
                template_name=f"raw-fleet/{f.fleet_id}",
                effective_resources=f.footprint,
                task_key=f.task_key,
            ))
        return entries

    # ── Fleet reservation (phase 1, opt-in) ─────────────────────────────────

    async def _classify_fleet_acquire(
        self,
        *,
        rollout_id: str,
        fleet_id: str | None,
        fleet_footprint: ResourceSpec | None,
        effective_resources: ResourceSpec,
    ) -> tuple[str, FleetReservation | None]:
        """Decide a fleet acquire's role and reserve its slot atomically.

        Returns ``(role, reservation)`` where ``role`` is ``"non_fleet"`` |
        ``"opener"`` | ``"companion"`` | ``"overflow"`` and ``reservation`` is
        the pinned :class:`FleetReservation` for a companion (else ``None``).
        All fleet bookkeeping decisions happen here under ``self._lock`` so
        concurrent acquires for one fleet serialize:

        - **companion** (fleet already reserved, and this container fits within
          the footprint): **reserves the member slot** (adds to
          ``reservation.members``) — so two concurrent companions of one fleet
          can't both pass the check and jointly exceed the footprint. The caller
          rolls the slot back on any later failure
          (``_release_fleet_member_locked``).
        - **overflow** (fleet reserved, but this container would exceed the
          footprint): the consumer's *actual* peak container concurrency
          exceeded its declared footprint — an unavoidable reality when the
          concurrency is dynamic and not perfectly predictable. Rather than
          hard-fail the acquire (which would break the consumer's task), the
          reservation **degrades gracefully**: this container is admitted via
          the NORMAL capacity-gated placement path (charged its own resources,
          co-located wherever it fits, queued if the cluster is full), NOT drawn
          from the reservation and NOT suppressed in accounting. The reservation
          still *guarantees* the declared footprint's worth of slots; anything
          beyond it is best-effort. No slot reserved, no member added.
        - **opener** (fleet not reserved, footprint declared): marks the fleet
          *opening* so a second concurrent opener for the same id is rejected.
          The reservation itself is created by the caller only once the lead
          container is up (mirrors the register-before-commit discipline).

        Raises :class:`XRLEnvError` only for a structural error (companion
        before any reservation exists; duplicate opener) — never for an
        over-budget companion (that's the graceful ``"overflow"`` path).
        """
        if fleet_id is None:
            return "non_fleet", None
        async with self._lock:
            existing = self._fleets.get(fleet_id)
            if existing is not None:
                used_cpu = sum(
                    r.cpu_request for r in existing.members.values()
                )
                used_mem = sum(
                    r.mem_request_bytes for r in existing.members.values()
                )
                if (
                    used_cpu + effective_resources.cpu_request
                    > existing.footprint.cpu_request
                    or used_mem + effective_resources.mem_request_bytes
                    > existing.footprint.mem_request_bytes
                ):
                    # Over the declared footprint — the task's real container
                    # concurrency exceeded what it reserved. Degrade gracefully
                    # (normal placement), don't hard-fail. No slot reserved.
                    LOGGER.warning(
                        "raw-container.coordinator.fleet-overflow fleet=%s: "
                        "companion exceeds the declared footprint (members use "
                        "cpu=%s/mem=%sB, + this cpu=%s/mem=%sB > footprint "
                        "cpu=%s/mem=%sB) — falling back to normal capacity-gated "
                        "placement (best-effort, not drawn from the "
                        "reservation). Raise xrlenv.fleet_cpu_request / "
                        "xrlenv.fleet_mem_request to reserve for this peak.",
                        fleet_id, used_cpu, used_mem,
                        effective_resources.cpu_request,
                        effective_resources.mem_request_bytes,
                        existing.footprint.cpu_request,
                        existing.footprint.mem_request_bytes,
                    )
                    return "overflow", None
                existing.members[rollout_id] = effective_resources
                existing.last_acquire_ts = time.time()
                return "companion", existing
            # No reservation yet for this fleet_id.
            if fleet_footprint is None:
                raise XRLEnvError(
                    f"fleet {fleet_id!r}: a companion (no footprint labels) "
                    f"arrived but no reservation exists — the fleet-opening "
                    f"acquire (which declares xrlenv.fleet_cpu_request / "
                    f"xrlenv.fleet_mem_request) must complete first. Consumer "
                    f"ordering bug, or the opener failed.",
                )
            if fleet_id in self._fleet_opening:
                raise XRLEnvError(
                    f"fleet {fleet_id!r}: a second fleet-opening acquire "
                    f"arrived while the first is still opening. Exactly one "
                    f"opener per fleet (MVP).",
                )
            self._fleet_opening.add(fleet_id)
            return "opener", None

    def _node_by_id(self, node_id: str) -> NodeTransport:
        """The live :class:`NodeTransport` for a fleet's pinned node — a
        companion is placed here directly (no re-scheduling). Raises if the
        node has detached: the fleet cannot continue on a gone node (MVP — no
        cross-node relocation)."""
        for n in self._scheduler.nodes:
            if getattr(n, "node_id", None) == node_id:
                return n
        raise XRLEnvError(
            f"fleet-reserved node {node_id!r} is no longer attached; cannot "
            f"place a companion (MVP: no cross-node relocation).",
        )

    def _release_fleet_member_locked(self, rollout_id: str) -> None:
        """Remove a rollout from whatever fleet holds it and drop the whole
        reservation once its **last** member is gone (invariant 2 — capacity
        released only on the final member's teardown). Caller holds
        ``self._lock``. Idempotent: a rollout in no fleet is a no-op.

        When the reservation is dropped, its persisted StateStore row is
        deleted too — the row is live metadata that must not outlive the fleet
        (not a historical log)."""
        for fleet_id, res in list(self._fleets.items()):
            if rollout_id in res.members:
                del res.members[rollout_id]
                if not res.members:
                    del self._fleets[fleet_id]
                    self._delete_fleet_reservation(fleet_id)
                return

    # ── Fleet reservation persistence (best-effort, like _update_record) ────

    def _persist_fleet_reservation(self, res: FleetReservation) -> None:
        """Persist an open fleet's tiny reservation row so a control-plane
        restart can recover its FOOTPRINT (CP-only; the node never sees it).
        Best-effort: no-op when no StateStore is wired or the store predates
        the fleet methods (older test doubles)."""
        if self._state is None:
            return
        record_fn = getattr(self._state, "record_fleet_reservation", None)
        if record_fn is None:
            return
        from xrlenv.control.state import FleetReservationRecord
        try:
            record_fn(FleetReservationRecord(
                fleet_id=res.fleet_id,
                node_id=res.node_id,
                footprint_json=res.footprint.model_dump_json(),
                task_key=res.task_key,
                owner=res.owner,
                opened_ts=res.opened_ts,
                last_acquire_ts=res.last_acquire_ts,
                container_runtime=res.container_runtime,
            ))
        except Exception:
            LOGGER.exception(
                "raw-container.coordinator: persisting fleet reservation %s "
                "failed; continuing (in-memory state is authoritative)",
                res.fleet_id,
            )

    def _touch_fleet_reservation(self, fleet_id: str, ts: float) -> None:
        """Refresh a reservation row's ``last_acquire_ts`` on a companion
        acquire (keeps the TTL grace window accurate across a restart).
        Best-effort."""
        if self._state is None:
            return
        touch_fn = getattr(self._state, "touch_fleet_reservation", None)
        if touch_fn is None:
            return
        with contextlib.suppress(Exception):
            touch_fn(fleet_id, last_acquire_ts=ts)

    def _delete_fleet_reservation(self, fleet_id: str) -> None:
        """Delete a reservation row when its fleet closes (last member gone /
        reclaim). Best-effort."""
        if self._state is None:
            return
        delete_fn = getattr(self._state, "delete_fleet_reservation", None)
        if delete_fn is None:
            return
        with contextlib.suppress(Exception):
            delete_fn(fleet_id)

    def _persist_compose_project(
        self, *, rollout_id: str, project_name: str, node_id: str,
        footprint: ResourceSpec, subnet_claims_v: tuple[str, ...], owner: str,
    ) -> None:
        """Persist a compose project's tiny live row so a CP restart can recover
        the footprint (CP-only reserve) + subnet claims the node can't re-derive.
        Best-effort: no-op without a StateStore or the compose methods (older
        doubles)."""
        if self._state is None:
            return
        record_fn = getattr(self._state, "record_compose_project", None)
        if record_fn is None:
            return
        import json

        from xrlenv.control.state import ComposeProjectStateRecord
        with contextlib.suppress(Exception):
            record_fn(ComposeProjectStateRecord(
                rollout_id=rollout_id,
                project_name=project_name,
                node_id=node_id,
                footprint_json=footprint.model_dump_json(),
                subnet_claims_json=json.dumps(list(subnet_claims_v)),
                owner=owner,
            ))

    def _delete_compose_project_row(self, rollout_id: str) -> bool:
        """Delete a compose project's row on confirmed teardown. Best-effort — returns True iff
        the row was ACTUALLY deleted (audit Low: the stale-row reaper counts only real
        deletions, not attempts, so a swallowed failure isn't reported as reclaimed)."""
        if self._state is None:
            return False
        delete_fn = getattr(self._state, "delete_compose_project", None)
        if delete_fn is None:
            return False
        try:
            delete_fn(rollout_id)
            return True
        except Exception:
            LOGGER.warning(
                "raw-container.coordinator: delete_compose_project row failed for rollout=%s "
                "(best-effort — the stale-row reaper will retry)", rollout_id, exc_info=True,
            )
            return False

    async def rebuild_fleets_from_state(
        self,
        *,
        now: float | None = None,
        reclaim_after_s: float = FLEET_RESERVATION_TTL_DEFAULT_S,
        allow_reclaim: bool = True,
        raise_on_error: bool = False,
    ) -> tuple[int, int]:
        """Reconstruct in-memory ``_fleets`` from persisted reservation rows +
        the live (re-adopted) fleet-member sessions. Idempotent — the raw-GC
        reconciler calls it every sweep, AFTER re-adoption has rebuilt the
        member sessions.

        The footprint comes from the persisted row (it never reached the node —
        spec 21); **membership comes from the live sessions** (``readopt`` only
        rebuilds a session for a container the node still reports, so a
        fleet-member session existing == that container is alive). This is the
        node-is-ground-truth rebuild the restart-safety decision calls for.

        - A persisted row with ≥1 live member and no live in-memory reservation
          yet ⇒ recreate the ``FleetReservation`` (footprint from the row,
          members from the live sessions).
        - A persisted row with **no** live members whose ``last_acquire_ts`` is
          older than ``reclaim_after_s`` ⇒ **reclaim** (delete the row) — but
          only when ``allow_reclaim`` (the reconciler passes ``False`` until its
          re-adoption grace elapses, so a not-yet-reconnected node's fleet isn't
          reclaimed prematurely).

        Returns ``(rebuilt, reclaimed)``. Best-effort: no-op without a
        StateStore or the fleet methods.
        """
        if self._state is None:
            return 0, 0
        list_fn = getattr(self._state, "list_fleet_reservations", None)
        if list_fn is None:
            return 0, 0
        clock = time.time() if now is None else now
        try:
            rows = list_fn()
        except Exception:
            LOGGER.exception(
                "raw-container.coordinator.rebuild_fleets: "
                "list_fleet_reservations failed",
            )
            # audit H11: on the readopt-on-connect path the caller must FAIL CLOSED when fleet
            # rows are unreadable (else it silently admits with fleets unaccounted). The periodic
            # sweep keeps the best-effort default.
            if raise_on_error:
                raise
            return 0, 0
        rebuilt = 0
        reclaim_ids: list[str] = []
        async with self._lock:
            # Live fleet-member sessions grouped by fleet_id, carrying each member's node so a row
            # can be validated against its members' node (audit H11) — a fleet is pinned to ONE
            # node, so a member on a different node than the row is corrupt inventory and must not
            # be folded in (that would suppress its load on one node while charging another).
            members_by_fleet: dict[str, dict[str, tuple[str | None, ResourceSpec]]] = {}
            for s in self._sessions.values():
                if s.fleet_id is not None:
                    members_by_fleet.setdefault(s.fleet_id, {})[
                        s.rollout_id
                    ] = (s.node_id, s.effective_resources)
            for row in rows:
                if row.fleet_id in self._fleets:
                    continue  # already live (steady state or already rebuilt)
                # Only members on THIS row's node count (node-ownership validation, H11).
                live = {
                    rid: res
                    for rid, (nid, res) in members_by_fleet.get(row.fleet_id, {}).items()
                    if nid == row.node_id
                }
                if live:
                    try:
                        footprint = ResourceSpec.model_validate_json(
                            row.footprint_json,
                        )
                    except Exception:
                        # A live fleet (the node reports member containers) whose declared PEAK
                        # is unreadable is a corruption signal. On the readopt-on-connect path
                        # (raise_on_error) we must FAIL CLOSED (audit H11): skipping leaves the
                        # peak un-reserved, so a later companion of this fleet could over-place
                        # into capacity the fleet still owns. The periodic sweep stays best-effort
                        # (skip): its members already charge their full standalone footprints, a
                        # conservative over-count until the next clean rebuild.
                        if raise_on_error:
                            LOGGER.error(
                                "raw-container.coordinator.rebuild_fleets: bad footprint_json "
                                "for LIVE fleet %s on reconnect — NOT admitting (fail closed, H11)",
                                row.fleet_id,
                            )
                            raise
                        LOGGER.warning(
                            "raw-container.coordinator.rebuild_fleets: bad "
                            "footprint_json for fleet %s; skipping",
                            row.fleet_id,
                        )
                        continue
                    self._fleets[row.fleet_id] = FleetReservation(
                        fleet_id=row.fleet_id,
                        node_id=row.node_id,
                        footprint=footprint,
                        members=dict(live),
                        opened_ts=row.opened_ts,
                        last_acquire_ts=row.last_acquire_ts,
                        task_key=row.task_key,
                        owner=row.owner,
                        # §5.3 — recover the fleet's runtime so rebuilt
                        # companions are still validated against it.
                        container_runtime=row.container_runtime,
                    )
                    rebuilt += 1
                elif (
                    allow_reclaim
                    and (clock - row.last_acquire_ts) > reclaim_after_s
                ):
                    reclaim_ids.append(row.fleet_id)
        for fid in reclaim_ids:
            self._delete_fleet_reservation(fid)
        if rebuilt or reclaim_ids:
            LOGGER.info(
                "raw-container.coordinator.rebuild_fleets: rebuilt=%d "
                "reclaimed=%d", rebuilt, len(reclaim_ids),
            )
        return rebuilt, len(reclaim_ids)

    async def reap_stale_compose_projects(
        self,
        *,
        now: float | None = None,
        reclaim_after_s: float = FLEET_RESERVATION_TTL_DEFAULT_S,
        allow_reclaim: bool = True,
    ) -> int:
        """Reclaim persisted compose rows with no live project past the TTL.

        The compose analog of the reclaim half of :meth:`rebuild_fleets_from_state`
        (the *rebuild* half is :meth:`readopt_compose_project`, driven per-node by
        the reconciler). A row is reclaimed (deleted) when its ``rollout_id`` has
        **no live** ``_compose_projects`` entry — re-adoption rebuilds one every
        sweep for any project whose node still reports the main container, so a row
        with no live project means the project is gone — AND ``created_ts`` is older
        than ``reclaim_after_s``. This is a backstop: the confirmed-teardown and
        node-loss paths delete rows directly; this catches a row those paths missed
        (e.g. a CP crash between the node-side down and ``_delete_compose_project_row``)
        so ``compose_projects`` doesn't accumulate ghosts that would falsely block
        subnet placement on restart.

        Gated on ``allow_reclaim`` (the reconciler passes ``False`` until its
        re-adoption grace elapses) so a not-yet-reconnected node's project isn't
        reclaimed before it has a chance to be re-adopted. Returns the count
        reclaimed. Best-effort: no-op without a StateStore or the compose methods.
        """
        if self._state is None or not allow_reclaim:
            return 0
        list_fn = getattr(self._state, "list_compose_projects", None)
        if list_fn is None:
            return 0
        clock = time.time() if now is None else now
        try:
            rows = list_fn()
        except Exception:
            LOGGER.exception(
                "raw-container.coordinator.reap_stale_compose: "
                "list_compose_projects failed",
            )
            return 0
        async with self._lock:
            reclaim_ids = [
                row.rollout_id for row in rows
                if row.rollout_id not in self._compose_projects
                and (clock - row.created_ts) > reclaim_after_s
            ]
        # Count only ACTUAL deletions (audit Low) — a swallowed delete failure leaves the row,
        # so the next sweep retries it; reporting it reclaimed would hide the lingering ghost.
        reclaimed = sum(
            1 for rid in reclaim_ids if self._delete_compose_project_row(rid)
        )
        if reclaimed:
            LOGGER.info(
                "raw-container.coordinator.reap_stale_compose: reclaimed=%d "
                "stale compose-project row(s)", reclaimed,
            )
        return reclaimed

    # ── Internal ───────────────────────────────────────────────────────────

    def _require_session(self, rollout_id: str) -> RawContainerSession:
        session = self._sessions.get(rollout_id)
        if session is None:
            raise XRLEnvError(
                f"raw-container session: rollout {rollout_id!r} not "
                f"found. Acquire first.",
            )
        return session

    # _pick_node was the P1.7.A.1 first-available stub; replaced
    # in P1.7.B.2 by the standard placement flow above (which calls
    # ``Scheduler.place(...)`` — same as case-1). The scheduler
    # raises clear errors when no node has the docker backend or
    # capacity is exhausted; no separate fallback needed here.
