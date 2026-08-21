"""StateStore Protocol + in-memory and sqlite implementations (spec 20).

Phase 0 keeps the schema small but consistent with the spec-20 shape so the
phase-1 redis migration is just a different backend behind the same Protocol.
Step bodies live in their own ``rollout_steps`` table (sqlite) / list (in-mem)
so a single step append is O(1) — task #12 then moves the body to disk-jsonl
under spec 00 invariant 8 ("state store metadata; blobs live on disk").

Identifiers are kept strictly separate per invariants 1 and 9: ``rollout_id``
and ``sandbox_id`` are independent columns; ``task_key`` is fairness, never
identity.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from xrlenv.types import RolloutStatus, Step, Trajectory

# ──────────────────────────────────────────────────────────────────────────────
# Records (the Protocol-level shape; all backends return / accept these)
# ──────────────────────────────────────────────────────────────────────────────


class RolloutRecord(BaseModel):
    """One row in the conceptual ``rollouts`` table."""

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    template: str
    status: RolloutStatus
    reason: str | None = None
    request_id: str | None = None
    task_key: str | None = None
    group_id: str | None = None
    owner_id: str = "default"
    project_id: str = "default"
    run_id: str = "default"
    node_id: str | None = None
    sandbox_id: str | None = None
    init_params: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    final_reward: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    trajectory_sink: str | None = None
    trajectory_node_id: str | None = None
    trajectory_uri: str | None = None
    trajectory_size_bytes: int | None = None
    created_at: float = Field(default_factory=time.time)
    last_touched_at: float = Field(default_factory=time.time)


class SandboxRecord(BaseModel):
    """One row in the conceptual ``sandboxes`` table."""

    model_config = ConfigDict(extra="forbid")

    sandbox_id: str
    backend: str
    backend_ref: str
    stub_endpoint: str
    template: str
    image: str | None = None
    node_id: str
    rollout_id: str | None = None
    status: str = "running"  # running | destroying | destroyed
    owner_count: int = 1
    created_at: float = Field(default_factory=time.time)
    effective_resources_json: str | None = None
    """Pattern A snapshot — JSON-serialised :class:`ResourceSpec` from
    the post-resolver overlay. ``None`` for Simple / Pattern B
    templates whose effective resources match the outer manifest's.
    The scheduler reads this when computing per-node load so a heavy
    per-task instance (e.g. ``sqlite-schema`` asking 2 CPU / 4 GiB
    while the outer manifest declares 1 CPU / 2 GiB) gets correctly
    counted instead of charged at the outer manifest's rate. See
    ``notes/phase-0-acceptance.md`` §2.5."""


class PendingRolloutRecord(BaseModel):
    """One row in the ``pending_rollouts`` admission queue (spec 20)."""

    model_config = ConfigDict(extra="forbid")

    pending_id: str
    template: str
    init_params: dict[str, Any]
    request_id: str | None = None
    task_key: str | None = None
    group_id: str | None = None
    owner_id: str = "default"
    """Tenant the queued acquire belongs to (multi-user fair-share). Carried
    on the in-memory waiter for the admission cap gate; not persisted in
    ``pending_rollouts`` today (restart recovery re-derives it)."""
    deadline_json: dict[str, Any] = Field(default_factory=dict)
    queue_partition: str = "default"
    submitted_at: float = Field(default_factory=time.time)


class FairnessOwnerOverride(BaseModel):
    """Operator override for one tenant's fair-share (multi-user)."""

    model_config = ConfigDict(extra="forbid")

    owner_id: str
    weight: float = 1.0
    """Legacy field retained so older operator state can still be read."""
    hard_cap: int | None = None
    """Owner-specific cap override. ``None`` means use the default cap."""
    uncapped: bool = False
    """When true the owner bypasses fair-share caps; scheduler resources still
    decide whether a new admission can actually run."""
    blocked: bool = False
    """When true the owner gets cap 0 — no *new* admissions. Running jobs are
    never killed (soft throttle); use cancel_group/cancel_rollout to reclaim."""


class FairnessPolicy(BaseModel):
    """Live, operator-tunable fair-share policy (multi-user).

    **Off by default**: ``capacity_basis is None`` means no per-owner cap is
    applied at all — the admission queue behaves exactly as before. Fairness
    engages only once an operator sets a default per-owner capacity via
    ``xrlenv fairshare set`` (or the admin panel). Read fresh on every drain
    pass, so edits take effect without a control-plane restart and never kill
    running work.
    """

    model_config = ConfigDict(extra="forbid")

    capacity_basis: int | None = None
    """Default per-owner concurrent-sandbox cap.

    ``None`` disables fairness. Real cluster capacity is still enforced by the
    scheduler/node resource gates; this value only throttles each owner so one
    tenant cannot keep taking new admissions forever.
    """
    floor: int = 1
    """Legacy minimum retained for older stored policy / CLI compatibility.

    The current per-owner cap policy does not divide capacity across active
    owners, so the floor is no longer part of cap computation.
    """
    overrides: dict[str, FairnessOwnerOverride] = Field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.capacity_basis is not None

    def cap_for(
        self, owner_id: str, _active_owners: set[str] | list[str],
    ) -> int | None:
        """The max concurrent sandboxes ``owner_id`` may hold right now.

        ``None`` means uncapped (fairness disabled). When enabled, the global
        ``capacity_basis`` is a default **per-owner** cap: every owner can be
        admitted up to that count if the scheduler has real resources. Paused
        blocked owners receive cap 0; uncapped owners bypass this gate; an
        owner ``hard_cap`` overrides the default cap.
        """
        if self.capacity_basis is None:
            return None
        my_ov = self.overrides.get(owner_id)
        if my_ov is not None and my_ov.blocked:
            return 0
        if my_ov is not None and my_ov.uncapped:
            return None
        if my_ov is not None and my_ov.hard_cap is not None:
            return my_ov.hard_cap
        return self.capacity_basis


class EventRecord(BaseModel):
    """One row in the append-only ``events`` log."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    ts: float
    rollout_id: str | None
    sandbox_id: str | None
    kind: str
    payload: dict[str, Any]


class NodeRecord(BaseModel):
    """One row in the ``nodes`` table — live registry mirror.

    The :class:`NodeRegistry` is the source of truth in the running
    control-plane process; this table is its persistent shadow so
    out-of-process callers (the operator CLI, future admin RPC clients)
    can see who's currently attached without going through gRPC.

    Status is binary: ``connected`` (heartbeating) or ``lost`` (the
    watchdog tripped). The normal register/deregister/heartbeat path never
    deletes a row — the registry keeps a history of attachments so an
    operator can spot a flapping node. The ONE exception is startup
    reconciliation: ``prune_lost_nodes(keep=...)`` reaps ``lost`` rows for
    node_ids absent from the current ``nodes.yaml`` roster (a decommissioned
    host — e.g. an IP-derived node_id orphaned by a cluster reboot), so the
    registry doesn't accumulate cruft across reboots. ``connected`` rows and
    rostered node_ids are always kept.
    Use ``list_nodes(status='connected')`` for the live cohort.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    status: str = "connected"
    connected_at: float
    last_seen_at: float
    stream_epoch: str | None = None
    instance_id: str | None = None
    backends: list[str] = Field(default_factory=list)
    # P6 step-2c (observability only) — the node's advertised CPU-isolation
    # capability (from NodeHello) + last-known pinnable-CPU counts (from the
    # heartbeat). Mirrored here so the out-of-process ``xrlenv nodes`` CLI +
    # admin ``/nodes`` view can show them. ``(0, 0)`` pinned counts = "unknown"
    # (pre-report / non-capable node). Nothing schedules on these — the floor
    # math + placement predicate land in later P6 steps.
    isolation_capable: bool = False
    pinned_cpus_free: int = 0
    pinned_cpus_total: int = 0


class AuditRecord(BaseModel):
    """One row in the append-only ``audit`` table (spec 19).

    Separate from :class:`EventRecord` because spec 20's retention
    matrix gives audit its own window (``audit_retention_days``, 30-day
    default) distinct from the generic events log's 14 days, *and* the
    columns are different (no rollout/sandbox FK; identity + scope live
    here instead). Both are swept by the state-retention janitor. Never
    store the raw bearer token — ``digest_hint`` is the first 6 hex chars
    of its SHA-256.
    """

    model_config = ConfigDict(extra="forbid")

    seq: int
    ts: float
    kind: str
    role: str | None = None
    digest_hint: str | None = None
    method: str | None = None
    source: str | None = None
    result: str = "ok"
    payload: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# P1.6.b — control-plane-driven image build records
# ──────────────────────────────────────────────────────────────────────────────


BuildPlanStatus = Literal[
    "in_flight", "completed", "partial_failure", "cancelled", "superseded",
]
BuildAssignmentStatus = Literal[
    # Eager/legacy: bin-packer placed it, queued for synchronous build
    # by the build coordinator (P1.6.b/c semantics).
    "pending",
    # P1.6.g (H3 lazy lifecycle, F1=2): intent declared at apply time;
    # build deferred to a later ``ensure_present`` call (rollout init,
    # admission pre-fetch, or operator-triggered warm-up).
    "registered",
    # Build is in flight on the node.
    "building",
    # Built successfully and currently present locally.
    "done",
    # Was ``done``, evicted from the local cache. The lazy hook
    # rebuilds on next ``ensure_present`` for this ref.
    "evicted",
    # Build attempt failed; ``error`` carries the message.
    "failed",
    # Operator-cancelled mid-build (or before the build started)
    # via ``xrlenv build cancel``. Distinct from ``failed`` so the
    # admin panel + status tooling can tell "we killed it" apart
    # from "build failed for a real reason."
    "cancelled",
]


class BuildPlanRecord(BaseModel):
    """One row in ``build_plans``."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    applied_at: float
    applied_by: str
    plan_json: str
    status: BuildPlanStatus
    name: str | None = None
    """Operator-friendly label echoed from ``BuildPlan.name``.
    Persisted so the admin panel can show the latest applied
    name without re-parsing ``plan_json``. Renames update this
    field on the next apply; the ``plan_id`` itself doesn't change."""


class BuildAssignmentRecord(BaseModel):
    """One row in ``build_plan_assignments``."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    node_id: str
    image_ref: str
    benchmark: str
    status: BuildAssignmentStatus
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Raw rollout records (P1.7.B.3 — case-2/3 evaluation harness tracking).
#
# Separate from RolloutRecord because the two have genuinely different
# shapes (case-1 carries trajectory steps + final_reward + per-rollout
# verifier output; case-2/3 raw containers carry container_id +
# container_name + harness-on-consumer-side metadata). See
# notes/p1-7-b-3-rollout-tracking-plan.md §2 D2.
# ──────────────────────────────────────────────────────────────────────────────


RawRolloutStatus = Literal[
    # AcquireContainerCommand dispatched; awaiting node ack.
    "acquiring",
    # Container alive on the chosen node; harness running.
    "running",
    # destroy_container succeeded (normal end-of-life). Paired
    # with ``acquire``; symmetric verb pair so the operator sees
    # "released" rather than the alarming-sounding "destroyed".
    "released",
    # Operator-initiated cancel before normal end (or sweep-stuck-
    # transient on control-plane restart).
    "cancelled",
    # Acquire-time error (image missing, scheduler rejected, etc.)
    # OR mid-run failure (node disconnect, exec-timeout cascade).
    # ``error`` field carries the diagnostic message.
    "failed",
    # GC-reclaimed by the raw-GC reconciler: the session outlived its
    # wall-clock deadline (``session_deadline_s`` / the default cap) or
    # its consumer-liveness TTL, and was force-destroyed cleanly.
    # Deliberately distinct from ``failed`` — the rollout's work did not
    # error; the platform reclaimed an over-budget or abandoned session.
    # ``error`` carries the reap reason so an operator can tell a reap
    # apart from a clean ``released``. See spec 09 GC layer.
    "reaped",
    # The control plane declined to PLACE the acquire within
    # ``queue_timeout_s`` — the pool was at capacity the whole wait, so
    # ``CapacityExhausted`` fired. Deliberately distinct from ``failed``
    # (like ``reaped``): the rollout's work never ran and nothing errored;
    # this is backpressure/pacing. A consumer that retries (the sweep's
    # infra-retry loop does) typically succeeds on a later attempt, so
    # counting these as failures paints a green run red. ``error`` carries
    # the original CapacityExhausted text plus the two operator levers
    # (raise ``queue_timeout_s`` / own the retry). See spec 13 (admin
    # categorization) + spec 20 (this enum).
    "capacity_rejected",
]


class RawRolloutRecord(BaseModel):
    """One row in ``raw_rollouts`` (P1.7.B.3).

    Tracks the lifecycle of a case-2/3 raw-container session
    (``Client.acquire_container`` flow + the docker-py drop-in
    ``containers.create`` flow). The audience's harness is opaque
    to the cluster; what we know is what comes over the spec-21
    wire (acquire / exec / destroy) plus two operator-facing
    metadata fields (``artifact_path``, ``displayed_name``) that
    the smoke driver provides via ``xrlenv.rollout_metadata(...)``.

    ``rollout_id`` is the cluster's primary key — durable across
    container churn. ``container_id`` is volatile (future
    snapshot/resume rotates it without touching ``rollout_id``).
    Spec-00 invariant 1 ("Sandbox identity ≠ rollout identity")
    applies to case-2/3 from day 1.
    """

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    status: RawRolloutStatus
    image: str
    node_id: str | None = None
    """``None`` during the brief ``acquiring`` window before the
    coordinator's scheduler decision lands. Set once the
    AcquireContainerCommand is dispatched."""
    container_id: str | None = None
    """``None`` until the node-side dispatcher acks the acquire
    with the docker-minted id. Volatile — future snapshot/resume
    can rotate this while ``rollout_id`` stays stable."""
    container_name: str | None = None
    artifact_path: str | None = None
    """From ``xrlenv.rollout_metadata(artifact_path=...)`` — the
    consumer-side filesystem path where the harness's per-instance
    artifacts (logs/run_evaluation/...) live. Admin's per-rollout
    detail page renders this directory inline iff the path
    resolves on the control-plane host's filesystem; otherwise
    shows the path as a string. Never uploaded over the wire."""
    displayed_name: str | None = None
    """From ``xrlenv.rollout_metadata(displayed_name=...)`` — the
    operator-friendly name for the admin's ``/rollouts`` row
    (e.g. ``"astropy__astropy-7166"`` instead of the synthetic
    rollout uuid). Optional; admin falls back to a short prefix
    of ``rollout_id`` when None."""
    task_key: str | None = None
    """The ``acquire_container(task_key=...)`` value — set
    explicitly by the SDK or promoted from the ``xrlenv.task_key``
    docker label by ``compat.create_container``. The scheduler reads
    this for anti-affinity (``max_runs_per_task``); admin surfaces
    it so operators can group rollouts of the same logical task.
    Optional; ``None`` when no anti-affinity was requested."""
    group_id: str | None = None
    """The ``xrlenv.group_id`` docker label, when set by the
    operator on the incoming labels dict. Never xrlenv-emitted —
    purely a caller-supplied annotation. Lets operators tie all
    rollouts of one batch / run / sweep together so admin's
    ``/rollouts`` can filter "show me this run's tasks" without
    needing a separate registry. Optional; ``None`` when absent."""
    fleet_id: str | None = None
    """The ``xrlenv.fleet_id`` docker label (fleet reservation, phase 1),
    parsed at acquire exactly like ``task_key``/``group_id``. Persisted so a
    control-plane restart can tell which re-adopted (node-confirmed-alive)
    containers belong to which fleet, and rebuild ``_fleets`` membership from
    the live containers — the footprint itself comes from the separate
    ``fleet_reservations`` row (spec 21: only ``fleet_id`` reaches the node).
    ``None`` for every non-fleet container."""
    owner_id: str = "default"
    """Tenant the acquiring consumer authenticated as (multi-user).
    Server-stamped from the verified bearer token at acquire time —
    never trusted from a client-supplied field — so admin's ``/rollouts``
    view can scope each user to their own sessions. ``"default"`` for the
    legacy shared token and for single-tenant / embedded-mode runs."""
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    """Set when status transitions to a terminal state — ``released`` /
    ``cancelled`` / ``failed`` / ``reaped``."""
    error: str | None = None
    """Populated when status == ``"failed"``, ``"reaped"`` (carries the
    raw-GC reap reason), or sometimes ``cancelled`` (if the cancel
    carried a reason)."""
    deadline_at: float | None = None
    """Resolved wall-clock reap deadline (epoch seconds) — the
    consumer-supplied ``session_deadline_s`` or the coordinator default,
    fixed at acquire time. Persisted so a control-plane restart can
    re-adopt the session with its ORIGINAL deadline instead of resetting
    the clock to ``created_at + default`` (audit P2). ``None`` on rows
    written before this field / by test doubles that don't set it."""
    effective_resources_json: str | None = None
    """The effective ``ResourceSpec`` the session was placed with
    (``model_dump_json``), so re-adoption restores the real CPU/memory
    footprint for scheduler load accounting rather than reverting to the
    raw default (audit P2). ``None`` = unknown → fall back to the
    default raw footprint."""
    container_runtime: str | None = None
    """The container runtime the session runs as (e.g. ``sysbox-runc``), persisted so a
    control-plane restart can restore it on re-adoption (audit H11). The per-node per-runtime
    concurrency cap (daemon protection) counts by ``RawContainerSession.container_runtime``; a
    re-adopted session that lost this would be counted as absent, letting the scheduler exceed
    the cap for surviving runtime-specific containers. ``None`` = default runtime."""


class FleetReservationRecord(BaseModel):
    """One row in ``fleet_reservations`` — tiny **live** scheduler metadata for
    one open fleet reservation (phase 1, opt-in).

    NOT a historical log and NOT blob storage (spec-00 invariant 6): the row is
    written when a fleet **opens** and **DELETED** when its last member is
    node-confirmed destroyed. One row per *currently open* fleet, so the table
    is bounded by live concurrency, never by cumulative rollout history — it
    does not grow with artifacts or container disk.

    Its sole purpose is control-plane restart recovery of the **footprint**:
    the footprint is consumer-declared CP-side accounting and never reaches the
    node (spec 21 — the node stays fleet-unaware), so it cannot be re-derived
    from live containers. On restart the footprint comes from this row while the
    live **member set** is rebuilt from the containers' ``xrlenv.fleet_id``
    labels (node = ground truth); a row with no live members past the
    opening/TTL window is reclaimed. Members are deliberately NOT stored here —
    the node's live containers are the source of truth for membership.
    """

    model_config = ConfigDict(extra="forbid")

    fleet_id: str
    node_id: str
    footprint_json: str
    """The declared peak :class:`~xrlenv.backends.base.ResourceSpec`
    (``model_dump_json``) — cpu + mem in v1 (disk is 0; disk stays per-member).
    Stored as JSON to mirror ``RawRolloutRecord.effective_resources_json`` and
    stay forward-compatible if the footprint gains axes later."""
    task_key: str | None = None
    owner: str = "default"
    opened_ts: float = Field(default_factory=time.time)
    last_acquire_ts: float = 0.0
    """Refreshed on every companion acquire — drives the
    ``fleet_reservation_ttl_s`` reclaim of a fleet whose consumer crashed
    without destroying cleanly (an EMPTY reservation past its TTL)."""
    container_runtime: str | None = None
    """§5.3 — the OCI runtime the fleet runs under (e.g. ``sysbox-runc``),
    recovered on CP restart so a rebuilt reservation still validates its
    companions' runtime and re-checks the pinned node advertises it. ``None``
    (the default, ordinary runc fleets) is unchanged; a store column that
    predates this field reads back as ``None`` (byte-compatible)."""


class ComposeProjectStateRecord(BaseModel):
    """One row in ``compose_projects`` — tiny **live** metadata for one open
    multi-service compose project (P1.7.C.2).

    Like :class:`FleetReservationRecord`, this is NOT a historical log and NOT
    blob storage (invariant 6): written when a project is acquired, **DELETED** on
    node-confirmed teardown. One row per *currently running* project. Its purpose
    is control-plane restart recovery of the CP-only accounting the node can't
    re-derive — the whole-stack **footprint** (the scheduler reserve) and the
    pinned **subnet claims** (node-exclusive anti-affinity). On restart the
    ``main`` session + ``_compose_projects`` are rebuilt from these rows correlated
    with the node's live containers (via the CP-stamped ``xrlenv.rollout_id`` /
    ``xrlenv.compose_project`` labels — node = ground truth for membership); a row
    whose node reports no live containers past the TTL is reclaimed. Keyed by
    ``rollout_id`` (1:1 with the project's ``main`` session)."""

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    project_name: str
    node_id: str
    footprint_json: str
    """The whole-stack :class:`~xrlenv.backends.base.ResourceSpec`
    (``model_dump_json``) — the scheduler reserve, CP-only, not re-derivable."""
    subnet_claims_json: str = "[]"
    """JSON list of pinned CIDRs (``networks.*.ipam.config[].subnet``) for
    node-exclusive subnet anti-affinity, rebuilt into ``_compose_projects`` on
    restart. ``[]`` for service-DNS-only projects."""
    owner: str = "default"
    created_ts: float = Field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory-construction helpers
# ──────────────────────────────────────────────────────────────────────────────


def _metadata_with_node_id(
    base: dict[str, Any] | None,
    node_id: str | None,
) -> dict[str, Any]:
    """Merge ``node_id`` into a trajectory metadata dict.

    ``Trajectory`` has no top-level ``node_id`` field — the rollout's
    home node is stored separately on :class:`RolloutRecord`. Callers
    that consume the sealed trajectory (the smoke driver, future
    `Client.replay`, the admin viewer) want it surfaced under
    ``trajectory.metadata["node_id"]`` so they can show "this rollout
    ran on AWS / GCP" without joining against the state store.
    """
    out = dict(base or {})
    if node_id is not None:
        out["node_id"] = node_id
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────────────────


class StateStore(Protocol):
    """Persistence surface consumed by the coordinator + scheduler.

    Implementations: :class:`InMemoryStateStore` (default for in-process tests
    and Slice 1) and :class:`SqliteStateStore` (Slice 2 production default for
    a single-process control plane). Phase 1 adds a redis-backed impl.

    All methods are sync; sqlite stays sync-first because phase-0 query
    volumes are tiny (~10 µs/query) and the simpler API is a better trade-off
    than threading a worker pool through every call site.
    """

    # Rollouts
    def insert_rollout(self, record: RolloutRecord) -> None: ...
    def get_rollout(self, rollout_id: str) -> RolloutRecord: ...
    def update_rollout(self, rollout_id: str, **fields: Any) -> RolloutRecord: ...
    def list_rollouts(self) -> list[RolloutRecord]: ...
    def append_step(self, rollout_id: str, step: Step) -> None: ...

    # Sandboxes
    def insert_sandbox(self, record: SandboxRecord) -> None: ...
    def get_sandbox(self, sandbox_id: str) -> SandboxRecord: ...
    def update_sandbox(self, sandbox_id: str, **fields: Any) -> SandboxRecord: ...
    def remove_sandbox(self, sandbox_id: str) -> None: ...
    def list_sandboxes(self) -> list[SandboxRecord]: ...

    # Idempotency
    def lookup_idempotent(self, request_id: str) -> str | None: ...
    def record_idempotent(self, request_id: str, rollout_id: str) -> None: ...

    # Events
    def append_event(
        self,
        kind: str,
        *,
        rollout_id: str | None = None,
        sandbox_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord: ...
    def events_since(self, seq: int) -> Iterable[EventRecord]: ...

    # Audit (spec 19) — separate retention schedule from events.
    def append_audit(
        self,
        kind: str,
        *,
        role: str | None = None,
        digest_hint: str | None = None,
        method: str | None = None,
        source: str | None = None,
        result: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> AuditRecord: ...
    def audit_since(self, seq: int) -> Iterable[AuditRecord]: ...

    # Live node registry mirror (spec 03 §"NodeRegistry persistence")
    def record_node_connected(
        self,
        node_id: str,
        *,
        backends: list[str] | None = None,
        stream_epoch: str | None = None,
        instance_id: str | None = None,
        isolation_capable: bool = False,
    ) -> NodeRecord: ...
    def record_node_disconnected(self, node_id: str) -> None: ...
    def update_node_seen(self, node_id: str, ts: float) -> None: ...
    def update_node_pinned_cpus(
        self, node_id: str, *, free: int, total: int,
    ) -> None: ...
    def update_node_health(self, node_id: str, health_json: str) -> None: ...
    def list_node_health(self) -> dict[str, str]: ...
    def update_node_aimd_limit(self, node_id: str, limit: int) -> None: ...
    def list_node_aimd_limits(self) -> dict[str, int]: ...
    def list_nodes(self, *, status: str | None = None) -> list[NodeRecord]: ...
    def prune_lost_nodes(self, *, keep: set[str]) -> list[str]: ...
    def prune_expired(
        self,
        *,
        now: float,
        audit_retention_days: int | None = 30,
        events_retention_days: int | None = 14,
        raw_rollout_retention_days: int | None = 14,
        batch_size: int = 10_000,
    ) -> dict[str, int]: ...

    # Admission queue
    def enqueue_pending(self, record: PendingRolloutRecord) -> None: ...
    def list_pending(self, *, partition: str | None = None) -> list[PendingRolloutRecord]: ...
    def remove_pending(self, pending_id: str) -> None: ...

    # Sealed trajectory (read-back)
    def seal_trajectory(self, rollout_id: str) -> Trajectory: ...

    # P1.6.b — control-plane-driven image builds
    def record_build_plan(
        self, *, plan_id: str, applied_by: str, plan_json: str,
        name: str | None = None,
    ) -> BuildPlanRecord: ...
    def update_build_plan_status(
        self, plan_id: str, status: BuildPlanStatus,
    ) -> None: ...
    def try_update_build_plan_status(
        self, plan_id: str, *,
        expected_current: BuildPlanStatus,
        new_status: BuildPlanStatus,
    ) -> bool:
        """Atomic CAS-style status flip.

        Updates ``plan_id``'s status to ``new_status`` only when the
        current persisted status equals ``expected_current``. Returns
        True on a successful flip, False when the plan's current
        status didn't match (or the plan doesn't exist).

        Use this in any code path that wants to *progress* a plan
        from one status to another without overwriting a concurrent
        flip set by a different code path. Canonical use: the build
        coordinator's apply() finalizer sets ``in_flight →
        completed/partial_failure`` via this CAS so an operator
        cancel that's already moved the plan to ``cancelled``
        doesn't get overwritten.
        """
        ...
    def get_build_plan(self, plan_id: str) -> BuildPlanRecord | None: ...
    def list_build_plans(
        self, *, status: BuildPlanStatus | None = None,
    ) -> list[BuildPlanRecord]: ...
    def record_assignment(self, record: BuildAssignmentRecord) -> None: ...
    def update_assignment_status(
        self,
        *,
        plan_id: str,
        node_id: str,
        image_ref: str,
        status: BuildAssignmentStatus,
        error: str | None = None,
    ) -> None: ...
    def list_assignments(
        self, plan_id: str,
    ) -> list[BuildAssignmentRecord]: ...
    def delete_assignments(self, plan_id: str) -> None:
        """Remove every assignment row for ``plan_id``. Called by the
        per-image-ref dispatch path on force re-apply so stale
        ``(node_id, image_ref)`` rows from a prior placement don't
        accumulate when the FFD bin-packer picks different nodes
        across runs (free disk drifts → different placement → same
        plan_id ends up with N + M rows instead of N)."""
        ...
    def find_registered_preferred_home(
        self, image_ref: str,
    ) -> str | None: ...
    """Audit P1.6.g-H2 (2026-05-05): return the most-recently-applied
    plan's ``preferred_home`` node id for a row that's still in
    ``status="registered"`` for ``image_ref``, or ``None`` if no
    deferred row exists. Used by the scheduler's image-affinity
    bonus to land first-rollout-of-a-deferred-ref on the planner's
    chosen node when feasible. Soft preference, not a hard
    constraint — falls back to other scoring on a tie."""

    # P1.7.B.3 — case-2/3 raw-rollout tracking
    def record_raw_rollout(self, record: RawRolloutRecord) -> None: ...
    def update_raw_rollout(
        self, rollout_id: str, **fields: Any,
    ) -> RawRolloutRecord: ...
    def get_raw_rollout(self, rollout_id: str) -> RawRolloutRecord | None: ...
    def list_raw_rollouts(
        self,
        *,
        status: RawRolloutStatus | None = None,
        since_after: float | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[RawRolloutRecord]: ...
    def count_raw_rollouts(
        self,
        *,
        status: RawRolloutStatus | None = None,
        since_after: float | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
    ) -> int: ...
    def aggregate_raw_rollouts_by_owner_status(
        self,
    ) -> dict[str, dict[str, int]]: ...
    def count_raw_rollouts_by_node(self) -> dict[str | None, int]: ...
    def raw_rollouts_created_span(
        self,
    ) -> tuple[float | None, float | None]: ...
    def owner_rollout_lifetime(self) -> dict[str, dict[str, int]]: ...
    def aggregate_raw_rollouts_all_time_by_owner_status(
        self,
    ) -> dict[str, dict[str, int]]: ...
    def lifetime_inception_ts(self) -> float | None: ...

    # Fleet reservation (phase 1) — tiny live rows, deleted on last-member
    # destroy; sole purpose is CP-restart footprint recovery.
    def record_fleet_reservation(
        self, record: FleetReservationRecord,
    ) -> None: ...
    def touch_fleet_reservation(
        self, fleet_id: str, *, last_acquire_ts: float,
    ) -> None: ...
    def delete_fleet_reservation(self, fleet_id: str) -> None: ...
    def list_fleet_reservations(self) -> list[FleetReservationRecord]: ...

    # Compose projects (P1.7.C.2) — tiny live rows, deleted on confirmed
    # teardown; sole purpose is CP-restart footprint + subnet-claim recovery.
    def record_compose_project(
        self, record: ComposeProjectStateRecord,
    ) -> None: ...
    def delete_compose_project(self, rollout_id: str) -> None: ...
    def list_compose_projects(self) -> list[ComposeProjectStateRecord]: ...


# ──────────────────────────────────────────────────────────────────────────────
# In-memory implementation (kept for tests + Slice 1 parity)
# ──────────────────────────────────────────────────────────────────────────────


class InMemoryStateStore:
    """Thread-safe in-memory store; mirrors the SqliteStateStore surface."""

    def __init__(self) -> None:
        self._rollouts: dict[str, RolloutRecord] = {}
        self._sandboxes: dict[str, SandboxRecord] = {}
        self._pending: dict[str, PendingRolloutRecord] = {}
        self._events: list[EventRecord] = []
        self._audit: list[AuditRecord] = []
        self._nodes: dict[str, NodeRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._build_plans: dict[str, BuildPlanRecord] = {}
        # Keyed by (plan_id, node_id, image_ref) for direct upsert.
        self._build_assignments: dict[
            tuple[str, str, str], BuildAssignmentRecord,
        ] = {}
        # P1.7.B.3 — raw rollouts (case-2/3 evaluation tracking).
        self._raw_rollouts: dict[str, RawRolloutRecord] = {}
        # Durable per-owner/status tallies of raw rollouts the retention GC has
        # removed — the all-time half of the /users scoreboard. {owner: {status: n}}
        self._owner_lifetime: dict[str, dict[str, int]] = {}
        # When the lifetime tally started accruing (store creation, for the
        # in-memory store). Mirrors SqliteStore's persisted schema_meta value.
        self._lifetime_inception: float = time.time()
        # Fleet reservation (phase 1) — fleet_id -> live reservation row.
        # Deleted on last-member destroy; bounded by open-fleet concurrency.
        self._fleet_reservations: dict[str, FleetReservationRecord] = {}
        self._compose_projects: dict[str, ComposeProjectStateRecord] = {}
        # Stage-1 admission/capacity — node_id -> latest health JSON.
        self._node_health: dict[str, str] = {}
        # Stage-3 — node_id -> latest AIMD admission limit.
        self._node_aimd_limits: dict[str, int] = {}
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._event_seq = 0
        self._audit_seq = 0

    # ── Rollouts ─────────────────────────────────────────────────────────────

    def insert_rollout(self, record: RolloutRecord) -> None:
        with self._lock:
            if record.rollout_id in self._rollouts:
                raise KeyError(f"rollout {record.rollout_id} already exists")
            self._rollouts[record.rollout_id] = record

    def get_rollout(self, rollout_id: str) -> RolloutRecord:
        with self._lock:
            record = self._rollouts.get(rollout_id)
            if record is None:
                raise KeyError(f"unknown rollout_id {rollout_id}")
            return record

    def update_rollout(self, rollout_id: str, **fields: Any) -> RolloutRecord:
        with self._lock:
            record = self._rollouts.get(rollout_id)
            if record is None:
                raise KeyError(f"unknown rollout_id {rollout_id}")
            for k, v in fields.items():
                setattr(record, k, v)
            record.last_touched_at = time.time()
            return record

    def list_rollouts(self) -> list[RolloutRecord]:
        with self._lock:
            return list(self._rollouts.values())

    def append_step(self, rollout_id: str, step: Step) -> None:
        with self._lock:
            record = self._rollouts.get(rollout_id)
            if record is None:
                raise KeyError(f"unknown rollout_id {rollout_id}")
            record.steps.append(step)
            record.final_reward += step.reward
            record.last_touched_at = time.time()

    # ── Sandboxes ────────────────────────────────────────────────────────────

    def insert_sandbox(self, record: SandboxRecord) -> None:
        with self._lock:
            if record.sandbox_id in self._sandboxes:
                raise KeyError(f"sandbox {record.sandbox_id} already exists")
            self._sandboxes[record.sandbox_id] = record

    def get_sandbox(self, sandbox_id: str) -> SandboxRecord:
        with self._lock:
            record = self._sandboxes.get(sandbox_id)
            if record is None:
                raise KeyError(f"unknown sandbox_id {sandbox_id}")
            return record

    def update_sandbox(self, sandbox_id: str, **fields: Any) -> SandboxRecord:
        with self._lock:
            record = self._sandboxes.get(sandbox_id)
            if record is None:
                raise KeyError(f"unknown sandbox_id {sandbox_id}")
            for k, v in fields.items():
                setattr(record, k, v)
            return record

    def remove_sandbox(self, sandbox_id: str) -> None:
        with self._lock:
            self._sandboxes.pop(sandbox_id, None)

    def list_sandboxes(self) -> list[SandboxRecord]:
        with self._lock:
            return list(self._sandboxes.values())

    # ── Idempotency ──────────────────────────────────────────────────────────

    def lookup_idempotent(self, request_id: str) -> str | None:
        with self._lock:
            return self._idempotency.get(request_id)

    def record_idempotent(self, request_id: str, rollout_id: str) -> None:
        with self._lock:
            self._idempotency[request_id] = rollout_id

    # ── Events ───────────────────────────────────────────────────────────────

    def append_event(
        self,
        kind: str,
        *,
        rollout_id: str | None = None,
        sandbox_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        with self._lock:
            self._event_seq += 1
            evt = EventRecord(
                seq=self._event_seq,
                ts=time.time(),
                rollout_id=rollout_id,
                sandbox_id=sandbox_id,
                kind=kind,
                payload=payload or {},
            )
            self._events.append(evt)
            return evt

    def events_since(self, seq: int) -> Iterable[EventRecord]:
        with self._lock:
            return [e for e in self._events if e.seq > seq]

    # ── Audit (spec 19) ──────────────────────────────────────────────────────

    def append_audit(
        self,
        kind: str,
        *,
        role: str | None = None,
        digest_hint: str | None = None,
        method: str | None = None,
        source: str | None = None,
        result: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> AuditRecord:
        with self._lock:
            self._audit_seq += 1
            row = AuditRecord(
                seq=self._audit_seq,
                ts=time.time(),
                kind=kind,
                role=role,
                digest_hint=digest_hint,
                method=method,
                source=source,
                result=result,
                payload=payload or {},
            )
            self._audit.append(row)
            return row

    def audit_since(self, seq: int) -> Iterable[AuditRecord]:
        with self._lock:
            return [r for r in self._audit if r.seq > seq]

    # ── Live node registry mirror ────────────────────────────────────────────

    def record_node_connected(
        self,
        node_id: str,
        *,
        backends: list[str] | None = None,
        stream_epoch: str | None = None,
        instance_id: str | None = None,
        isolation_capable: bool = False,
    ) -> NodeRecord:
        with self._lock:
            ts = time.time()
            existing = self._nodes.get(node_id)
            connected_at = existing.connected_at if (
                existing is not None and existing.status == "connected"
            ) else ts
            row = NodeRecord(
                node_id=node_id,
                status="connected",
                connected_at=connected_at,
                last_seen_at=ts,
                stream_epoch=stream_epoch,
                instance_id=instance_id,
                backends=backends or [],
                isolation_capable=isolation_capable,
                # Preserve last-known pinned counts across a re-register (they're
                # refreshed by the next heartbeat mirror, not carried on hello).
                pinned_cpus_free=existing.pinned_cpus_free if existing else 0,
                pinned_cpus_total=existing.pinned_cpus_total if existing else 0,
            )
            self._nodes[node_id] = row
            return row

    def record_node_disconnected(self, node_id: str) -> None:
        with self._lock:
            existing = self._nodes.get(node_id)
            if existing is None:
                return
            self._nodes[node_id] = existing.model_copy(
                update={"status": "lost", "last_seen_at": time.time()},
            )

    def update_node_seen(self, node_id: str, ts: float) -> None:
        with self._lock:
            existing = self._nodes.get(node_id)
            if existing is None:
                return
            self._nodes[node_id] = existing.model_copy(update={"last_seen_at": ts})

    def update_node_pinned_cpus(
        self, node_id: str, *, free: int, total: int,
    ) -> None:
        """P6 step-2c — mirror the node's last-known pinnable-CPU counts (from
        the heartbeat) onto its row for operator observability. No-op for an
        unknown node."""
        with self._lock:
            existing = self._nodes.get(node_id)
            if existing is None:
                return
            self._nodes[node_id] = existing.model_copy(update={
                "pinned_cpus_free": int(free),
                "pinned_cpus_total": int(total),
            })

    def update_node_health(self, node_id: str, health_json: str) -> None:
        with self._lock:
            self._node_health[node_id] = health_json

    def list_node_health(self) -> dict[str, str]:
        with self._lock:
            return dict(self._node_health)

    def update_node_aimd_limit(self, node_id: str, limit: int) -> None:
        with self._lock:
            self._node_aimd_limits[node_id] = limit

    def list_node_aimd_limits(self) -> dict[str, int]:
        with self._lock:
            return dict(self._node_aimd_limits)

    def list_nodes(self, *, status: str | None = None) -> list[NodeRecord]:
        with self._lock:
            rows = sorted(self._nodes.values(), key=lambda r: r.node_id)
            if status is None:
                return rows
            return [r for r in rows if r.status == status]

    def prune_lost_nodes(self, *, keep: set[str]) -> list[str]:
        with self._lock:
            pruned = [
                nid
                for nid, r in self._nodes.items()
                if r.status == "lost" and nid not in keep
            ]
            for nid in pruned:
                self._nodes.pop(nid, None)
                self._node_health.pop(nid, None)
                self._node_aimd_limits.pop(nid, None)
            return pruned

    def prune_expired(
        self,
        *,
        now: float,
        audit_retention_days: int | None = 30,
        events_retention_days: int | None = 14,
        raw_rollout_retention_days: int | None = 14,
        batch_size: int = 10_000,
    ) -> dict[str, int]:
        day = 86400.0
        terminal = {"released", "cancelled", "failed", "reaped", "capacity_rejected"}
        out = {"audit": 0, "events": 0, "raw_rollouts": 0}
        with self._lock:
            if audit_retention_days is not None:
                cutoff = now - audit_retention_days * day
                kept = [r for r in self._audit if r.ts >= cutoff]
                out["audit"] = len(self._audit) - len(kept)
                self._audit = kept
            if events_retention_days is not None:
                cutoff = now - events_retention_days * day
                kept_e = [e for e in self._events if e.ts >= cutoff]
                out["events"] = len(self._events) - len(kept_e)
                self._events = kept_e
            if raw_rollout_retention_days is not None:
                cutoff = now - raw_rollout_retention_days * day
                drop = [
                    rid
                    for rid, r in self._raw_rollouts.items()
                    if r.status in terminal
                    and (r.finished_at if r.finished_at is not None else r.created_at)
                    < cutoff
                ]
                for rid in drop:
                    # Preserve the pruned row's owner/status tally before delete
                    # so the /users lifetime scoreboard survives GC (mirror of the
                    # Sqlite path's same-transaction accumulate-then-delete).
                    r = self._raw_rollouts[rid]
                    per = self._owner_lifetime.setdefault(r.owner_id, {})
                    per[r.status] = per.get(r.status, 0) + 1
                    del self._raw_rollouts[rid]
                out["raw_rollouts"] = len(drop)
        return out

    # ── Admission queue ──────────────────────────────────────────────────────

    def enqueue_pending(self, record: PendingRolloutRecord) -> None:
        with self._lock:
            if record.pending_id in self._pending:
                raise KeyError(f"pending rollout {record.pending_id} already queued")
            self._pending[record.pending_id] = record

    def list_pending(self, *, partition: str | None = None) -> list[PendingRolloutRecord]:
        with self._lock:
            rows = sorted(self._pending.values(), key=lambda r: r.submitted_at)
            if partition is None:
                return rows
            return [r for r in rows if r.queue_partition == partition]

    def remove_pending(self, pending_id: str) -> None:
        with self._lock:
            self._pending.pop(pending_id, None)

    # ── Trajectory sealing ───────────────────────────────────────────────────

    def seal_trajectory(self, rollout_id: str) -> Trajectory:
        with self._lock:
            record = self._rollouts.get(rollout_id)
            if record is None:
                raise KeyError(f"unknown rollout_id {rollout_id}")
            return Trajectory(
                rollout_id=record.rollout_id,
                template=record.template,
                steps=list(record.steps),
                status=record.status,
                reason=record.reason,
                final_reward=record.final_reward,
                metadata=_metadata_with_node_id(record.metadata, record.node_id),
            )

    # ── P1.6.b — control-plane-driven image builds ──────────────────────

    def record_build_plan(
        self, *, plan_id: str, applied_by: str, plan_json: str,
        name: str | None = None,
    ) -> BuildPlanRecord:
        with self._lock:
            # Upsert: on re-apply (force=True or partial_failure
            # residual-only retry), advance applied_at + applied_by
            # so the admin /builds view reflects the latest operator
            # intent. Caller short-circuits no-op re-applies before
            # this call, so any record_build_plan call corresponds
            # to an apply that's about to do real work.
            now = time.time()
            existing = self._build_plans.get(plan_id)
            if existing is not None:
                record = existing.model_copy(update={
                    "applied_at": now,
                    "applied_by": applied_by,
                    "plan_json": plan_json,
                    "name": name,
                })
                self._build_plans[plan_id] = record
                return record
            record = BuildPlanRecord(
                plan_id=plan_id, applied_at=now,
                applied_by=applied_by, plan_json=plan_json,
                status="in_flight", name=name,
            )
            self._build_plans[plan_id] = record
            return record

    def update_build_plan_status(
        self, plan_id: str, status: BuildPlanStatus,
    ) -> None:
        with self._lock:
            record = self._build_plans.get(plan_id)
            if record is None:
                raise KeyError(f"unknown plan_id {plan_id}")
            self._build_plans[plan_id] = record.model_copy(update={"status": status})

    def try_update_build_plan_status(
        self, plan_id: str, *,
        expected_current: BuildPlanStatus,
        new_status: BuildPlanStatus,
    ) -> bool:
        with self._lock:
            record = self._build_plans.get(plan_id)
            if record is None or record.status != expected_current:
                return False
            self._build_plans[plan_id] = record.model_copy(
                update={"status": new_status},
            )
            return True

    def get_build_plan(self, plan_id: str) -> BuildPlanRecord | None:
        with self._lock:
            return self._build_plans.get(plan_id)

    def list_build_plans(
        self, *, status: BuildPlanStatus | None = None,
    ) -> list[BuildPlanRecord]:
        with self._lock:
            rows = list(self._build_plans.values())
        if status is None:
            return rows
        return [r for r in rows if r.status == status]

    def record_assignment(self, record: BuildAssignmentRecord) -> None:
        with self._lock:
            key = (record.plan_id, record.node_id, record.image_ref)
            self._build_assignments[key] = record

    def update_assignment_status(
        self,
        *,
        plan_id: str,
        node_id: str,
        image_ref: str,
        status: BuildAssignmentStatus,
        error: str | None = None,
    ) -> None:
        with self._lock:
            key = (plan_id, node_id, image_ref)
            record = self._build_assignments.get(key)
            if record is None:
                raise KeyError(f"unknown assignment {key}")
            now = time.time()
            update: dict[str, Any] = {"status": status, "error": error}
            if status == "building" and record.started_at is None:
                update["started_at"] = now
            if status in ("done", "failed", "cancelled"):
                update["completed_at"] = now
            self._build_assignments[key] = record.model_copy(update=update)

    def list_assignments(
        self, plan_id: str,
    ) -> list[BuildAssignmentRecord]:
        with self._lock:
            return [
                r for r in self._build_assignments.values()
                if r.plan_id == plan_id
            ]

    def delete_assignments(self, plan_id: str) -> None:
        with self._lock:
            stale = [
                key for key, r in self._build_assignments.items()
                if r.plan_id == plan_id
            ]
            for key in stale:
                del self._build_assignments[key]

    def find_registered_preferred_home(
        self, image_ref: str,
    ) -> str | None:
        with self._lock:
            # Walk plans newest-first; return the first registered
            # row's node_id (= preferred_home — the coordinator
            # persists the deferred row's preferred-home as the
            # row's ``node_id`` field).
            ordered = sorted(
                self._build_plans.values(),
                key=lambda p: p.applied_at, reverse=True,
            )
            for plan in ordered:
                for r in self._build_assignments.values():
                    if (
                        r.plan_id == plan.plan_id
                        and r.image_ref == image_ref
                        and r.status == "registered"
                    ):
                        return r.node_id
            return None

    # ── Raw rollouts (P1.7.B.3) ──────────────────────────────────────────────

    def record_raw_rollout(self, record: RawRolloutRecord) -> None:
        with self._lock:
            if record.rollout_id in self._raw_rollouts:
                raise KeyError(
                    f"raw rollout {record.rollout_id} already exists",
                )
            self._raw_rollouts[record.rollout_id] = record

    def update_raw_rollout(
        self, rollout_id: str, **fields: Any,
    ) -> RawRolloutRecord:
        with self._lock:
            record = self._raw_rollouts.get(rollout_id)
            if record is None:
                raise KeyError(f"unknown raw rollout {rollout_id}")
            updated = record.model_copy(update=fields)
            self._raw_rollouts[rollout_id] = updated
            return updated

    def get_raw_rollout(self, rollout_id: str) -> RawRolloutRecord | None:
        with self._lock:
            return self._raw_rollouts.get(rollout_id)

    def list_raw_rollouts(
        self,
        *,
        status: RawRolloutStatus | None = None,
        since_after: float | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[RawRolloutRecord]:
        with self._lock:
            rows = list(self._raw_rollouts.values())
        if status is not None:
            rows = [r for r in rows if r.status == status]
        if since_after is not None:
            rows = [r for r in rows if r.created_at >= since_after]
        if task_key is not None:
            rows = [r for r in rows if r.task_key == task_key]
        if group_id is not None:
            rows = [r for r in rows if r.group_id == group_id]
        if owner_id is not None:
            rows = [r for r in rows if r.owner_id == owner_id]
        # Newest-first; admin pages typically render in this order.
        rows.sort(key=lambda r: r.created_at, reverse=True)
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def count_raw_rollouts(
        self,
        *,
        status: RawRolloutStatus | None = None,
        since_after: float | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
    ) -> int:
        with self._lock:
            rows = list(self._raw_rollouts.values())
        if status is not None:
            rows = [r for r in rows if r.status == status]
        if since_after is not None:
            rows = [r for r in rows if r.created_at >= since_after]
        if task_key is not None:
            rows = [r for r in rows if r.task_key == task_key]
        if group_id is not None:
            rows = [r for r in rows if r.group_id == group_id]
        if owner_id is not None:
            rows = [r for r in rows if r.owner_id == owner_id]
        return len(rows)

    def aggregate_raw_rollouts_by_owner_status(self) -> dict[str, dict[str, int]]:
        with self._lock:
            rows = list(self._raw_rollouts.values())
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            per_status = out.setdefault(r.owner_id, {})
            per_status[r.status] = per_status.get(r.status, 0) + 1
        return out

    def count_raw_rollouts_by_node(self) -> dict[str | None, int]:
        with self._lock:
            rows = list(self._raw_rollouts.values())
        out: dict[str | None, int] = {}
        for r in rows:
            out[r.node_id] = out.get(r.node_id, 0) + 1
        return out

    def raw_rollouts_created_span(
        self,
    ) -> tuple[float | None, float | None]:
        with self._lock:
            times = [
                r.created_at
                for r in self._raw_rollouts.values()
                if getattr(r, "created_at", None) is not None
            ]
        return (min(times), max(times)) if times else (None, None)

    def owner_rollout_lifetime(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {o: dict(s) for o, s in self._owner_lifetime.items()}

    def aggregate_raw_rollouts_all_time_by_owner_status(
        self,
    ) -> dict[str, dict[str, int]]:
        # Live + lifetime under ONE lock → an atomic snapshot, so the retention
        # janitor can't move a row between the two reads and cause a double- or
        # under-count (mirror of the Sqlite single-query path).
        with self._lock:
            out: dict[str, dict[str, int]] = {}
            for r in self._raw_rollouts.values():
                per = out.setdefault(r.owner_id, {})
                per[r.status] = per.get(r.status, 0) + 1
            for owner, by_status in self._owner_lifetime.items():
                per = out.setdefault(owner, {})
                for status, n in by_status.items():
                    per[status] = per.get(status, 0) + n
            return out

    def lifetime_inception_ts(self) -> float | None:
        return self._lifetime_inception

    # ── Fleet reservation (phase 1) ─────────────────────────────────────────

    def record_fleet_reservation(
        self, record: FleetReservationRecord,
    ) -> None:
        with self._lock:
            # INSERT-OR-REPLACE: an opener creates one row; a re-record after a
            # restart rebuild simply refreshes it. Idempotent by design.
            self._fleet_reservations[record.fleet_id] = record

    def touch_fleet_reservation(
        self, fleet_id: str, *, last_acquire_ts: float,
    ) -> None:
        with self._lock:
            row = self._fleet_reservations.get(fleet_id)
            if row is not None:
                self._fleet_reservations[fleet_id] = row.model_copy(
                    update={"last_acquire_ts": last_acquire_ts},
                )

    def delete_fleet_reservation(self, fleet_id: str) -> None:
        with self._lock:
            self._fleet_reservations.pop(fleet_id, None)

    def list_fleet_reservations(self) -> list[FleetReservationRecord]:
        with self._lock:
            return list(self._fleet_reservations.values())

    def record_compose_project(
        self, record: ComposeProjectStateRecord,
    ) -> None:
        with self._lock:
            self._compose_projects[record.rollout_id] = record

    def delete_compose_project(self, rollout_id: str) -> None:
        with self._lock:
            self._compose_projects.pop(rollout_id, None)

    def list_compose_projects(self) -> list[ComposeProjectStateRecord]:
        with self._lock:
            return list(self._compose_projects.values())


# ──────────────────────────────────────────────────────────────────────────────
# Sqlite implementation (Slice 2 production default)
# ──────────────────────────────────────────────────────────────────────────────


_SCHEMA = """
-- journal_mode is set programmatically in __init__ BEFORE this schema runs
-- (see _apply_journal_mode) so a rollback-journal deployment never transiently
-- opens WAL / creates the mmap'd -shm. Do NOT set journal_mode here.
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id          TEXT PRIMARY KEY,
    template            TEXT NOT NULL,
    status              TEXT NOT NULL,
    reason              TEXT,
    request_id          TEXT,
    task_key            TEXT,
    group_id            TEXT,
    owner_id            TEXT NOT NULL DEFAULT 'default',
    project_id          TEXT NOT NULL DEFAULT 'default',
    run_id              TEXT NOT NULL DEFAULT 'default',
    node_id             TEXT,
    sandbox_id          TEXT,
    init_json           TEXT NOT NULL DEFAULT '{}',
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    final_reward        REAL NOT NULL DEFAULT 0.0,
    trajectory_sink     TEXT,
    trajectory_node_id  TEXT,
    trajectory_uri      TEXT,
    trajectory_size_bytes INTEGER,
    created_at          REAL NOT NULL,
    last_touched_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS rollouts_status_idx ON rollouts(status);
CREATE INDEX IF NOT EXISTS rollouts_created_at_idx
    ON rollouts(created_at DESC, rollout_id DESC);
CREATE INDEX IF NOT EXISTS rollouts_status_created_at_idx
    ON rollouts(status, created_at DESC, rollout_id DESC);
CREATE INDEX IF NOT EXISTS rollouts_template_created_at_idx
    ON rollouts(template, created_at DESC, rollout_id DESC);
CREATE INDEX IF NOT EXISTS rollouts_status_template_created_at_idx
    ON rollouts(status, template, created_at DESC, rollout_id DESC);
CREATE INDEX IF NOT EXISTS rollouts_request_id_idx ON rollouts(request_id);
CREATE INDEX IF NOT EXISTS rollouts_task_key_idx ON rollouts(task_key);
CREATE INDEX IF NOT EXISTS rollouts_group_id_idx ON rollouts(group_id);

CREATE TABLE IF NOT EXISTS rollout_steps (
    rollout_id   TEXT NOT NULL,
    step_index   INTEGER NOT NULL,
    action_json  TEXT NOT NULL,
    obs_json     TEXT NOT NULL,
    reward       REAL NOT NULL,
    done         INTEGER NOT NULL,
    truncated    INTEGER NOT NULL,
    info_json    TEXT NOT NULL DEFAULT '{}',
    ts           REAL NOT NULL,
    PRIMARY KEY (rollout_id, step_index),
    FOREIGN KEY (rollout_id) REFERENCES rollouts(rollout_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sandboxes (
    sandbox_id              TEXT PRIMARY KEY,
    backend                 TEXT NOT NULL,
    backend_ref             TEXT NOT NULL,
    stub_endpoint           TEXT NOT NULL,
    template                TEXT NOT NULL,
    image                   TEXT,
    node_id                 TEXT NOT NULL,
    rollout_id              TEXT,
    status                  TEXT NOT NULL DEFAULT 'running',
    owner_count             INTEGER NOT NULL DEFAULT 1,
    created_at              REAL NOT NULL,
    effective_resources_json TEXT
);
CREATE INDEX IF NOT EXISTS sandboxes_node_idx ON sandboxes(node_id);
CREATE INDEX IF NOT EXISTS sandboxes_status_idx ON sandboxes(status);

CREATE TABLE IF NOT EXISTS pending_rollouts (
    pending_id      TEXT PRIMARY KEY,
    template        TEXT NOT NULL,
    init_json       TEXT NOT NULL DEFAULT '{}',
    request_id      TEXT,
    task_key        TEXT,
    group_id        TEXT,
    deadline_json   TEXT NOT NULL DEFAULT '{}',
    queue_partition TEXT NOT NULL DEFAULT 'default',
    submitted_at    REAL NOT NULL
);

-- Multi-user fair-share policy (live, operator-tunable). Single global row
-- (id pinned to 1) + a per-owner override table. capacity_basis NULL means
-- fairness is disabled (the default) — the admission queue applies no cap.
CREATE TABLE IF NOT EXISTS fairness_global (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    capacity_basis  INTEGER,
    floor           INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS fairness_owner (
    owner_id        TEXT PRIMARY KEY,
    weight          REAL NOT NULL DEFAULT 1.0,
    hard_cap        INTEGER,
    uncapped        INTEGER NOT NULL DEFAULT 0,
    blocked         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS pending_partition_idx ON pending_rollouts(queue_partition, submitted_at);

CREATE TABLE IF NOT EXISTS idempotency (
    request_id  TEXT PRIMARY KEY,
    rollout_id  TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,
    rollout_id  TEXT,
    sandbox_id  TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_kind_idx ON events(kind);
CREATE INDEX IF NOT EXISTS events_rollout_idx ON events(rollout_id);

-- Spec 19 §"Audit logging" + spec 20 retention matrix: audit events are
-- separate from the generic events table so they roll on a longer
-- schedule (default 90 days). Phase-2 ships an optional signed archive
-- to object storage; phase 0 just keeps the rows here.
CREATE TABLE IF NOT EXISTS audit (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL NOT NULL,
    kind           TEXT NOT NULL,
    role           TEXT,                       -- node / consumer / operator / NULL
    digest_hint    TEXT,                       -- first 6 chars of SHA-256(token); NEVER the bytes
    method         TEXT,                       -- gRPC method or admin route
    source         TEXT,                       -- e.g. ip:port or 'local'
    result         TEXT NOT NULL DEFAULT 'ok', -- ok / denied / warned
    payload_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS audit_kind_idx ON audit(kind);
CREATE INDEX IF NOT EXISTS audit_ts_idx ON audit(ts);

-- Spec 03 §"NodeRegistry persistence" (Slice 4 follow-up): the live
-- in-process NodeRegistry mirrors register/deregister/heartbeat updates
-- here so out-of-process callers (the operator CLI, future admin RPC)
-- can read who's currently attached without going through gRPC.
CREATE TABLE IF NOT EXISTS nodes (
    node_id        TEXT PRIMARY KEY,
    status         TEXT NOT NULL,                    -- 'connected' | 'lost'
    connected_at   REAL NOT NULL,
    last_seen_at   REAL NOT NULL,
    stream_epoch   TEXT,
    instance_id    TEXT,
    backends_json  TEXT NOT NULL DEFAULT '[]',
    -- P6 step-2c (observability): isolation capability (NodeHello) + last-known
    -- pinnable-CPU counts (heartbeat). Nothing schedules on these yet.
    isolation_capable INTEGER NOT NULL DEFAULT 0,
    pinned_cpus_free  INTEGER NOT NULL DEFAULT 0,
    pinned_cpus_total INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS nodes_status_idx ON nodes(status);

-- Stage-1 admission/capacity observability
-- (notes/admission-stage-1-observability.md). One row per node
-- carrying the latest heartbeat's NodeHealthStats as JSON; the admin
-- "Cluster health" page renders it. A separate table (not columns on
-- ``nodes``) means a fresh ``CREATE TABLE IF NOT EXISTS`` covers it
-- with no ALTER migration on the existing ``nodes`` table.
CREATE TABLE IF NOT EXISTS node_health (
    node_id     TEXT PRIMARY KEY,
    health_json TEXT NOT NULL,
    updated_at  REAL NOT NULL
);

-- Stage-3 admission/capacity — the health-derived adaptive
-- concurrent-acquire limit per node, mirrored by the AIMD control
-- loop so the admin "Cluster health" page can show it out-of-process.
-- Separate table = a fresh CREATE TABLE IF NOT EXISTS, no migration.
CREATE TABLE IF NOT EXISTS node_aimd_limit (
    node_id         TEXT PRIMARY KEY,
    admission_limit INTEGER NOT NULL,
    updated_at      REAL NOT NULL
);

-- P1.6.b control-plane-driven image builds. ``build_plans`` carries
-- the operator-applied snapshot (one row per ``xrlenv build apply``);
-- ``build_plan_assignments`` carries the per-(node, image) row the
-- bin-packer produced. The pair is the source of truth for
-- ``xrlenv build status`` drift detection: drift = "snapshot says
-- image X on node Y but ``report_images`` doesn't see it." See
-- notes/phase-1-to-do.md "Slice P1.6 — control-plane-driven image
-- builds" for the 5-layer idempotency model.
CREATE TABLE IF NOT EXISTS build_plans (
    plan_id        TEXT PRIMARY KEY,                 -- sha256 of the canonicalised plan
    applied_at     REAL NOT NULL,
    applied_by     TEXT NOT NULL,                    -- operator token role / 'local' for in-process
    plan_json      TEXT NOT NULL,                    -- full normalised plan
    status         TEXT NOT NULL,                    -- 'in_flight' | 'completed' | 'partial_failure' | 'cancelled' | 'superseded'
    name           TEXT                              -- operator-friendly label (BuildPlan.name)
);
CREATE INDEX IF NOT EXISTS build_plans_status_idx ON build_plans(status);

CREATE TABLE IF NOT EXISTS build_plan_assignments (
    plan_id        TEXT NOT NULL,
    node_id        TEXT NOT NULL,
    image_ref      TEXT NOT NULL,
    benchmark      TEXT NOT NULL,
    -- One of: 'pending' (eager: queued for synchronous build) /
    -- 'registered' (lazy: build deferred to ensure_present) /
    -- 'building' / 'done' / 'evicted' (was done, fell out of cache;
    -- lazy re-build on next ensure_present) / 'failed'.
    status         TEXT NOT NULL,
    started_at     REAL,
    completed_at   REAL,
    error          TEXT,
    PRIMARY KEY (plan_id, node_id, image_ref)
);
CREATE INDEX IF NOT EXISTS build_plan_assignments_plan_idx
    ON build_plan_assignments(plan_id);
CREATE INDEX IF NOT EXISTS build_plan_assignments_status_idx
    ON build_plan_assignments(plan_id, status);

-- P1.7.B.3 — case-2/3 raw-container rollout tracking. Separate from
-- the ``rollouts`` table because the two carry genuinely different
-- shapes (case-1 has trajectory steps + final_reward + verifier
-- output; case-2/3 raw containers have container_id + container_name
-- + harness-on-consumer-side metadata). Additive table; existing
-- case-1 schema unchanged.
CREATE TABLE IF NOT EXISTS raw_rollouts (
    rollout_id      TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    image           TEXT NOT NULL,
    node_id         TEXT,
    container_id    TEXT,
    container_name  TEXT,
    artifact_path   TEXT,
    displayed_name  TEXT,
    task_key        TEXT,
    group_id        TEXT,
    fleet_id        TEXT,
    owner_id        TEXT NOT NULL DEFAULT 'default',
    created_at      REAL NOT NULL,
    finished_at     REAL,
    error           TEXT,
    deadline_at     REAL,
    effective_resources_json TEXT,
    container_runtime TEXT
);
CREATE INDEX IF NOT EXISTS raw_rollouts_status_idx
    ON raw_rollouts(status);
CREATE INDEX IF NOT EXISTS raw_rollouts_created_at_idx
    ON raw_rollouts(created_at DESC);
-- Durable per-owner/status rollout tallies. The retention GC folds each
-- raw_rollouts row it prunes into this table IN THE SAME TRANSACTION as the
-- DELETE, so the /users scoreboard stays all-time even though raw_rollouts
-- itself is only a retention window. Never GC'd (tiny: #owners x #statuses).
CREATE TABLE IF NOT EXISTS owner_rollout_lifetime (
    owner_id  TEXT NOT NULL,
    status    TEXT NOT NULL,
    count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (owner_id, status)
);
-- Small key/value scratch. Holds ``lifetime_inception_ts`` — the epoch when the
-- owner_rollout_lifetime tally first started accruing on this DB (set once, on
-- first open of a store carrying the feature). The /users page renders it so an
-- operator can see the boundary below which pre-feature rollouts are NOT counted.
CREATE TABLE IF NOT EXISTS schema_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
-- The admin /rollouts/raw list + count query filters by status and pages by
-- created_at; this composite lets one index satisfy the WHERE + ORDER BY
-- without a table scan (mirrors rollouts_status_created_at_idx on ``rollouts``).
CREATE INDEX IF NOT EXISTS raw_rollouts_status_created_at_idx
    ON raw_rollouts(status, created_at DESC);
-- The admin overview/health 'finished/failed in the last N min' tiles count
-- rows by ``status IN (...) AND finished_at >= cutoff``. This composite lets
-- SQLite seek straight to the small recent tail within each status partition,
-- instead of walking every ``released`` row (the vast majority) via the plain
-- status index and testing finished_at on each.
CREATE INDEX IF NOT EXISTS raw_rollouts_status_finished_at_idx
    ON raw_rollouts(status, finished_at DESC);
-- NOTE: indexes on the migration-added columns (owner_id / task_key / group_id)
-- are created in ``_migrate()``, AFTER their ``ALTER TABLE ADD COLUMN`` — a
-- legacy DB reaches this schema block before those columns exist, so a
-- ``CREATE INDEX ... (owner_id)`` here would fail with "no such column".

-- Fleet reservation (phase 1, opt-in). One row per CURRENTLY OPEN fleet —
-- tiny live scheduler metadata, DELETED when the fleet's last member is
-- node-confirmed destroyed (never a growing history; invariant 6). Separate
-- table = a fresh CREATE TABLE IF NOT EXISTS, no ALTER migration. Members are
-- NOT stored here: on CP restart the live member set is rebuilt from the
-- containers' xrlenv.fleet_id labels (node = ground truth); this row exists
-- only to recover the CP-only FOOTPRINT the node never sees.
CREATE TABLE IF NOT EXISTS fleet_reservations (
    fleet_id          TEXT PRIMARY KEY,
    node_id           TEXT NOT NULL,
    footprint_json    TEXT NOT NULL,
    task_key          TEXT,
    owner             TEXT NOT NULL DEFAULT 'default',
    opened_ts         REAL NOT NULL,
    last_acquire_ts   REAL NOT NULL,
    container_runtime TEXT
);

CREATE TABLE IF NOT EXISTS compose_projects (
    rollout_id         TEXT PRIMARY KEY,
    project_name       TEXT NOT NULL,
    node_id            TEXT NOT NULL,
    footprint_json     TEXT NOT NULL,
    subnet_claims_json TEXT NOT NULL DEFAULT '[]',
    owner              TEXT NOT NULL DEFAULT 'default',
    created_ts         REAL NOT NULL
);
"""


class SqliteStateStore:
    """sqlite3-backed StateStore (spec 20).

    ``check_same_thread=False`` lets the coordinator and the scheduler share one
    connection across asyncio tasks; sqlite3's own threading lock serializes
    concurrent statements safely. Per-call query volumes are tiny (single-row
    UPSERTs, small SELECTs by primary key) so the lock contention is in the noise.

    Journal mode is WAL by default but selectable via
    ``XRLENV_SQLITE_JOURNAL_MODE`` (see :meth:`_apply_journal_mode`) — set
    ``TRUNCATE`` on a network filesystem (Lustre/FSx), where WAL's mmap'd ``-shm``
    faults with a fatal ``SIGBUS``. Under a rollback-journal mode there is no
    ``-wal``, so the WAL checkpointer is neither scheduled nor run.
    """

    def __init__(self, db_path: Path, *, read_only: bool = False) -> None:
        self._path = db_path
        self._read_only = read_only
        self._lock = threading.Lock()
        # Set by ``close()``. Lets shutdown-sensitive callers (e.g.
        # ``NodeRegistry.deregister``, driven from a gRPC stream's
        # ``finally`` that can run *after* the runtime closed the
        # store) skip a doomed write quietly instead of raising
        # ``sqlite3.ProgrammingError: Cannot operate on a closed
        # database``.
        self._closed = False
        if read_only:
            # Pure reader (CLI status queries, deploy liveness probe): open the
            # file READ-ONLY so we never run PRAGMA journal_mode / schema DDL /
            # the inception stamp — i.e. never flip an existing TRUNCATE DB back
            # to WAL (which recreates the -shm mmap SIGBUS exposure on Lustre) or
            # create -wal/-shm sidecars. Assumes the schema already exists (the
            # control plane created it); a missing/unreadable file raises here,
            # which is the correct fail-closed signal for a liveness probe.
            self._conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            # A mode=ro reader still needs a SHARED lock, which a rollback-journal
            # writer (TRUNCATE/DELETE on Lustre) blocks for the brief window it
            # holds EXCLUSIVE at commit. With the default busy_timeout of 0 the
            # reader fails *immediately* with "database is locked" — under write
            # load that surfaces as a constant stream of admin-panel 500s. Wait
            # out the (millisecond-scale) commit window instead of failing hard.
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._journal_mode = "RO"
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                # Select + apply journal_mode BEFORE any schema DDL. WAL always
                # mmaps its -shm wal-index, and that mmap faults with a FATAL
                # SIGBUS on a network filesystem (Lustre/FSx) — see the CP-SIGBUS
                # incident. If the WAL PRAGMA ran first (in _SCHEMA) we'd
                # create/mmap -shm during CREATE TABLE + migration on every open,
                # reintroducing the exposure the override exists to remove. So set
                # it up front and never enter WAL on a rollback-journal deployment.
                self._apply_journal_mode()
                self._conn.executescript(_SCHEMA)
                self._migrate()
                self._ensure_lifetime_inception()
                self._conn.commit()
        except BaseException:
            # A failed init (e.g. an invalid XRLENV_SQLITE_JOURNAL_MODE, or a
            # failed conversion) must not leak the open connection/descriptor to
            # GC — the constructor raises, so no close() will ever be called.
            self._conn.close()
            raise

    # Durability-preserving journal modes only. MEMORY/OFF drop the rollback
    # journal (a crash mid-write corrupts the DB) and are rejected for the
    # control-plane store. WAL is the default; TRUNCATE/DELETE/PERSIST are the
    # rollback-journal modes for network-filesystem (no-mmap) deployments.
    _ALLOWED_JOURNAL_MODES = frozenset({"WAL", "TRUNCATE", "DELETE", "PERSIST"})

    def _apply_journal_mode(self) -> None:
        """Set SQLite's journal_mode from ``XRLENV_SQLITE_JOURNAL_MODE``.

        Fails CLOSED: an unset var → WAL (prior default); any other value must
        be one of :data:`_ALLOWED_JOURNAL_MODES` or we raise, rather than
        silently falling back to WAL (which on Lustre is the SIGBUS trap). The
        mode SQLite actually adopts is verified — a failed conversion (e.g. a
        second connection holding the DB) raises instead of running WAL unaware.
        """
        raw = os.environ.get("XRLENV_SQLITE_JOURNAL_MODE", "").strip()
        mode = raw.upper() if raw else "WAL"
        if mode not in self._ALLOWED_JOURNAL_MODES:
            raise ValueError(
                f"XRLENV_SQLITE_JOURNAL_MODE={raw!r} invalid; expected one of "
                f"{sorted(self._ALLOWED_JOURNAL_MODES)}"
            )
        got = self._conn.execute(f"PRAGMA journal_mode={mode}").fetchone()[0]
        if str(got).upper() != mode:
            raise RuntimeError(
                f"could not set SQLite journal_mode={mode}: SQLite reports "
                f"{got!r} (is another connection holding {self._path}?)"
            )
        # Fixed for the process; used to no-op the WAL checkpointer under a
        # rollback-journal mode (there is no -wal to checkpoint).
        self._journal_mode = mode

    def _ensure_lifetime_inception(self) -> None:
        """Stamp ``schema_meta['lifetime_inception_ts']`` once — the first time a
        store carrying the owner_rollout_lifetime feature opens this DB. SELECT
        first so steady-state opens take no write lock; ``INSERT OR IGNORE``
        closes the two-opens-at-once race."""
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'lifetime_inception_ts'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) "
                "VALUES ('lifetime_inception_ts', ?)",
                (str(time.time()),),
            )

    def lifetime_inception_ts(self) -> float | None:
        """Epoch when the /users lifetime tally started accruing on this DB
        (``None`` if unset). Totals are complete only for rollouts from this
        point on; rollouts pruned before it were lost before the feature."""
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'lifetime_inception_ts'"
        ).fetchone()
        if row is None:
            return None
        try:
            return float(row["value"])
        except (TypeError, ValueError):
            return None

    @property
    def is_closed(self) -> bool:
        """``True`` once :meth:`close` has run. Shutdown-race-sensitive
        callers check this before a mirror write rather than catching
        the closed-database error after the fact."""
        return self._closed

    def checkpoint_wal(self, *, truncate: bool = True) -> tuple[int, int, int]:
        """Force a WAL checkpoint so the ``-wal`` file can't grow without
        bound between the connection-close checkpoints.

        SQLite auto-checkpoints only *passively* at commit time (once the
        WAL passes ``wal_autocheckpoint`` frames), and a passive checkpoint
        cannot reclaim frames still pinned by an open reader. So a steady
        reader (the admin panel polling DB-backed tabs) alongside the
        control plane's steady write stream (per-heartbeat node mirrors +
        rollout-lifecycle rows) lets the WAL grow unbounded until the next
        process restart — the only other time it's checkpointed. A runaway
        51 GiB WAL on the shared filesystem was observed to stall the
        control-plane event loop badly enough to mass-mark every node lost.

        The checkpoint runs on a DEDICATED, short-lived connection — NOT the
        shared writer connection under ``self._lock``. A ``TRUNCATE`` of a
        large WAL (e.g. one inherited after an unclean shutdown, before the
        periodic checkpointer has had a chance to keep it small) can take
        many seconds on a network filesystem; holding ``self._lock`` for
        that long would block every heartbeat/rollout write behind it — the
        exact event-loop stall this checkpointer exists to prevent (audit
        residual, 2026-07-15). A separate connection lets SQLite's WAL
        protocol coordinate instead: the checkpoint folds what frames it can
        and returns ``busy`` (no truncation this round) if the writer/readers
        hold the WAL, WITHOUT ever blocking them. ``busy_timeout`` stays at
        the default 0 so a contended checkpoint returns immediately rather
        than waiting; the next tick retries. TRUNCATE still folds frames into
        the main DB even when it can't reset the file, so the WAL stays
        bounded under sustained write load.

        Returns SQLite's ``(busy, log_frames, checkpointed_frames)``;
        ``(0, 0, 0)`` if the store is already closed.
        """
        if self._closed:
            return (0, 0, 0)
        if getattr(self, "_journal_mode", "WAL") != "WAL":
            # Rollback-journal store (e.g. TRUNCATE on a network filesystem):
            # there is no -wal file, so `PRAGMA wal_checkpoint` is a no-op. Skip
            # the per-tick connection open entirely. Journal mode is fixed for
            # the process (set once in _apply_journal_mode).
            return (0, 0, 0)
        mode = "TRUNCATE" if truncate else "PASSIVE"
        # A fresh connection to the same file; created and used entirely in
        # the calling thread (the checkpointer's worker thread), so it never
        # crosses threads. It attaches to the existing WAL/-shm — no schema
        # or migration runs. Closed immediately after the one PRAGMA.
        conn = sqlite3.connect(str(self._path))
        try:
            row = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        finally:
            conn.close()
        if row is None:
            return (0, 0, 0)
        return (int(row[0]), int(row[1]), int(row[2]))

    def _migrate(self) -> None:
        """Best-effort additive migrations. Each step checks if the
        column / index already exists and is a no-op when present, so
        the same code runs against fresh databases (covered by
        ``_SCHEMA``) and dev databases that predate the column.
        """
        cur = self._conn.cursor()
        # Slice 9b: per-sandbox effective resources snapshot for
        # Pattern-A heterogeneous placements. Existing dev databases
        # lack the column; ALTER TABLE adds it as nullable.
        cols = {row["name"] for row in cur.execute("PRAGMA table_info(sandboxes)")}
        if "effective_resources_json" not in cols:
            cur.execute("ALTER TABLE sandboxes ADD COLUMN effective_resources_json TEXT")
        if "image" not in cols:
            cur.execute("ALTER TABLE sandboxes ADD COLUMN image TEXT")

        # P6 step-2c (observability): node isolation capability + last-known
        # pinnable-CPU counts. Pre-existing databases get 0 defaults (unknown /
        # non-capable) — fully backwards-compatible; nothing schedules on them.
        node_cols = {row["name"] for row in cur.execute("PRAGMA table_info(nodes)")}
        if "isolation_capable" not in node_cols:
            cur.execute(
                "ALTER TABLE nodes ADD COLUMN isolation_capable "
                "INTEGER NOT NULL DEFAULT 0",
            )
        if "pinned_cpus_free" not in node_cols:
            cur.execute(
                "ALTER TABLE nodes ADD COLUMN pinned_cpus_free "
                "INTEGER NOT NULL DEFAULT 0",
            )
        if "pinned_cpus_total" not in node_cols:
            cur.execute(
                "ALTER TABLE nodes ADD COLUMN pinned_cpus_total "
                "INTEGER NOT NULL DEFAULT 0",
            )

        # Operator-friendly plan label (build_plans.name). Pre-existing
        # plans show as ``(unnamed)`` in the admin panel until next apply.
        bp_cols = {row["name"] for row in cur.execute("PRAGMA table_info(build_plans)")}
        if "name" not in bp_cols:
            cur.execute("ALTER TABLE build_plans ADD COLUMN name TEXT")

        # Xrlenv-5: task_key + group_id columns on raw_rollouts so the
        # scheduler-side anti-affinity key (task_key) and the operator-
        # supplied grouping key (xrlenv.group_id label) are queryable
        # by admin's ``/rollouts`` view. Pre-existing databases get
        # NULL for both — fully backwards-compatible.
        raw_cols = {row["name"] for row in cur.execute("PRAGMA table_info(raw_rollouts)")}
        if "task_key" not in raw_cols:
            cur.execute("ALTER TABLE raw_rollouts ADD COLUMN task_key TEXT")
        if "group_id" not in raw_cols:
            cur.execute("ALTER TABLE raw_rollouts ADD COLUMN group_id TEXT")
        # Fleet reservation (phase 1): the container's xrlenv.fleet_id label,
        # persisted so a CP restart can rebuild fleet membership from the
        # re-adopted live containers. Pre-existing rows get NULL (non-fleet).
        if "fleet_id" not in raw_cols:
            cur.execute("ALTER TABLE raw_rollouts ADD COLUMN fleet_id TEXT")
        # audit H11: the container runtime (e.g. sysbox-runc), persisted so a CP restart
        # restores it on re-adoption and the per-node runtime concurrency cap still counts the
        # surviving container. Pre-existing rows get NULL (default runtime).
        if "container_runtime" not in raw_cols:
            cur.execute("ALTER TABLE raw_rollouts ADD COLUMN container_runtime TEXT")

        # Multi-user: owner_id on raw_rollouts so the case-2/3 drop-in
        # path carries the tenant the acquiring consumer authenticated as,
        # and admin's /rollouts view can scope per-owner. Pre-existing rows
        # default to 'default' (single-tenant) — fully backwards-compatible.
        if "owner_id" not in raw_cols:
            cur.execute(
                "ALTER TABLE raw_rollouts ADD COLUMN "
                "owner_id TEXT NOT NULL DEFAULT 'default'"
            )

        # Indexes on the migration-added columns (owner_id / task_key /
        # group_id) — created HERE, after the ALTER TABLEs above, because a
        # legacy DB hits ``_SCHEMA`` before these columns exist (a
        # ``CREATE INDEX ... (owner_id)`` in the schema block would fail with
        # "no such column"). ``IF NOT EXISTS`` keeps it a no-op on fresh DBs
        # and re-runs. Consumer-scoped admin views + fair-share aggregates
        # filter/group by owner_id; task_key / group_id back the admin filter
        # params and the cancel-group path — without these each was a scan.
        #
        # owner_id is a COMPOSITE (owner_id, status): the admin /users page
        # runs ``GROUP BY owner_id, status`` (aggregate_raw_rollouts_by_owner_
        # status). A single-column owner_id index can't cover that — it scans
        # every row for ``status`` and builds a temp b-tree for the group
        # (~2.7s on 168k rows). The composite makes it a covering index scan in
        # group order (no table lookup, no temp b-tree). Its leading owner_id
        # column also serves every owner-scoped filter, so it fully supersedes
        # the plain owner_id index — drop that if an earlier migration made it.
        cur.execute("DROP INDEX IF EXISTS raw_rollouts_owner_id_idx")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS raw_rollouts_owner_id_status_idx "
            "ON raw_rollouts(owner_id, status)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS raw_rollouts_task_key_idx "
            "ON raw_rollouts(task_key)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS raw_rollouts_group_id_idx "
            "ON raw_rollouts(group_id)"
        )

        # Audit P2: persist the resolved reap deadline + effective
        # ResourceSpec so a control-plane restart re-adopts a raw
        # session with its ORIGINAL deadline / footprint instead of
        # resetting to defaults. Pre-existing rows get NULL for both →
        # re-adoption falls back to the default (today's behaviour),
        # fully backwards-compatible.
        if "deadline_at" not in raw_cols:
            cur.execute("ALTER TABLE raw_rollouts ADD COLUMN deadline_at REAL")
        if "effective_resources_json" not in raw_cols:
            cur.execute(
                "ALTER TABLE raw_rollouts ADD COLUMN "
                "effective_resources_json TEXT"
            )

        # §5.3 — fleet_reservations.container_runtime (sysbox pool). Older
        # DBs lack the column; ALTER adds it as nullable (default None).
        fleet_cols = {
            row["name"]
            for row in cur.execute("PRAGMA table_info(fleet_reservations)")
        }
        if "container_runtime" not in fleet_cols:
            cur.execute(
                "ALTER TABLE fleet_reservations ADD COLUMN container_runtime TEXT"
            )

        fairness_owner_cols = {
            row["name"] for row in cur.execute("PRAGMA table_info(fairness_owner)")
        }
        if "uncapped" not in fairness_owner_cols:
            cur.execute(
                "ALTER TABLE fairness_owner "
                "ADD COLUMN uncapped INTEGER NOT NULL DEFAULT 0"
            )
        if "blocked" not in fairness_owner_cols:
            cur.execute(
                "ALTER TABLE fairness_owner "
                "ADD COLUMN blocked INTEGER NOT NULL DEFAULT 0"
            )
            if "paused" in fairness_owner_cols:
                # Old ``paused`` rows meant stop new admissions. Preserve that
                # operator intent under the explicit ``blocked`` name.
                cur.execute("UPDATE fairness_owner SET blocked = paused")

    @property
    def path(self) -> Path:
        """On-disk path to the sqlite database file (for the admin server
        + the operator CLI to open additional read-only connections)."""
        return self._path

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._conn.close()

    # ── Rollouts ─────────────────────────────────────────────────────────────

    def insert_rollout(self, record: RolloutRecord) -> None:
        with self._tx() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO rollouts (
                        rollout_id, template, status, reason, request_id, task_key, group_id,
                        owner_id, project_id, run_id, node_id, sandbox_id,
                        init_json, metadata_json, final_reward,
                        trajectory_sink, trajectory_node_id, trajectory_uri, trajectory_size_bytes,
                        created_at, last_touched_at
                    ) VALUES (
                        :rollout_id, :template, :status, :reason, :request_id, :task_key, :group_id,
                        :owner_id, :project_id, :run_id, :node_id, :sandbox_id,
                        :init_json, :metadata_json, :final_reward,
                        :trajectory_sink, :trajectory_node_id, :trajectory_uri, :trajectory_size_bytes,
                        :created_at, :last_touched_at
                    )
                    """,
                    _rollout_to_row(record),
                )
            except sqlite3.IntegrityError as exc:
                raise KeyError(f"rollout {record.rollout_id} already exists") from exc

    def get_rollout(self, rollout_id: str) -> RolloutRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM rollouts WHERE rollout_id = ?", (rollout_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown rollout_id {rollout_id}")
            steps = self._load_steps(rollout_id)
        return _rollout_from_row(row, steps)

    def update_rollout(self, rollout_id: str, **fields: Any) -> RolloutRecord:
        if not fields:
            return self.get_rollout(rollout_id)
        # Coerce dataclass-typed fields to their JSON / text representation.
        coerced: dict[str, Any] = {}
        for k, v in fields.items():
            if k in {"init_params"}:
                coerced["init_json"] = json.dumps(v)
            elif k == "metadata":
                coerced["metadata_json"] = json.dumps(v)
            elif k == "status" and isinstance(v, RolloutStatus):
                coerced["status"] = v.value
            else:
                coerced[k] = v
        coerced["last_touched_at"] = time.time()
        assignments = ", ".join(f"{k} = :{k}" for k in coerced)
        coerced["rollout_id"] = rollout_id

        with self._tx() as cur:
            cur.execute(
                f"UPDATE rollouts SET {assignments} WHERE rollout_id = :rollout_id",
                coerced,
            )
            if cur.rowcount == 0:
                raise KeyError(f"unknown rollout_id {rollout_id}")
        return self.get_rollout(rollout_id)

    def list_rollouts(self) -> list[RolloutRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM rollouts ORDER BY created_at"
            ).fetchall()
        return [_rollout_from_row(r, self._load_steps(r["rollout_id"])) for r in rows]

    def list_rollouts_page(
        self,
        *,
        status: str | None = None,
        template: str | None = None,
        created_after: float | None = None,
        owner_id: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[RolloutRecord], bool]:
        """Return one newest-first rollout page plus whether another page exists.

        ``owner_id`` scopes the page to one tenant (multi-user admin views);
        ``None`` returns every owner (operator/admin scope).
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")

        clauses: list[str] = []
        params: dict[str, object] = {
            "limit": limit + 1,
            "offset": offset,
        }
        if status:
            clauses.append("status = :status")
            params["status"] = status
        if template:
            clauses.append("template = :template")
            params["template"] = template
        if created_after is not None:
            clauses.append("created_at >= :created_after")
            params["created_after"] = created_after
        if owner_id is not None:
            clauses.append("owner_id = :owner_id")
            params["owner_id"] = owner_id
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM rollouts
                {where_sql}
                ORDER BY created_at DESC, rollout_id DESC
                LIMIT :limit OFFSET :offset
                """,
                params,
            ).fetchall()

        has_next = len(rows) > limit
        page_rows = rows[:limit]
        return (
            [
                _rollout_from_row(r, self._load_steps(r["rollout_id"]))
                for r in page_rows
            ],
            has_next,
        )

    def append_step(self, rollout_id: str, step: Step) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO rollout_steps (
                    rollout_id, step_index, action_json, obs_json, reward,
                    done, truncated, info_json, ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rollout_id,
                    step.index,
                    json.dumps(step.action),
                    json.dumps(step.obs),
                    step.reward,
                    int(step.done),
                    int(step.truncated),
                    json.dumps(step.info),
                    step.ts,
                ),
            )
            cur.execute(
                """
                UPDATE rollouts
                   SET final_reward = final_reward + ?,
                       last_touched_at = ?
                 WHERE rollout_id = ?
                """,
                (step.reward, time.time(), rollout_id),
            )

    # ── Sandboxes ────────────────────────────────────────────────────────────

    def insert_sandbox(self, record: SandboxRecord) -> None:
        with self._tx() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO sandboxes (
                        sandbox_id, backend, backend_ref, stub_endpoint, template,
                        image, node_id, rollout_id, status, owner_count, created_at,
                        effective_resources_json
                    ) VALUES (
                        :sandbox_id, :backend, :backend_ref, :stub_endpoint, :template,
                        :image, :node_id, :rollout_id, :status, :owner_count, :created_at,
                        :effective_resources_json
                    )
                    """,
                    record.model_dump(),
                )
            except sqlite3.IntegrityError as exc:
                raise KeyError(f"sandbox {record.sandbox_id} already exists") from exc

    def get_sandbox(self, sandbox_id: str) -> SandboxRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sandboxes WHERE sandbox_id = ?", (sandbox_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown sandbox_id {sandbox_id}")
        return _sandbox_from_row(row)

    def update_sandbox(self, sandbox_id: str, **fields: Any) -> SandboxRecord:
        if not fields:
            return self.get_sandbox(sandbox_id)
        assignments = ", ".join(f"{k} = :{k}" for k in fields)
        params = dict(fields)
        params["sandbox_id"] = sandbox_id
        with self._tx() as cur:
            cur.execute(
                f"UPDATE sandboxes SET {assignments} WHERE sandbox_id = :sandbox_id",
                params,
            )
            if cur.rowcount == 0:
                raise KeyError(f"unknown sandbox_id {sandbox_id}")
        return self.get_sandbox(sandbox_id)

    def remove_sandbox(self, sandbox_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM sandboxes WHERE sandbox_id = ?", (sandbox_id,))

    def list_sandboxes(self) -> list[SandboxRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sandboxes ORDER BY created_at"
            ).fetchall()
        return [_sandbox_from_row(r) for r in rows]

    # ── Idempotency ──────────────────────────────────────────────────────────

    def lookup_idempotent(self, request_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT rollout_id FROM idempotency WHERE request_id = ?", (request_id,)
            ).fetchone()
        return None if row is None else str(row["rollout_id"])

    def record_idempotent(self, request_id: str, rollout_id: str) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO idempotency (request_id, rollout_id, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    rollout_id = excluded.rollout_id,
                    created_at = excluded.created_at
                """,
                (request_id, rollout_id, time.time()),
            )

    # ── Events ───────────────────────────────────────────────────────────────

    def append_event(
        self,
        kind: str,
        *,
        rollout_id: str | None = None,
        sandbox_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        ts = time.time()
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO events (ts, kind, rollout_id, sandbox_id, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, kind, rollout_id, sandbox_id, json.dumps(payload or {})),
            )
            seq = cur.lastrowid
        assert seq is not None
        return EventRecord(
            seq=int(seq),
            ts=ts,
            rollout_id=rollout_id,
            sandbox_id=sandbox_id,
            kind=kind,
            payload=payload or {},
        )

    def events_since(self, seq: int) -> Iterable[EventRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE seq > ? ORDER BY seq", (seq,)
            ).fetchall()
        return [
            EventRecord(
                seq=int(r["seq"]),
                ts=float(r["ts"]),
                rollout_id=r["rollout_id"],
                sandbox_id=r["sandbox_id"],
                kind=r["kind"],
                payload=json.loads(r["payload_json"]),
            )
            for r in rows
        ]

    # ── Audit (spec 19) ──────────────────────────────────────────────────────

    def append_audit(
        self,
        kind: str,
        *,
        role: str | None = None,
        digest_hint: str | None = None,
        method: str | None = None,
        source: str | None = None,
        result: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> AuditRecord:
        ts = time.time()
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO audit (
                    ts, kind, role, digest_hint, method, source, result, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, kind, role, digest_hint, method, source, result,
                    json.dumps(payload or {}),
                ),
            )
            seq = cur.lastrowid
        assert seq is not None
        return AuditRecord(
            seq=int(seq), ts=ts, kind=kind,
            role=role, digest_hint=digest_hint,
            method=method, source=source, result=result,
            payload=payload or {},
        )

    def audit_since(self, seq: int) -> Iterable[AuditRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit WHERE seq > ? ORDER BY seq", (seq,)
            ).fetchall()
        return [
            AuditRecord(
                seq=int(r["seq"]),
                ts=float(r["ts"]),
                kind=r["kind"],
                role=r["role"],
                digest_hint=r["digest_hint"],
                method=r["method"],
                source=r["source"],
                result=r["result"],
                payload=json.loads(r["payload_json"]),
            )
            for r in rows
        ]

    # ── Live node registry mirror ────────────────────────────────────────────

    def record_node_connected(
        self,
        node_id: str,
        *,
        backends: list[str] | None = None,
        stream_epoch: str | None = None,
        instance_id: str | None = None,
        isolation_capable: bool = False,
    ) -> NodeRecord:
        ts = time.time()
        backends_json = json.dumps(backends or [])
        cap = 1 if isolation_capable else 0
        with self._tx() as cur:
            existing = cur.execute(
                "SELECT connected_at, status FROM nodes WHERE node_id = ?", (node_id,),
            ).fetchone()
            connected_at = (
                float(existing["connected_at"])
                if existing is not None and existing["status"] == "connected"
                else ts
            )
            # ``isolation_capable`` is refreshed on (re)connect from NodeHello.
            # pinned_cpus_* are NOT touched here — they're updated by the
            # heartbeat mirror; INSERT defaults them to 0 for a brand-new row and
            # the UPDATE branch leaves the last-known values intact.
            cur.execute(
                """
                INSERT INTO nodes (
                    node_id, status, connected_at, last_seen_at,
                    stream_epoch, instance_id, backends_json, isolation_capable
                ) VALUES (?, 'connected', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    status='connected',
                    connected_at=excluded.connected_at,
                    last_seen_at=excluded.last_seen_at,
                    stream_epoch=excluded.stream_epoch,
                    instance_id=excluded.instance_id,
                    backends_json=excluded.backends_json,
                    isolation_capable=excluded.isolation_capable
                """,
                (
                    node_id, connected_at, ts, stream_epoch, instance_id,
                    backends_json, cap,
                ),
            )
        return NodeRecord(
            node_id=node_id,
            status="connected",
            connected_at=connected_at,
            last_seen_at=ts,
            stream_epoch=stream_epoch,
            instance_id=instance_id,
            backends=backends or [],
            isolation_capable=isolation_capable,
        )

    def record_node_disconnected(self, node_id: str) -> None:
        ts = time.time()
        with self._tx() as cur:
            cur.execute(
                "UPDATE nodes SET status='lost', last_seen_at=? WHERE node_id = ?",
                (ts, node_id),
            )

    def update_node_seen(self, node_id: str, ts: float) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE nodes SET last_seen_at=? WHERE node_id = ?",
                (ts, node_id),
            )

    def update_node_pinned_cpus(
        self, node_id: str, *, free: int, total: int,
    ) -> None:
        """P6 step-2c — mirror the node's last-known pinnable-CPU counts (from
        the heartbeat) so out-of-process readers see them. No-op for an unknown
        node (the UPDATE simply matches no row)."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE nodes SET pinned_cpus_free=?, pinned_cpus_total=? "
                "WHERE node_id = ?",
                (int(free), int(total), node_id),
            )

    def update_node_health(self, node_id: str, health_json: str) -> None:
        """Stage-1 — upsert the node's latest heartbeat health stats."""
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO node_health (node_id, health_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    health_json=excluded.health_json,
                    updated_at=excluded.updated_at
                """,
                (node_id, health_json, time.time()),
            )

    def list_node_health(self) -> dict[str, str]:
        """Stage-1 — ``node_id -> health_json`` for every node that has
        reported. The admin "Cluster health" page parses each value."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, health_json FROM node_health",
            ).fetchall()
        return {r["node_id"]: r["health_json"] for r in rows}

    def update_node_aimd_limit(self, node_id: str, limit: int) -> None:
        """Stage-3 — upsert a node's adaptive admission limit."""
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO node_aimd_limit (node_id, admission_limit, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    admission_limit=excluded.admission_limit,
                    updated_at=excluded.updated_at
                """,
                (node_id, limit, time.time()),
            )

    def list_node_aimd_limits(self) -> dict[str, int]:
        """Stage-3 — ``node_id -> adaptive admission limit`` for every
        node the AIMD controller has produced a limit for."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, admission_limit FROM node_aimd_limit",
            ).fetchall()
        return {r["node_id"]: r["admission_limit"] for r in rows}

    def list_nodes(self, *, status: str | None = None) -> list[NodeRecord]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM nodes ORDER BY node_id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM nodes WHERE status = ? ORDER BY node_id",
                    (status,),
                ).fetchall()
        return [
            NodeRecord(
                node_id=r["node_id"],
                status=r["status"],
                connected_at=float(r["connected_at"]),
                last_seen_at=float(r["last_seen_at"]),
                stream_epoch=r["stream_epoch"],
                instance_id=r["instance_id"],
                backends=json.loads(r["backends_json"]),
                isolation_capable=bool(r["isolation_capable"]),
                pinned_cpus_free=int(r["pinned_cpus_free"]),
                pinned_cpus_total=int(r["pinned_cpus_total"]),
            )
            for r in rows
        ]

    def prune_lost_nodes(self, *, keep: set[str]) -> list[str]:
        """Startup reconciliation — reap ``lost`` node rows whose node_id is
        NOT in ``keep`` (the current nodes.yaml roster), plus their per-node
        satellite rows (``node_health``, ``node_aimd_limit``). Rollout history
        in ``raw_rollouts`` is deliberately left intact. Returns the pruned
        node_ids. See ``NodeRecord`` for why this is the one exception to the
        "rows are never deleted" rule."""
        with self._tx() as cur:
            rows = cur.execute(
                "SELECT node_id FROM nodes WHERE status = 'lost'"
            ).fetchall()
            pruned = [str(r["node_id"]) for r in rows if r["node_id"] not in keep]
            for nid in pruned:
                cur.execute("DELETE FROM nodes WHERE node_id = ?", (nid,))
                cur.execute("DELETE FROM node_health WHERE node_id = ?", (nid,))
                cur.execute("DELETE FROM node_aimd_limit WHERE node_id = ?", (nid,))
        return pruned

    def prune_expired(
        self,
        *,
        now: float,
        audit_retention_days: int | None = 30,
        events_retention_days: int | None = 14,
        raw_rollout_retention_days: int | None = 14,
        batch_size: int = 10_000,
    ) -> dict[str, int]:
        """Retention GC for the append-only / terminal tables (spec 20 matrix).
        Hard-deletes rows past their per-table retention window:

        - ``audit`` — ``ts`` older than ``audit_retention_days`` (spec 19 trail);
        - ``events`` — ``ts`` older than ``events_retention_days``;
        - ``raw_rollouts`` — TERMINAL rows (released/cancelled/failed/reaped/
          capacity_rejected) whose ``COALESCE(finished_at, created_at)`` is older
          than ``raw_rollout_retention_days``. Active rows (acquiring/running) are
          never touched; on-disk rollout artifacts are GC'd separately by the
          run-dir janitor.

        A ``None`` window skips that table. Deletes run in ``batch_size`` chunks
        so a large first purge doesn't hold one long write lock (WAL keeps
        readers unblocked between batches). NOTE: this frees pages for REUSE — it
        does NOT shrink ``state.db`` on disk. Run ``VACUUM`` (``xrlenv db vacuum``)
        once to return the freed space to the filesystem.
        """
        day = 86400.0

        def _batched(table: str, key_col: str, where: str, params: tuple[Any, ...]) -> int:
            total = 0
            while True:
                with self._tx() as cur:
                    # table / key_col / where are internal literals, never caller input.
                    cur.execute(
                        f"DELETE FROM {table} WHERE {key_col} IN "
                        f"(SELECT {key_col} FROM {table} WHERE {where} LIMIT ?)",
                        (*params, batch_size),
                    )
                    n = cur.rowcount or 0
                total += n
                if n < batch_size:
                    break
            return total

        def _prune_raw_rollouts_accumulating(cutoff: float) -> int:
            # Like _batched for raw_rollouts, but first folds each batch's
            # (owner_id, status) tallies into owner_rollout_lifetime so the
            # /users scoreboard stays all-time across GC. The accumulate + the
            # DELETE share ONE transaction per batch, so a row is counted iff
            # it's removed (crash-safe, never double-counts on retry).
            where = (
                "status IN ('released','cancelled','failed','reaped',"
                "'capacity_rejected') AND COALESCE(finished_at, created_at) < ?"
            )
            total = 0
            while True:
                with self._tx() as cur:
                    ids = [
                        r[0]
                        for r in cur.execute(
                            f"SELECT rollout_id FROM raw_rollouts WHERE {where} "
                            "LIMIT ?",
                            (cutoff, batch_size),
                        ).fetchall()
                    ]
                    if not ids:
                        break
                    placeholders = ",".join("?" * len(ids))
                    cur.execute(
                        "INSERT INTO owner_rollout_lifetime (owner_id, status, count) "
                        "SELECT owner_id, status, COUNT(*) FROM raw_rollouts "
                        f"WHERE rollout_id IN ({placeholders}) "
                        "GROUP BY owner_id, status "
                        "ON CONFLICT(owner_id, status) DO UPDATE SET "
                        "count = count + excluded.count",
                        ids,
                    )
                    cur.execute(
                        f"DELETE FROM raw_rollouts WHERE rollout_id IN ({placeholders})",
                        ids,
                    )
                    n = cur.rowcount or 0
                total += n
                if n < batch_size:
                    break
            return total

        out = {"audit": 0, "events": 0, "raw_rollouts": 0}
        if audit_retention_days is not None:
            out["audit"] = _batched(
                "audit", "seq", "ts < ?", (now - audit_retention_days * day,)
            )
        if events_retention_days is not None:
            out["events"] = _batched(
                "events", "seq", "ts < ?", (now - events_retention_days * day,)
            )
        if raw_rollout_retention_days is not None:
            out["raw_rollouts"] = _prune_raw_rollouts_accumulating(
                now - raw_rollout_retention_days * day
            )
        return out

    # ── Admission queue ──────────────────────────────────────────────────────

    def enqueue_pending(self, record: PendingRolloutRecord) -> None:
        with self._tx() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO pending_rollouts (
                        pending_id, template, init_json, request_id, task_key, group_id,
                        deadline_json, queue_partition, submitted_at
                    ) VALUES (
                        :pending_id, :template, :init_json, :request_id, :task_key, :group_id,
                        :deadline_json, :queue_partition, :submitted_at
                    )
                    """,
                    {
                        "pending_id": record.pending_id,
                        "template": record.template,
                        "init_json": json.dumps(record.init_params),
                        "request_id": record.request_id,
                        "task_key": record.task_key,
                        "group_id": record.group_id,
                        "deadline_json": json.dumps(record.deadline_json),
                        "queue_partition": record.queue_partition,
                        "submitted_at": record.submitted_at,
                    },
                )
            except sqlite3.IntegrityError as exc:
                raise KeyError(
                    f"pending rollout {record.pending_id} already queued"
                ) from exc

    def list_pending(self, *, partition: str | None = None) -> list[PendingRolloutRecord]:
        with self._lock:
            if partition is None:
                rows = self._conn.execute(
                    "SELECT * FROM pending_rollouts ORDER BY submitted_at"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM pending_rollouts WHERE queue_partition = ? "
                    "ORDER BY submitted_at",
                    (partition,),
                ).fetchall()
        return [_pending_from_row(r) for r in rows]

    def remove_pending(self, pending_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM pending_rollouts WHERE pending_id = ?", (pending_id,))

    # ── Trajectory sealing ───────────────────────────────────────────────────

    def seal_trajectory(self, rollout_id: str) -> Trajectory:
        record = self.get_rollout(rollout_id)
        return Trajectory(
            rollout_id=record.rollout_id,
            template=record.template,
            steps=list(record.steps),
            status=record.status,
            reason=record.reason,
            final_reward=record.final_reward,
            metadata=_metadata_with_node_id(record.metadata, record.node_id),
        )

    # ── P1.6.b — control-plane-driven image builds ──────────────────────

    def record_build_plan(
        self, *, plan_id: str, applied_by: str, plan_json: str,
        name: str | None = None,
    ) -> BuildPlanRecord:
        # Upsert: on re-apply (force=True or partial_failure
        # residual-only retry), advance applied_at + applied_by
        # so the admin /builds view reflects the latest operator
        # intent. Caller short-circuits no-op re-applies before
        # this call, so any record_build_plan call corresponds
        # to an apply that's about to do real work.
        now = time.time()
        with self._tx() as cur:
            existing = cur.execute(
                "SELECT plan_id, applied_at, applied_by, plan_json, status, name "
                "FROM build_plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
            if existing is not None:
                cur.execute(
                    "UPDATE build_plans SET applied_at = ?, applied_by = ?, "
                    "plan_json = ?, name = ? WHERE plan_id = ?",
                    (now, applied_by, plan_json, name, plan_id),
                )
                return BuildPlanRecord(
                    plan_id=existing["plan_id"],
                    applied_at=now,
                    applied_by=applied_by,
                    plan_json=plan_json,
                    status=cast(BuildPlanStatus, str(existing["status"])),
                    name=name,
                )
            cur.execute(
                "INSERT INTO build_plans (plan_id, applied_at, applied_by, "
                "plan_json, status, name) VALUES (?, ?, ?, ?, 'in_flight', ?)",
                (plan_id, now, applied_by, plan_json, name),
            )
            return BuildPlanRecord(
                plan_id=plan_id, applied_at=now, applied_by=applied_by,
                plan_json=plan_json, status="in_flight", name=name,
            )

    def update_build_plan_status(
        self, plan_id: str, status: BuildPlanStatus,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE build_plans SET status = ? WHERE plan_id = ?",
                (status, plan_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"unknown plan_id {plan_id}")

    def try_update_build_plan_status(
        self, plan_id: str, *,
        expected_current: BuildPlanStatus,
        new_status: BuildPlanStatus,
    ) -> bool:
        """SQL-level CAS: UPDATE ... WHERE plan_id=? AND status=?.
        The WHERE clause is evaluated atomically with the SET, so
        a concurrent ``update_build_plan_status(plan_id,
        "cancelled")`` from the cancel orchestrator either lands
        before our SET (rowcount=0, we report False) or after our
        SET (the cancel's update wins on the next transaction)."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE build_plans SET status = ? "
                "WHERE plan_id = ? AND status = ?",
                (new_status, plan_id, expected_current),
            )
            return bool(cur.rowcount > 0)

    def get_build_plan(self, plan_id: str) -> BuildPlanRecord | None:
        row = self._conn.execute(
            "SELECT plan_id, applied_at, applied_by, plan_json, status, name "
            "FROM build_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        if row is None:
            return None
        return BuildPlanRecord(
            plan_id=str(row["plan_id"]),
            applied_at=float(row["applied_at"]),
            applied_by=str(row["applied_by"]),
            plan_json=str(row["plan_json"]),
            status=cast(BuildPlanStatus, str(row["status"])),
            name=(str(row["name"]) if row["name"] is not None else None),
        )

    def list_build_plans(
        self, *, status: BuildPlanStatus | None = None,
    ) -> list[BuildPlanRecord]:
        if status is None:
            rows = self._conn.execute(
                "SELECT plan_id, applied_at, applied_by, plan_json, status, name "
                "FROM build_plans ORDER BY applied_at DESC",
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT plan_id, applied_at, applied_by, plan_json, status, name "
                "FROM build_plans WHERE status = ? ORDER BY applied_at DESC",
                (status,),
            ).fetchall()
        return [
            BuildPlanRecord(
                plan_id=str(r["plan_id"]),
                applied_at=float(r["applied_at"]),
                applied_by=str(r["applied_by"]),
                plan_json=str(r["plan_json"]),
                status=cast(BuildPlanStatus, str(r["status"])),
                name=(str(r["name"]) if r["name"] is not None else None),
            )
            for r in rows
        ]

    def record_assignment(self, record: BuildAssignmentRecord) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO build_plan_assignments "
                "(plan_id, node_id, image_ref, benchmark, status, "
                " started_at, completed_at, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.plan_id, record.node_id, record.image_ref,
                    record.benchmark, record.status,
                    record.started_at, record.completed_at, record.error,
                ),
            )

    def update_assignment_status(
        self,
        *,
        plan_id: str,
        node_id: str,
        image_ref: str,
        status: BuildAssignmentStatus,
        error: str | None = None,
    ) -> None:
        now = time.time()
        with self._tx() as cur:
            existing = cur.execute(
                "SELECT started_at FROM build_plan_assignments "
                "WHERE plan_id = ? AND node_id = ? AND image_ref = ?",
                (plan_id, node_id, image_ref),
            ).fetchone()
            if existing is None:
                raise KeyError(
                    f"unknown assignment ({plan_id}, {node_id}, {image_ref})",
                )
            started_at = existing["started_at"]
            if status == "building" and started_at is None:
                started_at = now
            completed_at: float | None = None
            if status in ("done", "failed", "cancelled"):
                completed_at = now
            cur.execute(
                "UPDATE build_plan_assignments "
                "SET status = ?, started_at = ?, completed_at = ?, error = ? "
                "WHERE plan_id = ? AND node_id = ? AND image_ref = ?",
                (status, started_at, completed_at, error,
                 plan_id, node_id, image_ref),
            )

    def list_assignments(
        self, plan_id: str,
    ) -> list[BuildAssignmentRecord]:
        rows = self._conn.execute(
            "SELECT plan_id, node_id, image_ref, benchmark, status, "
            "started_at, completed_at, error FROM build_plan_assignments "
            "WHERE plan_id = ? ORDER BY node_id, image_ref",
            (plan_id,),
        ).fetchall()
        return [
            BuildAssignmentRecord(
                plan_id=str(r["plan_id"]),
                node_id=str(r["node_id"]),
                image_ref=str(r["image_ref"]),
                benchmark=str(r["benchmark"]),
                status=cast(BuildAssignmentStatus, str(r["status"])),
                started_at=(
                    float(r["started_at"]) if r["started_at"] is not None else None
                ),
                completed_at=(
                    float(r["completed_at"]) if r["completed_at"] is not None else None
                ),
                error=(str(r["error"]) if r["error"] is not None else None),
            )
            for r in rows
        ]

    def delete_assignments(self, plan_id: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "DELETE FROM build_plan_assignments WHERE plan_id = ?",
                (plan_id,),
            )

    def find_registered_preferred_home(
        self, image_ref: str,
    ) -> str | None:
        # JOIN against build_plans so we get the most-recent plan's
        # row by applied_at. The status_idx on (plan_id, status)
        # already covers the per-plan filter; sqlite handles the
        # ORDER BY without scanning the whole table at the
        # 100s-of-plans scale we target in phase 1.
        row = self._conn.execute(
            "SELECT a.node_id FROM build_plan_assignments AS a "
            "JOIN build_plans AS p ON p.plan_id = a.plan_id "
            "WHERE a.image_ref = ? AND a.status = 'registered' "
            "ORDER BY p.applied_at DESC LIMIT 1",
            (image_ref,),
        ).fetchone()
        return str(row["node_id"]) if row is not None else None

    # ── Raw rollouts (P1.7.B.3) ──────────────────────────────────────────────

    def record_raw_rollout(self, record: RawRolloutRecord) -> None:
        with self._tx() as cur:
            existing = cur.execute(
                "SELECT 1 FROM raw_rollouts WHERE rollout_id = ?",
                (record.rollout_id,),
            ).fetchone()
            if existing is not None:
                raise KeyError(
                    f"raw rollout {record.rollout_id} already exists",
                )
            cur.execute(
                "INSERT INTO raw_rollouts ("
                "rollout_id, status, image, node_id, container_id, "
                "container_name, artifact_path, displayed_name, "
                "task_key, group_id, fleet_id, owner_id, "
                "created_at, finished_at, error, "
                "deadline_at, effective_resources_json, container_runtime"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.rollout_id, record.status, record.image,
                    record.node_id, record.container_id,
                    record.container_name, record.artifact_path,
                    record.displayed_name,
                    record.task_key, record.group_id, record.fleet_id,
                    record.owner_id,
                    record.created_at,
                    record.finished_at, record.error,
                    record.deadline_at, record.effective_resources_json,
                    record.container_runtime,
                ),
            )

    def update_raw_rollout(
        self, rollout_id: str, **fields: Any,
    ) -> RawRolloutRecord:
        # Fetch existing → apply updates → write back. Same shape as
        # update_rollout for case-1; the per-row dict overwrites
        # named columns only, leaving others untouched.
        allowed = {
            "status", "node_id", "container_id", "container_name",
            "artifact_path", "displayed_name",
            "task_key", "group_id", "fleet_id",
            "finished_at", "error",
            "deadline_at", "effective_resources_json", "container_runtime",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(
                f"update_raw_rollout: unknown fields {sorted(bad)}; "
                f"allowed: {sorted(allowed)}",
            )
        with self._tx() as cur:
            row = cur.execute(
                "SELECT rollout_id, status, image, node_id, container_id, "
                "container_name, artifact_path, displayed_name, "
                "task_key, group_id, fleet_id, owner_id, "
                "created_at, finished_at, error, "
                "deadline_at, effective_resources_json, container_runtime "
                "FROM raw_rollouts WHERE rollout_id = ?",
                (rollout_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown raw rollout {rollout_id}")
            current = self._row_to_raw_rollout(row)
            updated = current.model_copy(update=fields)
            cur.execute(
                "UPDATE raw_rollouts SET "
                "status = ?, node_id = ?, container_id = ?, "
                "container_name = ?, artifact_path = ?, "
                "displayed_name = ?, task_key = ?, group_id = ?, "
                "fleet_id = ?, "
                "finished_at = ?, error = ?, "
                "deadline_at = ?, effective_resources_json = ?, "
                "container_runtime = ? "
                "WHERE rollout_id = ?",
                (
                    updated.status, updated.node_id, updated.container_id,
                    updated.container_name, updated.artifact_path,
                    updated.displayed_name,
                    updated.task_key, updated.group_id,
                    updated.fleet_id,
                    updated.finished_at, updated.error,
                    updated.deadline_at, updated.effective_resources_json,
                    updated.container_runtime,
                    rollout_id,
                ),
            )
            return updated

    def get_raw_rollout(self, rollout_id: str) -> RawRolloutRecord | None:
        row = self._conn.execute(
            "SELECT rollout_id, status, image, node_id, container_id, "
            "container_name, artifact_path, displayed_name, "
            "task_key, group_id, fleet_id, owner_id, "
            "created_at, finished_at, error, "
            "deadline_at, effective_resources_json, container_runtime "
            "FROM raw_rollouts WHERE rollout_id = ?",
            (rollout_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_raw_rollout(row)

    def list_raw_rollouts(
        self,
        *,
        status: RawRolloutStatus | None = None,
        since_after: float | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[RawRolloutRecord]:
        sql = (
            "SELECT rollout_id, status, image, node_id, container_id, "
            "container_name, artifact_path, displayed_name, "
            "task_key, group_id, fleet_id, owner_id, "
            "created_at, finished_at, error, "
            "deadline_at, effective_resources_json, container_runtime FROM raw_rollouts"
        )
        clauses, params = self._raw_filter_clauses(
            status, since_after, task_key, group_id, owner_id,
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
            if offset:
                sql += " OFFSET ?"
                params.append(offset)
        elif offset:
            # SQLite requires LIMIT before OFFSET; use -1 as
            # "unbounded" per SQLite docs.
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_raw_rollout(r) for r in rows]

    def count_raw_rollouts(
        self,
        *,
        status: RawRolloutStatus | None = None,
        since_after: float | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) AS n FROM raw_rollouts"
        clauses, params = self._raw_filter_clauses(
            status, since_after, task_key, group_id, owner_id,
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        row = self._conn.execute(sql, params).fetchone()
        return int(row["n"]) if row is not None else 0

    def aggregate_raw_rollouts_by_owner_status(self) -> dict[str, dict[str, int]]:
        """Per-owner, per-status raw-rollout counts via a single ``GROUP BY``.

        Result is ``{owner_id: {status: count}}`` — O(owners x statuses), so it
        stays cheap regardless of how many rollouts the cluster has run.
        """
        rows = self._conn.execute(
            "SELECT owner_id, status, COUNT(*) AS n "
            "FROM raw_rollouts GROUP BY owner_id, status"
        ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(str(row["owner_id"]), {})[str(row["status"])] = int(
                row["n"]
            )
        return out

    def raw_rollouts_created_span(
        self,
    ) -> tuple[float | None, float | None]:
        """``(min, max)`` ``created_at`` across ``raw_rollouts`` (epoch secs).

        ``(None, None)`` when the table is empty. Backs the /users page's
        "stats reflect rollouts from X to Y" banner so the per-owner scoreboard
        is honestly labelled as a retention-windowed view — ``raw_rollouts`` is
        GC'd on ``--raw-rollout-retention-days`` — rather than misread as
        all-time. Single indexed aggregate, O(1)-ish.
        """
        row = self._conn.execute(
            "SELECT MIN(created_at) AS lo, MAX(created_at) AS hi FROM raw_rollouts"
        ).fetchone()
        if row is None or row["lo"] is None:
            return (None, None)
        return (float(row["lo"]), float(row["hi"]))

    def owner_rollout_lifetime(self) -> dict[str, dict[str, int]]:
        """Per-owner/status tallies of raw rollouts already removed by the
        retention GC — the durable half of the /users lifetime scoreboard.

        For all-time totals do NOT merge this with
        :meth:`aggregate_raw_rollouts_by_owner_status` in Python — reading the
        two as separate statements races the retention janitor (a row it moves
        live→lifetime between the reads is double- or under-counted). Use
        :meth:`aggregate_raw_rollouts_all_time_by_owner_status`, which reads both
        in one snapshot. This accessor is for inspection/debugging of the durable
        tally alone.
        """
        rows = self._conn.execute(
            "SELECT owner_id, status, count FROM owner_rollout_lifetime"
        ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(str(row["owner_id"]), {})[str(row["status"])] = int(
                row["count"]
            )
        return out

    def aggregate_raw_rollouts_all_time_by_owner_status(
        self,
    ) -> dict[str, dict[str, int]]:
        """All-time per-owner/status counts = live ``raw_rollouts`` + the
        durable ``owner_rollout_lifetime`` tally, in ONE query.

        Reading the two sources as separate statements races the retention
        janitor: it moves a row from live→lifetime atomically, so a row read as
        live *then* re-read as lifetime (or vice-versa) could be double- or
        under-counted. A single ``UNION ALL`` aggregate sees one consistent
        snapshot, so every rollout is counted exactly once.

        The single statement is necessary but NOT sufficient on its own: the
        janitor's accumulate-then-delete runs as one ``_tx()`` transaction on
        this *same* shared connection, and mid-transaction (lifetime row
        inserted, ``raw_rollouts`` row not yet deleted) it would over-count.
        So the read must also hold ``self._lock`` — that makes it mutually
        exclusive with ``_tx()`` and guarantees it only ever sees committed
        state, never a prune's uncommitted intermediate.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT owner_id, status, SUM(n) AS n FROM ("
                "  SELECT owner_id, status, COUNT(*) AS n FROM raw_rollouts"
                "    GROUP BY owner_id, status"
                "  UNION ALL"
                "  SELECT owner_id, status, count AS n FROM owner_rollout_lifetime"
                ") GROUP BY owner_id, status"
            ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(str(row["owner_id"]), {})[str(row["status"])] = int(
                row["n"]
            )
        return out

    def count_raw_rollouts_by_node(self) -> dict[str | None, int]:
        """All-time raw-rollout count per assigned node (``GROUP BY node_id``).

        A ``None`` key carries rollouts not yet assigned to a node (the brief
        ``acquiring`` window before placement).
        """
        rows = self._conn.execute(
            "SELECT node_id, COUNT(*) AS n FROM raw_rollouts GROUP BY node_id"
        ).fetchall()
        out: dict[str | None, int] = {}
        for row in rows:
            node_id = row["node_id"]
            out[str(node_id) if node_id is not None else None] = int(row["n"])
        return out

    # ── Admin overview/health aggregates (bounded — never load whole table) ──
    #
    # The admin overview + cluster-health pages need a handful of counts, not
    # the row bodies. Loading every ``raw_rollouts`` row to tally them in
    # Python was O(total rollouts) per page render — at 168k rows it took long
    # enough that the reader pinned the WAL and stalled the control plane.
    # These GROUP BY / windowed COUNT queries are O(#statuses) / index-bounded
    # instead. See [[wal-runaway-cp-stall]].

    def count_raw_rollouts_by_status(self) -> dict[str, int]:
        """Per-status raw-rollout counts via one ``GROUP BY`` (uses the
        status index). O(#statuses) regardless of table size."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM raw_rollouts GROUP BY status"
        ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def count_raw_rollouts_finished_since(
        self, since_ts: float, statuses: Sequence[str],
    ) -> int:
        """Count raw rollouts in ``statuses`` whose ``finished_at >= since_ts``
        — the 'finished/failed in the last N min' tiles, without a scan."""
        statuses = tuple(statuses)
        if not statuses:
            return 0
        placeholders = ",".join("?" * len(statuses))
        row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM raw_rollouts "
            f"WHERE status IN ({placeholders}) AND finished_at >= ?",
            (*statuses, since_ts),
        ).fetchone()
        return int(row["n"]) if row is not None else 0

    def count_raw_rollouts_created_since(self, since_ts: float) -> int:
        """Count raw rollouts with ``created_at >= since_ts`` (the failure-rate
        denominator over a recent window)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM raw_rollouts WHERE created_at >= ?",
            (since_ts,),
        ).fetchone()
        return int(row["n"]) if row is not None else 0

    def active_raw_node_ids(self) -> set[str]:
        """Distinct node_ids currently hosting a non-terminal raw rollout
        (``acquiring``/``running``). Small result — one row per busy node."""
        rows = self._conn.execute(
            "SELECT DISTINCT node_id FROM raw_rollouts "
            "WHERE status IN ('acquiring','running') AND node_id IS NOT NULL"
        ).fetchall()
        return {str(r["node_id"]) for r in rows}

    def running_raw_counts_by_node(self) -> dict[str, int]:
        """Per-node count of ``running`` raw rollouts (live containers).
        ``acquiring`` rows have no container yet, so they're excluded."""
        rows = self._conn.execute(
            "SELECT node_id, COUNT(*) AS n FROM raw_rollouts "
            "WHERE status = 'running' AND node_id IS NOT NULL GROUP BY node_id"
        ).fetchall()
        return {str(r["node_id"]): int(r["n"]) for r in rows}

    def list_long_running_raw(
        self, older_than_ts: float,
    ) -> list[RawRolloutRecord]:
        """Non-terminal raw rollouts (``acquiring``/``running``) created before
        ``older_than_ts`` — the small 'long-running/queued' subset the health
        page surfaces, fetched directly instead of scanning every row."""
        rows = self._conn.execute(
            "SELECT rollout_id, status, image, node_id, container_id, "
            "container_name, artifact_path, displayed_name, "
            "task_key, group_id, fleet_id, owner_id, "
            "created_at, finished_at, error, "
            "deadline_at, effective_resources_json, container_runtime FROM raw_rollouts "
            "WHERE status IN ('acquiring','running') AND created_at < ? "
            "ORDER BY created_at",
            (older_than_ts,),
        ).fetchall()
        return [self._row_to_raw_rollout(r) for r in rows]

    # ── Fleet reservation (phase 1) ─────────────────────────────────────────

    def record_fleet_reservation(
        self, record: FleetReservationRecord,
    ) -> None:
        # INSERT OR REPLACE: an opener writes one row; a restart-rebuild
        # re-records it. Idempotent by primary key (fleet_id).
        with self._tx() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO fleet_reservations ("
                "fleet_id, node_id, footprint_json, task_key, owner, "
                "opened_ts, last_acquire_ts, container_runtime"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.fleet_id, record.node_id, record.footprint_json,
                    record.task_key, record.owner, record.opened_ts,
                    record.last_acquire_ts, record.container_runtime,
                ),
            )

    def touch_fleet_reservation(
        self, fleet_id: str, *, last_acquire_ts: float,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE fleet_reservations SET last_acquire_ts = ? "
                "WHERE fleet_id = ?",
                (last_acquire_ts, fleet_id),
            )

    def delete_fleet_reservation(self, fleet_id: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "DELETE FROM fleet_reservations WHERE fleet_id = ?",
                (fleet_id,),
            )

    def list_fleet_reservations(self) -> list[FleetReservationRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fleet_reservations ORDER BY opened_ts"
            ).fetchall()
        return [
            FleetReservationRecord(
                fleet_id=str(r["fleet_id"]),
                node_id=str(r["node_id"]),
                footprint_json=str(r["footprint_json"]),
                task_key=(
                    str(r["task_key"]) if r["task_key"] is not None else None
                ),
                owner=str(r["owner"]),
                opened_ts=float(r["opened_ts"]),
                last_acquire_ts=float(r["last_acquire_ts"]),
                container_runtime=(
                    str(r["container_runtime"])
                    if r["container_runtime"] is not None
                    else None
                ),
            )
            for r in rows
        ]

    def record_compose_project(
        self, record: ComposeProjectStateRecord,
    ) -> None:
        # INSERT OR REPLACE: one row per project; a restart-rebuild re-records it.
        with self._tx() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO compose_projects ("
                "rollout_id, project_name, node_id, footprint_json, "
                "subnet_claims_json, owner, created_ts"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.rollout_id, record.project_name, record.node_id,
                    record.footprint_json, record.subnet_claims_json,
                    record.owner, record.created_ts,
                ),
            )

    def delete_compose_project(self, rollout_id: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "DELETE FROM compose_projects WHERE rollout_id = ?",
                (rollout_id,),
            )

    def list_compose_projects(self) -> list[ComposeProjectStateRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM compose_projects ORDER BY created_ts"
            ).fetchall()
        return [
            ComposeProjectStateRecord(
                rollout_id=str(r["rollout_id"]),
                project_name=str(r["project_name"]),
                node_id=str(r["node_id"]),
                footprint_json=str(r["footprint_json"]),
                subnet_claims_json=str(r["subnet_claims_json"]),
                owner=str(r["owner"]),
                created_ts=float(r["created_ts"]),
            )
            for r in rows
        ]

    @staticmethod
    def _raw_filter_clauses(
        status: RawRolloutStatus | None,
        since_after: float | None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if since_after is not None:
            clauses.append("created_at >= ?")
            params.append(since_after)
        if task_key is not None:
            clauses.append("task_key = ?")
            params.append(task_key)
        if group_id is not None:
            clauses.append("group_id = ?")
            params.append(group_id)
        if owner_id is not None:
            clauses.append("owner_id = ?")
            params.append(owner_id)
        return clauses, params

    @staticmethod
    def _row_to_raw_rollout(row: Any) -> RawRolloutRecord:
        keys = row.keys()
        return RawRolloutRecord(
            rollout_id=str(row["rollout_id"]),
            status=cast(RawRolloutStatus, str(row["status"])),
            image=str(row["image"]),
            node_id=row["node_id"] if row["node_id"] is not None else None,
            container_id=(
                row["container_id"] if row["container_id"] is not None
                else None
            ),
            container_name=(
                row["container_name"] if row["container_name"] is not None
                else None
            ),
            artifact_path=(
                row["artifact_path"] if row["artifact_path"] is not None
                else None
            ),
            displayed_name=(
                row["displayed_name"] if row["displayed_name"] is not None
                else None
            ),
            task_key=(
                row["task_key"] if row["task_key"] is not None else None
            ),
            group_id=(
                row["group_id"] if row["group_id"] is not None else None
            ),
            fleet_id=(
                row["fleet_id"]
                if "fleet_id" in keys and row["fleet_id"] is not None
                else None
            ),
            owner_id=(
                str(row["owner_id"]) if row["owner_id"] is not None else "default"
            ),
            created_at=float(row["created_at"]),
            finished_at=(
                float(row["finished_at"])
                if row["finished_at"] is not None else None
            ),
            error=row["error"] if row["error"] is not None else None,
            # Defensive column presence check (mirrors the managed
            # ``effective_resources_json`` reader) so a partial SELECT
            # or a pre-migration row can't KeyError.
            deadline_at=(
                float(row["deadline_at"])
                if "deadline_at" in keys
                and row["deadline_at"] is not None
                else None
            ),
            effective_resources_json=(
                row["effective_resources_json"]
                if "effective_resources_json" in keys
                and row["effective_resources_json"] is not None
                else None
            ),
            container_runtime=(
                row["container_runtime"]
                if "container_runtime" in keys
                and row["container_runtime"] is not None
                else None
            ),
        )

    # ── Multi-user fair-share (live, operator-tunable) ───────────────────────

    def get_fairness_policy(self) -> FairnessPolicy:
        """Read the current fair-share policy (global + per-owner overrides).

        Returns the disabled-by-default policy (``capacity_basis=None``) when
        nothing has been configured. Cheap — two indexed reads; the admission
        worker calls this once per drain pass so live edits apply promptly.
        """
        with self._lock:
            grow = self._conn.execute(
                "SELECT capacity_basis, floor FROM fairness_global WHERE id = 1",
            ).fetchone()
            orows = self._conn.execute(
                "SELECT owner_id, weight, hard_cap, uncapped, blocked "
                "FROM fairness_owner",
            ).fetchall()
        capacity_basis = (
            int(grow["capacity_basis"])
            if grow is not None and grow["capacity_basis"] is not None
            else None
        )
        floor = int(grow["floor"]) if grow is not None else 1
        overrides = {
            str(r["owner_id"]): FairnessOwnerOverride(
                owner_id=str(r["owner_id"]),
                weight=float(r["weight"]),
                hard_cap=(int(r["hard_cap"]) if r["hard_cap"] is not None else None),
                uncapped=bool(r["uncapped"]),
                blocked=bool(r["blocked"]),
            )
            for r in orows
        }
        return FairnessPolicy(
            capacity_basis=capacity_basis, floor=floor, overrides=overrides,
        )

    def set_fairness_global(
        self, *, capacity_basis: int | None, floor: int = 1,
    ) -> None:
        """Set the default per-owner cap. ``capacity_basis=None`` disables
        fairness (the admission queue stops applying owner caps)."""
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO fairness_global (id, capacity_basis, floor) "
                "VALUES (1, :cb, :fl) "
                "ON CONFLICT(id) DO UPDATE SET capacity_basis = :cb, floor = :fl",
                {"cb": capacity_basis, "fl": floor},
            )

    def set_fairness_owner(
        self,
        owner_id: str,
        *,
        weight: float = 1.0,
        hard_cap: int | None = None,
        uncapped: bool = False,
        blocked: bool = False,
    ) -> None:
        """Upsert one owner's override."""
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO fairness_owner "
                "(owner_id, weight, hard_cap, uncapped, blocked) "
                "VALUES (:o, :w, :h, :u, :b) "
                "ON CONFLICT(owner_id) DO UPDATE SET "
                "weight = :w, hard_cap = :h, uncapped = :u, blocked = :b",
                {
                    "o": owner_id, "w": weight, "h": hard_cap,
                    "u": 1 if uncapped else 0,
                    "b": 1 if blocked else 0,
                },
            )

    def clear_fairness_owner(self, owner_id: str) -> None:
        """Drop one owner's override (revert to default cap, not blocked)."""
        with self._tx() as cur:
            cur.execute(
                "DELETE FROM fairness_owner WHERE owner_id = ?", (owner_id,),
            )

    def running_counts_by_owner(self) -> dict[str, int]:
        """Cluster-wide count of *placed* sandboxes per owner.

        Counts gym/step rollouts in ``starting``/``running`` and raw (case-2/3)
        rollouts in ``running``. This is the per-owner usage the admission cap
        gate compares against ``FairnessPolicy.cap_for``.

        Raw ``acquiring`` is deliberately **excluded** (audit M3): a raw acquire
        writes its row as ``acquiring`` *before* it reaches the admission gate,
        so counting it would charge the candidate against its own cap — an
        otherwise-idle owner's first raw acquire would see ``running=1`` and
        park itself forever at ``cap=1``. Only post-placement sandboxes count;
        gym ``starting`` is post-placement (its record is created after the
        admission gate), so it stays. The brief raw ``acquiring`` window can let
        a burst slightly over-admit — acceptable versus a self-deadlock.
        """
        out: dict[str, int] = {}
        with self._lock:
            for row in self._conn.execute(
                "SELECT owner_id, COUNT(*) AS n FROM rollouts "
                "WHERE status IN ('starting', 'running') GROUP BY owner_id",
            ):
                key = str(row["owner_id"])
                out[key] = out.get(key, 0) + int(row["n"])
            for row in self._conn.execute(
                "SELECT owner_id, COUNT(*) AS n FROM raw_rollouts "
                "WHERE status = 'running' GROUP BY owner_id",
            ):
                key = str(row["owner_id"])
                out[key] = out.get(key, 0) + int(row["n"])
        return out

    # ── Internals ────────────────────────────────────────────────────────────

    @contextmanager
    def _tx(self) -> Any:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def _load_steps(self, rollout_id: str) -> list[Step]:
        rows = self._conn.execute(
            "SELECT * FROM rollout_steps WHERE rollout_id = ? ORDER BY step_index",
            (rollout_id,),
        ).fetchall()
        return [
            Step(
                index=int(r["step_index"]),
                action=json.loads(r["action_json"]),
                obs=json.loads(r["obs_json"]),
                reward=float(r["reward"]),
                done=bool(r["done"]),
                truncated=bool(r["truncated"]),
                info=json.loads(r["info_json"]),
                ts=float(r["ts"]),
            )
            for r in rows
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Row converters
# ──────────────────────────────────────────────────────────────────────────────


def _rollout_to_row(record: RolloutRecord) -> dict[str, Any]:
    return {
        "rollout_id": record.rollout_id,
        "template": record.template,
        "status": record.status.value,
        "reason": record.reason,
        "request_id": record.request_id,
        "task_key": record.task_key,
        "group_id": record.group_id,
        "owner_id": record.owner_id,
        "project_id": record.project_id,
        "run_id": record.run_id,
        "node_id": record.node_id,
        "sandbox_id": record.sandbox_id,
        "init_json": json.dumps(record.init_params),
        "metadata_json": json.dumps(record.metadata),
        "final_reward": record.final_reward,
        "trajectory_sink": record.trajectory_sink,
        "trajectory_node_id": record.trajectory_node_id,
        "trajectory_uri": record.trajectory_uri,
        "trajectory_size_bytes": record.trajectory_size_bytes,
        "created_at": record.created_at,
        "last_touched_at": record.last_touched_at,
    }


def _rollout_from_row(row: sqlite3.Row, steps: list[Step]) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=row["rollout_id"],
        template=row["template"],
        status=RolloutStatus(row["status"]),
        reason=row["reason"],
        request_id=row["request_id"],
        task_key=row["task_key"],
        group_id=row["group_id"],
        owner_id=row["owner_id"],
        project_id=row["project_id"],
        run_id=row["run_id"],
        node_id=row["node_id"],
        sandbox_id=row["sandbox_id"],
        init_params=json.loads(row["init_json"]),
        metadata=json.loads(row["metadata_json"]),
        steps=steps,
        final_reward=float(row["final_reward"]),
        trajectory_sink=row["trajectory_sink"],
        trajectory_node_id=row["trajectory_node_id"],
        trajectory_uri=row["trajectory_uri"],
        trajectory_size_bytes=row["trajectory_size_bytes"],
        created_at=float(row["created_at"]),
        last_touched_at=float(row["last_touched_at"]),
    )


def _sandbox_from_row(row: sqlite3.Row) -> SandboxRecord:
    # ``effective_resources_json`` was added in Slice 9b. Older rows
    # may have a NULL — which is what the field defaults to anyway.
    # ``sqlite3.Row`` supports ``.keys()``-based membership; defend
    # against in-memory non-Row mocks that might not.
    keys = row.keys() if hasattr(row, "keys") else []
    effective_resources_json = (
        row["effective_resources_json"]
        if "effective_resources_json" in keys
        else None
    )
    image = row["image"] if "image" in keys else None
    return SandboxRecord(
        sandbox_id=row["sandbox_id"],
        backend=row["backend"],
        backend_ref=row["backend_ref"],
        stub_endpoint=row["stub_endpoint"],
        template=row["template"],
        image=image,
        node_id=row["node_id"],
        rollout_id=row["rollout_id"],
        status=row["status"],
        owner_count=int(row["owner_count"]),
        created_at=float(row["created_at"]),
        effective_resources_json=effective_resources_json,
    )


def _pending_from_row(row: sqlite3.Row) -> PendingRolloutRecord:
    return PendingRolloutRecord(
        pending_id=row["pending_id"],
        template=row["template"],
        init_params=json.loads(row["init_json"]),
        request_id=row["request_id"],
        task_key=row["task_key"],
        group_id=row["group_id"],
        deadline_json=json.loads(row["deadline_json"]),
        queue_partition=row["queue_partition"],
        submitted_at=float(row["submitted_at"]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def new_id() -> str:
    return uuid.uuid4().hex
