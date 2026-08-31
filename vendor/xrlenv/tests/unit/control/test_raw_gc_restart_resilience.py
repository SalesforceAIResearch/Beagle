"""WS3 — control-plane restart resilience for raw sessions.

A CP restart wipes the coordinator's in-memory ``_sessions`` map, but
the durable ``raw_rollouts`` rows + node-side containers survive. Before
this fix the raw-GC reconciler treated the still-running containers as
node-only orphans and force-destroyed them, and the startup SQLite sweep
sealed the rows ``failed/lost-on-restart`` — killing live work on every
restart. These tests cover:

  * ``RawContainerCoordinator.readopt`` rebuilds a session from a row;
  * the reconcile sweep re-adopts restart survivors instead of
    force-destroying them;
  * the startup SQLite sweep protects re-adoptable ``running`` rows
    (those with a container_id) within the re-adoption grace, and seals
    them once the grace lapses (node never came back) and always seals
    ``acquiring`` ghosts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.raw_gc_reconciler import RawGCReconciler
from xrlenv.control.state import RawRolloutRecord


@dataclass
class _FakeTransport:
    node_id: str = "node-A"
    docker_container_ids: list[str] = field(default_factory=list)
    force_destroyed: list[str] = field(default_factory=list)

    def supported_backends(self) -> list[str]:
        return ["docker"]

    async def list_raw_container_ids(self, **_: Any) -> list[str]:
        return list(self.docker_container_ids)

    async def force_destroy_raw_container(self, *, container_id: str) -> None:
        self.force_destroyed.append(container_id)


@dataclass
class _FakeRegistry:
    transports: dict[str, _FakeTransport]

    @property
    def node_ids(self) -> list[str]:
        return list(self.transports.keys())

    def get(self, node_id: str) -> _FakeTransport | None:
        return self.transports.get(node_id)


@dataclass
class _FakeScheduler:
    nodes: list[Any] = field(default_factory=list)
    image_aware_placement: bool = True

    def place(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
        from xrlenv.errors import XRLEnvError
        raise XRLEnvError("placement not used in these tests")

    def commit_placement(self, *_: Any) -> None:  # pragma: no cover
        pass

    def release_placement(self, *_: Any) -> None:  # pragma: no cover
        pass


@dataclass
class _FakeState:
    rows: list[RawRolloutRecord] = field(default_factory=list)
    updates: list[dict[str, Any]] = field(default_factory=list)

    def list_raw_rollouts(
        self, *, status: str | None = None,
    ) -> list[RawRolloutRecord]:
        if status is None:
            return list(self.rows)
        return [r for r in self.rows if r.status == status]

    def get_raw_rollout(self, rollout_id: str) -> RawRolloutRecord | None:
        return next(
            (r for r in self.rows if r.rollout_id == rollout_id), None,
        )

    def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
        self.updates.append({"rollout_id": rollout_id, **fields})
        for i, row in enumerate(self.rows):
            if row.rollout_id == rollout_id:
                self.rows[i] = row.model_copy(update=fields)

    def record_raw_rollout(self, record: RawRolloutRecord) -> None:
        self.rows.append(record)


def _row(rollout_id: str, **kw: Any) -> RawRolloutRecord:
    base: dict[str, Any] = {
        "rollout_id": rollout_id,
        "status": "running",
        "image": "img:1",
        "node_id": "node-A",
        "container_id": f"c-{rollout_id}",
        "container_name": f"name-{rollout_id}",
        "created_at": time.time() - 100.0,
    }
    base.update(kw)
    return RawRolloutRecord(**base)


def _coord(nodes: list[Any] | None = None) -> RawContainerCoordinator:
    return RawContainerCoordinator(scheduler=_FakeScheduler(nodes=nodes or []))


# ──────────────────────────────────────────────────────────────────────────────
# coordinator.readopt
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_readopt_rebuilds_session_from_row() -> None:
    coord = _coord()
    transport = _FakeTransport(node_id="node-A")
    row = _row("r1", task_key="tk-1")

    assert await coord.readopt(row, transport) is True
    sessions = coord.list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s.rollout_id == "r1"
    assert s.node is transport
    assert s.node_id == "node-A"
    assert s.container_id == "c-r1"
    assert s.container_name == "name-r1"
    assert s.image == "img:1"
    assert s.task_key == "tk-1"
    # A deadline was reconstructed so an abandoned re-adopted session
    # can't leak past the default cap.
    assert s.deadline_at > 0.0


@pytest.mark.asyncio
async def test_readopt_restores_container_runtime_for_cap_accounting() -> None:
    # audit H11: a re-adopted sysbox-runc container must keep its runtime so the per-node
    # runtime concurrency cap counts it (else the scheduler over-places past the cap).
    coord = _coord()
    transport = _FakeTransport(node_id="node-A")
    row = _row("r1", container_runtime="sysbox-runc")

    assert await coord.readopt(row, transport) is True
    s = coord.list_sessions()[0]
    assert s.container_runtime == "sysbox-runc"
    # and it flows into the load entry the per-node runtime cap reads.
    entries = coord.iter_load_entries()
    assert len(entries) == 1
    assert entries[0].container_runtime == "sysbox-runc"


@pytest.mark.asyncio
async def test_readopt_is_idempotent_and_guards_missing_fields() -> None:
    coord = _coord()
    transport = _FakeTransport(node_id="node-A")
    row = _row("r1")

    assert await coord.readopt(row, transport) is True
    # Second call, SAME transport: already adopted → idempotent SUCCESS (True), no duplicate
    # insert. (Audit H11: readopt-on-connect treats False as "couldn't transfer ownership" and
    # fails closed, so an already-adopted survivor must report True — not False.)
    assert await coord.readopt(row, transport) is True
    assert len(coord.list_sessions()) == 1

    # Missing container_id / node_id → can't route → no-op (still False).
    assert await coord.readopt(_row("r2", container_id=None), transport) is False
    assert await coord.readopt(_row("r3", node_id=None), transport) is False
    assert {s.rollout_id for s in coord.list_sessions()} == {"r1"}


@pytest.mark.asyncio
async def test_readopt_reroutes_stale_generation_session_to_new_transport() -> None:
    # Audit H11 ownership transfer: a node-agent reconnect can leave the session routed through
    # the OLD transport (its _on_disconnected no-ops because the registry already points at the
    # replacement). A readopt against the NEW transport must RE-ROUTE the existing session, return
    # True (so readopt-on-connect admits rather than deadlocking), and not duplicate it.
    coord = _coord()
    old = _FakeTransport(node_id="node-A")
    new = _FakeTransport(node_id="node-A")
    row = _row("r1")

    assert await coord.readopt(row, old) is True
    assert coord.list_sessions()[0].node is old

    assert await coord.readopt(row, new) is True          # re-route, not a failure
    assert len(coord.list_sessions()) == 1                # not duplicated
    assert coord.list_sessions()[0].node is new           # now routes through the new transport


@pytest.mark.asyncio
async def test_readopt_stale_transport_does_not_steal_session_when_not_current() -> None:
    # audit H11: a DELAYED OLD readoption task (its stream superseded by a replacement) must NOT
    # transfer the session back to itself — that would let its failed-pass rollback DELETE the
    # session the replacement legitimately owns. The transfer is gated on is_current(): a stale
    # caller returns False and leaves the session on the current transport.
    coord = _coord()
    old = _FakeTransport(node_id="node-A")
    new = _FakeTransport(node_id="node-A")
    row = _row("r1")

    assert await coord.readopt(row, new) is True          # replacement adopts it
    assert coord.list_sessions()[0].node is new

    # the old pass resumes; it is NO LONGER current → must not steal.
    stole = await coord.readopt(row, old, is_current=lambda: False)
    assert stole is False                                  # transfer refused
    assert coord.list_sessions()[0].node is new           # still owned by the replacement
    # the current transport CAN still (idempotently) transfer.
    assert await coord.readopt(row, new, is_current=lambda: True) is True
    assert coord.list_sessions()[0].node is new


# ──────────────────────────────────────────────────────────────────────────────
# reconcile sweep re-adoption
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_readopts_survivor_instead_of_force_destroy() -> None:
    """A still-running container with a non-terminal row is re-adopted,
    NOT force-destroyed, and its row is left running."""
    transport = _FakeTransport(node_id="node-A", docker_container_ids=["c-r1"])
    coord = _coord(nodes=[transport])
    state = _FakeState(rows=[_row("r1")])
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
    )
    recon._started_at = time.time()  # restart grace active

    await recon.reconcile_once()

    assert [s.rollout_id for s in coord.list_sessions()] == ["r1"]
    assert transport.force_destroyed == []
    assert state.get_raw_rollout("r1").status == "running"


@pytest.mark.asyncio
async def test_sweep_force_destroys_true_orphan_without_row() -> None:
    """A container with NO non-terminal row is a genuine orphan — still
    force-destroyed (two-sweep confirmed), not re-adopted."""
    transport = _FakeTransport(node_id="node-A", docker_container_ids=["c-x"])
    coord = _coord(nodes=[transport])
    state = _FakeState(rows=[])  # no row for c-x
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
    )
    recon._started_at = time.time()

    await recon.reconcile_once()  # first sweep: observed node-only
    await recon.reconcile_once()  # second sweep: confirmed → destroy

    assert coord.list_sessions() == []
    assert transport.force_destroyed == ["c-x"]


# ──────────────────────────────────────────────────────────────────────────────
# startup SQLite sweep protection
# ──────────────────────────────────────────────────────────────────────────────


def test_startup_protects_readoptable_running_row_within_grace() -> None:
    """A running row with a container_id is NOT sealed at startup (within
    the re-adoption grace) even if the row is old — it's a re-adoption
    candidate, not a ghost."""
    coord = _coord()
    state = _FakeState(rows=[_row("r1", created_at=time.time() - 7200.0)])
    recon = RawGCReconciler(
        registry=_FakeRegistry({}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        running_stale_s=60.0,
        readopt_grace_s=300.0,
    )
    recon._started_at = time.time()  # just started → grace active

    flipped = recon._reconcile_sqlite(reason="lost-on-restart")

    assert flipped == 0
    assert state.get_raw_rollout("r1").status == "running"


def test_running_row_sealed_after_readopt_grace_lapses() -> None:
    """Once the grace has lapsed (node never reconnected), an
    un-re-adopted running-with-container row IS sealed."""
    coord = _coord()
    state = _FakeState(rows=[_row("r1", created_at=time.time() - 7200.0)])
    recon = RawGCReconciler(
        registry=_FakeRegistry({}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        running_stale_s=60.0,
        readopt_grace_s=300.0,
    )
    recon._started_at = time.time() - 10_000.0  # grace long lapsed

    flipped = recon._reconcile_sqlite(reason="lost-on-restart")

    assert flipped == 1
    assert state.get_raw_rollout("r1").status == "failed"


def test_startup_seals_acquiring_ghost_regardless_of_grace() -> None:
    """An ``acquiring`` row has no container to re-adopt — its acquire
    coroutine died with the process — so it's sealed at startup even
    within the grace."""
    coord = _coord()
    state = _FakeState(rows=[
        RawRolloutRecord(
            rollout_id="ra", status="acquiring", image="img:1",
            created_at=time.time() - 5.0,
        ),
    ])
    recon = RawGCReconciler(
        registry=_FakeRegistry({}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        readopt_grace_s=300.0,
    )
    recon._started_at = time.time()

    flipped = recon._reconcile_sqlite(reason="lost-on-restart")

    assert flipped == 1
    assert state.get_raw_rollout("ra").status == "failed"


# ──────────────────────────────────────────────────────────────────────────────
# Audit P2 — re-adoption restores persisted deadline + effective resources
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_readopt_restores_persisted_deadline_and_resources() -> None:
    """A re-adopted session restores the ORIGINAL reap deadline and
    effective ResourceSpec persisted at acquire — not the defaults — so a
    custom-deadline / custom-resource session survives a CP restart with
    correct semantics + scheduler load accounting."""
    from xrlenv.control.raw_container_service import _DEFAULT_RAW_RESOURCES

    coord = _coord()
    transport = _FakeTransport(node_id="node-A")
    custom = _DEFAULT_RAW_RESOURCES.model_copy(
        update={"cpu_limit": 8.0, "mem_limit_bytes": 32 * 1024**3},
    )
    row = _row(
        "r1",
        deadline_at=1_000_000.0,
        effective_resources_json=custom.model_dump_json(),
    )

    assert await coord.readopt(row, transport) is True
    s = coord.list_sessions()[0]
    assert s.deadline_at == 1_000_000.0
    assert s.effective_resources.cpu_limit == 8.0
    assert s.effective_resources.mem_limit_bytes == 32 * 1024**3


@pytest.mark.asyncio
async def test_readopt_falls_back_when_deadline_and_resources_absent() -> None:
    """Pre-P2 rows (deadline_at / effective_resources_json None) fall back
    to the default deadline + raw footprint — no regression for rows
    written before the field existed."""
    from xrlenv.control.raw_container_service import _DEFAULT_RAW_RESOURCES

    coord = _coord()
    transport = _FakeTransport(node_id="node-A")
    row = _row("r1")  # no deadline_at / effective_resources_json set

    assert await coord.readopt(row, transport) is True
    s = coord.list_sessions()[0]
    assert s.deadline_at > 0.0  # reconstructed from created_at + default
    assert s.effective_resources == _DEFAULT_RAW_RESOURCES


# ──────────────────────────────────────────────────────────────────────────────
# Audit P3 — disk-guard kill reason surfaced through the reconciler
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_coordinator_only_orphan_sealed_with_node_reap_reason() -> None:
    """When the node reports a disk-guard reap reason for a rollout, the
    reconciler seals the coordinator-only orphan via ``seal_orphan(reason=…)``
    so the row carries the real disk-pressure cause, not a generic
    teardown message.

    ``seal_orphan`` (NOT ``destroy``): the container is already gone on
    the node, so a wire-level destroy would only race — and once the node
    dropped its record it fails with a benign "not registered" error that
    ``destroy`` would wrongly seal ``failed``, burying the reap cause (the
    live-smoke regression this path guards)."""
    seal_calls: list[dict[str, Any]] = []

    class _RecordingCoord:
        # Only ``seal_orphan`` is exercised by a direct
        # _handle_coordinator_only call that succeeds (the
        # _lock/_sessions fallback path is for the XRLEnvError case, not
        # hit here). A ``destroy`` here would fail the test — the
        # reconciler must NOT wire-destroy a known-gone orphan.
        async def seal_orphan(
            self, *, rollout_id: str, container_id: str,
            reason: str | None = None,
        ) -> None:
            seal_calls.append({"rollout_id": rollout_id, "reason": reason})

    transport = _FakeTransport(node_id="node-A")
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=_RecordingCoord(),  # type: ignore[arg-type]
        state=None,
    )

    disk_reason = "disk-guard: reaped runaway raw container (writable ...)"
    await recon._handle_coordinator_only(
        "node-A", "c-1", "roll-A", reason=disk_reason,
    )
    assert seal_calls == [{"rollout_id": "roll-A", "reason": disk_reason}]

    # No reason (container vanished for some other cause) → clean teardown.
    await recon._handle_coordinator_only("node-A", "c-2", "roll-B")
    assert seal_calls[-1] == {"rollout_id": "roll-B", "reason": None}
