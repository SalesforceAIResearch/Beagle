"""WS3 edge cases — additional coverage for readopt and GC restart resilience.

Supplements test_raw_gc_restart_resilience.py with:

 * ``readopt`` returns False when the rollout_id is in ``_acquiring_ids``
   (in-flight acquire guard — can't re-adopt while still acquiring).
 * ``readopt`` with ``container_name=None`` (rare but valid on older rows)
   uses ``""`` not raises.
 * Startup sweep seals old ``running`` row with container_id AFTER the
   re-adoption grace has lapsed (regression guard for the main WS3 path).
 * Startup sweep does NOT seal an ``acquiring`` row that's in
   ``list_acquiring_ids`` (the in-flight set protects genuinely parked
   consumers, not just the time window).
 * Reconcile with a terminal row (status="released") sees the container
   as node-only but routes it through ``_handle_node_only`` as a routine
   deferred teardown — and force-destroys it after two sweeps.
 * A ``running`` row without a ``container_id`` is a genuine ghost and IS
   sealed (it can never be re-adopted — no container to point to).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.raw_gc_reconciler import RawGCReconciler
from xrlenv.control.state import RawRolloutRecord

# ── shared fakes (mirroring test_raw_gc_restart_resilience.py) ───────────────


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
        "created_at": time.time() - 200.0,
    }
    base.update(kw)
    return RawRolloutRecord(**base)


def _coord(nodes: list[Any] | None = None) -> RawContainerCoordinator:
    return RawContainerCoordinator(scheduler=_FakeScheduler(nodes=nodes or []))


# ── readopt guards ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_readopt_returns_false_when_rollout_in_acquiring_ids() -> None:
    """readopt must return False if the rollout_id is currently in
    ``_acquiring_ids`` — the acquire is still in flight, so the session
    does not need rebuilding and adding a second one would corrupt state."""
    coord = _coord()
    transport = _FakeTransport(node_id="node-A")
    row = _row("r-acq")

    # Simulate an in-flight acquire by adding to the private set directly.
    coord._acquiring_ids.add("r-acq")

    result = await coord.readopt(row, transport)  # type: ignore[arg-type]

    assert result is False
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_readopt_container_name_none_uses_empty_string() -> None:
    """A row with container_name=None (valid on old rows before that field
    was added) should readopt successfully, using "" for container_name."""
    coord = _coord()
    transport = _FakeTransport(node_id="node-A")
    row = _row("r-noname", container_name=None)

    result = await coord.readopt(row, transport)  # type: ignore[arg-type]

    assert result is True
    sessions = coord.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].container_name == ""


# ── startup SQLite sweep: in-flight acquiring row is NOT sealed ───────────────


def test_startup_does_not_seal_acquiring_row_in_acquiring_ids() -> None:
    """An acquiring row whose rollout_id is still in ``list_acquiring_ids``
    (genuinely in-flight, not a ghost) must not be sealed even within the
    startup sweep — the set-based check takes priority over any age window."""
    coord = _coord()
    row = RawRolloutRecord(
        rollout_id="ra-live",
        status="acquiring",
        image="img:1",
        created_at=time.time() - 5.0,  # young — would be sealed as ghost otherwise
    )
    state = _FakeState(rows=[row])
    # Register the rollout as actively acquiring.
    coord._acquiring_ids.add("ra-live")

    recon = RawGCReconciler(
        registry=_FakeRegistry({}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        readopt_grace_s=300.0,
    )
    recon._started_at = time.time()

    flipped = recon._reconcile_sqlite(reason="lost-on-restart")

    assert flipped == 0
    found = state.get_raw_rollout("ra-live")
    assert found is not None
    assert found.status == "acquiring"


# ── startup SQLite sweep: running row WITHOUT container_id is a ghost ─────────


def test_startup_seals_running_row_without_container_id_within_grace() -> None:
    """A 'running' row with no container_id cannot be re-adopted (there is
    no container to point to) — it's a genuine ghost and should be sealed
    even within the re-adoption grace period."""
    coord = _coord()
    # Old row with status=running but no container_id.
    row = RawRolloutRecord(
        rollout_id="r-no-container",
        status="running",
        image="img:1",
        node_id="node-A",
        container_id=None,       # the critical absence
        container_name=None,
        created_at=time.time() - 300.0,
    )
    state = _FakeState(rows=[row])
    recon = RawGCReconciler(
        registry=_FakeRegistry({}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        running_stale_s=60.0,
        readopt_grace_s=3600.0,  # very long grace — but should not protect this row
    )
    recon._started_at = time.time()  # grace is still active

    flipped = recon._reconcile_sqlite(reason="lost-on-restart")

    # Should be sealed: a running row without a container_id has nothing to re-adopt.
    assert flipped == 1
    found = state.get_raw_rollout("r-no-container")
    assert found is not None
    assert found.status == "failed"


# ── reconciler: terminal row node-only is force-destroyed after two sweeps ───


@pytest.mark.asyncio
async def test_terminal_row_node_only_force_destroyed_after_two_sweeps() -> None:
    """A container with a TERMINAL (released) row is node-only — the
    coordinator has no session for it. The reconciler must force-destroy it
    after two sweeps (two-sweep confirmation for node-only orphans).
    Re-adoption must NOT be attempted (row is terminal, so the re-adopt
    branch skips it)."""
    terminal_row = _row("r-done", status="released")
    transport = _FakeTransport(
        node_id="node-A",
        docker_container_ids=["c-r-done"],
    )
    coord = _coord(nodes=[transport])
    state = _FakeState(rows=[terminal_row])
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
    )
    recon._started_at = time.time()

    # First sweep: observes node-only; two-sweep gate holds; no destroy.
    await recon.reconcile_once()
    assert transport.force_destroyed == []

    # Second sweep: confirmed node-only → force-destroy.
    await recon.reconcile_once()
    assert transport.force_destroyed == ["c-r-done"]
    # Row status is NOT changed (terminal rows are not re-sealed).
    found = state.get_raw_rollout("r-done")
    assert found is not None
    assert found.status == "released"


# ── readopt: deadline reconstruction prevents abandoned-session leak ──────────


@pytest.mark.asyncio
async def test_readopt_deadline_is_reconstructed_from_row_created_at() -> None:
    """A re-adopted session's deadline_at is set to
    created_at + session_deadline_default_s. An abandoned re-adopted
    session (consumer never called destroy after CP restart) will be
    reaped by the deadline sweep once that time passes."""
    DEADLINE_S = 100.0  # short for the test
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(),
        session_deadline_default_s=DEADLINE_S,
    )
    transport = _FakeTransport(node_id="node-A")
    created_at = time.time() - 50.0  # 50s ago
    row = _row("r-deadline", created_at=created_at)

    await coord.readopt(row, transport)  # type: ignore[arg-type]

    sessions = coord.list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    # deadline_at should be approximately created_at + DEADLINE_S
    expected_deadline = created_at + DEADLINE_S
    assert abs(s.deadline_at - expected_deadline) < 1.0, (
        f"expected deadline ~{expected_deadline:.1f}, got {s.deadline_at:.1f}"
    )
    # Since created 50s ago with a 100s window, deadline is ~50s from now
    # (not already expired).
    assert s.deadline_at > time.time()


# ── sweep: re-adopt is not re-attempted after first success ───────────────────


@pytest.mark.asyncio
async def test_readopt_not_called_twice_for_same_container() -> None:
    """Once a container is re-adopted (session registered), a subsequent
    sweep for the same node should find the session in coord_set and NOT
    classify the container as node-only at all — no second readopt call."""
    transport = _FakeTransport(node_id="node-A", docker_container_ids=["c-r1"])
    coord = _coord(nodes=[transport])
    state = _FakeState(rows=[_row("r1")])
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
    )
    recon._started_at = time.time()

    # First sweep: re-adopts r1.
    await recon.reconcile_once()
    assert [s.rollout_id for s in coord.list_sessions()] == ["r1"]
    force_destroyed_after_first = list(transport.force_destroyed)

    # Second sweep: c-r1 is now in coord_set (session exists) → not node-only.
    await recon.reconcile_once()
    # No new force-destroys on the second sweep.
    assert transport.force_destroyed == force_destroyed_after_first
    # Session is still there (not cleared by the second sweep).
    assert [s.rollout_id for s in coord.list_sessions()] == ["r1"]
