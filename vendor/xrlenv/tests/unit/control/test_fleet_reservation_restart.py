"""Fleet reservation — restart-safety (step 5 / R7).

The footprint is CP-only (spec 21 keeps the node fleet-unaware), so a
control-plane restart can't re-derive it from live containers. The design:
persist a tiny footprint row per open fleet; on restart, re-adopt the live
containers (which restores each session's ``fleet_id`` from its persisted
per-container label) and rebuild ``_fleets`` from the footprint rows + those
node-confirmed-alive members. A memberless row past its TTL is reclaimed.

These tests drive the coordinator seam directly (``readopt`` +
``rebuild_fleets_from_state``) — the raw-GC reconciler just calls the latter
once per sweep after re-adoption.
"""

from __future__ import annotations

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.compat.metadata import (
    LABEL_FLEET_CPU_REQUEST,
    LABEL_FLEET_ID,
    LABEL_FLEET_MEM_REQUEST,
)
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.scheduler import Scheduler
from xrlenv.control.state import FleetReservationRecord, InMemoryStateStore
from xrlenv.control.template_catalog import TemplateCatalog

from tests.unit.control.test_raw_container_coordinator import _FakeNodeTransport

_GIB = 1024**3


def _opener_labels(fleet_id: str, cpu: float, mem_gb: int) -> dict[str, str]:
    return {
        LABEL_FLEET_ID: fleet_id,
        LABEL_FLEET_CPU_REQUEST: str(cpu),
        LABEL_FLEET_MEM_REQUEST: str(mem_gb * _GIB),
    }


def _companion_labels(fleet_id: str) -> dict[str, str]:
    return {LABEL_FLEET_ID: fleet_id}


def _coord(state: InMemoryStateStore, node: _FakeNodeTransport) -> RawContainerCoordinator:
    scheduler = Scheduler([node], catalog=TemplateCatalog(), state=state)
    coord = RawContainerCoordinator(scheduler=scheduler, state=state)
    scheduler.set_raw_session_provider(coord.iter_load_entries)
    return coord


def _node_cpu_load(coord: RawContainerCoordinator) -> float:
    return sum(
        e.effective_resources.cpu_request for e in coord.iter_load_entries()
    )


def _footprint(cpu: float, mem_gb: int) -> ResourceSpec:
    return ResourceSpec(
        cpu_request=cpu, cpu_limit=cpu, mem_request_bytes=mem_gb * _GIB,
        mem_limit_bytes=mem_gb * _GIB, disk_request_bytes=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Restart rebuild
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restart_rebuilds_fleet_from_rows_and_readopted_members() -> None:
    """The full restart path. A fleet (lead + companion) is open; the CP
    restarts (fresh coordinator, SAME StateStore, node reconnects). Re-adopting
    the live containers restores each session's fleet_id; the rebuild pass then
    reconstructs the reservation — footprint from the persisted row, members
    from the node-confirmed-alive sessions — and load accounting is back to
    footprint-once."""
    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")

    # ── Original control plane ──
    coord1 = _coord(state, node)
    lead = await coord1.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    node.next_container_id = "companion"
    comp = await coord1.acquire(
        image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
    )
    # Persisted: one footprint row + two raw rows carrying fleet_id.
    assert len(state.list_fleet_reservations()) == 1
    raw = {r.rollout_id: r for r in state.list_raw_rollouts()}
    assert raw[lead.rollout_id].fleet_id == "f1"
    assert raw[comp.rollout_id].fleet_id == "f1"

    # ── Control plane restarts ── fresh coordinator, same durable state,
    # same node (reconnected). In-memory fleet + session state is empty.
    coord2 = _coord(state, node)
    assert coord2._fleets == {}
    assert coord2._sessions == {}

    # Re-adopt the surviving containers (what the raw-GC reconciler does per
    # node). fleet_id is restored onto each rebuilt session.
    for row in state.list_raw_rollouts(status="running"):
        assert await coord2.readopt(row, node) is True
    assert coord2._sessions[lead.rollout_id].fleet_id == "f1"
    assert coord2._sessions[comp.rollout_id].fleet_id == "f1"
    # Before the rebuild, the fleet reservation doesn't exist yet, so the
    # members charge their full own resources (conservative over-count, never
    # a gap): 2 + 16 == 18 here.
    assert _node_cpu_load(coord2) == 18.0

    # Rebuild → reservation reconstructed from the footprint row + live members.
    rebuilt, reclaimed = await coord2.rebuild_fleets_from_state()
    assert (rebuilt, reclaimed) == (1, 0)
    res = coord2._fleets["f1"]
    assert res.node_id == "node-A"
    assert res.footprint.cpu_request == 18.0
    assert res.footprint.mem_request_bytes == 32 * _GIB
    assert set(res.members) == {lead.rollout_id, comp.rollout_id}
    # Load accounting is footprint-once again (18), members suppressed.
    assert _node_cpu_load(coord2) == 18.0


@pytest.mark.asyncio
async def test_rebuild_is_idempotent() -> None:
    """Calling the rebuild pass repeatedly (every sweep) doesn't duplicate or
    disturb an already-live reservation."""
    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node)
    await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    # The fleet is already live in-memory; rebuild must be a no-op.
    rebuilt, reclaimed = await coord.rebuild_fleets_from_state()
    assert (rebuilt, reclaimed) == (0, 0)
    assert set(coord._fleets) == {"f1"}
    assert _node_cpu_load(coord) == 18.0


@pytest.mark.asyncio
async def test_rebuild_skips_row_whose_node_has_not_reconnected() -> None:
    """A persisted row whose members haven't re-adopted yet (node still
    reconnecting) is neither rebuilt nor — with reclaim deferred — deleted; it
    waits for a later sweep."""
    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node)
    # A persisted footprint row with no live sessions (node not back yet).
    state.record_fleet_reservation(FleetReservationRecord(
        fleet_id="f1", node_id="node-B", footprint_json=_footprint(18, 32).model_dump_json(),
        opened_ts=500.0, last_acquire_ts=990.0,
    ))
    rebuilt, reclaimed = await coord.rebuild_fleets_from_state(
        now=1000.0, reclaim_after_s=600.0, allow_reclaim=False,
    )
    assert (rebuilt, reclaimed) == (0, 0)
    assert coord._fleets == {}
    assert len(state.list_fleet_reservations()) == 1  # kept, waiting


# ──────────────────────────────────────────────────────────────────────────────
# TTL reclaim
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reclaim_deletes_memberless_row_past_ttl() -> None:
    """Past the re-adoption grace, a memberless row older than the TTL is
    reclaimed (its node never came back, or its consumer crashed)."""
    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node)
    state.record_fleet_reservation(FleetReservationRecord(
        fleet_id="ghost", node_id="node-A",
        footprint_json=_footprint(18, 32).model_dump_json(),
        opened_ts=0.0, last_acquire_ts=100.0,
    ))
    rebuilt, reclaimed = await coord.rebuild_fleets_from_state(
        now=1000.0, reclaim_after_s=600.0, allow_reclaim=True,
    )
    assert (rebuilt, reclaimed) == (0, 1)
    assert state.list_fleet_reservations() == []
    assert coord._fleets == {}


@pytest.mark.asyncio
async def test_reclaim_deferred_within_ttl() -> None:
    """A memberless row still within its TTL is NOT reclaimed even when reclaim
    is allowed — its members may still be re-adopting."""
    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node)
    state.record_fleet_reservation(FleetReservationRecord(
        fleet_id="fresh", node_id="node-A",
        footprint_json=_footprint(18, 32).model_dump_json(),
        opened_ts=900.0, last_acquire_ts=990.0,  # 10s ago at now=1000
    ))
    rebuilt, reclaimed = await coord.rebuild_fleets_from_state(
        now=1000.0, reclaim_after_s=600.0, allow_reclaim=True,
    )
    assert (rebuilt, reclaimed) == (0, 0)
    assert len(state.list_fleet_reservations()) == 1


@pytest.mark.asyncio
async def test_rebuild_fleets_raise_on_error_fails_closed() -> None:
    # audit H11: rebuild swallows a state-list error and returns (0,0) by default (periodic sweep
    # best-effort), but the readopt-on-connect path passes raise_on_error=True so an unreadable
    # fleet table FAILS the pass instead of silently admitting with fleets unaccounted.
    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node)

    def _boom() -> list:
        raise RuntimeError("fleet table unreadable")
    state.list_fleet_reservations = _boom  # type: ignore[method-assign]

    assert await coord.rebuild_fleets_from_state() == (0, 0)          # default: best-effort
    with pytest.raises(RuntimeError, match="unreadable"):             # connect path: fail closed
        await coord.rebuild_fleets_from_state(raise_on_error=True)


@pytest.mark.asyncio
async def test_rebuild_fleets_bad_footprint_json_fails_closed_on_connect() -> None:
    # audit H11: a LIVE fleet (the node reports member containers) whose declared PEAK footprint
    # is unreadable is a corruption signal. The periodic sweep SKIPS it (best-effort — members
    # already charge full standalone footprints), but readopt-on-connect (raise_on_error=True)
    # must FAIL CLOSED — skipping would leave the peak un-reserved and let a later companion
    # over-place into capacity the fleet still owns.
    import datetime as _dt

    from xrlenv.control.raw_container_service import RawContainerSession

    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node)
    state.record_fleet_reservation(FleetReservationRecord(
        fleet_id="fbad", node_id="node-A", footprint_json="{not valid json",
        task_key="task-X", owner="o",
    ))
    coord._sessions["m1"] = RawContainerSession(  # a live member on the row's node
        rollout_id="m1", node=node, node_id="node-A", container_id="c-m1",
        container_name="n", image="img", created_at=_dt.datetime.now(_dt.UTC),
        fleet_id="fbad", effective_resources=_footprint(5, 8),
    )

    assert await coord.rebuild_fleets_from_state() == (0, 0)   # best-effort: skipped, not raised
    assert "fbad" not in coord._fleets
    # connect path: fail closed. Malformed footprint_json surfaces pydantic's ValidationError
    # (a ValueError subclass) out of ResourceSpec.model_validate_json.
    with pytest.raises(ValueError):
        await coord.rebuild_fleets_from_state(raise_on_error=True)


@pytest.mark.asyncio
async def test_load_charges_full_for_non_member_fleet_session() -> None:
    # audit H11: a session carrying a fleet_id whose rollout is NOT in the live reservation's
    # validated membership must charge its FULL footprint (not disk-only) — else its cpu/mem is
    # uncounted and enables over-placement.
    import datetime as _dt

    from xrlenv.control.raw_container_service import RawContainerSession

    state = InMemoryStateStore()
    node = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node)
    await coord.acquire(  # opener → live reservation f1 (18 cpu footprint), opener suppressed
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    # a mislabeled session: carries fleet_id=f1 but is NOT in the reservation's membership.
    coord._sessions["bogus"] = RawContainerSession(
        rollout_id="bogus", node=node, node_id="node-A", container_id="c-bogus",
        container_name="n", image="img", created_at=_dt.datetime.now(_dt.UTC),
        fleet_id="f1", effective_resources=_footprint(5, 8),
    )
    assert "bogus" not in coord._fleets["f1"].members
    # 18 (fleet footprint) + 5 (bogus charged FULL) == 23; a suppressed bogus would read 18.
    assert _node_cpu_load(coord) == 23.0


@pytest.mark.asyncio
async def test_rebuild_ignores_fleet_member_on_wrong_node() -> None:
    # audit H11: a fleet row pinned to node-A must not fold in a member session living on a
    # DIFFERENT node (corrupt inventory) — that would suppress the member's load on the wrong
    # node while charging node-A's footprint, enabling over-placement.
    import datetime as _dt

    from xrlenv.control.raw_container_service import RawContainerSession

    state = InMemoryStateStore()
    node_a = _FakeNodeTransport(node_id="node-A")
    coord = _coord(state, node_a)
    state.record_fleet_reservation(FleetReservationRecord(
        fleet_id="f1", node_id="node-A",
        footprint_json=_footprint(18, 32).model_dump_json(),
        opened_ts=900.0, last_acquire_ts=990.0,
    ))
    # a live member with fleet_id=f1 but on the WRONG node (node-B).
    coord._sessions["m"] = RawContainerSession(
        rollout_id="m", node=node_a, node_id="node-B",
        container_id="c-m", container_name="n-m", image="img",
        created_at=_dt.datetime.now(_dt.UTC),
        fleet_id="f1", effective_resources=_footprint(2, 4),
    )
    # within TTL so it's not reclaimed; the wrong-node member is simply not folded in.
    rebuilt, _ = await coord.rebuild_fleets_from_state(
        now=1000.0, reclaim_after_s=600.0, allow_reclaim=False,
    )
    assert rebuilt == 0                # not reconstructed from a wrong-node member
    assert "f1" not in coord._fleets
