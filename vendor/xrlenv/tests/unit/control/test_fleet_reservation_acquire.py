"""Fleet reservation — end-to-end acquire flow (phase 1, opt-in).

Exercises ``RawContainerCoordinator.acquire()``'s open-vs-companion branch
against a **real** ``Scheduler`` wired to the coordinator's load provider, so
the footprint reservation is checked as real capacity — not just bookkeeping.

Covers the §6 acceptance list + the audit's Step-3 findings:
- **F3** opener admits against the footprint; companion skips placement and
  lands on the pinned node; the no-fleet path is untouched.
- **F1** opener / companion failure cleanup (no leaked reservation or slot).
- **F2** concurrent companions serialize against one fleet row (can't both
  become members); duplicate opener rejected.
- Release: last-member destroy frees the reservation; node-loss frees it.
- **Graceful overflow**: a companion beyond the declared footprint is NOT
  hard-failed — it degrades to normal capacity-gated placement (charged its own
  resources, not a fleet member), so a mis-declared footprint never breaks a
  task.

All fleets are **synthetic + generic** (arbitrary cpu/mem numbers) — no
EvoClaw fixtures, roles, or sizes.
"""

from __future__ import annotations

import asyncio

import pytest
from xrlenv.compat.metadata import (
    LABEL_FLEET_CPU_REQUEST,
    LABEL_FLEET_ID,
    LABEL_FLEET_MEM_REQUEST,
)
from xrlenv.control.raw_container_service import (
    RawContainerCoordinator,
    RawContainerSession,
)
from xrlenv.control.scheduler import Scheduler
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import TemplateCatalog
from xrlenv.errors import CapacityExhausted, XRLEnvError

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


def _make(*, vcpus: int = 64) -> tuple[RawContainerCoordinator, Scheduler, _FakeNodeTransport]:
    node = _FakeNodeTransport(node_id="node-A")
    if vcpus != 64:
        from xrlenv.node.hw_probe import HardwareInfo
        node.hardware = lambda: HardwareInfo(  # type: ignore[method-assign]
            vcpus=vcpus, mem_bytes=128 * _GIB, disk_bytes=2000 * _GIB,
            has_kvm=True, has_gpu=False, gpu_model=None,
            kernel_version="6.0.0", platform="linux",
        )
    state = InMemoryStateStore()
    scheduler = Scheduler([node], catalog=TemplateCatalog(), state=state)
    coord = RawContainerCoordinator(scheduler=scheduler, state=state)
    scheduler.set_raw_session_provider(coord.iter_load_entries)
    return coord, scheduler, node


def _node_cpu_load(coord: RawContainerCoordinator) -> float:
    return sum(
        e.effective_resources.cpu_request for e in coord.iter_load_entries()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Opener
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_opener_reserves_footprint_and_tags_session() -> None:
    """A fleet-opening acquire reserves the whole footprint (cpu 18) on one
    node — not the lead's own 2 — creates the reservation, tags the session,
    and leaves no dangling pending."""
    coord, scheduler, _node = _make()

    lead = await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )

    assert lead.fleet_id == "f1"
    assert "f1" in coord._fleets
    res = coord._fleets["f1"]
    assert res.node_id == "node-A"
    assert res.footprint.cpu_request == 18.0
    assert set(res.members) == {lead.rollout_id}
    # Load reflects the FOOTPRINT once (18), not the lead's own 2.
    assert _node_cpu_load(coord) == 18.0
    # Handoff complete: pending dropped, opening marker cleared.
    assert len(scheduler._pending) == 0
    assert coord._fleet_opening == set()


# ──────────────────────────────────────────────────────────────────────────────
# Companion — the core anti-starvation scenario
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_companion_draws_from_reservation_even_when_free_pool_too_small() -> None:
    """The core scenario. On a 20-vCPU node, a fleet reserves footprint 18.
    A *non-fleet* 16-vCPU acquire is refused (only ~6 free). But the fleet's
    own 16-vCPU **companion** succeeds — it draws from the reservation and
    skips placement entirely, landing on the pinned node."""
    coord, scheduler, node = _make(vcpus=24)

    lead = await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )

    # A non-fleet 16-CPU acquire cannot fit — the fleet's footprint occupies
    # the node.
    node.next_container_id = "nonfleet"
    with pytest.raises(CapacityExhausted):
        await coord.acquire(image="busybox:1", cpu_limit=16)

    # The fleet's own companion (also 16 CPU) DOES fit — drawn from reservation.
    node.next_container_id = "companion"
    companion = await coord.acquire(
        image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
    )
    assert companion.fleet_id == "f1"
    assert companion.node_id == lead.node_id  # pinned to the reserved node
    assert set(coord._fleets["f1"].members) == {lead.rollout_id, companion.rollout_id}
    # No double-count: still the footprint 18, not 18 + 16.
    assert _node_cpu_load(coord) == 18.0
    # Companion skipped placement → it never minted a pending reservation.
    assert len(scheduler._pending) == 0


@pytest.mark.asyncio
async def test_companion_over_budget_falls_back_to_normal_placement() -> None:
    """A companion that would exceed the footprint is NOT hard-failed — it
    degrades gracefully: admitted via normal capacity-gated placement, charged
    its OWN resources (not drawn from the reservation, not a member). The
    reservation and its members are untouched; the overflow container carries
    no fleet_id in accounting so it's charged in full."""
    coord, _scheduler, node = _make()
    lead = await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    node.next_container_id = "c1"
    c1 = await coord.acquire(
        image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
    )  # 2 + 16 = 18, exactly at footprint

    node.next_container_id = "c2"
    # 18 + 1 = 19 > 18 → overflow → falls back to normal placement (succeeds).
    overflow = await coord.acquire(
        image="busybox:1", cpu_limit=1, labels=_companion_labels("f1"),
    )
    # It succeeded (a real session) and is NOT a fleet member (fleet_id cleared).
    assert overflow.fleet_id is None
    assert overflow.rollout_id not in coord._fleets["f1"].members
    # Reservation unchanged: still exactly the lead + the in-budget companion.
    assert set(coord._fleets["f1"].members) == {lead.rollout_id, c1.rollout_id}
    # Accounting: footprint 18 (fleet) + the overflow's OWN 1 cpu (charged in
    # full, not suppressed) == 19.
    assert _node_cpu_load(coord) == 19.0


# ──────────────────────────────────────────────────────────────────────────────
# Ordering / duplicate guards (F2)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_companion_before_any_reservation_raises() -> None:
    """A companion (fleet_id, no footprint labels) for a fleet that was never
    opened fails loud — the opener must complete first."""
    coord, _scheduler, _node = _make()
    with pytest.raises(XRLEnvError, match="must complete first"):
        await coord.acquire(image="busybox:1", labels=_companion_labels("ghost"))


@pytest.mark.asyncio
async def test_duplicate_opener_rejected() -> None:
    """Two concurrent openers for one fleet_id: the second is rejected while
    the first is still opening (guards the double-reservation window). Tested
    on the classification seam directly."""
    coord, _scheduler, _node = _make()
    from xrlenv.backends.base import ResourceSpec
    fp = ResourceSpec(
        cpu_request=18, cpu_limit=18, mem_request_bytes=32 * _GIB,
        mem_limit_bytes=32 * _GIB, disk_request_bytes=0,
    )
    role1, _ = await coord._classify_fleet_acquire(
        rollout_id="r1", fleet_id="f1", fleet_footprint=fp, effective_resources=fp,
    )
    assert role1 == "opener"
    assert "f1" in coord._fleet_opening
    with pytest.raises(XRLEnvError, match="second fleet-opening"):
        await coord._classify_fleet_acquire(
            rollout_id="r2", fleet_id="f1", fleet_footprint=fp,
            effective_resources=fp,
        )


@pytest.mark.asyncio
async def test_concurrent_companions_serialize_one_member_one_overflow() -> None:
    """F2: two companions launched concurrently, each 16 CPU, against a fleet
    with 16 CPU of headroom (footprint 18, lead 2). The lock serializes the
    budget check so they can't BOTH become members: exactly one joins the
    reservation (fleet_id set), the other overflows (fleet_id cleared, normal
    placement). BOTH succeed (graceful fallback — no hard failure)."""
    coord, _scheduler, node = _make()
    await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )

    async def _companion(cid: str) -> RawContainerSession:
        node.next_container_id = cid
        return await coord.acquire(
            image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
        )

    results = await asyncio.gather(_companion("a"), _companion("b"))
    members = [r for r in results if r.fleet_id == "f1"]
    overflows = [r for r in results if r.fleet_id is None]
    assert len(members) == 1  # exactly one joined the reservation
    assert len(overflows) == 1  # the other gracefully overflowed
    # Reservation holds lead + the one in-budget companion (not the overflow).
    assert len(coord._fleets["f1"].members) == 2
    # Load: footprint 18 + the overflow's own 16 (charged in full) == 34.
    assert _node_cpu_load(coord) == 34.0


# ──────────────────────────────────────────────────────────────────────────────
# Failure cleanup (F1)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_opener_failure_leaves_no_reservation_or_pending() -> None:
    """If the lead container's acquire fails, the opener leaks nothing — no
    ``FleetReservation``, no ``_fleet_opening`` marker, no ``_pending``
    footprint (release_placement dropped it)."""
    coord, scheduler, node = _make()
    node.raise_on_acquire = RuntimeError("simulated node failure")

    with pytest.raises(RuntimeError, match="simulated node failure"):
        await coord.acquire(
            image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
        )

    assert coord._fleets == {}
    assert coord._fleet_opening == set()
    assert len(scheduler._pending) == 0
    assert _node_cpu_load(coord) == 0.0


@pytest.mark.asyncio
async def test_companion_failure_rolls_back_slot() -> None:
    """If a companion's acquire fails, its reserved budget slot is rolled
    back — the reservation survives with just the surviving members, so a
    later companion still sees the right headroom."""
    coord, _scheduler, node = _make()
    lead = await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )

    node.raise_on_acquire = RuntimeError("companion node failure")
    node.next_container_id = "c1"
    with pytest.raises(RuntimeError, match="companion node failure"):
        await coord.acquire(
            image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
        )

    # Reservation intact, slot rolled back → only the lead remains.
    assert set(coord._fleets["f1"].members) == {lead.rollout_id}
    assert _node_cpu_load(coord) == 18.0
    # A fresh companion now fits again (slot really was freed).
    node.raise_on_acquire = None
    node.next_container_id = "c2"
    c2 = await coord.acquire(
        image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
    )
    assert set(coord._fleets["f1"].members) == {lead.rollout_id, c2.rollout_id}


# ──────────────────────────────────────────────────────────────────────────────
# Release
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_last_member_destroy_frees_reservation() -> None:
    """The reservation releases only when its LAST member is destroyed — not
    the first. Capacity returns to the free pool in one step."""
    coord, _scheduler, node = _make()
    lead = await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    node.next_container_id = "companion"
    companion = await coord.acquire(
        image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
    )

    # Destroy the companion → reservation persists (the lead still holds it).
    await coord.destroy(rollout_id=companion.rollout_id, container_id=companion.container_id)
    assert "f1" in coord._fleets
    assert set(coord._fleets["f1"].members) == {lead.rollout_id}
    assert _node_cpu_load(coord) == 18.0

    # Destroy the lead → last member gone → reservation released.
    await coord.destroy(rollout_id=lead.rollout_id, container_id=lead.container_id)
    assert coord._fleets == {}
    assert _node_cpu_load(coord) == 0.0


@pytest.mark.asyncio
async def test_node_lost_releases_fleet() -> None:
    """A lost node takes its pinned fleet with it — the reservation and all
    member sessions are dropped."""
    coord, _scheduler, node = _make()
    await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    node.next_container_id = "companion"
    await coord.acquire(
        image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
    )

    await coord.handle_node_lost("node-A")
    assert coord._fleets == {}
    assert coord._sessions == {}
    assert _node_cpu_load(coord) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Backward compatibility (the sacred default path)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_fleet_acquire_untouched() -> None:
    """A plain acquire (no fleet labels) takes the legacy path: no fleet_id on
    the session, no reservation, no opening marker — byte-for-byte behavior."""
    coord, _scheduler, _node = _make()
    session = await coord.acquire(image="busybox:1", cpu_limit=4, task_key="t1")

    assert session.fleet_id is None
    assert coord._fleets == {}
    assert coord._fleet_opening == set()
    # Charges the container's own 4 (not any footprint).
    assert _node_cpu_load(coord) == 4.0
    # A non-fleet acquire never writes a fleet reservation row.
    assert coord._state.list_fleet_reservations() == []


# ──────────────────────────────────────────────────────────────────────────────
# Restart-safety persistence (step 5 / R7) — tiny live row, deleted when done
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_opener_persists_footprint_row() -> None:
    """Opening a fleet writes exactly one tiny StateStore row carrying the
    CP-only footprint (so a restart can recover it) — plus node/task_key/owner,
    but NOT the member list (members are rebuilt from labels)."""
    coord, _scheduler, _node = _make()
    from xrlenv.backends.base import ResourceSpec

    await coord.acquire(
        image="busybox:1", cpu_limit=2, task_key="t1",
        owner_id="tenant-A", labels=_opener_labels("f1", 18, 32),
    )
    rows = coord._state.list_fleet_reservations()
    assert len(rows) == 1
    row = rows[0]
    assert row.fleet_id == "f1"
    assert row.node_id == "node-A"
    assert row.task_key == "t1"
    assert row.owner == "tenant-A"
    fp = ResourceSpec.model_validate_json(row.footprint_json)
    assert fp.cpu_request == 18.0
    assert fp.mem_request_bytes == 32 * _GIB
    assert fp.disk_request_bytes == 0  # v1: no fleet disk in the footprint


@pytest.mark.asyncio
async def test_row_deleted_when_last_member_destroyed() -> None:
    """The reservation row is live metadata — it must be DELETED when the
    fleet's last member is destroyed, never linger as history."""
    coord, _scheduler, node = _make()
    lead = await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    node.next_container_id = "companion"
    companion = await coord.acquire(
        image="busybox:1", cpu_limit=16, labels=_companion_labels("f1"),
    )
    assert len(coord._state.list_fleet_reservations()) == 1

    # Destroy the companion → row still present (lead holds the fleet).
    await coord.destroy(rollout_id=companion.rollout_id, container_id=companion.container_id)
    assert len(coord._state.list_fleet_reservations()) == 1
    # Destroy the lead → last member gone → row deleted.
    await coord.destroy(rollout_id=lead.rollout_id, container_id=lead.container_id)
    assert coord._state.list_fleet_reservations() == []


@pytest.mark.asyncio
async def test_row_deleted_on_node_lost() -> None:
    """A lost node drops the fleet — and its persisted row with it (the fleet
    can't survive its pinned node)."""
    coord, _scheduler, _node = _make()
    await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    assert len(coord._state.list_fleet_reservations()) == 1
    await coord.handle_node_lost("node-A")
    assert coord._state.list_fleet_reservations() == []


@pytest.mark.asyncio
async def test_overflow_companion_leaves_reservation_row_untouched() -> None:
    """An overflow companion (graceful fallback) must not corrupt or duplicate
    the persisted reservation row, and its OWN raw row must carry no fleet_id
    (it was admitted as an ordinary container, not a fleet member) so a restart
    rebuild won't mis-adopt it as a suppressed member."""
    coord, _scheduler, node = _make()
    await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    node.next_container_id = "big"
    overflow = await coord.acquire(
        image="busybox:1", cpu_limit=32, labels=_companion_labels("f1"),
    )
    # The reservation row is intact (one, still f1) — the overflow didn't touch it.
    rows = coord._state.list_fleet_reservations()
    assert len(rows) == 1
    assert rows[0].fleet_id == "f1"
    # The overflow container's own raw row carries no fleet association.
    assert coord._state.get_raw_rollout(overflow.rollout_id).fleet_id is None


@pytest.mark.asyncio
async def test_opener_failure_persists_no_row() -> None:
    """If the lead container never comes up, no reservation row is written
    (the row is created only at the success handoff, alongside the in-memory
    reservation)."""
    coord, _scheduler, node = _make()
    node.raise_on_acquire = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await coord.acquire(
            image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
        )
    assert coord._state.list_fleet_reservations() == []


# ──────────────────────────────────────────────────────────────────────────────
# §5.3 — fleet runtime consistency
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fleet_opener_persists_runtime_matching_companion_ok() -> None:
    """A fleet opened under sysbox-runc stores the runtime on the reservation,
    and a companion requesting the SAME runtime is admitted (drawn from the
    reservation) and runs under it on the pinned node."""
    from xrlenv.control.kwargs_policy import KwargsPolicy

    node = _FakeNodeTransport(node_id="node-A")
    node.supported_runtimes = lambda: ["runc", "sysbox-runc"]  # type: ignore[attr-defined]
    state = InMemoryStateStore()
    scheduler = Scheduler([node], catalog=TemplateCatalog(), state=state)
    coord = RawContainerCoordinator(
        scheduler=scheduler, state=state,
        kwargs_policy=KwargsPolicy(allowed_runtimes=("sysbox-runc",)),
    )
    scheduler.set_raw_session_provider(coord.iter_load_entries)

    await coord.acquire(
        image="dind:1", cpu_limit=2,
        labels=_opener_labels("f1", 18, 32), container_runtime="sysbox-runc",
    )
    assert coord._fleets["f1"].container_runtime == "sysbox-runc"

    node.next_container_id = "companion"
    comp = await coord.acquire(
        image="dind:1", cpu_limit=2,
        labels=_companion_labels("f1"), container_runtime="sysbox-runc",
    )
    assert comp.fleet_id == "f1"
    assert node.acquire_calls[-1]["container_runtime"] == "sysbox-runc"


@pytest.mark.asyncio
async def test_fleet_companion_runtime_mismatch_fails_loud() -> None:
    """Every fleet member must run under the same runtime. Companions skip
    Scheduler.place, so a companion requesting a DIFFERENT runtime than the
    opener is caught by an explicit guard and fails loud."""
    from xrlenv.control.kwargs_policy import KwargsPolicy

    node = _FakeNodeTransport(node_id="node-A")
    state = InMemoryStateStore()
    scheduler = Scheduler([node], catalog=TemplateCatalog(), state=state)
    coord = RawContainerCoordinator(
        scheduler=scheduler, state=state,
        kwargs_policy=KwargsPolicy(allowed_runtimes=("sysbox-runc",)),
    )
    scheduler.set_raw_session_provider(coord.iter_load_entries)

    # Opener runs the default runc (no container_runtime).
    await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )
    assert coord._fleets["f1"].container_runtime is None

    # Companion requests sysbox-runc → mismatch with the fleet's runc → loud.
    with pytest.raises(XRLEnvError, match="same runtime"):
        await coord.acquire(
            image="busybox:1", cpu_limit=2,
            labels=_companion_labels("f1"), container_runtime="sysbox-runc",
        )
