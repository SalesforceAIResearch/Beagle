"""Fleet reservation — R1 accounting seam (phase 1, opt-in).

Covers the *state + load-entry* slice: ``FleetReservation``, the two fleet
labels the coordinator parses (``_parse_fleet_labels``), the member disk-only
helper (``_fleet_member_disk_only``), and the two-loop ``iter_load_entries``.

The single correctness crux (R1): an open fleet contributes **exactly one**
footprint load entry for cpu+mem, its member containers contribute **no**
cpu/mem (only their own disk), and — most important — a deployment with **no**
fleets produces a byte-for-byte-identical load list to the pre-fleet code (the
golden backward-compat guarantee). See
``notes/fleet-reservation-r1-load-accounting.md`` +
``specs/10 §Fleet reservation accounting``.

Everything here is **synthetic + generic** — no EvoClaw fixtures, roles, or
sizes. Fleets are constructed directly on the coordinator's ``_sessions`` /
``_fleets`` maps because the admission flow that *populates* them from a live
acquire lands in a later slice; this slice proves the accounting seam in
isolation.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.compat.metadata import (
    LABEL_FLEET_CPU_REQUEST,
    LABEL_FLEET_ID,
    LABEL_FLEET_MEM_REQUEST,
    LABEL_GROUP_ID,
)
from xrlenv.control.raw_container_service import (
    FleetReservation,
    RawContainerCoordinator,
    RawContainerSession,
    _fleet_member_disk_only,
    _parse_fleet_labels,
)
from xrlenv.control.scheduler import RawSessionLoad
from xrlenv.errors import XRLEnvError

_GIB = 1024**3


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic sessions / fleets / coordinator (no scheduler calls)
# ──────────────────────────────────────────────────────────────────────────────


class _StubScheduler:
    """Minimal stand-in — the coordinator constructor only stores it; none
    of the seam tests here reach the scheduler."""

    nodes: ClassVar[list] = []
    image_aware_placement = False

    def place(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        ...

    def commit_placement(self, placement: object) -> None: ...  # pragma: no cover
    def release_placement(self, placement: object) -> None: ...  # pragma: no cover


def _spec(cpu: float, mem_bytes: int, disk_bytes: int) -> ResourceSpec:
    return ResourceSpec(
        cpu_request=cpu,
        cpu_limit=cpu,
        mem_request_bytes=mem_bytes,
        mem_limit_bytes=mem_bytes,
        disk_request_bytes=disk_bytes,
    )


def _session(
    rollout_id: str,
    *,
    node_id: str = "node-A",
    image: str = "busybox:1",
    cpu: float = 2.0,
    mem_bytes: int = 512 * 1024 * 1024,
    disk_bytes: int = 2 * _GIB,
    task_key: str | None = None,
    fleet_id: str | None = None,
) -> RawContainerSession:
    return RawContainerSession(
        rollout_id=rollout_id,
        node=object(),  # never touched by iter_load_entries
        node_id=node_id,
        container_id=f"c-{rollout_id}",
        container_name=f"name-{rollout_id}",
        image=image,
        created_at=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
        task_key=task_key,
        fleet_id=fleet_id,
        effective_resources=_spec(cpu, mem_bytes, disk_bytes),
    )


def _coord() -> RawContainerCoordinator:
    return RawContainerCoordinator(scheduler=_StubScheduler())


def _by_node_totals(
    entries: list[RawSessionLoad], node_id: str = "node-A",
) -> tuple[float, int, int]:
    """Sum (cpu_request, mem_request_bytes, disk_request_bytes) over the
    entries on one node — exactly what ``_gather_cluster_load`` +
    ``capacity`` do downstream."""
    on = [e for e in entries if e.node_id == node_id]
    return (
        sum(e.effective_resources.cpu_request for e in on),
        sum(e.effective_resources.mem_request_bytes for e in on),
        sum(e.effective_resources.disk_request_bytes for e in on),
    )


# ──────────────────────────────────────────────────────────────────────────────
# _parse_fleet_labels — the consumer→CP declaration hop
# ──────────────────────────────────────────────────────────────────────────────


def test_parse_fleet_labels_absent_is_not_a_fleet() -> None:
    """No labels / no ``xrlenv.fleet_id`` → ``(None, None)``: the common,
    zero-cost non-fleet path."""
    assert _parse_fleet_labels(None) == (None, None)
    assert _parse_fleet_labels({}) == (None, None)
    assert _parse_fleet_labels({LABEL_GROUP_ID: "g1"}) == (None, None)


def test_parse_fleet_labels_opener_yields_cpu_mem_footprint() -> None:
    """A fleet-opening acquire (fleet_id + both footprint labels) yields the
    id and a cpu+mem ResourceSpec. Disk is structurally 0 — v1 never puts
    disk in the footprint."""
    fleet_id, footprint = _parse_fleet_labels({
        LABEL_FLEET_ID: "fleet-1",
        LABEL_FLEET_CPU_REQUEST: "18",
        LABEL_FLEET_MEM_REQUEST: str(32 * _GIB),
    })
    assert fleet_id == "fleet-1"
    assert footprint is not None
    assert footprint.cpu_request == 18.0
    assert footprint.cpu_limit == 18.0
    assert footprint.mem_request_bytes == 32 * _GIB
    assert footprint.mem_limit_bytes == 32 * _GIB
    assert footprint.disk_request_bytes == 0  # v1: no fleet disk


def test_parse_fleet_labels_companion_has_id_but_no_footprint() -> None:
    """A companion carries only ``xrlenv.fleet_id`` — footprint was declared
    by the opener, not re-declared here → ``(fleet_id, None)``."""
    assert _parse_fleet_labels({LABEL_FLEET_ID: "fleet-1"}) == ("fleet-1", None)


def test_parse_fleet_labels_partial_footprint_raises() -> None:
    """fleet_id with only ONE of the two footprint labels is a declaration
    bug → fail loud (never silently degrade to per-container admission)."""
    with pytest.raises(XRLEnvError, match="incomplete"):
        _parse_fleet_labels({
            LABEL_FLEET_ID: "fleet-1",
            LABEL_FLEET_CPU_REQUEST: "18",
        })
    with pytest.raises(XRLEnvError, match="incomplete"):
        _parse_fleet_labels({
            LABEL_FLEET_ID: "fleet-1",
            LABEL_FLEET_MEM_REQUEST: str(_GIB),
        })


def test_parse_fleet_labels_malformed_footprint_raises() -> None:
    with pytest.raises(XRLEnvError, match="malformed"):
        _parse_fleet_labels({
            LABEL_FLEET_ID: "fleet-1",
            LABEL_FLEET_CPU_REQUEST: "not-a-number",
            LABEL_FLEET_MEM_REQUEST: str(_GIB),
        })


def test_parse_fleet_labels_nonpositive_footprint_raises() -> None:
    with pytest.raises(XRLEnvError, match="positive"):
        _parse_fleet_labels({
            LABEL_FLEET_ID: "fleet-1",
            LABEL_FLEET_CPU_REQUEST: "0",
            LABEL_FLEET_MEM_REQUEST: str(_GIB),
        })


# ──────────────────────────────────────────────────────────────────────────────
# _fleet_member_disk_only — the one documented member exception
# ──────────────────────────────────────────────────────────────────────────────


def test_fleet_member_disk_only_zeroes_cpu_mem_keeps_disk() -> None:
    """A fleet member's cpu+mem are footprint-covered, but its disk stays
    per-container — so we zero cpu+mem and keep disk verbatim."""
    out = _fleet_member_disk_only(_spec(cpu=16, mem_bytes=32 * _GIB, disk_bytes=7 * _GIB))
    assert out.cpu_request == 0.0
    assert out.cpu_limit == 0.0
    assert out.mem_request_bytes == 0
    assert out.mem_limit_bytes == 0
    assert out.disk_request_bytes == 7 * _GIB


# ──────────────────────────────────────────────────────────────────────────────
# iter_load_entries — golden backward-compat (the sacred default path)
# ──────────────────────────────────────────────────────────────────────────────


def test_iter_load_entries_no_fleets_is_byte_for_byte_legacy() -> None:
    """With ``_fleets`` empty and every session ``fleet_id is None``,
    ``iter_load_entries`` returns EXACTLY the pre-fleet list — one full
    entry per session, order preserved. This is the golden guarantee that
    tb2.1 / SWE-bench accounting is unchanged."""
    coord = _coord()
    coord._sessions = {
        "r1": _session("r1", image="busybox:1", cpu=2, task_key="task-A"),
        "r2": _session("r2", image="busybox:2", cpu=4, task_key="task-B"),
    }
    # The exact list the pre-fleet function produced.
    expected = [
        RawSessionLoad(
            node_id=s.node_id,
            template_name=f"raw-container/{s.image}",
            effective_resources=s.effective_resources,
            task_key=s.task_key,
        )
        for s in coord._sessions.values()
    ]
    assert coord.iter_load_entries() == expected


# ──────────────────────────────────────────────────────────────────────────────
# iter_load_entries — one open fleet: footprint once, members disk-only
# ──────────────────────────────────────────────────────────────────────────────


def test_open_fleet_charges_footprint_once_not_footprint_plus_members() -> None:
    """The R1 crux. Footprint cpu=20, members 2 + 16. Node cpu load must be
    **20** (the footprint) — not 38 (footprint + members, a double-count) and
    not 18 (members only, the starvation bug). mem is footprint-covered the
    same way; disk is the exception (per-member)."""
    coord = _coord()
    footprint = _spec(cpu=20, mem_bytes=40 * _GIB, disk_bytes=0)
    # members mirrors the real post-acquire state (raw_container_service.py registers each
    # companion in ``reservation.members`` at acquire); suppression is membership-scoped (H11).
    coord._fleets = {
        "fleet-1": FleetReservation(
            fleet_id="fleet-1",
            node_id="node-A",
            footprint=footprint,
            members={
                "lead": _spec(cpu=2, mem_bytes=4 * _GIB, disk_bytes=2 * _GIB),
                "companion": _spec(cpu=16, mem_bytes=32 * _GIB, disk_bytes=3 * _GIB),
            },
            task_key="task-X",
        ),
    }
    coord._sessions = {
        "lead": _session(
            "lead", cpu=2, mem_bytes=4 * _GIB, disk_bytes=2 * _GIB,
            fleet_id="fleet-1",
        ),
        "companion": _session(
            "companion", cpu=16, mem_bytes=32 * _GIB, disk_bytes=3 * _GIB,
            fleet_id="fleet-1",
        ),
    }

    entries = coord.iter_load_entries()
    # 2 member disk-only entries + 1 footprint entry.
    assert len(entries) == 3

    cpu, mem, disk = _by_node_totals(entries)
    assert cpu == 20.0                 # footprint once — the whole point
    assert mem == 40 * _GIB            # footprint once
    assert disk == (2 + 3) * _GIB      # members' own disk (footprint disk is 0)

    # Exactly one footprint entry, carrying the fleet's task_key + declared peak.
    fleet_entries = [e for e in entries if e.template_name == "raw-fleet/fleet-1"]
    assert len(fleet_entries) == 1
    assert fleet_entries[0].effective_resources == footprint
    assert fleet_entries[0].task_key == "task-X"

    # Member entries carry NO task_key (fairness counted once, on the fleet).
    member_entries = [
        e for e in entries if e.template_name.startswith("raw-fleet-member/")
    ]
    assert len(member_entries) == 2
    assert all(e.task_key is None for e in member_entries)
    assert all(e.effective_resources.cpu_request == 0.0 for e in member_entries)
    assert all(e.effective_resources.mem_request_bytes == 0 for e in member_entries)


def test_open_fleet_counts_as_one_fairness_unit() -> None:
    """A fleet is one ``(node, task_key)`` unit for ``max_runs_per_task`` —
    exactly one emitted entry carries the task_key, no matter how many
    members are up."""
    coord = _coord()
    coord._fleets = {
        "fleet-1": FleetReservation(
            fleet_id="fleet-1", node_id="node-A",
            footprint=_spec(20, 40 * _GIB, 0),
            members={rid: _spec(4, 512 * 1024 * 1024, 2 * _GIB) for rid in ("a", "b", "c")},
            task_key="task-X",
        ),
    }
    coord._sessions = {
        rid: _session(rid, cpu=4, fleet_id="fleet-1") for rid in ("a", "b", "c")
    }
    entries = coord.iter_load_entries()
    task_x_entries = [e for e in entries if e.task_key == "task-X"]
    assert len(task_x_entries) == 1


# ──────────────────────────────────────────────────────────────────────────────
# iter_load_entries — mixed / edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_mixed_fleet_and_plain_session() -> None:
    """One open fleet (footprint cpu 18) + one ordinary non-fleet session
    (cpu 4) → the plain session charges its full 4 alongside the fleet's 18,
    node total 22."""
    coord = _coord()
    coord._fleets = {
        "fleet-1": FleetReservation(
            fleet_id="fleet-1", node_id="node-A",
            footprint=_spec(18, 36 * _GIB, 0),
            members={"lead": _spec(2, 512 * 1024 * 1024, 2 * _GIB)},
            task_key="task-X",
        ),
    }
    coord._sessions = {
        "lead": _session("lead", cpu=2, fleet_id="fleet-1"),
        "plain": _session("plain", cpu=4, task_key="task-Y"),
    }
    entries = coord.iter_load_entries()
    cpu, _, _ = _by_node_totals(entries)
    assert cpu == 22.0
    # The plain session is unchanged: a full raw-container entry.
    plain = [e for e in entries if e.template_name == "raw-container/busybox:1"]
    assert len(plain) == 1
    assert plain[0].effective_resources.cpu_request == 4.0
    assert plain[0].task_key == "task-Y"


def test_member_without_live_reservation_charges_full_container() -> None:
    """A session carrying a ``fleet_id`` whose reservation is GONE (the brief
    last-member release window) must fall through to the full-container
    charge — a conservative over-count, never an accounting gap that would
    let a second placement over-place into still-occupied capacity."""
    coord = _coord()
    coord._fleets = {}  # reservation already dropped
    coord._sessions = {
        "orphan": _session("orphan", cpu=16, fleet_id="fleet-gone", task_key="task-X"),
    }
    entries = coord.iter_load_entries()
    assert len(entries) == 1
    # Charged as a normal container: full cpu + its own task_key restored.
    assert entries[0].template_name == "raw-container/busybox:1"
    assert entries[0].effective_resources.cpu_request == 16.0
    assert entries[0].task_key == "task-X"


def test_release_last_member_returns_footprint_to_free_pool() -> None:
    """Dropping the reservation (last-member destroy, simulated) removes the
    footprint entry in one step — capacity returns to its non-fleet baseline
    (invariant 2: released only on node-confirmed destroy)."""
    coord = _coord()
    coord._fleets = {
        "fleet-1": FleetReservation(
            fleet_id="fleet-1", node_id="node-A",
            footprint=_spec(20, 40 * _GIB, 0), members={}, task_key="task-X",
        ),
    }
    coord._sessions = {}  # last member already destroyed
    assert _by_node_totals(coord.iter_load_entries())[0] == 20.0  # still reserved

    # Reservation dropped → footprint gone, node back to zero raw load.
    coord._fleets = {}
    assert coord.iter_load_entries() == []


# ──────────────────────────────────────────────────────────────────────────────
# Integration — real Scheduler folds the seam correctly (_gather_cluster_load)
# ──────────────────────────────────────────────────────────────────────────────


def test_real_scheduler_sees_fleet_as_footprint_once() -> None:
    """Wire the seam into a real ``Scheduler`` via ``set_raw_session_provider``
    and confirm ``_gather_cluster_load`` folds a fleet to its footprint once
    (cpu 20, not 38 / 18) and counts it as a single fairness unit
    (task_count == 1)."""
    from xrlenv.control.scheduler import Scheduler
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.control.template_catalog import TemplateCatalog

    from tests.unit.control.test_raw_container_coordinator import (
        _FakeNodeTransport,
    )

    node = _FakeNodeTransport(node_id="node-A")
    scheduler = Scheduler(
        [node], catalog=TemplateCatalog(), state=InMemoryStateStore(),
        max_runs_per_task=4,
    )
    coord = _coord()
    coord._fleets = {
        "fleet-1": FleetReservation(
            fleet_id="fleet-1", node_id="node-A",
            footprint=_spec(20, 40 * _GIB, 0),
            members={
                "lead": _spec(2, 512 * 1024 * 1024, 2 * _GIB),
                "companion": _spec(16, 512 * 1024 * 1024, 3 * _GIB),
            },
            task_key="task-X",
        ),
    }
    coord._sessions = {
        "lead": _session("lead", cpu=2, disk_bytes=2 * _GIB, fleet_id="fleet-1"),
        "companion": _session(
            "companion", cpu=16, disk_bytes=3 * _GIB, fleet_id="fleet-1",
        ),
    }
    scheduler.set_raw_session_provider(coord.iter_load_entries)

    load = scheduler._gather_cluster_load(task_key="task-X")
    running, task_count = load["node-A"]
    total_cpu = sum(spec.cpu_request for _, spec in running)
    total_disk = sum(spec.disk_request_bytes for _, spec in running)
    assert total_cpu == 20.0            # footprint once
    assert total_disk == (2 + 3) * _GIB  # members' own disk still visible
    assert task_count == 1               # one fairness unit, not two members
