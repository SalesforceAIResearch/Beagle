"""P1.7.B.2 W6+W7+W8 — RawContainerCoordinator.plan_image_distribution.

Pins the operator-side FFD bin-packing flow:

- Coordinator snapshots per-node free disk via report_images.
- plan_opportunistic_placements assigns image → preferred_home.
- StateStore rows persisted with status="registered" so
  find_registered_preferred_home returns the assignment.
- eager_prefetch dispatches ensure_present per assignment.
- No StateStore wired → raises XRLEnvError (operator must configure
  one).
- Empty refs → empty result (no plan written).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.image_planner import ImageToPlace
from xrlenv.control.raw_container_service import (
    RawContainerCoordinator,
)
from xrlenv.control.state import (
    BuildAssignmentRecord,
    BuildPlanRecord,
)
from xrlenv.errors import XRLEnvError


@dataclass
class _FakeNodeReport:
    free_disk_bytes: int


@dataclass
class _FakeNode:
    node_id: str
    backends: list[str] = field(default_factory=lambda: ["docker"])
    free_disk_bytes: int = 100 * 1024 * 1024 * 1024  # 100 GiB
    ensure_present_calls: list[str] = field(default_factory=list)
    ensure_present_status: str = "ok"

    def supported_backends(self) -> list[str]:
        return list(self.backends)

    async def report_images(self) -> _FakeNodeReport:
        return _FakeNodeReport(free_disk_bytes=self.free_disk_bytes)

    async def ensure_present(
        self, image_ref: str, *, timeout_s: float = 0.0,
    ) -> tuple[str, str]:
        self.ensure_present_calls.append(image_ref)
        return (self.ensure_present_status, "")


@dataclass
class _FakeScheduler:
    nodes: list[_FakeNode]
    image_aware_placement: bool = True

    def place(self, *args: Any, **kwargs: Any) -> Any:  # unused here
        raise NotImplementedError


@dataclass
class _FakeState:
    """Mimics the relevant subset of StateStore for plan_image_distribution.
    Records build plans + assignments; supports the lookup."""

    plans: dict[str, BuildPlanRecord] = field(default_factory=dict)
    assignments: list[BuildAssignmentRecord] = field(default_factory=list)

    def record_build_plan(
        self, *, plan_id: str, applied_by: str, plan_json: str,
    ) -> BuildPlanRecord:
        record = BuildPlanRecord(
            plan_id=plan_id, applied_at=time.time(),
            applied_by=applied_by, plan_json=plan_json,
            status="in_flight",
        )
        self.plans[plan_id] = record
        return record

    def record_assignment(self, record: BuildAssignmentRecord) -> None:
        self.assignments.append(record)

    def find_registered_preferred_home(self, image_ref: str) -> str | None:
        for r in self.assignments:
            if r.image_ref == image_ref and r.status == "registered":
                return r.node_id
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Happy path + edge cases
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_image_distribution_writes_preferred_home_rows() -> None:
    """Two nodes, two refs that fit. After planning, each ref has a
    preferred_home row in StateStore."""
    n1 = _FakeNode(node_id="n1", free_disk_bytes=20 * 1024 * 1024 * 1024)
    n2 = _FakeNode(node_id="n2", free_disk_bytes=20 * 1024 * 1024 * 1024)
    state = _FakeState()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[n1, n2]), state=state,
    )

    rows = [
        ImageToPlace(image_ref="X:1", size_bytes=5 * 1024 * 1024 * 1024),
        ImageToPlace(image_ref="Y:1", size_bytes=5 * 1024 * 1024 * 1024),
    ]
    results = await coord.plan_image_distribution(rows=rows)

    assert len(results) == 2
    assert all(r.status == "placed" for r in results)
    # Both refs are placed; their homes are in {n1, n2}.
    homes = {r.preferred_home_node for r in results}
    assert homes <= {"n1", "n2"}
    # State rows recorded with status="registered".
    assert len(state.assignments) == 2
    assert all(a.status == "registered" for a in state.assignments)
    # find_registered_preferred_home returns them.
    assert state.find_registered_preferred_home("X:1") in {"n1", "n2"}
    assert state.find_registered_preferred_home("Y:1") in {"n1", "n2"}


@pytest.mark.asyncio
async def test_plan_image_distribution_records_deferred_when_no_room() -> None:
    """One small node, two big refs. FFD places one; the other goes
    deferred with a preferred_home (the still-most-free node)."""
    small = _FakeNode(
        node_id="small",
        free_disk_bytes=6 * 1024 * 1024 * 1024,  # only 6 GiB
    )
    state = _FakeState()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[small]), state=state,
    )

    rows = [
        ImageToPlace(image_ref="big-A:1", size_bytes=5 * 1024 * 1024 * 1024),
        ImageToPlace(image_ref="big-B:1", size_bytes=5 * 1024 * 1024 * 1024),
    ]
    results = await coord.plan_image_distribution(rows=rows)

    statuses = {r.image_ref: r.status for r in results}
    # Largest-first; first one fits, second one deferred.
    placed_count = sum(1 for v in statuses.values() if v == "placed")
    deferred_count = sum(1 for v in statuses.values() if v == "deferred")
    assert placed_count == 1
    assert deferred_count == 1


@pytest.mark.asyncio
async def test_plan_image_distribution_empty_rows_returns_empty() -> None:
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[_FakeNode(node_id="n1")]),
        state=_FakeState(),
    )
    results = await coord.plan_image_distribution(rows=[])
    assert results == []


@pytest.mark.asyncio
async def test_plan_image_distribution_no_state_raises() -> None:
    """Without a StateStore, the planner can't persist preferred_home
    rows — raise rather than silently succeed."""
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[_FakeNode(node_id="n1")]),
        # state=None
    )
    with pytest.raises(XRLEnvError, match="requires a StateStore"):
        await coord.plan_image_distribution(rows=[
            ImageToPlace(image_ref="X:1", size_bytes=1024),
        ])


@pytest.mark.asyncio
async def test_plan_image_distribution_no_docker_nodes_raises() -> None:
    """All nodes non-docker → no budget snapshot → clear error."""
    cube = _FakeNode(node_id="cube", backends=["cubesandbox"])
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[cube]), state=_FakeState(),
    )
    with pytest.raises(XRLEnvError, match="docker-capable"):
        await coord.plan_image_distribution(rows=[
            ImageToPlace(image_ref="X:1", size_bytes=1024),
        ])


@pytest.mark.asyncio
async def test_plan_image_distribution_skips_failed_report_images() -> None:
    """If one node's report_images blows up, the budget snapshot
    excludes it and FFD plans across the rest."""
    class _BadNode(_FakeNode):
        async def report_images(self) -> Any:
            raise RuntimeError("simulated transport hiccup")

    bad = _BadNode(node_id="bad")
    good = _FakeNode(node_id="good", free_disk_bytes=10 * 1024**3)
    state = _FakeState()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[bad, good]), state=state,
    )

    results = await coord.plan_image_distribution(rows=[
        ImageToPlace(image_ref="X:1", size_bytes=1024**3),
    ])
    # Placed on the good node; the bad one wasn't a candidate.
    placed = [r for r in results if r.status == "placed"]
    assert len(placed) == 1
    assert placed[0].preferred_home_node == "good"


# ──────────────────────────────────────────────────────────────────────────────
# Eager prefetch
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_eager_prefetch_dispatches_ensure_present() -> None:
    n1 = _FakeNode(node_id="n1", free_disk_bytes=20 * 1024**3)
    state = _FakeState()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[n1]), state=state,
    )

    await coord.plan_image_distribution(
        rows=[ImageToPlace(image_ref="X:1", size_bytes=1024**3)],
        eager_prefetch=True,
    )

    assert n1.ensure_present_calls == ["X:1"]


@pytest.mark.asyncio
async def test_plan_eager_prefetch_failure_marks_row_failed() -> None:
    """When ensure_present fails on a node, the corresponding row
    is marked status="failed" with the error captured. Other rows
    succeed."""
    class _BadEnsureNode(_FakeNode):
        async def ensure_present(
            self, image_ref: str, *, timeout_s: float = 0.0,
        ) -> tuple[str, str]:
            self.ensure_present_calls.append(image_ref)
            raise RuntimeError(f"simulated pull failure for {image_ref}")

    bad = _BadEnsureNode(node_id="n1", free_disk_bytes=20 * 1024**3)
    state = _FakeState()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[bad]), state=state,
    )

    results = await coord.plan_image_distribution(
        rows=[ImageToPlace(image_ref="X:1", size_bytes=1024**3)],
        eager_prefetch=True,
    )

    failed = [r for r in results if r.status == "failed"]
    assert len(failed) == 1
    assert "simulated pull failure" in (failed[0].error or "")


@pytest.mark.asyncio
async def test_plan_no_eager_prefetch_skips_ensure() -> None:
    """eager_prefetch=False (default) — no ensure_present calls."""
    n1 = _FakeNode(node_id="n1", free_disk_bytes=20 * 1024**3)
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[n1]), state=_FakeState(),
    )

    await coord.plan_image_distribution(
        rows=[ImageToPlace(image_ref="X:1", size_bytes=1024**3)],
    )  # eager_prefetch=False default

    assert n1.ensure_present_calls == []
