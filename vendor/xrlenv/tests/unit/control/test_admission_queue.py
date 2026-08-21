"""Tests for the AdmissionQueue (spec 03 / spec 20 ``pending_rollouts``)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.scheduler import Placement
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.errors import CapacityExhausted


def _manifest(name: str = "t") -> TemplateManifest:
    return TemplateManifest(
        name=name,
        version="0.1",
        digest=f"sha256:{name}",
        image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _placement(node_id: str = "node-A") -> Placement:
    node = MagicMock()
    node.node_id = node_id
    return Placement(node=node, backend="docker", score=1)


def _scheduler_returning(*, place_results: list[Any]) -> Any:
    """Scheduler whose ``place(...)`` consumes ``place_results``; the *last*
    entry is repeated indefinitely (so tests can model a node that never
    becomes available without painstakingly seeding N entries).
    """
    sched = MagicMock()
    # A1 / D18 (P1.2) — fake schedulers default to image-affinity OFF
    # so the admission queue's pre-fetch path stays out of the way of
    # tests that just want to drive ``place()`` directly. Tests that
    # exercise affinity construct a real scheduler.
    sched.image_aware_placement = False
    sched.nodes = []
    iterator = iter(place_results)
    last: list[Any] = []

    def _place(
        _manifest, *, task_key=None, backend=None,
        container_runtime=None,
        image_present=None, preferred_home_node=None,
        exclude_node_ids=None,
    ):
        try:
            v = next(iterator)
            last.clear()
            last.append(v)
        except StopIteration:
            v = last[0]
        if isinstance(v, BaseException):
            raise v
        return v

    sched.place.side_effect = _place
    return sched


# ──────────────────────────────────────────────────────────────────────────────
# Fast-path acquire: scheduler.place succeeds immediately
# ──────────────────────────────────────────────────────────────────────────────


async def test_acquire_returns_placement_immediately_on_capacity() -> None:
    state = InMemoryStateStore()
    sched = _scheduler_returning(place_results=[_placement("A")])
    q = AdmissionQueue(scheduler=sched, state=state)
    placement = await q.acquire(manifest=_manifest(), timeout_s=1.0)
    assert placement.node.node_id == "A"
    assert state.list_pending() == []


async def test_acquire_does_not_start_worker() -> None:
    """Fast-path acquire must not require start() to have been called."""
    state = InMemoryStateStore()
    sched = _scheduler_returning(place_results=[_placement("A")])
    q = AdmissionQueue(scheduler=sched, state=state)
    placement = await q.acquire(manifest=_manifest(), timeout_s=0.5)
    assert placement.node.node_id == "A"


async def test_acquire_passes_preferred_home_from_state_to_scheduler() -> None:
    """Audit P1.6.g-H2 (2026-05-05): when the manifest's image is a
    deferred (registered) row in the build snapshot, the queue must
    look up its preferred_home and forward it to ``scheduler.place()``
    so first-rollout placement honors the bin-packer's spread plan.
    """
    from xrlenv.control.state import BuildAssignmentRecord

    state = InMemoryStateStore()
    state.record_build_plan(
        plan_id="p1", applied_by="cli", plan_json="{}",
    )
    state.record_assignment(BuildAssignmentRecord(
        plan_id="p1", node_id="planner-home", image_ref="im/t:1",
        benchmark="b", status="registered",
    ))

    sched = _scheduler_returning(place_results=[_placement("A")])
    q = AdmissionQueue(scheduler=sched, state=state)
    await q.acquire(manifest=_manifest(), timeout_s=0.5)

    # The scheduler call must carry the preferred_home_node from the
    # build snapshot. Inspect the captured kwargs.
    assert sched.place.called
    call = sched.place.call_args
    assert call.kwargs.get("preferred_home_node") == "planner-home"


async def test_acquire_passes_none_preferred_home_when_no_registered_row(
) -> None:
    """When the build snapshot has no registered row for the
    manifest's image, ``preferred_home_node`` is ``None`` — the
    scheduler scores normally."""
    state = InMemoryStateStore()
    sched = _scheduler_returning(place_results=[_placement("A")])
    q = AdmissionQueue(scheduler=sched, state=state)
    await q.acquire(manifest=_manifest(), timeout_s=0.5)

    call = sched.place.call_args
    assert call.kwargs.get("preferred_home_node") is None


# ──────────────────────────────────────────────────────────────────────────────
# Queue-and-drain path
# ──────────────────────────────────────────────────────────────────────────────


async def test_acquire_enqueues_then_admits_when_capacity_frees() -> None:
    state = InMemoryStateStore()
    # First place() raises (queue), second place() succeeds (drain admits).
    sched = _scheduler_returning(
        place_results=[CapacityExhausted("full"), _placement("A")]
    )
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=0.05)
    await q.start()
    kick_task: asyncio.Task[None] | None = None
    try:
        # Kick after a tiny delay so the worker observes the queued waiter.
        async def kick_after() -> None:
            await asyncio.sleep(0.05)
            q.kick()

        kick_task = asyncio.create_task(kick_after())
        placement = await q.acquire(manifest=_manifest(), timeout_s=1.0)
        assert placement.node.node_id == "A"
        # Row was enqueued and then drained.
        assert state.list_pending() == []
    finally:
        if kick_task is not None:
            await kick_task
        await q.stop()


async def test_queued_acquire_preserves_container_runtime_on_drain_retry() -> None:
    """§5.3 — a queued sysbox acquire (CapacityExhausted → parked as a _Waiter)
    must be re-placed WITH its container_runtime on every drain retry, so it
    can't be admitted onto a node that doesn't advertise the runtime."""
    state = InMemoryStateStore()
    sched = _scheduler_returning(
        place_results=[CapacityExhausted("full"), _placement("A")]
    )
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=0.05)
    await q.start()
    kick_task: asyncio.Task[None] | None = None
    try:
        async def kick_after() -> None:
            await asyncio.sleep(0.05)
            q.kick()

        kick_task = asyncio.create_task(kick_after())
        await q.acquire(
            manifest=_manifest(), timeout_s=1.0,
            container_runtime="sysbox-runc",
        )
        # BOTH the initial (fast-path) place and the drain-retry place got the
        # runtime — the filter is never bypassed on the way through the queue.
        runtimes = [
            c.kwargs.get("container_runtime")
            for c in sched.place.call_args_list
        ]
        assert len(runtimes) >= 2
        assert all(r == "sysbox-runc" for r in runtimes)
    finally:
        if kick_task is not None:
            await kick_task
        await q.stop()


async def test_queued_acquire_preserves_exclude_node_ids_on_drain_retry() -> None:
    """D-AR-2026-07-07-B (audit P2) — the production admission path.

    A control-plane re-admit passes ``exclude_node_ids`` so a create-time
    saturated node is steered away from. When that re-admit can't fast-path
    place (node B momentarily full → CapacityExhausted → parked as a
    ``_Waiter``), the exclusion MUST be re-passed on every drain retry —
    otherwise the queued waiter could drain right back onto the failed node A,
    reintroducing the "A failed, B temporarily full, queued waiter lands on A"
    bug. The coordinator's own re-admit tests use the direct scheduler path;
    this pins the queued path.
    """
    state = InMemoryStateStore()
    # First place() (fast path, node B full) parks the waiter; the drain retry
    # then places on B. BOTH calls must carry the exclusion.
    sched = _scheduler_returning(
        place_results=[CapacityExhausted("B full"), _placement("node-B")]
    )
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=0.05)
    await q.start()
    kick_task: asyncio.Task[None] | None = None
    try:
        async def kick_after() -> None:
            await asyncio.sleep(0.05)
            q.kick()

        kick_task = asyncio.create_task(kick_after())
        placement = await q.acquire(
            manifest=_manifest(), timeout_s=1.0,
            exclude_node_ids=frozenset({"node-A"}),
        )
        # Drained onto the sibling, never back onto the excluded node A.
        assert placement.node.node_id == "node-B"
        # The exclusion was threaded to the fast-path place AND stored on the
        # _Waiter → re-passed on the drain retry. A regression that drops it
        # from acquire(), _Waiter, or the drain path fails here.
        excludes = [
            c.kwargs.get("exclude_node_ids")
            for c in sched.place.call_args_list
        ]
        assert len(excludes) >= 2  # fast-path + at least one drain retry
        assert all(e == frozenset({"node-A"}) for e in excludes)
        assert state.list_pending() == []
    finally:
        if kick_task is not None:
            await kick_task
        await q.stop()


async def test_acquire_times_out_when_queue_never_drains() -> None:
    state = InMemoryStateStore()
    sched = _scheduler_returning(
        place_results=[CapacityExhausted("first"), CapacityExhausted("still full")]
    )
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=0.05)
    await q.start()
    try:
        with pytest.raises(CapacityExhausted, match="queue_timeout_s"):
            await q.acquire(manifest=_manifest(), timeout_s=0.2)
        # Pending row should be cleaned up on timeout.
        assert state.list_pending() == []
    finally:
        await q.stop()


async def test_acquire_propagates_non_capacity_exception_from_scheduler() -> None:
    state = InMemoryStateStore()
    sched = _scheduler_returning(
        place_results=[CapacityExhausted("first"), RuntimeError("scheduler boom")]
    )
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=0.05)
    await q.start()
    kick_task: asyncio.Task[None] | None = None
    try:

        async def kick_after() -> None:
            await asyncio.sleep(0.05)
            q.kick()

        kick_task = asyncio.create_task(kick_after())
        with pytest.raises(RuntimeError, match="scheduler boom"):
            await q.acquire(manifest=_manifest(), timeout_s=1.0)
        assert state.list_pending() == []
    finally:
        if kick_task is not None:
            await kick_task
        await q.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Shutdown semantics
# ──────────────────────────────────────────────────────────────────────────────


async def test_stop_accepting_blocks_new_acquires() -> None:
    state = InMemoryStateStore()
    q = AdmissionQueue(
        scheduler=_scheduler_returning(place_results=[]), state=state
    )
    q.stop_accepting()
    with pytest.raises(CapacityExhausted, match="no longer accepting"):
        await q.acquire(manifest=_manifest(), timeout_s=0.1)


async def test_cancel_pending_resolves_waiters_with_exception() -> None:
    state = InMemoryStateStore()
    sched = _scheduler_returning(place_results=[CapacityExhausted("full")])
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=10.0)
    # Don't start the worker — we want the waiter to be queued without being admitted.
    waiter_task = asyncio.create_task(
        q.acquire(manifest=_manifest(), timeout_s=5.0)
    )
    await asyncio.sleep(0.05)  # let the acquire enqueue
    assert state.list_pending() != []

    q.cancel_pending()
    with pytest.raises(CapacityExhausted, match="cancelled on shutdown"):
        await waiter_task
    assert state.list_pending() == []


async def test_queue_status_reports_position_and_depth() -> None:
    """Stage-2: ``queue_status`` reports a request's 1-based FIFO
    position and the total queue depth; an unknown id reads as
    not-in-queue."""
    state = InMemoryStateStore()
    # Scheduler always full → every acquire queues (worker not started).
    sched = _scheduler_returning(
        place_results=[CapacityExhausted("full")] * 10,
    )
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=10.0)
    tasks = [
        asyncio.create_task(
            q.acquire(manifest=_manifest(), request_id=f"r-{i}", timeout_s=5.0),
        )
        for i in range(3)
    ]
    await asyncio.sleep(0.1)  # let all three enqueue
    try:
        # Each known request has a distinct position in 1..3, depth 3.
        seen = {q.queue_status(f"r-{i}") for i in range(3)}
        assert {pos for pos, _, _ in seen} == {1, 2, 3}
        assert all(depth == 3 and st == "queued" for _, depth, st in seen)
        # An unknown id: not in the queue, but depth still reported.
        assert q.queue_status("r-unknown") == (0, 3, "not_in_queue")
    finally:
        q.cancel_pending()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_wait_idle_returns_true_when_empty() -> None:
    state = InMemoryStateStore()
    q = AdmissionQueue(
        scheduler=_scheduler_returning(place_results=[]), state=state
    )
    assert await q.wait_idle(0.1) is True


# ──────────────────────────────────────────────────────────────────────────────
# Worker idempotency
# ──────────────────────────────────────────────────────────────────────────────


async def test_kick_is_safe_before_start() -> None:
    state = InMemoryStateStore()
    q = AdmissionQueue(
        scheduler=_scheduler_returning(place_results=[]), state=state
    )
    q.kick()  # must not raise


async def test_queue_status_is_owner_scoped() -> None:
    """Audit M1-residual: a scoped caller sees only their own queued requests;
    another tenant's request_id reads not_in_queue with no position/existence
    leak. owner_id=None keeps the original global view."""
    import asyncio

    from xrlenv.control.admission import _Waiter

    state = InMemoryStateStore()
    q = AdmissionQueue(
        scheduler=_scheduler_returning(place_results=[_placement()]),
        state=state,
    )
    loop = asyncio.get_running_loop()
    q._waiters["p1"] = _Waiter(
        manifest=_manifest(), task_key=None, future=loop.create_future(),
        owner_id="alice", request_id="r-alice",
    )
    q._waiters["p2"] = _Waiter(
        manifest=_manifest(), task_key=None, future=loop.create_future(),
        owner_id="bob", request_id="r-bob",
    )

    # alice sees only her own request: position 1 of 1 within her scope.
    assert q.queue_status("r-alice", "alice") == (1, 1, "queued")
    # alice asking about bob's request id → not_in_queue, depth is her own (1).
    assert q.queue_status("r-bob", "alice") == (0, 1, "not_in_queue")
    # Global view (owner None, single-tenant / no-auth) sees both.
    assert q.queue_status("r-bob") == (2, 2, "queued")
    assert q.queue_status("r-alice") == (1, 2, "queued")
