"""Focused tests for bookkeeping invariants in RawContainerCoordinator.destroy()
when a consumer NodeCommandTimeout is swallowed (resilient teardown).

audit H8 / invariant 2: a consumer destroy that TIMES OUT is not node-confirmed. It still
doesn't raise to the consumer (the work is done), but capacity is NOT released — the session +
its liveness bookkeeping are RETAINED and teardown is deferred to the raw-GC reconciler, which
frees capacity only on confirmed absence.

This file verifies:
1. Consumer timeout clears ``_destroying`` (so it can't get stuck True) — the deadline/liveness
   sweep + a retry can then re-attempt.
2. Consumer timeout RETAINS liveness bookkeeping (_last_seen_at, _inflight_rpcs, _heartbeated)
   — the session is still live/charging until node-confirmed teardown.
3. ``admission.kick()`` does NOT fire on consumer timeout (no capacity was freed).
4. ``is_destroying()`` returns False after a consumer timeout, but the session is RETAINED.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.errors import NodeCommandTimeout, XRLEnvError

# ──────────────────────────────────────────────────────────────────────────────
# Minimal fakes (mirror the pattern from test_raw_container_coordinator.py)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeRecord:
    rollout_id: str
    container_id: str
    container_name: str
    image: str


@dataclass
class _FakeNodeTransport:
    node_id: str = "node-A"
    backends: list[str] = field(default_factory=lambda: ["docker"])
    next_container_id: str = "container-001"

    def supported_backends(self) -> list[str]:
        return list(self.backends)

    def hardware(self) -> Any:
        from xrlenv.node.hw_probe import HardwareInfo
        return HardwareInfo(
            vcpus=64, mem_bytes=128 * 1024 ** 3, disk_bytes=2000 * 1024 ** 3,
            has_kvm=True, has_gpu=False, gpu_model=None,
            kernel_version="6.0.0", platform="linux",
        )

    async def query_image(self, image: str) -> Any:
        @dataclass
        class _R:
            present: bool = True
        return _R()

    async def acquire_container(self, **kwargs: Any) -> _FakeRecord:
        return _FakeRecord(
            rollout_id=kwargs["rollout_id"],
            container_id=self.next_container_id,
            container_name=f"name-of-{self.next_container_id}",
            image=kwargs["image"],
        )

    async def container_exec(self, **kwargs: Any) -> dict:
        return {"exit_code": 0, "stdout": b"", "stderr": b"", "timed_out": False}

    async def destroy_container(self, **kwargs: Any) -> None:
        raise NodeCommandTimeout("node A: destroy timed out after 300.0s")


@dataclass
class _FakePlacement:
    node: Any
    backend: str = "docker"
    score: float = 1.0
    reservation_id: str = "fake-res-0"


@dataclass
class _FakeScheduler:
    nodes: list[Any]
    image_aware_placement: bool = True
    place_calls: list[dict] = field(default_factory=list)
    commit_calls: list[Any] = field(default_factory=list)
    release_calls: list[Any] = field(default_factory=list)
    _next_reservation: int = 0

    def place(self, manifest: Any, *, task_key: Any = None, backend: Any = None,
              image_present: Any = None, preferred_home_node: Any = None) -> _FakePlacement:
        for node in self.nodes:
            if "docker" in node.supported_backends():
                rid = f"fake-res-{self._next_reservation}"
                self._next_reservation += 1
                return _FakePlacement(node=node, backend="docker", score=1.0, reservation_id=rid)
        raise XRLEnvError("no node supports backend 'docker'")

    def commit_placement(self, placement: Any) -> None:
        self.commit_calls.append(placement.reservation_id)

    def release_placement(self, placement: Any) -> None:
        self.release_calls.append(placement.reservation_id)


@dataclass
class _FakeAdmission:
    _placement: Any = None
    kick_calls: int = 0

    async def acquire(self, *, manifest: Any, task_key: Any = None,
                      request_id: Any = None, owner_id: str = "default",
                      backend: Any = None,
                      timeout_s: float | None = None) -> Any:
        return self._placement

    def kick(self) -> None:
        self.kick_calls += 1

    def queue_status(self, request_id: str) -> tuple[int, int, str]:
        return (0, 0, "not_in_queue")


def _make_coord(node: _FakeNodeTransport) -> RawContainerCoordinator:
    return RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consumer_timeout_clears_destroying_flag() -> None:
    """``_destroying`` must not be left True after a swallowed consumer
    timeout — is_destroying() must return False once destroy() returns."""
    node = _FakeNodeTransport()
    coord = _make_coord(node)
    session = await coord.acquire(image="busybox:1")

    # Consumer destroy — no reason → timeout is swallowed.
    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
    )

    assert coord.is_destroying(session.rollout_id) is False, (
        "_destroying flag must be cleared after a consumer-timeout destroy"
    )


@pytest.mark.asyncio
async def test_consumer_timeout_retains_liveness_bookkeeping() -> None:
    """audit H8: a consumer-timeout destroy is not node-confirmed, so the session is RETAINED
    and its liveness bookkeeping (``_last_seen_at`` / ``_inflight_rpcs`` / ``_heartbeated``)
    stays — the session is still live/charging until the reconciler confirms teardown and frees
    it (which is where liveness is finally dropped)."""
    node = _FakeNodeTransport()
    coord = _make_coord(node)
    session = await coord.acquire(image="busybox:1")
    rollout_id = session.rollout_id

    # Inject synthetic liveness state to make the retention observable.
    import time
    coord._last_seen_at[rollout_id] = time.time()  # type: ignore[attr-defined]
    coord._inflight_rpcs[rollout_id] = 2           # type: ignore[attr-defined]
    coord._heartbeated.add(rollout_id)              # type: ignore[attr-defined]

    await coord.destroy(
        rollout_id=rollout_id,
        container_id=session.container_id,
    )

    # RETAINED — the session is still live until node-confirmed teardown.
    assert [s.rollout_id for s in coord.list_sessions()] == [rollout_id]
    assert rollout_id in coord._last_seen_at        # type: ignore[attr-defined]
    assert rollout_id in coord._inflight_rpcs       # type: ignore[attr-defined]
    assert rollout_id in coord._heartbeated         # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_consumer_timeout_does_not_fire_admission_kick() -> None:
    """audit H8: a consumer-timeout destroy released NO capacity (session retained), so
    ``admission.kick()`` must NOT fire — waking a waiter would let it re-place against capacity
    that is still charged (an invariant-2 violation)."""
    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=1.0)
    admission = _FakeAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0  # reset counter from acquire's kick

    # Consumer destroy — timeout swallowed, but capacity NOT freed.
    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
    )

    assert admission.kick_calls == 0, (
        "admission.kick() must NOT fire after a consumer-timeout destroy — "
        "the session's capacity is retained, not freed"
    )


@pytest.mark.asyncio
async def test_consumer_timeout_is_destroying_false_but_session_retained() -> None:
    """``is_destroying()`` must be False once the consumer timeout is swallowed (so the
    deadline/liveness sweep can re-attempt), but the session is RETAINED (audit H8)."""
    node = _FakeNodeTransport()
    coord = _make_coord(node)
    session = await coord.acquire(image="busybox:1")

    assert coord.is_destroying(session.rollout_id) is False

    destroy_started = asyncio.Event()

    async def _slow_timeout_destroy(**kwargs: Any) -> None:
        destroy_started.set()
        raise NodeCommandTimeout("slow node")

    node.destroy_container = _slow_timeout_destroy  # type: ignore[method-assign]

    task = asyncio.create_task(
        coord.destroy(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
        )
    )

    await destroy_started.wait()
    await asyncio.sleep(0)
    await task

    assert coord.is_destroying(session.rollout_id) is False, (
        "is_destroying must be False after consumer timeout is swallowed"
    )
    # RETAINED — capacity held until the reconciler confirms teardown.
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]
