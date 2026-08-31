"""P1.7.B.2 — image-affinity scheduling for raw-container acquire.

Pins the new placement flow:

- Coordinator fans out ``query_image(image)`` per acquire.
- Scheduler.place receives the ``image_present`` map.
- preferred_home_node lookup hits StateStore.find_registered_preferred_home.
- ``ensure_image_present`` flag flows through to the node-transport call.
- Pre-flight ``query_image`` on the winner raises XRLEnvError when the
  scheduler's snapshot was stale AND ensure_image_present=False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.errors import XRLEnvError


@dataclass
class _FakeQueryReply:
    present: bool


@dataclass
class _FakeRecord:
    rollout_id: str
    container_id: str
    container_name: str
    image: str


@dataclass
class _FakeNode:
    node_id: str
    backends: list[str] = field(default_factory=lambda: ["docker"])
    has_image: bool = True
    query_calls: list[str] = field(default_factory=list)
    acquire_calls: list[dict] = field(default_factory=list)

    def supported_backends(self) -> list[str]:
        return list(self.backends)

    async def query_image(self, image: str) -> _FakeQueryReply:
        self.query_calls.append(image)
        return _FakeQueryReply(present=self.has_image)

    async def acquire_container(self, **kwargs: Any) -> _FakeRecord:
        self.acquire_calls.append(kwargs)
        return _FakeRecord(
            rollout_id=kwargs["rollout_id"],
            container_id=f"c-{self.node_id}",
            container_name=f"name-c-{self.node_id}",
            image=kwargs["image"],
        )


@dataclass
class _FakePlacement:
    node: Any
    backend: str = "docker"
    score: float = 1.0
    reservation_id: str = "fake-res-0"


@dataclass
class _FakeScheduler:
    nodes: list[_FakeNode]
    image_aware_placement: bool = True
    last_image_present: dict[str, bool] | None = None
    last_preferred_home: str | None = None
    last_task_key: str | None = None
    pick_node: _FakeNode | None = None  # if set, force this winner
    # P1.7.A leak-fix lifecycle counters. ``RawContainerCoordinator.acquire``
    # now calls ``commit_placement`` on success and ``release_placement``
    # on failure; the fake records both so any test asserting the call
    # pattern can introspect, and so the production code's calls don't
    # AttributeError on a fake that pre-dates the lifecycle.
    commit_calls: list[Any] = field(default_factory=list)
    release_calls: list[Any] = field(default_factory=list)
    _next_reservation: int = 0

    def place(
        self,
        manifest: Any,
        *,
        task_key: Any = None,
        backend: Any = None,
        image_present: Any = None,
        preferred_home_node: Any = None,
    ) -> _FakePlacement:
        self.last_image_present = image_present
        self.last_preferred_home = preferred_home_node
        self.last_task_key = task_key
        reservation_id = f"fake-res-{self._next_reservation}"
        self._next_reservation += 1
        if self.pick_node is not None:
            return _FakePlacement(
                node=self.pick_node, reservation_id=reservation_id,
            )
        # Default: prefer a node that has the image; else first docker node.
        if image_present:
            for node in self.nodes:
                if image_present.get(node.node_id):
                    return _FakePlacement(
                        node=node, reservation_id=reservation_id,
                    )
        for node in self.nodes:
            if "docker" in node.supported_backends():
                return _FakePlacement(
                    node=node, reservation_id=reservation_id,
                )
        raise XRLEnvError("no docker-capable node")

    def commit_placement(self, placement: Any) -> None:
        self.commit_calls.append(placement.reservation_id)

    def release_placement(self, placement: Any) -> None:
        self.release_calls.append(placement.reservation_id)


@dataclass
class _FakeState:
    preferred_home_by_image: dict[str, str] = field(default_factory=dict)

    def find_registered_preferred_home(self, image: str) -> str | None:
        return self.preferred_home_by_image.get(image)


# ──────────────────────────────────────────────────────────────────────────────
# Image-affinity fan-out + scheduler integration
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_runs_query_image_fan_out_to_all_nodes() -> None:
    """The coordinator queries every backend-capable node concurrently
    so the scheduler's image_present map is fresh."""
    n1 = _FakeNode(node_id="n1", has_image=True)
    n2 = _FakeNode(node_id="n2", has_image=False)
    sched = _FakeScheduler(nodes=[n1, n2])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1")

    assert n1.query_calls == ["busybox:1", "busybox:1"]  # fan-out + pre-flight
    assert n2.query_calls == ["busybox:1"]  # only fan-out (n2 wasn't picked)
    assert sched.last_image_present == {"n1": True, "n2": False}


@pytest.mark.asyncio
async def test_acquire_passes_preferred_home_to_scheduler() -> None:
    """When StateStore has a preferred_home for the image, the
    coordinator forwards it to ``Scheduler.place(preferred_home_node=...)``."""
    n1 = _FakeNode(node_id="n1")
    sched = _FakeScheduler(nodes=[n1])
    state = _FakeState(preferred_home_by_image={"my-image:1": "n1"})
    coord = RawContainerCoordinator(scheduler=sched, state=state)

    await coord.acquire(image="my-image:1")

    assert sched.last_preferred_home == "n1"


@pytest.mark.asyncio
async def test_acquire_preferred_home_none_when_no_state() -> None:
    """Coordinator without a StateStore reference passes
    preferred_home_node=None — placement still works."""
    n1 = _FakeNode(node_id="n1")
    sched = _FakeScheduler(nodes=[n1])
    coord = RawContainerCoordinator(scheduler=sched)  # no state

    await coord.acquire(image="busybox:1")

    assert sched.last_preferred_home is None


@pytest.mark.asyncio
async def test_acquire_passes_task_key_through() -> None:
    """task_key reaches the scheduler so anti-affinity fires."""
    n1 = _FakeNode(node_id="n1")
    sched = _FakeScheduler(nodes=[n1])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1", task_key="bench/instance-A")

    assert sched.last_task_key == "bench/instance-A"


@pytest.mark.asyncio
async def test_acquire_winner_lands_via_image_affinity() -> None:
    """Two-node scenario: only n2 has the image. Scheduler picks n2;
    acquire_container lands on n2."""
    n1 = _FakeNode(node_id="n1", has_image=False)
    n2 = _FakeNode(node_id="n2", has_image=True)
    sched = _FakeScheduler(nodes=[n1, n2])
    coord = RawContainerCoordinator(scheduler=sched)

    session = await coord.acquire(image="busybox:1")

    assert session.node_id == "n2"
    assert n1.acquire_calls == []
    assert len(n2.acquire_calls) == 1


# ──────────────────────────────────────────────────────────────────────────────
# ensure_image_present opt-out flag
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_ensure_image_present_default_true_passes_to_node() -> None:
    n1 = _FakeNode(node_id="n1")
    sched = _FakeScheduler(nodes=[n1])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1")

    assert n1.acquire_calls[0]["ensure_image_present"] is True


@pytest.mark.asyncio
async def test_acquire_ensure_image_present_false_propagates() -> None:
    n1 = _FakeNode(node_id="n1")
    sched = _FakeScheduler(nodes=[n1])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1", ensure_image_present=False)

    assert n1.acquire_calls[0]["ensure_image_present"] is False


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight query_image on the winner (D19 mirror)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_preflight_passes_when_image_present() -> None:
    """Happy path: pre-flight confirms image present, acquire proceeds."""
    n1 = _FakeNode(node_id="n1", has_image=True)
    sched = _FakeScheduler(nodes=[n1])
    coord = RawContainerCoordinator(scheduler=sched)

    session = await coord.acquire(image="busybox:1")

    assert session.node_id == "n1"
    # query_calls = [fan-out, pre-flight]
    assert len(n1.query_calls) == 2


@pytest.mark.asyncio
async def test_acquire_preflight_raises_on_stale_with_strict_mode() -> None:
    """Pre-flight finds image absent + ensure_image_present=False →
    raise rather than fall through to a wedge / opaque pull-then-timeout."""
    n1 = _FakeNode(node_id="n1", has_image=False)
    sched = _FakeScheduler(nodes=[n1])
    # Force scheduler to pick n1 even though image_present says False
    # (mirrors a stale-snapshot race).
    sched.pick_node = n1
    coord = RawContainerCoordinator(scheduler=sched)

    with pytest.raises(XRLEnvError, match="ensure_image_present is False"):
        await coord.acquire(image="busybox:1", ensure_image_present=False)
    # Acquire never dispatched.
    assert n1.acquire_calls == []


@pytest.mark.asyncio
async def test_acquire_preflight_absent_with_default_proceeds() -> None:
    """ensure_image_present=True (default) tolerates a stale snapshot —
    the node-side ``ensure_present`` would pull. Coordinator just
    proceeds with the acquire."""
    n1 = _FakeNode(node_id="n1", has_image=False)
    sched = _FakeScheduler(nodes=[n1])
    sched.pick_node = n1
    coord = RawContainerCoordinator(scheduler=sched)

    session = await coord.acquire(image="busybox:1")  # default True

    assert session.node_id == "n1"
    assert len(n1.acquire_calls) == 1
    assert n1.acquire_calls[0]["ensure_image_present"] is True


@pytest.mark.asyncio
async def test_acquire_preflight_query_failure_doesnt_abort() -> None:
    """A pre-flight ``query_image`` RPC failure (transport hiccup,
    timeout) is logged + swallowed; acquire proceeds. The node-side
    ensure_present is the real safety net."""
    class _N(_FakeNode):
        async def query_image(self, image: str) -> _FakeQueryReply:
            self.query_calls.append(image)
            if len(self.query_calls) == 1:
                # First call (fan-out): succeed
                return _FakeQueryReply(present=True)
            # Second call (pre-flight): blow up
            raise RuntimeError("simulated transport hiccup")

    n1 = _N(node_id="n1")
    sched = _FakeScheduler(nodes=[n1])
    coord = RawContainerCoordinator(scheduler=sched)

    session = await coord.acquire(image="busybox:1")

    assert session.node_id == "n1"
    assert len(n1.acquire_calls) == 1
