"""Issue #18 follow-up B — unit tests for queue_timeout_s plumb-through
and queue_wait_s surfacing.

Covers:
1. Proto field round-trip: AcquireContainerRequest.queue_timeout_s and
   AcquireContainerResponse.queue_wait_s set/read correctly.
2. End-to-end plumb-through via InProcessTransport: Client.acquire_container
   reaches CoordinatorRolloutService/RawContainerCoordinator with the value
   the caller passed; default (None) arrives as DEFAULT_QUEUE_TIMEOUT_S (24h).
3. queue_wait_s surfacing: RawContainerSession / RawAcquireResult /
   ClusterContainerSession carry the measured wait end to end; fast path
   produces 0.0.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.client.client import Client
from xrlenv.client.container_session import ClusterContainerSession
from xrlenv.client.transport import InProcessTransport
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.service import RawAcquireResult

# ──────────────────────────────────────────────────────────────────────────────
# Re-use the established fake shapes from test_raw_container_coordinator
# (copied here to keep this file self-contained and avoid import coupling).
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
    acquire_calls: list[dict] = field(default_factory=list)
    next_container_id: str = "container-001"
    has_image: bool = True

    def supported_backends(self) -> list[str]:
        return list(self.backends)

    def hardware(self) -> Any:
        from xrlenv.node.hw_probe import HardwareInfo
        return HardwareInfo(
            vcpus=64, mem_bytes=128 * 1024**3, disk_bytes=2000 * 1024**3,
            has_kvm=True, has_gpu=False, gpu_model=None,
            kernel_version="6.0.0", platform="linux",
        )

    async def query_image(self, image: str) -> Any:
        @dataclass
        class _R:
            present: bool
        return _R(present=self.has_image)

    async def acquire_container(self, **kwargs: Any) -> _FakeRecord:
        self.acquire_calls.append(kwargs)
        return _FakeRecord(
            rollout_id=kwargs["rollout_id"],
            container_id=self.next_container_id,
            container_name=f"name-of-{self.next_container_id}",
            image=kwargs["image"],
        )

    async def container_exec(self, **kwargs: Any) -> dict[str, Any]:
        return {"exit_code": 0, "stdout": b"", "stderr": b"", "timed_out": False}

    async def destroy_container(self, **kwargs: Any) -> None:
        pass

    async def container_put_archive(self, **kwargs: Any) -> None:
        pass

    async def container_get_archive(self, **kwargs: Any) -> bytes:
        return b""

    def container_exec_stream(self, **kwargs: Any) -> Any:
        async def _gen() -> Any:
            yield {"stdout": b"", "stderr": b"", "done": True,
                   "exit_code": 0, "timed_out": False}
        return _gen()


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

    def place(
        self, manifest: Any, *, task_key: Any = None, backend: Any = None,
        image_present: Any = None, preferred_home_node: Any = None,
    ) -> _FakePlacement:
        self.place_calls.append({
            "manifest_image": getattr(manifest, "image", None),
            "task_key": task_key, "backend": backend,
            "image_present": image_present,
            "preferred_home_node": preferred_home_node,
        })
        for node in self.nodes:
            if "docker" in node.supported_backends():
                reservation_id = f"fake-res-{self._next_reservation}"
                self._next_reservation += 1
                return _FakePlacement(
                    node=node, backend="docker", score=1.0,
                    reservation_id=reservation_id,
                )
        from xrlenv.errors import XRLEnvError
        raise XRLEnvError("no node supports backend 'docker'")

    def commit_placement(self, placement: Any) -> None:
        self.commit_calls.append(placement.reservation_id)

    def release_placement(self, placement: Any) -> None:
        self.release_calls.append(placement.reservation_id)


@dataclass
class _FakeAdmission:
    acquire_calls: list[dict] = field(default_factory=list)
    kick_calls: int = 0
    raise_on_acquire: Exception | None = None
    _placement: Any = None
    _wait_s: float = 0.0  # simulated queue wait before returning

    async def acquire(
        self, *, manifest: Any, task_key: Any = None,
        request_id: Any = None, owner_id: str = "default",
        backend: Any = None, timeout_s: float | None = None,
    ) -> Any:
        self.acquire_calls.append({
            "manifest_image": getattr(manifest, "image", None),
            "task_key": task_key, "request_id": request_id,
            "backend": backend, "timeout_s": timeout_s,
        })
        if self.raise_on_acquire is not None:
            raise self.raise_on_acquire
        if self._wait_s > 0:
            await asyncio.sleep(self._wait_s)
        return self._placement

    def kick(self) -> None:
        self.kick_calls += 1

    def queue_status(self, request_id: str) -> tuple[int, int, str]:
        return (0, 0, "not_in_queue")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Proto round-trip
# ──────────────────────────────────────────────────────────────────────────────


def test_acquire_container_request_queue_timeout_s_field() -> None:
    """AcquireContainerRequest.queue_timeout_s (field 22) sets and reads
    back the exact float value through protobuf."""
    from xrlenv.api._pb2 import rollout_control_pb2 as rpb

    req = rpb.AcquireContainerRequest(queue_timeout_s=7200.0)
    assert req.queue_timeout_s == pytest.approx(7200.0)

    # 0.0 is the "use server default" sentinel.
    req_default = rpb.AcquireContainerRequest()
    assert req_default.queue_timeout_s == pytest.approx(0.0)


def test_acquire_container_response_queue_wait_s_field() -> None:
    """AcquireContainerResponse.queue_wait_s (field 5) sets and reads back
    the measured wait."""
    from xrlenv.api._pb2 import rollout_control_pb2 as rpb

    resp = rpb.AcquireContainerResponse(queue_wait_s=3.75)
    assert resp.queue_wait_s == pytest.approx(3.75)

    # Fast path: field absent → 0.0 default.
    resp_fast = rpb.AcquireContainerResponse()
    assert resp_fast.queue_wait_s == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. End-to-end plumb-through via InProcessTransport
#
# Strategy: use a minimal RolloutService fake that records the
# queue_timeout_s value it received via acquire_container. This avoids
# constructing a full RolloutCoordinator (which requires a Scheduler,
# StateStore, TemplateCatalog, etc.) while still exercising the full
# Client → InProcessTransport → service forwarding chain.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _SpyRolloutService:
    """Minimal RolloutService that records acquire_container kwargs."""

    acquire_calls: list[dict] = field(default_factory=list)
    _next_result: RawAcquireResult = field(
        default_factory=lambda: RawAcquireResult(
            rollout_id="r-spy", container_id="c-spy",
            container_name="cname-spy", node_id="node-spy",
        ),
    )

    async def acquire_container(self, **kwargs: Any) -> RawAcquireResult:
        self.acquire_calls.append(kwargs)
        return self._next_result

    # The rest of the RolloutService Protocol is unused for these tests;
    # type-ignore keeps the dataclass clean.


@pytest.mark.asyncio
async def test_queue_timeout_s_explicit_value_flows_through_inprocess_transport() -> None:
    """Client.acquire_container(queue_timeout_s=1234.0) should reach the
    RolloutService.acquire_container call with queue_timeout_s=1234.0.

    Tests the InProcessTransport layer — it substitutes the cluster
    default for None but passes explicit values unchanged."""
    spy = _SpyRolloutService()
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    await client.acquire_container(image="busybox:1", queue_timeout_s=1234.0)

    assert len(spy.acquire_calls) == 1
    assert spy.acquire_calls[0]["queue_timeout_s"] == pytest.approx(1234.0)


@pytest.mark.asyncio
async def test_cpu_mem_limit_flow_through_inprocess_transport() -> None:
    """P0a — Client.acquire_container(cpu_limit=, mem_limit_bytes=) must
    reach the RolloutService.acquire_container call unchanged."""
    spy = _SpyRolloutService()
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    await client.acquire_container(
        image="busybox:1", cpu_limit=4.0, mem_limit_bytes=8 * 1024**3,
    )

    assert spy.acquire_calls[0]["cpu_limit"] == pytest.approx(4.0)
    assert spy.acquire_calls[0]["mem_limit_bytes"] == 8 * 1024**3


@pytest.mark.asyncio
async def test_cpu_mem_limit_thread_through_coordinator_service() -> None:
    """P0a — CoordinatorRolloutService.acquire_container threads the
    harness CPU/memory request to RawContainerCoordinator.acquire.
    Regression guard: a missing thread-through here means the gRPC path
    silently drops the limits (the rollout endpoint passes them in)."""
    from xrlenv.control.service import CoordinatorRolloutService

    @dataclass
    class _SpyCoordinator:
        acquire_calls: list[dict] = field(default_factory=list)

        async def acquire(self, **kwargs: Any) -> Any:
            self.acquire_calls.append(kwargs)
            # RawContainerSession and RawAcquireResult share the field
            # names CoordinatorRolloutService.acquire_container reads.
            return RawAcquireResult(
                rollout_id="r-spy", container_id="c-spy",
                container_name="cname-spy", node_id="node-spy",
            )

    spy_coord = _SpyCoordinator()
    svc = CoordinatorRolloutService(
        None,  # type: ignore[arg-type]  # case-1 coordinator unused here
        raw_container_coordinator=spy_coord,  # type: ignore[arg-type]
    )

    await svc.acquire_container(
        image="busybox:1", cpu_limit=6.0, mem_limit_bytes=2 * 1024**3,
    )

    assert spy_coord.acquire_calls[0]["cpu_limit"] == pytest.approx(6.0)
    assert spy_coord.acquire_calls[0]["mem_limit_bytes"] == 2 * 1024**3


@pytest.mark.asyncio
async def test_cpu_isolation_flows_through_inprocess_transport() -> None:
    """P6 — Client.acquire_container(cpu_isolation=REQUIRED) must reach the
    RolloutService.acquire_container call unchanged (the in-process transport
    passes the scalar through; it does not derive)."""
    from xrlenv.backends.base import CpuIsolation

    spy = _SpyRolloutService()
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    await client.acquire_container(
        image="busybox:1", cpu_isolation=CpuIsolation.REQUIRED,
    )

    assert spy.acquire_calls[0]["cpu_isolation"] is CpuIsolation.REQUIRED


@pytest.mark.asyncio
async def test_cpu_isolation_defaults_off_via_inprocess_transport() -> None:
    """P6 — no cpu_isolation kwarg → OFF reaches the service (safe default,
    never accidental isolation)."""
    from xrlenv.backends.base import CpuIsolation

    spy = _SpyRolloutService()
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    await client.acquire_container(image="busybox:1")

    assert spy.acquire_calls[0]["cpu_isolation"] is CpuIsolation.OFF


@pytest.mark.asyncio
async def test_cpu_isolation_threads_through_coordinator_service() -> None:
    """P6 — CoordinatorRolloutService.acquire_container threads the derived
    cpu_isolation contract to RawContainerCoordinator.acquire. Regression
    guard: a missing thread-through here means the field is dropped between the
    ingress derivation and the coordinator (where it is stamped on the
    effective ResourceSpec)."""
    from xrlenv.backends.base import CpuIsolation
    from xrlenv.control.service import CoordinatorRolloutService

    @dataclass
    class _SpyCoordinator:
        acquire_calls: list[dict] = field(default_factory=list)

        async def acquire(self, **kwargs: Any) -> Any:
            self.acquire_calls.append(kwargs)
            return RawAcquireResult(
                rollout_id="r-spy", container_id="c-spy",
                container_name="cname-spy", node_id="node-spy",
            )

    spy_coord = _SpyCoordinator()
    svc = CoordinatorRolloutService(
        None,  # type: ignore[arg-type]
        raw_container_coordinator=spy_coord,  # type: ignore[arg-type]
    )

    await svc.acquire_container(
        image="busybox:1", cpu_isolation=CpuIsolation.REQUIRED,
    )

    assert spy_coord.acquire_calls[0]["cpu_isolation"] is CpuIsolation.REQUIRED


@pytest.mark.asyncio
async def test_runtime_limits_thread_through_coordinator_service() -> None:
    """P0b — CoordinatorRolloutService.acquire_container threads the
    harness RuntimeLimits to RawContainerCoordinator.acquire."""
    from xrlenv.backends.base import RuntimeLimits
    from xrlenv.control.service import CoordinatorRolloutService

    @dataclass
    class _SpyCoordinator:
        acquire_calls: list[dict] = field(default_factory=list)

        async def acquire(self, **kwargs: Any) -> Any:
            self.acquire_calls.append(kwargs)
            return RawAcquireResult(
                rollout_id="r-spy", container_id="c-spy",
                container_name="cname-spy", node_id="node-spy",
            )

    spy_coord = _SpyCoordinator()
    svc = CoordinatorRolloutService(
        None,  # type: ignore[arg-type]
        raw_container_coordinator=spy_coord,  # type: ignore[arg-type]
    )

    await svc.acquire_container(
        image="busybox:1",
        runtime_limits=RuntimeLimits(pids_limit=2048, readonly_rootfs=True),
    )

    rl = spy_coord.acquire_calls[0]["runtime_limits"]
    assert rl.pids_limit == 2048
    assert rl.readonly_rootfs is True


@pytest.mark.asyncio
async def test_queue_timeout_s_none_defaults_to_24h_via_inprocess_transport() -> None:
    """Audit M1: ``Client.acquire_container()`` with no queue_timeout_s
    (None) must reach the service as the 24h ``DEFAULT_QUEUE_TIMEOUT_S``
    — the in-process transport used to hardcode 3600s, bypassing the
    Stage-2 single-default contract that the gRPC path honours."""
    from xrlenv.control.admission import DEFAULT_QUEUE_TIMEOUT_S

    spy = _SpyRolloutService()
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    await client.acquire_container(image="busybox:1")

    assert spy.acquire_calls[0]["queue_timeout_s"] == pytest.approx(
        DEFAULT_QUEUE_TIMEOUT_S,
    )
    assert DEFAULT_QUEUE_TIMEOUT_S == 86_400.0  # 24h, exact


# ──────────────────────────────────────────────────────────────────────────────
# 3. queue_wait_s surfacing
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_wait_s_is_below_threshold_on_fast_path() -> None:
    """When admission returns without delay, the session's queue_wait_s is
    below the 1-second WARN threshold (the coordinator measures wall-clock
    time, so it's a tiny positive value rather than exactly 0.0)."""
    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    admission_placement = _FakePlacement(node=node, backend="docker", score=1.0)
    admission = _FakeAdmission(_placement=admission_placement, _wait_s=0.0)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    session = await coord.acquire(image="busybox:1")

    # Sub-microsecond wall-clock read — well below the 1s WARN threshold.
    assert session.queue_wait_s < 1.0


@pytest.mark.asyncio
async def test_queue_wait_s_is_nonzero_when_admission_sleeps() -> None:
    """When the admission queue introduces latency, the resulting session
    carries a non-zero queue_wait_s that reflects the measured wait."""
    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    admission_placement = _FakePlacement(node=node, backend="docker", score=1.0)
    # 0.05s is fast enough for a unit test but detectable.
    admission = _FakeAdmission(_placement=admission_placement, _wait_s=0.05)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    session = await coord.acquire(image="busybox:1")

    assert session.queue_wait_s > 0.0


@pytest.mark.asyncio
async def test_raw_acquire_result_carries_queue_wait_s() -> None:
    """CoordinatorRolloutService.acquire_container propagates queue_wait_s
    from RawContainerSession into the returned RawAcquireResult, which the
    Client wraps into ClusterContainerSession."""
    spy = _SpyRolloutService(
        _next_result=RawAcquireResult(
            rollout_id="r-2", container_id="c-2",
            container_name="cname-2", node_id="node-A",
            queue_wait_s=2.5,
        ),
    )
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    session = await client.acquire_container(image="busybox:1")

    assert isinstance(session, ClusterContainerSession)
    assert session.queue_wait_s == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_raw_acquire_result_queue_wait_s_zero_on_fast_path() -> None:
    """Fast-path acquire (no queuing) yields queue_wait_s==0.0 on the
    ClusterContainerSession."""
    spy = _SpyRolloutService()  # default RawAcquireResult has queue_wait_s=0.0
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    session = await client.acquire_container(image="busybox:1")

    assert session.queue_wait_s == pytest.approx(0.0)


def test_cluster_container_session_exposes_queue_wait_s() -> None:
    """ClusterContainerSession stores queue_wait_s from RawAcquireResult
    and surfaces it via the property."""
    from tests.unit.client.test_cluster_container_session import _FakeTransport

    transport = _FakeTransport(
        next_acquire=RawAcquireResult(
            rollout_id="r-99", container_id="c-99",
            container_name="cname-99", node_id="node-A",
            queue_wait_s=4.5,
        ),
    )
    session = ClusterContainerSession(transport, transport.next_acquire)
    assert session.queue_wait_s == pytest.approx(4.5)


def test_raw_acquire_result_default_queue_wait_s() -> None:
    """RawAcquireResult.queue_wait_s defaults to 0.0 (fast path)."""
    result = RawAcquireResult(
        rollout_id="r-1", container_id="c-1",
        container_name="cn-1", node_id="node-A",
    )
    assert result.queue_wait_s == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 — session_deadline_s plumb-through
# ──────────────────────────────────────────────────────────────────────────────


def test_acquire_container_request_session_deadline_s_field() -> None:
    """AcquireContainerRequest.session_deadline_s (field 23) sets and
    reads back; proto3 default is 0.0 ('use server default')."""
    from xrlenv.api._pb2 import rollout_control_pb2 as rpb

    req = rpb.AcquireContainerRequest(session_deadline_s=7200.0)
    assert req.session_deadline_s == pytest.approx(7200.0)
    assert rpb.AcquireContainerRequest().session_deadline_s == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_session_deadline_s_explicit_value_flows_through_inprocess() -> None:
    """Client.acquire_container(session_deadline_s=...) reaches the
    RolloutService.acquire_container call unchanged."""
    spy = _SpyRolloutService()
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    await client.acquire_container(
        image="busybox:1", session_deadline_s=900.0,
    )

    assert spy.acquire_calls[0]["session_deadline_s"] == pytest.approx(900.0)


@pytest.mark.asyncio
async def test_session_deadline_s_none_flows_through_as_none() -> None:
    """No session_deadline_s kwarg → None reaches the service, which
    lets the coordinator apply its default cap (unlike queue_timeout_s,
    the transport does NOT substitute a number here — None is the
    'use coordinator default' signal)."""
    spy = _SpyRolloutService()
    transport = InProcessTransport(spy)  # type: ignore[arg-type]
    client = Client(transport)

    await client.acquire_container(image="busybox:1")

    assert spy.acquire_calls[0]["session_deadline_s"] is None
