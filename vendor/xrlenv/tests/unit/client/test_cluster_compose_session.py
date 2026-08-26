"""P1.7.C.2 (step 4a) — consumer SDK surface for multi-service compose projects.

``Client.acquire_compose_project`` → :class:`ClusterComposeSession`, plus the
transport wiring (InProcess + gRPC) it rides on. The session shares the whole
``exec`` / ``put_archive`` / ``get_archive`` / ``exec_stream`` / ``apply_egress``
surface with :class:`ClusterContainerSession` (all bound to the project's ``main``
service container), and differs only in construction (from a
``RawComposeAcquireResult``) and teardown (``destroy`` downs the whole project via
``destroy_compose_project``). Fakes stand in for the transport / service / gRPC
stub so no live control plane is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.client.client import Client
from xrlenv.client.container_session import (
    ClusterComposeSession,
    ClusterContainerSession,
)
from xrlenv.client.transport import GrpcClientTransport, InProcessTransport
from xrlenv.control.raw_container_service import RawComposeAcquireResult
from xrlenv.control.service import RawExecResult


def _result(**over: Any) -> RawComposeAcquireResult:
    base: dict[str, Any] = {
        "rollout_id": "r-1",
        "node_id": "node-A",
        "main_container_id": "cidmainfull",
        "main_container_name": "proj-main",
        "project_name": "proj",
        "service_container_ids": {"main": "cidmainfull", "pg": "cidpgfull"},
        "queue_wait_s": 1.5,
    }
    base.update(over)
    return RawComposeAcquireResult(**base)


@dataclass
class _FakeTransport:
    """Records compose acquire/destroy + exec/archive routing."""

    next_acquire: RawComposeAcquireResult = field(default_factory=_result)
    acquire_calls: list[dict] = field(default_factory=list)
    destroy_compose_calls: list[dict] = field(default_factory=list)
    exec_calls: list[dict] = field(default_factory=list)
    put_archive_calls: list[dict] = field(default_factory=list)
    get_archive_calls: list[dict] = field(default_factory=list)
    raise_on_destroy: Exception | None = None

    async def acquire_compose_project(self, **kwargs: Any) -> RawComposeAcquireResult:
        self.acquire_calls.append(kwargs)
        return self.next_acquire

    async def destroy_compose_project(self, **kwargs: Any) -> None:
        if self.raise_on_destroy:
            raise self.raise_on_destroy
        self.destroy_compose_calls.append(kwargs)

    async def container_exec(self, **kwargs: Any) -> RawExecResult:
        self.exec_calls.append(kwargs)
        return RawExecResult(exit_code=0, stdout=b"hi\n", stderr=b"", timed_out=False)

    async def container_put_archive(self, **kwargs: Any) -> None:
        self.put_archive_calls.append(kwargs)

    async def container_get_archive(self, **kwargs: Any) -> bytes:
        self.get_archive_calls.append(kwargs)
        return b"<tar>"

    async def heartbeat_many(self, rollout_ids: list[str]) -> None: ...
    async def close(self) -> None: ...


# ──────────────────────────────────────────────────────────────────────────────
# ClusterComposeSession
# ──────────────────────────────────────────────────────────────────────────────


def test_session_is_a_container_session_subclass() -> None:
    # so a harness typed against ClusterContainerSession accepts a compose session
    assert issubclass(ClusterComposeSession, ClusterContainerSession)


def test_session_carries_compose_result_attrs() -> None:
    t = _FakeTransport()
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    assert s.rollout_id == "r-1"
    assert s.container_id == "cidmainfull"  # exec/archive target = main service
    assert s.container_name == "proj-main"
    assert s.node_id == "node-A"
    assert s.queue_wait_s == 1.5
    assert s.project_name == "proj"
    assert s.service_container_ids == {"main": "cidmainfull", "pg": "cidpgfull"}
    # the map is a copy — callers can't mutate internal state
    s.service_container_ids["pg"] = "hacked"
    assert s.service_container_ids["pg"] == "cidpgfull"


@pytest.mark.asyncio
async def test_exec_and_archive_target_main_container() -> None:
    t = _FakeTransport()
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    await s.exec(["echo", "hi"], timeout_s=5.0)
    await s.put_archive("/tmp", b"<in>")
    await s.get_archive("/logs")

    assert t.exec_calls[0]["container_id"] == "cidmainfull"
    assert t.exec_calls[0]["rollout_id"] == "r-1"
    assert t.put_archive_calls[0]["container_id"] == "cidmainfull"
    assert t.get_archive_calls[0]["container_id"] == "cidmainfull"


@pytest.mark.asyncio
async def test_destroy_downs_whole_project() -> None:
    t = _FakeTransport()
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    await s.destroy()

    assert s.destroyed is True
    # routes to compose-down (whole stack), NOT destroy_container
    assert t.destroy_compose_calls == [
        {"rollout_id": "r-1", "project_name": "proj"},
    ]


@pytest.mark.asyncio
async def test_destroy_is_idempotent() -> None:
    t = _FakeTransport()
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    await s.destroy()
    await s.destroy()
    await s.destroy()

    assert len(t.destroy_compose_calls) == 1


@pytest.mark.asyncio
async def test_failed_destroy_stays_usable_and_retryable() -> None:
    # P1 (audit): compose teardown is STRICT server-side — a failed `docker
    # compose down` RETAINS the session + capacity for retry (invariant 2). So a
    # failed destroy must NOT mark the session destroyed; a second call retries.
    t = _FakeTransport(raise_on_destroy=RuntimeError("down failed"))
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="down failed"):
        await s.destroy()
    assert s.destroyed is False  # NOT marked destroyed → still usable for retry
    assert t.destroy_compose_calls == []  # the failed down committed nothing

    # a retry (down now succeeds) completes and marks destroyed
    t.raise_on_destroy = None
    await s.destroy()
    assert s.destroyed is True
    assert t.destroy_compose_calls == [
        {"rollout_id": "r-1", "project_name": "proj"},
    ]


@pytest.mark.asyncio
async def test_apply_egress_fails_loud_for_compose() -> None:
    # P2 (audit): the inherited apply_egress would restrict only `main`'s netns,
    # silently leaving sidecars unrestricted. Until project-network egress lands
    # (design §4.3) it must fail loud rather than under-enforce.
    from xrlenv.backends.egress import EgressAllowlist

    t = _FakeTransport()
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="project-network"):
        await s.apply_egress(EgressAllowlist(rules=()))


@pytest.mark.asyncio
async def test_async_context_manager_downs_project_on_exit() -> None:
    t = _FakeTransport()
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    async with s as entered:
        assert entered is s
        await s.exec(["true"])

    assert s.destroyed is True
    assert len(t.destroy_compose_calls) == 1


@pytest.mark.asyncio
async def test_async_context_manager_downs_project_on_exception() -> None:
    t = _FakeTransport()
    s = ClusterComposeSession(t, t.next_acquire)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="boom"):
        async with s:
            raise ValueError("boom")

    assert s.destroyed is True
    assert len(t.destroy_compose_calls) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Client.acquire_compose_project
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_client_acquire_returns_compose_session_and_forwards_args() -> None:
    t = _FakeTransport()
    client = Client(t)  # type: ignore[arg-type]

    session = await client.acquire_compose_project(
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app@sha256:abc", "postgres:14"],
        footprint_cpu=6.0,
        footprint_mem_bytes=8 * 1024**3,
        project_name="proj",
        task_key="tw_522753",
    )

    assert isinstance(session, ClusterComposeSession)
    assert session.project_name == "proj"
    call = t.acquire_calls[0]
    assert call["footprint_cpu"] == 6.0
    assert call["footprint_mem_bytes"] == 8 * 1024**3
    assert call["images"] == ["ns/app@sha256:abc", "postgres:14"]
    assert call["task_key"] == "tw_522753"


@pytest.mark.asyncio
async def test_client_acquire_registers_keepalive_destroy_deregisters() -> None:
    t = _FakeTransport()
    client = Client(t)  # type: ignore[arg-type]

    session = await client.acquire_compose_project(
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app:main"],
        footprint_cpu=2.0,
        footprint_mem_bytes=2 * 1024**3,
    )
    assert "r-1" in client._keepalive._ids  # beating the project's main session
    await session.destroy()
    assert "r-1" not in client._keepalive._ids  # deregistered on down
    await client.close()


@pytest.mark.asyncio
async def test_failed_destroy_keeps_keepalive_until_confirmed() -> None:
    # P1 (audit): a failed down must keep the session HEARTBEATING so the CP
    # doesn't liveness-reap a project the caller is still tearing down; the
    # keepalive is dropped only after a confirmed down.
    t = _FakeTransport(raise_on_destroy=RuntimeError("down failed"))
    client = Client(t)  # type: ignore[arg-type]

    session = await client.acquire_compose_project(
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app:main"],
        footprint_cpu=2.0,
        footprint_mem_bytes=2 * 1024**3,
    )
    assert "r-1" in client._keepalive._ids
    with pytest.raises(RuntimeError, match="down failed"):
        await session.destroy()
    assert "r-1" in client._keepalive._ids  # still beating → not reaped mid-retry

    t.raise_on_destroy = None  # confirmed retry deregisters
    await session.destroy()
    assert "r-1" not in client._keepalive._ids
    await client.close()


# ──────────────────────────────────────────────────────────────────────────────
# InProcessTransport — footprint scalar → ResourceSpec at the service boundary
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inprocess_builds_footprint_resource_spec() -> None:
    calls: list[dict] = []

    class _FakeService:
        async def acquire_compose_project(self, **kwargs: Any) -> RawComposeAcquireResult:
            calls.append(kwargs)
            return _result()

        async def destroy_compose_project(self, **kwargs: Any) -> None:
            calls.append({"destroy": kwargs})

    transport = InProcessTransport(_FakeService())  # type: ignore[arg-type]
    await transport.acquire_compose_project(
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app:main"],
        footprint_cpu=6.0,
        footprint_mem_bytes=8 * 1024**3,
    )

    fp = calls[0]["footprint"]
    # request==limit packing; disk unset; whole-stack reserve
    assert fp.cpu_request == 6.0
    assert fp.cpu_limit == 6.0
    assert fp.mem_request_bytes == 8 * 1024**3
    assert fp.mem_limit_bytes == 8 * 1024**3
    assert fp.disk_request_bytes == 0
    # ``None`` queue_timeout resolved to the cluster default, not left None
    assert calls[0]["queue_timeout_s"] is not None


@pytest.mark.asyncio
async def test_inprocess_destroy_forwards_project_name() -> None:
    calls: list[dict] = []

    class _FakeService:
        async def destroy_compose_project(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    transport = InProcessTransport(_FakeService())  # type: ignore[arg-type]
    await transport.destroy_compose_project(rollout_id="r-1", project_name="proj")
    assert calls == [{"rollout_id": "r-1", "project_name": "proj"}]


# ──────────────────────────────────────────────────────────────────────────────
# GrpcClientTransport — request building + response unpacking
# ──────────────────────────────────────────────────────────────────────────────


def _bare_grpc(stub: object) -> GrpcClientTransport:
    t = object.__new__(GrpcClientTransport)
    t._token = None  # type: ignore[attr-defined]
    t._stub = stub  # type: ignore[attr-defined]
    return t


class _RecordingStub:
    def __init__(self) -> None:
        self.acquire_req: Any = None
        self.destroy_req: Any = None
        self.destroy_timeout: float | None = None

    async def AcquireComposeProject(
        self, req: Any, metadata: object = None, timeout: float | None = None,
    ) -> Any:
        self.acquire_req = req
        return rpb.AcquireComposeProjectResponse(
            rollout_id="r-9", node_id="node-B",
            main_container_id="cidmain", main_container_name="p-main",
            project_name="p", queue_wait_s=2.0,
            service_container_ids={"main": "cidmain", "db": "ciddb"},
        )

    async def DestroyComposeProject(
        self, req: Any, metadata: object = None, timeout: float | None = None,
    ) -> Any:
        self.destroy_req = req
        self.destroy_timeout = timeout
        return SimpleNamespace()


@pytest.mark.asyncio
async def test_grpc_acquire_builds_request_and_unpacks_response() -> None:
    stub = _RecordingStub()
    transport = _bare_grpc(stub)

    rec = await transport.acquire_compose_project(
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app@sha256:abc", "postgres:14"],
        footprint_cpu=6.0,
        footprint_mem_bytes=8 * 1024**3,
        main_service="main",
        project_name="p",
        task_key="tk",
        group_id="g",
        labels={"k": "v"},
        up_timeout_s=300.0,
    )

    req = stub.acquire_req
    assert req.compose_yaml == "services:\n  main: {}\n"
    assert list(req.images) == ["ns/app@sha256:abc", "postgres:14"]
    assert req.main_service == "main"
    assert req.project_name == "p"
    assert req.task_key == "tk"
    assert req.group_id == "g"
    assert dict(req.labels) == {"k": "v"}
    assert req.up_timeout_s == 300.0
    # footprint proto: request==limit
    assert req.footprint.cpu_request == 6.0
    assert req.footprint.cpu_limit == 6.0
    assert req.footprint.mem_request_bytes == 8 * 1024**3
    assert req.footprint.mem_limit_bytes == 8 * 1024**3
    # request_id stamped for the queue-status poller
    assert req.request_id
    # response unpacked into the duck-typed result the session consumes
    assert isinstance(rec, RawComposeAcquireResult)
    assert rec.rollout_id == "r-9"
    assert rec.node_id == "node-B"
    assert rec.main_container_id == "cidmain"
    assert rec.project_name == "p"
    assert rec.service_container_ids == {"main": "cidmain", "db": "ciddb"}
    assert rec.queue_wait_s == 2.0


@pytest.mark.asyncio
async def test_grpc_acquire_defaults_use_proto_sentinels() -> None:
    stub = _RecordingStub()
    transport = _bare_grpc(stub)

    await transport.acquire_compose_project(
        compose_yaml="services:\n  main: {}\n",
        images=[],
        footprint_cpu=1.0,
        footprint_mem_bytes=1024,
    )
    req = stub.acquire_req
    assert req.main_service == "main"       # default
    assert req.queue_timeout_s == 0.0       # None → 0.0 = "server default"
    assert req.session_deadline_s == 0.0
    assert req.up_timeout_s == 0.0
    assert list(req.images) == []
    assert req.project_name == ""           # unset


@pytest.mark.asyncio
async def test_grpc_destroy_builds_request_with_deadline() -> None:
    stub = _RecordingStub()
    transport = _bare_grpc(stub)

    await transport.destroy_compose_project(rollout_id="r-9", project_name="p")

    assert stub.destroy_req.rollout_id == "r-9"
    assert stub.destroy_req.project_name == "p"
    # teardown carries a bounded backstop deadline (never blocks forever)
    assert stub.destroy_timeout is not None and stub.destroy_timeout > 0


async def test_compose_session_also_exposes_liveness_at_risk() -> None:
    """ClusterComposeSession must mirror base-class fields it does not inherit.

    It deliberately does NOT route through ``super().__init__`` — the compose
    acquire result has a different shape — so any field added to the base class
    silently goes missing here unless mirrored. Adding ``liveness_probe`` to the
    base broke every compose acquire with a TypeError until this class took it
    too; a compose consumer must be able to read the same signal.
    """
    from xrlenv.client.container_session import ClusterComposeSession

    class _R:
        rollout_id = "r1"
        main_container_id = "c1"
        main_container_name = "n1"
        node_id = "node-A"
        queue_wait_s = 0.0
        project_name = "proj"
        service_container_ids: ClassVar[dict[str, str]] = {}

    at_risk = False
    session = ClusterComposeSession(
        object(), _R(), liveness_probe=lambda: at_risk,  # type: ignore[arg-type]
    )
    assert session.liveness_at_risk is False
    at_risk = True
    assert session.liveness_at_risk is True

    assert ClusterComposeSession(object(), _R()).liveness_at_risk is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_liveness_at_risk_reflects_real_keepalive_state_via_acquire_compose_project() -> None:
    """Same real-path regression as the container-session test, for compose.

    ``ClusterComposeSession`` deliberately does not call
    ``ClusterContainerSession.__init__`` (see ``test_compose_session_also_...``
    above), which is exactly the kind of seam where a session-handle field can
    be wired correctly on paper and still not reach the object a harness
    actually holds. This drives the real ``Client.acquire_compose_project``
    call -- not a hand-built ``ClusterComposeSession(..., liveness_probe=...)``
    -- and checks the returned session reports the real keepalive state.
    """
    import asyncio

    class _FailableTransport(_FakeTransport):
        fail: bool = True

        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            if self.fail:
                raise RuntimeError("control plane unreachable")

    transport = _FailableTransport()
    client = Client(transport)  # type: ignore[arg-type]
    client._keepalive._interval_s = 0.02
    client._keepalive._beat_budget_s = 0.01

    session = await client.acquire_compose_project(
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app@sha256:abc"],
        footprint_cpu=1.0,
        footprint_mem_bytes=1024,
    )
    assert session.liveness_at_risk is False  # nothing has failed yet

    await asyncio.sleep(0.3)
    assert session.liveness_at_risk is True, (
        "compose session.liveness_at_risk did not go True on the real "
        "acquire_compose_project() path despite sustained heartbeat failure"
    )

    transport.fail = False
    await asyncio.sleep(0.2)
    assert session.liveness_at_risk is False, (
        "compose session.liveness_at_risk did not clear on the real "
        "acquire_compose_project() path after the keepalive recovered"
    )

    await session.destroy()
