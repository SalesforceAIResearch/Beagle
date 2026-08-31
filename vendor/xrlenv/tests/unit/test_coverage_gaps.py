"""Additional unit tests closing coverage gaps found in Slice 1.

Covers paths not exercised by the prior 134 tests:
- coordinator._step_result_from_payload defensive None-reward handling
- coordinator.step when sandbox_id is missing
- coordinator._node_for when node disappears from scheduler
- coordinator: scheduler placement failure propagation
- state: seal_trajectory on unknown rollout_id
- state: update_sandbox on unknown sandbox_id
- state: list_rollouts with multiple records
- envs/base: SyncEnvAdapter step timeout via __step_timeout_s envelope
- envs/base: SyncEnvAdapter _do_teardown closes env.close()
- sandbox_stub/server: commands with env dict passthrough
- sandbox_stub/server: StubServer constructor validation
- stub_client: higher-level env_setup / env_step / env_teardown / commands
- node/agent: stats method dispatch
- client: Client as context manager; rollout with Deadline merges init_params
- client: Client.rollout passes request_id / task_key / group_id
- service: CoordinatorRolloutService thin wrapper (happy path)
- types: Step dataclass field access
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from xrlenv.backends.base import ResourceSpec, SandboxHandle
from xrlenv.client.client import Client
from xrlenv.control.coordinator import (
    RolloutCoordinator,
    _step_result_from_payload,
)
from xrlenv.control.service import (
    CoordinatorRolloutService,
    StartRolloutRequest,
    StartRolloutResponse,
)
from xrlenv.control.state import InMemoryStateStore, RolloutRecord
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.envs.base import StepTimeout, SyncEnvAdapter
from xrlenv.errors import BackendCapabilityMissing, RolloutFailed
from xrlenv.node.agent import NodeAgent, NodeAgentConfig
from xrlenv.node.stub_client import StubClient
from xrlenv.sandbox_stub.server import StubServer, build_app
from xrlenv.types import Action, Deadline, Observation, RolloutStatus, Step, StepResult

# ── Shared helpers ─────────────────────────────────────────────────────────────


def _make_manifest(name: str = "t", backend: str = "docker") -> TemplateManifest:
    return TemplateManifest(
        name=name,
        version="0.1",
        digest="sha256:abc",
        image="im:1",

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


class FakeNodeAgent:
    node_id: str = "fake-node"

    def __init__(
        self,
        *,
        raise_on_setup: Exception | None = None,
        raise_on_create: Exception | None = None,
    ) -> None:
        self._raise_on_setup = raise_on_setup
        self._raise_on_create = raise_on_create
        self.destroyed: list[str] = []
        self._sandboxes: dict[str, Any] = {}

    def supported_backends(self) -> list[str]:
        return ["docker"]

    async def create_sandbox(self, **_kw: Any) -> SandboxHandle:
        if self._raise_on_create is not None:
            raise self._raise_on_create
        return SandboxHandle(
            id="sb-1", backend="docker", backend_ref="ctr", stub_endpoint="tcp://127.0.0.1:9"
        )

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        self.destroyed.append(sb.id)

    async def env_setup(self, _sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
        if self._raise_on_setup is not None:
            raise self._raise_on_setup
        return {"obs": "hello"}

    async def env_step(
        self, _sb: SandboxHandle, action: Any, **_kw: Any,
    ) -> dict[str, Any]:
        return {"obs": action, "reward": 0.0, "done": False, "truncated": False, "info": {}}

    async def env_teardown(self, _sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
        return {"status": "ok"}

    async def _stub_for(self, _sb: SandboxHandle) -> Any:
        m = MagicMock()
        m.commands = AsyncMock(return_value={"exit_code": 0})
        return m

    async def query_image(self, _image: str) -> Any:
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _make_coordinator(
    agent: FakeNodeAgent | None = None,
    manifest: TemplateManifest | None = None,
) -> tuple[RolloutCoordinator, InMemoryStateStore]:
    if agent is None:
        agent = FakeNodeAgent()
    if manifest is None:
        manifest = _make_manifest()
    catalog = TemplateCatalog()
    catalog.register(manifest)
    from xrlenv.control.scheduler import Placement

    sched = MagicMock()
    sched.place.return_value = Placement(node=agent, backend="docker", score=1)
    sched.nodes = [agent]
    state = InMemoryStateStore()
    return RolloutCoordinator(catalog=catalog, scheduler=sched, state=state), state


# ── coordinator._step_result_from_payload ─────────────────────────────────────


def test_step_result_from_payload_none_reward_becomes_zero() -> None:
    """None reward (e.g. not-yet-computed field) must not cause TypeError.
    The happy-path mapping + missing-key defaults are exercised end-to-end by
    test_coordinator.py via real step calls; only this defensive None-handling
    is worth a dedicated unit test."""
    r = _step_result_from_payload({"reward": None})
    assert r.reward == pytest.approx(0.0)


# ── coordinator.step — sandbox_id missing ─────────────────────────────────────


async def test_step_raises_when_sandbox_id_is_none() -> None:
    """A RUNNING rollout record with no sandbox_id is a buggy state; step must raise."""
    coord, state = _make_coordinator()
    rid, _ = await coord.start_rollout(template_name="t", init={})
    # Manually clear sandbox_id to simulate a corrupted record.
    state.update_rollout(rid, sandbox_id=None)

    with pytest.raises(RolloutFailed, match="no sandbox bound"):
        await coord.step(rid, "action")


# ── coordinator._node_for — node disappeared ──────────────────────────────────


async def test_step_raises_when_node_disappears() -> None:
    """If the node disappears from the scheduler after rollout start, step must raise."""
    agent = FakeNodeAgent()
    coord, _state = _make_coordinator(agent=agent)
    rid, _ = await coord.start_rollout(template_name="t", init={})

    # Remove the node from the scheduler's node list.
    coord._scheduler.nodes = []

    with pytest.raises(RuntimeError, match="disappeared from scheduler"):
        await coord.step(rid, "action")


# ── coordinator: scheduler placement failure ──────────────────────────────────


async def test_start_rollout_propagates_scheduler_placement_failure() -> None:
    """BackendCapabilityMissing from scheduler.place propagates to the caller.

    NOTE: The coordinator does NOT catch scheduler errors in start_rollout —
    the scheduler.place() call happens *before* the try/except block wrapping
    _bootstrap_sandbox. So the raw BackendCapabilityMissing propagates, not
    a RolloutFailed. This documents the current contract.
    """
    catalog = TemplateCatalog()
    catalog.register(_make_manifest())
    sched = MagicMock()
    sched.place.side_effect = BackendCapabilityMissing("no docker node")
    sched.nodes = []
    state = InMemoryStateStore()
    coord = RolloutCoordinator(catalog=catalog, scheduler=sched, state=state)

    with pytest.raises(BackendCapabilityMissing):
        await coord.start_rollout(template_name="t", init={})


# ── InMemoryStateStore gaps ────────────────────────────────────────────────────


def test_seal_trajectory_raises_for_unknown_rollout() -> None:
    store = InMemoryStateStore()
    with pytest.raises(KeyError):
        store.seal_trajectory("does-not-exist")


def test_update_sandbox_raises_for_unknown_sandbox() -> None:
    store = InMemoryStateStore()
    with pytest.raises(KeyError):
        store.update_sandbox("ghost-sandbox", status="destroyed")


def test_list_rollouts_returns_all_records() -> None:
    store = InMemoryStateStore()
    for i in range(3):
        store.insert_rollout(
            RolloutRecord(rollout_id=f"r{i}", template="t", status=RolloutStatus.RUNNING)
        )
    assert len(store.list_rollouts()) == 3


# ── SyncEnvAdapter — step timeout via __step_timeout_s envelope ───────────────


class _SlowSyncAdapter(SyncEnvAdapter):
    """Wraps a sync env that sleeps; used to test timeout enforcement."""

    def _do_setup(self, init_params: dict[str, Any]) -> Observation:
        return {"kind": "slow.greeting"}

    def _do_step(self, action: Action) -> StepResult:
        import time

        sleep_s = action.get("sleep_s", 0.0) if isinstance(action, dict) else 0.0
        time.sleep(sleep_s)
        return StepResult(obs={"done": False})


async def test_sync_env_adapter_step_timeout_raises_step_timeout() -> None:
    adapter = _SlowSyncAdapter(sandbox_id="sb-timeout-test")
    await adapter.setup({"setup_timeout_s": 5.0})
    # Inject a very short per-step timeout via the envelope.
    action = {"sleep_s": 5.0, "__step_timeout_s": 0.05}
    with pytest.raises(StepTimeout):
        await adapter.step(action)
    await adapter.teardown()


async def test_sync_env_adapter_step_uses_default_timeout_when_no_envelope() -> None:
    """A fast step without __step_timeout_s must complete normally."""
    adapter = _SlowSyncAdapter(sandbox_id="sb-fast-test")
    await adapter.setup({"setup_timeout_s": 5.0})
    result = await adapter.step({"sleep_s": 0.0})
    assert result.obs == {"done": False}
    await adapter.teardown()


# ── SyncEnvAdapter _do_teardown closes env ────────────────────────────────────


class _ClosableEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ClosingAdapter(SyncEnvAdapter):
    def __init__(self) -> None:
        super().__init__(sandbox_id="close-test")
        self._closable = _ClosableEnv()

    def _do_setup(self, init_params: dict[str, Any]) -> Observation:
        self._state.env = self._closable
        return {}

    def _do_step(self, action: Action) -> StepResult:
        return StepResult(obs={})


async def test_sync_env_adapter_teardown_calls_env_close() -> None:
    adapter = _ClosingAdapter()
    await adapter.setup({})
    await adapter.teardown()
    assert adapter._closable.closed is True


# ── sandbox_stub/server: /commands with env dict ──────────────────────────────


@pytest.fixture
async def stub_client() -> TestClient:  # type: ignore[return]
    app = build_app()
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_commands_with_env_passthrough(stub_client: TestClient) -> None:
    resp = await stub_client.post(
        "/commands",
        json={"cmd": ["sh", "-c", "echo $MY_VAR"], "env": {"MY_VAR": "xrlenv_test"}, "cwd": "/tmp"},
    )
    body = await resp.json()
    assert body["exit_code"] == 0
    assert "xrlenv_test" in body["stdout"]


# ── sandbox_stub/server: StubServer constructor validation ────────────────────


def test_stub_server_requires_uds_or_port() -> None:
    with pytest.raises(ValueError, match="uds_path or bind_port"):
        StubServer()


def test_stub_server_rejects_both_uds_and_port() -> None:
    with pytest.raises(ValueError, match="not both"):
        StubServer(uds_path="/tmp/x.sock", bind_port=9999)


def test_stub_server_uds_only_is_valid() -> None:
    server = StubServer(uds_path="/tmp/test.sock")
    assert server._uds_path == "/tmp/test.sock"


def test_stub_server_tcp_only_is_valid() -> None:
    server = StubServer(bind_port=8888)
    assert server._bind_port == 8888


# ── StubClient higher-level methods ──────────────────────────────────────────


def _build_full_stub_app() -> web.Application:
    """Mini aiohttp app that simulates the stub's env endpoints."""
    app = web.Application()

    async def healthz(_req: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def env_setup(req: web.Request) -> web.Response:
        body = await req.json()
        return web.json_response(
            {"obs": {"kind": "test.greeting"}, "capabilities": {}, "sandbox_id": body.get("sandbox_id")}
        )

    async def env_step(req: web.Request) -> web.Response:
        body = await req.json()
        return web.json_response(
            {"obs": body.get("action"), "reward": 0.0, "done": False, "truncated": False, "info": {}}
        )

    async def env_teardown(_req: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def commands(req: web.Request) -> web.Response:
        body = await req.json()
        return web.json_response(
            {"exit_code": 0, "stdout": " ".join(body.get("cmd", [])), "stderr": "", "timed_out": False}
        )

    app.router.add_get("/healthz", healthz)
    app.router.add_post("/env/setup", env_setup)
    app.router.add_post("/env/step", env_step)
    app.router.add_post("/env/teardown", env_teardown)
    app.router.add_post("/commands", commands)
    return app


async def test_stub_client_env_setup() -> None:
    async with TestClient(TestServer(_build_full_stub_app())) as tc:
        endpoint = f"tcp://127.0.0.1:{tc.port}"
        async with StubClient(endpoint) as client:
            result = await client.env_setup("m", "C", {}, sandbox_id="sb-1")
        assert result["obs"]["kind"] == "test.greeting"


async def test_stub_client_env_step() -> None:
    async with TestClient(TestServer(_build_full_stub_app())) as tc:
        endpoint = f"tcp://127.0.0.1:{tc.port}"
        async with StubClient(endpoint) as client:
            result = await client.env_step({"cmd": "ls"})
        assert result["obs"] == {"cmd": "ls"}


async def test_stub_client_env_teardown() -> None:
    async with TestClient(TestServer(_build_full_stub_app())) as tc:
        endpoint = f"tcp://127.0.0.1:{tc.port}"
        async with StubClient(endpoint) as client:
            result = await client.env_teardown()
        assert result["status"] == "ok"


async def test_stub_client_commands() -> None:
    async with TestClient(TestServer(_build_full_stub_app())) as tc:
        endpoint = f"tcp://127.0.0.1:{tc.port}"
        async with StubClient(endpoint) as client:
            result = await client.commands(["echo", "hi"], timeout_s=5.0)
        assert result["exit_code"] == 0
        assert "echo" in result["stdout"]


# ── node/agent: stats method ──────────────────────────────────────────────────


class _StatsBackend:
    name = "docker"

    async def create(self, *_: Any, **__: Any) -> SandboxHandle:
        return SandboxHandle(id="s", backend="docker", backend_ref="r", stub_endpoint="tcp://127.0.0.1:1")

    async def destroy(self, _sb: SandboxHandle) -> None:
        pass

    async def stats(self, _sb: SandboxHandle) -> Any:
        from xrlenv.backends.base import ResourceUsage

        return ResourceUsage(cpu_seconds=3.0, rss_bytes=2048, disk_bytes=0, rx_bytes=0, tx_bytes=0)


async def test_node_agent_stats_delegates_to_backend() -> None:
    cfg = NodeAgentConfig(node_id="n1", backends={"docker": _StatsBackend()})  # type: ignore[arg-type]
    agent = NodeAgent(cfg)
    handle = SandboxHandle(id="s", backend="docker", backend_ref="r", stub_endpoint="tcp://127.0.0.1:1")
    usage = await agent.stats(handle)
    assert usage.cpu_seconds == pytest.approx(3.0)
    assert usage.rss_bytes == 2048


# ── Client context manager and Deadline merging ───────────────────────────────


class _FakeService:
    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
        self.last_req = req
        return StartRolloutResponse(rollout_id="r-fake", init_obs={"hi": True})

    async def step(self, rollout_id: str, action: object) -> StepResult:
        return StepResult(obs={}, reward=0.0, done=True)

    async def finish(self, rollout_id: str) -> Any:
        from xrlenv.types import Trajectory

        return Trajectory(
            rollout_id=rollout_id, template="t", steps=[], status=RolloutStatus.FINISHED,
            reason=None, final_reward=0.0
        )

    async def cancel(self, rollout_id: str, reason: str) -> Any:
        from xrlenv.types import Trajectory

        return Trajectory(
            rollout_id=rollout_id, template="t", steps=[], status=RolloutStatus.CANCELLED,
            reason=reason, final_reward=0.0
        )


async def test_client_as_context_manager_calls_close() -> None:
    svc = _FakeService()
    closed = False
    client = Client.in_process(svc)

    original_close = client.close

    async def patched_close() -> None:
        nonlocal closed
        closed = True
        await original_close()

    client.close = patched_close  # type: ignore[method-assign]
    async with client:
        pass
    assert closed is True


async def test_client_rollout_with_deadline_merges_init_params() -> None:
    svc = _FakeService()
    client = Client.in_process(svc)
    deadline = Deadline(hard_s=60.0, step_timeout_s=10.0, setup_timeout_s=20.0)

    await client.rollout("t", init={"user_key": "val"}, deadline=deadline)
    req = svc.last_req
    assert req.init.get("step_timeout_s") == 10.0
    assert req.init.get("setup_timeout_s") == 20.0
    assert req.init.get("user_key") == "val"


async def test_client_rollout_passes_identifiers() -> None:
    svc = _FakeService()
    client = Client.in_process(svc)
    await client.rollout(
        "t", request_id="req-42", task_key="task-key", group_id="group-1"
    )
    req = svc.last_req
    assert req.request_id == "req-42"
    assert req.task_key == "task-key"
    assert req.group_id == "group-1"


# ── CoordinatorRolloutService thin wrapper ────────────────────────────────────


async def test_coordinator_service_start_rollout_delegates() -> None:
    from unittest.mock import MagicMock

    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateManifest,
    )

    coord_mock = AsyncMock()
    coord_mock.start_rollout.return_value = ("r-1", {"obs": "first"})
    # Slice 4.5: service.start_rollout looks up the template's reward.mode
    # from the coordinator's catalog so the SDK can validate consumer_final
    # presence at call time.
    manifest = TemplateManifest(
        name="t", version="0.1", digest="sha256:t", image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )
    catalog_mock = MagicMock()
    catalog_mock.get.return_value = manifest
    type(coord_mock).catalog = catalog_mock
    svc = CoordinatorRolloutService(coord_mock)  # type: ignore[arg-type]

    req = StartRolloutRequest(template="t", init={"k": "v"}, request_id="req-x")
    resp = await svc.start_rollout(req)
    assert resp.rollout_id == "r-1"
    assert resp.init_obs == {"obs": "first"}
    assert resp.reward_mode == "env_step"
    coord_mock.start_rollout.assert_awaited_once_with(
        template_name="t", init={"k": "v"}, request_id="req-x",
        task_key=None, group_id=None, owner_id="default", deadline=None,
        backend=None, network=None,
    )


async def test_coordinator_service_step_delegates() -> None:
    coord_mock = AsyncMock()
    coord_mock.step.return_value = StepResult(obs={"x": 1}, reward=0.5, done=False)
    svc = CoordinatorRolloutService(coord_mock)  # type: ignore[arg-type]

    result = await svc.step("r-1", "my-action")
    assert result.reward == pytest.approx(0.5)
    coord_mock.step.assert_awaited_once_with("r-1", "my-action")


# ── types.Step dataclass ──────────────────────────────────────────────────────


def test_step_dataclass_stores_all_fields() -> None:
    s = Step(
        index=3,
        action={"cmd": "ls"},
        obs={"stdout": "file.txt"},
        reward=1.0,
        done=True,
        truncated=False,
        info={"steps": 3},
        ts=0.42,
    )
    assert s.index == 3
    assert s.action == {"cmd": "ls"}
    assert s.reward == pytest.approx(1.0)
    assert s.done is True
    assert s.ts == pytest.approx(0.42)
