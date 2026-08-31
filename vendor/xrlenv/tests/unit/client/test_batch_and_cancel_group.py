"""Tests for Slice 4 consumer-SDK additions: cancel_group, batch_rollout,
end-to-end idle-TTL through the coordinator.

Uses an in-process Client + LocalRuntime backed by a fake-node FakeBackend
so we don't need Docker. The runtime's deadline + idle-TTL watchers are
real asyncio tasks; the coordinator's terminate path is real.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import (
    ExecChunk,
    ExecResult,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxBackend,
    SandboxCapabilities,
    SandboxHandle,
    ServiceSpec,
    SnapshotID,
    TemplateRef,
)
from xrlenv.client.client import Client
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.scheduler import Placement
from xrlenv.control.service import CoordinatorRolloutService
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.types import RolloutStatus

# ──────────────────────────────────────────────────────────────────────────────
# Fake node + helpers
# ──────────────────────────────────────────────────────────────────────────────


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _manifest(name: str = "t") -> TemplateManifest:
    return TemplateManifest(
        name=name, version="0.1", digest=f"sha256:{name}", image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
        hard_s_default=10.0,
    )


class _FakeNode:
    """Minimal NodeTransport stand-in. Issues unique sandbox ids and runs an
    EnvAdapter loop that returns done after `max_steps` steps.
    """

    def __init__(self, node_id: str = "fake") -> None:
        self.node_id = node_id
        self._created = 0
        self._max_steps_per_sb: dict[str, int] = {}
        self._steps_per_sb: dict[str, int] = {}

    def supported_backends(self) -> list[str]:
        return ["docker"]

    def hardware(self) -> HardwareInfo:
        return _hw()

    async def create_sandbox(self, **_: Any) -> SandboxHandle:
        self._created += 1
        sid = f"sb-{self._created}"
        return SandboxHandle(
            id=sid, backend="docker", backend_ref=f"cid-{self._created}",
            stub_endpoint="tcp://127.0.0.1:0",
        )

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        self._max_steps_per_sb.pop(sb.id, None)
        self._steps_per_sb.pop(sb.id, None)

    async def env_setup(
        self,
        sb: SandboxHandle,
        *,
        adapter_module: str,
        adapter_class: str,
        init_params: dict[str, Any],
        **_kw: Any,
    ) -> dict[str, Any]:
        self._max_steps_per_sb[sb.id] = int(init_params.get("max_steps") or 1)
        self._steps_per_sb[sb.id] = 0
        return {"obs": {"kind": "first"}}

    async def env_step(
        self, sb: SandboxHandle, action: Any, **_kw: Any,
    ) -> dict[str, Any]:
        self._steps_per_sb[sb.id] += 1
        done = self._steps_per_sb[sb.id] >= self._max_steps_per_sb[sb.id]
        return {
            "obs": {"step": self._steps_per_sb[sb.id]},
            "reward": 1.0,
            "done": done,
            "truncated": False,
            "info": {},
        }

    async def env_teardown(self, sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
        return {"status": "ok"}

    async def run_in_sandbox(
        self, sb: SandboxHandle, cmd: list[str], **_: Any
    ) -> ExecResult:
        return ExecResult(exit_code=0)

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=0.0, rss_bytes=0, disk_bytes=0, rx_bytes=0, tx_bytes=0)

    async def query_image(self, _image: str) -> Any:
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _build_runtime(*, default_idle_ttl_s: float = 120.0) -> tuple[Client, RolloutCoordinator, InMemoryStateStore, _FakeNode]:
    node = _FakeNode()
    catalog = TemplateCatalog()
    catalog.register(_manifest())

    sched = MagicMock()
    sched.place.return_value = Placement(node=node, backend="docker", score=1)  # type: ignore[arg-type]
    sched.nodes = [node]

    state = InMemoryStateStore()
    admission = AdmissionQueue(scheduler=sched, state=state)
    coordinator = RolloutCoordinator(
        catalog=catalog,
        scheduler=sched,
        state=state,
        admission=admission,
        default_idle_ttl_s=default_idle_ttl_s,
    )
    service = CoordinatorRolloutService(coordinator)
    client = Client.in_process(service)
    return client, coordinator, state, node


# ──────────────────────────────────────────────────────────────────────────────
# cancel_group
# ──────────────────────────────────────────────────────────────────────────────


async def test_cancel_group_seals_matching_rollouts() -> None:
    client, coord, state, _node = _build_runtime()
    sessions = []
    for _ in range(3):
        s = await client.rollout(
            template="t", init={"max_steps": 100}, group_id="grp-A"
        )
        sessions.append(s)
    # One in a different group — should NOT be cancelled.
    other = await client.rollout(
        template="t", init={"max_steps": 100}, group_id="grp-B"
    )

    report = await client.cancel_group("grp-A", reason="test")

    assert sorted(report.cancelled) == sorted(s.rollout_id for s in sessions)
    assert report.already_terminal == ()

    for s in sessions:
        rec = state.get_rollout(s.rollout_id)
        assert rec.status == RolloutStatus.CANCELLED
        assert rec.reason == "test"

    # The unrelated-group rollout is untouched.
    assert state.get_rollout(other.rollout_id).status == RolloutStatus.RUNNING

    # Drain by cancelling the orphan + close the coordinator's watchers cleanly.
    await client.cancel_rollout(other.rollout_id)
    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_cancel_group_reports_already_terminal() -> None:
    client, coord, _state, _node = _build_runtime()
    s = await client.rollout(template="t", init={"max_steps": 100}, group_id="grp")
    await client.cancel_rollout(s.rollout_id)  # pre-terminate

    report = await client.cancel_group("grp", reason="late")
    assert report.cancelled == ()
    assert report.already_terminal == (s.rollout_id,)

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_cancel_group_empty_group_returns_empty_report() -> None:
    client, coord, _state, _node = _build_runtime()
    report = await client.cancel_group("nobody-here", reason="x")
    assert report.cancelled == ()
    assert report.already_terminal == ()

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


# ──────────────────────────────────────────────────────────────────────────────
# batch_rollout
# ──────────────────────────────────────────────────────────────────────────────


async def test_batch_rollout_buckets_finished() -> None:
    client, coord, _state, _node = _build_runtime()

    async def policy(_obs: Any) -> Any:
        return {"cmd": "noop"}

    inits = [{"max_steps": 2} for _ in range(4)]
    result = await client.batch_rollout(
        template="t", inits=inits, policy=policy, concurrency=2
    )
    assert len(result.finished) == 4
    assert result.truncated == []
    assert result.failed == []
    for traj in result.finished:
        assert traj.status == RolloutStatus.FINISHED
        assert pytest.approx(traj.final_reward) == 2.0  # 2 steps x reward=1.0

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_batch_rollout_buckets_idle_ttl_truncation_as_truncated() -> None:
    """Audit M1 (commit 1c27026): a rollout reaped by idle TTL between steps
    must surface in result.truncated, not result.failed.

    Reproduction: idle_ttl_s=0.05; policy sleeps 0.1s before each action so
    the watcher fires while the consumer is computing the next move. The
    next session.step() hits a TRUNCATED rollout and Coordinator.step()
    must raise RolloutTruncated (not RolloutFailed/not_running) so
    batch_rollout's exception bucketing places it in `truncated`.
    """
    client, coord, _state, _node = _build_runtime(default_idle_ttl_s=0.05)

    async def slow_policy(_obs: Any) -> Any:
        await asyncio.sleep(0.1)
        return {"cmd": "noop"}

    result = await client.batch_rollout(
        template="t",
        inits=[{"max_steps": 100}],
        policy=slow_policy,
        concurrency=1,
    )
    assert len(result.truncated) == 1, (
        f"expected 1 truncated, got finished={len(result.finished)} "
        f"truncated={len(result.truncated)} failed={len(result.failed)}"
    )
    assert result.failed == []
    assert result.finished == []
    sealed = result.truncated[0]
    assert sealed.status == RolloutStatus.TRUNCATED
    assert sealed.reason == "idle_ttl"

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_batch_rollout_validates_annotation_lengths() -> None:
    client, coord, _state, _node = _build_runtime()

    async def policy(_obs: Any) -> Any:
        return {}

    with pytest.raises(ValueError, match="task_keys length"):
        await client.batch_rollout(
            template="t",
            inits=[{}, {}],
            policy=policy,
            task_keys=["only-one"],
        )

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_batch_rollout_passes_group_ids_through_to_records() -> None:
    client, coord, state, _node = _build_runtime()

    async def policy(_obs: Any) -> Any:
        return {}

    await client.batch_rollout(
        template="t",
        inits=[{"max_steps": 1}, {"max_steps": 1}],
        policy=policy,
        group_ids=["grp-X", "grp-X"],
        task_keys=["task-A", "task-B"],
        concurrency=2,
    )
    groups = sorted(r.group_id for r in state.list_rollouts() if r.group_id)
    assert groups == ["grp-X", "grp-X"]
    keys = sorted(r.task_key for r in state.list_rollouts() if r.task_key)
    assert keys == ["task-A", "task-B"]

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


# ──────────────────────────────────────────────────────────────────────────────
# Idle TTL through the coordinator
# ──────────────────────────────────────────────────────────────────────────────


async def test_idle_ttl_reaps_abandoned_rollout() -> None:
    """Coordinator arms idle-TTL on start_rollout; if the consumer never steps
    or heartbeats, the watcher reaps the rollout as truncated/idle_ttl.
    """
    client, coord, state, _node = _build_runtime(default_idle_ttl_s=0.1)
    s = await client.rollout(template="t", init={"max_steps": 100})

    # Wait past the idle window without touching.
    await asyncio.sleep(0.3)

    rec = state.get_rollout(s.rollout_id)
    assert rec.status == RolloutStatus.TRUNCATED
    assert rec.reason == "idle_ttl"
    assert coord.idle_ttl_watcher.has_watcher(s.rollout_id) is False

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_session_heartbeat_resets_idle_ttl() -> None:
    """Explicit Session.heartbeat() keeps the rollout alive past the window."""
    client, coord, state, _node = _build_runtime(default_idle_ttl_s=0.15)
    s = await client.rollout(template="t", init={"max_steps": 100})

    # Heartbeat every 50ms for ~300ms — should never let the 150ms idle
    # window elapse.
    for _ in range(6):
        await asyncio.sleep(0.05)
        await s.heartbeat()

    rec = state.get_rollout(s.rollout_id)
    assert rec.status == RolloutStatus.RUNNING

    # Stop touching; let the watcher fire.
    await asyncio.sleep(0.3)
    rec = state.get_rollout(s.rollout_id)
    assert rec.status == RolloutStatus.TRUNCATED
    assert rec.reason == "idle_ttl"

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_step_implicitly_resets_idle_ttl() -> None:
    """Each step counts as a touch (spec 02 §"Consumer-side heartbeat")."""
    client, coord, state, _node = _build_runtime(default_idle_ttl_s=0.15)
    s = await client.rollout(template="t", init={"max_steps": 100})

    # Step every 50ms for ~300ms — implicit touch each time.
    for _ in range(6):
        await asyncio.sleep(0.05)
        await s.step({})

    rec = state.get_rollout(s.rollout_id)
    assert rec.status == RolloutStatus.RUNNING

    # Cancel cleanly to free the deadline watcher.
    await client.cancel_rollout(s.rollout_id)
    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


# ──────────────────────────────────────────────────────────────────────────────
# Fake-backend dummy (kept here to placate any missed wiring)
# ──────────────────────────────────────────────────────────────────────────────


class _UnusedBackend(SandboxBackend):
    """Defensive stub so the test module imports SandboxBackend cleanly even
    though the FakeNode doesn't reach into the backend layer at all.
    """

    name = "_unused"
    capabilities = SandboxCapabilities(
        supports_snapshot=False,
        supports_chainable_snapshot=False,
        live_state_captured=False,
        supports_port_forward=False,
        supports_gpu=False,
        isolation_class="none",
        fast_create_p50_ms=0,
    )

    async def create(
        self,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
    ) -> SandboxHandle:
        raise NotImplementedError

    async def destroy(self, sb: SandboxHandle) -> None:
        raise NotImplementedError

    def exec(
        self,
        sb: SandboxHandle,
        cmd: list[str],
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[ExecChunk]:
        raise NotImplementedError

    async def read_file(self, sb: SandboxHandle, path: str) -> bytes:
        raise NotImplementedError

    async def write_file(self, sb: SandboxHandle, path: str, data: bytes) -> None:
        raise NotImplementedError

    def read_file_stream(self, sb: SandboxHandle, path: str) -> AsyncIterator[bytes]:
        raise NotImplementedError

    async def write_file_stream(
        self, sb: SandboxHandle, path: str, src: AsyncIterator[bytes]
    ) -> None:
        raise NotImplementedError

    async def spawn_service(self, sb: SandboxHandle, spec: ServiceSpec) -> object:
        raise NotImplementedError

    async def spawn_services(
        self, sb: SandboxHandle, specs: list[ServiceSpec]
    ) -> list[object]:
        raise NotImplementedError

    async def port_forward(self, sb: SandboxHandle, internal_port: int) -> str:
        raise NotImplementedError

    async def snapshot(self, sb: SandboxHandle) -> SnapshotID:
        raise NotImplementedError

    async def restore(self, snapshot: SnapshotID) -> SandboxHandle:
        raise NotImplementedError

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        raise NotImplementedError
