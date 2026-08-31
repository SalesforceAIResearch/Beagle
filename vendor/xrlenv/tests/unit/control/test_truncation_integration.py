"""Integration test: hard-deadline expiry → coordinator → truncation flow.

Uses a FakeNodeAgent (no Docker) so we can drive a slow ``env_step`` past
the watcher's deadline without timing real container destroy. Verifies:

- The deadline watcher sets the truncate event mid-step
- The coordinator races env_step vs the event and raises RolloutTruncated
- Truncation skips env_teardown (sandbox dies; pinned thread goes with it)
- The trajectory seals as TRUNCATED with reason="hard_deadline"
- The deadline-watcher entry is cleaned up
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import ResourceSpec, SandboxHandle
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.scheduler import Placement
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.errors import RolloutTruncated
from xrlenv.types import Deadline, RolloutStatus


def _manifest(hard_s: float = 0.05) -> TemplateManifest:
    return TemplateManifest(
        name="t",
        version="0.1",
        digest="sha256:t",
        image="im/t:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
        hard_s_default=hard_s,
    )


class _FakeNode:
    def __init__(self) -> None:
        self.node_id = "fake"
        self.create_calls = 0
        self.destroy_calls = 0
        self.teardown_calls = 0
        self._slow_step_started = asyncio.Event()
        self.slow_step_seconds = 5.0  # block long past the deadline

    def supported_backends(self) -> list[str]:
        return ["docker"]

    async def create_sandbox(self, **_: Any) -> SandboxHandle:
        self.create_calls += 1
        return SandboxHandle(
            id=f"sb-{self.create_calls}",
            backend="docker",
            backend_ref="cid",
            stub_endpoint="tcp://127.0.0.1:0",
        )

    async def env_setup(self, _handle: Any, **__: Any) -> dict[str, Any]:
        return {"obs": "first"}

    async def env_step(
        self, _handle: Any, _action: Any, **__: Any,
    ) -> dict[str, Any]:
        self._slow_step_started.set()
        await asyncio.sleep(self.slow_step_seconds)
        return {"obs": {}, "reward": 0.0, "done": False, "truncated": False, "info": {}}

    async def env_teardown(self, _handle: Any, **__: Any) -> dict[str, Any]:
        self.teardown_calls += 1
        return {"status": "ok"}

    async def destroy_sandbox(self, _handle: Any) -> None:
        self.destroy_calls += 1

    async def query_image(self, _image: str) -> Any:
        # A1 / D19 (P1.2) — coordinator pre-flight check expects every
        # NodeTransport to answer ``query_image``. Default fakes
        # cover the happy path (image present); tests asserting the
        # absent-image branch construct a fake that overrides this.
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _make_coordinator(hard_s: float = 0.05) -> tuple[RolloutCoordinator, _FakeNode, InMemoryStateStore]:
    node = _FakeNode()
    catalog = TemplateCatalog()
    catalog.register(_manifest(hard_s=hard_s))

    sched = MagicMock()
    sched.place.return_value = Placement(node=node, backend="docker", score=1)  # type: ignore[arg-type]
    sched.nodes = [node]

    state = InMemoryStateStore()
    coord = RolloutCoordinator(
        catalog=catalog, scheduler=sched, state=state
    )
    return coord, node, state


# ──────────────────────────────────────────────────────────────────────────────
# Truncation flow
# ──────────────────────────────────────────────────────────────────────────────


async def test_hard_deadline_truncates_in_flight_step() -> None:
    coord, _node, state = _make_coordinator(hard_s=0.05)
    rid, _obs = await coord.start_rollout(
        template_name="t", init={}, deadline=Deadline(hard_s=0.05)
    )

    with pytest.raises(RolloutTruncated):
        await coord.step(rid, {"cmd": "noop"})

    record = state.get_rollout(rid)
    assert record.status == RolloutStatus.TRUNCATED
    assert record.reason == "hard_deadline"


async def test_truncation_skips_env_teardown() -> None:
    coord, node, _state = _make_coordinator(hard_s=0.05)
    rid, _obs = await coord.start_rollout(
        template_name="t", init={}, deadline=Deadline(hard_s=0.05)
    )
    with pytest.raises(RolloutTruncated):
        await coord.step(rid, {"cmd": "noop"})
    # spec 02: hard_deadline kills the sandbox unilaterally; teardown does
    # not run because the pinned env thread can't be safely preempted.
    assert node.teardown_calls == 0
    # ...but the sandbox WAS destroyed.
    assert node.destroy_calls == 1


async def test_deadline_watcher_cleared_after_truncation() -> None:
    coord, _node, _state = _make_coordinator(hard_s=0.05)
    rid, _obs = await coord.start_rollout(
        template_name="t", init={}, deadline=Deadline(hard_s=0.05)
    )
    with pytest.raises(RolloutTruncated):
        await coord.step(rid, {"cmd": "noop"})
    # No leftover watcher entry.
    assert coord.deadline_watcher.has_watcher(rid) is False


async def test_deadline_uses_manifest_default_when_unspecified() -> None:
    """Deadline=None → manifest's hard_s_default is honored."""
    coord, _node, state = _make_coordinator(hard_s=0.05)
    rid, _obs = await coord.start_rollout(template_name="t", init={})  # no deadline
    with pytest.raises(RolloutTruncated):
        await coord.step(rid, {"cmd": "noop"})
    record = state.get_rollout(rid)
    assert record.status == RolloutStatus.TRUNCATED


# ──────────────────────────────────────────────────────────────────────────────
# Normal-finish path is unaffected by the deadline machinery
# ──────────────────────────────────────────────────────────────────────────────


async def test_normal_finish_cancels_watcher_and_calls_teardown() -> None:
    coord, node, _state = _make_coordinator(hard_s=10.0)
    node.slow_step_seconds = 0.0  # fast steps; no truncation race
    rid, _obs = await coord.start_rollout(
        template_name="t", init={}, deadline=Deadline(hard_s=10.0)
    )
    await coord.step(rid, {"cmd": "noop"})
    traj = await coord.finish(rid)
    assert traj.status == RolloutStatus.FINISHED
    # Normal finish path: teardown DID run, sandbox destroyed exactly once.
    assert node.teardown_calls == 1
    assert node.destroy_calls == 1
    # Watcher removed.
    assert coord.deadline_watcher.has_watcher(rid) is False
