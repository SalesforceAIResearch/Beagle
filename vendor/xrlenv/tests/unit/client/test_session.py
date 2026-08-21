"""Tests for RolloutSession against an in-memory fake transport."""

from __future__ import annotations

import pytest
from xrlenv.client.session import RolloutSession
from xrlenv.client.transport import ClientTransport
from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse
from xrlenv.types import Action, RolloutStatus, StepResult, Trajectory


class FakeTransport:
    """Records calls + returns scripted StepResults."""

    def __init__(self, scripted: list[StepResult]) -> None:
        self._scripted = list(scripted)
        self.finish_calls = 0
        self.cancel_calls = 0
        self.cancel_reasons: list[str] = []

    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
        return StartRolloutResponse(rollout_id="r1", init_obs={"hi": True})

    async def step(self, rollout_id: str, action: Action) -> StepResult:
        return self._scripted.pop(0)

    async def finish(self, rollout_id: str) -> Trajectory:
        self.finish_calls += 1
        return Trajectory(
            rollout_id=rollout_id,
            template="t",
            steps=[],
            status=RolloutStatus.FINISHED,
            reason=None,
            final_reward=0.0,
        )

    async def cancel(self, rollout_id: str, reason: str) -> Trajectory:
        self.cancel_calls += 1
        self.cancel_reasons.append(reason)
        return Trajectory(
            rollout_id=rollout_id,
            template="t",
            steps=[],
            status=RolloutStatus.CANCELLED,
            reason=reason,
            final_reward=0.0,
        )

    async def close(self) -> None:
        return None


def _session(transport: ClientTransport) -> RolloutSession:
    return RolloutSession(
        transport=transport,
        rollout_id="r1",
        initial_obs={"hi": True},
        template="t",
    )


async def test_done_after_step_returning_done() -> None:
    t = FakeTransport([StepResult(obs={"x": 1}, reward=0.5, done=True)])
    s = _session(t)
    async with s:
        result = await s.step({"cmd": "noop"})
        assert result.done is True
        assert s.done is True
    assert t.finish_calls == 1


async def test_truncated_marks_done() -> None:
    t = FakeTransport([StepResult(obs={}, reward=0.0, truncated=True)])
    s = _session(t)
    async with s:
        await s.step("x")
        assert s.truncated is True
        assert s.done is True


async def test_step_after_done_raises() -> None:
    t = FakeTransport([StepResult(obs={}, reward=0.0, done=True)])
    s = _session(t)
    async with s:
        await s.step("x")
        with pytest.raises(RuntimeError):
            await s.step("y")


async def test_exception_in_with_block_triggers_cancel() -> None:
    t = FakeTransport([StepResult(obs={}, reward=0.0)])
    s = _session(t)
    with pytest.raises(RuntimeError):
        async with s:
            raise RuntimeError("boom")
    assert t.cancel_calls == 1
    # The session captures ``<type>: <message>`` in the reason so the
    # admin page / replay caller can see the cause without spelunking
    # coordinator.log. Match by prefix + content rather than equality
    # so future exception-message tweaks don't break this test.
    assert len(t.cancel_reasons) == 1
    assert t.cancel_reasons[0].startswith("aborted_with_exception: RuntimeError")
    assert "boom" in t.cancel_reasons[0]


async def test_trajectory_property_before_close_raises() -> None:
    t = FakeTransport([])
    s = _session(t)
    with pytest.raises(RuntimeError):
        _ = s.trajectory


async def test_reward_sum_accumulates() -> None:
    t = FakeTransport(
        [
            StepResult(obs={}, reward=0.4),
            StepResult(obs={}, reward=0.3, done=True),
        ]
    )
    s = _session(t)
    async with s:
        await s.step("a")
        await s.step("b")
    assert pytest.approx(s.reward_sum) == 0.7
    assert s.steps_taken == 2


async def test_idempotent_finish() -> None:
    t = FakeTransport([])
    s = _session(t)
    await s.finish()
    await s.finish()
    assert t.finish_calls == 1
