"""Tests for xrlenv/types.py — RolloutStatus, Deadline, StepResult, Trajectory."""

from __future__ import annotations

import pytest
from xrlenv.types import Deadline, RolloutStatus, Step, StepResult, Trajectory

# ── RolloutStatus.is_terminal ─────────────────────────────────────────────────

@pytest.mark.parametrize("status", [
    RolloutStatus.FINISHED,
    RolloutStatus.TRUNCATED,
    RolloutStatus.CANCELLED,
    RolloutStatus.FAILED,
])
def test_is_terminal_returns_true_for_terminal_statuses(status: RolloutStatus) -> None:
    assert status.is_terminal is True


@pytest.mark.parametrize("status", [
    RolloutStatus.QUEUED,
    RolloutStatus.STARTING,
    RolloutStatus.RUNNING,
    RolloutStatus.CANCELLING,
    RolloutStatus.FINISHING,
])
def test_is_terminal_returns_false_for_transient_statuses(status: RolloutStatus) -> None:
    assert status.is_terminal is False


def test_rollout_status_is_str_enum() -> None:
    assert RolloutStatus.RUNNING == "running"
    assert str(RolloutStatus.FINISHED) == "finished"


# ── Deadline ──────────────────────────────────────────────────────────────────

def test_deadline_requires_hard_s() -> None:
    d = Deadline(hard_s=60.0)
    assert d.hard_s == 60.0
    assert d.soft_s is None
    assert d.step_timeout_s is None


def test_deadline_with_all_overrides() -> None:
    d = Deadline(
        hard_s=300.0,
        soft_s=200.0,
        setup_timeout_s=30.0,
        step_timeout_s=10.0,
        teardown_timeout_s=5.0,
    )
    assert d.soft_s == 200.0
    assert d.setup_timeout_s == 30.0
    assert d.step_timeout_s == 10.0


def test_deadline_is_frozen() -> None:
    from pydantic import ValidationError

    d = Deadline(hard_s=10.0)
    with pytest.raises(ValidationError):
        d.hard_s = 20.0  # type: ignore[misc]


# ── StepResult ───────────────────────────────────────────────────────────────

def test_step_result_defaults() -> None:
    r = StepResult(obs={"x": 1})
    assert r.reward == 0.0
    assert r.done is False
    assert r.truncated is False
    assert r.info == {}


def test_step_result_with_all_fields() -> None:
    r = StepResult(obs=None, reward=1.5, done=True, truncated=False, info={"k": "v"})
    assert r.reward == 1.5
    assert r.done is True
    assert r.info == {"k": "v"}


# ── Trajectory ───────────────────────────────────────────────────────────────

def test_trajectory_steps_list() -> None:
    step = Step(
        index=0, action="act", obs={"o": 1},
        reward=0.5, done=False, truncated=False, info={}, ts=0.1,
    )
    traj = Trajectory(
        rollout_id="r1",
        template="t",
        steps=[step],
        status=RolloutStatus.FINISHED,
        reason=None,
        final_reward=0.5,
    )
    assert len(traj.steps) == 1
    assert traj.steps[0].index == 0
    assert traj.final_reward == 0.5
    assert traj.metadata == {}


def test_trajectory_metadata_defaults_to_empty_dict() -> None:
    traj = Trajectory(
        rollout_id="r2",
        template="t",
        steps=[],
        status=RolloutStatus.CANCELLED,
        reason="consumer_cancelled",
        final_reward=0.0,
    )
    assert traj.metadata == {}
