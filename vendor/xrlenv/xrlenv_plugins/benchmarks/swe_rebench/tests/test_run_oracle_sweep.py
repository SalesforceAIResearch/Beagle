"""Unit tests for the SWE-rebench oracle sweep's pure logic (no network, no cluster)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from xrlenv_plugins.benchmarks.swe_rebench.run_oracle_sweep import (
    _INFRA_RETRY_EXCEPTIONS,
    _resolve_tasks,
    _rewards,
    _summarise,
    _task_key,
    _trial_passes,
)


def _trial(
    task_id: str = "demo__repo-1",
    *,
    rewards: dict[str, Any] | None = None,
    exception_type: str | None = None,
) -> SimpleNamespace:
    """A duck-typed stand-in for harbor's TrialResult (only the fields the gate
    reads)."""
    return SimpleNamespace(
        config=SimpleNamespace(task=SimpleNamespace(path=Path("/cache/x") / task_id)),
        verifier_result=SimpleNamespace(rewards=rewards),
        exception_info=(
            None if exception_type is None
            else SimpleNamespace(exception_type=exception_type)
        ),
    )


# ── the pass gate ─────────────────────────────────────────────────────────────


def test_pass_on_resolved_instance() -> None:
    """reward.txt == 1 -> harbor rewards={"reward": 1.0} -> PASS."""
    assert _trial_passes(_trial(rewards={"reward": 1.0})) == (True, None)


def test_fail_on_unresolved_instance() -> None:
    ok, reason = _trial_passes(_trial(rewards={"reward": 0.0}))
    assert ok is False
    assert reason is not None and "rewards=" in reason


def test_fail_when_any_reward_key_is_zero() -> None:
    """The gate is 'every reward > 0' — a side key at 0 must not pass."""
    ok, _ = _trial_passes(_trial(rewards={"reward": 1.0, "extra": 0.0}))
    assert ok is False


def test_pass_with_multiple_positive_rewards() -> None:
    assert _trial_passes(_trial(rewards={"reward": 1, "extra": 0.5}))[0] is True


def test_fail_on_missing_verifier_result_names_the_exception() -> None:
    ok, reason = _trial_passes(_trial(rewards=None, exception_type="NodeLost"))
    assert ok is False
    assert reason is not None and "NodeLost" in reason


def test_fail_on_missing_verifier_result_without_exception() -> None:
    ok, reason = _trial_passes(_trial(rewards=None))
    assert ok is False
    assert reason == "no verifier result (no reward file written)"


def test_fail_on_unparseable_reward() -> None:
    ok, _ = _trial_passes(_trial(rewards={"reward": "not-a-number"}))
    assert ok is False


def test_rewards_coerces_ints_to_float() -> None:
    assert _rewards(_trial(rewards={"reward": 1})) == {"reward": 1.0}


def test_rewards_none_on_empty_dict() -> None:
    """An empty rewards dict is 'nothing graded', not 'all rewards positive' —
    otherwise `all(...)` over an empty set would vacuously PASS."""
    assert _rewards(_trial(rewards={})) is None
    assert _trial_passes(_trial(rewards={}))[0] is False


def test_task_key_is_the_shard_dir_name() -> None:
    assert _task_key(_trial("astropy__astropy-1")) == "astropy__astropy-1"


# ── retry policy ──────────────────────────────────────────────────────────────


def test_infra_retry_set_is_infra_only() -> None:
    """The retry set must never include a content/verification outcome — that is
    --content-retries' job, and it is visible in the artifacts."""
    assert set(_INFRA_RETRY_EXCEPTIONS) == {
        "CapacityExhausted",
        "ControlPlaneLost",
        "NodeLost",
        "NodeCommandTimeout",
        "SessionReaped",
    }


# ── task resolution ───────────────────────────────────────────────────────────


def _make_shard(tmp_path: Path, *names: str) -> Path:
    shard = tmp_path / "swe-rebench"
    for name in names:
        (shard / name / "solution").mkdir(parents=True)
        (shard / name / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    return shard


def test_resolve_tasks_defaults_to_every_gateable_task(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path, "b", "a")
    (shard / "no-solution").mkdir()  # no solve.sh -> not gateable
    assert _resolve_tasks(shard, None) == ["a", "b"]


def test_resolve_tasks_honors_an_explicit_subset(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path, "a", "b", "c")
    assert _resolve_tasks(shard, "c, a") == ["c", "a"]  # caller order preserved


def test_resolve_tasks_fails_loud_on_unknown_task(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path, "a")
    with pytest.raises(SystemExit, match="unknown / non-gateable"):
        _resolve_tasks(shard, "a,ghost")


def test_resolve_tasks_fails_loud_on_empty_selection(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path, "a")
    with pytest.raises(SystemExit, match="selected no tasks"):
        _resolve_tasks(shard, " , ")


def test_resolve_tasks_fails_loud_on_empty_shard(tmp_path: Path) -> None:
    shard = tmp_path / "swe-rebench"
    shard.mkdir()
    with pytest.raises(SystemExit, match="no oracle-gateable tasks"):
        _resolve_tasks(shard, None)


# ── exit code ─────────────────────────────────────────────────────────────────


def test_summarise_exits_zero_only_when_every_oracle_solved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    all_pass = [_trial("a", rewards={"reward": 1.0}),
                _trial("b", rewards={"reward": 1.0})]
    assert _summarise(all_pass, 2, tmp_path) == 0

    mixed = [_trial("a", rewards={"reward": 1.0}),
             _trial("b", rewards={"reward": 0.0})]
    assert _summarise(mixed, 2, tmp_path) == 1
    out = capsys.readouterr().out
    assert "1 / 2 oracle(s) solved." in out
    assert "'b'" in out


def test_summarise_counts_a_missing_trial_as_unsolved(tmp_path: Path) -> None:
    """`expected` is the requested count — a task whose trial never came back
    must not be able to green the gate."""
    assert _summarise([_trial("a", rewards={"reward": 1.0})], 2, tmp_path) == 1
