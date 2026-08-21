"""Unit tests for the LHTB oracle sweep's pure logic (dense-reward gate; no cluster)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from xrlenv_plugins.benchmarks.lhtb import run_oracle_sweep as s


def _tr(rewards, *, exc=None, name="t"):
    return SimpleNamespace(
        task_name=name,
        verifier_result=SimpleNamespace(rewards=rewards),
        exception_info=(SimpleNamespace(exception_type=exc) if exc else None),
    )


def test_dense_reward_positive_passes() -> None:
    # LHTB reward is a partial-credit float; > 0 = the oracle produced a gradable result
    assert s._trial_passes(_tr({"reward": 0.72}))[0] is True
    assert s._trial_passes(_tr({"reward": 1.0}))[0] is True


def test_reward_zero_fails() -> None:
    ok, reason = s._trial_passes(_tr({"reward": 0.0}))
    assert not ok and reason == "reward=0.0"


def test_exception_fails() -> None:
    ok, reason = s._trial_passes(_tr({"reward": 1.0}, exc="AgentTimeoutError"))
    assert not ok and "AgentTimeoutError" in reason


def test_no_rewards_fails() -> None:
    assert s._trial_passes(_tr(None))[0] is False
    assert s._trial_passes(_tr({}))[0] is False


def test_reward_value_prefers_reward_key_then_falls_back_to_max() -> None:
    # Prefer the canonical "reward" key even when a raw diagnostic metric in the same
    # dict is larger (chess-mate emits reference_white_moves=120 alongside reward=0.9),
    # so a max-over-values would grab the raw metric, not the [0,1] reward.
    assert s._reward_value(_tr({"reward": 0.3, "bonus": 0.9})) == 0.3
    # Fall back to max-over-values only when a verifier writes no "reward" key.
    assert s._reward_value(_tr({"score": 0.4, "bonus": 0.7})) == 0.7
    # Non-numeric reward -> None (not a crash).
    assert s._reward_value(_tr({"reward": "oops"})) is None


def test_task_key_is_dir_basename_not_namespaced_task_name() -> None:
    # Regression (2026-07-31 ci false-fail): the content-retry loop keys `best` on
    # _task_key — the requested id == shard dir name (basename of config.task.path) —
    # NOT trial_result.task_name. LHTB task.toml names are "long-horizon-terminal-bench
    # /<id>", so keying `best` on task_name never matched the requested bare ids: every
    # task read as non-passing (re-ran all retries) and the tally showed 0/N at reward>0.
    tr = SimpleNamespace(
        config=SimpleNamespace(
            task=SimpleNamespace(path="/cache/lhtb/su2-airfoil-regression"),
        ),
        task_name="long-horizon-terminal-bench/su2-airfoil-regression",
    )
    assert s._task_key(tr) == "su2-airfoil-regression"


def test_resolve_tasks(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    for name in ("2048", "sokoban"):
        (shard / name).mkdir(parents=True)
        (shard / name / "task.toml").write_text("[environment]\n")
    assert s._resolve_tasks(shard, None) == ["2048", "sokoban"]
    assert s._resolve_tasks(shard, "sokoban") == ["sokoban"]
    with pytest.raises(SystemExit, match="unknown task"):
        s._resolve_tasks(shard, "nope")


def test_raw_reward_reads_raw_reward_txt(tmp_path: Path) -> None:
    # a game trial that wrote both normalized reward.txt and raw_reward.txt
    vdir = tmp_path / "2048__abc" / "verifier"
    vdir.mkdir(parents=True)
    (vdir / "raw_reward.txt").write_text("6.6445\n")
    tr = SimpleNamespace(trial_uri=f"file://{tmp_path / '2048__abc'}")
    assert s._raw_reward(tr) == 6.6445


def test_raw_reward_none_when_absent(tmp_path: Path) -> None:
    # snake_maze-style trial: no raw_reward.txt (reward is band-derived directly)
    (tmp_path / "snake__xyz" / "verifier").mkdir(parents=True)
    tr = SimpleNamespace(trial_uri=f"file://{tmp_path / 'snake__xyz'}")
    assert s._raw_reward(tr) is None
    # and no trial_uri at all -> None
    assert s._raw_reward(SimpleNamespace(trial_uri=None)) is None


def test_resolve_tasks_empty_selector_raises(tmp_path: object) -> None:
    # audit M5/Low: a present-but-empty --tasks ("" or ",") must FAIL, not fall through.
    import pytest as _pytest
    import xrlenv_plugins.benchmarks.lhtb.run_oracle_sweep as _s
    for empty in ("", ",", " , "):
        with _pytest.raises(SystemExit, match="selected no tasks"):
            _s._resolve_tasks(tmp_path, empty)  # type: ignore[arg-type]
