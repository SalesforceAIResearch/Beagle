"""Unit tests for the FrontierSWE oracle sweep's pure logic (no cluster).

The distinctive gate: harbor 0.20's strict ``VerifierResult`` can't ingest
FrontierSWE's rich ``reward.json`` (``subscores`` list + ``additional_data`` dict),
so ``verifier_result`` is None and the trial carries a ``ValidationError`` — on
EVERY task. We therefore grade from the **downloaded** ``reward.json`` on disk, and
a harbor ``exception_info`` is IGNORED whenever a gradeable reward.json is present.
These tests pin exactly that behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from xrlenv_plugins.benchmarks.frontier_swe import run_oracle_sweep as s


def _write_reward(trials_dir: Path, trial_name: str, payload: dict) -> None:
    """Write a reward.json at harbor's layout: <trials_dir>/<trial>/verifier/reward.json."""
    vdir = trials_dir / trial_name / "verifier"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "reward.json").write_text(json.dumps(payload))


def _tr(
    *,
    trials_dir: Path | None = None,
    trial_name: str = "t-0",
    task_path: str = "/cache/frontier-swe/ffmpeg-swscale-rewrite",
    verifier_rewards: dict | None = None,
    exc: str | None = None,
) -> SimpleNamespace:
    """A duck-typed trial_result. ``verifier_rewards`` populates harbor's parsed
    result (the flat-schema / future-harbor path); the disk reward.json (if written)
    is the FrontierSWE rich-schema path."""
    exception_info = SimpleNamespace(exception_type=exc) if exc else None
    verifier_result = (
        SimpleNamespace(rewards=verifier_rewards) if verifier_rewards is not None
        else None
    )
    config = SimpleNamespace(
        task=SimpleNamespace(path=task_path),
        trials_dir=str(trials_dir) if trials_dir is not None else None,
        trial_name=trial_name,
    )
    return SimpleNamespace(
        config=config,
        verifier_result=verifier_result,
        exception_info=exception_info,
    )


# ── the rich-schema disk path (the crux) ──────────────────────────────────────


def test_pass_from_disk_reward_positive_despite_harbor_validationerror(tmp_path: Path) -> None:
    # EVERY FrontierSWE trial carries a ValidationError exception (harbor can't parse
    # the rich reward.json) — but a gradeable reward.json is on disk, so it PASSES.
    _write_reward(
        tmp_path, "t-0",
        {"score": 0.9, "reward": 0.9,
         "subscores": [{"subtask": "x", "score": 1}],
         "additional_data": {"reason": "ok", "total_time_ms": 1234}},
    )
    tr = _tr(trials_dir=tmp_path, exc="ValidationError")
    ok, reason = s._trial_passes(tr)
    assert ok
    assert reason is None
    assert s._reward_value(tr) == 0.9


def test_fail_from_disk_reward_zero(tmp_path: Path) -> None:
    _write_reward(
        tmp_path, "t-0",
        {"score": 0.0, "reward": 0.0, "subscores": [], "additional_data": {"reason": "fail"}},
    )
    ok, reason = s._trial_passes(_tr(trials_dir=tmp_path, exc="ValidationError"))
    assert not ok
    assert reason == "reward=0.0"


def test_fail_when_no_reward_json_and_exception(tmp_path: Path) -> None:
    # No reward.json on disk (verifier never produced output) + a harbor exception
    # -> a REAL failure, surfaced with the exception type.
    ok, reason = s._trial_passes(_tr(trials_dir=tmp_path, exc="NodeLost"))
    assert not ok
    assert "NodeLost" in reason
    assert "no gradeable reward.json" in reason


def test_fail_when_no_reward_json_no_exception(tmp_path: Path) -> None:
    ok, reason = s._trial_passes(_tr(trials_dir=tmp_path, exc=None))
    assert not ok
    assert "no gradeable reward.json" in reason


def test_score_fallback_when_reward_key_absent(tmp_path: Path) -> None:
    # emit_reward always writes both, but be defensive: fall back to "score".
    _write_reward(tmp_path, "t-0", {"score": 0.5, "subscores": [], "additional_data": {}})
    assert s._reward_value(_tr(trials_dir=tmp_path)) == 0.5


# ── the flat-schema / future-harbor path (verifier_result populated) ──────────


def test_pass_from_verifier_result_when_populated() -> None:
    # If a future harbor (or a flat-schema task) populates verifier_result.rewards,
    # we use it directly — no disk read needed.
    ok, reason = s._trial_passes(_tr(verifier_rewards={"reward": 1.0}))
    assert ok
    assert reason is None


def test_verifier_result_takes_precedence_over_disk(tmp_path: Path) -> None:
    _write_reward(tmp_path, "t-0", {"reward": 0.0, "subscores": [], "additional_data": {}})
    # verifier_result says pass; it is preferred over the disk reward.json.
    tr = _tr(trials_dir=tmp_path, verifier_rewards={"reward": 1.0})
    assert s._reward_value(tr) == 1.0
    assert s._trial_passes(tr)[0] is True


# ── reward extraction + side metrics ──────────────────────────────────────────


def test_reward_from_json_obj_non_numeric_is_none() -> None:
    assert s._reward_from_json_obj({"reward": "oops"}) is None
    assert s._reward_from_json_obj({}) is None
    assert s._reward_from_json_obj([1, 2, 3]) is None
    assert s._reward_from_json_obj({"reward": 3}) == 3.0


def test_side_metrics_skips_container_values(tmp_path: Path) -> None:
    _write_reward(
        tmp_path, "t-0",
        {"score": 0.9, "reward": 0.9, "extra_scalar": 2,
         "subscores": [{"a": 1}], "additional_data": {"k": "v"}, "flag": True},
    )
    side = s._side_metrics(_tr(trials_dir=tmp_path))
    # keeps the scalar non-reward key; drops the list, the dict, the bool, and
    # the reward/score keys themselves.
    assert side == {"extra_scalar": 2}


def test_task_key_is_dir_basename() -> None:
    tr = _tr(task_path="/cache/frontier-swe/git-to-zig")
    assert s._task_key(tr) == "git-to-zig"


# ── task resolution (solve.sh anchor) ─────────────────────────────────────────


def _mk_gateable(shard: Path, name: str, *, with_solve: bool = True) -> None:
    (shard / name).mkdir(parents=True)
    (shard / name / "task.toml").write_text("[environment]\n")
    if with_solve:
        sol = shard / name / "solution"
        sol.mkdir()
        (sol / "solve.sh").write_text("#!/bin/sh\n")


def test_resolve_tasks_only_gateable(tmp_path: Path) -> None:
    shard = tmp_path / "frontier-swe"
    _mk_gateable(shard, "a")
    _mk_gateable(shard, "b")
    _mk_gateable(shard, "withheld", with_solve=False)  # no solve.sh -> not gateable

    assert s._resolve_tasks(shard, None) == ["a", "b"]  # all gateable, sorted
    assert s._resolve_tasks(shard, "b") == ["b"]  # subset
    with pytest.raises(SystemExit, match=r"non-gateable|unknown"):
        s._resolve_tasks(shard, "withheld")  # present but no solve.sh
    with pytest.raises(SystemExit, match=r"non-gateable|unknown"):
        s._resolve_tasks(shard, "nope")


def test_resolve_tasks_empty_selector_raises(tmp_path: Path) -> None:
    shard = tmp_path / "frontier-swe"
    _mk_gateable(shard, "a")
    for empty in ("", ",", " , ", ",,"):
        with pytest.raises(SystemExit, match="selected no tasks"):
            s._resolve_tasks(shard, empty)
    assert s._resolve_tasks(shard, None) == ["a"]  # None (absent) -> all


def test_infra_retry_set_excludes_validationerror() -> None:
    # The reward-schema ValidationError must NOT be retried as infra (it's expected +
    # handled by grade-from-artifact, not a transient blip).
    assert "ValidationError" not in s._INFRA_RETRY_EXCEPTIONS
    assert "CapacityExhausted" in s._INFRA_RETRY_EXCEPTIONS
