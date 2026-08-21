"""Per-task rollout-artifact persistence + the result.json resume record (base.py).

A per-task harness (docker drop-in) writes ``run_dir/<benchmark>/<task_id>/`` — patch.diff, the raw
native trajectory (+ best-effort ATIF), and, after grading, result.json — so the run is inspectable
and resume can read done-state back from the harness's own tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beagle.benchmarks.base import read_result_json, write_result_json, write_rollout_artifacts
from beagle.types import RolloutStatus, Task, TaskResult, TrajectoryRef


def test_write_rollout_artifacts_persists_patch_and_raw_trajectory(tmp_path) -> None:
    task = Task(task_id="django__django-1", benchmark="swe-bench-verified")
    result = TaskResult(
        task_id="django__django-1", benchmark="swe-bench-verified", patch="THE PATCH",
        trajectory=TrajectoryRef(path=Path("mini.traj.json"), format="mini-swe"),
        trajectory_text='{"messages": []}')
    write_rollout_artifacts(result, task, tmp_path, agent_name="mini-swe")

    task_dir = tmp_path / "swe-bench-verified" / "django__django-1"
    assert result.artifact_dir == task_dir                       # grader writes its report here too
    assert (task_dir / "patch.diff").read_text() == "THE PATCH"
    # the raw native stream is the reliable artifact (named from the TrajectoryRef)
    assert (task_dir / "agent" / "mini.traj.json").read_text() == '{"messages": []}'
    assert not (task_dir / "agent" / "stderr.log").exists()      # no stderr kept on success


def test_write_rollout_artifacts_persists_stderr_on_failure(tmp_path) -> None:
    # On failure the full stderr is kept (the one-line `error` is only its first meaningful line),
    # even when the trial left no patch/trajectory — the agent dir is created for it.
    task = Task(task_id="pytest-dev__pytest-1", benchmark="swe-bench-verified")
    result = TaskResult(
        task_id="pytest-dev__pytest-1", benchmark="swe-bench-verified",
        status=RolloutStatus.FAILED, error="monet exited rc=1: Error: gateway 429",
        stderr_text="(node:7) [UNDICI-EHPA] Warning: ...\nError: gateway 429 after 6 retries\n")
    write_rollout_artifacts(result, task, tmp_path, agent_name="monet")

    log = tmp_path / "swe-bench-verified" / "pytest-dev__pytest-1" / "agent" / "stderr.log"
    assert log.read_text().splitlines()[-1] == "Error: gateway 429 after 6 retries"


def test_write_rollout_artifacts_stamps_agent_version_into_atif(tmp_path) -> None:
    # the trajectory's agent.version must be the real ref (the code under eval), never "unknown"
    pytest.importorskip("harbor")
    task = Task(task_id="t1", benchmark="b")
    mini_traj = json.dumps({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "ok", "extra": {"actions": [{"command": "ls"}]}}]})
    result = TaskResult(
        task_id="t1", benchmark="b", trajectory_text=mini_traj,
        trajectory=TrajectoryRef(path=Path("mini.traj.json"), format="mini-swe"))
    write_rollout_artifacts(result, task, tmp_path, agent_name="mini-swe", agent_version="abc123sha")

    atif = json.loads((tmp_path / "b" / "t1" / "agent" / "trajectory.json").read_text())
    assert atif["agent"]["name"] == "mini-swe" and atif["agent"]["version"] == "abc123sha"


def test_write_rollout_artifacts_is_noop_without_run_dir() -> None:
    task = Task(task_id="t", benchmark="b")
    result = TaskResult(task_id="t", benchmark="b", patch="x")
    write_rollout_artifacts(result, task, None)   # no run_dir → no write, no crash
    assert result.artifact_dir is None


def test_write_rollout_artifacts_is_noop_without_benchmark(tmp_path) -> None:
    # a task with no benchmark (e.g. the harbor shim's synthetic Task) → nowhere to root a subtree
    result = TaskResult(task_id="t", patch="x")
    write_rollout_artifacts(result, Task(task_id="t"), tmp_path)
    assert result.artifact_dir is None and not any(tmp_path.iterdir())


def test_result_json_roundtrips(tmp_path) -> None:
    task_dir = tmp_path / "b" / "t1"
    task_dir.mkdir(parents=True)
    r = TaskResult(
        task_id="t1", benchmark="b", status=RolloutStatus.COMPLETED, resolved=True, reward=1.0,
        num_turns=3, tokens={"prompt": 5, "completion": 2}, artifact_dir=task_dir)
    write_result_json(r)

    d = json.loads((task_dir / "result.json").read_text())
    assert d["task_id"] == "t1" and d["resolved"] is True and d["reward"] == 1.0

    back = read_result_json(task_dir / "result.json")
    assert back is not None
    assert back.task_id == "t1" and back.benchmark == "b" and back.resolved
    assert back.reward == 1.0 and back.num_turns == 3 and back.tokens == {"prompt": 5, "completion": 2}
    assert back.status is RolloutStatus.COMPLETED and back.artifact_dir == task_dir


def test_read_result_json_missing_or_malformed_is_none(tmp_path) -> None:
    assert read_result_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_result_json(bad) is None
