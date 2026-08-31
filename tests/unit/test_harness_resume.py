"""HarborHarness resume seam + helper coverage.

Tests for the code paths that landed in the current session:
- ``_error_string``: dict / object / partial / None variants
- ``_result_from_harbor_json``: reward thresholds, missing verifier, on-disk path
- ``HarborHarness.completed``: happy path, missing bench_dir, malformed result.json,
  pass@k task ids, empty items list
- ``_job_dir``: relative trial_dir, all-None, empty rows

All tests are hermetic — no cluster, no harbor import required except where
``pytest.importorskip`` guards it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from beagle.benchmarks.harness.drivers import _error_string
from beagle.rollout.runner import _job_dir as runner_job_dir


# ---------------------------------------------------------------------------
# _error_string
# ---------------------------------------------------------------------------

def test_error_string_dict_both_fields() -> None:
    assert _error_string({"exception_type": "T", "exception_message": "msg"}) == "T: msg"


def test_error_string_dict_type_only() -> None:
    assert _error_string({"exception_type": "T", "exception_message": None}) == "T"


def test_error_string_dict_message_only() -> None:
    assert _error_string({"exception_type": None, "exception_message": "bad"}) == "bad"


def test_error_string_dict_neither_returns_none() -> None:
    assert _error_string({}) is None
    assert _error_string({"exception_type": "", "exception_message": ""}) is None


def test_error_string_object_type_and_message() -> None:
    class _E:
        exception_type = "AgentTimeoutError"
        exception_message = "deadline exceeded"

    assert _error_string(_E()) == "AgentTimeoutError: deadline exceeded"


def test_error_string_object_no_attributes_returns_none() -> None:
    class _E:
        exception_type = None
        exception_message = None

    assert _error_string(_E()) is None


def test_error_string_none_input_is_none() -> None:
    assert _error_string(None) is None


# ---------------------------------------------------------------------------
# _result_from_harbor_json
# ---------------------------------------------------------------------------

def test_result_from_harbor_json_reward_above_threshold() -> None:
    """reward=1.0 → resolved=True, status=COMPLETED, no error."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json
    from beagle.types import RolloutStatus

    d = {
        "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_result": {"n_input_tokens": 200, "n_output_tokens": 30},
    }
    r = _result_from_harbor_json(d, "task1", Path("/bench/task1"))
    assert r.task_id == "task1"
    assert r.reward == 1.0
    assert r.resolved is True
    assert r.error is None
    assert r.status is RolloutStatus.COMPLETED
    # no n_cache_tokens → whole prompt is uncached; cache split still present (all zero cache)
    assert r.tokens == {"prompt": 200, "completion": 30,
                        "input_uncached": 200, "cache_read": 0, "cache_write": 0}
    assert r.artifact_dir == Path("/bench/task1")


def test_result_from_harbor_json_preserves_cache_split() -> None:
    """harbor's n_cache_tokens (cached subset of n_input) → beagle's cache_read, so run.json's cache
    buckets are non-zero. prompt stays the FULL input; input_uncached = prompt - cache."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json

    d = {
        "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_result": {"n_input_tokens": 366122, "n_cache_tokens": 336894, "n_output_tokens": 9144},
    }
    r = _result_from_harbor_json(d, "t", Path("/d"))
    assert r.tokens == {"prompt": 366122, "completion": 9144,
                        "input_uncached": 29228, "cache_read": 336894, "cache_write": 0}
    # invariant: prompt == input_uncached + cache_read + cache_write
    t = r.tokens
    assert t["prompt"] == t["input_uncached"] + t["cache_read"] + t["cache_write"]


def test_tokens_from_harbor_guards_and_nones() -> None:
    """The reconstruction helper: None counters → all zero; cache is clamped to prompt so a bogus
    n_cache > n_input can't make input_uncached negative."""
    from beagle.benchmarks.harness.drivers import _tokens_from_harbor

    assert _tokens_from_harbor(None, None, None) == {
        "prompt": 0, "completion": 0, "input_uncached": 0, "cache_read": 0, "cache_write": 0}
    # cache clamped to prompt (never negative uncached)
    assert _tokens_from_harbor(100, 250, 5) == {
        "prompt": 100, "completion": 5, "input_uncached": 0, "cache_read": 100, "cache_write": 0}


def test_result_from_harbor_json_partial_reward_not_resolved() -> None:
    """reward=0.4 < 1.0 threshold → resolved=False."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json

    d = {"verifier_result": {"rewards": {"reward": 0.4}}}
    r = _result_from_harbor_json(d, "task2", Path("/bench/task2"))
    assert r.reward == 0.4
    assert r.resolved is False
    assert r.error is None


def test_result_from_harbor_json_zero_reward_not_resolved() -> None:
    """reward=0.0 → resolved=False (boundary: reward is NOT None but 0 < 1.0)."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json

    d = {"verifier_result": {"rewards": {"reward": 0.0}}}
    r = _result_from_harbor_json(d, "t", Path("/d"))
    assert r.reward == 0.0
    assert r.resolved is False


def test_result_from_harbor_json_no_verifier_reward_is_none() -> None:
    """verifier_result absent → reward=None → resolved=False."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json

    r = _result_from_harbor_json({}, "t", Path("/d"))
    assert r.reward is None
    assert r.resolved is False
    assert r.error is None


def test_result_from_harbor_json_with_exception_info() -> None:
    """exception_info present → error string set, status=FAILED."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json
    from beagle.types import RolloutStatus

    d = {
        "exception_info": {"exception_type": "AgentTimeoutError", "exception_message": "timed out"},
    }
    r = _result_from_harbor_json(d, "t", Path("/d"))
    assert r.error == "AgentTimeoutError: timed out"
    assert r.status is RolloutStatus.FAILED
    assert r.resolved is False


def test_result_from_harbor_json_missing_agent_result_tokens_default_zero() -> None:
    """Missing agent_result → tokens default to 0."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json

    r = _result_from_harbor_json({"verifier_result": {"rewards": {"reward": 1.0}}},
                                 "t", Path("/d"))
    assert r.tokens == {"prompt": 0, "completion": 0,
                        "input_uncached": 0, "cache_read": 0, "cache_write": 0}


# ---------------------------------------------------------------------------
# HarborHarness.completed
# ---------------------------------------------------------------------------

def test_harbor_completed_returns_empty_when_no_items(tmp_path) -> None:
    from beagle.benchmarks.harness import HarborHarness

    h = HarborHarness()
    assert h.completed([], run_dir=tmp_path) == []


def test_harbor_completed_returns_empty_when_bench_dir_missing(tmp_path) -> None:
    """When there is no <benchmark> dir under run_dir, nothing is considered done."""
    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    items = [(Task(task_id="t1", benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    h = HarborHarness()
    # run_dir exists but has no terminal_bench_2_1 sub-dir
    result = h.completed(items, run_dir=tmp_path)
    assert result == []


def test_harbor_completed_reads_back_finished_trial(tmp_path) -> None:
    """Happy path: a result.json written by harbor is read back as a TaskResult."""
    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench_dir = tmp_path / "terminal_bench_2_1"
    trial_dir = bench_dir / "task_abc"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_result": {"n_input_tokens": 10, "n_output_tokens": 5},
    }))

    items = [(Task(task_id="task_abc", benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    h = HarborHarness()
    results = h.completed(items, run_dir=tmp_path)

    assert len(results) == 1
    assert results[0].task_id == "task_abc"
    assert results[0].resolved is True
    assert results[0].reward == 1.0


def test_harbor_completed_skips_task_without_result_json(tmp_path) -> None:
    """A trial dir with no result.json → task is not in done-set."""
    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench_dir = tmp_path / "terminal_bench_2_1"
    (bench_dir / "task_no_result").mkdir(parents=True)
    # No result.json written

    items = [(Task(task_id="task_no_result", benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    h = HarborHarness()
    assert h.completed(items, run_dir=tmp_path) == []


def test_harbor_completed_tolerates_malformed_result_json(tmp_path) -> None:
    """A malformed result.json silently re-queues the task (never crashes resume)."""
    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench_dir = tmp_path / "terminal_bench_2_1"
    trial_dir = bench_dir / "task_bad"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text("not valid { json")

    items = [(Task(task_id="task_bad", benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    h = HarborHarness()
    # Must NOT raise; task is simply absent from done-set
    result = h.completed(items, run_dir=tmp_path)
    assert result == []


def test_harbor_completed_handles_pass_at_k_trial_name(tmp_path) -> None:
    """Pass@k: task_id='orig__s0' written by select_and_sample → found via exact path."""
    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench_dir = tmp_path / "terminal_bench_2_1"
    # harbor writes trial at bench_dir / task.task_id = 'orig__s0'
    trial_dir = bench_dir / "orig__s0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps(
        {"verifier_result": {"rewards": {"reward": 1.0}}}
    ))

    items = [(Task(task_id="orig__s0", benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    h = HarborHarness()
    results = h.completed(items, run_dir=tmp_path)

    assert len(results) == 1
    assert results[0].task_id == "orig__s0"
    assert results[0].resolved is True


def test_harbor_completed_finds_via_glob_suffix(tmp_path) -> None:
    """A trial written with a __suffix (from harbor adding its own hash) is found via glob."""
    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench_dir = tmp_path / "terminal_bench_2_1"
    # Hypothetical: harbor appended a unique suffix
    trial_dir = bench_dir / "task1__ABCD1234"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps(
        {"verifier_result": {"rewards": {"reward": 0.0}},
         "exception_info": None}
    ))

    # The task is queried by its base id 'task1' (no __suffix)
    items = [(Task(task_id="task1", benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    h = HarborHarness()
    results = h.completed(items, run_dir=tmp_path)

    assert len(results) == 1
    assert results[0].task_id == "task1"
    assert results[0].reward == 0.0


def test_harbor_completed_recovers_truncated_long_id(tmp_path) -> None:
    """The load-bearing fix: harbor/pier truncate the trial DIR slug to 32 chars + a hash, so a glob
    on the full (long) task_id misses it. completed() must match on the id the result.json RECORDS."""
    from beagle.benchmarks.harness import PierHarness
    from beagle.types import Task, TaskContext

    full = "arktype-json-schema-refs-dependencies"          # 37 chars > 32
    bench_dir = tmp_path / "deep-swe"
    trial_dir = bench_dir / f"{full[:32]}__QVk4tHk"          # truncated slug, harbor-style
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({
        "task_id": {"path": f"/cache/deep-swe/{full}"},      # recorded full identity
        "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_result": {"n_input_tokens": 100, "n_output_tokens": 20},
    }))

    items = [(Task(task_id=full, benchmark="deep-swe"), TaskContext(image=None))]
    results = PierHarness().completed(items, run_dir=tmp_path)
    assert [r.task_id for r in results] == [full]            # found despite the truncated dir
    assert results[0].resolved is True


def test_harbor_completed_flags_silent_no_attempt_as_error(tmp_path) -> None:
    """A trial where the agent never ran (0 input+output tokens, unresolved, no error) — a failed
    repo clone/bootstrap — is stamped a NoAttempt error (category E) instead of a fake reward=0,
    so it's no longer mistaken for a genuine capability failure (category F, no error)."""
    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench_dir = tmp_path / "terminal_bench_2_1"
    trial_dir = bench_dir / "never-ran"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": 0.0}},
        "agent_result": {"n_input_tokens": 0, "n_output_tokens": 0},
    }))

    items = [(Task(task_id="never-ran", benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    r = HarborHarness().completed(items, run_dir=tmp_path)[0]
    assert r.error is not None and r.error.startswith("NoAttempt")


def test_result_from_harbor_json_no_attempt_gated_on_agent_actually_ran() -> None:
    """No-attempt fires only when the agent truly didn't run — keyed on ACTIVITY (a native stream /
    harbor tokens / a positive reward), NOT harbor's lossy 0-token count. A graded reward=0 with any
    sign of the agent running is a genuine fail, not a no-attempt; only a graded failure with zero
    activity (setup/clone died) is flagged; an ungraded (reward None) trial is left alone."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json

    ran = {"verifier_result": {"rewards": {"reward": 0.0}},        # harbor tokens present → ran
           "agent_result": {"n_input_tokens": 500, "n_output_tokens": 12}}
    assert _result_from_harbor_json(ran, "t", Path("/d")).error is None
    resolved = {"verifier_result": {"rewards": {"reward": 1.0}}}   # resolved → never flagged
    assert _result_from_harbor_json(resolved, "t", Path("/d")).error is None
    ungraded = {"verifier_result": {"rewards": {"reward": None}}, "agent_result": {}}
    assert _result_from_harbor_json(ungraded, "t", Path("/d")).error is None       # reward None → no guess
    no_attempt = {"verifier_result": {"rewards": {"reward": 0.0}}, "agent_result": {}}  # graded 0, no activity
    assert (_result_from_harbor_json(no_attempt, "t", Path("/d")).error or "").startswith("NoAttempt")


def test_tokens_and_ran_read_from_native_stream_not_harbor(tmp_path) -> None:
    """Regression (the real bug): opencode on the harbor path reports 0 tokens in ``agent_result``
    even after a full run — tokens live in the native stream. Both the no-attempt gate AND the token
    totals must read the CANONICAL stream, so a genuine reward=0 fail with a real stream is NOT
    mislabeled ``NoAttempt`` and its tokens aren't dropped to zero."""
    from beagle.benchmarks.harness.drivers import _result_from_harbor_json

    trial = tmp_path / "t__x"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "opencode.stream.jsonl").write_text(json.dumps(
        {"type": "step_finish",
         "part": {"tokens": {"input": 300, "output": 20, "cache": {"read": 100, "write": 0}}}}) + "\n")
    d = {"verifier_result": {"rewards": {"reward": 0.0}},          # graded fail
         "agent_result": {"n_input_tokens": 0, "n_output_tokens": 0}}   # harbor LOST the tokens
    r = _result_from_harbor_json(d, "t", trial)
    assert r.error is None and r.resolved is False                # genuine-fail, NOT a false NoAttempt
    assert r.tokens["prompt"] == 400 and r.tokens["cache_read"] == 100   # from the stream, not 0


def test_usage_from_agent_dir_none_when_no_stream(tmp_path) -> None:
    from beagle.benchmarks.agent_usage import usage_from_agent_dir

    (tmp_path / "agent").mkdir()
    assert usage_from_agent_dir(tmp_path / "agent") is None        # no recognized stream → None (never ran)
    assert usage_from_agent_dir(tmp_path / "nonexistent") is None  # missing dir → None


def test_harbor_resume_reconciles_subset_by_delete_and_full_config(monkeypatch, tmp_path) -> None:
    """Design-A reconcile: on a resume, HarborHarness deletes exactly the to_run trial dirs and hands
    harbor the STORED **full** config (so harbor's config check passes and it re-runs only the deleted
    trials, skipping the rest), then returns the re-run subset's fresh results keyed by FULL id."""
    pytest.importorskip("harbor")
    import harbor
    from harbor.models.job.config import JobConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench = "deep-swe"
    job_dir = tmp_path / bench
    job_dir.mkdir(parents=True)
    long_id = "arktype-json-schema-refs-dependencies"        # 37 chars → dir slug truncated to 32
    task_paths = {tid: tmp_path / "tasks" / tid for tid in ("t_keep", long_id, "t_err")}
    for p in task_paths.values():
        p.mkdir(parents=True)

    # The "original" run's stored full JobConfig (3 tasks) + 3 native trial dirs.
    stored = JobConfig(
        job_name=bench, jobs_dir=str(tmp_path), n_concurrent_trials=1,
        environment=EnvironmentConfig(import_path="pkg:Env"), agents=[AgentConfig()],
        tasks=[TaskConfig(path=str(p), trial_name=t) for t, p in task_paths.items()])
    (job_dir / "config.json").write_text(stored.model_dump_json())

    def _seed(tid: str, dirslug: str, reward: float, error: str | None = None) -> None:
        d = job_dir / dirslug
        d.mkdir()
        rj = {"task_id": {"path": f"/tasks/{tid}"},         # recorded FULL id (dir slug is truncated)
              "verifier_result": {"rewards": {"reward": reward}},
              "agent_result": {"n_input_tokens": 9, "n_output_tokens": 3}}
        if error:
            rj["exception_info"] = {"exception_type": error, "exception_message": "x"}
        (d / "result.json").write_text(json.dumps(rj))

    _seed("t_keep", "t_keep__A", 1.0)                        # resolved → must survive
    _seed(long_id, f"{long_id[:32]}__B", 0.0, error="GitCloneError")   # to_run (long id) → delete+rerun
    _seed("t_err", "t_err__C", 0.0, error="GitCloneError")             # to_run → delete+rerun

    want = {long_id, "t_err"}
    cap: dict = {}

    class _FakeJob:
        @classmethod
        async def create(cls, config):  # noqa: ANN001
            cap["config"] = config
            return cls()

        async def run(self):
            for tid in want:                                # simulate harbor re-running the deleted trials
                d = job_dir / f"{tid[:32]}__NEW"
                d.mkdir(parents=True, exist_ok=True)
                (d / "result.json").write_text(json.dumps({
                    "task_id": {"path": f"/tasks/{tid}"},
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "agent_result": {"n_input_tokens": 20, "n_output_tokens": 8}}))
            return SimpleNamespace(trial_results=[])

    monkeypatch.setattr(harbor, "Job", _FakeJob)
    items = [(Task(task_id=t, benchmark=bench, extras={"harbor_task_dir": str(task_paths[t])}),
              TaskContext(image=None)) for t in (long_id, "t_err")]

    out = HarborHarness()._run_job(items, AgentConfig(), run_dir=tmp_path, parallelism=1,
                                   resuming=True)

    # 1) harbor got the STORED full config (3 tasks), not the 2-task subset → no FileExistsError
    assert len(cap["config"].tasks) == 3
    # 2) the kept trial survived; the two to_run dirs were deleted (re-run into __NEW)
    assert (job_dir / "t_keep__A").exists()
    assert not (job_dir / f"{long_id[:32]}__B").exists() and not (job_dir / "t_err__C").exists()
    # 3) returns the re-run subset's FRESH results, keyed by FULL id (incl the truncated-dir long id)
    assert {r.task_id for r in out} == want
    assert all(r.resolved and r.reward == 1.0 for r in out)


def test_harbor_run_job_without_resuming_does_not_reconcile(monkeypatch, tmp_path) -> None:
    """A plain (non-resume) re-invocation must NOT delete/re-run existing trials — the reconcile is
    gated on ``resuming``. Without it, _run_job builds a fresh JobConfig and leaves the tree alone."""
    pytest.importorskip("harbor")
    import harbor
    from harbor.models.job.config import JobConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

    from beagle.benchmarks.harness import HarborHarness
    from beagle.types import Task, TaskContext

    bench = "deep-swe"
    job_dir = tmp_path / bench
    job_dir.mkdir(parents=True)
    task_dir = tmp_path / "tasks" / "t1"
    task_dir.mkdir(parents=True)
    stored = JobConfig(
        job_name=bench, jobs_dir=str(tmp_path), n_concurrent_trials=1,
        environment=EnvironmentConfig(import_path="pkg:Env"), agents=[AgentConfig()],
        tasks=[TaskConfig(path=str(task_dir), trial_name="t1")])
    (job_dir / "config.json").write_text(stored.model_dump_json())
    (job_dir / "t1__A").mkdir()
    (job_dir / "t1__A" / "result.json").write_text(json.dumps(
        {"task_id": {"path": "/tasks/t1"}, "verifier_result": {"rewards": {"reward": 1.0}}}))

    cap: dict = {}

    class _FakeJob:
        @classmethod
        async def create(cls, config):  # noqa: ANN001
            cap["config"] = config
            return cls()

        async def run(self):
            return SimpleNamespace(trial_results=[])

    monkeypatch.setattr(harbor, "Job", _FakeJob)
    items = [(Task(task_id="t1", benchmark=bench, extras={"harbor_task_dir": str(task_dir)}),
              TaskContext(image=None))]
    HarborHarness()._run_job(items, AgentConfig(), run_dir=tmp_path, parallelism=1)  # resuming=False

    assert (job_dir / "t1__A").exists()                      # existing trial NOT deleted
    # fresh-path config was BUILT (env = HarborHarness's import path), not the stored "pkg:Env"
    assert cap["config"].environment.import_path != "pkg:Env"


# ---------------------------------------------------------------------------
# _job_dir (runner.py)
# ---------------------------------------------------------------------------

def test_job_dir_relative_to_run_dir(tmp_path) -> None:
    """Normal case: trial_dir under run_dir → relative benchmark path."""
    rows = [{"trial_dir": str(tmp_path / "terminal_bench" / "task1")}]
    result = runner_job_dir(rows, tmp_path)
    assert result == "terminal_bench"


def test_job_dir_outside_run_dir_falls_back_to_name() -> None:
    """trial_dir NOT under run_dir → parent.name used as fallback."""
    rows = [{"trial_dir": "/some/other/path/terminal_bench/task1"}]
    result = runner_job_dir(rows, Path("/results/RID"))
    assert result == "terminal_bench"


def test_job_dir_none_trial_dir_returns_none() -> None:
    assert runner_job_dir([{"trial_dir": None}], Path("/r")) is None


def test_job_dir_empty_rows_returns_none() -> None:
    assert runner_job_dir([], Path("/r")) is None


def test_job_dir_skips_none_finds_first_valid(tmp_path) -> None:
    """Rows with some None trial_dirs → first non-None is used."""
    rows = [
        {"trial_dir": None},
        {"trial_dir": str(tmp_path / "bench" / "task2")},
    ]
    result = runner_job_dir(rows, tmp_path)
    assert result == "bench"


# ---------------------------------------------------------------------------
# _check_config_drift
# ---------------------------------------------------------------------------

def test_check_config_drift_malformed_run_json_raises_clear_error(tmp_path) -> None:
    """A malformed run.json (partial write / corruption) on --resume raises a clear
    RuntimeError with guidance — NOT an opaque JSONDecodeError (P1-3 fix)."""
    from beagle.rollout.runner import _check_config_drift

    run_dir = tmp_path / "RID"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("INVALID JSON {{{")

    with pytest.raises(RuntimeError, match="unreadable"):
        _check_config_drift(run_dir, "sha256:abc", resume=True)


def test_check_config_drift_no_hash_in_run_json_allows_resume(tmp_path) -> None:
    """run.json exists but has no config_hash → treated as no prior hash, resume allowed."""
    from beagle.rollout.runner import _check_config_drift

    run_dir = tmp_path / "RID"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"run_id": "RID"}))
    # Should NOT raise
    _check_config_drift(run_dir, "sha256:anything", resume=True)


def test_check_config_drift_no_run_json_allows_resume(tmp_path) -> None:
    """No run.json yet → resume is a no-op, no drift raised."""
    from beagle.rollout.runner import _check_config_drift

    run_dir = tmp_path / "NEW"
    run_dir.mkdir()
    _check_config_drift(run_dir, "sha256:abc", resume=True)  # must not raise


def test_check_config_drift_resume_false_ignores_existing(tmp_path) -> None:
    """resume=False → drift is never checked, even with a changed hash."""
    from beagle.rollout.runner import _check_config_drift

    run_dir = tmp_path / "RID"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"config_hash": "sha256:old"}))
    # resume=False → should NOT raise even with different hash
    _check_config_drift(run_dir, "sha256:different", resume=False)
