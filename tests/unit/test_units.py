"""Unit tests for the real (non-stub) logic that ships in Phases A–B.

Pure-Python, stdlib-only: no cluster / Docker / harbor. Complements the
shape/smoke tests in ``test_skeleton.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import beagle as bgl
from beagle.benchmarks.base import BenchmarkSpec
from beagle.benchmarks.grader import InBandGrader
from beagle.benchmarks.harness.drivers import _trial_to_result
from beagle.benchmarks.source import select_and_sample
from beagle.benchmarks.webarena_infinity import WaiSource
from beagle.registry import Registry
from beagle.rollout.runtime.runtime import ContainerResources, _resource_run_flags
from beagle.rollout.runtime.transport import GitClone, git_clone_argv
from beagle.rollout.runtime.xrlenv_runtime import _translate_run_args
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult


# --- source: select_and_sample ----------------------------------------------


def _items(*ids: str):
    return [(Task(task_id=i), TaskContext(image=None)) for i in ids]


def test_select_and_sample_order_exclude_expand() -> None:
    items = _items("t0", "t1", "t2")
    # task_ids preserves requested order (not source order)
    sel = select_and_sample(items, BenchmarkSpec(name="b", task_ids=["t2", "t0"]))
    assert [t.task_id for t, _ in sel] == ["t2", "t0"]
    # exclude applied to the full set
    sel = select_and_sample(items, BenchmarkSpec(name="b", exclude_task_ids=["t1"]))
    assert [t.task_id for t, _ in sel] == ["t0", "t2"]
    # num_samples expands each selected task (pass@k), preserving selection order
    sel = select_and_sample(items, BenchmarkSpec(name="b", task_ids=["t0"], num_samples=2))
    assert [t.task_id for t, _ in sel] == ["t0__s0", "t0__s1"]
    sel = select_and_sample(items, BenchmarkSpec(name="b", task_ids=["t0", "t1"], num_samples=2))
    assert [t.task_id for t, _ in sel] == ["t0__s0", "t0__s1", "t1__s0", "t1__s1"]


def test_select_and_sample_missing_id_raises() -> None:
    with pytest.raises(KeyError):
        select_and_sample(_items("t0"), BenchmarkSpec(name="b", task_ids=["nope"]))


def test_taskdataset_from_benchmark_routes_through_load_tasks(monkeypatch) -> None:
    # The dataloader entry: task selection lives on the BenchmarkSpec (via load_tasks),
    # not on whoever runs the eval.
    import beagle.benchmarks.registry as reg

    monkeypatch.setattr(reg, "load_tasks", lambda spec: iter(_items("a", "b")))
    ds = bgl.TaskDataset.from_benchmark(BenchmarkSpec(name="demo"))
    assert len(ds) == 2 and ds.task_ids == ["a", "b"]


# --- grader: InBandGrader (regression for the denominator fix) ---------------


def test_inband_grader_counts_errored_as_zero() -> None:
    # An errored trial (reward=None) scores 0 and stays in the denominator, so the
    # fitness isn't inflated by dropping the failure.
    results = [
        TaskResult(task_id="a", reward=1.0),
        TaskResult(task_id="b", reward=None, error="boom"),
    ]
    rep = InBandGrader().grade(results, runtime=None, run_dir=None)  # type: ignore[arg-type]
    assert rep.num_tasks == 2
    assert rep.num_resolved == 1
    assert rep.score == 0.5  # (1.0 + 0.0) / 2, NOT 1.0/1
    assert rep.per_task == {"a": 1.0, "b": 0.0}


# --- harness: _trial_to_result (duck-typed harbor TrialResult) ---------------


class _VR:
    def __init__(self, rewards):
        self.rewards = rewards


class _Exc:
    def __init__(self, t):
        self.exception_type = t


class _Trial:
    def __init__(self, name, rewards=None, exc=None):
        self.trial_name = name
        self.verifier_result = _VR(rewards) if rewards is not None else None
        self.exception_info = _Exc(exc) if exc else None

    def compute_token_cost_totals(self):
        return (100, 20, 50, 0.01)   # (n_input incl cache, n_cache, n_output, cost)


def test_trial_to_result_pass_and_error() -> None:
    ok = _trial_to_result(_Trial("chess__x", rewards={"reward": 1.0}), jobs_dir=Path("/j"), job_name="run")
    assert (ok.task_id, ok.reward, ok.resolved, ok.error) == ("chess__x", 1.0, True, None)
    assert ok.status is RolloutStatus.COMPLETED
    assert str(ok.artifact_dir) == "/j/run/chess__x"
    # harbor n_cache (20) → cache_read; prompt stays full input; input_uncached = 100 - 20
    assert ok.tokens == {"prompt": 100, "completion": 50,
                        "input_uncached": 80, "cache_read": 20, "cache_write": 0}

    partial = _trial_to_result(_Trial("y", rewards={"reward": 0.4}), jobs_dir=Path("/j"), job_name="run")
    assert partial.reward == 0.4 and not partial.resolved  # below the 1.0 threshold

    err = _trial_to_result(_Trial("z", exc="AuthDenied"), jobs_dir=Path("/j"), job_name="run")
    assert err.reward is None and not err.resolved and err.error == "AuthDenied"
    assert err.status is RolloutStatus.FAILED


def test_error_string_is_full_and_substring_matchable() -> None:
    # Fix #2: downstream infra/timeout tolerance greps the message, so capture "<type>: <msg>"
    # (not just the class name) — from both the live ExceptionInfo object and the on-disk dict.
    from beagle.benchmarks.harness.drivers import _error_string

    class _E:
        exception_type = "AgentTimeoutError"
        exception_message = "deadline exceeded"

    assert _error_string(_E()) == "AgentTimeoutError: deadline exceeded"
    assert _error_string({"exception_type": "X", "exception_message": "fetch failed"}) == "X: fetch failed"
    assert _error_string({"exception_type": "X", "exception_message": ""}) == "X"  # type-only fallback
    assert _error_string(None) is None


# --- ported runtime helpers --------------------------------------------------


def test_resource_run_flags() -> None:
    assert _resource_run_flags(ContainerResources(cpu_limit=2.0)) == ["--cpus", "2.0"]
    assert _resource_run_flags(ContainerResources(mem_limit_bytes=1024)) == ["--memory", "1024"]
    assert _resource_run_flags(ContainerResources()) == []
    with pytest.raises(NotImplementedError):
        _resource_run_flags(ContainerResources(disk_limit_bytes=1))
    with pytest.raises(NotImplementedError):
        _resource_run_flags(ContainerResources(gpus=1))


def test_translate_run_args() -> None:
    assert _translate_run_args(["--entrypoint", ""]) == {"entrypoint": [""]}
    assert _translate_run_args(["--entrypoint", "x"]) == {"entrypoint": "x"}
    assert _translate_run_args([]) == {}
    with pytest.raises(NotImplementedError):
        _translate_run_args(["--bogus", "1"])
    with pytest.raises(ValueError):
        _translate_run_args(["--entrypoint"])


# --- git_clone_argv: SHA vs branch/tag (regression for the --branch-rejects-SHA bug)


def test_git_clone_argv_branch_uses_fast_path() -> None:
    argv = git_clone_argv(GitClone(repo_url="https://h/r", ref="develop", container_path="/a"))
    assert argv[:5] == ["git", "clone", "--depth", "1", "--branch"]
    assert "develop" in argv and "/a" in argv


def test_git_clone_argv_sha_fetches_and_checks_out() -> None:
    # A commit SHA can't be cloned via --branch; must fetch-by-commit + checkout.
    sha = "b8264d2b8b8c5ddf6d5eb4ad8d48cc9fea89552b"
    argv = git_clone_argv(GitClone(repo_url="https://h/r", ref=sha, container_path="/a"))
    assert argv[0] == "sh"
    script = argv[2]
    assert "--branch" not in script
    assert "git fetch -q --depth 1 origin" in script and "git checkout -q FETCH_HEAD" in script
    assert argv[-3:] == ["https://h/r", sha, "/a"]  # repo/ref/dir passed as args, not interpolated


def test_git_clone_argv_empty_ref_clones_default_branch() -> None:
    # A ref-less clone must NOT pass --branch (git clone --branch HEAD/"" fails).
    argv = git_clone_argv(GitClone(repo_url="https://h/r", ref="", container_path="/a"))
    assert "--branch" not in argv
    assert argv == ["git", "clone", "--depth", "1", "https://h/r", "/a"]


def test_git_clone_argv_token_injected_by_name_only() -> None:
    # token appears only as the var name in the wrapper; branch + SHA both wrap.
    for ref in ("main", "b8264d2b8b8c5ddf6d5eb4ad8d48cc9fea89552b"):
        argv = git_clone_argv(GitClone("https://h/r", ref, "/a", token_env="GH_TOKEN"))
        assert argv[0] == "sh" and "$GH_TOKEN" in argv[2]
        assert "ghp_" not in argv[2]  # no secret value, only the name


# --- registry ----------------------------------------------------------------


def test_registry_dup_and_missing() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    assert reg.get("a") == 1 and "a" in reg
    with pytest.raises(KeyError):
        reg.register("a", 2)  # duplicate
    with pytest.raises(KeyError):
        reg.get("missing")


# --- config: fail-loud on unknown keys --------------------------------------


def test_config_rejects_unknown_agent_key() -> None:
    with pytest.raises(ValueError):
        bgl.BeagleConfig.from_dict(
            {"evolvee": {"name": "monet", "bogus": 1}, "evolver": {"name": "cursor"}}
        )


# --- WaiSource stub is loud (not silent-empty) -------------------------------


def test_wai_source_raises_not_silent_empty() -> None:
    with pytest.raises(NotImplementedError):
        list(WaiSource().tasks(BenchmarkSpec(name="webarena-infinity")))
