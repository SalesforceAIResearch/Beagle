"""Additional coverage for beagle/algorithms/darwinx/eval.py.

Covers the branches not exercised by test_darwinx_eval.py:
- ``to_darwinx_run_json``: empty results list, reward=None+resolved=True fallback,
  all-errored, partial reward
- ``write_darwinx_run_json``: no prior run.json (run.beagle.json not created),
  called twice (second call overwrites run.beagle.json — document the behavior)
- ``_effective_reward``: all three branches
- ``translate_config``: minimal config (no runtime), missing runtime block
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beagle.algorithms.darwinx import eval as ev
from beagle.types import TaskResult


# ---------------------------------------------------------------------------
# _effective_reward
# ---------------------------------------------------------------------------

def test_effective_reward_numeric() -> None:
    assert ev._effective_reward({"reward": 0.5}) == 0.5
    assert ev._effective_reward({"reward": 1.0}) == 1.0
    assert ev._effective_reward({"reward": 0.0}) == 0.0


def test_effective_reward_none_resolved_true_gives_one() -> None:
    assert ev._effective_reward({"reward": None, "resolved": True}) == 1.0


def test_effective_reward_none_resolved_false_gives_zero() -> None:
    assert ev._effective_reward({"reward": None, "resolved": False}) == 0.0


# ---------------------------------------------------------------------------
# to_darwinx_run_json: edge cases
# ---------------------------------------------------------------------------

def test_to_darwinx_run_json_empty_results() -> None:
    d = ev.to_darwinx_run_json([])
    assert d["per_task_results"] == []
    assert d["errors"] == []
    assert d["totals"] == {"num_tasks": 0, "num_tasks_resolved": 0, "num_tasks_errored": 0}


def test_to_darwinx_run_json_reward_none_resolved_true_counts_as_resolved() -> None:
    """reward=None + resolved=True → effective reward=1.0 → counts toward num_tasks_resolved."""
    r = TaskResult(task_id="t1", resolved=True, reward=None)
    d = ev.to_darwinx_run_json([r])
    assert d["totals"]["num_tasks_resolved"] == 1


def test_to_darwinx_run_json_partial_reward_not_resolved_in_totals() -> None:
    """reward=0.5 < 1.0 → not counted as resolved in totals (effective_reward < 1.0)."""
    r = TaskResult(task_id="t1", resolved=False, reward=0.5)
    d = ev.to_darwinx_run_json([r])
    assert d["totals"]["num_tasks_resolved"] == 0
    assert d["totals"]["num_tasks_errored"] == 0


def test_to_darwinx_run_json_all_errored() -> None:
    """All tasks errored → errors list populated, num_tasks_resolved=0."""
    results = [
        TaskResult(task_id="t1", resolved=False, reward=None, error="boom1"),
        TaskResult(task_id="t2", resolved=False, reward=None, error="boom2"),
    ]
    d = ev.to_darwinx_run_json(results)
    assert d["totals"]["num_tasks_errored"] == 2
    assert d["totals"]["num_tasks_resolved"] == 0
    assert len(d["errors"]) == 2
    # error messages are full strings (for substring matching by the algorithm)
    assert all(e["message"] for e in d["errors"])


def test_to_darwinx_run_json_suffix_stripped_from_task_id() -> None:
    """Harbor trial suffix (e.g. __ABC) is stripped from task_id in per_task_results."""
    r = TaskResult(task_id="my-task__CAFEBABE", resolved=True, reward=1.0)
    d = ev.to_darwinx_run_json([r])
    assert d["per_task_results"][0]["task_id"] == "my-task"


def test_to_darwinx_run_json_no_suffix_unchanged() -> None:
    """A task_id without a __ suffix is kept as-is."""
    r = TaskResult(task_id="plain-task", resolved=True, reward=1.0)
    d = ev.to_darwinx_run_json([r])
    assert d["per_task_results"][0]["task_id"] == "plain-task"


# ---------------------------------------------------------------------------
# write_darwinx_run_json: run.beagle.json preservation
# ---------------------------------------------------------------------------

def test_write_darwinx_run_json_no_prior_run_json_creates_no_beagle_json(tmp_path) -> None:
    """If run.json doesn't exist yet, there's nothing to preserve → run.beagle.json absent."""
    ev.write_darwinx_run_json(tmp_path, [])
    assert not (tmp_path / "run.beagle.json").exists()
    assert (tmp_path / "run.json").exists()


def test_write_darwinx_run_json_preserves_clean_summary(tmp_path) -> None:
    """When run.json exists (beagle clean shape), it's renamed to run.beagle.json."""
    (tmp_path / "run.json").write_text('{"benchmarks": {}, "totals": {}}')
    ev.write_darwinx_run_json(tmp_path, [])

    compat = json.loads((tmp_path / "run.json").read_text())
    assert "per_task_results" in compat
    clean = json.loads((tmp_path / "run.beagle.json").read_text())
    assert "benchmarks" in clean


def test_run_eval_supports_pass_at_k(tmp_path) -> None:
    """num_samples>1 is supported: the eval runs k samples and to_darwinx_run_json collapses them
    to k rows under the base id (the driver's avg@k merge groups them by that id)."""
    from types import SimpleNamespace

    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: {name: gpt-5.5}\nagent: {name: monet, config: {}}\n"
                   "benchmark: {name: terminal_bench_2_1, task_ids: [t1], num_samples: 3}\n")
    samples = [TaskResult(task_id=f"t1__s{i}", resolved=(i == 1), reward=float(i == 1))
               for i in range(3)]
    run_dir = ev.run_eval(cfg, results_root=tmp_path / "out", run_id="RID",
                          _evaluate=lambda config, **k: SimpleNamespace(results=samples))
    run_json = json.loads((run_dir / "run.json").read_text())
    assert [r["task_id"] for r in run_json["per_task_results"]] == ["t1", "t1", "t1"]
    assert run_json["totals"] == {"num_tasks": 1, "num_tasks_resolved": 1, "num_tasks_errored": 0}


def test_write_darwinx_run_json_is_idempotent_preserves_first_clean(tmp_path) -> None:
    """A second call must NOT clobber the preserved clean summary with the (already
    reshaped) compat run.json — the FIRST clean summary is kept (P1-4 fix)."""
    (tmp_path / "run.json").write_text('{"benchmarks": {"b": {"score": 0.5}}}')  # clean shape
    ev.write_darwinx_run_json(tmp_path, [])                   # first call: clean → beagle.json
    ev.write_darwinx_run_json(tmp_path, [])                   # second call: must not overwrite it
    preserved = json.loads((tmp_path / "run.beagle.json").read_text())
    assert preserved == {"benchmarks": {"b": {"score": 0.5}}}  # the ORIGINAL clean summary
    assert "per_task_results" in json.loads((tmp_path / "run.json").read_text())


# ---------------------------------------------------------------------------
# translate_config: minimal + consumer_token rename
# ---------------------------------------------------------------------------

def test_translate_config_minimal_no_runtime() -> None:
    """A config with no runtime block → RunConfig built with defaults."""
    cfg = ev.translate_config({
        "model": {"name": "gpt-5.5"},
        "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1"},
    })
    assert cfg.model.name == "gpt-5.5"
    assert cfg.agent.name == "monet"
    assert cfg.benchmark.name == "terminal_bench_2_1"


def test_translate_config_consumer_token_renamed_to_token() -> None:
    cfg = ev.translate_config({
        "model": {"name": "gpt-5.5"},
        "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1"},
        "runtime": {"kind": "xrlenv-cluster", "consumer_token": "TOK"},
    })
    assert cfg.runtime.token == "TOK"


def test_translate_config_grpc_secure_dropped_silently() -> None:
    """grpc_secure is an unexpressible field — it's dropped without error."""
    cfg = ev.translate_config({
        "model": {"name": "gpt-5.5"},
        "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1"},
        "runtime": {"kind": "xrlenv-cluster", "grpc_secure": True},
    })
    assert cfg.runtime.kind == "xrlenv-cluster"


def test_translate_config_unknown_field_fails_loud() -> None:
    """RunConfig has extra='forbid' — unknown fields from coding-bench divergence fail loud."""
    with pytest.raises(Exception, match="extra"):
        ev.translate_config({
            "model": {"name": "gpt-5.5"},
            "agent": {"name": "monet", "config": {}},
            "benchmark": {"name": "terminal_bench_2_1"},
            "unknown_field": "oops",
        })


# ---------------------------------------------------------------------------
# Pass@k + suffix stripping interaction (design risk regression)
# ---------------------------------------------------------------------------

def test_pass_at_k_task_ids_collapse_to_base_id_in_darwinx() -> None:
    """When num_samples>1, to_darwinx_run_json strips the __s0/__s1 suffix.
    Both samples become the same base task_id in per_task_results — the algorithm
    would average them (or see duplicates). This is a known design risk when
    combining pass@k with the DarwinX adapter; document it with a test."""
    results = [
        TaskResult(task_id="orig__s0", resolved=True, reward=1.0),
        TaskResult(task_id="orig__s1", resolved=False, reward=0.0),
    ]
    d = ev.to_darwinx_run_json(results)
    task_ids = [r["task_id"] for r in d["per_task_results"]]
    # Both collapse to 'orig' — this IS the current behavior; the test documents it.
    assert task_ids == ["orig", "orig"]
    # The algorithm would see two rows for 'orig'; behavior depends on its deduplication.
    # This is a P1 design risk: DarwinX adapter should not be used with num_samples > 1
    # until explicit pass@k support is added.


# ---------------------------------------------------------------------------
# pass@k: k samples collapse to k rows under one base id (the driver groups them)
# ---------------------------------------------------------------------------

def test_base_task_id_strips_trial_and_sample_suffix() -> None:
    assert ev._base_task_id("ars__s1__deadbeef") == "ars"   # harbor trial + pass@k sample
    assert ev._base_task_id("ars__s1") == "ars"             # sample only
    assert ev._base_task_id("ars__deadbeef") == "ars"       # trial only
    assert ev._base_task_id("ars") == "ars"


def test_to_darwinx_run_json_passk_collapses_and_counts_pass_at_k() -> None:
    # pass@3 on one task: samples __s0/__s1/__s2 → 3 rows under the base id; totals over the 1
    # distinct task, resolved because one sample passed.
    results = [
        TaskResult(task_id="ars__s0", resolved=False, reward=0.0),
        TaskResult(task_id="ars__s1", resolved=True, reward=1.0),
        TaskResult(task_id="ars__s2", resolved=False, reward=0.0),
    ]
    d = ev.to_darwinx_run_json(results)
    assert [row["task_id"] for row in d["per_task_results"]] == ["ars", "ars", "ars"]
    assert d["totals"] == {"num_tasks": 1, "num_tasks_resolved": 1, "num_tasks_errored": 0}


def test_to_darwinx_run_json_passk_all_fail_not_resolved() -> None:
    results = [TaskResult(task_id=f"ars__s{i}", resolved=False, reward=0.0) for i in range(3)]
    d = ev.to_darwinx_run_json(results)
    assert d["totals"] == {"num_tasks": 1, "num_tasks_resolved": 0, "num_tasks_errored": 0}
