"""Additional coverage for beagle/rollout/run_record.py.

Covers the branches not exercised by the primary test_run_record.py:
- ``_compact``: nested empty-dict pruning, list with mixed content, scalar passthrough
- ``per_task_row``: artifact_dir set vs None
- ``compute_totals``: zero tasks (division-by-zero guard), all resolved, all errored
- ``benchmark_summary``: all-resolved, all-errored, empty rows
- ``_sum_tokens``: missing/None token dicts
- ``write_run_json``: creates parent dirs, overwrites existing file atomically
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beagle.rollout.run_record import (
    _compact,
    benchmark_summary,
    compute_totals,
    per_task_row,
    write_run_json,
)
from beagle.types import TaskResult


# ---------------------------------------------------------------------------
# _compact
# ---------------------------------------------------------------------------

def test_compact_drops_nested_empty_dict() -> None:
    """A dict that becomes {} after pruning is itself dropped."""
    assert _compact({"outer": {"inner": None}}) == {}


def test_compact_keeps_nonempty_list_items() -> None:
    """A non-empty list is kept; empty list is dropped; list items are recursed."""
    assert _compact({"a": [], "b": [1, None]}) == {"b": [1, None]}


def test_compact_scalar_passthrough() -> None:
    assert _compact(42) == 42
    assert _compact("hello") == "hello"
    assert _compact(False) is False


def test_compact_preserves_zero_and_false_in_dict() -> None:
    """0, False are not pruned (only None / '' / {} / [] are)."""
    assert _compact({"x": 0, "y": False, "z": None}) == {"x": 0, "y": False}


# ---------------------------------------------------------------------------
# per_task_row
# ---------------------------------------------------------------------------

def test_per_task_row_with_artifact_dir(tmp_path) -> None:
    r = TaskResult(task_id="t1", resolved=True, reward=1.0,
                   artifact_dir=tmp_path / "trial")
    row = per_task_row(r, benchmark="b")
    assert row["trial_dir"] == str(tmp_path / "trial")
    assert row["task_id"] == "t1" and row["benchmark"] == "b"


def test_per_task_row_without_artifact_dir() -> None:
    r = TaskResult(task_id="t2", resolved=False)
    row = per_task_row(r, benchmark="b")
    assert row["trial_dir"] is None


def test_per_task_row_error_field() -> None:
    r = TaskResult(task_id="t3", error="boom")
    row = per_task_row(r, benchmark="b")
    assert row["error"] == "boom"
    assert row["resolved"] is False


# ---------------------------------------------------------------------------
# compute_totals
# ---------------------------------------------------------------------------

def test_compute_totals_zero_tasks_no_division_error() -> None:
    """Zero tasks → resolved_rate = 0.0, not ZeroDivisionError."""
    t = compute_totals([], num_benchmarks=0)
    assert t["resolved_rate"] == 0.0
    assert t["num_tasks"] == 0
    assert t["num_attempted"] == 0


def test_compute_totals_all_resolved() -> None:
    rows = [
        per_task_row(TaskResult(task_id="t1", resolved=True, reward=1.0,
                                tokens={"prompt": 5, "completion": 2}), benchmark="b"),
        per_task_row(TaskResult(task_id="t2", resolved=True, reward=1.0,
                                tokens={"prompt": 5, "completion": 2}), benchmark="b"),
    ]
    t = compute_totals(rows, num_benchmarks=1)
    assert t["resolved_rate"] == 1.0
    assert t["num_resolved"] == 2 and t["num_errored"] == 0
    assert t["num_attempted"] == 2


def test_compute_totals_all_errored() -> None:
    rows = [
        per_task_row(TaskResult(task_id="t1", error="err"), benchmark="b"),
        per_task_row(TaskResult(task_id="t2", error="err"), benchmark="b"),
    ]
    t = compute_totals(rows, num_benchmarks=1)
    assert t["num_errored"] == 2 and t["num_resolved"] == 0
    assert t["num_attempted"] == 0
    assert t["resolved_rate"] == 0.0


def test_compute_totals_tokens_summed_across_rows() -> None:
    rows = [
        per_task_row(TaskResult(task_id="t1", tokens={"prompt": 100, "completion": 10}), benchmark="b"),
        per_task_row(TaskResult(task_id="t2", tokens={"prompt": 50, "completion": 5}), benchmark="b"),
    ]
    t = compute_totals(rows, num_benchmarks=1)
    assert t["tokens"] == {"prompt": 150, "completion": 15, "total": 165,
                           "input_uncached": 0, "cache_read": 0, "cache_write": 0}


# ---------------------------------------------------------------------------
# benchmark_summary
# ---------------------------------------------------------------------------

def test_benchmark_summary_empty_rows() -> None:
    b = benchmark_summary([], score=0.0, job_dir=None, wall_time_sec=None)
    assert b["num_tasks"] == 0 and b["score"] == 0.0
    assert b["tokens"]["total"] == 0


def test_benchmark_summary_all_resolved() -> None:
    rows = [
        per_task_row(TaskResult(task_id="t1", resolved=True, reward=1.0,
                                tokens={"prompt": 10, "completion": 1}), benchmark="b"),
    ]
    b = benchmark_summary(rows, score=1.0, job_dir="bench", wall_time_sec=5.0)
    assert b["num_resolved"] == 1 and b["num_errored"] == 0
    assert b["score"] == 1.0 and b["job_dir"] == "bench"


# ---------------------------------------------------------------------------
# _sum_tokens: missing keys handled
# ---------------------------------------------------------------------------

def test_sum_tokens_tolerates_missing_keys() -> None:
    from beagle.rollout.run_record import _sum_tokens

    rows = [
        {"tokens": {}},           # no prompt/completion → treated as 0
        {"tokens": None},          # None → treated as {}
        {"tokens": {"prompt": 5}}, # only prompt → completion + cache buckets = 0
    ]
    assert _sum_tokens(rows) == {"prompt": 5, "completion": 0, "total": 5,
                                 "input_uncached": 0, "cache_read": 0, "cache_write": 0}


# ---------------------------------------------------------------------------
# write_run_json
# ---------------------------------------------------------------------------

def test_write_run_json_creates_parent_dirs(tmp_path) -> None:
    """Deeply nested run_dir is created automatically."""
    run_dir = tmp_path / "a" / "b" / "c"
    path = write_run_json(run_dir, {"run_id": "RID"})
    assert path == run_dir / "run.json"
    assert json.loads(path.read_text())["run_id"] == "RID"


def test_write_run_json_overwrites_existing(tmp_path) -> None:
    """Second write replaces the first atomically (no .tmp leftover)."""
    run_dir = tmp_path / "RID"
    write_run_json(run_dir, {"v": 1})
    write_run_json(run_dir, {"v": 2})
    assert json.loads((run_dir / "run.json").read_text())["v"] == 2
    assert not (run_dir / "run.json.tmp").exists()
