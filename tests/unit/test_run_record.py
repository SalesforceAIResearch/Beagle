"""The run.json record — thin benchmark-keyed summary + resume ledger + atomic write."""

from __future__ import annotations

import json

from beagle.config import RunConfig
from beagle.rollout.run_record import (
    assemble_run_record,
    benchmark_summary,
    compact_config,
    compute_totals,
    per_task_row,
    write_run_json,
)
from beagle.types import TaskResult


def _rows() -> list[dict]:
    return [per_task_row(r, benchmark="terminal_bench_2_1") for r in [
        TaskResult(task_id="t1", resolved=True, reward=1.0, tokens={"prompt": 100, "completion": 10}),
        TaskResult(task_id="t2", resolved=False, reward=0.0, tokens={"prompt": 50, "completion": 5}),
        TaskResult(task_id="t3", error="boom"),  # infra crash — attempted excludes it
    ]]


def test_compute_totals_is_additive_only() -> None:
    t = compute_totals(_rows(), num_benchmarks=1, wall_time_sec=12.5)
    assert t["num_benchmarks"] == 1 and t["num_tasks"] == 3 and t["num_attempted"] == 2
    assert t["num_resolved"] == 1 and t["num_errored"] == 1
    assert t["resolved_rate"] == 1 / 3          # a raw count ratio, not a cross-benchmark score
    assert "score" not in t                     # score is per-benchmark only (not additive)
    assert t["tokens"] == {"prompt": 150, "completion": 15, "total": 165,
                           "input_uncached": 0, "cache_read": 0, "cache_write": 0}
    assert t["wall_time_sec"] == 12.5


def test_benchmark_summary_derives_aggregate() -> None:
    b = benchmark_summary(_rows(), score=0.5, job_dir="terminal_bench_2_1", wall_time_sec=3.0)
    assert b["score"] == 0.5 and b["job_dir"] == "terminal_bench_2_1"
    assert b["num_tasks"] == 3 and b["num_resolved"] == 1 and b["num_errored"] == 1
    assert b["tokens"]["total"] == 165


def test_compact_config_drops_empties_and_defaults() -> None:
    cfg = RunConfig.from_dict({
        "model": {"name": "gpt-5.5"},  # provider "" / api_base None / params {} should vanish
        "agent": {"name": "monet", "config": {"monet_args": ["--provider", "gw"]}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},  # num_samples=1, tag="main" default
        "runtime": {"kind": "xrlenv-cluster"}, "parallelism": 2,
    })
    c = compact_config(cfg)
    assert c["model"] == {"name": "gpt-5.5"}          # empties pruned
    assert c["agent"]["config"]["monet_args"] == ["--provider", "gw"]
    assert c["benchmark"] == {"name": "terminal_bench_2_1", "task_ids": ["t1"]}  # tag/num_samples dropped
    assert "tag" not in c["benchmark"] and "num_samples" not in c["benchmark"]
    assert c["runtime"] == {"kind": "xrlenv-cluster"} and c["parallelism"] == 2  # non-defaults kept


def test_assemble_thin_record_and_atomic_write(tmp_path) -> None:
    cfg = RunConfig.from_dict({
        "model": {"name": "gpt-5.5"},
        "agent": {"name": "monet", "config": {"x": 1}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1", "t2", "t3"]},
    })
    rows = _rows()
    benches = {"terminal_bench_2_1": benchmark_summary(
        rows, score=0.5, job_dir="terminal_bench_2_1", wall_time_sec=3.0)}
    rec = assemble_run_record(
        run_id="RID", config=cfg, benchmarks=benches,
        totals=compute_totals(rows, num_benchmarks=1), config_hash="sha256:abc", config_path="c.yaml",
    )
    run_dir = tmp_path / "RID"
    back = json.loads(write_run_json(run_dir, rec).read_text())

    # Thin, benchmark-keyed: config embedded (compacted), NO top-level model/agent/metrics/per_task.
    assert back["run_id"] == "RID" and back["config_hash"] == "sha256:abc"
    assert back["config"]["model"] == {"name": "gpt-5.5"}
    assert "model" not in back and "per_task_results" not in back and "metrics" not in back
    assert back["benchmarks"]["terminal_bench_2_1"]["score"] == 0.5
    assert back["benchmarks"]["terminal_bench_2_1"]["job_dir"] == "terminal_bench_2_1"
    assert back["totals"]["num_tasks"] == 3 and back["totals"]["resolved_rate"] == 1 / 3
    assert not (run_dir / "run.json.tmp").exists()  # atomic: no leftover temp
    assert not (run_dir / "tasks.jsonl").exists()   # no house ledger — derived from harbor trees
