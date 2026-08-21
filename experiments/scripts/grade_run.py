#!/usr/bin/env python3
"""Grade an existing run's patches WITHOUT re-running the agents (recover an interrupted two-phase run).

    set -a; source .env; set +a          # XRLENV_* so swebench eval routes to the cluster docker drop-in
    python experiments/scripts/grade_run.py --config <eval.yaml> --run-dir <existing run dir> [--dry-run]

For a two-phase benchmark (SWE-bench) whose agents produced ``patch.diff`` per trial but whose grading
never ran (no ``result.json``), this reconstructs ``TaskResult``s from the patches, runs the benchmark's
grader (swebench batch eval → cluster docker drop-in; NO gateway needed) at ``parallelism_eval_patches``,
writes each trial's ``result.json`` + the canonical ``<benchmark>/result.json``, and assembles
``run.json``. Afterwards ``beagle evaluate --config <eval.yaml> --resume`` reruns only the still-missing
tasks (they now have ``result.json`` and are skipped).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def _trial_tokens(agent_dir: Path) -> dict:
    """Best-effort cache-split tokens from the trial's ATIF final_metrics (for run.json aggregation)."""
    try:
        fm = json.loads((agent_dir / "trajectory.json").read_text()).get("final_metrics") or {}
    except (OSError, ValueError):
        return {}
    p = int(fm.get("total_prompt_tokens") or 0)
    cached = min(int(fm.get("total_cached_tokens") or 0), p)
    comp = int(fm.get("total_completion_tokens") or 0)
    return {"prompt": p, "completion": comp, "total": p + comp,
            "input_uncached": p - cached, "cache_read": cached, "cache_write": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run's eval config yaml")
    ap.add_argument("--run-dir", required=True, help="the existing run dir (with per-trial patch.diff)")
    ap.add_argument("--dry-run", action="store_true", help="reconstruct + report; grade NOTHING")
    args = ap.parse_args()

    import beagle as bgl
    from beagle.cli._canonical import build_evaluation, load
    from beagle.rollout import run_record as rr
    from beagle.rollout.run_id import build_run_id, compute_config_hash
    from beagle.rollout.runtime import RuntimeConfig as RtCfg
    from beagle.rollout.runtime import build_runtime
    from beagle.types import RolloutStatus, TaskResult

    run_cfg, _ = build_evaluation(load(args.config))
    run_dir = Path(args.run_dir)
    bench_name = run_cfg.benchmark.name
    bench_dir = run_dir / bench_name
    eval_par = run_cfg.parallelism_eval_patches or run_cfg.parallelism
    if not bench_dir.is_dir():
        raise SystemExit(f"no benchmark dir {bench_dir}")

    # Reconstruct a TaskResult per trial that produced a patch (result.json doesn't store the patch).
    results, no_patch = [], 0
    for td in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        pf = td / "patch.diff"
        patch = pf.read_text() if pf.exists() else ""
        if not patch.strip():
            no_patch += 1
        results.append(TaskResult(task_id=td.name, benchmark=bench_name,
                                  status=RolloutStatus.COMPLETED, patch=patch or None,
                                  tokens=_trial_tokens(td / "agent"), artifact_dir=td))
    gradeable = sum(1 for r in results if (r.patch or "").strip())
    print(f"benchmark   : {bench_name}")
    print(f"trials found: {len(results)}  (gradeable patches: {gradeable}, empty/no patch: {no_patch})")
    print(f"eval fan-out: parallelism_eval_patches={run_cfg.parallelism_eval_patches} "
          f"→ using {eval_par}")
    print(f"runtime     : {run_cfg.runtime.kind}   run_dir: {run_dir}")
    if (run_dir / "run.json").exists():
        print("  ⚠ run.json already exists — it will be overwritten")
    if args.dry_run:
        print("\nDRY RUN — grading nothing. Re-run without --dry-run to grade on the cluster.")
        return

    run_id = build_run_id(run_cfg, compute_config_hash(run_cfg.model_dump(mode="json")))
    rt = build_runtime(RtCfg(kind=run_cfg.runtime.kind, run_id=run_id))

    t0 = time.monotonic()
    ts_start = datetime.now(timezone.utc)
    print(f"\ngrading {gradeable} patches (max_workers={eval_par}) …")
    report = bgl.benchmarks.get(bench_name).grader().grade(
        results, runtime=rt, run_dir=run_dir, parallelism=eval_par)
    print(f"graded: {report.num_resolved}/{report.num_tasks} resolved  score={report.score:.4f}")

    # Assemble run.json (mirrors the Runner's post-grade steps for one benchmark group).
    rows = [rr.per_task_row(r, benchmark=bench_name) for r in results]
    wall = time.monotonic() - t0
    summary = rr.benchmark_summary(rows, score=report.score, job_dir=bench_name, wall_time_sec=wall)
    totals = rr.compute_totals(rows, num_benchmarks=1, wall_time_sec=wall)
    record = rr.assemble_run_record(
        run_id=run_id, config=run_cfg, benchmarks={bench_name: summary}, totals=totals,
        config_hash=compute_config_hash(run_cfg.model_dump(mode="json")), config_path=args.config,
        environment=rr.capture_environment(),
        timestamp_start=ts_start.isoformat(), timestamp_end=datetime.now(timezone.utc).isoformat())
    rr.write_run_json(run_dir, record)
    print(f"wrote {run_dir}/run.json  +  {bench_dir}/result.json (canonical)")
    print("\nnext: rerun the still-missing tasks with\n"
          f"  beagle evaluate --config {args.config} --resume --run-dir {run_dir}")


if __name__ == "__main__":
    main()
