#!/usr/bin/env python3
"""run_oracle_sweep.py — the SWE-bench Pro correctness gate on the xrlenv cluster.

Runs harbor's OracleAgent (``solution/solve.sh`` = the gold patch) on every task of the
populated ``swebench-pro`` shard through the shared cluster environment
(``xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster``) and reports which tasks the
oracle actually resolves (verifier reward > 0). An oracle FAIL is a corpus/plumbing defect
(reward ceiling 0 → poison for RL), never a model signal: inspect
``<jobs-dir>/<job-id>/<task>/verifier/`` (``stdout.log``, ``stderr.log``, ``output.json``,
``reward.json``) and either fix the plumbing (build_cache.py) or EXCLUDE the instance in
run_full_sweep.sh with a reason.

Mirrors ``terminal_bench_2_1/run_oracle_sweep.py``: the same ``import_path`` wiring, the same
two retry layers (``--retries`` = infra-transient exceptions only; ``--content-retries`` =
per-task re-runs of non-passing tasks, solved if ANY attempt passes), the same pass gate
(every verifier reward > 0), exit 0 iff every requested task passes.

    export XRLENV_BENCHMARK_CACHE=<cache root>      # + XRLENV_GRPC_HOST/_PORT/_TOKEN (or all of them in .env)
    .venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/run_oracle_sweep.py --tasks <id>,<id> --max-workers 4
    .venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/run_oracle_sweep.py --max-workers 32 --content-retries 1
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Any

DATASET_DIR = "swebench-pro"
ENV_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"

# Infra-transient exception TYPE NAMES the trial queue may retry — ONLY these. A task-content
# failure (verifier reward 0, patch apply failure, test timeout) is never re-rolled by this
# layer; --content-retries owns per-task re-runs.
_INFRA_RETRY_EXCEPTIONS = frozenset({
    "CapacityExhausted", "ControlPlaneLost", "NodeLost", "NodeCommandTimeout", "SessionReaped",
})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    return _repo_root() / "tmp" / "sanity-checks"


def _default_job_id() -> str:
    return "swebench-pro-oracle__" + _dt.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")


def _dataset_root(args: argparse.Namespace) -> Path:
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env(args.dest)
    root = args.dest or os.environ.get("XRLENV_BENCHMARK_CACHE")
    if not root:
        raise SystemExit("no cache root — pass --dest or set XRLENV_BENCHMARK_CACHE (the path build_cache.py wrote to).")
    from xrlenv_plugins.benchmarks.swebench_pro.build_cache import shard_dir

    dataset_root = shard_dir(root)        # <root>/swebench-pro, or .../golden_patches when only that is populated
    if not dataset_root.is_dir():
        raise SystemExit(f"{dataset_root} not found — build the cache first:\n"
                         f"  python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --stage all --dest {root}")
    return dataset_root


def _ids_from_arg(arg: str) -> list[str]:
    """A file of ids (one per line, ``#`` comments) or a comma list. Pro ids are ~100 chars, so a comma list
    of a few of them exceeds NAME_MAX — never stat() a string that contains a comma."""
    if "," not in arg:
        try:
            p = Path(arg)
            if p.is_file():
                return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        except OSError:
            pass
    return [t.strip() for t in arg.split(",") if t.strip()]


def _resolve_tasks(dataset_root: Path, tasks_arg: str | None) -> list[str]:
    """``--tasks`` = comma list or a file of ids (one per line, ``#`` comments); default = every
    task with a solution/solve.sh under the shard."""
    if tasks_arg is not None:
        want = _ids_from_arg(tasks_arg)
        if not want:
            raise SystemExit(f"--tasks {tasks_arg!r} selected no tasks")
        missing = [t for t in want if not (dataset_root / t / "solution" / "solve.sh").is_file()]
        if missing:
            raise SystemExit(f"unknown task(s) under {dataset_root}: {missing[:5]}{' …' if len(missing) > 5 else ''}")
        return want
    found = sorted(p.parent.parent.name for p in dataset_root.glob("*/solution/solve.sh"))
    if not found:
        raise SystemExit(f"no tasks with solution/solve.sh under {dataset_root}")
    return found


def _build_job_config(*, task_ids: list[str], dataset_root: Path, jobs_dir: Path, job_id: str, n_concurrent_trials: int,
                      override_cpus: int | None = None, override_memory_mb: int | None = None, cpus_multiplier: float = 1.0,
                      memory_multiplier: float = 1.0, cpu_pinning: bool = True, timeout_multiplier: float = 1.0,
                      retries: int = 0) -> Any:
    """One harbor JobConfig: OracleAgent in the cluster environment, one trial per task path.
    ``cpu_pinning`` defaults ON: the Go/JS suites scale their worker count to nproc, and inside
    a CFS-quota container nproc is the HOST core count (a lesson from swe-rebench v2)."""
    from harbor.models.job.config import JobConfig, RetryConfig  # type: ignore[import-untyped]
    from harbor.models.trial.config import (  # type: ignore[import-untyped]
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
    )

    env_kwargs: dict[str, Any] = {}
    if cpus_multiplier != 1.0:
        env_kwargs["xrlenv_cpu_multiplier"] = cpus_multiplier
    if memory_multiplier != 1.0:
        env_kwargs["xrlenv_mem_multiplier"] = memory_multiplier
    if cpu_pinning:
        env_kwargs["xrlenv_cpu_pinning"] = True
    return JobConfig(
        job_name=job_id, jobs_dir=jobs_dir, n_concurrent_trials=n_concurrent_trials, timeout_multiplier=timeout_multiplier,
        retry=RetryConfig(max_retries=retries, include_exceptions=set(_INFRA_RETRY_EXCEPTIONS), min_wait_sec=2.0,
                          wait_multiplier=1.0, max_wait_sec=10.0),
        environment=EnvironmentConfig(import_path=ENV_IMPORT_PATH, override_cpus=override_cpus,
                                      override_memory_mb=override_memory_mb, kwargs=env_kwargs),
        agents=[AgentConfig()],
        tasks=[TaskConfig(path=dataset_root / tid) for tid in task_ids],
    )


def _task_key(trial_result: Any) -> str:
    """The requested id = the shard dir name (NOT task.toml's namespaced ``name``)."""
    return Path(trial_result.config.task.path).name


REWARD_KEY = "reward"


def _trial_passes(trial_result: Any) -> tuple[bool, str | None]:
    """Pass = completed, verifier rewards recorded, ``rewards["reward"] > 0`` (upstream's resolved flag).
    The other numeric fields in reward.json (f2p/p2p counts, n_parsed …) are diagnostics — an instance
    with zero PASS_TO_PASS tests legitimately reports ``p2p_total: 0``."""
    if trial_result.exception_info is not None:
        return False, f"exception: {trial_result.exception_info.exception_type}"
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None or vr.rewards.get(REWARD_KEY) is None:
        return False, "no verifier reward recorded"
    r = vr.rewards[REWARD_KEY]
    if not (r > 0):
        detail = {k: v for k, v in vr.rewards.items() if k in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")}
        return False, f"reward={r} {detail}"
    return True, None


def _summarise(trial_results: list[Any], expected: int, jobs_dir: Path, job_id: str) -> int:
    passed, failed = 0, []
    print("\n=== swebench-pro oracle sweep ===")
    for tr in sorted(trial_results, key=_task_key):
        ok, reason = _trial_passes(tr)
        rewards = tr.verifier_result.rewards if tr.verifier_result is not None else None
        print(f"  [{'PASS' if ok else 'FAIL'}] {_task_key(tr):96s} rewards={rewards}{'' if ok else '  (' + str(reason) + ')'}")
        passed += ok
        if not ok:
            failed.append(_task_key(tr))
    print(f"\n{passed} / {expected} oracle(s) resolved.")
    if failed:
        print(f"failed: {failed}\nAn oracle failure here is a corpus/plumbing defect, not a model signal. Inspect "
              f"{jobs_dir / job_id}/<task>/verifier/{{stdout.log,stderr.log,output.json,reward.json}}; fix build_cache.py "
              f"or EXCLUDE the instance (with a reason) in run_full_sweep.sh.")
    return 0 if passed == expected else 1


async def _run(args: argparse.Namespace) -> int:
    import harbor  # type: ignore[import-untyped]

    if not os.environ.get("XRLENV_GRPC_HOST"):
        raise SystemExit("XRLENV_GRPC_HOST is not set — the sweep runs on the xrlenv cluster; export XRLENV_GRPC_HOST / "
                         "XRLENV_GRPC_PORT / XRLENV_CONSUMER_TOKEN (source .env) first.")
    dataset_root = _dataset_root(args)
    task_ids = _resolve_tasks(dataset_root, args.tasks)
    jobs_dir = Path(args.jobs_dir).expanduser()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or _default_job_id()
    print(f"running {len(task_ids)} oracle(s) from {dataset_root} on the xrlenv cluster at "
          f"{os.environ['XRLENV_GRPC_HOST']}:{os.environ.get('XRLENV_GRPC_PORT', '50051')} (concurrency={args.max_workers})\n"
          f"artifacts: {jobs_dir / job_id}", file=sys.stderr)

    best: dict[str, Any] = {}
    remaining = list(task_ids)
    cr = int(args.content_retries)
    for attempt in range(1 + cr):
        jid = job_id if attempt == 0 else f"{job_id}-retry{attempt}"
        config = _build_job_config(task_ids=remaining, dataset_root=dataset_root, jobs_dir=jobs_dir, job_id=jid,
                                   n_concurrent_trials=args.max_workers, override_cpus=args.override_cpus,
                                   override_memory_mb=args.override_memory_mb, cpus_multiplier=args.cpus_multiplier,
                                   memory_multiplier=args.memory_multiplier, cpu_pinning=args.cpu_pinning,
                                   timeout_multiplier=args.timeout_multiplier, retries=args.retries)
        job = await harbor.Job.create(config)
        job_result = await job.run()
        for tr in job_result.trial_results:
            key = _task_key(tr)
            if key not in best or (not _trial_passes(best[key])[0] and _trial_passes(tr)[0]):
                best[key] = tr
        remaining = [t for t in remaining if t not in best or not _trial_passes(best[t])[0]]
        if not remaining:
            break
        if attempt < cr:
            print(f"content-retry {attempt + 1}/{cr}: re-running {len(remaining)} non-passing task(s)", file=sys.stderr)
    from xrlenv_plugins.benchmarks._sweep_retry import consolidate_retry_dirs
    consolidate_retry_dirs(jobs_dir, job_id, cr)
    return _summarise([best[t] for t in task_ids if t in best], len(task_ids), jobs_dir, job_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swebench-pro run_oracle_sweep", allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=None, help="cache ROOT (default $XRLENV_BENCHMARK_CACHE); the shard is <root>/swebench-pro (or <root>/swebench-pro/golden_patches when only that level is populated)")
    p.add_argument("--tasks", default=None, help="comma list of instance ids OR a file of ids (default: every populated task)")
    p.add_argument("--max-workers", type=int, default=8, help="trial concurrency (the admission queue gates load; size it to free cluster cpu/mem)")
    p.add_argument("--jobs-dir", default=str(_default_jobs_dir()))
    p.add_argument("--job-id", default=None)
    p.add_argument("--retries", type=int, default=6, help="per-trial INFRA-only retries (CapacityExhausted, ControlPlaneLost, …)")
    p.add_argument("--content-retries", type=int, default=0, help="per-task re-runs of non-passing tasks (solved if ANY attempt passes)")
    p.add_argument("--timeout-multiplier", type=float, default=1.0)
    p.add_argument("--override-cpus", type=int, default=None)
    p.add_argument("--override-memory-mb", type=int, default=None)
    p.add_argument("--cpus-multiplier", type=float, default=1.0)
    p.add_argument("--memory-multiplier", type=float, default=1.0)
    p.add_argument("--cpu-pinning", dest="cpu_pinning", action="store_true", default=True)
    p.add_argument("--no-cpu-pinning", dest="cpu_pinning", action="store_false")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from xrlenv.observability.logging import configure_logging
        configure_logging()
    except Exception:  # pragma: no cover
        pass
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
