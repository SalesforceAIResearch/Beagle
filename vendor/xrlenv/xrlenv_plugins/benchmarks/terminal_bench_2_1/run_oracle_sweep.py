#!/usr/bin/env python3
"""Run the terminal-bench-2-1 oracle sweep against the patched cache.

Runs harbor's OracleAgent for every task in the patched shared cache and
reports which tasks the oracle actually solves (verifier reward > 0). This
is the corpus-quality gate: a task the *oracle* can't solve is poison for
RL training — its reward ceiling is 0, so no policy can ever earn signal —
and for this dataset an oracle failure usually means an unpinned dependency
drifted on PyPI (see ``build_cache.py``'s ``PATCHES``). Workflow: run this
sweep → add a pin for each failure → re-run → confirm green.

Reads tasks from ``$XRLENV_BENCHMARK_CACHE/terminal-bench-2-1`` (the cache
``build_cache.py`` writes). **Runs ALL tasks by default**; pass ``--tasks``
to select a subset.

Each task's oracle runs **on the xrlenv cluster** — the runner uses the
cluster harbor environment, which routes every container acquire/exec
through the control plane to a scheduler-chosen node. So it needs a running
control plane, addressed via env vars (the same ones the docker-py drop-in
and the harbor cluster plug-in read):

    export XRLENV_GRPC_HOST=<control-plane-host>   # required
    export XRLENV_GRPC_PORT=50051                  # optional (default 50051)
    export XRLENV_CONSUMER_TOKEN=<token>           # required if CP has auth
    export XRLENV_GRPC_SECURE=true                 # optional (TLS)

Each task's image must be present on the nodes (the cluster pulls each
task's ``docker_image`` on first acquire; tasks without a prebuilt need an
operator-built+pushed image). The oracle also needs network inside the
container (it ``git clone``s and ``pip install``s). Per-trial artifacts
(``agent/oracle.txt``, ``verifier/reward.txt``, ...) are downloaded back to
``<jobs-dir>/<job-id>/`` (harbor convention) — default jobs-dir is the repo's
gitignored ``tmp/`` and job-id is a timestamped
``tb21-oracle-sweep__YYYY-MM-DD__HH-MM-SS``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Any

# Dataset shard subdir name — must match build_cache.py's DATASET_DIR (the
# name harbor's export produces and the shard consumers' _locate_task_dir
# sees). Kept local so this runner is self-contained.
DATASET_DIR = "terminal-bench-2-1"

# Cluster harbor environment: routes every container acquire/exec through
# the control plane to a scheduler-chosen node, so the sweep actually runs
# on the dev cluster (not the local Docker daemon). Reads XRLENV_GRPC_HOST /
# _PORT / _CONSUMER_TOKEN / _GRPC_SECURE from the environment.
ENV_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"

# Infra-transient errors the trial queue may retry — and ONLY these. They match on
# ``type(e).__name__`` (harbor's RetryConfig). tb2.1 is all-runc (no sysbox cap),
# so the load-bearing case here is capacity blips at high --max-workers on a small
# cluster + control-plane/node hiccups + the HF-429/connection errors on concurrent
# model/dataset pulls the oracle solves do. Acquire is harbor's FIRST (cheapest) setup
# step, so the DOMINANT retry is a fail-fast acquire — before solve.sh runs — and the
# task usually runs once. But a retry re-runs the WHOLE trial in a FRESH container, so a
# post-acquire infra error (e.g. ``NodeCommandTimeout`` on an exec) can re-execute
# solve.sh — a new container each time, so only EXTERNAL side effects carry, never the
# recorded result. The job records ONE trial result per task (waited-then-passed counts
# once, not a re-roll). Task-content failures (AgentTimeoutError, verifier errors) stay
# OUT of this set — harbor's default ``exclude_exceptions`` also lists them — so a
# genuinely-failed oracle
# (the corpus defect this sweep exists to catch) is NEVER re-rolled into a fluke
# pass. Keep this set to capacity / control-plane / node blips only.
_INFRA_RETRY_EXCEPTIONS = frozenset({
    "CapacityExhausted",   # admission queue timed out waiting for a slot
    "ControlPlaneLost",    # CP restarted under the run
    "NodeLost",            # node dropped its stream mid-acquire
    "NodeCommandTimeout",  # a node RPC deadline (teardown / exec) tripped
})


def _repo_root() -> Path:
    # xrlenv_plugins/benchmarks/terminal_bench_2_1/run_oracle_sweep.py
    #   parents: [0]=terminal_bench_2_1 [1]=benchmarks [2]=xrlenv_plugins
    #            [3]=repo root
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    """``<repo>/tmp`` — the harbor ``jobs_dir`` (a per-run ``<job-id>/`` subdir
    is created under it). The repo's ``tmp/`` is gitignored."""
    return _repo_root() / "tmp"


def _default_job_id() -> str:
    """Harbor job-run convention: a timestamped id so each run gets its own
    ``<jobs-dir>/<job-id>/`` dir instead of colliding on a fixed name."""
    return "tb21-oracle-sweep__" + _dt.datetime.now().strftime(
        "%Y-%m-%d__%H-%M-%S",
    )


def _dataset_root(args: argparse.Namespace) -> Path:
    # Hard-reject the retired cache env var/path (renamed 2026-07-31) BEFORE the
    # root read: a caller still on XRLENV_HARBOR_CACHE / .../xrlenv_harbor_cache
    # would sweep the wrong (stale/absent) cache and mistake it for a corpus defect.
    # Lazy import to match plugin style (plugin -> xrlenv is allowed).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env(args.dest)
    root = args.dest or os.environ.get("XRLENV_BENCHMARK_CACHE")
    if not root:
        raise SystemExit(
            "no cache root — pass --dest or set XRLENV_BENCHMARK_CACHE (the "
            "path build_cache.py wrote to).",
        )
    dataset_root = Path(root).expanduser() / DATASET_DIR
    if not dataset_root.is_dir():
        raise SystemExit(
            f"{dataset_root} not found — build the cache first:\n"
            f"  python xrlenv_plugins/benchmarks/terminal_bench_2_1/"
            f"build_cache.py --stage all --dest {root}",
        )
    return dataset_root


def _resolve_tasks(dataset_root: Path, tasks_arg: str | None) -> list[str]:
    if tasks_arg is not None:   # None=absent; '' is present-but-empty (audit M5)
        want = [t.strip() for t in tasks_arg.split(",") if t.strip()]
        if not want:
            raise SystemExit(f"--tasks {tasks_arg!r} selected no tasks (audit M5)")
        missing = [
            t for t in want
            if not (dataset_root / t / "solution" / "solve.sh").is_file()
        ]
        if missing:
            raise SystemExit(
                f"unknown task(s) under {dataset_root}: {missing}",
            )
        return want
    # Default: every task with a solution/solve.sh.
    found = sorted(
        p.parent.parent.name
        for p in dataset_root.glob("*/solution/solve.sh")
    )
    if not found:
        raise SystemExit(f"no tasks with solution/solve.sh under {dataset_root}")
    return found


def _build_job_config(
    *, task_ids: list[str], dataset_root: Path, jobs_dir: Path,
    job_id: str, n_concurrent_trials: int,
    override_cpus: int | None = None, override_memory_mb: int | None = None,
    cpus_multiplier: float = 1.0, memory_multiplier: float = 1.0,
    cpu_pinning: bool = False,
    timeout_multiplier: float = 1.0, retries: int = 0,
) -> Any:
    """Compose a harbor JobConfig: OracleAgent (``AgentConfig()`` default) in
    the cluster xrlenv harbor environment, one trial per task path.

    ``override_cpus`` / ``override_memory_mb`` (when set) force every task's
    cpu/mem to one absolute value instead of its ``task.toml`` value.
    ``cpus_multiplier`` / ``memory_multiplier`` instead *scale* each task's
    declared cpu/mem (e.g. 2.0 = double), preserving the corpus's relative
    sizing — a headroom / contention ablation. They compose with the absolute
    overrides (scale is applied on top of the effective value). Passed to the
    cluster plugin via ``environment.kwargs``. ``timeout_multiplier`` scales
    harbor's agent/verifier timeouts.
    """
    from harbor.models.job.config import (  # type: ignore[import-untyped]
        JobConfig,
        RetryConfig,
    )
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
        job_name=job_id,
        jobs_dir=jobs_dir,
        n_concurrent_trials=n_concurrent_trials,
        timeout_multiplier=timeout_multiplier,
        # NB: the field is ``max_retries`` (a prior ``max_attempts=`` was silently
        # dropped by pydantic → retries were always 0). include_exceptions gates
        # retries to infra-transient errors ONLY (see _INFRA_RETRY_EXCEPTIONS) so a
        # genuinely-failed oracle — the corpus defect this sweep exists to catch —
        # is never re-rolled into a fluke pass. Short near-constant backoff: the
        # acquire's own queue_timeout already provides the "wait for a slot", so we
        # want the trial back in the admission queue quickly.
        retry=RetryConfig(
            max_retries=retries,
            include_exceptions=set(_INFRA_RETRY_EXCEPTIONS),
            min_wait_sec=2.0,
            wait_multiplier=1.0,
            max_wait_sec=10.0,
        ),
        environment=EnvironmentConfig(
            import_path=ENV_IMPORT_PATH,
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            kwargs=env_kwargs,
        ),
        agents=[AgentConfig()],
        tasks=[
            TaskConfig(path=dataset_root / tid) for tid in task_ids
        ],
    )


def _task_key(trial_result: Any) -> str:
    """Requested task id = the shard dir name (basename of the task path).

    NOT ``trial_result.task_name``: that is the task.toml ``name``, which for some
    benchmarks carries a ``namespace/`` prefix (deep-swe ``datacurve/<id>``, lhtb
    ``long-horizon-terminal-bench/<id>``). The content-retry loop matches results back
    to the requested ids (bare dir names from ``--tasks`` / the shard), so it MUST key
    on the same identifier the caller asked for — else every task looks non-passing and
    the tally reads 0/N even when the oracle scored reward>0 (harbor & pier alike).
    """
    return Path(trial_result.config.task.path).name


def _trial_passes(trial_result: Any) -> tuple[bool, str | None]:
    """Pass = trial completed with verifier rewards fully populated and every
    value > 0. Returns ``(passed, reason)``; reason is None on success."""
    if trial_result.exception_info is not None:
        return False, f"exception: {trial_result.exception_info.exception_type}"
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None or not vr.rewards:
        return False, "no verifier rewards recorded"
    failures = [k for k, v in vr.rewards.items() if not (v > 0)]
    if failures:
        return False, f"non-positive reward(s): {failures}"
    return True, None


def _summarise(trial_results: list[Any], expected: int, jobs_dir: Path) -> int:
    passed = 0
    failed: list[str] = []
    print("\n=== terminal-bench-2-1 oracle sweep ===")
    for tr in sorted(trial_results, key=lambda r: r.task_name):
        ok, reason = _trial_passes(tr)
        rewards = (
            tr.verifier_result.rewards
            if tr.verifier_result is not None else None
        )
        mark = "PASS" if ok else "FAIL"
        detail = "" if ok else f"  ({reason})"
        print(f"  [{mark}] {tr.task_name:40s} rewards={rewards}{detail}")
        if ok:
            passed += 1
        else:
            failed.append(tr.task_name)

    print(f"\n{passed} / {expected} oracle(s) solved.")
    if failed:
        print(
            f"failed: {failed}\n"
            f"An oracle failure here is a corpus defect, not a model signal. "
            f"For this dataset it usually means an unpinned dependency "
            f"drifted — inspect the per-trial verifier output under "
            f"{jobs_dir}/, then add a pin to build_cache.py's PATCHES table "
            f"and re-run `build_cache.py --stage patch` + this sweep.",
        )
    return 0 if passed == expected else 1


async def _run(args: argparse.Namespace) -> int:
    import harbor  # type: ignore[import-untyped]

    if not os.environ.get("XRLENV_GRPC_HOST"):
        raise SystemExit(
            "XRLENV_GRPC_HOST is not set — this sweep runs on the xrlenv "
            "cluster, so it needs the control-plane address. Export it (plus "
            "XRLENV_CONSUMER_TOKEN if the CP has auth) before running, e.g.:\n"
            "  export XRLENV_GRPC_HOST=<control-plane-host>\n"
            "  export XRLENV_GRPC_PORT=50051\n"
            "  export XRLENV_CONSUMER_TOKEN=<token>\n"
            "Then a per-task oracle is scheduled onto a cluster node — verify "
            "with `xrlenv rollouts` / the admin panel while it runs.",
        )

    dataset_root = _dataset_root(args)
    task_ids = _resolve_tasks(dataset_root, args.tasks)

    jobs_dir = Path(args.jobs_dir).expanduser()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or _default_job_id()

    cp = os.environ.get("XRLENV_GRPC_HOST")
    cp_port = os.environ.get("XRLENV_GRPC_PORT", "50051")
    res_note = ""
    if args.cpus_multiplier != 1.0 or args.memory_multiplier != 1.0:
        res_note = (
            f", resource mult [cpu={args.cpus_multiplier}, "
            f"mem={args.memory_multiplier}]"
        )
    print(
        f"running {len(task_ids)} oracle(s) from {dataset_root} on the xrlenv "
        f"cluster at {cp}:{cp_port} (concurrency={args.max_workers}{res_note})\n"
        f"artifacts: {jobs_dir / job_id}",
        file=sys.stderr,
    )

    # Content-retry loop: run the tasks, then re-run ONLY the non-passing ones (by
    # _trial_passes — this benchmark's own rule) up to --content-retries times. A task
    # counts as solved if ANY attempt passes; a reward=0 flake (e.g. a transient DNS
    # blip) gets a fresh trial, while a genuine failure persists across attempts. This
    # is the per-task content-retry the run_full_sweep.sh wrapper used to own — now here
    # so every driver (the wrapper AND the ci runner) gets it. Distinct from --retries,
    # which is infra-transient-only and never re-rolls a reward outcome.
    best: dict[str, Any] = {}
    remaining = list(task_ids)
    cr = int(args.content_retries)
    for attempt in range(1 + cr):
        jid = job_id if attempt == 0 else f"{job_id}-retry{attempt}"
        config = _build_job_config(
            task_ids=remaining,
            dataset_root=dataset_root,
            jobs_dir=jobs_dir,
            job_id=jid,
            n_concurrent_trials=args.max_workers,
            override_cpus=args.override_cpus,
            override_memory_mb=args.override_memory_mb,
            cpus_multiplier=args.cpus_multiplier,
            memory_multiplier=args.memory_multiplier,
            cpu_pinning=args.cpu_pinning,
            timeout_multiplier=args.timeout_multiplier,
            retries=args.retries,
        )
        job = await harbor.Job.create(config)
        job_result = await job.run()
        for tr in job_result.trial_results:
            key = _task_key(tr)           # requested id (dir name), NOT the namespaced task_name
            if key not in best or (
                not _trial_passes(best[key])[0] and _trial_passes(tr)[0]
            ):
                best[key] = tr            # first result, or an upgrade fail -> pass
        remaining = [t for t in remaining
                     if t not in best or not _trial_passes(best[t])[0]]
        if not remaining:
            break
        if attempt < cr:
            print(f"content-retry {attempt + 1}/{cr}: re-running {len(remaining)} "
                  f"non-passing task(s): {', '.join(remaining)}", file=sys.stderr)
    # Fold the content-retry rounds' sibling -retryN dirs back into the attempt-0
    # dir so the operator sees ONE result set per task, not stray retry folders.
    from xrlenv_plugins.benchmarks._sweep_retry import consolidate_retry_dirs
    consolidate_retry_dirs(jobs_dir, job_id, cr)
    return _summarise(
        [best[t] for t in task_ids if t in best], len(task_ids), jobs_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # audit H9: NO prefix abbreviation (--des / --de / --d) may resolve to --dest —
        # the cache root must come only from XRLENV_BENCHMARK_CACHE, never a CLI override
        # smuggled past the wrapper's exact-form reject.
        allow_abbrev=False,
        prog="run_oracle_sweep",
        description=(
            "Run the terminal-bench-2-1 oracle against the patched cache and "
            "report which tasks the oracle solves (reward > 0). Runs ALL "
            "tasks by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help="Harbor cache ROOT (the dataset lives at "
        "<dest>/terminal-bench-2-1/). Defaults to $XRLENV_BENCHMARK_CACHE.",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids to run. Default: every task in the "
        "cache.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Trial-level concurrency (harbor n_concurrent_trials). Default 4. "
        "Each oracle runs a full container build — tune to the box.",
    )
    p.add_argument(
        "--jobs-dir",
        default=str(_default_jobs_dir()),
        help="Harbor jobs_dir; each run lands in a timestamped "
        "<jobs-dir>/<job-id>/ subdir. Defaults to the repo's gitignored "
        "tmp/ (<repo>/tmp).",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Run label under <jobs-dir>/. Defaults to a timestamped "
        "tb21-oracle-sweep__YYYY-MM-DD__HH-MM-SS (harbor convention).",
    )
    p.add_argument(
        "--override-cpus",
        type=int,
        default=None,
        help="Force every task to this many cpus instead of its task.toml "
        "value (ablation for the honor-task-resources change).",
    )
    p.add_argument(
        "--override-memory-mb",
        type=int,
        default=None,
        help="Force every task to this much memory (MiB) instead of its "
        "task.toml value.",
    )
    p.add_argument(
        "--cpus-multiplier",
        type=float,
        default=1.0,
        help="Scale every task's declared cpus by this factor (e.g. 2.0 = "
        "double), preserving relative sizing — a headroom/contention ablation. "
        "Composes with --override-cpus. Default 1.0.",
    )
    p.add_argument(
        "--memory-multiplier",
        type=float,
        default=1.0,
        help="Scale every task's declared memory by this factor (e.g. 2.0 = "
        "double). Composes with --override-memory-mb. Default 1.0.",
    )
    p.add_argument(
        "--cpu-pinning",
        action="store_true",
        help="Opt every task in the job into cpuset pinning (dedicated cores, "
        "nproc == declared cpus). Ablation knob for timing-sensitive tasks.",
    )
    p.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Scale harbor's agent/verifier timeouts (e.g. 2.0 for slower "
        "nodes). Default 1.0.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Max trial retries for INFRA-transient errors ONLY (capacity / "
        "control-plane / node blips + HF-429/connection errors on concurrent "
        "model pulls — see _INFRA_RETRY_EXCEPTIONS). Lets xrlenv absorb high "
        "--max-workers on a small cluster: an acquire that can't get a slot fails "
        "fast + retries. Common case is a fail-fast acquire (before solve.sh); a "
        "post-acquire infra error re-runs the whole attempt in a fresh container "
        "(one result per task either way — never double-counted). Task-content "
        "failures (AgentTimeoutError, verifier errors) are NEVER retried, so a broken "
        "oracle still surfaces. Default 6; a run with no capacity pressure uses 0.",
    )
    p.add_argument(
        "--content-retries",
        type=int,
        default=0,
        help="Per-TASK content-retry rounds: after a run, re-run the tasks that came "
        "back non-passing (reward=0 by _trial_passes) up to N more times; a task is "
        "solved if ANY attempt passes. Catches nondeterministic reward=0 flakes (a "
        "transient DNS/verifier blip) that --retries (infra-only) deliberately never "
        "re-rolls. Default 0 (run_full_sweep.sh + the ci runner pass 2).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from xrlenv.observability.logging import configure_logging

        configure_logging()
    except Exception:  # pragma: no cover — logging is best-effort
        pass
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
