#!/usr/bin/env python3
"""Run the DeepSWE oracle sweep against the shared cache, on the xrlenv cluster.

Runs pier's default ``OracleAgent`` (applies the task's ``solution/`` reference and
commits) for every task in the ``deep-swe`` shard (built by ``build_cache.py``) and
reports which the oracle solves (verifier ``reward > 0``). This is the corpus-quality
gate: a task the *oracle* can't solve is poison for RL — its reward ceiling is 0 — so
under the oracle a non-passing task is a plumbing/content bug, not a model signal.

Each task runs **on the xrlenv cluster** via the pier cluster environment
(``xrlenv_plugins.pier:XrlenvPierEnvironmentCluster``), which routes every container
acquire/exec through the control plane to a scheduler-chosen node. So it needs a
running control plane, addressed via env:

    export XRLENV_GRPC_HOST=<control-plane-host>   # required
    export XRLENV_GRPC_PORT=50051                  # optional (default 50051)
    export XRLENV_CONSUMER_TOKEN=<token>           # required if CP has auth
    export XRLENV_GRPC_SECURE=true                 # optional (TLS)

**Images.** DeepSWE tasks ship a prebuilt ``[environment] docker_image`` on public
ECR, so — like terminal-bench-2-1 — the cluster resolves each task's image directly
from its ``docker_image`` (no ``IMAGE_TEMPLATE`` needed). Warm them onto nodes ahead
of time with ``build_plan_gen.py`` + ``xrlenv build apply`` (Path 1, direct ECR
pull), or rely on lazy first-acquire pull. See the plan §3b for the optional
pull-through-mirror rewrite (Path 2).

**Separate verifier.** DeepSWE grades in a fresh container (``environment_mode=
"separate"``); the pier cluster env resolves the verifier base image from the tests
``FROM`` / parent ``docker_image`` and uploads ``/tests`` itself (the plugin handles
this — no sweep flag needed).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Any

# Dataset shard subdir name — must match build_cache.py's SHARD.
SHARD = "deep-swe"

ENV_IMPORT_PATH = "xrlenv_plugins.pier:XrlenvPierEnvironmentCluster"

# Infra-transient errors the trial queue may retry — and ONLY these. They match on
# ``type(e).__name__`` (pier's RetryConfig.include_exceptions). This lets a consumer
# request any concurrency while xrlenv transparently paces a capacity-CAPPED runtime:
# an at-cap acquire fails fast with ``CapacityExhausted`` (see the pier env's
# ``_ACQUIRE_QUEUE_TIMEOUT_S``) and the trial queue retries. Acquire is the first
# (cheapest) setup step, so the DOMINANT retry is a fail-fast acquire — before
# ``solve.sh`` runs — and the task usually runs once. But a retry re-runs the WHOLE
# trial in a FRESH container, so a post-acquire infra error (e.g. ``NodeCommandTimeout``
# on an exec) can re-execute ``solve.sh`` — a new container each time, so only EXTERNAL
# side effects carry, never the recorded result. ``job.run()`` records ONE trial result
# per task; a waited-then-passed task counts once (not double-counted). Task-content
# failures (AgentTimeoutError, verifier errors) are deliberately OUT of this set — a
# genuinely-failed task is never re-rolled.
_INFRA_RETRY_EXCEPTIONS = frozenset({
    "CapacityExhausted",   # admission queue timed out waiting for a runtime slot
    "ControlPlaneLost",    # CP restarted under the run
    "NodeLost",            # node dropped its stream mid-acquire
    "NodeCommandTimeout",  # a node RPC deadline (teardown / exec) tripped
})


def _repo_root() -> Path:
    # xrlenv_plugins/benchmarks/deep_swe/run_oracle_sweep.py
    #   parents: [0]=deep_swe [1]=benchmarks [2]=xrlenv_plugins [3]=repo root
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    return _repo_root() / "tmp"


def _default_job_id() -> str:
    return "deepswe-oracle-sweep__" + _dt.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")


def _shard_root(args: argparse.Namespace) -> Path:
    # Hard-reject the retired cache env var/path before reading the root (renamed
    # 2026-07-31: XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, xrlenv_harbor_cache
    # -> xrlenv_benchmark_cache). The old var/path points at stale/absent data, so
    # running against it yields unreliable results — fail loud instead. Lazy import
    # to match the plugin style (plugin -> xrlenv core).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env(args.dest)
    root = args.dest or os.environ.get("XRLENV_BENCHMARK_CACHE")
    if not root:
        raise SystemExit(
            "no cache root — pass --dest or set XRLENV_BENCHMARK_CACHE (the path "
            "build_cache.py wrote to).",
        )
    shard_root = Path(root).expanduser() / SHARD
    if not shard_root.is_dir():
        raise SystemExit(
            f"{shard_root} not found — build the cache first:\n"
            f"  python xrlenv_plugins/benchmarks/deep_swe/build_cache.py "
            f"--stage all --dest {root}",
        )
    return shard_root


def _resolve_tasks(shard_root: Path, tasks_arg: str | None) -> list[str]:
    if tasks_arg is not None:   # None=absent; '' is present-but-empty (audit M5)
        want = [t.strip() for t in tasks_arg.split(",") if t.strip()]
        if not want:
            raise SystemExit(f"--tasks {tasks_arg!r} selected no tasks (audit M5)")
        missing = [t for t in want if not (shard_root / t / "task.toml").is_file()]
        if missing:
            raise SystemExit(f"unknown task(s) under {shard_root}: {missing}")
        return want
    found = sorted(p.parent.name for p in shard_root.glob("*/task.toml"))
    if not found:
        raise SystemExit(f"no tasks with task.toml under {shard_root}")
    return found


def _build_job_config(
    *, task_ids: list[str], shard_root: Path, jobs_dir: Path, job_id: str,
    n_concurrent_trials: int, override_cpus: int | None,
    override_memory_mb: int | None, cpus_multiplier: float,
    memory_multiplier: float, cpu_pinning: bool, timeout_multiplier: float,
    retries: int,
) -> Any:
    from pier.models.job.config import JobConfig, RetryConfig
    from pier.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

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
        # include_exceptions gates retries to infra-transient errors ONLY (see
        # _INFRA_RETRY_EXCEPTIONS) so eval signal is never re-rolled. Short
        # near-constant backoff: the acquire's own queue_timeout provides the
        # "wait for a slot", so we want the trial back in the admission queue fast.
        retry=RetryConfig(
            max_retries=retries,
            include_exceptions=set(_INFRA_RETRY_EXCEPTIONS),
            min_wait_sec=2.0,
            wait_multiplier=1.0,
            max_wait_sec=10.0,
        ),
        environment=EnvironmentConfig(
            # import_path selects our cluster env; type=None so it takes precedence.
            import_path=ENV_IMPORT_PATH,
            type=None,
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            kwargs=env_kwargs,
        ),
        agents=[AgentConfig()],  # default -> OracleAgent (runs solution/solve.sh)
        tasks=[TaskConfig(path=shard_root / tid) for tid in task_ids],
    )


# The verifier grade is binary and lives under the "reward" key of reward.json;
# the other keys (f2p/p2p totals + fractions, partial) are side metrics that may be
# legitimately 0. Key the pass gate on "reward" ONLY (an "all values > 0" gate would
# false-FAIL a passing task whose p2p_total or partial is 0).
_REWARD_KEY = "reward"


def _reward_value(trial_result: Any) -> float | None:
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None:
        return None
    raw = vr.rewards.get(_REWARD_KEY)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


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
    """Pass = no exception AND ``reward.json``'s ``reward`` value > 0."""
    if trial_result.exception_info is not None:
        return False, f"exception: {trial_result.exception_info.exception_type}"
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None or not vr.rewards:
        return False, "no verifier rewards recorded"
    reward = _reward_value(trial_result)
    if reward is None:
        return False, f"no {_REWARD_KEY!r} key in rewards={list(vr.rewards)}"
    if reward > 0:
        return True, None
    return False, f"{_REWARD_KEY}={reward}"


def _side_metrics(trial_result: Any) -> dict[str, Any]:
    """The non-``reward`` keys of reward.json (f2p/p2p buckets, partial), reported
    for context but NOT used to decide pass/fail."""
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None:
        return {}
    return {k: v for k, v in vr.rewards.items() if k != _REWARD_KEY}


def _summarise(trial_results: list[Any], expected: int, jobs_dir: Path) -> int:
    passed = 0
    failed: list[str] = []
    print("\n=== deep-swe oracle sweep ===")
    for tr in sorted(trial_results, key=lambda r: r.task_name):
        ok, reason = _trial_passes(tr)
        reward = _reward_value(tr)
        side = _side_metrics(tr)
        mark = "PASS" if ok else "FAIL"
        detail = "" if ok else f"  ({reason})"
        side_str = f"  {side}" if side else ""
        print(f"  [{mark}] {tr.task_name:44s} reward={reward}{side_str}{detail}")
        if ok:
            passed += 1
        else:
            failed.append(tr.task_name)

    print(f"\n{passed} / {expected} oracle(s) solved.")
    if failed:
        print(
            f"failed: {failed}\n"
            f"Under the oracle a failure is a plumbing/content bug, not a model "
            f"signal — inspect the per-trial output under {jobs_dir}/ "
            f"(agent/, verifier/reward.json, trial logs).",
        )
    return 0 if passed == expected else 1


async def _run(args: argparse.Namespace) -> int:
    from pier.job import Job

    if not os.environ.get("XRLENV_GRPC_HOST"):
        raise SystemExit(
            "XRLENV_GRPC_HOST is not set — this sweep runs on the xrlenv "
            "cluster, so it needs the control-plane address. Export it (plus "
            "XRLENV_CONSUMER_TOKEN if the CP has auth) before running, e.g.:\n"
            "  export XRLENV_GRPC_HOST=<control-plane-host>\n"
            "  export XRLENV_GRPC_PORT=50051\n"
            "  export XRLENV_CONSUMER_TOKEN=<token>",
        )

    shard_root = _shard_root(args)
    task_ids = _resolve_tasks(shard_root, args.tasks)

    jobs_dir = Path(args.jobs_dir).expanduser()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or _default_job_id()

    cp = os.environ.get("XRLENV_GRPC_HOST")
    cp_port = os.environ.get("XRLENV_GRPC_PORT", "50051")
    print(
        f"running {len(task_ids)} oracle(s) from {shard_root} on the xrlenv "
        f"cluster at {cp}:{cp_port} (concurrency={args.max_workers})\n"
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
            shard_root=shard_root,
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
        job = await Job.create(config)
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
    return _summarise([best[t] for t in task_ids if t in best], len(task_ids), jobs_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # audit H9: NO prefix abbreviation (--des / --de / --d) may resolve to --dest —
        # the cache root must come only from XRLENV_BENCHMARK_CACHE, never a CLI override
        # smuggled past the wrapper's exact-form reject.
        allow_abbrev=False,
        prog="run_oracle_sweep",
        description=(
            "Run the deep-swe oracle against the shared cache and report which "
            "tasks the oracle solves (reward.json reward > 0). Runs ALL tasks by "
            "default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help=f"Cache ROOT (the shard lives at <dest>/{SHARD}/). Defaults to "
        "$XRLENV_BENCHMARK_CACHE.",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids to run. Default: every task in the shard.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Trial-level concurrency (pier n_concurrent_trials). Default 4.",
    )
    p.add_argument(
        "--jobs-dir",
        default=str(_default_jobs_dir()),
        help="pier jobs_dir; each run lands in a <jobs-dir>/<job-id>/ subdir. "
        "Defaults to the repo's gitignored tmp/.",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Run label under <jobs-dir>/. Defaults to a timestamped "
        "deepswe-oracle-sweep__YYYY-MM-DD__HH-MM-SS.",
    )
    p.add_argument(
        "--override-cpus",
        type=int,
        default=None,
        help="Force every task to this many cpus instead of its task.toml value.",
    )
    p.add_argument(
        "--override-memory-mb",
        type=int,
        default=None,
        help="Force every task to this much memory (MiB) instead of task.toml.",
    )
    p.add_argument(
        "--cpus-multiplier",
        type=float,
        default=1.0,
        help="Scale every task's declared cpus by this factor (headroom "
        "ablation). Composes with --override-cpus. Default 1.0.",
    )
    p.add_argument(
        "--memory-multiplier",
        type=float,
        default=1.0,
        help="Scale every task's declared memory by this factor. Default 1.0.",
    )
    p.add_argument(
        "--cpu-pinning",
        action="store_true",
        help="Opt every task in the job into cpuset pinning (nproc == declared "
        "cpus). Ablation knob for build-heavy SWE tasks that scale to nproc.",
    )
    p.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Scale pier's agent/verifier timeouts (e.g. 2.0 for slow nodes). "
        "Default 1.0.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Max trial retries for INFRA-transient errors only (capacity / "
        "control-plane / node blips — see _INFRA_RETRY_EXCEPTIONS). Task-content "
        "failures are NEVER retried. Default 6.",
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
