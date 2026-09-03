#!/usr/bin/env python3
"""Run the SWE-rebench oracle sweep against the shared cache, on the xrlenv cluster.

Runs harbor's default ``OracleAgent`` (applies each task's shipped
``solution/solve.sh`` reference) for every task in the ``swe-rebench`` shard
(built by ``build_cache.py``) and reports which the oracle solves (verifier
``reward > 0``). This is the corpus-quality gate: a task the *oracle* can't
solve is poison for RL — its reward ceiling is 0 — so under the oracle a
non-passing task is a plumbing/content bug, not a model signal.

Each task runs **on the xrlenv cluster** via the harbor cluster environment
(``xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster``), which routes every
container acquire/exec through the control plane to a scheduler-chosen node. So it
needs a running control plane, addressed via env:

    export XRLENV_GRPC_HOST=<control-plane-host>   # required
    export XRLENV_GRPC_PORT=50051                  # optional (default 50051)
    export XRLENV_CONSUMER_TOKEN=<token>           # required if CP has auth
    export XRLENV_GRPC_SECURE=true                 # optional (TLS)

**Images.** SWE-rebench tasks get a prebuilt Docker Hub ``[environment]
docker_image`` written by ``build_cache.py --stage repin``, so — like
terminal-bench-2-1 and frontier-swe — the cluster resolves each task's image
directly from its ``docker_image``; there is **no** ``xrlenv_image_template`` to
inject and nothing to build. Warm them ahead of time with ``build_plan_gen.py`` +
``xrlenv build apply``, or rely on lazy first-acquire pull.

**Grading is stock harbor — no benchmark-specific seam.** Every task's
``tests/test.sh`` runs the repo's test command, hands the log to the corpus-wide
``tests/parser.py`` (byte-identical across all 860), and writes a flat
``/logs/verifier/reward.txt`` of ``0`` or ``1`` — resolved iff every
``FAIL_TO_PASS`` and ``PASS_TO_PASS`` test passes. harbor 0.20's ``Verifier``
parses that into ``VerifierResult.rewards = {"reward": <float>}``, which
:func:`_trial_passes` reads directly. Nothing is re-implemented here (the
``parser.py`` resolution rule stays upstream's).
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
SHARD = "swe-rebench"

ENV_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"

# Infra-transient errors the trial queue may retry — and ONLY these. They match on
# ``type(e).__name__`` (harbor's RetryConfig.include_exceptions). This lets a consumer
# request any concurrency while xrlenv transparently paces a capacity-CAPPED runtime:
# an at-cap acquire fails fast with ``CapacityExhausted`` and the trial queue retries.
# Acquire is harbor's first (cheapest) setup step, so the DOMINANT retry is a fail-fast
# acquire — before solve.sh runs — and a task usually runs once. A post-acquire infra
# error re-runs the WHOLE trial in a FRESH container, so only EXTERNAL side effects
# carry, never the recorded result. Task-content failures are NEVER re-rolled here
# (that is --content-retries' job, and it is visible in the artifacts).
_INFRA_RETRY_EXCEPTIONS = frozenset({
    "CapacityExhausted",   # admission queue timed out waiting for a runtime slot
    "ControlPlaneLost",    # CP restarted under the run
    "NodeLost",            # node dropped its stream mid-acquire
    "NodeCommandTimeout",  # a node RPC deadline (teardown / exec) tripped
    # The control plane destroyed the session out from under us — a stalled
    # consumer past the quarantine horizon, a lost node, a deadline. The
    # rollout's work never failed; the platform reclaimed it, so a fresh
    # acquire is the correct response, not a content result.
    "SessionReaped",
})


def _repo_root() -> Path:
    # xrlenv_plugins/benchmarks/swe_rebench/run_oracle_sweep.py
    #   parents: [0]=swe_rebench [1]=benchmarks [2]=xrlenv_plugins [3]=repo root
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    return _repo_root() / "tmp"


def _default_job_id() -> str:
    return "swe-rebench-oracle-sweep__" + _dt.datetime.now().strftime(
        "%Y-%m-%d__%H-%M-%S",
    )


def _shard_root(args: argparse.Namespace) -> Path:
    # Hard-reject the retired cache env var/path before reading the root (renamed
    # 2026-07-31: XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, xrlenv_harbor_cache
    # -> xrlenv_benchmark_cache). Lazy import to match the plugin style.
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
            f"  python xrlenv_plugins/benchmarks/swe_rebench/build_cache.py "
            f"--stage all --dest {root}",
        )
    return shard_root


def _resolve_tasks(shard_root: Path, tasks_arg: str | None) -> list[str]:
    """Oracle-gateable tasks = those that ship ``solution/solve.sh``. All 860
    SWE-rebench tasks do, but the discovery keys on the solve.sh anchor anyway so a
    partially-populated shard can't silently enumerate an un-gateable task."""
    if tasks_arg is not None:   # None=absent; '' is present-but-empty
        want = [t.strip() for t in tasks_arg.split(",") if t.strip()]
        if not want:
            raise SystemExit(f"--tasks {tasks_arg!r} selected no tasks")
        missing = [
            t for t in want
            if not (shard_root / t / "solution" / "solve.sh").is_file()
        ]
        if missing:
            raise SystemExit(
                f"unknown / non-gateable task(s) under {shard_root} "
                f"(no solution/solve.sh): {missing[:10]}"
                + (f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""),
            )
        return want
    found = sorted(
        p.parent.parent.name
        for p in shard_root.glob("*/solution/solve.sh")
    )
    if not found:
        raise SystemExit(
            f"no oracle-gateable tasks (solution/solve.sh) under {shard_root}",
        )
    return found


def _build_job_config(
    *, task_ids: list[str], shard_root: Path, jobs_dir: Path, job_id: str,
    n_concurrent_trials: int, override_cpus: int | None,
    override_memory_mb: int | None, cpus_multiplier: float,
    memory_multiplier: float, cpu_pinning: bool, timeout_multiplier: float,
    retries: int,
) -> Any:
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
        agents=[AgentConfig()],  # default -> OracleAgent (runs solution/solve.sh)
        tasks=[TaskConfig(path=shard_root / tid) for tid in task_ids],
    )


# ── Pass gate ─────────────────────────────────────────────────────────────────
# SWE-rebench writes a flat /logs/verifier/reward.txt of 0 or 1 (resolved iff every
# FAIL_TO_PASS + PASS_TO_PASS test passed, per the corpus-wide tests/parser.py), and
# harbor parses it into rewards={"reward": <float>}. So the gate is the tb2.1/seta
# shape: EVERY reward key must be > 0 (single-key in practice).


def _rewards(trial_result: Any) -> dict[str, float] | None:
    vr = getattr(trial_result, "verifier_result", None)
    rewards = getattr(vr, "rewards", None) if vr is not None else None
    if not rewards:
        return None
    try:
        return {k: float(v) for k, v in rewards.items()}
    except (TypeError, ValueError):
        return None


def _trial_passes(trial_result: Any) -> tuple[bool, str | None]:
    """Pass = the verifier reported EVERY reward > 0 (i.e. reward.txt == 1, the
    instance resolved). A harbor ``exception_info`` — the trial never produced a
    verifier result at all — is a failure with the exception named."""
    rewards = _rewards(trial_result)
    if rewards is None:
        exc = getattr(trial_result, "exception_info", None)
        if exc is not None:
            return False, f"exception: {exc.exception_type}"
        return False, "no verifier result (no reward file written)"
    if all(v > 0 for v in rewards.values()):
        return True, None
    return False, f"rewards={rewards}"


def _task_key(trial_result: Any) -> str:
    """Requested task id = the shard dir name (basename of the task path) — the
    same identifier the caller passed in ``--tasks``, so the content-retry loop
    matches results back correctly."""
    return Path(trial_result.config.task.path).name


def _summarise(trial_results: list[Any], expected: int, jobs_dir: Path) -> int:
    passed = 0
    failed: list[str] = []
    print("\n=== swe-rebench oracle sweep ===")
    for tr in sorted(trial_results, key=lambda r: _task_key(r)):
        ok, reason = _trial_passes(tr)
        rewards = _rewards(tr)
        mark = "PASS" if ok else "FAIL"
        detail = "" if ok else f"  ({reason})"
        print(f"  [{mark}] {_task_key(tr):55s} rewards={rewards}{detail}")
        if ok:
            passed += 1
        else:
            failed.append(_task_key(tr))

    print(f"\n{passed} / {expected} oracle(s) solved.")
    if failed:
        print(
            f"failed ({len(failed)}): {failed}\n"
            f"Under the oracle a failure is a plumbing/content bug, not a model "
            f"signal — inspect the per-trial output under {jobs_dir}/ "
            f"(agent/, verifier/reward.txt, verifier/report.json, trial logs). "
            f"report.json names which FAIL_TO_PASS / PASS_TO_PASS tests did not pass.",
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
    # _trial_passes) up to --content-retries times. A task counts as solved if ANY
    # attempt passes; a reward=0 flake gets a fresh trial while a genuine failure
    # persists. Distinct from --retries (infra-transient-only, never re-rolls a
    # reward outcome).
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
        job = await harbor.Job.create(config)
        job_result = await job.run()
        for tr in job_result.trial_results:
            key = _task_key(tr)
            if key not in best or (
                not _trial_passes(best[key])[0] and _trial_passes(tr)[0]
            ):
                best[key] = tr
        remaining = [t for t in remaining
                     if t not in best or not _trial_passes(best[t])[0]]
        if not remaining:
            break
        if attempt < cr:
            shown = ", ".join(remaining[:20])
            more = f" ... (+{len(remaining) - 20} more)" if len(remaining) > 20 else ""
            print(f"content-retry {attempt + 1}/{cr}: re-running {len(remaining)} "
                  f"non-passing task(s): {shown}{more}", file=sys.stderr)
    # Fold the content-retry rounds' sibling -retryN dirs back into the attempt-0
    # dir so the operator sees ONE result set per task.
    from xrlenv_plugins.benchmarks._sweep_retry import consolidate_retry_dirs
    consolidate_retry_dirs(jobs_dir, job_id, cr)
    return _summarise(
        [best[t] for t in task_ids if t in best], len(task_ids), jobs_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # NO prefix abbreviation may resolve to --dest — the cache root must come
        # only from XRLENV_BENCHMARK_CACHE, never a CLI override smuggled past the
        # wrapper's exact-form reject.
        allow_abbrev=False,
        prog="run_oracle_sweep",
        description=(
            "Run the swe-rebench oracle against the shared cache and report which "
            "tasks the oracle solves (verifier reward > 0). Runs ALL oracle-gateable "
            "tasks by default."
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
        help="Comma-separated task ids to run. Default: every oracle-gateable task "
        "(ships solution/solve.sh) in the shard.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Trial-level concurrency (harbor n_concurrent_trials). Default 8. "
        "Callers may request ANY value — xrlenv paces capacity via fail-fast "
        "acquire + the infra-only --retries; never lower it to make a red run green.",
    )
    p.add_argument(
        "--jobs-dir",
        default=str(_default_jobs_dir()),
        help="harbor jobs_dir; each run lands in a <jobs-dir>/<job-id>/ subdir. "
        "Defaults to the repo's gitignored tmp/.",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Run label under <jobs-dir>/. Defaults to a timestamped "
        "swe-rebench-oracle-sweep__YYYY-MM-DD__HH-MM-SS.",
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
        "cpus). Ablation knob for build-heavy tasks that scale to nproc.",
    )
    p.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Scale harbor's agent/verifier timeouts (e.g. 2.0 for slow nodes). "
        "Default 1.0 — the gate runs at native budget.",
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
        "solved if ANY attempt passes. Catches nondeterministic reward=0 flakes that "
        "--retries (infra-only) deliberately never re-rolls. Default 0 "
        "(run_full_sweep.sh passes 2).",
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
