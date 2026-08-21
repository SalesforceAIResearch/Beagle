"""terminal-bench-2 onboarding smoke — drives the **upstream**
``harbor`` runner against either the local Docker daemon or an
xrlenv cluster.

The audience's contract
=======================

Both modes drive ``harbor.Job.run()`` directly. The only difference
between modes is which ``import_path`` the harbor ``EnvironmentConfig``
points at:

- ``--local`` →
  ``xrlenv_plugins.harbor:XrlenvHarborEnvironment``
  (harbor's stock LocalDocker behaviour, with xrlenv kwargs recorded
  on the instance for observability).
- **default (cluster mode)** →
  ``xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster``
  (cluster-routed: ``acquire_container`` / ``session.exec`` /
  ``put_archive`` / ``get_archive`` / ``destroy`` instead of local
  ``docker compose`` + ``docker cp``).

This is exactly the UX harbor users already know from picking ``e2b``,
``modal``, or ``daytona`` — the integration is one ``import_path``
swap in ``job.yaml``.

Operator setup (cluster mode)
=============================

The audience doesn't run xrlenv-specific commands inside the smoke.
The OPERATOR runs them once at cluster bring-up:

1. ``xrlenv up`` — boot the control plane (idempotent).
2. **Pre-build per-task images on each node** via
   ``examples/benchmarks-onboarding/terminal-bench-2/scripts/
   build-task-images.sh``. terminal-bench-2 task images are not on
   a public registry, so the cluster looks up
   ``hb__<task_id>`` locally on the chosen node; missing images
   fail fast with a clear ``ImageNotFound``. Real build-on-acquire
   (``HarborImageBuilder`` + acquire→build→re-acquire fallback) is
   **P1.7.C.2** — out of scope for this slice.
3. Set ``XRLENV_GRPC_HOST`` / ``XRLENV_GRPC_PORT`` /
   ``XRLENV_CONSUMER_TOKEN`` in the consumer's shell.

The audience then runs this smoke unchanged.

Default task set
================

8 phase-0 acceptance tasks (see :data:`SMOKE_TASKS`). ``--all``
runs every task in the harbor cache; ``--tasks task1,task2`` runs
an explicit subset.

Concurrency is the operator's choice
====================================

``--max-workers`` defaults to **1 (serial)**. The smoke threads
the value into harbor's native ``JobConfig.n_concurrent_trials``
which spawns N concurrent trials inside one harbor process.
harbor itself uses asyncio internally per-trial, so this is
already an event-loop-scoped concurrency model — no need for an
external ``ThreadPoolExecutor`` wrapper.

For tb2 tasks the per-task wall-clock is dominated by
``solve.sh`` execution inside the container; concurrency >1 wins
when N nodes are available to fan out across.

Artifact archiving
==================

Always archives — harbor itself writes per-trial outputs
(``trial.log``, ``result.json``, ``verifier/``, ``agent/``,
``artifacts/``) under ``<jobs_dir>/<job_name>/<trial_name>/``.
This smoke sets ``jobs_dir = <repo>/tmp/`` (gitignored) and
``job_name = <job-id>``. ``--save-artifacts <PATH>`` overrides
``jobs_dir``; ``--job-id <NAME>`` overrides the timestamped
default.

Oracle policy
=============

Default agent is harbor's own :class:`harbor.agents.oracle.OracleAgent`
which copies the task's ``solution/solve.sh`` into the container and
runs it. Pass criterion: every trial has ``verifier_result.rewards``
fully populated with positive values. A failing trial under the
oracle policy is a plumbing bug, not a model-eval signal.

Usage
=====

::

    # Local 8-task smoke (baseline against local Docker daemon):
    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \\
        --local

    # Cluster 8-task smoke, env-var-driven, serial:
    export XRLENV_GRPC_HOST=127.0.0.1
    export XRLENV_GRPC_PORT=50051
    export XRLENV_CONSUMER_TOKEN=$(cat ~/.xrlenv/secrets/consumer.token)
    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py

    # Cluster 8-task smoke, 4-way concurrent:
    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \\
        --max-workers 4

    # Custom task list:
    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \\
        --tasks fix-git,dna-insert

    # Sweep across every cached task:
    .venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \\
        --all
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import logging
import os
import pprint
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("xrlenv.smoke.terminal-bench-2-onboarding")

# 8 phase-0 acceptance tasks. Each has a prebuilt
# ``alexgshaw/<task>:20251031`` image on Docker Hub; the cluster's
# ``ensure_image_present`` pulls on first acquire. A task added to
# this list whose ``task.toml`` doesn't ship a ``docker_image``
# field needs either an operator-built+pushed image or P1.7.C.2's
# build-on-acquire (currently deferred); otherwise acquire fails
# fast with ``ImageNotFound``.
SMOKE_TASKS: tuple[str, ...] = (
    "fix-git",
    "build-pov-ray",
    "overfull-hbox",
    "cobol-modernization",
    "prove-plus-comm",
    "constraints-scheduling",
    "nginx-request-logging",
    "dna-insert",
)

LOCAL_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironment"
CLUSTER_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"


# ──────────────────────────────────────────────────────────────────────────────
# Path + cache discovery (mirrors xrlenv_plugins/benchmarks/terminal_bench_2/
# examples/tb2_acceptance_smoke.py so the operator's harbor cache layout
# works for both onboarding paths.)
# ──────────────────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    """``<repo>/tmp/`` — gitignored; harbor's ``jobs_dir`` default."""
    return _repo_root() / "tmp"


def _default_job_id() -> str:
    """``smoke-terminal-bench-2-YYYYMMDD-HHMMSS`` UTC default."""
    return _dt.datetime.utcnow().strftime(
        "smoke-terminal-bench-2-%Y%m%d-%H%M%S",
    )


def _harbor_cache_root() -> Path:
    # Fail loud on the RETIRED XRLENV_HARBOR_CACHE var / .../xrlenv_harbor_cache path (audit
    # M17) rather than silently ignoring a stale legacy env and falling through to the default.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env()
    explicit = os.environ.get("XRLENV_BENCHMARK_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    # Phase-0 example default (populate-harbor-cache.sh clones the tb2 catalog here).
    return Path("~/.cache/harbor/tasks").expanduser()


def _locate_task_dir(task_id: str) -> Path:
    """Find ``<cache>/<task_id>/`` (flat layout) or
    ``<cache>/<hash>/<task_id>/`` (sharded layout). Mirrors the
    in-tree ``_locate_solution_dir`` shape so both onboarding paths
    consume the same harbor download flow."""
    cache_root = _harbor_cache_root()
    flat = cache_root / task_id
    if (flat / "solution").is_dir():
        return flat
    if cache_root.is_dir():
        for hash_dir in cache_root.iterdir():
            if not hash_dir.is_dir():
                continue
            candidate = hash_dir / task_id
            if (candidate / "solution").is_dir():
                return candidate
    raise SystemExit(
        f"terminal-bench-2 task {task_id!r}: directory with solution/ "
        f"not found under {cache_root}. Populate the harbor cache (see "
        f"xrlenv_plugins/benchmarks/terminal_bench_2/README.md), or "
        f"override the cache root via ``XRLENV_BENCHMARK_CACHE``.",
    )


def _discover_all_tasks() -> list[str]:
    """Walk the harbor cache and return every task that has a
    ``solution/solve.sh`` — the universe ``--all`` runs over."""
    cache_root = _harbor_cache_root()
    if not cache_root.is_dir():
        raise SystemExit(
            f"--all requires a populated harbor cache at {cache_root}. "
            f"See xrlenv_plugins/benchmarks/terminal_bench_2/README.md.",
        )
    seen: set[str] = set()
    for top in sorted(cache_root.iterdir()):
        if not top.is_dir():
            continue
        # Flat layout: <cache>/<task_id>/solution/solve.sh
        if (top / "solution" / "solve.sh").is_file():
            seen.add(top.name)
            continue
        # Sharded layout: <cache>/<hash>/<task_id>/solution/solve.sh
        for inner in sorted(top.iterdir()):
            if inner.is_dir() and (inner / "solution" / "solve.sh").is_file():
                seen.add(inner.name)
    if not seen:
        raise SystemExit(
            f"--all walked {cache_root} but found no task directories "
            f"with a solution/solve.sh — is the cache populated?",
        )
    return sorted(seen)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="terminal-bench-2-onboarding-smoke",
        description=(
            "Run the upstream harbor runner against either local "
            "Docker (--local) or an xrlenv cluster (default). The "
            "audience picks the right ``import_path`` in their "
            "harbor job.yaml — this smoke does the same swap "
            "programmatically. No xrlenv-specific CLI knobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--local", action="store_true",
        help="Use the local-mode harbor Environment "
             "(XrlenvHarborEnvironment). Baseline mode for verifying "
             "the smoke works at all before pointing it at the cluster.",
    )
    p.add_argument(
        "--tasks", default=None,
        help="Comma-separated task ids. Overrides the default "
             "8-task smoke set. Mutually exclusive with --all.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Run every task in the harbor cache (whatever has a "
             "solution/solve.sh). Mutually exclusive with --tasks. "
             "Pre-build all images first via "
             "scripts/build-task-images.sh on each node.",
    )
    p.add_argument(
        "--max-workers", type=int, default=1,
        help="Trial-level concurrency (default 1 = serial). Threads "
             "into harbor's JobConfig.n_concurrent_trials. harbor "
             "uses asyncio internally per-trial, so this is already "
             "an event-loop-scoped concurrency model — no external "
             "ThreadPoolExecutor wrapper required.",
    )
    p.add_argument(
        "--save-artifacts", default=None, metavar="PATH",
        help="Override harbor's ``jobs_dir``. Default is "
             f"``{_default_jobs_dir()}/`` (gitignored).",
    )
    p.add_argument(
        "--job-id", default=None,
        help="Label under <jobs_dir>/. Default is "
             "``smoke-terminal-bench-2-YYYYMMDD-HHMMSS`` UTC.",
    )
    return p


def _resolve_task_list(args: argparse.Namespace) -> list[str]:
    if args.all and args.tasks:
        raise SystemExit("--all and --tasks are mutually exclusive.")
    if args.all:
        return _discover_all_tasks()
    if args.tasks:
        return [s.strip() for s in args.tasks.split(",") if s.strip()]
    return list(SMOKE_TASKS)


def _build_job_config(
    *,
    task_ids: list[str],
    local: bool,
    jobs_dir: Path,
    job_id: str,
    n_concurrent_trials: int,
) -> Any:
    """Compose harbor's ``JobConfig``. Mirrors the equivalent
    ``job.yaml`` an end-user would write. Picks the right
    ``import_path`` for the chosen mode; agents defaults to harbor's
    OracleAgent (``AgentConfig()`` with no explicit name)."""
    from harbor.models.job.config import JobConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
    )

    import_path = LOCAL_IMPORT_PATH if local else CLUSTER_IMPORT_PATH
    return JobConfig(
        job_name=job_id,
        jobs_dir=jobs_dir,
        n_concurrent_trials=n_concurrent_trials,
        environment=EnvironmentConfig(import_path=import_path),
        agents=[AgentConfig()],
        tasks=[
            TaskConfig(path=_locate_task_dir(task_id))
            for task_id in task_ids
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pass / fail aggregation.
# ──────────────────────────────────────────────────────────────────────────────


def _trial_passes(trial_result: Any) -> tuple[bool, str | None]:
    """Pass = trial completed with verifier_result.rewards fully
    populated and every value > 0. Returns ``(passed, reason)``;
    reason is None on success."""
    if trial_result.exception_info is not None:
        return False, (
            f"exception: "
            f"{trial_result.exception_info.exception_type}"
        )
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None or not vr.rewards:
        return False, "no verifier rewards recorded"
    failures = [k for k, v in vr.rewards.items() if not (v > 0)]
    if failures:
        return False, f"non-positive reward(s): {failures}"
    return True, None


def _summarise(
    trial_results: list[Any], expected: int,
) -> tuple[int, dict[str, Any]]:
    print("\n=== terminal-bench-2 onboarding smoke ===")
    passed = 0
    failed: list[str] = []
    per_trial: list[dict[str, Any]] = []
    for tr in sorted(trial_results, key=lambda r: r.task_name):
        ok, reason = _trial_passes(tr)
        if ok:
            passed += 1
        else:
            failed.append(tr.task_name)
        rewards = (
            tr.verifier_result.rewards
            if tr.verifier_result is not None else None
        )
        row = {
            "task": tr.task_name,
            "trial": tr.trial_name,
            "passed": ok,
            "rewards": rewards,
            "reason": reason,
        }
        per_trial.append(row)
        pprint.pp(row)
    print(
        f"\n{passed} / {expected} trial(s) passed under the oracle "
        f"policy.",
    )
    if failed:
        print(
            f"failed: {failed}\n"
            f"Under the oracle policy, a non-passing trial is a "
            f"plumbing bug, not a model-eval signal — check the "
            f"per-trial directory under <jobs_dir>/<job_name>/ for "
            f"trial.log + verifier outputs.",
        )
    summary = {
        "expected": expected,
        "passed": passed,
        "failed": failed,
        "trials": per_trial,
    }
    return (0 if passed == expected else 1), summary


# ──────────────────────────────────────────────────────────────────────────────
# Entry point.
# ──────────────────────────────────────────────────────────────────────────────


async def _run(args: argparse.Namespace) -> int:
    import harbor

    task_ids = _resolve_task_list(args)
    LOGGER.info(
        "running %d task(s) in %s mode at concurrency=%d: %s",
        len(task_ids),
        "local" if args.local else "cluster",
        args.max_workers,
        ", ".join(task_ids),
    )

    if not args.local:
        if not os.environ.get("XRLENV_GRPC_HOST"):
            raise SystemExit(
                "cluster mode: XRLENV_GRPC_HOST not set. Either pass "
                "--local for the baseline, or export "
                "XRLENV_GRPC_HOST / XRLENV_GRPC_PORT / "
                "XRLENV_CONSUMER_TOKEN before running this smoke "
                "(typically done by the operator's deploy script "
                "alongside ``xrlenv up``). The cluster Environment "
                "lazy-constructs an xrlenv.Client from these env vars.",
            )
        LOGGER.info(
            "cluster mode: control plane at %s:%s",
            os.environ.get("XRLENV_GRPC_HOST"),
            os.environ.get("XRLENV_GRPC_PORT", "50051"),
        )
        LOGGER.info(
            "image distribution: the cluster pulls each task's "
            "``docker_image`` (typically ``alexgshaw/<task>:<rev>`` "
            "on Docker Hub for the phase-0 set) via "
            "``ensure_image_present`` on first acquire. Tasks "
            "without a ``docker_image`` field need either an "
            "operator-built+pushed image or P1.7.C.2's "
            "build-on-acquire (deferred).",
        )

    jobs_dir = (
        Path(args.save_artifacts).expanduser()
        if args.save_artifacts else _default_jobs_dir()
    )
    job_id = args.job_id or _default_job_id()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    config = _build_job_config(
        task_ids=task_ids,
        local=args.local,
        jobs_dir=jobs_dir,
        job_id=job_id,
        n_concurrent_trials=args.max_workers,
    )

    LOGGER.info("artifacts: %s", jobs_dir / job_id)

    job = await harbor.Job.create(config)
    job_result = await job.run()

    exit_code, _summary = _summarise(
        list(job_result.trial_results), len(task_ids),
    )
    print(f"artifacts at {jobs_dir / job_id}")
    return exit_code


def main() -> int:
    args = _build_parser().parse_args()
    from xrlenv.observability.logging import configure_logging
    configure_logging()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
