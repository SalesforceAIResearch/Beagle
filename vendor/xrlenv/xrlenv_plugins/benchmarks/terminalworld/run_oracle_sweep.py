#!/usr/bin/env python3
"""Run the TerminalWorld ``verified`` oracle sweep against the shared cache.

Runs harbor's OracleAgent for every task in the ``terminalworld-verified`` shard
(built by ``build_cache.py``) and reports which the oracle actually solves
(verifier reward > 0). This is the corpus-quality gate: a task the *oracle* can't
solve is poison for RL — its reward ceiling is 0 — so under the oracle a
non-passing task is a plumbing/content bug, not a model signal.

Each task's oracle runs **on the xrlenv cluster** via the cluster harbor
environment (``xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster``), which
routes every container acquire/exec through the control plane to a
scheduler-chosen node. So it needs a running control plane, addressed via env:

    export XRLENV_GRPC_HOST=<control-plane-host>   # required
    export XRLENV_GRPC_PORT=50051                  # optional (default 50051)
    export XRLENV_CONSUMER_TOKEN=<token>           # required if CP has auth
    export XRLENV_GRPC_SECURE=true                 # optional (TLS)

**Image resolution.** Unlike terminal-bench-2-1 (whose tasks ship a prebuilt
``docker_image``), TerminalWorld tasks ship only a ``Dockerfile``; the sibling
``xrlenv_plugins/images_build/`` scripts build+push each to
``<registry>/terminalworld-verified/<id>:main``. This sweep constructs the
template ``<registry>/terminalworld-verified/{task_id}:main`` from ``--registry``
or ``$XRLENV_PRIVATE_REGISTRY_HOST(:_PORT)`` and passes it to the cluster plugin
as the ``xrlenv_image_template`` per-run kwarg via ``EnvironmentConfig``.

**Sysbox tasks.** A task marked for the sysbox pool (``build_cache.py --stage
sysbox``, which writes ``[environment.env] XRLENV_CONTAINER_RUNTIME`` into its
task.toml) is routed to a sysbox-runc node by the cluster plugin — no sweep flag
needed. Pass ``--tasks tw_245733`` to run just the decisive probe.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Any

# Dataset shard subdir name — must match build_cache.py's SHARD and the image
# namespace the images_build scripts push under.
SHARD = "terminalworld-verified"

ENV_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"

# Infra-transient errors the trial queue may retry — and ONLY these. They match on
# ``type(e).__name__`` (harbor's RetryConfig). The point is to let a consumer
# request any concurrency while xrlenv transparently paces a capacity-CAPPED
# runtime: when the per-node sysbox cap holds an acquire, the harbor adapter fails
# it fast with ``CapacityExhausted`` (see environment._ACQUIRE_QUEUE_TIMEOUT_S) and
# the trial queue retries. Acquire is harbor's FIRST (cheapest) setup step, so the
# DOMINANT retry is a fail-fast acquire — before ``solve.sh`` runs — and the task
# usually runs once. But a retry re-runs the WHOLE trial in a FRESH container, so a
# post-acquire infra error (e.g. ``NodeCommandTimeout`` on an exec) can re-execute
# ``solve.sh`` — a new container each time, so only EXTERNAL side effects carry, never
# the recorded result. The job records ONE trial result per task; a waited-then-passed
# task counts once (not double-counted), with the retry counter the only trace.
# Task-content failures (AgentTimeoutError, verifier errors) are deliberately OUT of
# this set — harbor's default ``exclude_exceptions`` also lists them — so a
# genuinely-failed task is
# NEVER re-rolled into a fluke pass. Keep this set to capacity / control-plane /
# node blips only.
_INFRA_RETRY_EXCEPTIONS = frozenset({
    "CapacityExhausted",   # admission queue timed out waiting for a runtime slot
    "ControlPlaneLost",    # CP restarted under the run
    "NodeLost",            # node dropped its stream mid-acquire
    "NodeCommandTimeout",  # a node RPC deadline (teardown / exec) tripped
})


def _repo_root() -> Path:
    # xrlenv_plugins/benchmarks/terminalworld/run_oracle_sweep.py
    #   parents: [0]=terminalworld [1]=benchmarks [2]=xrlenv_plugins [3]=repo root
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    """``<repo>/tmp`` — the harbor ``jobs_dir`` (a per-run ``<job-id>/`` subdir
    is created under it). The repo's ``tmp/`` is gitignored."""
    return _repo_root() / "tmp"


def _default_job_id() -> str:
    return "tw-oracle-sweep__" + _dt.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")


def _shard_root(args: argparse.Namespace) -> Path:
    # Reject the retired cache var/path before reading it — a sweep against the
    # wrong cache would roll stale/absent tasks into an unreliable result set.
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
            f"  python xrlenv_plugins/benchmarks/terminalworld/build_cache.py "
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


def _resolve_registry(explicit: str | None) -> str | None:
    """``host:port`` for the private registry the TW images were pushed to.
    ``--registry`` wins, else ``$XRLENV_PRIVATE_REGISTRY_HOST(:_PORT)``. Returns
    None if unresolved (the sweep will omit the xrlenv_image_template kwarg,
    which requires the tasks to carry a docker_image in their task.toml)."""
    if explicit:
        return explicit
    host = os.environ.get("XRLENV_PRIVATE_REGISTRY_HOST")
    if not host:
        return None
    port = os.environ.get("XRLENV_PRIVATE_REGISTRY_PORT", "5011")
    return f"{host}:{port}"


def _ensure_image_template(registry: str | None) -> str:
    """Compose the image template so the cluster plugin resolves each TW task to
    ``<registry>/terminalworld-verified/{task_id}:main``, built from ``registry``.
    The sweep hands it to the adapter per-run via the ``xrlenv_image_template``
    kwarg (a scoped handoff — NOT a process-global env var)."""
    if not registry:
        raise SystemExit(
            "cannot resolve TW images — set --registry <host:port> (or "
            "$XRLENV_PRIVATE_REGISTRY_HOST[:_PORT]). TerminalWorld tasks ship a "
            "Dockerfile (no prebuilt docker_image), so the sweep needs the "
            "private-registry namespace the images_build scripts pushed to.",
        )
    return f"{registry}/{SHARD}/{{task_id}}:main"


def _build_job_config(
    *, task_ids: list[str], shard_root: Path, jobs_dir: Path, job_id: str,
    n_concurrent_trials: int, override_cpus: int | None,
    override_memory_mb: int | None, cpus_multiplier: float,
    memory_multiplier: float, cpu_pinning: bool, timeout_multiplier: float,
    retries: int, image_template: str | None = None,
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
    if image_template:
        env_kwargs["xrlenv_image_template"] = image_template
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
        # retries to infra-transient errors ONLY (see _INFRA_RETRY_EXCEPTIONS) so
        # eval signal is never re-rolled. Short near-constant backoff: the acquire's
        # own queue_timeout already provides the "wait for a slot", so we want the
        # trial back in the admission queue quickly — a long backoff is dead time
        # where it isn't competing for a freed slot.
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
        tasks=[TaskConfig(path=shard_root / tid) for tid in task_ids],
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
    print("\n=== terminalworld-verified oracle sweep ===")
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
            f"Under the oracle a failure is a plumbing/content bug, not a model "
            f"signal — inspect the per-trial output under {jobs_dir}/ "
            f"(agent/oracle.txt, verifier/reward.txt, trial.log).",
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
    template = _ensure_image_template(_resolve_registry(args.registry))

    jobs_dir = Path(args.jobs_dir).expanduser()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or _default_job_id()

    cp = os.environ.get("XRLENV_GRPC_HOST")
    cp_port = os.environ.get("XRLENV_GRPC_PORT", "50051")
    print(
        f"running {len(task_ids)} oracle(s) from {shard_root} on the xrlenv "
        f"cluster at {cp}:{cp_port} (concurrency={args.max_workers})\n"
        f"images: {template}\n"
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
            image_template=template,
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
    return _summarise([best[t] for t in task_ids if t in best], len(task_ids), jobs_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # audit H9: NO prefix abbreviation (--des / --de / --d) may resolve to --dest —
        # the cache root must come only from XRLENV_BENCHMARK_CACHE, never a CLI override
        # smuggled past the wrapper's exact-form reject.
        allow_abbrev=False,
        prog="run_oracle_sweep",
        description=(
            "Run the terminalworld-verified oracle against the shared cache and "
            "report which tasks the oracle solves (reward > 0). Runs ALL tasks "
            "by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help=f"Harbor cache ROOT (the shard lives at <dest>/{SHARD}/). Defaults "
        "to $XRLENV_BENCHMARK_CACHE.",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids to run. Default: every task in the shard.",
    )
    p.add_argument(
        "--registry",
        default=None,
        help="Private registry host:port the TW images were pushed to (used to "
        "build the xrlenv_image_template kwarg). Defaults to "
        "$XRLENV_PRIVATE_REGISTRY_HOST[:_PORT].",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Trial-level concurrency (harbor n_concurrent_trials). Default 4.",
    )
    p.add_argument(
        "--jobs-dir",
        default=str(_default_jobs_dir()),
        help="Harbor jobs_dir; each run lands in a timestamped "
        "<jobs-dir>/<job-id>/ subdir. Defaults to the repo's gitignored tmp/.",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Run label under <jobs-dir>/. Defaults to a timestamped "
        "tw-oracle-sweep__YYYY-MM-DD__HH-MM-SS.",
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
        "cpus). Ablation knob for timing-sensitive tasks.",
    )
    p.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Scale harbor's agent/verifier timeouts (e.g. 2.0 for slow nodes). "
        "Default 1.0.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Max trial retries for INFRA-transient errors only (capacity / "
        "control-plane / node blips — see _INFRA_RETRY_EXCEPTIONS). This is how "
        "xrlenv absorbs high concurrency against a capacity-capped runtime. Each retry "
        "re-runs the WHOLE trial in a FRESH container; the DOMINANT case is a fail-fast "
        "acquire BEFORE solve.sh runs (so the task usually runs once), but a post-acquire "
        "infra error (e.g. NodeCommandTimeout on an exec) can re-execute solve.sh — the "
        "recorded result is still one outcome per task (re-graded), so a waited-then-passed "
        "trial is a clean pass. Task-content failures (AgentTimeoutError, verifier errors) "
        "are NEVER retried. Default 6 (~7 attempts x ~240 s queue budget ≈ 28 min of "
        "slot-wait coverage); a run with no capacity pressure uses 0 of them.",
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
