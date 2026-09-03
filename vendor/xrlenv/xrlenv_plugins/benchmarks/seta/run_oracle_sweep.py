"""seta (camel-ai/seta-env) onboarding smoke — drives the **upstream**
``harbor`` runner against either the local Docker daemon or an xrlenv cluster.

seta-env is harbor-compatible (``task.toml`` + ``environment/`` +
``solution/solve.sh`` + ``tests/``), so the integration is the **same one
``import_path`` swap** terminal-bench-2 uses:

- ``--local`` → ``xrlenv_plugins.harbor:XrlenvHarborEnvironment``
  (harbor builds each task's ``environment/Dockerfile`` and runs against the
  local Docker daemon — the baseline).
- **default (cluster mode)** →
  ``xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`` (acquires a container
  on a scheduler-picked node and routes exec / archive through xrlenv).

The one seta-specific wrinkle vs terminal-bench-2: seta tasks ship a
**Dockerfile, not a prebuilt ``docker_image``**. We built each task once and
pushed it to the private registry as ``<registry>/seta-env/<id>:main`` (see
``scripts/`` here and ``deploy/registry``). In cluster mode this
smoke passes ``xrlenv_image_template`` as a per-run kwarg so the adapter resolves
each task to its private-registry image — no per-task config, no subclass.

Operator setup (cluster mode), once at bring-up::

    xrlenv up                                          # control plane
    python xrlenv_plugins/benchmarks/seta/build_cache.py --stage populate
    export XRLENV_GRPC_HOST=<control-plane-host>             # dev control plane
    export XRLENV_GRPC_PORT=50051
    export XRLENV_CONSUMER_TOKEN=$(cat ~/.xrlenv/secrets/consumer.token)

Then::

    # Cluster 8-task oracle smoke (default registry <registry-host>:5011):
    .venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py

    # Local baseline (harbor builds the Dockerfile on this host):
    .venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --local

    # Specific tasks / concurrency / a different registry:
    .venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py \\
        --tasks 0,42,100 --max-workers 4 --registry <registry-host>:5011

    # Sweep every cached task (skips the known-unbuildable blacklist):
    .venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --all --max-workers 8

Oracle policy: harbor's ``OracleAgent`` copies each task's ``solution/solve.sh``
into the container and runs it; the verifier writes ``reward.txt`` to
``/logs/verifier/``. A non-passing trial under the oracle is a plumbing bug, not
a model-eval signal.
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

LOGGER = logging.getLogger("xrlenv.smoke.seta-onboarding")

# 8 small contiguous tasks that built + pushed cleanly (none on the blacklist).
SMOKE_TASKS: tuple[str, ...] = tuple(str(i) for i in range(8))

LOCAL_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironment"
CLUSTER_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"

# Infra-transient errors the trial queue may retry — and ONLY these (matched on
# ``type(e).__name__`` via harbor's RetryConfig). Lets a caller request any
# concurrency while xrlenv paces a capacity-CAPPED runtime: an at-cap acquire fails
# fast with CapacityExhausted and the trial re-queues. Acquire is harbor's FIRST
# (cheapest) setup step, so the DOMINANT retry is a fail-fast acquire (before solve.sh
# runs) and the task usually runs once. But a retry re-runs the WHOLE trial in a FRESH
# container, so a post-acquire infra error (e.g. NodeCommandTimeout on an exec) can
# re-execute solve.sh — a new container each time, so only EXTERNAL side effects carry,
# never the recorded result. The job records ONE result per task (waited-then-passed
# counts once). Task-content failures stay OUT — a failed task is never re-rolled into a
# fluke pass. (Mirrors the terminalworld / tb2.1 oracle sweeps.)
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

# Image-ref pieces. The registry host:port comes from
# XRLENV_PRIVATE_REGISTRY_HOST/PORT in .env (no hard-coded default — cluster mode
# fails fast if unset). These two default to how the build pushed the images.
DEFAULT_NAMESPACE = "seta-env"
DEFAULT_IMAGE_TAG = "main"

# seta tasks live under a dedicated shard of the unified harbor cache:
# ``<cache>/seta-env/<id>/``. The shard name namespaces seta tasks away from
# terminal-bench-2's tasks in the same cache — deterministic, no flat-layout
# guesswork. The populate script writes exactly this; the flat ``<cache>/<id>``
# layout is intentionally NOT supported (hard fail).
SETA_CACHE_SHARD = "seta-env"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_jobs_dir() -> Path:
    return _repo_root() / "tmp"


def _default_job_id() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("smoke-seta-%Y%m%d-%H%M%S")


def _harbor_cache_root() -> Path:
    """The unified harbor cache ROOT: ``$XRLENV_BENCHMARK_CACHE``, else fail loud.

    seta and terminal-bench-2 share one cache; seta tasks live under their own
    ``seta-env/`` shard (:data:`SETA_CACHE_SHARD`), so they never collide with tb2's.
    ``benchmark_cache_root`` is the single implementation — it rejects the retired
    XRLENV_HARBOR_CACHE var / xrlenv_harbor_cache path (renamed 2026-07-31) and raises
    when nothing is set. This used to fall back to a home-directory cache instead, which
    answers an operator error with a plausible-but-wrong directory. Lazy import to match
    the plugin style (plugin -> xrlenv is allowed).
    """
    from xrlenv_plugins.benchmarks._benchmark_cache import benchmark_cache_root

    return Path(benchmark_cache_root()).expanduser()


def _seta_shard() -> Path:
    """The seta shard of the harbor cache. Every seta task is
    ``<cache>/seta-env/<id>/``."""
    return _harbor_cache_root() / SETA_CACHE_SHARD


def _registry_from_env() -> str | None:
    """The private registry the seta images live in, from the SAME env vars the
    node / build-push scripts use (``XRLENV_PRIVATE_REGISTRY_HOST`` +
    ``XRLENV_PRIVATE_REGISTRY_PORT``, normally in ``.env``). Returns ``None`` when
    the host is unset — cluster mode then fails fast. ``--registry`` overrides."""
    host = os.environ.get("XRLENV_PRIVATE_REGISTRY_HOST")
    if not host:
        return None
    port = os.environ.get("XRLENV_PRIVATE_REGISTRY_PORT", "5011")
    return f"{host}:{port}"


def _configure_cluster_from_env(args: argparse.Namespace) -> str:
    """Resolve + validate the cluster config from ``.env`` (which ``import
    xrlenv`` auto-loads — no ``source .env`` needed), PRINT what will be used,
    compose the harbor image template (RETURNED for the sweep to pass to the
    adapter via the ``xrlenv_image_template`` kwarg — no process-global env var),
    and FAIL FAST listing every missing required value rather than erroring late
    (e.g. ``AuthDenied`` at acquire)."""
    from xrlenv.client import load_dotenv
    load_dotenv()  # idempotent; ensures .env is applied even if auto-load was off

    grpc_host = os.environ.get("XRLENV_GRPC_HOST")
    grpc_port = os.environ.get("XRLENV_GRPC_PORT", "50051")
    token = os.environ.get("XRLENV_CONSUMER_TOKEN")
    registry = args.registry or _registry_from_env()
    template = (
        f"{registry}/{args.namespace}/{{task_id}}:{args.image_tag}"
        if registry
        else None
    )

    token_state = f"set ({len(token)} chars)" if token else "<MISSING>"
    print("=== seta smoke — cluster config (read from .env + shell env) ===", flush=True)
    print(f"  control plane  : {grpc_host or '<MISSING>'}:{grpc_port}", flush=True)
    print(f"  consumer token : {token_state}", flush=True)
    print(f"  image template : {template or '<MISSING>'}", flush=True)
    print("================================================================", flush=True)

    missing = []
    if not grpc_host:
        missing.append("XRLENV_GRPC_HOST  — dev control plane (e.g. <control-plane-host>)")
    if not token:
        missing.append("XRLENV_CONSUMER_TOKEN  — a valid dev consumer token")
    if not template:
        missing.append(
            "XRLENV_PRIVATE_REGISTRY_HOST  — registry holding the seta images "
            "(or pass --registry)",
        )
    if missing:
        raise SystemExit(
            "cluster mode is missing required config:\n  - "
            + "\n  - ".join(missing)
            + "\n\nSet these in the repo-root .env (the smoke reads it "
            "automatically — no `source .env` needed), or run with --local for "
            "the no-cluster baseline.",
        )
    assert template is not None  # guaranteed non-None by the missing-check above
    return template


def _blacklist_ids() -> set[str]:
    """Task ids known-unbuildable upstream (no image in the registry). Read from
    ``xrlenv_plugins/benchmarks/seta/black_list.txt``; first token of each
    non-blank, non-``#`` line. Missing file → empty (don't filter)."""
    path = _repo_root() / "xrlenv_plugins/benchmarks/seta/black_list.txt"
    ids: set[str] = set()
    if not path.is_file():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.add(s.split()[0])
    return ids


def _locate_task_dir(task_id: str) -> Path:
    """The seta task dir ``<cache>/seta-env/<id>/`` (with the ``solution/``
    harbor's OracleAgent reads). Deterministic: no flat-layout fallback — a
    missing task is a hard fail with a clear message."""
    task_dir = _seta_shard() / task_id
    if (task_dir / "solution" / "solve.sh").is_file():
        return task_dir
    raise SystemExit(
        f"seta task {task_id!r}: {task_dir}/solution/solve.sh not found. seta tasks live "
        f"under the '{SETA_CACHE_SHARD}/' shard of the harbor cache; run "
        f"build_cache.py --stage populate. (The flat <cache>/<id> layout is "
        f"intentionally unsupported.)",
    )


def _discover_all_tasks() -> list[str]:
    """Every seta task in the ``seta-env/`` shard (``solution/solve.sh`` present),
    MINUS the blacklist. The shard is seta's namespace within the shared cache, so
    there is no cross-benchmark guesswork."""
    shard = _seta_shard()
    if not shard.is_dir():
        raise SystemExit(
            f"--all requires the seta shard {shard}. Run "
            f"build_cache.py --stage populate first.",
        )
    blacklist = _blacklist_ids()
    seen = {
        d.name for d in shard.iterdir()
        if d.is_dir() and (d / "solution" / "solve.sh").is_file()
    }
    excluded = seen & blacklist
    if excluded:
        LOGGER.info(
            "excluding %d blacklisted (upstream-unbuildable) task(s): %s",
            len(excluded), ", ".join(sorted(excluded)),
        )
    runnable = sorted(
        seen - blacklist, key=lambda s: (int(s) if s.isdigit() else 1 << 30, s),
    )
    if not runnable:
        raise SystemExit(
            f"--all found no runnable seta tasks under {shard} "
            f"(solution/solve.sh, not blacklisted) — is the shard populated?",
        )
    return runnable


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # audit H9: NO prefix abbreviation (--des / --de / --d) may resolve to --dest —
        # the cache root must come only from XRLENV_BENCHMARK_CACHE, never a CLI override
        # smuggled past the wrapper's exact-form reject.
        allow_abbrev=False,
        prog="seta-oracle-sweep",
        description=(
            "Run the upstream harbor runner over seta-env tasks against local "
            "Docker (--local) or an xrlenv cluster (default). Same import_path "
            "swap as terminal-bench-2; cluster mode resolves each task to its "
            "private-registry image via the xrlenv_image_template per-run kwarg."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--local", action="store_true",
                   help="Local-mode harbor Environment (builds environment/"
                        "Dockerfile on this host). Baseline before the cluster.")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task ids (overrides the 8-task default). "
                        "Mutually exclusive with --all.")
    p.add_argument("--all", action="store_true",
                   help="Every cached task (minus the upstream blacklist). "
                        "Mutually exclusive with --tasks.")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Trial concurrency (default 1). Threads into harbor's "
                        "JobConfig.n_concurrent_trials.")
    p.add_argument("--registry", default=None,
                   help="Private registry host:port holding the seta images. "
                        "Default: $XRLENV_PRIVATE_REGISTRY_HOST:"
                        "$XRLENV_PRIVATE_REGISTRY_PORT from .env (required in "
                        "cluster mode if not passed). Cluster mode only.")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                   help=f"Image repo namespace — the '<ns>' in "
                        f"<registry>/<ns>/<id>:<tag>. Must match what the build "
                        f"pushed (IMAGE_NAMESPACE in build_plan_gen.py); change "
                        f"only if you re-tagged elsewhere. Default {DEFAULT_NAMESPACE}.")
    p.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG,
                   help=f"Image tag — the '<tag>' in <registry>/<ns>/<id>:<tag> "
                        f"(the git ref the images were built from). Default "
                        f"{DEFAULT_IMAGE_TAG}.")
    p.add_argument("--save-artifacts", "--jobs-dir", dest="save_artifacts",
                   default=None, metavar="PATH",
                   help=f"Override harbor's jobs_dir (default "
                        f"{_default_jobs_dir()}/, gitignored). --jobs-dir is an alias — "
                        f"unifies seta with the other benchmarks' run_oracle_sweep.py.")
    p.add_argument("--job-id", default=None,
                   help="Label under <jobs_dir>/. Default smoke-seta-<UTC>.")
    p.add_argument("--retries", type=int, default=6, metavar="N",
                   help="Max trial retries for INFRA-transient errors ONLY (capacity / "
                        "control-plane / node blips — see _INFRA_RETRY_EXCEPTIONS). "
                        "Task-content failures are NEVER re-rolled. Default 6 (matches the "
                        "other benchmarks + benchmarks.yaml `retries: 6`; the full wrapper "
                        "relies on this default, so 0 left full-mode seta with no infra "
                        "retries — audit M5).")
    p.add_argument("--content-retries", type=int, default=0, metavar="N",
                   help="Per-TASK content-retry rounds: after a run, re-run the tasks that "
                        "came back non-passing (reward=0 by _trial_passes) up to N more "
                        "times; a task is solved if ANY attempt passes. Catches "
                        "nondeterministic reward=0 flakes (a transient DNS/verifier blip) "
                        "that --retries (infra-only) deliberately never re-rolls. Default 0.")
    return p


def _resolve_task_list(args: argparse.Namespace) -> list[str]:
    if args.all and args.tasks is not None:
        raise SystemExit("--all and --tasks are mutually exclusive.")
    if args.all:
        return _discover_all_tasks()
    if args.tasks is not None:   # None=absent; '' is present-but-empty (audit M5)
        requested = [s.strip() for s in args.tasks.split(",") if s.strip()]
        if not requested:
            raise SystemExit(f"--tasks {args.tasks!r} selected no tasks (audit M5)")
        blacklisted = [t for t in requested if t in _blacklist_ids()]
        if blacklisted:
            LOGGER.warning(
                "requested task(s) %s are on the upstream blacklist "
                "(no image in the registry) — they will fail at acquire. "
                "See xrlenv_plugins/benchmarks/seta/black_list.txt.",
                blacklisted,
            )
        return requested
    return list(SMOKE_TASKS)


def _build_job_config(
    *, task_ids: list[str], local: bool, jobs_dir: Path, job_id: str,
    n_concurrent_trials: int, retries: int = 0, image_template: str | None = None,
) -> Any:
    from harbor.models.job.config import JobConfig, RetryConfig
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
        # Infra-transient retries ONLY (include_exceptions gates to
        # _INFRA_RETRY_EXCEPTIONS) so eval signal is never re-rolled. Short
        # near-constant backoff: the acquire's own queue_timeout is the "wait for a
        # slot", so re-queue fast rather than idle on a long backoff.
        retry=RetryConfig(
            max_retries=retries,
            include_exceptions=set(_INFRA_RETRY_EXCEPTIONS),
            min_wait_sec=2.0,
            wait_multiplier=1.0,
            max_wait_sec=10.0,
        ),
        environment=EnvironmentConfig(
            import_path=import_path,
            kwargs=(
                {"xrlenv_image_template": image_template} if image_template else {}
            ),
        ),
        agents=[AgentConfig()],
        tasks=[TaskConfig(path=_locate_task_dir(t)) for t in task_ids],
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
    if trial_result.exception_info is not None:
        return False, f"exception: {trial_result.exception_info.exception_type}"
    vr = trial_result.verifier_result
    if vr is None or vr.rewards is None or not vr.rewards:
        return False, "no verifier rewards recorded"
    failures = [k for k, v in vr.rewards.items() if not (v > 0)]
    if failures:
        return False, f"non-positive reward(s): {failures}"
    return True, None


def _summarise(trial_results: list[Any], expected: int) -> tuple[int, dict[str, Any]]:
    print("\n=== seta onboarding smoke ===")
    passed = 0
    failed: list[str] = []
    per_trial: list[dict[str, Any]] = []
    for tr in sorted(trial_results, key=lambda r: r.task_name):
        ok, reason = _trial_passes(tr)
        if ok:
            passed += 1
        else:
            failed.append(tr.task_name)
        rewards = tr.verifier_result.rewards if tr.verifier_result is not None else None
        row = {"task": tr.task_name, "trial": tr.trial_name, "passed": ok,
               "rewards": rewards, "reason": reason}
        per_trial.append(row)
        pprint.pp(row)
    print(f"\n{passed} / {expected} trial(s) passed under the oracle policy.")
    if failed:
        print(f"failed: {failed}\nUnder the oracle, a non-passing trial is a "
              f"plumbing bug — check <jobs_dir>/<job_name>/<trial>/trial.log.")
    return (0 if passed == expected else 1), {
        "expected": expected, "passed": passed, "failed": failed, "trials": per_trial,
    }


async def _run(args: argparse.Namespace) -> int:
    import harbor

    task_ids = _resolve_task_list(args)
    LOGGER.info(
        "running %d seta task(s) in %s mode at concurrency=%d: %s",
        len(task_ids), "local" if args.local else "cluster",
        args.max_workers, ", ".join(task_ids),
    )

    image_template: str | None = None
    if not args.local:
        # Reads .env, prints the resolved config, composes the image template, and
        # fails fast if anything required is missing.
        image_template = _configure_cluster_from_env(args)

    jobs_dir = (
        Path(args.save_artifacts).expanduser()
        if args.save_artifacts else _default_jobs_dir()
    )
    job_id = args.job_id or _default_job_id()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("artifacts: %s", jobs_dir / job_id)

    # Content-retry loop: run the tasks, then re-run ONLY the non-passing ones (by
    # _trial_passes — this benchmark's own rule) up to --content-retries times. A task
    # counts as solved if ANY attempt passes; a reward=0 flake (e.g. a transient DNS
    # blip) gets a fresh trial, while a genuine failure persists across attempts.
    # Distinct from --retries, which is infra-transient-only and never re-rolls a
    # reward outcome. content_retries=0 (default) runs the loop exactly once —
    # behavior-identical to a single job run.
    best: dict[str, Any] = {}
    remaining = list(task_ids)
    cr = int(args.content_retries)
    for attempt in range(1 + cr):
        jid = job_id if attempt == 0 else f"{job_id}-retry{attempt}"
        config = _build_job_config(
            task_ids=remaining, local=args.local, jobs_dir=jobs_dir, job_id=jid,
            n_concurrent_trials=args.max_workers, retries=args.retries,
            image_template=image_template,
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
            LOGGER.info(
                "content-retry %d/%d: re-running %d non-passing task(s): %s",
                attempt + 1, cr, len(remaining), ", ".join(remaining),
            )

    # Fold the content-retry rounds' sibling -retryN dirs back into the attempt-0
    # dir so the operator sees ONE result set per task, not stray retry folders.
    from xrlenv_plugins.benchmarks._sweep_retry import consolidate_retry_dirs
    consolidate_retry_dirs(jobs_dir, job_id, cr)
    exit_code, _summary = _summarise(
        [best[t] for t in task_ids if t in best], len(task_ids),
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
