"""SWE-bench Verified onboarding smoke — runs the **upstream**
``swebench`` harness against either the local Docker daemon or an
xrlenv cluster, with the only swap being the docker client.

Two modes
=========

- ``--local`` — uses ``docker.from_env()``. Identical to what you'd
  see running swebench's own ``run_evaluation`` CLI on this host;
  serves as a baseline.
- **default (cluster mode)** — uses ``xrlenv.from_env()``, which
  reads connection config from the environment. Set
  ``XRLENV_GRPC_HOST`` / ``XRLENV_GRPC_PORT`` /
  ``XRLENV_CONSUMER_TOKEN`` (and optionally ``XRLENV_GRPC_SECURE``)
  in the consumer's shell; the smoke driver itself contains
  literally one xrlenv-specific line: ``client = xrlenv.from_env()``.

The drop-in contract
====================

This smoke deliberately contains **exactly one** xrlenv-specific
piece of code:

.. code-block:: python

    client = xrlenv.from_env()

Everything else is upstream swebench. No
``Client.acquire_container(...)``, no
``client.plan_image_distribution(...)``, no xrlenv-shaped pre-loop
setup. Compare with the original swebench harness, which uses
``client = docker.from_env()`` — that's the *only* line that
changes when you onboard a swebench-style harness onto xrlenv.

The smoke does NOT vendor a parallel implementation of swebench's
grading. It calls
``swebench.harness.run_evaluation.run_instance`` directly and lets
upstream's ``get_eval_report`` write the per-instance
``report.json`` exactly as it would in a non-xrlenv run.

Concurrency is the user's choice
================================

``--max-workers`` defaults to **1 (serial)**. The smoke runs
concurrent acquires only when the operator explicitly opts in.
Why: different harnesses tolerate different concurrency models.
swebench's own ``run_instances`` is thread-pool-shaped and is
thread-safe; benchmarks like OSWorld carry thread-unsafe internals
(in-container subprocess management, hooks, etc.) and would break
under threading — those run with ``--max-workers=1`` (serial) or
the operator wraps the smoke with ``multiprocessing`` for
process-per-instance isolation, or ``asyncio`` for an event-loop-
shaped harness, etc. xrlenv's docker-py drop-in is concurrency-
neutral: any model the operator picks works because the drop-in
itself doesn't share mutable state across calls — it's purely
mechanism, never policy. See the README's "Concurrency is the
operator's choice" section.

Operator setup (cluster mode)
=============================

The audience does NOT run xrlenv-specific commands inside the
smoke. The OPERATOR runs them once at cluster bring-up, in order:

1. ``xrlenv up`` — boot the control plane (one-shot; idempotent).
2. **Optional, recommended for ``--all`` sweeps**:
   ``xrlenv images plan --refs refs/all-verified.txt
   --eager-prefetch`` — FFD bin-packs the cluster's image bytes,
   pre-fetches each preferred_home node before consumers start.
   For 8-instance smokes the reactive image-affinity scheduler
   handles distribution naturally; the CLI is for closed-set
   batch sweeps.
3. Set ``XRLENV_GRPC_HOST`` / ``XRLENV_GRPC_PORT`` /
   ``XRLENV_CONSUMER_TOKEN`` in the consumer's shell (typically
   in the operator's deploy script that invokes the smoke).

The audience then runs this smoke unchanged.

Artifact archiving
==================

By default the smoke runs swebench's harness in a tempdir that is
deleted at exit, mirroring how an isolated CI run would behave —
nothing leaks into the working tree.

Pass ``--save-artifacts`` to keep the harness's per-instance
artifact tree (``logs/run_evaluation/<run_id>/<model>/<instance>/``)
under ``<repo>/tmp/<job-id>/`` (default; gitignored) so the
operator can inspect grader output, container logs, and the gold
patch the smoke applied. The flag accepts an optional explicit
path:

::

    --save-artifacts                 # repo/tmp/ (default)
    --save-artifacts ~/scratch/jobs  # explicit path

Layout matches ``tests/smoke/test_swebench_drop_in.py`` so
operators can navigate either smoke's output the same way.

Default task set
================

8 instances spanning 3 repos x 3 difficulty bands (see
:data:`SMOKE_INSTANCES`). Pass ``--all`` for the full
500-instance Verified set, or ``--instances a,b,c`` for a custom
list.

Oracle policy
=============

For an "is this plumbing correct" smoke, every instance is paired
with the **dataset's gold patch as the prediction**. Upstream's
grader applies the gold patch + runs the per-repo test command;
since gold-patch == correct-fix, every instance should resolve.
A non-resolving instance under the oracle policy is a plumbing
bug, not a model evaluation result.

Usage
=====

::

    # Local 8-instance smoke (baseline against local Docker daemon):
    .venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \\
        --local

    # Cluster 8-instance smoke, env-var-driven, serial:
    export XRLENV_GRPC_HOST=127.0.0.1
    export XRLENV_GRPC_PORT=50051
    export XRLENV_CONSUMER_TOKEN=$(cat ~/.xrlenv/secrets/consumer.token)
    .venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py

    # Cluster 8-instance smoke, 4-way concurrent (only safe for
    # thread-safe harnesses; swebench is):
    .venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \\
        --max-workers 4

    # Keep harness artifacts under <repo>/tmp/<job-id>/:
    .venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \\
        --max-workers 4 --save-artifacts

    # Full 500-instance Verified sweep (cluster mode, concurrent):
    .venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \\
        --all --max-workers 8 --save-artifacts

    # Custom instance list:
    .venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \\
        --instances django__django-11099,sympy__sympy-13615
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import logging
import os
import pprint
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import xrlenv
from xrlenv.observability.logging import configure_logging

LOGGER = logging.getLogger("xrlenv.smoke.swebench-onboarding")

# 8-instance smoke set: 3 repos x 3 difficulty bands. Cheap to run
# end-to-end (~5-15 min wall-clock) and covers the breadth that a
# multi-node cluster's image-affinity scheduler exercises.
SMOKE_INSTANCES: tuple[str, ...] = (
    "astropy__astropy-7166",
    "django__django-11099",
    "sympy__sympy-18189",
    "astropy__astropy-12907",
    "astropy__astropy-14182",
    "sympy__sympy-13615",
    "django__django-11138",
    "sympy__sympy-12489",
)

# Upstream-published prebuilt-image namespace. Setting this on
# ``make_test_spec`` flips ``TestSpec.is_remote_image`` to True,
# which makes the harness call ``client.images.get(...)`` and
# ``client.images.pull(...)`` instead of building locally. In
# cluster mode, the chosen node's ImageCacheManager handles the
# pull; in local mode, the local Docker daemon does.
SWEBENCH_NAMESPACE = "swebench"


# ──────────────────────────────────────────────────────────────────────────────
# Artifact-archiving helpers (inlined from tests/smoke/_artifacts.py).
#
# The shape MUST match `tests/smoke/test_swebench_drop_in.py` so an
# operator navigating either smoke's output finds the same layout.
# Inlined rather than imported because `examples/` shouldn't depend
# on `tests/` — keeps the onboarding directory self-contained.
# ──────────────────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    """`<repo>/` resolved from this file's location.

    smoke.py lives at examples/benchmarks-onboarding/swebench-verified/,
    so the repo root is three parents up.
    """
    return Path(__file__).resolve().parents[3]


def _default_save_artifacts_root() -> Path:
    """``<repo>/tmp/`` — gitignored; default ``--save-artifacts`` dest.

    Created lazily on first use; operators don't need to mkdir
    ahead of time. Mirrors
    ``tests/smoke/_artifacts.default_save_artifacts_root``.
    """
    return _repo_root() / "tmp"


def _default_job_id() -> str:
    """``smoke-swebench-verified-YYYYMMDD-HHMMSS`` UTC timestamp
    default for ``--job-id``. Lexicographically sortable so manual
    ``ls`` lists newest last. Mirrors
    ``tests/smoke/_artifacts.default_job_id`` shape; the
    benchmark-name prefix lets multiple smokes share ``tmp/``
    without collision."""
    return _dt.datetime.utcnow().strftime(
        "smoke-swebench-verified-%Y%m%d-%H%M%S",
    )


# ──────────────────────────────────────────────────────────────────────────────
# HF offline mode (mirrors tests/smoke/test_swebench_drop_in.py).
#
# By default, ``datasets.load_dataset`` does revision-check HEAD
# requests + legacy-loader 404 probes (``SWE-bench_Verified.py``,
# ``dataset_infos.json``, ``.huggingface.yaml``) on every call, even
# when the parquet is fully cached locally. In offline mode, it
# reads straight from the HF cache with zero network and zero httpx
# INFO chatter. The flags are read at *import* time from
# ``HF_DATASETS_OFFLINE`` / ``HF_HUB_OFFLINE`` env vars; once those
# libs are imported, env-var flips have no effect — we mutate the
# in-memory module globals instead, restored on exit.
# ──────────────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _hf_offline_mode() -> Iterator[None]:
    """Force HF offline mode for the duration of the block.

    Mirrors ``tests/smoke/test_swebench_drop_in.py::_hf_offline_mode``;
    inlined rather than imported to keep the onboarding directory
    self-contained (examples/ should not depend on tests/).
    """
    import datasets.config as _ds_cfg
    import huggingface_hub.constants as _hf_const

    prev_ds_offline = _ds_cfg.HF_DATASETS_OFFLINE
    prev_hub_offline = _hf_const.HF_HUB_OFFLINE
    _ds_cfg.HF_DATASETS_OFFLINE = True
    _hf_const.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        _ds_cfg.HF_DATASETS_OFFLINE = prev_ds_offline
        _hf_const.HF_HUB_OFFLINE = prev_hub_offline


def _load_swebench_dataset_cached(
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
) -> list[dict[str, Any]]:
    """Wrap ``swebench.harness.run_evaluation.load_swebench_dataset``
    with the offline-first / online-fallback pattern from
    ``tests/smoke/test_swebench_drop_in.py``.

    Cached runs (the common case after the first invocation) read
    straight from ``~/.cache/huggingface/datasets/`` with zero
    network traffic. Cold runs (fresh host, or operator nuked the
    HF cache) transparently fall back to an online fetch which
    populates the cache so subsequent runs are quiet.
    """
    from swebench.harness.run_evaluation import load_swebench_dataset

    with _hf_offline_mode():
        try:
            return load_swebench_dataset(dataset_name, split)
        except (FileNotFoundError, ConnectionError, OSError):
            # Cache miss — fall through to online below. The
            # noisy 404-probe pattern on a true first run is the
            # one-time price for populating the cache.
            LOGGER.info(
                "HF cache miss for %s; falling back to online fetch "
                "(populates ~/.cache/huggingface for next time)",
                dataset_name,
            )
    return load_swebench_dataset(dataset_name, split)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swebench-verified-smoke",
        description=(
            "Run the upstream swebench harness against either local "
            "Docker (--local) or an xrlenv cluster (default). Cluster "
            "mode reads connection config from XRLENV_GRPC_HOST / "
            "XRLENV_GRPC_PORT / XRLENV_CONSUMER_TOKEN — there are NO "
            "xrlenv-specific CLI flags on this smoke. The audience's "
            "harness contains one xrlenv line: ``xrlenv.from_env()``."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--local", action="store_true",
        help="Use docker.from_env() against the local Docker daemon. "
             "Baseline mode for verifying the harness works at all "
             "before pointing it at the cluster.",
    )
    p.add_argument(
        "--instances", default=None,
        help="Comma-separated instance_ids to run. Overrides the "
             "default 8-instance smoke set. Mutually exclusive "
             "with --all.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Run the full 500-instance Verified set. Mutually "
             "exclusive with --instances. Operator should run "
             "``xrlenv images plan --refs refs/all-verified.txt "
             "--eager-prefetch`` first for optimal image distribution.",
    )
    p.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-instance test timeout in seconds (default 1800 = "
             "30 min, matches swebench default).",
    )
    p.add_argument(
        "--run-id", default="xrlenv-onboarding-smoke",
        help="Run id passed to swebench's run_instance (controls "
             "log dir naming + container names).",
    )
    p.add_argument(
        "--max-workers", type=int, default=1,
        help="Concurrency at the driver level (default 1 = serial). "
             "Concurrency is the operator's choice — pick a model "
             "that matches your harness's thread-safety. swebench's "
             "harness is thread-safe; ``--max-workers N`` runs N "
             "instances concurrently via ThreadPoolExecutor. For "
             "harnesses with thread-unsafe internals (e.g. OSWorld), "
             "stay at 1 or wrap with multiprocessing externally. The "
             "xrlenv drop-in itself is concurrency-neutral.",
    )
    p.add_argument(
        "--save-artifacts", nargs="?",
        default=None, const=str(_default_save_artifacts_root()),
        metavar="PATH",
        help=f"Keep harness artifacts (logs/run_evaluation/...) under "
             f"<PATH>/<job-id>/. Without an explicit path defaults to "
             f"``{_default_save_artifacts_root()}`` (gitignored). "
             f"Without the flag, artifacts live in a tempdir reaped at "
             f"exit. Layout matches tests/smoke/test_swebench_drop_in.py.",
    )
    p.add_argument(
        "--job-id", default=None,
        help="Label under <save-artifacts>/. Default is "
             "``smoke-YYYYMMDD-HHMMSS`` UTC. Useful for tagging runs "
             "with model/version labels (e.g. claude-opus-4-7-50-v1).",
    )
    return p


def _resolve_instance_list(args: argparse.Namespace) -> list[str]:
    if args.all and args.instances:
        raise SystemExit(
            "--all and --instances are mutually exclusive."
        )
    if args.all:
        instances = _load_swebench_dataset_cached()
        return [inst["instance_id"] for inst in instances]
    if args.instances:
        return [s.strip() for s in args.instances.split(",") if s.strip()]
    return list(SMOKE_INSTANCES)


def _build_docker_client(args: argparse.Namespace) -> Any:
    """Return either a real ``docker.DockerClient`` (local mode) or
    an ``xrlenv.XrlenvDockerClient`` (cluster mode, env-var-driven).

    In cluster mode the entire xrlenv-specific surface is one line.
    The swebench harness consumes either transparently — that's
    the whole point of the docker-py drop-in.
    """
    if args.local:
        import docker
        LOGGER.info("local mode: docker.from_env()")
        return docker.from_env()
    # Cluster mode: env-var-driven xrlenv.from_env() with NO kwargs.
    # The audience's harness reads literally one xrlenv-specific
    # line. Operator sets XRLENV_GRPC_HOST etc. at deploy time.
    if not os.environ.get("XRLENV_GRPC_HOST"):
        raise SystemExit(
            "cluster mode: XRLENV_GRPC_HOST not set. Either pass "
            "--local for the baseline, or export XRLENV_GRPC_HOST / "
            "XRLENV_GRPC_PORT / XRLENV_CONSUMER_TOKEN before running "
            "this smoke (typically done by the operator's deploy "
            "script alongside ``xrlenv up``).",
        )
    LOGGER.info(
        "cluster mode: xrlenv.from_env() — host=%s port=%s",
        os.environ.get("XRLENV_GRPC_HOST"),
        os.environ.get("XRLENV_GRPC_PORT", "50051"),
    )
    return xrlenv.from_env()


def _load_dataset_for_instances(
    instance_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch upstream's full Verified dataset and return the subset
    matching ``instance_ids`` keyed by id. Reads from the HF cache
    when populated; falls back to an online fetch on cold cache."""
    LOGGER.info(
        "loading SWE-bench_Verified dataset (SWE-bench/SWE-bench_Verified, "
        "test split, HF cache-first)",
    )
    instances = _load_swebench_dataset_cached()
    by_id = {inst["instance_id"]: inst for inst in instances}
    missing = [iid for iid in instance_ids if iid not in by_id]
    if missing:
        raise SystemExit(
            f"unknown instance_id(s): {missing}. Run with --all to "
            f"see the full list, or check that you're on the latest "
            f"swebench package (pip install -U swebench).",
        )
    return {iid: by_id[iid] for iid in instance_ids}


def _run_one_instance(
    instance: dict[str, Any], client: Any, *,
    run_id: str, timeout: int,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Drive a single instance through upstream's run_instance.

    Prediction = the dataset's gold ``patch`` (oracle policy). All
    image-related calls (``client.images.get`` / ``.pull``) become
    no-ops in cluster mode — the chosen node's ImageCacheManager
    pulls on first acquire when needed. The harness's normal
    ``container.exec_run`` etc. flow is unchanged.

    P1.7.B.3: wraps ``run_instance(...)`` in
    ``with xrlenv.rollout_metadata(artifact_path=..., displayed_name=...):``
    so the cluster's RawRolloutRecord carries:

    - ``displayed_name = instance_id`` — the admin's
      ``/rollouts`` row reads "astropy__astropy-7166" instead of
      a synthetic uuid.
    - ``artifact_path = <artifact_root>/logs/run_evaluation/
      <run_id>/<model>/<instance_id>/`` — when set,
      admin's per-rollout detail page best-effort-renders this
      directory's contents inline if the path resolves on the
      control-plane host's filesystem; otherwise displays the
      path string for the operator to navigate to externally.
      Only set when ``--save-artifacts`` was passed (otherwise
      the harness writes into a tempdir that's reaped at exit
      and the path wouldn't be useful).
    """
    from swebench.harness.constants import (
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
    )
    from swebench.harness.run_evaluation import make_test_spec, run_instance

    instance_id = instance["instance_id"]
    test_spec = make_test_spec(
        instance,
        # Flip is_remote_image=True so the harness skips local
        # image building and only probes presence.
        namespace=SWEBENCH_NAMESPACE,
        instance_image_tag="latest",
    )
    pred: dict[str, Any] = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: "xrlenv-onboarding-oracle",
        KEY_PREDICTION: instance["patch"],
    }
    LOGGER.info("[%s] starting", instance_id)
    # Build the per-instance artifact_path under the user-chosen
    # save-root. swebench's run_instance writes to
    # ``<cwd>/logs/run_evaluation/<run_id>/<model>/<instance_id>/``;
    # we cd'd into ``artifact_root`` upstream so the directory
    # below resolves correctly post-run.
    instance_artifact_path: str | None = None
    if artifact_root is not None:
        instance_artifact_path = str(
            artifact_root / "logs" / "run_evaluation"
            / run_id / "xrlenv-onboarding-oracle" / instance_id
        )
    with xrlenv.rollout_metadata(
        artifact_path=instance_artifact_path,
        displayed_name=instance_id,
    ):
        result = run_instance(
            test_spec=test_spec,
            pred=pred,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id=run_id,
            timeout=timeout,
        )
    LOGGER.info(
        "[%s] resolved=%s",
        instance_id, result.get("resolved"),
    )
    return result


def _drive_instances(
    instances: dict[str, dict[str, Any]],
    client: Any,
    *,
    run_id: str,
    timeout: int,
    max_workers: int,
    artifact_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the per-instance loop. ``max_workers=1`` is serial; >1
    uses ``concurrent.futures.ThreadPoolExecutor``.

    Concurrency is the operator's policy choice — see the module
    docstring's "Concurrency is the user's choice" section. The
    xrlenv drop-in itself is concurrency-neutral; this function
    picks ONE model (threading) because swebench's harness is
    thread-safe. Operators with thread-unsafe harnesses stay at
    serial or wrap the smoke with a different mechanism externally.

    ``artifact_root``: when ``--save-artifacts`` was passed, this is
    the persistent root the cwd was set to before this loop. Used
    to compute the per-instance ``artifact_path`` that flows into
    the cluster's RawRolloutRecord via ``xrlenv.rollout_metadata``.
    None when artifacts are reaped at exit (no point recording a
    tempdir path on the cluster's record).
    """
    results: dict[str, dict[str, Any]] = {}

    def _safe_run(
        iid: str, inst: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        try:
            return iid, _run_one_instance(
                inst, client, run_id=run_id, timeout=timeout,
                artifact_root=artifact_root,
            )
        except Exception as exc:
            LOGGER.exception("[%s] crashed", iid)
            return iid, {
                "instance_id": iid,
                "completed": False,
                "resolved": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if max_workers <= 1:
        for iid, inst in instances.items():
            _, results[iid] = _safe_run(iid, inst)
        return results

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="swebench-smoke",
    ) as pool:
        futures = [
            pool.submit(_safe_run, iid, inst)
            for iid, inst in instances.items()
        ]
        for fut in concurrent.futures.as_completed(futures):
            iid, result = fut.result()
            results[iid] = result
    return results


def _summarise(
    results: dict[str, dict[str, Any]], expected: int,
) -> tuple[int, dict[str, Any]]:
    """Print one line per instance + a final tally. Returns
    ``(exit_code, summary_dict)`` — the dict is what's archived to
    ``summary-*.json`` under save-artifacts."""
    print("\n=== swebench-verified onboarding smoke ===")
    resolved = 0
    failed: list[str] = []
    per_instance: list[dict[str, Any]] = []
    for iid, result in sorted(results.items()):
        ok = bool(result.get("resolved"))
        if ok:
            resolved += 1
        else:
            failed.append(iid)
        row = {
            "instance_id": iid,
            "completed": result.get("completed"),
            "resolved": ok,
            "error": result.get("error"),
        }
        per_instance.append(row)
        pprint.pp(row)
    print(
        f"\n{resolved} / {expected} instance(s) resolved under the "
        f"oracle policy.",
    )
    if failed:
        print(
            f"failed: {failed}\n"
            f"Under the gold-patch oracle policy, a non-resolved "
            f"instance is a plumbing bug, not a model-eval signal — "
            f"check the per-instance log under "
            f"``logs/run_evaluation/<run_id>/<model>/<instance_id>/``.",
        )
    summary = {
        "expected": expected,
        "resolved": resolved,
        "failed": failed,
        "instances": per_instance,
    }
    return (0 if resolved == expected else 1), summary


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _build_parser().parse_args()
    configure_logging()

    instance_ids = _resolve_instance_list(args)
    # Log the full list — fits on one comma-joined line whether 8
    # or 500. The earlier "first 3:" truncation was a readability
    # tweak that read like an execution count; the unambiguous
    # form prints all of them.
    LOGGER.info(
        "running %d instance(s) at concurrency=%d: %s",
        len(instance_ids), args.max_workers,
        ", ".join(instance_ids),
    )
    instances = _load_dataset_for_instances(instance_ids)
    client = _build_docker_client(args)

    save_root = Path(args.save_artifacts) if args.save_artifacts else None
    job_id = args.job_id or _default_job_id()

    # swebench's run_instance writes ``logs/run_evaluation/<run_id>/
    # <model>/<instance>/`` cwd-relative. Two artifact-storage
    # modes, controlled by where we cwd to before the per-instance
    # loop:
    #
    # **Direct-write** (``--save-artifacts`` set): cwd =
    # ``<save_root>/<job_id>/``. Per-instance reports
    # (report.json, run_instance.log, test_output.txt, patch.diff)
    # become durable the moment swebench writes them — Ctrl-C /
    # crash / mid-run failure leaves completed instances' reports
    # intact. Trade-off vs the tempdir-then-copy pattern: a buggy
    # smoke could pollute the save_root path with stray files.
    # For an onboarding smoke that's an acceptable exchange for
    # the per-instance-durability win.
    #
    # **Tempdir** (no flag): cwd = a fresh tempdir reaped at exit.
    # True ephemeral; nothing leaks into the working tree.
    #
    # cwd is set ONCE before the loop. Concurrent threads share
    # the process cwd, so the set-and-restore is loop-scoped, not
    # per-instance. swebench's path computation reads cwd at
    # ``run_instance`` call time, not at module import, so the
    # set-once-before-loop pattern works under any concurrency
    # model the operator picks.
    prev_cwd = Path.cwd()
    try:
        if save_root is not None:
            artifact_root = save_root / job_id
            artifact_root.mkdir(parents=True, exist_ok=True)
            LOGGER.info("save-artifacts: %s/", artifact_root)
            os.chdir(artifact_root)
            try:
                results = _drive_instances(
                    instances, client,
                    run_id=args.run_id,
                    timeout=args.timeout,
                    max_workers=args.max_workers,
                    # Per-instance ``artifact_path`` flows into
                    # the cluster's RawRolloutRecord so admin's
                    # detail page can list the harness's logs.
                    artifact_root=artifact_root,
                )
            finally:
                os.chdir(prev_cwd)
            exit_code, summary = _summarise(results, len(instances))
            # Per-run summary file. Small + written at end; the
            # per-instance reports above are the load-bearing
            # artifacts and were durable from the moment swebench
            # wrote them. Microsecond-precision timestamp so
            # back-to-back invocations don't clobber each other.
            run_ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            summary_path = artifact_root / f"summary-{run_ts}.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"artifacts at {artifact_root}")
            return exit_code

        # Tempdir mode: reaped at exit.
        with tempfile.TemporaryDirectory(
            prefix="xrlenv-swebench-onboarding-smoke-",
        ) as td:
            os.chdir(td)
            try:
                results = _drive_instances(
                    instances, client,
                    run_id=args.run_id,
                    timeout=args.timeout,
                    max_workers=args.max_workers,
                )
            finally:
                os.chdir(prev_cwd)
            exit_code, _summary = _summarise(results, len(instances))
            return exit_code
    finally:
        # Clean teardown — for the xrlenv drop-in this shuts the
        # owned runner + Client down. For docker.from_env() it
        # closes the underlying HTTP session.
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
