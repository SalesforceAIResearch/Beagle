"""run_oracle_sweep.py — the SWE-bench Verified correctness gate.

Drives the **upstream** swebench harness (``run_instance`` + ``get_eval_report``)
against the xrlenv cluster via the **docker-py drop-in** — the only xrlenv-specific
line is ``client = xrlenv.from_env()`` (``--local`` uses ``docker.from_env()`` for a
baseline). This is NOT the harbor/pier ``import_path`` shape: swebench has its own
harness that speaks docker-py, so we swap the docker client and let the harness run
unmodified (GUIDELINE §7.1 / the docker-py-drop-in path).

Oracle policy: each instance's prediction is the **dataset's gold patch**, read from
the cache (``build_cache.py`` materialized ``<cache>/swebench-verified/<id>/``). The
harness applies it + runs the per-repo tests + writes ``report.json`` with
``resolved: true/false`` (upstream's own field — we never invent a resolution rule).
Every gold-patch-as-prediction should resolve; a non-resolving instance under the
oracle is a plumbing/content bug, not a model signal.

Contract (GUIDELINE §3.3): reads instances from the cache (fail loud if unpopulated),
requires ``XRLENV_GRPC_HOST`` in cluster mode, exit 0 iff **every** instance resolved,
per-run ``summary.json`` under ``<jobs-dir>/<job-id>/``. Two retry layers live HERE, in this
driver (the shell wrapper only forwards their flags): ``--retries`` re-runs a trial on
INFRA-transient errors ONLY, and ``--content-retries`` re-runs an UNRESOLVED instance to
distinguish a one-off flake from a real regression.

Usage::

    # 8-instance smoke on the cluster (env-var-driven):
    python xrlenv_plugins/benchmarks/swebench_verified/run_oracle_sweep.py --smoke
    # a subset:
    python .../run_oracle_sweep.py --tasks django__django-11099,sympy__sympy-13615
    # the whole cached corpus, 8-way concurrent, infra-retry 6:
    python .../run_oracle_sweep.py --all --max-workers 8 --retries 6 --jobs-dir ./tmp
    # local docker baseline (no cluster):
    python .../run_oracle_sweep.py --smoke --local
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Any

import xrlenv
from xrlenv.backends.base import CpuIsolation
from xrlenv.observability.logging import configure_logging

LOGGER = logging.getLogger("xrlenv.swebench-verified.oracle")

# Self-contained per the plugin convention (the other benchmarks avoid intra-
# namespace-package imports, which mypy double-counts). Kept in sync with
# build_cache.py / build_plan_gen.py.
SHARD = "swebench-verified"
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

# Setting namespace="swebench" on make_test_spec flips TestSpec.is_remote_image=True,
# so the harness pulls the prebuilt image instead of building locally.
SWEBENCH_NAMESPACE = "swebench"

# Per-task cpuset PINNING (P2/P6) for instances whose tests spawn OpenMP/BLAS thread pools
# sized to the VISIBLE cpu count (``os.cpu_count()`` / affinity = every host core), NOT the
# container's CFS quota. Unpinned (quota-only), those host-core-many threads thrash against the
# 2-core quota (spin-barrier stall) and the eval HANGS — 0 tests complete before the timeout.
# Pinning fixes affinity to the CPU budget so the pools match the CPU the container can use.
# The docker-py-drop-in analogue of harbor/tb2.1's ``XRLENV_CPU_PINNING`` task-env marker —
# PER-TASK, not global. The map's value is the isolation LEVEL:
#   * REQUIRED — pin-or-reschedule: the pin can NEVER silently degrade to quota-only (the state
#     that hangs it). Placed ONLY on isolation-capable nodes with free pinnable cores; a
#     transient PinCapacityExhausted is re-admitted on a sibling capable node by the infra-retry
#     layer. Use for instances that deterministically hang unpinned (a degrade = a hang).
#   * BEST_EFFORT — pins when the node ledger has cores, else degrades to quota-only. Lower
#     scheduling constraint (lands anywhere), but a degrade under ledger pressure risks the hang.
# scikit-learn-14710 is REQUIRED (user call): verified it hangs >1800 s quota-only and passes
# (80/80) pinned to 2 cores, on a dev worker AND end-to-end on the prod cluster. Add an id here
# only with a reproduced hang + the fix confirmed under pinning.
_CPU_PINNED_INSTANCES: dict[str, CpuIsolation] = {
    "scikit-learn__scikit-learn-14710": CpuIsolation.REQUIRED,
}

# Infra-transient errors the driver may retry — and ONLY these (matched on
# type(e).__name__). Same set the harbor/pier oracle sweeps gate their RetryConfig to;
# here the drop-in surfaces them as exceptions out of run_instance's acquire step, so
# we retry at the driver level. A content outcome (resolved=False) is NEVER retried.
_INFRA_RETRY_EXCEPTIONS = frozenset({
    "CapacityExhausted",
    "PinCapacityExhausted",   # node-specific pin/ledger race — re-acquire may land on a good node
    "ControlPlaneLost",
    "NodeLost",
    "NodeCommandTimeout",
})


class _InfraFailure(Exception):
    """A cluster-infra failure that swebench's ``run_instance`` swallowed or wrapped.

    Upstream catches EvaluationError / BuildImageError / *every* exception inside
    ``run_instance`` and returns ``{"completed": false, "resolved": false}`` from its
    ``finally`` — and at container creation it further WRAPS the exception as
    ``BuildImageError`` — so an ``_INFRA_RETRY_EXCEPTIONS`` raised by the xrlenv cluster client
    (capacity / control-plane / node blip) never propagates to ``_run_with_retries`` and a
    transient infra failure would be misreported as an oracle/content failure (audit H1).

    We recover the signal from a STRUCTURED per-attempt record the xrlenv docker drop-in stamps
    at the mechanism boundary of EACH container operation — acquire, and post-acquire exec
    (batched + streaming) / archive / teardown — BEFORE the harness can obscure it (audit M8),
    not by parsing ``run_instance.log`` (logs are diagnostics, not retry-policy input). ``kind``
    is the real failure type; the adapter raises this only for an infra kind, so the infra-only
    retry absorbs it."""

    def __init__(self, kind: str, instance_id: str) -> None:
        super().__init__(f"{kind} (recorded at the op boundary for {instance_id})")
        self.kind = kind


def _take_infra_failure(client: Any, instance_id: str) -> str | None:
    """Pop the structured failure KIND the xrlenv drop-in recorded for this instance, or None.
    The drop-in records a cluster failure for ANY container operation (acquire / exec /
    streaming exec / archive / teardown) keyed by the rollout ``displayed_name`` (=
    ``instance_id``) BEFORE the harness wraps (BuildImageError) or swallows it — this adapter
    owns the policy of which kinds warrant a retry. A ``--local`` run uses a real docker client
    with no such API -> None (nothing recorded)."""
    take = getattr(getattr(client, "api", None), "take_infra_failure", None)
    kind = take(instance_id) if callable(take) else None
    return kind if isinstance(kind, str) else None


def _clear_infra_failure(client: Any, instance_id: str) -> None:
    """Drop any stale infra record for this instance before a fresh attempt, so evidence never
    crosses infra/content retries or reruns (audit M8)."""
    clear = getattr(getattr(client, "api", None), "clear_infra_failure", None)
    if callable(clear):
        clear(instance_id)


def _cache_shard() -> Path:
    # Resolve the root through the shared guard+resolver: the cache env/path were renamed
    # (audit: retired XRLENV_HARBOR_CACHE / .../xrlenv_harbor_cache -> unreliable results); it
    # HARD-REJECTS the legacy var/path (and errors if unset) before any cache read. Lazy import
    # matches plugin style.
    from xrlenv_plugins.benchmarks._benchmark_cache import benchmark_cache_root
    root = benchmark_cache_root()
    shard = Path(root).expanduser() / SHARD
    if not shard.is_dir():
        raise SystemExit(
            f"cache shard not found at {shard} — run:\n"
            f"  python xrlenv_plugins/benchmarks/swebench_verified/build_cache.py --stage all",
        )
    return shard


def _present_instances(shard: Path) -> list[str]:
    return sorted(d.name for d in shard.iterdir()
                  if not d.name.startswith(".")   # skip build_cache temp/stale siblings (M7)
                  and (d / "instance.json").is_file())


def _load_cached_instance(shard: Path, instance_id: str) -> dict[str, Any]:
    """The full upstream row from the cache (build_cache wrote instance.json)."""
    inst_dir = shard / instance_id
    anchor = inst_dir / "instance.json"
    # Direct ``--tasks`` entry bypasses the wrapper's list-green completeness gate, so re-check
    # containment here: a symlinked instance dir or a symlinked anchor would read out of the
    # shard (audit Low). lstat (is_symlink) is checked BEFORE is_file (which follows).
    if inst_dir.is_symlink() or anchor.is_symlink():
        raise SystemExit(
            f"refusing to read a symlinked cache entry for {instance_id!r} "
            f"({anchor}): an out-of-shard target is not trusted.",
        )
    if not anchor.is_file():
        raise SystemExit(
            f"instance {instance_id!r} not in the cache ({anchor} missing) — "
            f"populate it: build_cache.py --instances {instance_id}",
        )
    row: dict[str, Any] = json.loads(anchor.read_text())
    # The embedded instance_id feeds upstream + artifact paths downstream. A corrupt/custom
    # cache could carry a traversal id even though the DIR name is safe — require them to
    # agree (audit Low: cached-identity containment).
    embedded = str(row.get("instance_id", ""))
    if embedded != instance_id:
        raise SystemExit(
            f"cache corruption at {anchor}: embedded instance_id {embedded!r} != the dir "
            f"name {instance_id!r}; refusing to trust it.",
        )
    return row


def _safe_instance_id(iid: str) -> str:
    """Reject an id that isn't a BARE path component. Ids are joined onto the cache and
    artifact roots (``shard / iid``, ``artifact_root / … / iid``), so ``../../etc`` or an
    absolute path would escape the intended tree (audit Low: path containment)."""
    if not iid or iid in (".", "..") or iid != Path(iid).name:
        raise SystemExit(f"unsafe instance id {iid!r}: must be a bare name (no '/' or '..')")
    return iid


def _resolve_task_list(args: argparse.Namespace, shard: Path) -> list[str]:
    if args.all and args.tasks is not None:
        raise SystemExit("--all and --tasks are mutually exclusive.")
    if args.all:
        ids = _present_instances(shard)
        if not ids:
            raise SystemExit(f"cache shard {shard} is empty — run build_cache.py --stage all")
        return ids  # enumerated from the shard — already bare component names
    if args.tasks is not None:   # None=absent; '' is present-but-empty (audit M5)
        ids = [s.strip() for s in args.tasks.split(",") if s.strip()]
        if not ids:
            raise SystemExit(f"--tasks {args.tasks!r} selected no instances (audit M5)")
        return [_safe_instance_id(i) for i in ids]   # user input — containment-check
    return list(SMOKE_INSTANCES)


def _build_docker_client(local: bool) -> Any:
    """A real docker.DockerClient (--local) or the xrlenv docker-py drop-in.
    The cluster path is the one xrlenv-specific line."""
    if local:
        import docker
        LOGGER.info("local mode: docker.from_env()")
        return docker.from_env()
    if not os.environ.get("XRLENV_GRPC_HOST"):
        raise SystemExit(
            "cluster mode: XRLENV_GRPC_HOST not set. Pass --local for the local "
            "baseline, or export XRLENV_GRPC_HOST / XRLENV_GRPC_PORT / "
            "XRLENV_CONSUMER_TOKEN (usually from .env) first.",
        )
    LOGGER.info("cluster mode: xrlenv.from_env() — host=%s port=%s",
                os.environ.get("XRLENV_GRPC_HOST"), os.environ.get("XRLENV_GRPC_PORT", "50051"))
    return xrlenv.from_env()


def _run_one_instance(
    instance: dict[str, Any], client: Any, *, run_id: str, timeout: int,
    artifact_root: Path | None,
    namespace: str = SWEBENCH_NAMESPACE, instance_image_tag: str = "latest",
) -> dict[str, Any]:
    """One instance through upstream ``run_instance`` with the gold patch as the
    prediction. Image calls become no-ops in cluster mode (the node pulls on
    acquire). Returns upstream's result dict (carries ``resolved``).

    ``namespace`` / ``instance_image_tag`` pick the per-instance image the eval requests
    (default ``swebench/…:latest``). They MUST match whatever ``build_plan_gen`` warmed —
    a custom/mirrored plan (``--namespace``/``--tag``) otherwise warms an image the sweep
    never pulls, and the sweep falls back to the default ``swebench`` public image (audit
    M6)."""
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
    from swebench.harness.run_evaluation import make_test_spec, run_instance

    instance_id = instance["instance_id"]
    test_spec = make_test_spec(
        instance, namespace=namespace, instance_image_tag=instance_image_tag,
    )
    pred = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: "xrlenv-oracle",
        KEY_PREDICTION: instance["patch"],
    }
    instance_artifact_path: str | None = None
    if artifact_root is not None:
        instance_artifact_path = str(
            artifact_root / "logs" / "run_evaluation" / run_id / "xrlenv-oracle" / instance_id,
        )
    # Clear any stale infra record from a prior attempt so evidence never crosses retries
    # (audit M8). The drop-in stamps a fresh record at acquire if this attempt hits infra.
    _clear_infra_failure(client, instance_id)
    # Per-task cpuset pinning for the OpenMP/BLAS-oversubscription instances (see
    # _CPU_PINNED_INSTANCES for the level rationale). OFF (the default, every other instance)
    # is unchanged — no pinning, quota-only, exactly as before.
    cpu_isolation = _CPU_PINNED_INSTANCES.get(instance_id, CpuIsolation.OFF)
    with xrlenv.rollout_metadata(
        artifact_path=instance_artifact_path, displayed_name=instance_id,
        cpu_isolation=cpu_isolation,
    ):
        result = run_instance(
            test_spec=test_spec, pred=pred, rm_image=False, force_rebuild=False,
            client=client, run_id=run_id, timeout=timeout,
        )
    # Per-instance result at DEBUG — a 500-instance sweep would otherwise emit 500 INFO lines
    # into the benchmark log (the flood). run_benchmarks reads progress from disk (report.json),
    # not this log, and _summarise prints the failures + the final tally; so DEBUG keeps the log
    # clean (matches deep_swe) without losing any signal. Content-retry rounds still log at INFO.
    LOGGER.debug("[%s] resolved=%s", instance_id, (result or {}).get("resolved"))
    # An UNCOMPLETED run may be a swallowed/WRAPPED infra failure (audit H1/M8): upstream
    # returns completed=false both for a genuine content/build failure AND for an infra blip it
    # caught + wrapped (BuildImageError at acquire, or a swallowed post-acquire exec/archive
    # error). The drop-in recorded the failure KIND at the mechanism boundary (structured, not
    # scraped from logs); THIS adapter owns the policy — only an _INFRA_RETRY_EXCEPTIONS kind is
    # re-rolled by the infra layer. Any other kind (or none) is a content failure, returned
    # as-is. The record is popped either way so it never leaks into a later attempt.
    if not (result or {}).get("completed"):
        kind = _take_infra_failure(client, instance_id)
        if kind is not None and kind in _INFRA_RETRY_EXCEPTIONS:
            raise _InfraFailure(kind, instance_id)
    return result or {"instance_id": instance_id, "resolved": False, "completed": False}


def _run_with_retries(
    instance: dict[str, Any], client: Any, *, run_id: str, timeout: int,
    artifact_root: Path | None, retries: int,
    namespace: str = SWEBENCH_NAMESPACE, instance_image_tag: str = "latest",
) -> dict[str, Any]:
    """Infra-only retry: re-run on an _INFRA_RETRY_EXCEPTIONS type, up to ``retries``
    times. A content outcome (resolved=False) is returned as-is, never re-rolled."""
    iid = instance["instance_id"]
    attempt = 0
    while True:
        try:
            # Each infra retry gets a DISTINCT run_id so it doesn't overwrite the prior
            # attempt's run_instance.log / upstream report (audit M8) — the log is the
            # evidence we classify on, and a reused run_id would also hit upstream's report
            # cache. attempt 0 keeps the bare run_id.
            attempt_run_id = run_id if attempt == 0 else f"{run_id}-infra{attempt}"
            return _run_one_instance(
                instance, client, run_id=attempt_run_id, timeout=timeout,
                artifact_root=artifact_root,
                namespace=namespace, instance_image_tag=instance_image_tag,
            )
        except Exception as exc:
            # Infra-transient if the exception type is in the set (upstream re-raised it)
            # OR we recovered an infra signature from run_instance.log (upstream swallowed
            # it — audit H1). Content failures never reach here (they return completed=false
            # with no infra signature).
            name = exc.kind if isinstance(exc, _InfraFailure) else type(exc).__name__
            if (isinstance(exc, _InfraFailure) or name in _INFRA_RETRY_EXCEPTIONS) \
                    and attempt < retries:
                attempt += 1
                LOGGER.warning("[%s] infra-transient %s — retry %d/%d", iid, name, attempt, retries)
                continue
            LOGGER.exception("[%s] failed (%s)", iid, name)
            return {"instance_id": iid, "completed": False, "resolved": False,
                    "error": f"{name}: {exc}"}


def _drive(
    instances: dict[str, dict[str, Any]], client: Any, *, run_id: str, timeout: int,
    max_workers: int, artifact_root: Path | None, retries: int,
    namespace: str = SWEBENCH_NAMESPACE, instance_image_tag: str = "latest",
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    def _one(iid: str, inst: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return iid, _run_with_retries(
            inst, client, run_id=run_id, timeout=timeout,
            artifact_root=artifact_root, retries=retries,
            namespace=namespace, instance_image_tag=instance_image_tag,
        )

    if max_workers <= 1:
        for iid, inst in instances.items():
            _, results[iid] = _one(iid, inst)
        return results
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="swebench-oracle",
    ) as pool:
        for fut in concurrent.futures.as_completed(
            [pool.submit(_one, iid, inst) for iid, inst in instances.items()]
        ):
            iid, res = fut.result()
            results[iid] = res
    return results


def _summarise(results: dict[str, dict[str, Any]], expected: int) -> tuple[int, dict[str, Any]]:
    print("\n=== swebench-verified oracle sweep ===")
    resolved = 0
    failed: list[str] = []
    per_instance: list[dict[str, Any]] = []
    for iid, res in sorted(results.items()):
        ok = bool(res.get("resolved"))
        resolved += 1 if ok else 0
        row = {"instance_id": iid, "completed": res.get("completed"),
               "resolved": ok, "error": res.get("error")}
        per_instance.append(row)
        if not ok:
            failed.append(iid)
            # One concise line per FAILURE only — a green run stays quiet (no 500-instance
            # per-line dump that floods the sweep log); full per-instance detail lives in
            # summary.json + the per-instance run_evaluation artifacts.
            err = res.get("error")
            print(f"  [FAIL] {iid}" + (f"  ({err})" if err else ""))
    print(f"\n{resolved} / {expected} instance(s) resolved under the oracle policy.")
    if failed:
        print(f"failed (plumbing/content bug under the gold-patch oracle): {failed}")
    summary = {"expected": expected, "resolved": resolved, "failed": failed,
               "instances": per_instance}
    return (0 if resolved == expected and failed == [] else 1), summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # audit H9: NO prefix abbreviation (--des / --de / --d) may resolve to --dest —
        # the cache root must come only from XRLENV_BENCHMARK_CACHE, never a CLI override
        # smuggled past the wrapper's exact-form reject.
        allow_abbrev=False,
        prog="swebench-verified-oracle-sweep", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--tasks", "--instances", dest="tasks", default=None,
                     help="Comma-separated instance ids (default: the 8 smoke set). "
                          "--instances is an alias. Mutually exclusive with --all.")
    sel.add_argument("--all", action="store_true", help="Every cached instance.")
    sel.add_argument("--smoke", action="store_true", help="The 8 smoke instances (default).")
    p.add_argument("--local", action="store_true",
                   help="docker.from_env() against the local daemon (baseline).")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Driver concurrency (default 1 serial). swebench's harness is thread-safe.")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-instance test timeout (s), default 1800 (swebench default).")
    p.add_argument("--run-id", default="xrlenv-oracle-sweep",
                   help="swebench run id (controls log dir + container names).")
    p.add_argument("--namespace", default=SWEBENCH_NAMESPACE,
                   help="Per-instance image namespace the eval requests (default "
                        f"{SWEBENCH_NAMESPACE!r}, the public Docker Hub images). MUST match "
                        "the namespace build_plan_gen warmed (--namespace) — a mirrored plan "
                        "otherwise warms an image this sweep never pulls (audit M6).")
    p.add_argument("--instance-image-tag", default="latest",
                   help="Per-instance image tag the eval requests (default 'latest'). MUST "
                        "match build_plan_gen's --tag. Note 'latest' is mutable — pin a "
                        "digest-stable tag for a reproducible run.")
    p.add_argument("--jobs-dir", default="./tmp",
                   help="Artifact root; per-run tree lands under <jobs-dir>/<job-id>/.")
    p.add_argument("--job-id", default="swebench-verified-oracle",
                   help="Label under <jobs-dir>/.")
    p.add_argument("--retries", type=int, default=0, metavar="N",
                   help="Max trial retries for INFRA-transient errors ONLY (capacity / "
                        "control-plane / node blips). Content failures are never re-rolled.")
    p.add_argument("--content-retries", type=int, default=0, metavar="N",
                   help="Per-INSTANCE content-retry rounds: after a run, re-run instances "
                        "that came back UNRESOLVED up to N more times; an instance is solved "
                        "if ANY attempt resolves. Catches nondeterministic unresolved flakes "
                        "that --retries (infra-only) deliberately never re-rolls.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging()

    shard = _cache_shard()
    instance_ids = _resolve_task_list(args, shard)
    LOGGER.info("running %d instance(s) @ concurrency=%d: %s",
                len(instance_ids), args.max_workers, ", ".join(instance_ids))
    instances = {iid: _load_cached_instance(shard, iid) for iid in instance_ids}
    client = _build_docker_client(args.local)

    artifact_root = (Path(args.jobs_dir).expanduser() / args.job_id).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("artifacts: %s/", artifact_root)

    # swebench's run_instance writes logs/run_evaluation/... cwd-relative, so cd into
    # the artifact root for the loop (set once; restored after). Threads share cwd.
    prev_cwd = Path.cwd()
    try:
        os.chdir(artifact_root)
        try:
            # Content-retry loop: re-run ONLY the unresolved instances up to
            # --content-retries times (an instance is solved if ANY attempt resolves).
            # Catches nondeterministic unresolved flakes; --retries stays infra-only.
            best: dict[str, dict[str, Any]] = {}
            remaining = list(instances)
            cr = int(args.content_retries)
            for attempt in range(1 + cr):
                subset = {iid: instances[iid] for iid in remaining}
                # Each content-retry needs a DISTINCT upstream run_id (audit H2): upstream
                # keys report.json by (run_id, model, instance_id) and returns the cached
                # report immediately if it exists, so reusing args.run_id would make every
                # retry a no-op that re-reads the first attempt's unresolved report. The
                # XRLEnv job identity (job_id / artifact_root) stays; only the upstream
                # cache key rotates. attempt 0 keeps the bare run_id for the common case.
                attempt_run_id = args.run_id if attempt == 0 else f"{args.run_id}-retry{attempt}"
                results = _drive(
                    subset, client, run_id=attempt_run_id, timeout=args.timeout,
                    max_workers=args.max_workers, artifact_root=artifact_root,
                    retries=args.retries, namespace=args.namespace,
                    instance_image_tag=args.instance_image_tag,
                )
                for iid, res in results.items():
                    if iid not in best or (not best[iid].get("resolved") and res.get("resolved")):
                        best[iid] = res
                remaining = [iid for iid in remaining if not (best.get(iid) or {}).get("resolved")]
                if not remaining:
                    break
                if attempt < cr:
                    LOGGER.info("content-retry %d/%d: re-running %d unresolved: %s",
                                attempt + 1, cr, len(remaining), ", ".join(remaining))
        finally:
            os.chdir(prev_cwd)
        # Fold the per-attempt run_evaluation folders (<run_id>-retryN / -infraN) into
        # the main <run_id> folder so the operator sees ONE eval-log set per instance,
        # not stray retry folders. summary.json (below) is already the merged truth.
        from xrlenv_plugins.benchmarks._sweep_retry import consolidate_swebench_eval_dirs
        consolidate_swebench_eval_dirs(artifact_root, args.run_id)
        exit_code, summary = _summarise(best, len(instances))
        (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"artifacts + summary at {artifact_root}")
        return exit_code
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
