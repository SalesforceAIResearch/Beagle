"""DarwinX eval adapter — the DarwinX-specific glue over the general :func:`beagle.eval.evaluate`.

A vendored self-evolution algorithm evaluates a candidate by shelling ``python -m runner.run
<cfg> --results-root <dir>`` and then reading ``<results-root>/runs/<run_id>/run.json`` (its
``per_task_results[]`` rows). This module is the thin, DarwinX-specific wrapper: it (1) translates
the algorithm's ``Config``-shaped yaml into beagle's :class:`RunConfig`, (2) delegates the actual
evaluation to :func:`beagle.eval.evaluate` (harbor.Job, native trees, run_id — the general seam),
and (3) re-serializes the result into the run.json shape the algorithm reads (approach (a): the
algorithm's parser is untouched; beagle's clean summary is preserved beside it as
``run.beagle.json``). The general "eval an agent on a benchmark" logic lives in
:mod:`beagle.eval`, not here.

CLI-compatible with ``python -m runner.run`` so it can shadow it:

    python -m beagle.algorithms.darwinx.eval <config> --results-root <dir> \
        [--run-id ID] [--campaign-id X] [--include-task-name T ...]

See ``notes/darwinx-dropin-contract.md`` §3.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Callable

from beagle.types import TaskResult

# The harbor trial-name suffix (``<task>__<hash>``) the Runner carries on TaskResult.task_id.
# The algorithm keys per-task rows on the *base* task id, so we strip it.
_TRIAL_SUFFIX_RE = re.compile(r"__[A-Za-z0-9]+$")
# pass@k: ``select_and_sample`` expands one task into ``<task>__s0``/``__s1``/… — strip that too so
# a task's k samples collapse to k rows under one base id (the driver groups by base id for avg@k).
_SAMPLE_SUFFIX_RE = re.compile(r"__s\d+$")


def _base_task_id(task_id: str) -> str:
    """Strip harbor's trial suffix then the pass@k sample suffix → the base task id."""
    return _SAMPLE_SUFFIX_RE.sub("", _TRIAL_SUFFIX_RE.sub("", task_id))


def translate_config(raw: dict[str, Any]):
    """A ``runner.Config``-shaped dict (the algorithm's eval config format) → beagle :class:`RunConfig`.

    The two are a near-verbatim port (see ``notes/study/gap-config.md``); only the runtime
    block needs massaging. Unknown fields still hit ``RunConfig``'s ``extra='forbid'`` and
    fail loud — that's the signal to extend this translator, not to loosen the schema.
    """
    from beagle.config import RunConfig

    d = copy.deepcopy(raw)
    rt = d.get("runtime")
    if isinstance(rt, dict):
        if "consumer_token" in rt and "token" not in rt:
            rt["token"] = rt.pop("consumer_token")       # renamed field
        rt.pop("grpc_secure", None)                       # unexpressible TLS toggle (contract #3)
    bench = d.get("benchmark")
    if isinstance(bench, dict):
        # The algorithm's `benchmark.dataset` is its OWN dataset reference (a benchmark-suite relative
        # path, e.g. "benchmarks/terminal_bench/vendor"); beagle's loaders resolve tasks
        # themselves — harbor-family from `$XRLENV_BENCHMARK_CACHE` — and `BenchmarkSpec.dataset`
        # means a task-source PATH override, a different thing. Drop it so we don't mis-glob an
        # empty/foreign dir instead of the cache. (task_ids/name/etc. are the same in both.)
        bench.pop("dataset", None)
    return RunConfig.from_dict(d)


def _effective_reward(row: dict[str, Any]) -> float:
    """Mirror the algorithm's reward fallback: numeric reward, else 1.0 if resolved."""
    r = row.get("reward")
    return float(r) if r is not None else (1.0 if row.get("resolved") else 0.0)


def to_darwinx_run_json(results: list[TaskResult]) -> dict[str, Any]:
    """Serialize the Runner's TaskResults into the run.json shape the algorithm reads.

    Contract (``codingbench_eval.parse_run_json`` / ``_build_eval_result``): ``per_task_results``
    rows ``{task_id, resolved, reward|None, error|None, tokens}``, a job-level ``errors`` list
    ``{task_id, kind, message, traceback}`` for errored tasks (its transient/infra tolerance
    substring-matches these — so ``error`` must be a full readable string, contract #2), and
    ``totals.{num_tasks, num_tasks_resolved, num_tasks_errored}``. Task ids are the base name
    (harbor's ``__<hash>`` trial suffix + the pass@k ``__s<n>`` sample suffix stripped), so a
    task's k samples appear as k rows under one base id — the driver's avg@k / best-of merge
    groups them by that id. ``totals`` is over DISTINCT base tasks (pass@k: resolved if any sample
    resolved), since the driver recomputes from the rows anyway.
    """
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for r in results:
        tid = _base_task_id(r.task_id)
        rows.append({
            "task_id": tid,
            "resolved": bool(r.resolved),
            "reward": r.reward,
            "error": r.error,
            "tokens": dict(r.tokens),
        })
        if r.error:
            errors.append({"task_id": tid, "kind": "error", "message": r.error, "traceback": ""})
    by_base: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_base.setdefault(row["task_id"], []).append(row)
    return {
        "per_task_results": rows,
        "errors": errors,
        "totals": {
            "num_tasks": len(by_base),
            "num_tasks_resolved": sum(1 for rs in by_base.values()
                                      if any(_effective_reward(x) >= 1.0 for x in rs)),
            "num_tasks_errored": sum(1 for rs in by_base.values() if all(x.get("error") for x in rs)),
        },
    }


def write_darwinx_run_json(run_dir: Path, results: list[TaskResult]) -> Path:
    """Write ``run_dir/run.json`` in the algorithm's shape, preserving beagle's own clean
    summary as ``run.beagle.json``. Returns the run.json path."""
    import json

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    clean, preserved = run_dir / "run.json", run_dir / "run.beagle.json"
    # Preserve beagle's clean summary once; a re-run must not clobber it with the (already
    # reshaped) compat run.json from a prior pass — keep the FIRST clean summary.
    if clean.exists() and not preserved.exists():
        clean.replace(preserved)
    path = run_dir / "run.json"
    path.write_text(json.dumps(to_darwinx_run_json(results), indent=2, default=str))
    return path


def run_eval(
    config_path: str | Path,
    *,
    results_root: str | Path,
    run_id: str | None = None,
    include_task_name: list[str] | None = None,
    campaign_id: str | None = None,
    _evaluate: Callable[..., Any] | None = None,
) -> Path:
    """Translate a candidate config, evaluate it via :func:`beagle.eval.evaluate`, and emit
    the algorithm-shaped ``run.json`` at ``<results-root>/runs/<run_id>/`` (where the algorithm
    snapshots + reads). The ``_evaluate`` hook exists only for hermetic tests (no cluster)."""
    import yaml

    from xrlenv import rollout_metadata

    from beagle.eval import evaluate
    from beagle.rollout.interrupt import stop_group_on_sigint
    from beagle.rollout.run_id import build_run_id, compute_config_hash

    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    if include_task_name:  # Harbor's --include-task-name selects the task subset
        raw.setdefault("benchmark", {})["task_ids"] = list(include_task_name)
    cfg = translate_config(raw)
    # pass@k (num_samples>1): the eval expands into <task>__s0/…, and to_darwinx_run_json collapses
    # them to k rows under the base id for the driver's avg@k merge — no guard needed.

    run_id = run_id or build_run_id(cfg, compute_config_hash(cfg.model_dump(mode="json")))
    run_dir = Path(results_root) / "runs" / run_id     # the layout the algorithm discovers

    do_eval = _evaluate or evaluate
    # Tag every container this candidate acquires with ``xrlenv.group_id=run_id`` — harbor/swebench
    # acquire through the xrlenv drop-in and inherit the tag via the contextvar (no runtime object
    # on this path) — and on Ctrl-C actively terminate that group so containers are freed now rather
    # than lingering ~120 s for the raw-liveness reaper. A terminal Ctrl-C reaches the whole process
    # group, so each candidate subprocess tears down its own run.
    with stop_group_on_sigint(run_id), rollout_metadata(group_id=run_id):
        result = do_eval(cfg, run_id=run_id, run_dir=run_dir, campaign_id=campaign_id)
    write_darwinx_run_json(run_dir, list(result.results))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="beagle.algorithms.darwinx.eval",
        description="Evaluate a candidate config through beagle's eval; emit an "
                    "algorithm-shaped run.json (runner.run-compatible).",
    )
    p.add_argument("config", help="path to the eval config (RunConfig-shaped yaml)")
    p.add_argument("--results-root", required=True, help="root under which runs/<run_id>/ is written")
    p.add_argument("--run-id", default=None)
    p.add_argument("--campaign-id", default=None)
    p.add_argument("--include-task-name", action="append", default=None,
                   help="restrict to this task (repeatable) — maps to benchmark.task_ids")
    args = p.parse_args(argv)

    run_dir = run_eval(
        args.config, results_root=args.results_root, run_id=args.run_id,
        include_task_name=args.include_task_name, campaign_id=args.campaign_id,
    )
    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
