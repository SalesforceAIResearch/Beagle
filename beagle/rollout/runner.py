"""The unified ``Runner`` — roll out any agent over any (mix of) benchmark(s) and
write a run directory (``run.json`` + the benchmarks' native artifact trees).

Defining property: it dispatches each task to that task's **own benchmark harness**
(resolved from ``Task.benchmark``) and lets the harness drive the rollout natively — so
a mixed dataset "just works". The harness owns concurrency (harbor.Job fans out
internally); the Runner **groups by benchmark and batch-rolls each group**, grades,
reduces to a run record, and handles resume. It does NOT spawn a thread pool.

What the Runner owns: run identity, the run dir + ``run.json`` contract, resume/skip,
grading + reduction. What it does NOT own: rollout mechanics or eval logic — those live
in the benchmark harness / grader (:mod:`beagle.benchmarks`).
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from beagle.rollout.resume import plan_resume
from beagle.types import TaskResult

if TYPE_CHECKING:
    from beagle.agents.core.base import Runnable
    from beagle.config import RunConfig
    from beagle.data.dataset import TaskDataset


@contextlib.contextmanager
def _tag_group(run_id: str) -> Iterator[None]:
    """Tag every container acquired in this scope with ``xrlenv.group_id=run_id``.

    That label is what a Ctrl-C teardown (``terminate_raw_group(run_id)``) matches to actively kill
    THIS run's container cohort, instead of leaving it for xrlenv's ~120 s raw-liveness reaper. The
    harbor/pier path acquires through the xrlenv drop-in and reads this contextvar (there is no
    runtime object to stamp it on that path); the docker runtime additionally stamps it in
    ``acquire()``.

    Set HERE — the single rollout seam — so every entry point (``beagle evaluate`` / the SDK /
    evolve) inherits it uniformly. Previously only the darwinx (evolve) caller wrapped the call, so
    ``beagle evaluate`` shipped UNTAGGED containers on the harbor/pier path (``group_id`` unset →
    ``terminate_raw_group`` a no-op on Ctrl-C). Best-effort: a no-op if xrlenv isn't importable
    (pure-local / hermetic tests), since group tagging is a cluster concern.
    """
    try:
        from xrlenv import rollout_metadata
    except Exception:  # noqa: BLE001 — xrlenv not importable ⇒ nothing to tag
        yield
        return
    with rollout_metadata(group_id=run_id):
        yield


@dataclass
class RunResult:
    """Aggregate outcome of a rollout batch."""

    run_id: str
    results: list[TaskResult] = field(default_factory=list)
    #: Root under which each benchmark wrote its native artifacts (also holds run.json).
    artifact_dir: Path | None = None
    totals: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """The run's fitness scalar (resolved-rate over all tasks in the record)."""
        return float(self.metrics.get("score", 0.0))


def _check_config_drift(run_dir: Path, config_hash: str, *, resume: bool,
                        force: bool = False) -> str | None:
    """Refuse to resume across a config change — the drift guard, keyed on the prior
    ``run.json`` ``config_hash``. (Per-task done-state is read from each harness's native
    tree via ``harness.completed`` at rollout time, not from a house ledger.)

    Returns the prior (drifted) hash when ``force`` overrides a mismatch — the caller records it
    in ``run.json`` as ``config_hash_drift`` — else ``None``. ``force`` (``--force-resume``) is the
    escape hatch for "I tweaked a knob, keep the finished tasks"; it downgrades the hard error to a
    warning."""
    run_json = run_dir / "run.json"
    if not resume or not run_json.exists():
        return None
    try:
        prior_hash = json.loads(run_json.read_text()).get("config_hash")
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"cannot resume {run_dir.name}: its run.json is unreadable ({e}). "
            f"Remove it and start a fresh run."
        ) from None
    if prior_hash is not None and prior_hash != config_hash:
        if not force:
            raise RuntimeError(
                f"cannot resume {run_dir.name}: config changed (run.json {prior_hash} != {config_hash}). "
                f"Start a fresh run, roll back the config, or pass --force-resume to keep finished tasks."
            )
        print(f"[runner] ⚠ --force-resume: config drifted ({prior_hash} != {config_hash}); resuming "
              f"anyway — {run_dir.name}/run.json will mix both configs (recorded as config_hash_drift).")
        return prior_hash
    return None


def _job_dir(rows: list[dict[str, Any]], run_dir: Path) -> str | None:
    """The benchmark's native subtree, relative to the run dir (the pointer into harbor's
    ``<benchmark>/`` tree). Derived from a trial dir so it reflects whatever the harness
    actually wrote — the Runner imposes nothing."""
    for r in rows:
        td = r.get("trial_dir")
        if td:
            parent = Path(td).parent
            try:
                return str(parent.relative_to(run_dir))
            except ValueError:
                return parent.name
    return None


class Runner:
    """Drives rollouts through each task's native benchmark harness.

    ``runtime`` is the container substrate (``None`` on the harbor path — harbor owns the
    trial environment). ``parallelism`` is forwarded to the harness. ``results_root`` is
    where run directories are created (default ``results/``).
    """

    def __init__(
        self,
        runtime: Any = None,
        *,
        parallelism: int = 1,
        eval_parallelism: int | None = None,
        results_root: str | Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.parallelism = parallelism
        # Patch-eval fan-out for a two-phase grader (SWE-bench); None → reuse ``parallelism``.
        self.eval_parallelism = eval_parallelism
        self.results_root = Path(results_root) if results_root else Path("results")

    def run(
        self,
        agent: Runnable,
        dataset: TaskDataset,
        *,
        config: RunConfig,
        run_id: str | None = None,
        run_dir: str | Path | None = None,
        resume: bool = False,
        retry_errors: bool = False,
        retry_unresolved: bool = False,
        only_task_ids: set[str] | None = None,
        force_resume: bool = False,
        config_path: str | Path | None = None,
        campaign_id: str | None = None,
    ) -> RunResult:
        """Roll out ``agent`` over ``dataset`` → a run dir with ``run.json`` + native artifacts.

        ``run_dir`` sets the output dir explicitly (verbatim); otherwise it defaults to
        ``results_root/<run_id>``. ``run_id`` names the leaf of that default and is
        recorded in ``run.json`` regardless. The Runner stays harness-agnostic: it writes
        ``run.json`` here and hands each benchmark's harness this ``run_dir`` — the harness
        then writes its OWN native subtree under it (harbor → ``<benchmark>/<trial>/``;
        docker/native-runner → their upstream layout). No house structure is imposed.
        """
        from beagle import benchmarks
        from beagle.rollout import run_record as rr
        from beagle.rollout.retry import better_attempt
        from beagle.rollout.run_id import build_run_id, compute_config_hash

        retry = config.retry            # infra + content retry policy for this run
        config_hash = compute_config_hash(config.model_dump(mode="json"))
        run_id = run_id or build_run_id(config, config_hash)
        run_dir = Path(run_dir) if run_dir else self.results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # The three resume/retry flags are independent (each re-runs its own category); ANY of them
        # means we're resuming into an existing run_dir — read the prior tree + gate on config drift.
        resuming = resume or retry_errors or retry_unresolved
        drift = _check_config_drift(run_dir, config_hash, resume=resuming, force=force_resume)

        # Group tasks by their benchmark; each group batch-rolls through its native harness.
        groups: dict[str, list] = {}
        for task, ctx in dataset:
            groups.setdefault(task.benchmark, []).append((task, ctx))

        ts_start = datetime.now(timezone.utc)
        t0 = time.monotonic()

        def _run_group(name: str) -> tuple[str, list[TaskResult], list[dict[str, Any]], dict[str, Any]]:
            """Roll out + grade one benchmark's tasks. Self-contained (writes only its own
            ``run_dir/<benchmark>/`` subtree), so groups are safe to run concurrently."""
            g0 = time.monotonic()
            items = groups[name]
            bench = benchmarks.get(name)
            # A benchmark's ``options`` block can override the harbor/pier cluster Environment
            # (env_import_path) — e.g. a local, non-cluster tb2 run. Pass the kwarg ONLY when set,
            # so a benchmark with the plain ``harness(self)`` signature keeps working.
            _eip = next((b.options.get("env_import_path")
                         for b in config.all_benchmarks() if b.name == name), None)
            harness = bench.harness(env_import_path=_eip) if _eip else bench.harness()

            # Resume: ask the harness what's already done (read from ITS native tree, not a house
            # ledger), then let ``plan_resume`` (shared with ``--dry-run``) decide what re-runs — each
            # flag re-runs its own category (--resume→missing, --retry-errors→error, --retry-unresolved
            # →error+genuine-fail); a resolved task is always kept. A missing task that no flag covers
            # is neither re-run nor kept (left absent). Fails loud if --retry-unresolved finds a trial
            # with no gradeable signal (no reward, no error) — 'unresolved' can't be told from 'ungraded'.
            prior = harness.completed(items, run_dir=run_dir) if resuming else []
            plan = plan_resume(items, prior, resume=resume, retry_errors=retry_errors,
                               retry_unresolved=retry_unresolved, only_task_ids=only_task_ids,
                               label=name)
            done = plan.keep
            rerun = set(plan.rerun_ids)
            to_run = [(t, c) for t, c in items if t.task_id in rerun]

            # Roll out with the retry policy: infra-retry rides `retry` INTO the harness (harbor's
            # RetryConfig / the per-task loop); CONTENT-retry is HERE — re-run still-unresolved
            # tasks up to ``retry.content`` rounds, keeping the best attempt per task (a task is
            # solved if ANY round resolves). Round 0 is the base job; later rounds write a
            # ``<benchmark>-retry<N>`` sibling dir (harbor's own resume won't skip them).
            best: dict[str, TaskResult] = {}
            remaining = to_run
            for attempt in range(1 + (retry.content if retry else 0)):
                if not remaining:
                    break
                produced = harness.rollout(
                    agent, remaining, runtime=self.runtime, run_dir=run_dir,
                    parallelism=self.parallelism, retry=retry, attempt=attempt,
                    resuming=resuming,
                )
                for r in produced:
                    if better_attempt(best.get(r.task_id), r):
                        best[r.task_id] = r
                remaining = [(t, c) for t, c in to_run
                             if not (best.get(t.task_id) and best[t.task_id].resolved)]
            group_results = list(done.values()) + list(best.values())
            report = bench.grader().grade(
                group_results, runtime=self.runtime, run_dir=run_dir,
                parallelism=self.eval_parallelism or self.parallelism)   # patch-eval fan-out (SWE-bench)
            rows = [rr.per_task_row(r, benchmark=name) for r in group_results]
            summary = rr.benchmark_summary(
                rows, score=report.score, job_dir=_job_dir(rows, run_dir),
                wall_time_sec=time.monotonic() - g0,
            )
            return name, group_results, rows, summary

        # Benchmark groups are independent, so a MIX of benchmarks runs them CONCURRENTLY
        # (bounded by ``parallelism``) — otherwise a 2-benchmark mix serializes (all of tb2.1,
        # then all of swe). A single group or ``parallelism<=1`` keeps the sequential path.
        # (``parallelism`` is also the intra-benchmark task fan-out inside ``harness.rollout``.)
        # Tag each group's containers with xrlenv.group_id=run_id (the label a Ctrl-C teardown
        # targets). Wrap per group so the tag is set INSIDE the thread that rolls out — it must
        # survive the ThreadPoolExecutor below, and this one seam covers every entry point.
        def _tagged_run_group(name: str):
            with _tag_group(run_id):
                return _run_group(name)

        names = list(groups)
        if self.parallelism > 1 and len(names) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(self.parallelism, len(names))) as ex:
                completed = list(ex.map(_tagged_run_group, names))
        else:
            completed = [_tagged_run_group(n) for n in names]

        # Assemble in the groups' insertion order so run.json is stable regardless of finish order.
        by_name = {name: (gr, rows, summary) for name, gr, rows, summary in completed}
        results: list[TaskResult] = []
        rows_by_bench: dict[str, list[dict[str, Any]]] = {}
        benchmarks_out: dict[str, Any] = {}
        for name in names:
            gr, rows, summary = by_name[name]
            results.extend(gr)
            rows_by_bench[name] = rows
            benchmarks_out[name] = summary
        wall = time.monotonic() - t0
        ts_end = datetime.now(timezone.utc)

        all_rows = [r for brows in rows_by_bench.values() for r in brows]
        totals = rr.compute_totals(all_rows, num_benchmarks=len(rows_by_bench), wall_time_sec=wall)
        record = rr.assemble_run_record(
            run_id=run_id, config=config, benchmarks=benchmarks_out, totals=totals,
            config_hash=config_hash, config_path=config_path, campaign_id=campaign_id,
            environment=rr.capture_environment(),
            timestamp_start=ts_start.isoformat(), timestamp_end=ts_end.isoformat(),
            config_hash_drift=drift,
        )
        rr.write_run_json(run_dir, record)
        # metrics: a small in-memory summary for callers (RunResult.score = resolved_rate).
        metrics = {"score": totals["resolved_rate"], "num_resolved": totals["num_resolved"],
                   "num_tasks": totals["num_tasks"],
                   "per_benchmark": {n: b["score"] for n, b in benchmarks_out.items()}}
        return RunResult(run_id=run_id, results=results, artifact_dir=run_dir, totals=totals, metrics=metrics)


__all__ = ["RunResult", "Runner"]
