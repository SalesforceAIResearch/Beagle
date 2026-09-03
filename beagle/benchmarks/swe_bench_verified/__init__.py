"""SWE-bench Verified (override case — not harbor-cache, patch-graded).

Tasks come from HuggingFace (not the harbor cache), the agent produces a patch
(``DockerHarness``), and grading runs the upstream swebench evaluator on that patch
via ``xrlenv.from_env()`` — a :class:`PatchEvalGrader`. Everything benchmark-specific
lives here; the deps (``datasets``, ``swebench``) are lazy-imported behind the
``beagle[swe-bench]`` extra so they don't weigh on core.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

from beagle.benchmarks.base import (
    Benchmark,
    BenchmarkHarness,
    BenchmarkSpec,
    GradeReport,
    Grader,
    TaskSource,
    write_result_json,
)
from beagle.benchmarks.grader import PatchEvalGrader
from beagle.benchmarks.harness import DockerHarness
from beagle.benchmarks.registry import register
from beagle.benchmarks.source import TaskItem, select_and_sample
from beagle.rollout.runtime import ContainerRuntime
from beagle.types import Task, TaskContext, TaskResult

LOGGER = logging.getLogger(__name__)

_BENCH = "swe-bench-verified"

#: HuggingFace dataset the upstream harness loads test specs from.
_DATASET = "SWE-bench/SWE-Bench_Verified"
_SPLIT = "test"

#: Label the single-entry prediction under (any stable string works; the
#: report dir is namespaced by this via ``model_name_or_path``).
_MODEL_NAME = "beagle"


class _SweBenchSource(TaskSource):
    """Enumerate SWE-bench Verified from HuggingFace and derive per-instance images."""

    def tasks(self, spec: BenchmarkSpec) -> Iterator[TaskItem]:
        from datasets import load_dataset  # lazy: beagle[swe-bench]

        ds = load_dataset(spec.dataset or "SWE-bench/SWE-Bench_Verified", split=spec.split or "test")
        items = [self._to_item(row) for row in ds]
        yield from select_and_sample(items, spec)

    def _to_item(self, row: dict) -> TaskItem:
        iid = row["instance_id"]
        # Deterministic upstream image formula (shared with mini-swe-agent).
        image = f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"
        task = Task(
            task_id=iid,
            problem_statement=row["problem_statement"],
            repo_url=row.get("repo", ""),
            base_commit=row.get("base_commit", ""),
            benchmark=_BENCH,
            extras={"hints": row.get("hints_text", "")},
        )
        ctx = TaskContext(
            image=image,
            repo_path="/testbed",
            shell_preamble="source /opt/miniconda3/bin/activate testbed",
            benchmark_name=_BENCH,
            # This benchmark has no engine deadline (docker drop-in) and the dataset ships no
            # per-instance agent budget — upstream's 1800 s is the EVALUATION timeout, not the
            # agent's. So there is genuinely nothing to inherit, and the run config's
            # ``agent.timeout`` is the bound. Left None deliberately rather than inventing a
            # per-instance number here: see ``resolve_agent_timeout``.
            agent_timeout_s=None,
        )
        return task, ctx


class SweBenchGrader(PatchEvalGrader):
    """Score patches with the upstream swebench evaluator — in ONE native batch.

    :meth:`grade` assembles a single ``predictions.json`` from all non-empty patches and calls
    ``swebench.harness.run_evaluation.main`` **once** with ``max_workers=parallelism`` (swebench's
    own cross-instance concurrency — not N serial calls), then distributes its native per-instance
    outputs (``report.json`` / ``run_instance.log`` / ``test_output.txt`` / ``eval.sh`` /
    ``patch.diff``) into each task's ``<run_dir>/<benchmark>/<task_id>/`` dir and reads
    ``resolved`` (1.0/0.0). Empty-patch / errored tasks score 0 without spending a container.
    Artifacts stay byte-compatible with upstream (``predictions.json`` + the
    ``logs/run_evaluation/…`` tree). :meth:`evaluate_patch` keeps a single-instance path for the
    generic :class:`PatchEvalGrader` contract.

    On the xrlenv cluster the evaluator's containers route through a ``docker.from_env`` drop-in
    installed *before* swebench imports docker (see :meth:`_install_xrlenv_drop_in`). ``runtime`` is
    accepted to satisfy the contract, but the evaluator drives its own containers.
    """

    def grade(
        self, results: list[TaskResult], *, runtime: ContainerRuntime, run_dir: Path,
        parallelism: int = 1,
    ) -> GradeReport:
        bench = (results[0].benchmark if results else "") or _BENCH
        bench_dir = Path(run_dir) / bench
        gradeable_ids = {r.task_id for r in results if r.error is None and (r.patch or "").strip()}
        gradeable = [r for r in results if r.task_id in gradeable_ids]
        # A RESUMED already-graded task comes back with its reward set but no patch (result.json doesn't
        # store the patch), so it isn't in gradeable_ids — PRESERVE its reward instead of re-scoring it
        # to 0, and note it's excluded from the swebench eval below, so we never re-run its container.
        # A fresh / empty-patch task (reward is None) defaults to 0.0.
        per_task: dict[str, float] = {
            r.task_id: (float(r.reward) if r.reward is not None else 0.0) for r in results}
        for r in results:
            if r.error is None and r.task_id not in gradeable_ids and r.reward is None:
                # Empty patch = no gradeable attempt. Score 0, but flag a retryable NoAttempt error —
                # otherwise a bare reward=0 reads as a genuine capability failure and hides from
                # --retry-errors (the case that used to need a manual `rm result.json`).
                r.error = "NoAttempt: empty patch (agent produced no diff)"
                LOGGER.info("swe-bench %s: empty patch — flagged NoAttempt (retryable), scoring 0.0",
                            r.task_id)

        if gradeable:
            bench_dir.mkdir(parents=True, exist_ok=True)
            # ONE predictions.json (dict keyed by instance_id — swebench's native batch schema).
            predictions = {r.task_id: {"instance_id": r.task_id, "model_name_or_path": _MODEL_NAME,
                                       "model_patch": r.patch} for r in gradeable}
            predictions_path = bench_dir / "predictions.json"
            predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
            run_id = f"beagle-{uuid.uuid4().hex[:8]}"
            # The rollout progress bar is done; mark the (silent-until-now) grading phase so the run
            # isn't a blank terminal while swebench evaluates the batch.
            print(f"[{bench}] grading {len(gradeable)} patch(es) with swebench "
                  f"(parallelism {max(1, parallelism)})…", file=sys.stderr, flush=True)
            self._invoke_run_evaluation(
                predictions_path, sorted(gradeable_ids), out_dir=bench_dir, run_id=run_id,
                max_workers=max(1, parallelism))
            logs_model = bench_dir / "logs" / "run_evaluation" / run_id / _MODEL_NAME.replace("/", "__")
            for r in gradeable:
                per_task[r.task_id] = self._collect_instance(logs_model, bench_dir, r)
            n_resolved = sum(1 for r in gradeable if per_task[r.task_id] >= 1.0)
            print(f"[{bench}] graded {len(gradeable)} patch(es): {n_resolved} resolved.",
                  file=sys.stderr, flush=True)

        for r in results:
            r.reward = per_task[r.task_id]
            r.resolved = per_task[r.task_id] >= 1.0
            write_result_json(r)
        report = GradeReport(
            num_tasks=len(results),
            num_resolved=sum(1 for r in results if r.resolved),
            score=(sum(per_task.values()) / len(per_task)) if per_task else 0.0,
            eval_dir=bench_dir, per_task=per_task)
        # Canonical, predictably-named benchmark-level summary. swebench's own aggregate is
        # ``<model>.<run_id>.json`` — a per-run-id name that's awkward to find + parse — so we ALSO
        # write the fixed ``<benchmark>/result.json`` that a harbor job leaves, giving downstream tooling
        # ONE stable path across benchmark families. swebench's native report + predictions.json stay
        # for upstream byte-compat.
        bench_dir.mkdir(parents=True, exist_ok=True)
        (bench_dir / "result.json").write_text(json.dumps({
            "benchmark": bench,
            "num_tasks": report.num_tasks,
            "num_resolved": report.num_resolved,
            "num_errored": sum(1 for r in results if r.error is not None),
            "score": report.score,
            "per_task": {r.task_id: {"resolved": r.resolved, "reward": r.reward} for r in results},
        }, indent=2), encoding="utf-8")
        return report

    def evaluate_patch(self, task_id: str, patch: str, *, runtime: ContainerRuntime) -> float:
        # Single-instance path (the PatchEvalGrader contract); grade() uses the batch above. Empty
        # patch → 0 without spinning up the evaluator (which would just report an empty patch).
        if not patch or not patch.strip():
            LOGGER.info("swe-bench %s: empty patch, scoring 0.0 without evaluation", task_id)
            return 0.0
        run_id = f"beagle-{task_id}-{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory(prefix="beagle-swebench-") as tmp:
            out_dir = Path(tmp)
            predictions_path = out_dir / "predictions.json"
            predictions_path.write_text(json.dumps(
                {task_id: {"instance_id": task_id, "model_name_or_path": _MODEL_NAME,
                           "model_patch": patch}}), encoding="utf-8")
            self._invoke_run_evaluation(predictions_path, [task_id], out_dir=out_dir,
                                        run_id=run_id, max_workers=1)
            report = (out_dir / "logs" / "run_evaluation" / run_id
                      / _MODEL_NAME.replace("/", "__") / task_id / "report.json")
            return self._read_report(report, task_id)

    def _invoke_run_evaluation(
        self, predictions_path: Path, instance_ids: list[str], *, out_dir: Path, run_id: str,
        max_workers: int,
    ) -> None:
        """Run swebench's evaluator with cwd=``out_dir`` so its native ``logs/run_evaluation/…``
        tree lands under ``out_dir``. On the cluster (``XRLENV_GRPC_HOST`` set) the docker-py
        drop-in is installed first — it MUST precede swebench's docker import."""
        if os.environ.get("XRLENV_GRPC_HOST"):
            self._install_xrlenv_drop_in()
        # Lazy import — AFTER the drop-in. swebench is behind the ``beagle[swe-bench]`` extra.
        try:
            from swebench.harness.run_evaluation import main as run_eval  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - exercised via the error path
            raise RuntimeError(
                "swe-bench grading requires the upstream 'swebench' package. "
                "Install it with the extra: `pip install 'beagle[swe-bench]'`."
            ) from e
        # Resolve to ABSOLUTE before the chdir — run_dir may be relative (``./tmp/…``), and the
        # chdir below would otherwise break a relative predictions path.
        predictions_path, out_dir = Path(predictions_path).resolve(), Path(out_dir).resolve()
        prev_cwd = Path.cwd()
        os.chdir(out_dir)
        try:
            run_eval(
                dataset_name=_DATASET, split=_SPLIT, instance_ids=list(instance_ids),
                predictions_path=str(predictions_path), max_workers=max_workers,
                force_rebuild=False, cache_level="env", clean=False, open_file_limit=4096,
                run_id=run_id, timeout=1800,
                # Pull the pre-published eval images (fast; the only path viable under the cluster
                # drop-in, which doesn't dispatch ``docker build``).
                namespace="swebench", rewrite_reports=False, modal=False, report_dir=".")
        finally:
            os.chdir(prev_cwd)

    @staticmethod
    def _collect_instance(logs_model: Path, bench_dir: Path, r: TaskResult) -> float:
        """Copy swebench's native per-instance files into the task's dir (co-located with the
        rollout artifacts) and read ``report.json`` → 1.0/0.0. ``logs_model`` =
        ``<bench_dir>/logs/run_evaluation/<run_id>/<model>``."""
        src = logs_model / r.task_id
        dst = r.artifact_dir or (bench_dir / r.task_id)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("report.json", "run_instance.log", "test_output.txt", "eval.sh", "patch.diff"):
            f = src / name
            if f.exists():
                shutil.copy2(f, dst / name)
        return SweBenchGrader._read_report(src / "report.json", r.task_id)

    @staticmethod
    def _read_report(report_path: Path, task_id: str) -> float:
        """A per-instance ``report.json`` → ``resolved`` (1.0/0.0). Missing/malformed → 0.0 with a
        logged reason: swebench leaves a tail of unscored instances (transient pulls, timeouts, a
        few deterministic upstream-harness failures) — scored 0 but never a *silent* zero."""
        if not report_path.exists():
            LOGGER.warning(
                "swe-bench %s: no report.json at %s — scoring 0.0 (instance did not demonstrably "
                "resolve; check the swebench eval log for the underlying error)", task_id, report_path)
            return 0.0
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("swe-bench %s: report.json at %s is not valid JSON — scoring 0.0",
                           task_id, report_path)
            return 0.0
        return 1.0 if bool(report.get(task_id, {}).get("resolved")) else 0.0

    @staticmethod
    def _install_xrlenv_drop_in() -> None:
        """Swap ``docker.from_env`` for xrlenv's so swebench's harness routes every
        container creation through the xrlenv cluster. MUST run before swebench
        imports docker. xrlenv reads its connection config from ``XRLENV_GRPC_HOST``
        / ``XRLENV_GRPC_PORT`` / ``XRLENV_CONSUMER_TOKEN`` / ``XRLENV_GRPC_SECURE``.
        """
        import docker  # noqa: PLC0415

        try:
            from xrlenv.compat.docker_client import from_env as xrlenv_from_env  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - exercised via the error path
            raise RuntimeError(
                "Cluster grading (XRLENV_GRPC_HOST set) requires the vendored "
                "xrlenv to be installed (xrlenv.compat.docker_client)."
            ) from e
        docker.from_env = xrlenv_from_env  # type: ignore[assignment]

    @staticmethod
    def _read_resolved(run_dir: Path, run_id: str, task_id: str) -> float:
        """Read this instance's ``report.json`` → ``resolved`` (1.0/0.0).

        Layout (relative to ``run_dir``, from the chdir above)::

            logs/run_evaluation/<run_id>/<model_name>/<instance_id>/report.json

        A missing report means the instance didn't demonstrably resolve — swebench
        leaves a small tail of unscored instances (transient pulls, timeouts, a few
        deterministic upstream-harness failures). We score those 0.0 but log the
        reason so it isn't a silent zero.
        """
        report_path = (
            run_dir
            / "logs"
            / "run_evaluation"
            / run_id
            / _MODEL_NAME.replace("/", "__")
            / task_id
            / "report.json"
        )
        if not report_path.exists():
            LOGGER.warning(
                "swe-bench %s: no report.json at %s — scoring 0.0 (instance did "
                "not demonstrably resolve; check the swebench eval log for the "
                "underlying error)",
                task_id,
                report_path,
            )
            return 0.0
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning(
                "swe-bench %s: report.json at %s is not valid JSON — scoring 0.0",
                task_id,
                report_path,
            )
            return 0.0
        # The report is keyed by instance_id: report[task_id]["resolved"].
        resolved = bool(report.get(task_id, {}).get("resolved"))
        return 1.0 if resolved else 0.0


@register("swe-bench-verified")
class SweBenchVerified(Benchmark):
    """SWE-bench Verified: HF source + docker drop-in harness + patch-eval grader."""

    name = _BENCH

    def additional_info_post(self, task: Task, ctx: TaskContext) -> str | None:
        """Append the instance's ``hints_text`` (the maintainer's original-issue hints) after the
        problem statement — the one bit of SWE-bench **data** that would otherwise be dropped.
        Empty hints → ``None`` (payload is just the problem). This is data, not framing: how to
        fix the bug (workflow, don't-touch-tests, don't-commit) is the agent's own to say."""
        hints = (task.extras or {}).get("hints", "").strip()
        if not hints:
            return None
        return f"## Hints from the original issue\n\n{hints}"

    def source(self) -> TaskSource:
        return _SweBenchSource()

    def harness(self, env_import_path: str | None = None) -> BenchmarkHarness:
        if env_import_path:      # docker drop-in — no cluster Environment; warn rather than drop silently
            import warnings
            warnings.warn(f"{self.name}: env_import_path is ignored — this benchmark's docker harness "
                          "has no cluster Environment.", stacklevel=2)
        return DockerHarness()

    def grader(self) -> Grader:
        return SweBenchGrader()


__all__ = ["SweBenchVerified", "SweBenchGrader"]
