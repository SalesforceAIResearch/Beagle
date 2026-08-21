"""Reusable graders — building blocks, not a closed taxonomy.

The framework defines only the open :class:`~beagle.benchmarks.base.Grader`
interface. These are common implementations a benchmark can reuse; a benchmark with
a shape none of these fit just implements ``Grader`` directly. Nothing here is a
"kind" the framework special-cases.

* :class:`InBandGrader` — the reward was produced *during* the rollout (harbor
  verifier, WAI verifier). Grading is a reduction; zero benchmark-specific code.
* :class:`PatchEvalGrader` — the agent produced a patch; a separate evaluator scores
  it. A base class: a concrete grader implements :meth:`PatchEvalGrader.evaluate_patch`
  (running the benchmark's own evaluator, typically in an xrlenv container). Its
  dependencies are imported lazily and gated behind an optional extra.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from beagle.benchmarks.base import Grader, GradeReport
from beagle.rollout.runtime import ContainerRuntime
from beagle.types import TaskResult


class InBandGrader(Grader):
    """Reduce rewards the rollout already produced. The default for in-band benchmarks."""

    def __init__(self, resolved_threshold: float = 1.0) -> None:
        self.resolved_threshold = resolved_threshold

    def grade(
        self, results: list[TaskResult], *, runtime: ContainerRuntime, run_dir: Path,
        parallelism: int = 1,
    ) -> GradeReport:
        # A missing reward (an errored/crashed trial) scores 0 and is counted in
        # the denominator — dropping failures would inflate the fitness scalar the
        # algorithm optimizes. (Errored trials remain distinguishable via
        # TaskResult.error / status for callers that want to exclude infra flakes.)
        # ``parallelism`` is unused — the reward is already in-band; grading is a pure reduce.
        def _reward(r: TaskResult) -> float:
            return r.reward if r.reward is not None else 0.0

        per_task = {r.task_id: _reward(r) for r in results}
        num_resolved = sum(
            1
            for r in results
            if r.resolved or (r.reward is not None and r.reward >= self.resolved_threshold)
        )
        score = (sum(per_task.values()) / len(per_task)) if per_task else 0.0
        return GradeReport(
            num_tasks=len(results),
            num_resolved=num_resolved,
            score=score,
            eval_dir=run_dir,
            per_task=per_task,
        )


class PatchEvalGrader(Grader):
    """Base for graders that run an evaluator on the agent's produced patch.

    Concrete subclasses implement :meth:`evaluate_patch` (SWE-bench Verified runs the
    upstream evaluator via ``xrlenv.from_env()``; SWE-bench Pro runs vendored
    per-instance scripts). This base handles the reduce; each subclass owns only the
    per-task evaluation + its lazy-imported deps.
    """

    @abstractmethod
    def evaluate_patch(self, task_id: str, patch: str, *, runtime: ContainerRuntime) -> float:
        """Score one patch (e.g. fraction of tests passing, or 0/1)."""
        raise NotImplementedError

    def grade(
        self, results: list[TaskResult], *, runtime: ContainerRuntime, run_dir: Path,
        parallelism: int = 1,
    ) -> GradeReport:
        # Generic per-task path: score each patch independently, fanning out to ``parallelism``
        # workers (each ``evaluate_patch`` blocks on an evaluator container — threads release the
        # GIL). A benchmark whose evaluator has a NATIVE batch mode (SWE-bench's ``run_evaluation``
        # with ``max_workers``) should override :meth:`grade` instead of paying N separate calls.
        def _score(r: TaskResult) -> float:
            return 0.0 if r.error is not None else self.evaluate_patch(
                r.task_id, r.patch or "", runtime=runtime)

        if parallelism > 1 and len(results) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="beagle-grade") as ex:
                scores = list(ex.map(_score, results))
        else:
            scores = [_score(r) for r in results]

        per_task: dict[str, float] = {}
        for r, score in zip(results, scores):
            r.reward = score
            r.resolved = score >= 1.0
            per_task[r.task_id] = score
        return GradeReport(
            num_tasks=len(results),
            num_resolved=sum(1 for r in results if r.resolved),
            score=(sum(per_task.values()) / len(per_task)) if per_task else 0.0,
            eval_dir=run_dir,
            per_task=per_task,
        )


__all__ = ["InBandGrader", "PatchEvalGrader"]
