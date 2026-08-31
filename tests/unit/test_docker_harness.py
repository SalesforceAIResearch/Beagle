"""DockerHarness.run — the swebench-style rollout: delegate to the agent's run (produce a patch),
grading is the benchmark's PatchEvalGrader (separate). Hermetic (no container)."""

from __future__ import annotations

import pytest

from beagle.benchmarks.harness import DockerHarness
from beagle.rollout.binding import GenericBinding, HarborBinding
from beagle.types import Task, TaskContext, TaskResult


def test_run_delegates_to_the_agents_binding() -> None:
    seen: dict = {}

    def _agent_run(task, task_ctx, *, runtime):  # noqa: ANN001 — the agent's run callable
        seen.update(task=task.task_id, repo=task_ctx.repo_path, runtime=runtime)
        return TaskResult(task_id=task.task_id, patch="THE DIFF")

    res = DockerHarness().run(
        GenericBinding(run=_agent_run),
        Task(task_id="astropy__astropy-12907", problem_statement="fix it", benchmark="swe-bench-verified"),
        TaskContext(image="swebench/img", repo_path="/testbed"),
        runtime="RT",
    )
    assert res.patch == "THE DIFF"
    assert seen == {"task": "astropy__astropy-12907", "repo": "/testbed", "runtime": "RT"}


def test_run_rejects_a_non_generic_binding() -> None:
    with pytest.raises(TypeError, match="generic binding"):
        DockerHarness().run(HarborBinding(import_path="m:A"),
                            Task(task_id="t", problem_statement="p", benchmark="swe-bench-verified"),
                            TaskContext(image=None), runtime=None)
