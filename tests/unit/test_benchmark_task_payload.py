"""Benchmark task-payload assembly: the raw ``problem_statement`` is left untouched, and the
agent-facing ``instruction`` is ``additional_info_pre`` + task + ``additional_info_post`` (blank
sections dropped). beagle ships **no** prompt framing — the agent supplies its own. The hooks
carry benchmark **data** only (SWE-bench's ``hints_text`` is the one instance in the tree).
See ``notes/task-prompt-injection.md``."""

from __future__ import annotations

from beagle.benchmarks.base import Benchmark, BenchmarkSpec, TaskSource
from beagle.types import Task, TaskContext


class _FakeSource(TaskSource):
    def tasks(self, spec):  # noqa: ANN001
        yield (Task(task_id="t1", problem_statement="RAW GOAL", benchmark="fake",
                    extras={"hints": "HINT"}),
               TaskContext(image=None, repo_path="/work"))


class _FakeBench(Benchmark):
    name = "fake"

    def source(self) -> TaskSource:
        return _FakeSource()

    def harness(self):  # noqa: ANN201 (not exercised by load_tasks)
        return None

    def grader(self):  # noqa: ANN201
        return None


def test_no_hooks_payload_is_raw_task() -> None:
    (task, _), = list(_FakeBench().load_tasks(BenchmarkSpec(name="fake")))
    # raw stays raw, and with no hooks the payload equals the raw task
    assert task.problem_statement == "RAW GOAL"
    assert task.instruction == "RAW GOAL"
    assert task.prompt() == "RAW GOAL"


class _HookedBench(_FakeBench):
    name = "hooked"

    def additional_info_pre(self, task, ctx):  # noqa: ANN001
        return "PRE INFO"

    def additional_info_post(self, task, ctx):  # noqa: ANN001
        return f"POST for {task.task_id} @ {ctx.repo_path}"


def test_pre_and_post_hooks_wrap_the_raw_task_in_order() -> None:
    (task, _), = list(_HookedBench().load_tasks(BenchmarkSpec(name="hooked")))
    assert task.problem_statement == "RAW GOAL"                       # never overwritten
    assert task.instruction == "PRE INFO\n\nRAW GOAL\n\nPOST for t1 @ /work"
    assert task.prompt() == task.instruction


class _PostOnlyBlankBench(_FakeBench):
    name = "postblank"

    def additional_info_post(self, task, ctx):  # noqa: ANN001
        return "   "  # blank/whitespace hook is dropped, not glued on


def test_blank_hook_is_dropped() -> None:
    (task, _), = list(_PostOnlyBlankBench().load_tasks(BenchmarkSpec(name="postblank")))
    assert task.instruction == "RAW GOAL"


def test_prompt_falls_back_to_raw_for_a_directly_constructed_task() -> None:
    # e.g. the harbor shim builds a Task from harbor's native instruction.md (no load_tasks)
    t = Task(task_id="trial", problem_statement="HARBOR INSTRUCTION")
    assert t.instruction == ""
    assert t.prompt() == "HARBOR INSTRUCTION"


def test_swebench_post_hook_renders_hints_as_data() -> None:
    from beagle.benchmarks.swe_bench_verified import SweBenchVerified

    bench = SweBenchVerified()
    ctx = TaskContext(image=None, repo_path="/testbed")
    with_hints = Task(task_id="django__django-1", problem_statement="fix it",
                      extras={"hints": "look at forms.py"})
    assert bench.additional_info_post(with_hints, ctx) == (
        "## Hints from the original issue\n\nlook at forms.py")
    # empty hints → no post section at all (payload is just the problem)
    no_hints = Task(task_id="x", problem_statement="fix it", extras={"hints": ""})
    assert bench.additional_info_post(no_hints, ctx) is None
