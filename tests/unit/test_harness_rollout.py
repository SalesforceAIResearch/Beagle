"""The base per-task ``BenchmarkHarness.rollout`` fan-out — the non-harbor concurrency path.

Harbor benchmarks let harbor's Job engine drive concurrency (``HarborHarness`` overrides rollout);
a benchmark with no engine of its own (docker drop-in like SWE-bench) uses this default, which owns
the fan-out — ported from coding-bench's ``runner.executor`` (proven at 128-way for swe-bench-verified):
serial at ``parallelism<=1``, else a ThreadPoolExecutor over per-task ``run``, capturing per-task
failures so one bad task can't sink the batch.
"""

from __future__ import annotations

import threading

import pytest

from beagle.benchmarks.base import BenchmarkHarness
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult


class _Agent:
    def rollout_binding(self, ctx):
        return None


def _items(*ids: str):
    return [(Task(task_id=i, benchmark="b"), TaskContext(image=None)) for i in ids]


class _BarrierHarness(BenchmarkHarness):
    """Each ``run`` blocks on a shared barrier — it only returns once N tasks are in flight AT ONCE,
    so the call completes iff the harness actually ran them concurrently."""

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=5)

    def run(self, binding, task, task_ctx, *, runtime):
        self.barrier.wait()
        return TaskResult(task_id=task.task_id, resolved=True, reward=1.0)


def test_rollout_parallel_runs_tasks_concurrently(tmp_path) -> None:
    # parallelism=2 must run both tasks at once — a 2-party barrier returns only when BOTH are in
    # flight. If the base rollout serialized, the first run() would block forever → BrokenBarrierError.
    h = _BarrierHarness(parties=2)
    out = list(h.rollout(_Agent(), _items("t1", "t2"), runtime=None, run_dir=tmp_path, parallelism=2))
    assert {r.task_id for r in out} == {"t1", "t2"} and all(r.resolved for r in out)


def test_rollout_serial_at_parallelism_1(tmp_path) -> None:
    # parallelism=1 → inline serial (no pool). A 1-party barrier is a no-op, so each run returns.
    h = _BarrierHarness(parties=1)
    out = list(h.rollout(_Agent(), _items("t1", "t2", "t3"), runtime=None, run_dir=tmp_path, parallelism=1))
    assert [r.task_id for r in out] == ["t1", "t2", "t3"]   # serial preserves submission order


class _RaisingHarness(BenchmarkHarness):
    def run(self, binding, task, task_ctx, *, runtime):
        if task.task_id == "boom":
            raise RuntimeError("kaboom")
        return TaskResult(task_id=task.task_id, resolved=True, reward=1.0)


@pytest.mark.parametrize("parallelism", [1, 3])
def test_rollout_captures_per_task_failure(parallelism: int, tmp_path) -> None:
    # One task raising must NOT sink the batch — it's captured as a FAILED TaskResult (error
    # recorded), the rest complete. Holds on both the serial and the threaded path.
    h = _RaisingHarness()
    out = {r.task_id: r for r in h.rollout(_Agent(), _items("ok1", "boom", "ok2"),
                                           runtime=None, run_dir=tmp_path, parallelism=parallelism)}
    assert len(out) == 3
    assert out["ok1"].resolved and out["ok2"].resolved
    assert out["boom"].status is RolloutStatus.FAILED
    assert out["boom"].error is not None and "kaboom" in out["boom"].error


def test_rollout_progress_line(monkeypatch, capsys, tmp_path) -> None:
    # The non-harbor fan-out (docker drop-in like SWE-bench) prints a live progress line — harbor has
    # its own bar, this path had nothing. Force the non-TTY branch (plain lines, deterministic).
    import beagle.benchmarks.base as base
    monkeypatch.setattr(base.sys.stderr, "isatty", lambda: False, raising=False)
    h = _RaisingHarness()
    list(h.rollout(_Agent(), _items("ok1", "boom", "ok2"),
                   runtime=None, run_dir=tmp_path, parallelism=1))
    err = capsys.readouterr().err
    assert "[b] rollout" in err          # prefixed by the benchmark name
    assert "3/3" in err                  # reaches the total
    assert "1 err" in err                # the failed task is surfaced in the line


def test_rollout_progress_helper_tty() -> None:
    # Unit-level: the TTY branch updates in place (\r) and closes with a newline.
    import io

    from beagle.benchmarks.base import _RolloutProgress

    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    tty = _Tty()
    p = _RolloutProgress("swe-bench-verified", 2, 1, stream=tty)
    p.update(TaskResult(task_id="a", resolved=True, reward=1.0))
    p.update(TaskResult(task_id="b", resolved=True, reward=1.0))
    p.close()
    out = tty.getvalue()
    assert "[swe-bench-verified] rollout: 2 task(s), parallelism 1" in out  # start line up front
    assert "\r" in out                                          # in-place update on a TTY
    assert "[swe-bench-verified] rollout 2/2" in out
    assert out.endswith("\n")                                   # closed with a fresh line


def test_harbor_harness_env_import_path_is_overridable() -> None:
    # #12/5: a local (non-cluster) harbor run can point the harness at a different Environment class
    # via the constructor — no monkeypatching the class attribute, no env var.
    from beagle.benchmarks.harness.drivers import HarborHarness, PierHarness

    assert HarborHarness().ENV_IMPORT_PATH == "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"
    assert PierHarness().ENV_IMPORT_PATH == "xrlenv_plugins.pier:XrlenvPierEnvironmentCluster"
    over = HarborHarness(env_import_path="my.mod:LocalEnv")
    assert over.ENV_IMPORT_PATH == "my.mod:LocalEnv"                    # instance override applied
    assert HarborHarness().ENV_IMPORT_PATH != "my.mod:LocalEnv"        # class default not mutated


def test_benchmark_harness_threads_env_import_path() -> None:
    # #20 item 1: a benchmark's options.env_import_path reaches the harbor/pier harness through
    # Benchmark.harness(env_import_path=...) — the seam the runner uses from the config.
    from beagle.benchmarks.harness.benchmark import HarborBenchmark
    from beagle.benchmarks.harness.drivers import HarborHarness

    class _B(HarborBenchmark):
        name = "t-env-import"

    h = _B().harness(env_import_path="my.mod:LocalEnv")
    assert isinstance(h, HarborHarness) and h.ENV_IMPORT_PATH == "my.mod:LocalEnv"
    assert _B().harness().ENV_IMPORT_PATH == HarborHarness.ENV_IMPORT_PATH   # default when unset
