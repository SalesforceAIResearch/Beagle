"""``HarborBenchmark`` — the zero-code base for harbor-family benchmarks.

A benchmark onboarded into xrlenv (build_cache + oracle green) is runnable in
beagle by subclassing this and doing nothing else: tasks come from the harbor
cache, rollouts run through harbor's native trial driver, and the reward the
harbor verifier already produced is read in-band. Only benchmarks that diverge from
this (different task source or grading) override a method.
"""

from __future__ import annotations

from typing import ClassVar

from beagle.benchmarks.base import (
    Benchmark,
    BenchmarkHarness,
    Grader,
    TaskSource,
)
from beagle.benchmarks.grader import InBandGrader
from beagle.benchmarks.harness.drivers import HarborHarness
from beagle.benchmarks.source import HarborCache


class HarborBenchmark(Benchmark):
    """Harbor-family benchmark with all three pluggables defaulted.

    Set :attr:`cache_name` if the harbor cache dir differs from the registry name.
    """

    #: Harbor cache subdirectory; defaults to the registered ``name``.
    cache_name: ClassVar[str] = ""

    def task_env(self) -> dict[str, str]:
        """Container facts the AGENT phase needs, which the framework doesn't supply.

        Harbor hands the agent an instruction and a container, but not the task's working
        directory or interpreter — its own shim resolves the cwd with ``pwd``. That is enough
        for a benchmark whose images declare a ``WORKDIR`` and put the task's tools on ``PATH``
        (terminal-bench), and wrong for one that leaves both to the harness (SWE-rebench: the
        agent lands in ``/`` with base conda while the verifier activates the task env itself).

        Return either/both of:

        * ``repo_path_cmd`` — a shell snippet PRINTING the task's working dir; run in-container
          in place of the bare ``pwd`` probe. Per-task resolution therefore costs nothing: the
          snippet is evaluated inside each trial.
        * ``shell_preamble`` — sourced before the agent's ``cd``, so exports survive it.

        Same two facts SWE-bench Verified puts on its ``TaskContext`` directly
        (``swe_bench_verified``); this is the harbor path's seam for them. Default ``{}`` = the
        framework's own behaviour, unchanged.
        """
        return {}

    def source(self) -> TaskSource:
        # name = canonical registry name (Task.benchmark identity → the Runner resolves
        # the benchmark back via benchmarks.get); cache_name = the (possibly hyphenated)
        # cache subdir on disk. Collapsing the two leaks the cache name into task identity
        # and breaks benchmarks.get(task.benchmark) at run time.
        return HarborCache(self.name, cache_name=self.cache_name or None)

    def harness(self, env_import_path: str | None = None) -> BenchmarkHarness:
        return HarborHarness(env_import_path=env_import_path, task_env=self.task_env())

    def grader(self) -> Grader:
        return InBandGrader()


__all__ = ["HarborBenchmark"]
