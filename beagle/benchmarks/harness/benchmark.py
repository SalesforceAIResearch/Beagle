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

    def source(self) -> TaskSource:
        # name = canonical registry name (Task.benchmark identity → the Runner resolves
        # the benchmark back via benchmarks.get); cache_name = the (possibly hyphenated)
        # cache subdir on disk. Collapsing the two leaks the cache name into task identity
        # and breaks benchmarks.get(task.benchmark) at run time.
        return HarborCache(self.name, cache_name=self.cache_name or None)

    def harness(self, env_import_path: str | None = None) -> BenchmarkHarness:
        return HarborHarness(env_import_path=env_import_path)

    def grader(self) -> Grader:
        return InBandGrader()


__all__ = ["HarborBenchmark"]
