"""Benchmark integration.

Public surface::

    import beagle as bgl
    bgl.benchmarks.available()                 # -> ["swe-bench-verified", "terminal_bench_2_1"]
    bench = bgl.benchmarks.get("terminal_bench_2_1")
    tasks = bgl.benchmarks.load_tasks(spec)

A benchmark is onboarded once in xrlenv; beagle consumes it. Each `Benchmark` is
three pluggables — `source`, `harness`, `grader` — with defaults:

* **Harbor-family** → subclass `HarborBenchmark`, done: tasks from the harbor cache
  (`HarborCache`), harbor trial harness, in-band reward (`InBandGrader`). Zero code.
* **Otherwise** → provide a `TaskSource` and/or a `Grader`. Grading is an *open*
  interface; `InBandGrader` / `PatchEvalGrader` are reusable building blocks, not a
  fixed set of kinds.

Adding a benchmark is a one-file drop under `beagle/benchmarks/` (auto-discovered).
"""

from __future__ import annotations

import importlib
import pkgutil

from beagle.benchmarks.base import (
    Benchmark,
    BenchmarkHarness,
    BenchmarkSpec,
    GradeReport,
    Grader,
    TaskSource,
)
from beagle.benchmarks.grader import InBandGrader, PatchEvalGrader
from beagle.benchmarks.harness import (
    DockerHarness,
    HarborBenchmark,
    HarborHarness,
    NativeRunnerHarness,
    PierHarness,
)
from beagle.benchmarks.registry import BENCHMARKS, available, get, load_tasks, register
from beagle.benchmarks.source import HarborCache

# Framework modules/packages (interfaces + reusable building blocks) — not benchmarks.
_FRAMEWORK_MODULES = {"base", "harness", "registry", "source", "grader"}


def _autodiscover() -> None:
    """Import every benchmark module/subpackage so its ``@register`` runs."""
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name in _FRAMEWORK_MODULES:
            continue
        importlib.import_module(f"{__name__}.{info.name}")


_autodiscover()

__all__ = [
    # factory / dispatch
    "BENCHMARKS",
    "register",
    "get",
    "load_tasks",
    "available",
    # contract
    "Benchmark",
    "BenchmarkHarness",
    "BenchmarkSpec",
    "TaskSource",
    "Grader",
    "GradeReport",
    # reusable building blocks
    "HarborBenchmark",
    "HarborCache",
    "HarborHarness",
    "PierHarness",
    "DockerHarness",
    "NativeRunnerHarness",
    "InBandGrader",
    "PatchEvalGrader",
]
