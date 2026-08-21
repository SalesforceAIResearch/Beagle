"""The benchmark registry and top-level dispatch helpers.

Benchmarks self-register under their canonical name via :func:`register`. The
runner resolves a task's benchmark via :func:`get` (keyed by ``Task.benchmark``),
which is what lets a mixed dataset dispatch each task to the right native harness.
Like agents, benchmarks are a one-file drop — decorate and place under
``beagle/benchmarks/``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from beagle.benchmarks.base import Benchmark, BenchmarkSpec
from beagle.registry import Registry
from beagle.types import Task, TaskContext

#: name -> Benchmark subclass. Populated by the ``@register`` decorator at import.
BENCHMARKS: Registry[type[Benchmark]] = Registry("benchmark")


def register(name: str) -> Callable[[type[Benchmark]], type[Benchmark]]:
    """Class decorator: register a benchmark under ``name`` and stamp it on the class."""

    def _decorate(cls: type[Benchmark]) -> type[Benchmark]:
        cls.name = name
        BENCHMARKS.register(name, cls)
        return cls

    return _decorate


def get(name: str) -> Benchmark:
    """Instantiate the benchmark registered under ``name``."""
    return BENCHMARKS.get(name)()


def load_tasks(spec: BenchmarkSpec) -> Iterator[tuple[Task, TaskContext]]:
    """Stream ``(Task, TaskContext)`` for ``spec`` from its benchmark loader."""
    return get(spec.name).load_tasks(spec)


def available() -> list[str]:
    return BENCHMARKS.names()


__all__ = ["BENCHMARKS", "register", "get", "load_tasks", "available"]
