"""``TaskDataset`` — an indexable collection of tasks, the PyTorch ``Dataset``.

A dataset is a flat, ordered sequence of ``(Task, TaskContext)`` pairs. It is
benchmark-agnostic: tasks remember their origin via ``Task.benchmark`` so a mixed
dataset can still be graded and weighted per source. The container operations
here are fully implemented; construction *from a benchmark* defers to
:mod:`beagle.benchmarks` loaders (stubbed until those land).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from beagle.types import Task, TaskContext, TaskId

if TYPE_CHECKING:
    from beagle.benchmarks.base import BenchmarkSpec

#: The unit of a dataset: a task plus everything needed to stand up its container.
TaskItem = tuple[Task, TaskContext]


@dataclass
class TaskDataset:
    """An ordered, filterable set of tasks.

    Parameters
    ----------
    items:
        The ``(Task, TaskContext)`` pairs.
    name:
        Human label (usually the benchmark name, or ``"mixture"``).
    benchmark_spec:
        The :class:`BenchmarkSpec` this dataset was built from, when it came from a single
        benchmark (via :meth:`from_benchmark`). Kept so the Trainer can derive the eval
        ``RunConfig`` in the direct (PyTorch-UX) path without a separate config. ``None`` for
        ad-hoc datasets and mixtures.
    """

    items: list[TaskItem]
    name: str = ""
    benchmark_spec: BenchmarkSpec | None = None
    #: Every benchmark contributing to this dataset. One entry for a single-benchmark dataset,
    #: several for a mixture — where ``benchmark_spec`` is ``None`` precisely because there is
    #: no single one. Without this a mixture is unscoreable: the Trainer derives the eval
    #: config from ``benchmark_spec``, so a mixture would resolve to no benchmark at all.
    benchmark_specs: list[BenchmarkSpec] = field(default_factory=list)

    # -- sequence protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TaskItem:
        return self.items[idx]

    def __iter__(self) -> Iterator[TaskItem]:
        return iter(self.items)

    @property
    def task_ids(self) -> list[TaskId]:
        return [t.task_id for t, _ in self.items]

    # -- transforms (return new datasets, never mutate) ----------------------

    def _like(self, items: list[TaskItem]) -> TaskDataset:
        """A new dataset over ``items`` keeping this one's provenance.

        Transforms must carry ``benchmark_spec``/``benchmark_specs`` through, or a derived
        dataset — notably the val half of a split — becomes unscoreable for no visible reason.
        """
        return TaskDataset(items, name=self.name, benchmark_spec=self.benchmark_spec,
                           benchmark_specs=list(self.benchmark_specs))

    def filter(self, predicate: Callable[[Task], bool]) -> TaskDataset:
        return self._like([(t, c) for t, c in self.items if predicate(t)])

    def select(self, task_ids: list[TaskId]) -> TaskDataset:
        """Keep only ``task_ids``, preserving the requested order. Missing ids raise."""
        by_id = {t.task_id: (t, c) for t, c in self.items}
        try:
            picked = [by_id[i] for i in task_ids]
        except KeyError as e:
            raise KeyError(f"task id {e.args[0]!r} not in dataset {self.name!r}") from None
        return self._like(picked)

    def concat(self, other: TaskDataset) -> TaskDataset:
        merged: list[BenchmarkSpec] = list(self.benchmark_specs)
        for spec in other.benchmark_specs:
            if spec not in merged:
                merged.append(spec)
        # Two datasets from different benchmarks have no single primary any more.
        primary = self.benchmark_spec if self.benchmark_spec == other.benchmark_spec else None
        return TaskDataset(self.items + other.items, name=self.name or other.name,
                           benchmark_spec=primary, benchmark_specs=merged)

    def split(self, val_fraction: float = 0.2) -> tuple[TaskDataset, TaskDataset]:
        """Deterministic tail split into ``(train, val)``.

        A plain positional split (no shuffle) so runs are reproducible without a
        seed. Callers wanting stratified/shuffled splits should compose their own
        sampler (see :mod:`beagle.data.sampler`).
        """
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        n_val = int(round(len(self.items) * val_fraction))
        cut = len(self.items) - n_val
        return self._like(self.items[:cut]), self._like(self.items[cut:])

    # -- construction from a benchmark ---------------------------------------

    @classmethod
    def from_benchmark(cls, spec: object) -> TaskDataset:
        """Materialize a dataset from a benchmark — a declarative
        :class:`~beagle.config.BenchmarkConfig` or a runtime :class:`BenchmarkSpec`.

        A ``BenchmarkConfig`` (the declarative surface) is converted to its spec first. Delegates
        to :func:`beagle.benchmarks.load_tasks` (which applies the spec's ``task_ids`` /
        ``exclude_task_ids`` / ``num_samples`` selection), collecting the streamed
        ``(Task, TaskContext)`` pairs; the resolved spec is kept as ``benchmark_spec``.
        """
        from beagle.benchmarks.registry import load_tasks  # lazy: avoid import cycle

        if hasattr(spec, "to_spec"):     # a BenchmarkConfig (declarative) → its BenchmarkSpec
            spec = spec.to_spec()
        return cls(list(load_tasks(spec)), name=getattr(spec, "name", ""),  # type: ignore[arg-type]
                   benchmark_spec=spec, benchmark_specs=[spec])  # type: ignore[arg-type,list-item]


__all__ = ["TaskItem", "TaskDataset"]
