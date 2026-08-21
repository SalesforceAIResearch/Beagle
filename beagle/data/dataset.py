"""``TaskDataset`` — an indexable collection of tasks, the PyTorch ``Dataset``.

A dataset is a flat, ordered sequence of ``(Task, TaskContext)`` pairs. It is
benchmark-agnostic: tasks remember their origin via ``Task.benchmark`` so a mixed
dataset can still be graded and weighted per source. The container operations
here are fully implemented; construction *from a benchmark* defers to
:mod:`beagle.benchmarks` loaders (stubbed until those land).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
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

    def filter(self, predicate: Callable[[Task], bool]) -> TaskDataset:
        return TaskDataset([(t, c) for t, c in self.items if predicate(t)], name=self.name)

    def select(self, task_ids: list[TaskId]) -> TaskDataset:
        """Keep only ``task_ids``, preserving the requested order. Missing ids raise."""
        by_id = {t.task_id: (t, c) for t, c in self.items}
        try:
            picked = [by_id[i] for i in task_ids]
        except KeyError as e:
            raise KeyError(f"task id {e.args[0]!r} not in dataset {self.name!r}") from None
        return TaskDataset(picked, name=self.name)

    def concat(self, other: TaskDataset) -> TaskDataset:
        return TaskDataset(self.items + other.items, name=self.name or other.name)

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
        train = TaskDataset(self.items[:cut], name=self.name)
        val = TaskDataset(self.items[cut:], name=self.name)
        return train, val

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
                   benchmark_spec=spec)  # type: ignore[arg-type]


__all__ = ["TaskItem", "TaskDataset"]
