"""``DataMixture`` — mix tasks from multiple benchmarks into one dataset.

This is the dataloader of the design plot. A mixture names several benchmark
components, each with a weight and an optional cap, and produces a unified
:class:`~beagle.data.dataset.TaskDataset` with a reproducible train/val split.

Because every :class:`~beagle.types.Task` remembers its ``benchmark``, grading
and per-source metrics still work downstream even after tasks are interleaved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beagle.data.dataset import TaskDataset, TaskItem
from beagle.data.sampler import ConcatSampler, TaskSampler


@dataclass
class MixtureComponent:
    """One benchmark's contribution to a mixture.

    ``benchmark_spec`` is a benchmark spec (see :mod:`beagle.benchmarks`) or an
    already-built :class:`TaskDataset`. ``weight`` sets its sampling frequency;
    ``limit`` caps how many of its tasks enter the mixture.
    """

    benchmark_spec: object
    weight: float = 1.0
    limit: int | None = None


@dataclass
class DataMixture:
    """A weighted mixture of benchmark datasets.

    Examples
    --------
    >>> mix = DataMixture([                                  # doctest: +SKIP
    ...     MixtureComponent(tb21_spec, weight=3.0),
    ...     MixtureComponent(swev_spec, weight=1.0, limit=100),
    ... ], val_fraction=0.1)
    >>> train_ds, val_ds = mix.split()
    """

    components: list[MixtureComponent] = field(default_factory=list)
    sampler: TaskSampler = field(default_factory=ConcatSampler)
    val_fraction: float = 0.0

    @classmethod
    def from_config(cls, config: dict) -> DataMixture:
        """Build a mixture from a plain-dict config (the ``data_config`` in the API).

        Expected shape::

            {"components": [{"benchmark": {...}, "weight": 3.0, "limit": 100}, ...],
             "sampler": "concat" | "weighted_round_robin",
             "val_fraction": 0.1}

        Each ``benchmark`` sub-dict is parsed into a benchmark spec and resolved to
        a dataset lazily in :meth:`materialize`.
        """
        # TODO(impl): parse components into MixtureComponent(benchmark_spec=...),
        # map "sampler" string to a TaskSampler.
        raise NotImplementedError("DataMixture.from_config not yet implemented")

    def _resolve(self, spec: object) -> TaskDataset:
        """Turn a component's ``benchmark_spec`` into a concrete dataset."""
        if isinstance(spec, TaskDataset):
            return spec
        # TODO: TaskDataset.from_benchmark(spec)
        raise NotImplementedError("benchmark-spec resolution not yet implemented")

    def materialize(self) -> TaskDataset:
        """Resolve every component and combine them into one dataset via the sampler."""
        groups: list[list[TaskItem]] = []
        weights: list[float] = []
        for comp in self.components:
            ds = self._resolve(comp.benchmark_spec)
            items = ds.items if comp.limit is None else ds.items[: comp.limit]
            groups.append(items)
            weights.append(comp.weight)
        combined = self.sampler.combine(groups, weights)
        return TaskDataset(combined, name="mixture")

    def split(self) -> tuple[TaskDataset, TaskDataset]:
        """Materialize, then split into ``(train, val)`` by :attr:`val_fraction`."""
        return self.materialize().split(self.val_fraction)


__all__ = ["MixtureComponent", "DataMixture"]
