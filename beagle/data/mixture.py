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
from beagle.data.sampler import ConcatSampler, TaskSampler, WeightedRoundRobinSampler

#: Sampler names accepted in a ``data_config``, mapped to their implementations.
_SAMPLERS: dict[str, type[TaskSampler]] = {
    "concat": ConcatSampler,
    "weighted_round_robin": WeightedRoundRobinSampler,
}


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
        from beagle.config import BenchmarkConfig  # lazy: avoid an import cycle via config

        if not isinstance(config, dict):
            raise TypeError(f"data mixture config must be a dict, got {type(config).__name__}")

        raw_components = config.get("components") or []
        if not raw_components:
            raise ValueError("data mixture config needs at least one entry under 'components'")

        components: list[MixtureComponent] = []
        for i, raw in enumerate(raw_components):
            if not isinstance(raw, dict):
                raise TypeError(f"component {i} must be a dict, got {type(raw).__name__}")
            benchmark = raw.get("benchmark")
            if benchmark is None:
                raise ValueError(f"component {i} has no 'benchmark'")
            # A dict is the declarative surface; anything else is assumed to be an
            # already-built spec or TaskDataset and is passed through to _resolve.
            spec = BenchmarkConfig(**benchmark) if isinstance(benchmark, dict) else benchmark
            components.append(
                MixtureComponent(
                    benchmark_spec=spec,
                    weight=float(raw.get("weight", 1.0)),
                    limit=raw.get("limit"),
                )
            )

        sampler_name = config.get("sampler", "concat")
        try:
            sampler = _SAMPLERS[sampler_name]()
        except KeyError:
            raise ValueError(
                f"unknown sampler {sampler_name!r}; expected one of {sorted(_SAMPLERS)}"
            ) from None

        return cls(
            components=components,
            sampler=sampler,
            val_fraction=float(config.get("val_fraction", 0.0)),
        )

    def _resolve(self, spec: object) -> TaskDataset:
        """Turn a component's ``benchmark_spec`` into a concrete dataset."""
        if isinstance(spec, TaskDataset):
            return spec
        return TaskDataset.from_benchmark(spec)

    def materialize(self) -> TaskDataset:
        """Resolve every component and combine them into one dataset via the sampler.

        The result carries ``benchmark_specs`` (all components) but no ``benchmark_spec`` —
        a mixture genuinely has no single benchmark, and the plural is what lets the Trainer
        still derive an eval config for it.
        """
        groups: list[list[TaskItem]] = []
        weights: list[float] = []
        specs = []
        for comp in self.components:
            ds = self._resolve(comp.benchmark_spec)
            items = ds.items if comp.limit is None else ds.items[: comp.limit]
            groups.append(items)
            weights.append(comp.weight)
            specs.extend(s for s in ds.benchmark_specs if s not in specs)
        combined = self.sampler.combine(groups, weights)
        return TaskDataset(combined, name="mixture", benchmark_specs=specs)

    def split(self) -> tuple[TaskDataset, TaskDataset]:
        """Materialize, then split into ``(train, val)`` by :attr:`val_fraction`."""
        return self.materialize().split(self.val_fraction)


__all__ = ["MixtureComponent", "DataMixture"]
