"""Samplers decide how per-benchmark task lists are combined into one stream.

Kept separate from :class:`~beagle.data.mixture.DataMixture` so the *policy*
(concatenate, weighted round-robin, proportional) is swappable without touching
the mixture container. A sampler is deterministic given its inputs (any
randomness is seeded) so mixed datasets are reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from beagle.data.dataset import TaskItem


class TaskSampler(ABC):
    """Combine several ordered task lists into a single ordered task list."""

    @abstractmethod
    def combine(self, groups: list[list[TaskItem]], weights: list[float]) -> list[TaskItem]:
        """Return the mixed, ordered task items.

        ``groups[i]`` are the (already limit-applied) items of component ``i`` and
        ``weights[i]`` its relative weight.
        """
        raise NotImplementedError


class ConcatSampler(TaskSampler):
    """Concatenate components in order. Weights ignored. The trivial default."""

    def combine(self, groups: list[list[TaskItem]], weights: list[float]) -> list[TaskItem]:
        out: list[TaskItem] = []
        for g in groups:
            out.extend(g)
        return out


class WeightedRoundRobinSampler(TaskSampler):
    """Interleave components so each appears with frequency proportional to weight.

    Deterministic largest-remainder interleave (no RNG). Useful when a training
    stream should see benchmarks intermixed rather than in blocks.
    """

    def combine(self, groups: list[list[TaskItem]], weights: list[float]) -> list[TaskItem]:
        # TODO(impl): stride-schedule each group by its normalized weight and
        # merge. Left as a stub in the skeleton; ConcatSampler is the default.
        raise NotImplementedError("WeightedRoundRobinSampler.combine not yet implemented")


__all__ = ["TaskSampler", "ConcatSampler", "WeightedRoundRobinSampler"]
