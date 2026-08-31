"""Samplers decide how per-benchmark task lists are combined into one stream.

Kept separate from :class:`~beagle.data.mixture.DataMixture` so the *policy*
(concatenate, weighted round-robin, proportional) is swappable without touching
the mixture container. A sampler is deterministic given its inputs (any
randomness is seeded) so mixed datasets are reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction

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

    Deterministic stride schedule (no RNG). Useful when a training stream should see
    benchmarks intermixed rather than in blocks.

    Every item of every group is emitted -- weight sets the *order*, not the quantity. The
    requested ratio therefore holds only while all groups still have items; once a group runs
    out the remainder follows in the ratio of whatever is left. Cap a component with
    ``MixtureComponent.limit`` if the ratio should hold across the whole stream.
    """

    def combine(self, groups: list[list[TaskItem]], weights: list[float]) -> list[TaskItem]:
        if len(groups) != len(weights):
            raise ValueError(
                f"got {len(groups)} groups but {len(weights)} weights; they must correspond"
            )

        # Stride scheduling. Each group draws at intervals of 1/weight along a shared virtual
        # clock, and at every step the group whose next draw is earliest emits one item. A
        # weight-3 group therefore draws three times as often as a weight-1 one.
        #
        # The clock is Fraction rather than float because ties are common (equal weights make
        # every group's draw times coincide) and float rounding would break them
        # inconsistently, so two runs of the same mixture could order differently. Ties resolve
        # to the lowest index, which makes the whole schedule reproducible without a seed.
        strides: list[Fraction | None] = []
        for i, (group, weight) in enumerate(zip(groups, weights)):
            if not group:
                strides.append(None)
                continue
            if weight <= 0:
                # Silently dropping tasks from a training mixture is the kind of bug that
                # surfaces as a mysteriously weak model weeks later, so refuse instead.
                raise ValueError(
                    f"component {i} has weight {weight!r}; weights must be > 0. "
                    "To exclude a component, remove it from the mixture."
                )
            strides.append(Fraction(1) / Fraction(weight))

        total = sum(len(g) for g in groups)
        cursors = [0] * len(groups)
        next_draw: list[Fraction | None] = list(strides)

        out: list[TaskItem] = []
        while len(out) < total:
            pick = -1
            earliest: Fraction | None = None
            for i, stride in enumerate(strides):
                if stride is None or cursors[i] >= len(groups[i]):
                    continue  # exhausted or empty; the rest simply close ranks
                draw = next_draw[i]
                assert draw is not None
                if earliest is None or draw < earliest:
                    earliest, pick = draw, i
            if pick < 0:  # unreachable while len(out) < total, but keeps the loop bounded
                break
            out.append(groups[pick][cursors[pick]])
            cursors[pick] += 1
            next_draw[pick] = next_draw[pick] + strides[pick]  # type: ignore[operator]
        return out


__all__ = ["TaskSampler", "ConcatSampler", "WeightedRoundRobinSampler"]
