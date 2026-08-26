"""Use ``atelier.verifier`` scores as a fitness signal for self_evolve.

self_evolve's parent selection ranks nodes by a single ``score`` column,
which today is the binary pass-rate (∑ rewards / num_tasks). Binary
pass/fail throws away the per-trajectory quality signal that
LLM-as-a-Verifier extracts.

This module computes a **verifier-augmented fitness score** that blends:
- the original pass-rate (the ground-truth signal — verifier shouldn't
  overrule the actual benchmark verdict), and
- the verifier's per-trajectory soundness score (a noise-resistant
  continuous signal that breaks ties between nodes with similar
  pass-rates).

Two integration points:

1. **As a tiebreaker** (recommended starting point): use raw pass-rate as
   the dominant score; only consult verifier when pass-rates tie.
2. **As a blended score**: ``score = (1 - α) * pass_rate + α * verifier``
   for some α in [0, 1]. Replaces self_evolve's ``score`` column directly.

Approach (2) gives the most search benefit per the LLM-as-a-Verifier
paper, but also the most opportunity for the verifier to mislead the
search if it's mis-calibrated. We default to (2) with α=0.3 (modest
verifier influence) and let callers override.

The module is a pure computation — given a candidate's per-task results
and a verifier instance, it produces a fitness scalar. Wiring it into
self_evolve's score column happens in a separate small patch to
``self_evolve/pipeline.py`` (write-side) and ``parent_selection.py``
(read-side); both already consume ``Node.score``, so the integration
surface is minimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .verifier import TrajectoryAssessment, TrajectoryInput, Verifier


DEFAULT_VERIFIER_WEIGHT = 0.3
"""α in ``score = (1-α)*pass_rate + α*verifier``. Conservative default
gives the verifier ~30% of the signal, so it can break ties + lift
genuinely-better trajectories but can't outvote the actual benchmark."""


@dataclass(frozen=True)
class FitnessComponents:
    """The two raw inputs to fitness blending, kept separately so the
    campaign report can show both components for diagnosis."""

    pass_rate: float
    """Fraction of search-subset tasks the candidate passed."""

    verifier_mean: float
    """Mean of per-trajectory aggregated verifier scores, in [0, 1].
    Computed over all trajectories that have a verifier assessment.
    If no trajectories have been scored, this is 0.0."""

    n_assessed: int
    """Number of trajectories the verifier was actually run on (may be
    less than the search-subset size if the caller skipped tasks)."""


@dataclass(frozen=True)
class FitnessScore:
    """Verifier-augmented fitness, with components preserved."""

    candidate_id: str
    components: FitnessComponents
    alpha: float
    """The blending weight used. Stored so cross-run comparisons stay
    consistent (don't blend with different alphas)."""

    @property
    def value(self) -> float:
        """Final scalar fitness in [0, 1]: ``(1-α)*pass_rate + α*verifier``."""
        return (
            (1.0 - self.alpha) * self.components.pass_rate
            + self.alpha * self.components.verifier_mean
        )

    def to_summary(self) -> str:
        c = self.components
        return (
            f"fitness[{self.candidate_id}] = "
            f"{(1 - self.alpha):.2f}*{c.pass_rate:.3f} + "
            f"{self.alpha:.2f}*{c.verifier_mean:.3f} "
            f"(n={c.n_assessed}) = {self.value:.3f}"
        )


# ─── Computation ──────────────────────────────────────────────────────────


def compute_fitness(
    candidate_id: str,
    *,
    task_rewards: Mapping[str, float],
    assessments: Mapping[str, TrajectoryAssessment],
    alpha: float = DEFAULT_VERIFIER_WEIGHT,
) -> FitnessScore:
    """Blend self_evolve's pass-rate with verifier scores.

    ``task_rewards``: per-task reward (0.0 / 1.0 typically) for every
    task in the search subset. Pass-rate = mean of values.

    ``assessments``: per-task ``TrajectoryAssessment`` from the verifier.
    The keys (task_ids) should be a subset of ``task_rewards`` keys —
    we don't enforce that, but verifier-mean only uses keys present.

    ``alpha``: blending weight, in [0, 1]. 0 = ignore verifier (pure
    self_evolve), 1 = ignore pass-rate (pure verifier).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    pass_rate = (
        sum(task_rewards.values()) / len(task_rewards)
        if task_rewards
        else 0.0
    )

    if assessments:
        verifier_mean = sum(
            a.aggregated_score for a in assessments.values()
        ) / len(assessments)
    else:
        verifier_mean = 0.0

    components = FitnessComponents(
        pass_rate=pass_rate,
        verifier_mean=verifier_mean,
        n_assessed=len(assessments),
    )
    return FitnessScore(
        candidate_id=candidate_id, components=components, alpha=alpha
    )


# ─── Helpers for self_evolve integration ──────────────────────────────────


@dataclass
class FitnessRunner:
    """Compute fitness for a candidate by running the verifier on its
    trajectories.

    Encapsulates the (verifier, alpha, trajectory_loader) triple so the
    self_evolve orchestrator only needs to call ``run_for_candidate``.
    """

    verifier: Verifier
    alpha: float = DEFAULT_VERIFIER_WEIGHT
    """Verifier-blending weight (default 0.3)."""

    skip_failed_trajectories: bool = True
    """If True, only score trajectories whose pass/fail reward was ≥1.0.
    Failed trajectories' verifier scores are mostly informative for the
    proposer, not for ranking — and the API budget is dominated by these
    calls. Set False to score everything (more expensive)."""

    def run_for_candidate(
        self,
        candidate_id: str,
        *,
        task_rewards: Mapping[str, float],
        trajectories: Mapping[str, TrajectoryInput],
    ) -> FitnessScore:
        """Score one candidate's trajectories and blend into a fitness.

        ``trajectories`` maps task_id → ``TrajectoryInput`` (caller is
        responsible for loading transcripts from disk).

        Trajectories not in ``trajectories`` are scored solely via
        pass-rate (no verifier contribution). This is the common path
        when ``skip_failed_trajectories=True`` — failed tasks don't get a
        verifier score, so they only count toward pass-rate.
        """
        assessments: dict[str, TrajectoryAssessment] = {}
        for task_id, input_ in trajectories.items():
            if (
                self.skip_failed_trajectories
                and task_rewards.get(task_id, 0.0) < 1.0
            ):
                continue
            assessments[task_id] = self.verifier.score(input_)

        return compute_fitness(
            candidate_id,
            task_rewards=task_rewards,
            assessments=assessments,
            alpha=self.alpha,
        )


__all__ = [
    "DEFAULT_VERIFIER_WEIGHT",
    "FitnessComponents",
    "FitnessScore",
    "FitnessRunner",
    "compute_fitness",
]
