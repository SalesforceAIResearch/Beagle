"""Test-time Best-of-N trajectory selection via verifier tournament.

After a candidate has been promoted by Atelier (passed scope, honeypot, and
transfer gates), we can further lift its measured pass rate at *test time*
by sampling multiple trajectories per task and selecting the best one with
an LLM-as-a-Verifier tournament.

The selection procedure (per the LLM-as-a-Verifier paper):

1. Sample ``N`` trajectories per task with temperature > 0 to get diversity.
2. For each pair of trajectories on a task, score both via the verifier
   (which decomposes the eval into multiple criteria, runs K repeated
   verifications, and aggregates across G score-token granularities).
3. Run a round-robin tournament — each trajectory plays every other once;
   the trajectory with the most pairwise wins on a task is selected.
4. Use the selected trajectory's reward as the task's score.

This module owns the **tournament + selection** layer. The actual scoring
function — score(trajectory) → real — is plugged in via the
``VerifierScorer`` protocol; the concrete LLM-as-a-Verifier implementation
lives in ``atelier/verifier.py`` (week 2 deliverable).

The Harbor wiring for **sampling** N trajectories (re-running Monet+candidate
with different seeds / temperatures) is provided by the caller, also via a
``TrajectorySampler`` protocol.

This separation means the tournament logic is independently unit-testable
with simple fake samplers and scorers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol


# ─── Trajectory dataclass ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Trajectory:
    """One sampled trajectory for one task.

    Atelier doesn't care about the trajectory's internal structure — it
    just needs a stable identifier so the scorer can be cached, a reward
    (for downstream aggregation), and the per-task identifier.
    """

    task_id: str
    """The task this trajectory is for."""

    sample_index: int
    """0-based index of this sample among the N drawn for this task."""

    reward: float
    """The verifier's pass/fail reward (0.0 / 1.0 or partial)."""

    trial_dir: str | None = None
    """Optional path to the trial directory for diagnosis."""

    extra: Mapping[str, object] = field(default_factory=dict)
    """Free-form metadata, e.g. trajectory length, token counts."""


# ─── Sampler + Scorer protocols ───────────────────────────────────────────


class TrajectorySampler(Protocol):
    """Callable that samples N trajectories for a single task."""

    def __call__(
        self, *, task_id: str, candidate_id: str, n: int
    ) -> list[Trajectory]:
        ...


class VerifierScorer(Protocol):
    """Callable that returns a verifier score for one trajectory.

    The real implementation runs LLM-as-a-Verifier; the protocol is kept
    minimal so unit tests can swap in a deterministic fake.
    """

    def __call__(self, *, trajectory: Trajectory) -> float:
        ...


# ─── Tournament selection ────────────────────────────────────────────────


@dataclass(frozen=True)
class TournamentRecord:
    """The full per-task tournament outcome (kept for diagnosis)."""

    task_id: str
    """The task this tournament was run for."""

    trajectories: tuple[Trajectory, ...]
    """All N candidates ordered by ``sample_index``."""

    scores: tuple[float, ...]
    """Verifier score for each trajectory (parallel to ``trajectories``)."""

    wins: tuple[int, ...]
    """Pairwise wins per trajectory (parallel to ``trajectories``)."""

    selected_index: int
    """Index into ``trajectories`` of the chosen trajectory."""

    @property
    def selected(self) -> Trajectory:
        return self.trajectories[self.selected_index]


def round_robin_tournament(
    trajectories: list[Trajectory],
    scorer: VerifierScorer,
) -> TournamentRecord:
    """Score every trajectory, run pairwise round-robin, return record.

    Selection rule: trajectory with the most pairwise wins.

    Tiebreaks (in order):
    1. higher verifier score (raw value),
    2. higher reward,
    3. lower sample_index.

    Pairwise win condition: ``score(a) > score(b)``. Equal-score pairs are
    a draw — both get +0.5 wins; this softens ties without changing
    expectation. Numerical equality is broken by trajectories' sample
    indices for full determinism.
    """
    if not trajectories:
        raise ValueError("round_robin_tournament: no trajectories to score")

    task_ids = {t.task_id for t in trajectories}
    if len(task_ids) != 1:
        raise ValueError(
            f"round_robin_tournament: trajectories must share a task_id; "
            f"got {sorted(task_ids)}"
        )
    [task_id] = task_ids

    n = len(trajectories)
    scores = [scorer(trajectory=t) for t in trajectories]
    wins: list[float] = [0.0 for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = scores[i], scores[j]
            if si > sj:
                wins[i] += 1.0
            elif sj > si:
                wins[j] += 1.0
            else:
                wins[i] += 0.5
                wins[j] += 0.5

    # Pick the winner with tiebreaks.
    def sort_key(idx: int) -> tuple:
        return (
            -wins[idx],
            -scores[idx],
            -trajectories[idx].reward,
            trajectories[idx].sample_index,
        )

    order = sorted(range(n), key=sort_key)
    selected_index = order[0]

    return TournamentRecord(
        task_id=task_id,
        trajectories=tuple(trajectories),
        scores=tuple(scores),
        wins=tuple(int(w) if w.is_integer() else w for w in wins),  # type: ignore[union-attr]
        selected_index=selected_index,
    )


# ─── Best-of-N driver ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class BestOfNResult:
    """Aggregate result across all tasks evaluated with Best-of-N."""

    candidate_id: str
    n: int
    """The N used (samples per task)."""

    records: tuple[TournamentRecord, ...]
    """Per-task tournament records, one per task."""

    @property
    def selected_trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(r.selected for r in self.records)

    @property
    def n_tasks(self) -> int:
        return len(self.records)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.records if r.selected.reward >= 1.0)

    @property
    def pass_rate(self) -> float:
        if self.n_tasks == 0:
            return 0.0
        return self.n_passed / self.n_tasks

    @property
    def mean_selected_reward(self) -> float:
        if self.n_tasks == 0:
            return 0.0
        return sum(r.selected.reward for r in self.records) / self.n_tasks


def run_best_of_n(
    candidate_id: str,
    *,
    task_ids: list[str],
    n: int,
    sampler: TrajectorySampler,
    scorer: VerifierScorer,
) -> BestOfNResult:
    """Run the full Best-of-N selection over a list of tasks.

    For each task:
      1. Sample ``n`` trajectories via ``sampler``.
      2. Run round-robin tournament with ``scorer``.
      3. Record the selected trajectory + full tournament for diagnosis.

    No early termination — every task is evaluated. This is a test-time
    scaling pass, not a search.
    """
    if n < 1:
        raise ValueError(f"Best-of-N: n must be >= 1, got {n}")

    records: list[TournamentRecord] = []
    for task_id in task_ids:
        trajectories = sampler(
            task_id=task_id, candidate_id=candidate_id, n=n
        )
        if len(trajectories) != n:
            raise RuntimeError(
                f"sampler returned {len(trajectories)} trajectories for "
                f"task {task_id!r}; expected exactly n={n}. The sampler "
                "should retry transient failures internally; if it can't "
                "produce n trajectories the candidate's eval is unreliable."
            )
        rec = round_robin_tournament(trajectories, scorer)
        records.append(rec)

    return BestOfNResult(
        candidate_id=candidate_id, n=n, records=tuple(records)
    )


# ─── Lift vs Pass@1 ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BestOfNLift:
    """Δ between the candidate's Pass@1 baseline and its Best-of-N rate."""

    candidate_id: str
    pass_at_1_rate: float
    """Standard single-trajectory pass rate (typically already known from
    self_evolve's final-eval step)."""

    best_of_n_rate: float
    """Pass rate using verifier-tournament selection over n trajectories."""

    n: int
    """The N used."""

    @property
    def absolute_lift(self) -> float:
        return self.best_of_n_rate - self.pass_at_1_rate

    @property
    def relative_lift(self) -> float:
        """As a fraction of the Pass@1 baseline. Inf when Pass@1 is 0."""
        if self.pass_at_1_rate == 0.0:
            return float("inf") if self.best_of_n_rate > 0.0 else 0.0
        return self.absolute_lift / self.pass_at_1_rate

    def to_summary(self) -> str:
        return (
            f"best-of-n: {self.pass_at_1_rate:.3f} (Pass@1) → "
            f"{self.best_of_n_rate:.3f} (Best-of-{self.n}); "
            f"Δ {self.absolute_lift:+.3f} ({self.relative_lift:+.1%})"
        )


__all__ = [
    "Trajectory",
    "TrajectorySampler",
    "VerifierScorer",
    "TournamentRecord",
    "BestOfNResult",
    "BestOfNLift",
    "round_robin_tournament",
    "run_best_of_n",
]
