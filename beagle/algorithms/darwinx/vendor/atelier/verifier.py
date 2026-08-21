"""LLM-as-a-Verifier — trajectory scoring via criteria decomposition,
repeated verification, and score-token granularity.

Used in two places inside Atelier:

1. ``best_of_n.py``: as a ``VerifierScorer`` that compares candidate
   trajectories at test time and selects the best per task via
   round-robin tournament.
2. ``verifier_fitness.py``: as a fine-grained fitness signal for
   ``self_evolve``'s parent selection, replacing binary pass / fail with
   continuous trajectory quality.

The algorithm (per the LLM-as-a-Verifier paper, arXiv:2509.16187 style):

- **Criteria decomposition (C criteria)**: split "is this trajectory
  good?" into multiple sub-questions, score each independently, then
  aggregate. A single weighted-mean over criteria is the default; we
  preserve the per-criterion vector so callers can use other
  aggregations (geometric mean, product, etc.).
- **Repeated verification (K runs)**: re-score each criterion K times to
  damp the LLM's noise floor. Average across K.
- **Score granularity (G score tokens)**: the model produces a single
  digit response (1–9). We capture the *probability distribution* over
  those tokens (via the model's top-K logprobs) and compute the
  expected value rather than picking the argmax. With G=9 we get
  effectively 9 distinct score levels per call instead of the binary
  "is this a 7 or an 8?" question the LLM would naively answer.

Cost-wise, the default (C=5, K=3, G=9) is 15 LLM calls per trajectory.
At ~0.001 per call for a small frontier model, that's ~$0.015 per
trajectory; a Best-of-4 over the 89 TB-2 tasks is ~$5 to score.

This module owns the algorithm. The actual logprob-returning model call
is delegated to a ``VerifierBackend`` Protocol; the OpenAI-compatible
implementation lives in ``verifier_backend_openai.py`` (next commit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


# ─── Score tokens ─────────────────────────────────────────────────────────


DEFAULT_SCORE_TOKENS: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)
"""Single-digit tokens, all single-token in every common BPE tokenizer
(unlike "10" which splits). G=9 gives 9 distinct expected-value levels
across [1, 9]."""


# ─── Criterion + distribution + score ─────────────────────────────────────


@dataclass(frozen=True)
class Criterion:
    """One axis along which to evaluate a trajectory.

    The ``description`` is rendered into the verifier prompt to tell the
    model what to focus on. The ``name`` is a short identifier used in
    logs + the ``TrajectoryAssessment`` record.

    ``weight`` is used by ``TrajectoryAssessment.aggregated_score`` to
    take a weighted mean across criteria; default 1.0 (uniform).
    """

    name: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class ScoreDistribution:
    """Probability distribution over the score tokens for one model call.

    ``probs`` is parallel to the ``score_tokens`` tuple — ``probs[i]``
    is the probability the model picked token ``score_tokens[i]``. The
    distribution is normalized so it sums to ~1.0 (the backend may discard
    probability mass on tokens outside ``score_tokens``).
    """

    probs: tuple[float, ...]
    score_values: tuple[int, ...]
    """Numeric value of each score token (e.g., (1, 2, …, 9))."""

    def __post_init__(self) -> None:
        if len(self.probs) != len(self.score_values):
            raise ValueError(
                f"probs ({len(self.probs)}) and score_values "
                f"({len(self.score_values)}) length mismatch"
            )

    @property
    def expected_value(self) -> float:
        """E[score] under the distribution, in the raw score range."""
        return sum(v * p for v, p in zip(self.score_values, self.probs))

    @property
    def normalized(self) -> float:
        """E[score] mapped to [0, 1] using min/max of score_values."""
        if not self.score_values:
            return 0.0
        lo, hi = min(self.score_values), max(self.score_values)
        if hi == lo:
            return 1.0  # degenerate; pick anything in [0,1]
        return (self.expected_value - lo) / (hi - lo)


@dataclass(frozen=True)
class CriterionScore:
    """Aggregated score for one criterion across K verification repeats."""

    criterion: Criterion
    distributions: tuple[ScoreDistribution, ...]

    @property
    def n_repeats(self) -> int:
        return len(self.distributions)

    @property
    def mean_normalized(self) -> float:
        """Mean of per-repeat normalized expected values, in [0, 1]."""
        if not self.distributions:
            return 0.0
        return sum(d.normalized for d in self.distributions) / len(
            self.distributions
        )

    @property
    def std_normalized(self) -> float:
        """Sample standard deviation across K repeats. Useful for
        confidence reporting; ~0 when the model is consistent."""
        n = len(self.distributions)
        if n < 2:
            return 0.0
        m = self.mean_normalized
        return math.sqrt(
            sum((d.normalized - m) ** 2 for d in self.distributions)
            / (n - 1)
        )


@dataclass(frozen=True)
class TrajectoryAssessment:
    """A full multi-criterion assessment for one trajectory."""

    trajectory_id: str
    """Caller-supplied identifier (e.g., a Trial ID)."""

    criterion_scores: tuple[CriterionScore, ...]

    @property
    def aggregated_score(self) -> float:
        """Weighted mean across criteria, in [0, 1].

        Default aggregation; callers can also access individual criterion
        scores for alternative aggregations (geometric mean, hard-floor
        minimum, etc.).
        """
        if not self.criterion_scores:
            return 0.0
        total_weight = sum(cs.criterion.weight for cs in self.criterion_scores)
        if total_weight <= 0:
            return 0.0
        return (
            sum(
                cs.criterion.weight * cs.mean_normalized
                for cs in self.criterion_scores
            )
            / total_weight
        )

    @property
    def min_score(self) -> float:
        """Lowest per-criterion score. Useful as a conservative aggregation
        ('the trajectory is only as strong as its weakest criterion')."""
        if not self.criterion_scores:
            return 0.0
        return min(cs.mean_normalized for cs in self.criterion_scores)


# ─── Trajectory input ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrajectoryInput:
    """The data needed to score one trajectory.

    Atelier doesn't take a position on transcript formatting — the caller
    constructs this from a self_evolve trial directory (or wherever the
    trajectory is recorded) and we hand it off to the verifier as is.
    """

    trajectory_id: str
    task_id: str
    task_instruction: str
    """The task statement the agent was asked to solve."""

    transcript: str
    """Rendered transcript of the agent's actions + outputs (e.g.,
    self_evolve's ``agent/transcript.md``)."""

    workspace_summary: str = ""
    """Optional: short text summary of the final workspace state (e.g.,
    `ls -la`, key file contents). Helps the verifier check the agent's
    output was correct without re-running."""

    extra: dict = field(default_factory=dict)
    """Free-form metadata available to the prompt template."""


# ─── Backend protocol ─────────────────────────────────────────────────────


class VerifierBackend(Protocol):
    """A callable that scores one prompt and returns a probability
    distribution over the score tokens.

    Implementations are responsible for:

    1. Sending the prompt to a model whose API exposes top-K logprobs
       (OpenAI chat completions, or any OpenAI-compatible endpoint;
       Claude does NOT expose logprobs, so a Claude-only backend is
       not supported).
    2. Locating the digit-token logprobs in the top-K response.
    3. Normalizing the discovered logprobs into a probability
       distribution over ``score_tokens``. Tokens absent from the
       top-K should get probability 0 (or a small floor) before
       normalization.
    """

    def score(
        self, *, prompt: str, score_tokens: tuple[str, ...]
    ) -> ScoreDistribution: ...


# ─── Prompt template ─────────────────────────────────────────────────────


DEFAULT_PROMPT_TEMPLATE = """You are an expert evaluator of LLM-based coding agents on terminal-based tasks.

# Task the agent was asked to solve
{task_instruction}

# Agent's full trajectory (transcript of actions and outputs)
{transcript}

# Final workspace state
{workspace_summary}

# Evaluation criterion
{criterion_name}: {criterion_description}

# Your job
Rate the agent's performance on THIS single criterion on a 1–9 scale, where:
- 1 = catastrophic failure on this criterion
- 5 = neutral / acceptable
- 9 = exemplary

Respond with ONLY the single digit (1, 2, 3, 4, 5, 6, 7, 8, or 9). No
other words, no punctuation. Your entire response must be exactly one
character long."""


# Self-Grounded Verification (SGV) variant — mitigates LLM "agreement bias"
# (over-validating agent behaviour; arXiv/OpenReview SGV) by forcing the judge
# to commit to its OWN priors about a correct trajectory BEFORE it reads/judges
# the candidate, and to reason step-wise + adversarially. Same {placeholders}
# and same single-digit 1–9 output as DEFAULT_PROMPT_TEMPLATE, so the consensus/
# scoring/fitness path is unchanged — only the judgment is hardened.
SGV_PROMPT_TEMPLATE = """You are a SKEPTICAL expert evaluator of LLM-based coding agents on terminal-based tasks. Your default stance is doubt: agents frequently appear to succeed while actually failing or succeeding by luck. Do not over-credit plausible-looking behaviour.

# Task the agent was asked to solve
{task_instruction}

# Evaluation criterion
{criterion_name}: {criterion_description}

# Step 1 — Priors FIRST (before reading the transcript)
Ignoring the agent's transcript entirely, think about what a GENUINELY strong
trajectory for THIS task would necessarily contain on this criterion: the
concrete steps/checks/evidence that must be present, and the tell-tale signs of
a fake, lucky, or incomplete success. Hold these priors fixed.

# Step 2 — Now read the agent's trajectory and final workspace
## Agent's full trajectory (transcript of actions and outputs)
{transcript}

## Final workspace state
{workspace_summary}

# Step 3 — Adversarial, step-wise judgment
Walk the trajectory and check it AGAINST your Step-1 priors: did the agent
actually perform the necessary steps/checks (cite where), or is the apparent
success unsupported / coincidental / unverified? Actively look for the
lucky-or-fake explanation before accepting success.

# Your job
Rate the agent on THIS single criterion on a 1–9 scale, where:
- 1 = catastrophic failure on this criterion
- 5 = neutral / acceptable
- 9 = exemplary AND your priors are concretely satisfied with cited evidence
Penalize unsupported/coincidental success: if the priors are not demonstrably
met in the trajectory, do not score above 5.

Respond with ONLY the single digit (1, 2, 3, 4, 5, 6, 7, 8, or 9). No
other words, no punctuation. Your entire response must be exactly one
character long."""


# ─── Default criteria ─────────────────────────────────────────────────────


DEFAULT_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        name="task_completion",
        description=(
            "Did the agent successfully complete the task as specified? "
            "Specifically: did it produce the artifacts the task asked for "
            "(files, command output, state changes) in the form requested?"
        ),
    ),
    Criterion(
        name="solution_correctness",
        description=(
            "Is the agent's solution genuinely correct, or did it pass via "
            "shortcuts (writing exact expected output, monkey-patching the "
            "test harness, exploiting verifier weaknesses, copying answers "
            "from elsewhere in the workspace)? Penalize gaming heavily."
        ),
    ),
    Criterion(
        name="verification",
        description=(
            "Did the agent verify its own work before declaring done? "
            "E.g., re-running the test, inspecting the output, checking "
            "edge cases. An agent that confidently declares done without "
            "verification is low on this criterion even if it succeeded."
        ),
    ),
    Criterion(
        name="cleanup",
        description=(
            "Did the agent leave the workspace in the expected state? "
            "Specifically: stray build artifacts, temp files, debugging "
            "remnants, partial edits — these are penalties on byte-strict "
            "graders even if the underlying solution was correct."
        ),
    ),
    Criterion(
        name="efficiency",
        description=(
            "Did the agent reach the solution using a reasonable number "
            "of turns and tool calls? Heavy thrashing — looping on the "
            "same error, exploring the same file many times — signals "
            "either a capability gap or a workflow that won't generalize."
        ),
    ),
)


TB2_EXTRA_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        name="error_recovery",
        description=(
            "When intermediate steps failed (tool returning non-zero, "
            "file write rejected, build error, test failure mid-run), "
            "did the agent diagnose the error and recover, or did it "
            "give up / repeat the same failing call / declare done "
            "anyway? TB-2 rewards persistence and self-correction; an "
            "agent that pivots to a workable approach after an error "
            "scores higher than one that ignores the failure or "
            "thrashes on it."
        ),
        weight=1.2,
    ),
    Criterion(
        name="task_interpretation",
        description=(
            "Did the agent correctly interpret what the task was asking "
            "for, or did it solve a slightly different (often easier) "
            "problem? Watch for: skipping requirements (e.g. 'must "
            "handle edge case X' ignored), partial coverage of a "
            "multi-part spec, misreading 'find all' as 'find any', "
            "incorrect output format despite computing the right value. "
            "TB-2 graders are strict about specification compliance."
        ),
        weight=1.1,
    ),
)
"""Two extra criteria targeting failure modes specific to TB-2:
error recovery (key for long-running terminal tasks) and task
interpretation (key for spec-heavy graders). Slightly upweighted
because these correlate strongly with pass/fail on TB-2 in
exploratory eval. Used alongside ``DEFAULT_CRITERIA`` via
``TB2_CRITERIA`` (or explicitly per-call)."""


TB2_CRITERIA: tuple[Criterion, ...] = DEFAULT_CRITERIA + TB2_EXTRA_CRITERIA
"""TB-2-tuned criteria set. ~7 criteria × K=3 = ~21 calls per
trajectory, ~$0.021 per trajectory with gpt-5.4-mini. Best-of-4 on
TB-2's 89 tasks ≈ $7.50 to score one candidate. Default verifier
trajectory cost was 5 criteria × K=3 = 15 calls ($5.30 for the same
candidate), so the TB-2 profile adds ~$2 in scoring spend per
campaign final-eval."""


def criteria_for_profile(profile: str) -> tuple[Criterion, ...]:
    """Look up a criteria tuple by profile name.

    Profiles:
      - ``default`` (5 generic criteria — original LLM-as-Verifier set)
      - ``tb2`` (5 generic + 2 TB-2-specific = 7 criteria)

    Unknown profile → falls back to ``default`` with a logged warning.
    Used by ``verifier_fitness.compute_fitness`` to plumb env-controlled
    criteria selection (``ATELIER_VERIFIER_CRITERIA_PROFILE``).
    """
    p = profile.strip().lower()
    if p == "default":
        return DEFAULT_CRITERIA
    if p == "tb2":
        return TB2_CRITERIA
    import logging
    logging.getLogger("atelier.verifier").warning(
        "unknown criteria profile %r; falling back to 'default'", profile
    )
    return DEFAULT_CRITERIA


# ─── Verifier ─────────────────────────────────────────────────────────────


@dataclass
class Verifier:
    """Compose criteria + a backend + repeat count into a callable scorer.

    The default config (C=5, K=3, G=9) implements the
    LLM-as-a-Verifier algorithm from arXiv:2509.16187 with the
    refinements described in the module docstring.
    """

    backend: VerifierBackend
    criteria: tuple[Criterion, ...] = DEFAULT_CRITERIA
    k_repeats: int = 3
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    score_tokens: tuple[str, ...] = DEFAULT_SCORE_TOKENS

    def __post_init__(self) -> None:
        if self.k_repeats < 1:
            raise ValueError(f"k_repeats must be >= 1, got {self.k_repeats}")
        if not self.criteria:
            raise ValueError("at least one criterion is required")
        if not self.score_tokens:
            raise ValueError("at least one score token is required")

    # ─── Prompt construction ────────────────────────────────────────────

    def _build_prompt(
        self, trajectory: TrajectoryInput, criterion: Criterion
    ) -> str:
        return self.prompt_template.format(
            task_instruction=trajectory.task_instruction,
            transcript=trajectory.transcript,
            workspace_summary=trajectory.workspace_summary or "(none)",
            criterion_name=criterion.name,
            criterion_description=criterion.description,
        )

    # ─── Per-criterion + full scoring ────────────────────────────────────

    def score_one_criterion(
        self, trajectory: TrajectoryInput, criterion: Criterion
    ) -> CriterionScore:
        """Score one trajectory on one criterion, K times."""
        distributions: list[ScoreDistribution] = []
        for _ in range(self.k_repeats):
            prompt = self._build_prompt(trajectory, criterion)
            dist = self.backend.score(
                prompt=prompt, score_tokens=self.score_tokens
            )
            distributions.append(dist)
        return CriterionScore(
            criterion=criterion, distributions=tuple(distributions)
        )

    def score(self, trajectory: TrajectoryInput) -> TrajectoryAssessment:
        """Run the full verifier: every criterion × K repeats."""
        scores: list[CriterionScore] = []
        for criterion in self.criteria:
            scores.append(self.score_one_criterion(trajectory, criterion))
        return TrajectoryAssessment(
            trajectory_id=trajectory.trajectory_id,
            criterion_scores=tuple(scores),
        )

    # ─── best_of_n.VerifierScorer adapter ──────────────────────────────

    def to_scorer(self):
        """Adapt the verifier to the simpler ``best_of_n.VerifierScorer``
        signature: ``(trajectory) -> float``.

        Used to plug the verifier into ``best_of_n.run_best_of_n``.
        The ``best_of_n.Trajectory`` carries trial info; the caller is
        responsible for loading a ``TrajectoryInput`` from that and
        invoking the verifier. We keep this thin to make wiring obvious.
        """

        def scorer(*, trajectory) -> float:
            # The best_of_n Trajectory holds task_id + reward + index + extra.
            # The caller passes the full TrajectoryInput in extra["input"];
            # falling back to a minimal input lets tests use simple fakes.
            inp = trajectory.extra.get("input") if hasattr(trajectory, "extra") else None
            if not isinstance(inp, TrajectoryInput):
                raise RuntimeError(
                    "to_scorer expected trajectory.extra['input'] to be a "
                    "TrajectoryInput; got " + repr(type(inp).__name__)
                )
            return self.score(inp).aggregated_score

        return scorer


# ─── Helpers for backend implementations ─────────────────────────────────


def logprobs_to_distribution(
    logprob_map: dict[str, float],
    *,
    score_tokens: tuple[str, ...],
    score_values: tuple[int, ...] | None = None,
    missing_logprob: float = -20.0,
) -> ScoreDistribution:
    """Convert a {token: logprob} map (from the model's top-K response)
    into a normalized probability distribution over the score tokens.

    Tokens absent from ``logprob_map`` are treated as if they had
    log-probability ``missing_logprob`` (default = -20, ~2e-9). This
    keeps the distribution proper when the model strongly prefers one
    token (and the others fall outside top-K).

    ``score_values`` defaults to ``(int(t) for t in score_tokens)``;
    pass it explicitly when score tokens aren't digits.
    """
    if score_values is None:
        score_values = tuple(int(t) for t in score_tokens)
    if len(score_values) != len(score_tokens):
        raise ValueError(
            "score_tokens and score_values must have the same length"
        )

    # Pull logprobs (using fallback for missing tokens) and exponentiate.
    raw_probs = []
    for tok in score_tokens:
        lp = logprob_map.get(tok, missing_logprob)
        raw_probs.append(math.exp(lp))

    total = sum(raw_probs)
    if total <= 0:
        # Pathological: model returned no usable logprobs. Use uniform.
        n = len(score_tokens)
        probs = tuple([1.0 / n] * n)
    else:
        probs = tuple(p / total for p in raw_probs)

    return ScoreDistribution(probs=probs, score_values=score_values)


__all__ = [
    "Criterion",
    "ScoreDistribution",
    "CriterionScore",
    "TrajectoryAssessment",
    "TrajectoryInput",
    "VerifierBackend",
    "Verifier",
    "DEFAULT_SCORE_TOKENS",
    "DEFAULT_CRITERIA",
    "DEFAULT_PROMPT_TEMPLATE",
    "SGV_PROMPT_TEMPLATE",
    "logprobs_to_distribution",
]
