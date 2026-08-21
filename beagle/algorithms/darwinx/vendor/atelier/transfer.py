"""Layers 6 & 7 — cross-model and cross-benchmark transfer gates.

Atelier promotes a candidate from "passed self_evolve's eval" to
"upstream into monet_code" only if it also survives *transfer* checks:

- **Cross-model transfer (Layer 6)**: the same candidate run with a
  different model family (e.g., search used Opus → check with Sonnet
  or Gemini) should not regress more than θ pp on a small task subset.
  If it does, the candidate likely overfits to its proposer's model.

- **Cross-benchmark transfer (Layer 7)**: the same candidate run on a
  smoke slice of a different benchmark (e.g., SWE-bench Verified) should
  not regress more than θ pp vs that benchmark's baseline. If it does,
  the candidate likely overfits to TB-2's task distribution.

Both gates share a generic comparison shape — pass rate of candidate vs
pass rate of baseline on the same task subset, with a regression
threshold. They differ only in *which* runner produces the pass rate
(different model family vs. different benchmark substrate).

This module implements the shared shape (``TransferGate``) plus two
typed configurations (``CROSS_MODEL_DEFAULT``, ``CROSS_BENCHMARK_DEFAULT``)
so calls sites read self-documenting. The actual Harbor / harness wiring
(how to run a candidate against an arbitrary task subset on an arbitrary
model) is delegated to ``TransferEvaluator`` Protocols, just as in
``honeypot.py``.

Relationship to ``honeypot.py``:

- Honeypot: candidate's pass-rate **rising** above baseline is bad
  (signal: learned to game).
- Transfer: candidate's pass-rate **falling** below baseline is bad
  (signal: overfit to its training distribution).

Both modules share the dataclasses' shape (``TaskResult``, ``Result``,
``Delta``); they're named differently here to keep the import space clean
and the dataflow self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class TransferMode(Enum):
    """Whether transfer regressions gate or just get logged."""

    MEASURE = "measure"
    GATE = "gate"


# ─── Results & deltas ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransferTaskResult:
    task_id: str
    passed: bool
    reward: float


@dataclass(frozen=True)
class TransferResult:
    """Aggregate result on the transfer task subset."""

    candidate_id: str
    """The candidate being evaluated."""

    eval_name: str
    """Identifier for this transfer eval (e.g., 'cross_model_sonnet',
    'cross_benchmark_swebench')."""

    task_results: tuple[TransferTaskResult, ...]

    @property
    def n_tasks(self) -> int:
        return len(self.task_results)

    @property
    def pass_rate(self) -> float:
        if self.n_tasks == 0:
            return 0.0
        return sum(1 for t in self.task_results if t.passed) / self.n_tasks


@dataclass(frozen=True)
class TransferDelta:
    """Candidate minus baseline on the transfer task subset."""

    candidate: TransferResult
    baseline: TransferResult

    @property
    def pass_rate_delta(self) -> float:
        return self.candidate.pass_rate - self.baseline.pass_rate

    @property
    def n_regressed(self) -> int:
        """Tasks the baseline passed but the candidate fails (the
        regression signal)."""
        baseline_passes = {
            t.task_id for t in self.baseline.task_results if t.passed
        }
        candidate_passes = {
            t.task_id for t in self.candidate.task_results if t.passed
        }
        return len(baseline_passes - candidate_passes)


# ─── Config ───────────────────────────────────────────────────────────────


DEFAULT_REGRESSION_THRESHOLD_PASS_RATE = -0.02
"""Reject if candidate's pass-rate is more than 2pp BELOW baseline. The
threshold is signed (negative = regression magnitude tolerated)."""

DEFAULT_REGRESSION_THRESHOLD_REGRESSED_COUNT = 3
"""Reject if candidate newly fails ≥this many tasks the baseline passed."""


@dataclass
class TransferConfig:
    """One transfer gate's configuration (cross-model or cross-benchmark)."""

    eval_name: str
    """Identifier for the gate (e.g., 'cross_model_sonnet')."""

    mode: TransferMode = TransferMode.MEASURE

    threshold_pass_rate_delta: float = DEFAULT_REGRESSION_THRESHOLD_PASS_RATE
    threshold_regressed: int = DEFAULT_REGRESSION_THRESHOLD_REGRESSED_COUNT

    task_ids: tuple[str, ...] = field(default_factory=tuple)
    """Task IDs to evaluate. Specific to the eval (TB-2 task names for
    cross-model; SWE-bench-Verified slice for cross-benchmark)."""

    max_tasks: int = 20
    """Cap on tasks per candidate; small to keep transfer-eval cost low."""


# Pre-built configurations callers can import for self-documenting code.
CROSS_MODEL_DEFAULT = TransferConfig(
    eval_name="cross_model",
    mode=TransferMode.MEASURE,
    max_tasks=20,
)

CROSS_BENCHMARK_DEFAULT = TransferConfig(
    eval_name="cross_benchmark",
    mode=TransferMode.MEASURE,
    max_tasks=10,  # smaller — SWE-bench slice is just smoke-level
)


# ─── Evaluator protocol ───────────────────────────────────────────────────


class TransferEvaluator(Protocol):
    """A callable that evaluates (candidate, task) on the transfer substrate.

    Substrate-specific (cross-model: same TB-2 task on a different model;
    cross-benchmark: a SWE-bench task with monet_code at the candidate's
    commit). Adapters live alongside the relevant Harbor configs; this
    module only owns the comparison logic.
    """

    def __call__(
        self, *, candidate_id: str, task_id: str
    ) -> TransferTaskResult: ...


# ─── Decision ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransferDecision:
    accept: bool
    delta: TransferDelta
    mode: TransferMode
    eval_name: str
    reason: str = ""

    def to_summary(self) -> str:
        prefix = f"transfer[{self.eval_name}]:"
        d = self.delta
        if self.accept:
            return (
                f"{prefix} accept ({self.mode.value}) — "
                f"Δ pass-rate {d.pass_rate_delta:+.3f}, "
                f"{d.n_regressed} newly-regressed"
            )
        return (
            f"{prefix} reject ({self.mode.value}) — "
            f"{self.reason}; Δ pass-rate {d.pass_rate_delta:+.3f}, "
            f"{d.n_regressed} newly-regressed"
        )


# ─── Scoring + decide ────────────────────────────────────────────────────


def score_candidate(
    candidate_id: str,
    *,
    config: TransferConfig,
    evaluator: TransferEvaluator,
) -> TransferResult:
    """Run a candidate on the transfer task subset via the configured
    evaluator. Errors propagate (mirrors honeypot.score_candidate)."""
    results: list[TransferTaskResult] = []
    for task_id in config.task_ids:
        r = evaluator(candidate_id=candidate_id, task_id=task_id)
        results.append(r)
    return TransferResult(
        candidate_id=candidate_id,
        eval_name=config.eval_name,
        task_results=tuple(results),
    )


def compute_delta(
    *, candidate: TransferResult, baseline: TransferResult
) -> TransferDelta:
    if candidate.eval_name != baseline.eval_name:
        raise ValueError(
            "transfer.compute_delta: candidate.eval_name "
            f"({candidate.eval_name!r}) != baseline.eval_name "
            f"({baseline.eval_name!r})"
        )
    return TransferDelta(candidate=candidate, baseline=baseline)


def decide(
    delta: TransferDelta, *, config: TransferConfig
) -> TransferDecision:
    """Apply the configured mode + thresholds to a delta.

    - MEASURE: always accept; reason empty.
    - GATE: reject if delta.pass_rate_delta < threshold_pass_rate_delta
      (note: thresholds are SIGNED — negative-more-negative = bigger
      regression = reject) OR n_regressed > threshold_regressed.
    """
    if config.mode is TransferMode.MEASURE:
        return TransferDecision(
            accept=True, delta=delta, mode=config.mode, eval_name=config.eval_name
        )

    # GATE
    if delta.pass_rate_delta < config.threshold_pass_rate_delta:
        return TransferDecision(
            accept=False,
            delta=delta,
            mode=config.mode,
            eval_name=config.eval_name,
            reason=(
                f"pass-rate delta {delta.pass_rate_delta:+.3f} < "
                f"threshold {config.threshold_pass_rate_delta:+.3f}"
            ),
        )
    if delta.n_regressed > config.threshold_regressed:
        return TransferDecision(
            accept=False,
            delta=delta,
            mode=config.mode,
            eval_name=config.eval_name,
            reason=(
                f"newly-regressed count {delta.n_regressed} > "
                f"threshold {config.threshold_regressed}"
            ),
        )
    return TransferDecision(
        accept=True, delta=delta, mode=config.mode, eval_name=config.eval_name
    )


# ─── Gate adapter (to plug into atelier.gate.AtelierGate) ────────────────


@dataclass
class TransferGateAdapter:
    """Adapts a transfer eval into the ``atelier.gate.TransferGate`` Protocol.

    Pass an instance of this to ``AtelierGate.cross_model_gate`` /
    ``cross_benchmark_gate`` to wire the transfer check into the
    orchestrator. Requires a baseline pre-computed once per campaign.
    """

    config: TransferConfig
    evaluator: TransferEvaluator
    baseline: TransferResult

    @property
    def name(self) -> str:
        return self.config.eval_name

    def __call__(
        self, *, node_id: str, candidate_id: str
    ) -> "LayerResult":
        from .gate import LayerResult, LayerStatus

        result = score_candidate(
            candidate_id, config=self.config, evaluator=self.evaluator
        )
        delta = compute_delta(candidate=result, baseline=self.baseline)
        decision = decide(delta, config=self.config)
        status = LayerStatus.ACCEPT if decision.accept else LayerStatus.REJECT
        return LayerResult(
            name=self.config.eval_name,
            status=status,
            summary=decision.to_summary(),
            payload=decision,
        )


__all__ = [
    "TransferMode",
    "TransferTaskResult",
    "TransferResult",
    "TransferDelta",
    "TransferConfig",
    "TransferEvaluator",
    "TransferDecision",
    "TransferGateAdapter",
    "CROSS_MODEL_DEFAULT",
    "CROSS_BENCHMARK_DEFAULT",
    "DEFAULT_REGRESSION_THRESHOLD_PASS_RATE",
    "DEFAULT_REGRESSION_THRESHOLD_REGRESSED_COUNT",
    "score_candidate",
    "compute_delta",
    "decide",
]
