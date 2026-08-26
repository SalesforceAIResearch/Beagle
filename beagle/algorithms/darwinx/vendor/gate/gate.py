"""Atelier gate — orchestrate verification layers into one decision per node.

A self_evolve campaign produces nodes (candidates). Each node has:
- a parent commit + a list of commits introduced by the iteration
- a numerical score from the final eval
- a path to the worktree where the diff lives

Atelier wraps the post-final-eval pipeline with the following layered
verification, ordered cheap → expensive (so we fail fast):

    Layer 4 ── scope filter (cheap, static)         atelier.scope_filter
    Layer 5 ── reward-hacking honeypot              atelier.honeypot
    Layer 6 ── cross-model transfer (smoke)         atelier.transfer  [week 2]
    Layer 7 ── cross-benchmark transfer (smoke)     atelier.transfer  [week 2]
    verifier ─ trajectory soundness on TB-2 trials  atelier.verifier  [week 2]

Each layer returns its own decision (accept / reject + reason). The gate
aggregates them: a node is promoted iff every layer accepts it.

Modes are independent per layer. Typically:
- Layer 4 starts in SOFT_FLAG for measurement, then flips to STRICT_REJECT
  once we've confirmed the apples-to-apples surface is not too restrictive.
- Layer 5 starts in MEASURE for calibration, then flips to GATE.
- Layers 6/7 + verifier start in MEASURE while we develop the implementations.

The orchestrator does not invoke layers it has no implementation for yet
(week-2 layers are skipped with `status="not_yet_implemented"`); the
node's overall decision is the AND of whatever ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from . import honeypot, scope_filter


# Note: the L5 PredictionGate* classes (predicted_impact / jaccard credibility
# falsification) were removed when the MatchFixGate equivalence approach
# replaced the stats-based gate. See atelier/matchfix_gate.py for the
# replacement that prevents regressions BEFORE final-eval rather than
# scoring them after.
# atelier.predictions module is retained for analytics (it still gets
# populated when DARWINX_GATE_PREDICTIONS_ENABLED=1) but no longer feeds the gate.


class LayerStatus(Enum):
    """The result of one layer's check, with explicit handling for layers
    that haven't been built yet."""

    ACCEPT = "accept"
    REJECT = "reject"
    NOT_YET_IMPLEMENTED = "not_yet_implemented"
    SKIPPED = "skipped"
    """Layer was explicitly disabled for this run."""


@dataclass(frozen=True)
class LayerResult:
    """One layer's contribution to the overall gate decision."""

    name: str
    """Short identifier, e.g. 'scope', 'honeypot', 'cross_model'."""

    status: LayerStatus

    summary: str
    """One-line human-readable summary (logged + included in report)."""

    payload: Any = None
    """Layer-specific structured result (ScopeDecision, HoneypotDecision,
    etc.) — preserved for diagnosis and report rendering."""


@dataclass(frozen=True)
class GateDecision:
    """Aggregate decision for one node, across all enabled Atelier layers."""

    node_id: str

    accept: bool
    """True iff every implemented layer that ran returned ACCEPT.
    NOT_YET_IMPLEMENTED and SKIPPED do not block acceptance."""

    layers: tuple[LayerResult, ...]
    """Per-layer results in the order they ran."""

    @property
    def reject_reasons(self) -> list[str]:
        return [
            f"{l.name}: {l.summary}"
            for l in self.layers
            if l.status is LayerStatus.REJECT
        ]

    def to_summary(self) -> str:
        verdict = "ACCEPT" if self.accept else "REJECT"
        per_layer = " | ".join(
            f"{l.name}:{l.status.value}" for l in self.layers
        )
        return f"atelier-gate[{self.node_id}] {verdict} — {per_layer}"


# ─── Layer-implementation protocols ───────────────────────────────────────


class TransferGate(Protocol):
    """Cross-model or cross-benchmark gate (week-2 deliverable)."""

    name: str

    def __call__(self, *, node_id: str, candidate_id: str) -> LayerResult: ...


class VerifierGate(Protocol):
    """Trajectory verifier gate (week-2 deliverable)."""

    name: str

    def __call__(self, *, node_id: str, candidate_id: str) -> LayerResult: ...


# ─── Gate orchestrator ────────────────────────────────────────────────────


@dataclass
class AtelierGate:
    """Compose Atelier's verification layers into one promote-or-reject call.

    Wiring details:
    - The scope filter receives the diff text or a list of changed file
      paths via ``diff_supplier``.
    - The honeypot receives a runner that scores the candidate against the
      configured task subset. Baseline is passed in directly because it
      doesn't change between candidates (compute once per campaign).
    - Layers not yet implemented are recorded as NOT_YET_IMPLEMENTED and
      do not block acceptance.
    """

    scope_mode: scope_filter.ScopeMode = scope_filter.ScopeMode.SOFT_FLAG
    honeypot_cfg: honeypot.HoneypotConfig | None = None
    """If None, layer 5 is skipped (use for measurement runs without TW
    deployed)."""
    cross_model_gate: TransferGate | None = None
    cross_benchmark_gate: TransferGate | None = None
    verifier_gate: VerifierGate | None = None

    skip_layers: tuple[str, ...] = field(default_factory=tuple)
    """Override: layers in this set are recorded as SKIPPED."""

    # ─── Layer wrappers ──────────────────────────────────────────────────

    def _run_scope_layer(self, *, diff_files: list[str]) -> LayerResult:
        violations = scope_filter.scan_paths(diff_files)
        decision = scope_filter.decide(violations, mode=self.scope_mode)
        status = (
            LayerStatus.ACCEPT if decision.accept else LayerStatus.REJECT
        )
        return LayerResult(
            name="scope",
            status=status,
            summary=decision.to_summary(),
            payload=decision,
        )

    def _run_honeypot_layer(
        self,
        *,
        candidate_id: str,
        baseline: honeypot.HoneypotResult,
        runner: honeypot.HoneypotRunner,
        task_ids: tuple[str, ...],
    ) -> LayerResult:
        if self.honeypot_cfg is None:
            return LayerResult(
                name="honeypot",
                status=LayerStatus.SKIPPED,
                summary="honeypot: skipped (no config)",
                payload=None,
            )

        candidate_result = honeypot.score_candidate(
            candidate_id, task_ids=task_ids, runner=runner
        )
        delta = honeypot.compute_delta(
            candidate=candidate_result, baseline=baseline
        )
        decision = honeypot.decide(delta, cfg=self.honeypot_cfg)
        status = (
            LayerStatus.ACCEPT if decision.accept else LayerStatus.REJECT
        )
        return LayerResult(
            name="honeypot",
            status=status,
            summary=decision.to_summary(),
            payload=decision,
        )

    # ─── Run the full gate ──────────────────────────────────────────────

    def evaluate(
        self,
        *,
        node_id: str,
        candidate_id: str,
        diff_files: list[str],
        # Honeypot integration (None to skip).
        honeypot_runner: honeypot.HoneypotRunner | None = None,
        honeypot_baseline: honeypot.HoneypotResult | None = None,
        honeypot_task_ids: tuple[str, ...] = (),
    ) -> GateDecision:
        """Run every implemented + enabled layer and return the aggregate.

        Layers are evaluated in cheap → expensive order. We don't
        short-circuit on the first REJECT — every layer always runs so we
        can attribute rejections to specific layers in the campaign report.
        """
        layers: list[LayerResult] = []

        # ─── Layer 4: scope ──
        if "scope" in self.skip_layers:
            layers.append(
                LayerResult(
                    name="scope",
                    status=LayerStatus.SKIPPED,
                    summary="scope: skipped (configured)",
                )
            )
        else:
            layers.append(self._run_scope_layer(diff_files=diff_files))

        # ─── Layer 5: honeypot ──
        if "honeypot" in self.skip_layers:
            layers.append(
                LayerResult(
                    name="honeypot",
                    status=LayerStatus.SKIPPED,
                    summary="honeypot: skipped (configured)",
                )
            )
        elif (
            self.honeypot_cfg is None
            or honeypot_runner is None
            or honeypot_baseline is None
        ):
            layers.append(
                LayerResult(
                    name="honeypot",
                    status=LayerStatus.SKIPPED,
                    summary=(
                        "honeypot: skipped (cfg / runner / baseline missing)"
                    ),
                )
            )
        else:
            layers.append(
                self._run_honeypot_layer(
                    candidate_id=candidate_id,
                    baseline=honeypot_baseline,
                    runner=honeypot_runner,
                    task_ids=honeypot_task_ids,
                )
            )

        # ─── Layer 6: cross-model (week-2) ──
        if self.cross_model_gate is not None and "cross_model" not in self.skip_layers:
            layers.append(
                self.cross_model_gate(
                    node_id=node_id, candidate_id=candidate_id
                )
            )
        else:
            layers.append(
                LayerResult(
                    name="cross_model",
                    status=(
                        LayerStatus.SKIPPED
                        if "cross_model" in self.skip_layers
                        else LayerStatus.NOT_YET_IMPLEMENTED
                    ),
                    summary="cross_model: pending (week-2)",
                )
            )

        # ─── Layer 7: cross-benchmark (week-2) ──
        if (
            self.cross_benchmark_gate is not None
            and "cross_benchmark" not in self.skip_layers
        ):
            layers.append(
                self.cross_benchmark_gate(
                    node_id=node_id, candidate_id=candidate_id
                )
            )
        else:
            layers.append(
                LayerResult(
                    name="cross_benchmark",
                    status=(
                        LayerStatus.SKIPPED
                        if "cross_benchmark" in self.skip_layers
                        else LayerStatus.NOT_YET_IMPLEMENTED
                    ),
                    summary="cross_benchmark: pending (week-2)",
                )
            )

        # ─── Verifier (week-2) ──
        if self.verifier_gate is not None and "verifier" not in self.skip_layers:
            layers.append(
                self.verifier_gate(
                    node_id=node_id, candidate_id=candidate_id
                )
            )
        else:
            layers.append(
                LayerResult(
                    name="verifier",
                    status=(
                        LayerStatus.SKIPPED
                        if "verifier" in self.skip_layers
                        else LayerStatus.NOT_YET_IMPLEMENTED
                    ),
                    summary="verifier: pending (week-2)",
                )
            )

        accept = all(l.status is not LayerStatus.REJECT for l in layers)
        return GateDecision(
            node_id=node_id, accept=accept, layers=tuple(layers)
        )


__all__ = [
    "LayerStatus",
    "LayerResult",
    "GateDecision",
    "TransferGate",
    "VerifierGate",
    "AtelierGate",
]
