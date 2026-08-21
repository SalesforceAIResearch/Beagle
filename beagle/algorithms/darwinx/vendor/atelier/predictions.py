"""Predicted-impact + falsification data model for Atelier-X.

Adopts NexAU-AHE's central methodological contribution: every candidate
must commit a structured **predicted_impact** alongside its diff
(failure_evidence, root_cause, targeted_fix, predicted should_pass /
should_fail / at_risk lists). The next iteration's eval FALSIFIES the
prediction. Chronic over-predictors get a parent-sampling penalty.

This module owns:

1. **Schema**: dataclasses for ``PredictedImpact`` and
   ``PredictionCredibility``.
2. **Parser**: extracts ``predicted_impact:`` YAML block from the
   cursor-agent's review-step text (the natural emission point — the
   review step has just seen the diff + mini-eval and is the right
   place to commit predictions).
3. **Comparator**: given a candidate's predictions + the next
   iteration's actual flips, computes Jaccard accuracy +
   over-prediction rate + per-task call-out.
4. **Storage helpers**: write predictions / credibility to per-node
   JSON sidecars under ``reports/<campaign>/atelier/predictions/``.

The atelier-gate's L5 (``prediction_credibility``) layer consumes
these sidecars; ``parent_selection`` (Zeyuan's existing picker) gets a
new ranking signal ``prediction_credibility_score`` blended into the
archive sampling probability.

Why YAML-block extraction rather than structured tool calls? Because:
- We can't modify cursor-agent's CLI to accept structured output.
- We want the prediction to live alongside the proposer's natural
  reasoning (which is markdown).
- YAML is forgiving on minor formatting variation.
- A regex finds the block; ``yaml.safe_load`` parses it.

If the proposer omits the block, we record an empty prediction and
penalize the parent in L5. No silent degradation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


__all__ = [
    "FailureEvidence",
    "PredictedImpact",
    "PredictionCredibility",
    "parse_predicted_impact",
    "compare_predictions_with_actual",
    "load_predictions",
    "save_predictions",
    "load_credibility",
    "save_credibility",
    "rolling_credibility_score",
]


logger = logging.getLogger("atelier.predictions")


# ─── Schema ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FailureEvidence:
    """One failure that motivated the proposed edit.

    Sourced — the proposer must point at a specific task + trace
    excerpt. Empty ``trace_excerpt`` is allowed (the proposer may
    cite the failure without quoting the trace) but discouraged;
    L5 penalizes proposers who omit evidence.
    """

    task: str
    trace_excerpt: str = ""


@dataclass(frozen=True)
class PredictedImpact:
    """What the proposer claims its candidate will do.

    Adopted from NexAU-AHE's per-edit commit format. Required fields:
    - ``should_pass``: tasks that were failing, candidate claims will pass
    - ``should_fail``: tasks tolerated as still failing (no claim of fix)
    - ``at_risk``: tasks at risk of regression, with mitigation noted
    - ``failure_evidence``: trace-sourced motivation for the edit
    - ``root_cause``: WHY it failed, not just WHAT failed
    - ``targeted_fix``: the concrete edit that addresses the cause
    """

    should_pass: tuple[str, ...] = ()
    should_fail: tuple[str, ...] = ()
    at_risk: tuple[str, ...] = ()
    failure_evidence: tuple[FailureEvidence, ...] = ()
    root_cause: str = ""
    targeted_fix: str = ""

    @property
    def is_empty(self) -> bool:
        """True iff the proposer didn't commit any meaningful prediction.

        Used by L5: empty predictions count as 0 credibility (worse
        than a wrong prediction, since they're an *abdication* of
        falsifiability).
        """
        return (
            not self.should_pass
            and not self.failure_evidence
            and not self.root_cause.strip()
            and not self.targeted_fix.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_pass": list(self.should_pass),
            "should_fail": list(self.should_fail),
            "at_risk": list(self.at_risk),
            "failure_evidence": [asdict(e) for e in self.failure_evidence],
            "root_cause": self.root_cause,
            "targeted_fix": self.targeted_fix,
        }


@dataclass(frozen=True)
class PredictionCredibility:
    """Result of falsifying a candidate's PredictedImpact against
    the next iteration's actual flips.

    Computed per (parent_node_id, child_node_id) pair: the child's
    predictions about which tasks would flip are compared against
    what actually happened when the child was eval'd.
    """

    parent_node_id: str
    child_node_id: str

    predicted_should_pass: tuple[str, ...]
    actually_passed: tuple[str, ...]
    """Tasks the child flipped fail→pass (in the child's solved_tasks
    minus the parent's solved_tasks)."""

    predicted_at_risk: tuple[str, ...]
    actually_regressed: tuple[str, ...]
    """Tasks the child flipped pass→fail (in parent's solved_tasks
    minus child's solved_tasks)."""

    jaccard_accuracy: float
    """|predicted_pass ∩ actually_passed| / |predicted_pass ∪ actually_passed|.
    1.0 = perfect, 0.0 = nothing in common. Defined as 1.0 when both
    sets are empty (no prediction, no flip — vacuously honest)."""

    over_prediction_rate: float
    """|predicted_pass - actually_passed| / max(1, |predicted_pass|).
    Fraction of predicted flips that didn't materialize. High = the
    proposer is making up improvements that don't exist."""

    under_prediction_rate: float
    """|actually_passed - predicted_pass| / max(1, |actually_passed|).
    Fraction of actual flips that the proposer didn't predict. High
    = the proposer is lucky-improving, not deliberately improving."""

    regression_surprise_rate: float
    """|actually_regressed - predicted_at_risk| / max(1, |actually_regressed|).
    Fraction of regressions the proposer failed to anticipate. High
    = the proposer doesn't understand the impact of its own edits."""

    is_empty_prediction: bool
    """True iff the proposer didn't commit a prediction at all.
    Counted as 0 credibility in rolling scores."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_node_id": self.parent_node_id,
            "child_node_id": self.child_node_id,
            "predicted_should_pass": list(self.predicted_should_pass),
            "actually_passed": list(self.actually_passed),
            "predicted_at_risk": list(self.predicted_at_risk),
            "actually_regressed": list(self.actually_regressed),
            "jaccard_accuracy": round(self.jaccard_accuracy, 4),
            "over_prediction_rate": round(self.over_prediction_rate, 4),
            "under_prediction_rate": round(self.under_prediction_rate, 4),
            "regression_surprise_rate": round(self.regression_surprise_rate, 4),
            "is_empty_prediction": self.is_empty_prediction,
        }


# ─── Parser ───────────────────────────────────────────────────────────────


# Match `predicted_impact:` followed by a YAML block.
# The block ends at the first non-indented non-empty line OR at EOF.
#
# We accept it inside a fenced code block (```yaml ... ```) OR as
# a top-level YAML block. The cursor-agent's review-step output is
# markdown so most cases come through fenced.
_BLOCK_RE = re.compile(
    r"(?:```(?:ya?ml)?\s*\n)?"
    r"^predicted_impact\s*:\s*\n"
    r"(?P<body>(?:[ \t]+.*\n|\s*\n)+)"
    r"(?:```\s*\n?)?",
    re.MULTILINE,
)


def parse_predicted_impact(text: str) -> PredictedImpact:
    """Extract a ``predicted_impact:`` YAML block from cursor-agent text.

    Returns ``PredictedImpact()`` (empty) if no block is found or if
    parsing fails. Failures are logged, not raised — L5 handles empty
    predictions as a credibility penalty rather than a hard error.
    """
    if not text:
        return PredictedImpact()

    match = _BLOCK_RE.search(text)
    if not match:
        return PredictedImpact()

    body = match.group("body")
    # Strip the leading indentation so yaml.safe_load sees a proper mapping.
    lines = body.splitlines()
    # Find the minimum non-blank-line indentation.
    indents = [
        len(line) - len(line.lstrip(" "))
        for line in lines if line.strip()
    ]
    min_indent = min(indents) if indents else 0
    stripped = "\n".join(
        line[min_indent:] if line.strip() else "" for line in lines
    )

    try:
        import yaml  # lazy import — atelier shouldn't hard-depend on yaml
    except ImportError:
        logger.warning("yaml package not available; cannot parse predicted_impact")
        return PredictedImpact()

    try:
        parsed = yaml.safe_load(stripped)
    except yaml.YAMLError as e:
        logger.warning("predicted_impact YAML parse failed: %s", e)
        return PredictedImpact()

    if not isinstance(parsed, dict):
        return PredictedImpact()

    def _as_tuple(v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            return (v,) if v.strip() else ()
        if isinstance(v, (list, tuple)):
            return tuple(str(x) for x in v if x is not None and str(x).strip())
        return ()

    def _as_failure_evidence(v: Any) -> tuple[FailureEvidence, ...]:
        if v is None:
            return ()
        if not isinstance(v, list):
            return ()
        out = []
        for item in v:
            if isinstance(item, dict):
                task = str(item.get("task", "")).strip()
                excerpt = str(item.get("trace_excerpt", "")).strip()
                if task:
                    out.append(FailureEvidence(task=task, trace_excerpt=excerpt))
            elif isinstance(item, str):
                # bare task name, no excerpt
                if item.strip():
                    out.append(FailureEvidence(task=item.strip()))
        return tuple(out)

    return PredictedImpact(
        should_pass=_as_tuple(parsed.get("should_pass")),
        should_fail=_as_tuple(parsed.get("should_fail")),
        at_risk=_as_tuple(parsed.get("at_risk")),
        failure_evidence=_as_failure_evidence(parsed.get("failure_evidence")),
        root_cause=str(parsed.get("root_cause", "") or "").strip(),
        targeted_fix=str(parsed.get("targeted_fix", "") or "").strip(),
    )


# ─── Comparator ──────────────────────────────────────────────────────────


def compare_predictions_with_actual(
    *,
    parent_node_id: str,
    child_node_id: str,
    predicted: PredictedImpact,
    parent_solved: Iterable[str],
    child_solved: Iterable[str],
) -> PredictionCredibility:
    """Falsify a candidate's predictions against the next iteration's eval.

    ``parent_solved`` + ``child_solved`` are the sets of tasks that the
    parent and child nodes solved in their final-eval (typically
    drawn from ``Node.solved_tasks_json``).

    Computes Jaccard accuracy, over-prediction rate, under-prediction
    rate, and regression surprise rate. Empty predictions get
    ``is_empty_prediction=True`` and zeros across the board (L5 treats
    this as 0 credibility).
    """
    parent_set = set(parent_solved)
    child_set = set(child_solved)

    actually_passed = sorted(child_set - parent_set)
    actually_regressed = sorted(parent_set - child_set)

    predicted_pass = set(predicted.should_pass)
    predicted_at_risk = set(predicted.at_risk)
    actually_passed_set = set(actually_passed)
    actually_regressed_set = set(actually_regressed)

    is_empty = predicted.is_empty

    # Jaccard accuracy on should_pass ↔ actually_passed.
    if not predicted_pass and not actually_passed_set:
        jaccard = 1.0   # vacuously honest: no claim, no flip
    else:
        union = predicted_pass | actually_passed_set
        inter = predicted_pass & actually_passed_set
        jaccard = len(inter) / max(1, len(union))

    over = (
        len(predicted_pass - actually_passed_set) / max(1, len(predicted_pass))
        if predicted_pass else 0.0
    )
    under = (
        len(actually_passed_set - predicted_pass) / max(1, len(actually_passed_set))
        if actually_passed_set else 0.0
    )
    surprise = (
        len(actually_regressed_set - predicted_at_risk) / max(1, len(actually_regressed_set))
        if actually_regressed_set else 0.0
    )

    return PredictionCredibility(
        parent_node_id=parent_node_id,
        child_node_id=child_node_id,
        predicted_should_pass=tuple(sorted(predicted_pass)),
        actually_passed=tuple(actually_passed),
        predicted_at_risk=tuple(sorted(predicted_at_risk)),
        actually_regressed=tuple(actually_regressed),
        jaccard_accuracy=jaccard,
        over_prediction_rate=over,
        under_prediction_rate=under,
        regression_surprise_rate=surprise,
        is_empty_prediction=is_empty,
    )


# ─── Rolling per-parent credibility score ────────────────────────────────


def rolling_credibility_score(
    credibilities: Iterable[PredictionCredibility],
    *,
    window: int = 5,
    empty_penalty: float = 0.0,
) -> float:
    """Compute a rolling per-parent credibility score.

    Takes the LAST ``window`` credibility records for one parent
    (chronologically ordered) and computes the mean jaccard accuracy.
    Empty predictions contribute ``empty_penalty`` (default 0.0)
    rather than being skipped — empty predictions ARE evidence of
    proposer abdication.

    Returns a float in [0, 1]. Used by L5 to penalize chronic
    over-predictors in parent sampling.
    """
    creds = list(credibilities)[-window:]
    if not creds:
        return 1.0   # no data → neutral; don't penalize a fresh parent
    scores = [
        empty_penalty if c.is_empty_prediction else c.jaccard_accuracy
        for c in creds
    ]
    return sum(scores) / len(scores)


# ─── Storage helpers ─────────────────────────────────────────────────────


def _predictions_path(
    reports_root: Path, campaign: str, child_node_id: str
) -> Path:
    """Latest-prediction sidecar path. Backward-compatible — readers
    that want "the most recent prediction" go through this path."""
    return (
        Path(reports_root)
        / campaign
        / "atelier"
        / "predictions"
        / f"{child_node_id}.predicted.json"
    )


def _per_iter_predictions_path(
    reports_root: Path, campaign: str, child_node_id: str, iteration: int,
) -> Path:
    """Per-iteration prediction sidecar path. Preserves the FULL history
    of what the proposer claimed in iter 1, iter 2, ..., so we can
    compute credibility against the BEST per-iter prediction rather than
    the latest (which is often degraded — see ab_overnight_3 lesson).
    """
    return (
        Path(reports_root)
        / campaign
        / "atelier"
        / "predictions"
        / f"{child_node_id}.iter_{iteration:02d}.predicted.json"
    )


def _credibility_path(
    reports_root: Path, campaign: str, child_node_id: str
) -> Path:
    """Sidecar path for the falsification result (predicted vs actual)."""
    return (
        Path(reports_root)
        / campaign
        / "atelier"
        / "predictions"
        / f"{child_node_id}.credibility.json"
    )


def save_predictions(
    *,
    reports_root: Path,
    campaign: str,
    child_node_id: str,
    predicted: PredictedImpact,
    iteration: int | None = None,
) -> Path:
    """Persist the parsed prediction.

    If ``iteration`` is provided, also writes a per-iteration sidecar
    at ``<node>.iter_NN.predicted.json`` alongside the
    backward-compatible ``<node>.predicted.json`` (which always holds
    the LATEST prediction for the simple reader).

    Why both: today's lesson from `ab_overnight_3_e2_atelier` showed
    the proposer often makes its BEST prediction in iter 1 and
    degrades over later iters. Preserving the full per-iter history
    lets `load_best_prediction()` find the maximally-honest prediction
    across the lineage rather than just the last one.
    """
    path = _predictions_path(reports_root, campaign, child_node_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = predicted.to_dict()
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)

    # Per-iter sidecar (only when iteration is known — e.g., during the
    # pipeline's review step). Other callers (offline analysis) can omit.
    if iteration is not None and iteration >= 0:
        iter_path = _per_iter_predictions_path(
            reports_root, campaign, child_node_id, iteration,
        )
        iter_tmp = iter_path.with_suffix(".json.tmp")
        iter_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        iter_tmp.replace(iter_path)

    return path


def load_per_iter_predictions(
    *,
    reports_root: Path,
    campaign: str,
    child_node_id: str,
) -> list[tuple[int, "PredictedImpact"]]:
    """Load every per-iter prediction sidecar for a node.

    Returns a list of ``(iteration, PredictedImpact)`` pairs sorted by
    iteration. Empty list if no per-iter sidecars exist (e.g., older
    runs that pre-date the per-iter feature).
    """
    pred_dir = Path(reports_root) / campaign / "atelier" / "predictions"
    if not pred_dir.is_dir():
        return []
    pattern = f"{child_node_id}.iter_*.predicted.json"
    out: list[tuple[int, PredictedImpact]] = []
    for path in sorted(pred_dir.glob(pattern)):
        # File name: <node>.iter_NN.predicted.json — extract NN
        m = re.search(r"\.iter_(\d+)\.predicted\.json$", path.name)
        if not m:
            continue
        iteration = int(m.group(1))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pred = PredictedImpact(
            should_pass=tuple(data.get("should_pass") or ()),
            should_fail=tuple(data.get("should_fail") or ()),
            at_risk=tuple(data.get("at_risk") or ()),
            failure_evidence=tuple(
                FailureEvidence(**fe) for fe in (data.get("failure_evidence") or [])
            ),
            root_cause=data.get("root_cause", ""),
            targeted_fix=data.get("targeted_fix", ""),
        )
        out.append((iteration, pred))
    return sorted(out, key=lambda t: t[0])


def load_best_prediction(
    *,
    reports_root: Path,
    campaign: str,
    child_node_id: str,
    parent_solved: Iterable[str],
    child_solved: Iterable[str],
) -> tuple[int, "PredictedImpact", "PredictionCredibility"] | None:
    """Find the per-iter prediction with the HIGHEST jaccard accuracy
    against the eventual actual flips. Returns ``None`` if no per-iter
    sidecars exist.

    This is the "max-of-history" credibility — the proposer's best
    moment across iterations. Use alongside the latest-prediction
    credibility for a fuller picture: a lineage with high best-of-iter
    accuracy but low latest-iter accuracy is *degrading*, which is a
    distinct failure mode from "consistently inaccurate".
    """
    per_iter = load_per_iter_predictions(
        reports_root=reports_root, campaign=campaign,
        child_node_id=child_node_id,
    )
    if not per_iter:
        return None
    best: tuple[int, PredictedImpact, PredictionCredibility] | None = None
    for iteration, pred in per_iter:
        cred = compare_predictions_with_actual(
            parent_node_id="__per_iter__",   # unused for this comparison
            child_node_id=child_node_id,
            predicted=pred,
            parent_solved=parent_solved,
            child_solved=child_solved,
        )
        if best is None or cred.jaccard_accuracy > best[2].jaccard_accuracy:
            best = (iteration, pred, cred)
    return best


def load_predictions(
    *,
    reports_root: Path,
    campaign: str,
    child_node_id: str,
) -> PredictedImpact | None:
    """Reload a previously-saved prediction. Returns ``None`` if missing."""
    path = _predictions_path(reports_root, campaign, child_node_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return PredictedImpact(
        should_pass=tuple(data.get("should_pass") or ()),
        should_fail=tuple(data.get("should_fail") or ()),
        at_risk=tuple(data.get("at_risk") or ()),
        failure_evidence=tuple(
            FailureEvidence(**fe) for fe in (data.get("failure_evidence") or [])
        ),
        root_cause=data.get("root_cause", ""),
        targeted_fix=data.get("targeted_fix", ""),
    )


def save_credibility(
    *,
    reports_root: Path,
    campaign: str,
    child_node_id: str,
    credibility: PredictionCredibility,
) -> Path:
    """Persist the falsification result."""
    path = _credibility_path(reports_root, campaign, child_node_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(credibility.to_dict(), indent=2), encoding="utf-8"
    )
    tmp.replace(path)
    return path


def load_credibility(
    *,
    reports_root: Path,
    campaign: str,
    child_node_id: str,
) -> PredictionCredibility | None:
    """Reload a credibility record. Returns ``None`` if missing."""
    path = _credibility_path(reports_root, campaign, child_node_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return PredictionCredibility(
            parent_node_id=data["parent_node_id"],
            child_node_id=data["child_node_id"],
            predicted_should_pass=tuple(data.get("predicted_should_pass") or ()),
            actually_passed=tuple(data.get("actually_passed") or ()),
            predicted_at_risk=tuple(data.get("predicted_at_risk") or ()),
            actually_regressed=tuple(data.get("actually_regressed") or ()),
            jaccard_accuracy=float(data.get("jaccard_accuracy", 0.0)),
            over_prediction_rate=float(data.get("over_prediction_rate", 0.0)),
            under_prediction_rate=float(data.get("under_prediction_rate", 0.0)),
            regression_surprise_rate=float(data.get("regression_surprise_rate", 0.0)),
            is_empty_prediction=bool(data.get("is_empty_prediction", False)),
        )
    except (KeyError, TypeError):
        return None
