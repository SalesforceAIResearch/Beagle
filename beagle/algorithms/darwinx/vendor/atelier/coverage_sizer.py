"""LLM-driven adaptive K-sizer for the equivalence gate probes (v8).

v5's MatchFixGate used a fixed K=6 probe size. Empirically this gave
12% coverage of `parent.solved` and produced a catastrophic false-
negative in v7 (candidate `c22ff235`: 6/6 probes passed, final-eval
Δ=-0.315, 28 of 45 unprobed tasks regressed).

This module asks the LLM (one focused call) to size K based on the
semantic analyzer's risk_level + modified_surfaces:

  • additive-only edits (`bundled_skills`, new sub_agents): K=4
  • single-tool edits: K=8-12
  • core-control-loop / evidence_classifier edits: K=18-25
  • unknown surfaces: default K (caller's default, usually 6)

The sized K is then handed to the existing probe-selection +
execution stages. No other gate logic changes.

Failure modes (all degrade to caller's default K):
  • LLM call error
  • JSON parse failure
  • LLM returns K outside [K_MIN, K_MAX]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .matchfix_gate import LLMBackend, SemanticAnalysis


__all__ = [
    "CoverageRecommendation",
    "K_MIN",
    "K_MAX",
    "recommend_k",
]


_LOG = logging.getLogger("atelier.coverage_sizer")


K_MIN = 4
K_MAX = 30


@dataclass(frozen=True)
class CoverageRecommendation:
    """LLM's recommendation for the equivalence gate's probe K."""

    k: int
    """The recommended probe count, already clamped to [K_MIN, K_MAX]
    and bounded above by parent_solved_count."""

    rationale: str
    """One-sentence rubric-row match."""

    raw_response: str = ""
    """Full LLM reply for debugging."""

    is_fallback: bool = False
    """True iff the call failed and ``k`` is the caller's default."""


_SIZER_SCHEMA_HINT = """Reply with EXACTLY one JSON object (no prose outside it):

```
{
  "k": <integer in [K_MIN, K_MAX]>,
  "rationale": "<one-sentence rubric-row match>"
}
```"""


def recommend_k(
    *,
    semantic_analysis: SemanticAnalysis,
    parent_solved_count: int,
    default_k: int,
    llm: LLMBackend,
    prompt_template: str,
) -> CoverageRecommendation:
    """Ask the LLM to size K. Never raises.

    Always returns a usable recommendation:
      • If LLM succeeds: clamp recommendation to
        [K_MIN, min(K_MAX, parent_solved_count)]
      • If LLM fails: return ``default_k`` (also clamped) with
        ``is_fallback=True``

    The caller can opt out of LLM-sizing entirely by setting
    ``ATELIER_EQUIVALENCE_PROBE_K_FIXED=1`` upstream and just using
    the env-configured K.
    """
    effective_max = max(K_MIN, min(K_MAX, parent_solved_count or K_MAX))

    # If pool too small for meaningful K, just use everything.
    if parent_solved_count is not None and parent_solved_count <= K_MIN:
        return CoverageRecommendation(
            k=max(1, parent_solved_count),
            rationale=(
                f"parent_solved_count={parent_solved_count} <= K_MIN; "
                f"probe all solved tasks"
            ),
            is_fallback=False,
        )

    surfaces = ", ".join(semantic_analysis.modified_surfaces) or "(none)"
    prompt = (
        prompt_template
        .replace("{modified_surfaces}", surfaces)
        .replace("{risk_level}", semantic_analysis.risk_level)
        .replace(
            "{analysis_rationale}",
            semantic_analysis.rationale or "(none provided)",
        )
        .replace("{parent_solved_count}", str(parent_solved_count))
        .replace("{default_k}", str(default_k))
        .replace("{k_min}", str(K_MIN))
        .replace("{k_max}", str(effective_max))
    )
    prompt = prompt + "\n\n" + _SIZER_SCHEMA_HINT

    try:
        raw = llm.complete(prompt=prompt)
    except Exception as e:  # noqa: BLE001
        _LOG.warning(
            "coverage sizer LLM call failed (%s) — falling back to "
            "default K=%d", e, default_k,
        )
        return CoverageRecommendation(
            k=max(K_MIN, min(effective_max, default_k)),
            rationale=f"LLM error: {e}",
            is_fallback=True,
        )

    parsed = _extract_json_object(raw)
    if parsed is None:
        _LOG.warning(
            "coverage sizer: could not parse JSON (raw=%r) — fallback K=%d",
            raw[:200], default_k,
        )
        return CoverageRecommendation(
            k=max(K_MIN, min(effective_max, default_k)),
            rationale="JSON parse failure",
            raw_response=raw,
            is_fallback=True,
        )

    try:
        k_raw = int(parsed.get("k", default_k))
    except (TypeError, ValueError):
        k_raw = default_k

    # Clamp into the allowed window. parent_solved_count caps from above.
    k_clamped = max(K_MIN, min(effective_max, k_raw))
    rationale = str(parsed.get("rationale", "")).strip() or "(no rationale)"
    if k_clamped != k_raw:
        rationale = (
            f"{rationale} [clamped from LLM-requested {k_raw} to {k_clamped}]"
        )

    _LOG.info(
        "coverage sizer: surfaces=[%s] risk=%s → K=%d (pool=%d). %s",
        surfaces, semantic_analysis.risk_level, k_clamped,
        parent_solved_count, rationale[:160],
    )
    return CoverageRecommendation(
        k=k_clamped,
        rationale=rationale,
        raw_response=raw,
    )


# ─── Helpers ────────────────────────────────────────────────────────────────


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    for r in (_JSON_FENCE_RE, _FIRST_OBJ_RE):
        m = r.search(raw)
        if m:
            try:
                obj = json.loads(m.group(1) if r is _JSON_FENCE_RE else m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None
