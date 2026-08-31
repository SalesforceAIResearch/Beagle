"""LLM-driven parent selector (v8 replacement for PerformanceNovelty math).

v7's ``PerformanceNoveltyStrategy`` ranked parents by
``score × √novelty`` over binary task-success vectors. In practice the
cosine-novelty metric confounded "valuable diversity" with "different
solved set because the candidate broke things", so a catastrophic
regressor (score 0.258, lost 28 tasks) was picked as parent by three
downstream pipelines.

The v8 design swaps that brittle math for one focused LLM call per
pipeline spawn:

  • Input: a list of ``ArchiveCard`` records (one per eligible parent).
    Each card carries the signals an experienced human would weigh:
    score, score-delta vs root, gate verdict, modified surfaces,
    improved / regressed tasks, child count, and "is this lineage
    safe (>= root.score)?".
  • Output: ``{ "picked_node_id": "...", "rationale": "..." }``.
  • The LLM can SEE that c9428a06 regressed and skip it; it can ALSO
    see "root has been picked 4 times, sibling X has 0 children and
    might be worth exploring" and trade off accordingly.

The contract: this module ONLY makes a recommendation. The
``LLMParentSelectionStrategy`` integration in ``parent_selection.py``
validates the LLM's pick against the candidate set and falls back to
``MixedHighScoreStrategy`` on any failure (LLM down, bad JSON,
node_id not in the set). This module is judgment-not-truth: the
gate, the final-eval, and the strategy contract are the real safety
net.

Mirrors the MatchFixAgent pattern (arXiv 2509.16187): one focused
LLM call with structured input + structured output, anchored to
observable signals (scores, verdicts).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .matchfix_gate import LLMBackend


__all__ = [
    "ArchiveCard",
    "ParentPickResult",
    "build_archive_card",
    "render_archive_cards_for_prompt",
    "select_parent_with_llm",
]


_LOG = logging.getLogger("atelier.llm_parent_selector")


# ─── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchiveCard:
    """Compact record the LLM picker reasons over for one candidate parent.

    Fields are deliberately the signals a thoughtful human would weigh —
    not just the raw score — so the LLM can distinguish "novel because
    exploring" from "novel because broke things".
    """

    node_id: str
    score: float
    parent_node_id: str | None

    score_delta_vs_root: float
    """``node.score - root.score``. Positive means improver, negative
    means regressor."""

    n_children_already: int
    """How many descendants this node already has in the archive.
    High values signal an over-explored branch."""

    modified_surfaces: tuple[str, ...]
    """From the equivalence verdict's semantic_analysis. Empty when
    the gate didn't produce a verdict (e.g., very early node)."""

    gate_verdict: str
    """One of ``"EQUIVALENT"`` / ``"MODIFIED"`` / ``"INCONCLUSIVE"`` /
    ``"unknown"``. Comes from the equivalence sidecar."""

    gate_accept: bool
    """Whether the gate accepted this candidate for final-eval."""

    improved_tasks: tuple[str, ...]
    """Tasks this node newly solved (vs its parent). Capped to ~8 for
    prompt size."""

    regressed_tasks: tuple[str, ...]
    """Tasks this node broke (vs its parent). Capped to ~8."""

    is_safe_lineage: bool
    """``score >= root_score`` — useful at-a-glance flag for the LLM."""

    iter_count: int = 0
    """Iteration depth from root (root has iter_count=0)."""


@dataclass(frozen=True)
class ParentPickResult:
    """Output of ``select_parent_with_llm``.

    ``picked_node_id`` is None when the LLM call failed or returned an
    invalid pick (the caller should fall back to a deterministic
    strategy in that case).
    """

    picked_node_id: str | None
    rationale: str
    """LLM's explanation of WHY this node was picked. Surfaced in the
    sidecar for human auditing + retrospective analysis."""

    raw_response: str = ""
    """The full LLM reply for debugging parser drift. Not serialized
    to the sidecar (often >2 KB)."""

    fallback_reason: str | None = None
    """When ``picked_node_id is None``, why the fallback path is
    needed: ``"empty_archive"`` / ``"llm_error"`` /
    ``"parse_error"`` / ``"invalid_pick"``."""


# ─── ArchiveCard construction ───────────────────────────────────────────────


def build_archive_card(
    *,
    node_id: str,
    score: float | None,
    parent_node_id: str | None,
    root_score: float,
    n_children_already: int,
    modified_surfaces: Sequence[str] = (),
    gate_verdict: str = "unknown",
    gate_accept: bool = False,
    improved_tasks: Sequence[str] = (),
    regressed_tasks: Sequence[str] = (),
    iter_count: int = 0,
) -> ArchiveCard:
    """Assemble a card from the tree row + sidecar reads.

    The caller (the strategy adapter in ``parent_selection.py``) is
    responsible for the sidecar reads + the iter_count walk —
    keeping this module purely functional + easy to test.
    """
    s = float(score) if score is not None else 0.0
    return ArchiveCard(
        node_id=str(node_id),
        score=s,
        parent_node_id=parent_node_id,
        score_delta_vs_root=s - float(root_score),
        n_children_already=int(n_children_already),
        modified_surfaces=tuple(modified_surfaces),
        gate_verdict=str(gate_verdict),
        gate_accept=bool(gate_accept),
        improved_tasks=tuple(improved_tasks)[:8],
        regressed_tasks=tuple(regressed_tasks)[:8],
        is_safe_lineage=s >= float(root_score),
        iter_count=int(iter_count),
    )


# ─── Prompt rendering ──────────────────────────────────────────────────────


def render_archive_cards_for_prompt(
    cards: Sequence[ArchiveCard], *, root_score: float,
) -> str:
    """Build the markdown block that the LLM picker reasons over.

    Each card is rendered as a compact YAML-ish entry; the picker's
    prompt template surrounds this with instructions.
    """
    if not cards:
        return "_(archive is empty)_"
    lines: list[str] = []
    for c in cards:
        delta = (
            f"{c.score_delta_vs_root:+.3f}"
            if c.score_delta_vs_root is not None else "—"
        )
        safe = "safe" if c.is_safe_lineage else "DEGRADED"
        verdict = c.gate_verdict
        accept = "ACCEPT" if c.gate_accept else "REJECT"
        surfaces = ",".join(c.modified_surfaces) or "(none recorded)"
        improved = ", ".join(c.improved_tasks[:5]) or "—"
        regressed = ", ".join(c.regressed_tasks[:5]) or "—"
        lines.append(
            f"- node_id: `{c.node_id}`\n"
            f"  score: {c.score:.3f} (Δ_vs_root={delta}, {safe})\n"
            f"  gate: {verdict} ({accept})\n"
            f"  iter_depth: {c.iter_count}, n_children_already: {c.n_children_already}\n"
            f"  modified_surfaces: {surfaces}\n"
            f"  improved_tasks: {improved}\n"
            f"  regressed_tasks: {regressed}"
        )
    return "\n\n".join(lines)


# ─── Selection ──────────────────────────────────────────────────────────────


_PICKER_SCHEMA_HINT = """Reply with EXACTLY one JSON object (no prose outside it):

```
{
  "picked_node_id": "<node_id from the candidate list above>",
  "rationale": "<one paragraph: why this candidate, why not the alternatives, what direction you expect the next iter to go>"
}
```

The picked_node_id MUST be one of the node_ids listed above (no inventing new ones)."""


def select_parent_with_llm(
    *,
    cards: Sequence[ArchiveCard],
    root_score: float,
    n_pipelines_remaining: int,
    llm: LLMBackend,
    prompt_template: str,
    top_k_cap: int = 10,
) -> ParentPickResult:
    """Call the LLM picker. Pure judgment; never raises.

    Args:
      cards: ordered list of eligible parent candidates. The strategy
        adapter is responsible for fetching + ordering — typically by
        score descending, then truncated to ``top_k_cap``.
      root_score: the campaign root's final-eval score. Used in the
        prompt header for context ("you're trying to outperform 0.573").
      n_pipelines_remaining: how many more steps the campaign will
        spawn. Affects the exploration-exploitation balance the LLM
        should adopt (early = explore more; late = exploit best).
      llm: the LLMBackend implementation.
      prompt_template: rendered with `{archive_cards}`, `{root_score}`,
        `{n_pipelines_remaining}` placeholders.
      top_k_cap: cap on candidates shown to the LLM (default 10).
        Beyond this, lower-scoring candidates are dropped before
        rendering. Keeps the prompt bounded.
    """
    if not cards:
        return ParentPickResult(
            picked_node_id=None,
            rationale="archive is empty — no eligible parent",
            fallback_reason="empty_archive",
        )

    # Cap candidates to keep prompt size bounded.
    shown = list(cards)[:top_k_cap]
    candidates_block = render_archive_cards_for_prompt(
        shown, root_score=root_score,
    )

    prompt = (
        prompt_template
        .replace("{archive_cards}", candidates_block)
        .replace("{root_score}", f"{root_score:.3f}")
        .replace(
            "{n_pipelines_remaining}", str(int(n_pipelines_remaining)),
        )
        .replace("{n_candidates}", str(len(shown)))
    )
    prompt = prompt + "\n\n" + _PICKER_SCHEMA_HINT

    try:
        raw = llm.complete(prompt=prompt)
    except Exception as e:  # noqa: BLE001 — picker must never raise
        _LOG.warning("LLM parent picker call failed: %s", e)
        return ParentPickResult(
            picked_node_id=None,
            rationale=f"LLM call failed: {e}",
            fallback_reason="llm_error",
        )

    parsed = _extract_json_object(raw)
    if parsed is None:
        _LOG.warning(
            "LLM parent picker: could not parse JSON (raw=%r)", raw[:200],
        )
        return ParentPickResult(
            picked_node_id=None,
            rationale="JSON parse failure on LLM reply",
            raw_response=raw,
            fallback_reason="parse_error",
        )

    picked = str(parsed.get("picked_node_id", "")).strip()
    rationale = str(parsed.get("rationale", "")).strip()
    valid_ids = {c.node_id for c in shown}
    if picked not in valid_ids:
        _LOG.warning(
            "LLM parent picker: returned id %r not in candidate set "
            "(size=%d). Rationale was: %s",
            picked, len(shown), rationale[:200],
        )
        return ParentPickResult(
            picked_node_id=None,
            rationale=(
                f"LLM returned invalid id {picked!r} (not in candidate "
                f"set of size {len(shown)})"
            ),
            raw_response=raw,
            fallback_reason="invalid_pick",
        )

    return ParentPickResult(
        picked_node_id=picked,
        rationale=rationale or "(no rationale provided by LLM)",
        raw_response=raw,
    )


# ─── Helpers ────────────────────────────────────────────────────────────────


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object in ``raw``, or None.

    Mirrors ``matchfix_gate._extract_json_object`` — kept inline to
    avoid module-level cross-import bloat.
    """
    if not raw:
        return None
    candidates: list[str] = []
    m = _JSON_FENCE_RE.search(raw)
    if m:
        candidates.append(m.group(1))
    m2 = _FIRST_OBJ_RE.search(raw)
    if m2:
        candidates.append(m2.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
