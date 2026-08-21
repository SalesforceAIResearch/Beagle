"""GEA-style sibling-trace pooling for the analyze prompt.

Adopts the second half of Group-Evolving Agents (arXiv:2602.04837 §3.2):
each new iteration's proposer doesn't reason in isolation from its
parent — it also sees the *aggregated evidence* of K-1 "sibling"
parents (other recent high-PN nodes from the archive). The pooled
evidence covers:

    1. The sibling's verdict-stage outcome (decision, modified
       surfaces, n_regressions)
    2. Tasks the sibling newly solved (improvement signal)
    3. Tasks the sibling regressed (avoidance signal)
    4. LTM lessons the sibling persisted (if any survived rollback)
    5. A compact failing-trial digest (so the proposer can spot
       common failure modes across branches)

The "group experience" pattern dramatically beats independent
branches on SWE-bench Verified (GEA: 71.0% vs DGM: 56.7% per Table 1),
because tool innovations from one branch get re-discovered+integrated
by other branches rather than dying.

This module exposes:
- ``select_siblings()`` — pick K-1 siblings using the same PN scoring
  but excluding the focus node's lineage + already-chosen siblings.
- ``render_sibling_evidence()`` — produce the markdown block to inline
  into the analyze.md prompt template.

The pipeline integration lives in
``self_evolve.pipeline._render_analyze_prompt``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


__all__ = [
    "SiblingCard",
    "select_siblings",
    "render_sibling_evidence",
    "load_node_ancestors",
]


_LOG = logging.getLogger("atelier.sibling_pool")


# ─── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SiblingCard:
    """One sibling's evidence rendered for the analyze prompt.

    Built from the sibling's tree-row + its equivalence sidecar +
    its LTM contributions. Designed to be small enough that K=2-3
    siblings fit comfortably in the analyze prompt budget.
    """

    node_id: str
    score: float
    parent_node_id: str | None
    modified_surfaces: tuple[str, ...]
    """From the sibling's equivalence verdict.semantic_analysis.
    Empty if the gate didn't run (e.g., very old node)."""

    verdict_decision: str
    """EQUIVALENT / MODIFIED / INCONCLUSIVE / unknown."""

    n_probe_pass: int
    n_probe_fail: int

    improved_tasks: tuple[str, ...]
    regressed_tasks: tuple[str, ...]

    ltm_lessons: tuple[str, ...]
    """Lessons this sibling persisted to LongTermMemory (and that
    weren't rolled back). Capped to ~3 to keep cards compact."""

    sample_failing_trial: str
    """First ~400 chars of one regressed-trial digest. Empty when
    sibling had no regressions or no digest available."""


# ─── Selection ──────────────────────────────────────────────────────────────


def load_node_ancestors(
    *, node_id: str, parent_of: dict[str, str | None],
) -> set[str]:
    """Walk the parent chain to collect all ancestors of ``node_id``.

    ``parent_of[child] = parent_id_or_None``. Returns the set of
    ancestor node ids (excludes ``node_id`` itself). Caps walk at 100
    hops to avoid pathological loops in malformed trees.
    """
    seen: set[str] = set()
    current = parent_of.get(node_id)
    hops = 0
    while current and current not in seen and hops < 100:
        seen.add(current)
        current = parent_of.get(current)
        hops += 1
    return seen


def select_siblings(
    *,
    focus_node_id: str,
    candidates: Sequence[tuple[str, float]],
    parent_of: dict[str, str | None],
    k: int = 2,
    exclude_lineage: bool = True,
    exclude_node_ids: Iterable[str] = (),
) -> list[str]:
    """Pick ``k-1`` sibling node ids from ``candidates``.

    Args:
        focus_node_id: the node this pipeline's worker will branch off
            (its lineage is excluded when ``exclude_lineage=True``).
        candidates: ``[(node_id, pn_score), ...]`` ordered by PN score
            descending. Same ranking produced by
            ``atelier.novelty.rank_by_pn``.
        parent_of: full archive parent map for lineage exclusion.
        k: group size (focus + k-1 siblings). When k <= 1, returns
            an empty list (sibling-pool disabled).
        exclude_lineage: if True, drop any candidate that's an
            ancestor or descendant of focus_node_id.
        exclude_node_ids: additional ids to filter out (e.g., siblings
            already picked by other pipelines in the same iteration).

    Returns:
        Up to ``k-1`` node ids, ordered by PN score descending.
    """
    n_siblings = max(0, k - 1)
    if n_siblings == 0:
        return []
    excluded = set(exclude_node_ids) | {focus_node_id}
    if exclude_lineage:
        excluded |= load_node_ancestors(
            node_id=focus_node_id, parent_of=parent_of,
        )
        # Also exclude direct children (descendants are less useful as
        # "siblings" — they're already informed by focus).
        for nid, par in parent_of.items():
            if par == focus_node_id:
                excluded.add(nid)

    picked: list[str] = []
    for node_id, _pn in candidates:
        if node_id in excluded:
            continue
        picked.append(node_id)
        if len(picked) >= n_siblings:
            break
    return picked


# ─── Evidence rendering ─────────────────────────────────────────────────────


def _load_equivalence_verdict(
    *, reports_root: Path, campaign: str, node_id: str,
) -> dict | None:
    """Load the equivalence sidecar for a node; None if missing /
    malformed (best-effort)."""
    path = (
        Path(reports_root) / campaign / "atelier" / "equivalence"
        / f"{node_id}.equivalence.json"
    )
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_node_ltm_contributions(
    *, reports_root: Path, campaign: str, node_id: str,
    max_lessons: int = 3,
) -> tuple[str, ...]:
    """Read the campaign's LongTermMemory file and pull out entries
    whose source_node == this node, skipping rolled-back ones.

    Returns up to ``max_lessons`` lesson body strings."""
    try:
        from atelier import long_term_memory
    except ImportError:
        return ()
    try:
        entries = long_term_memory.load_memory(
            reports_root=reports_root, campaign=campaign,
        )
        rejected = long_term_memory.load_rejected_sources(
            reports_root=reports_root, campaign=campaign,
        )
    except Exception:  # noqa: BLE001
        return ()
    if node_id in rejected:
        return ()
    contribs = [
        e.body.strip()[:300]
        for e in entries
        if e.source_node == node_id
    ]
    return tuple(contribs[:max_lessons])


def build_sibling_card(
    *,
    sibling_node_id: str,
    sibling_score: float,
    sibling_parent_id: str | None,
    sibling_improved_tasks: Sequence[str],
    sibling_regressed_tasks: Sequence[str],
    reports_root: Path,
    campaign: str,
) -> SiblingCard:
    """Assemble a SiblingCard from tree data + sidecar files.

    All sidecar reads are best-effort — missing files just degrade
    individual fields without raising.
    """
    verdict = _load_equivalence_verdict(
        reports_root=reports_root, campaign=campaign, node_id=sibling_node_id,
    )
    modified_surfaces: tuple[str, ...] = ()
    decision = "unknown"
    n_pass = 0
    n_fail = 0
    sample_digest = ""
    if verdict is not None:
        sa = verdict.get("semantic_analysis") or {}
        modified_surfaces = tuple(sa.get("modified_surfaces", []) or [])
        decision = str(verdict.get("decision", "unknown"))
        per_task = verdict.get("per_task_results") or {}
        n_pass = sum(1 for v in per_task.values() if float(v) >= 1.0)
        n_fail = len(per_task) - n_pass
        digests = verdict.get("failure_digests") or {}
        if digests:
            first_task, first_digest = next(iter(digests.items()))
            sample_digest = f"({first_task}) {str(first_digest)[:400]}"

    ltm_lessons = _load_node_ltm_contributions(
        reports_root=reports_root, campaign=campaign, node_id=sibling_node_id,
    )

    # Cap task lists to keep the prompt bounded.
    return SiblingCard(
        node_id=sibling_node_id,
        score=float(sibling_score or 0.0),
        parent_node_id=sibling_parent_id,
        modified_surfaces=modified_surfaces,
        verdict_decision=decision,
        n_probe_pass=n_pass,
        n_probe_fail=n_fail,
        improved_tasks=tuple(sibling_improved_tasks)[:8],
        regressed_tasks=tuple(sibling_regressed_tasks)[:8],
        ltm_lessons=ltm_lessons,
        sample_failing_trial=sample_digest,
    )


def render_sibling_evidence(
    *,
    focus_node_id: str,
    focus_score: float | None,
    siblings: Sequence[SiblingCard],
) -> str:
    """Build the markdown block to inline into the analyze prompt.

    Returns empty string when ``siblings`` is empty (sibling-pool
    feature effectively disabled — analyze.md template should
    conditionally include this block based on emptiness).
    """
    if not siblings:
        return ""

    lines: list[str] = []
    focus_s = f"{focus_score:.3f}" if focus_score is not None else "?"
    lines.append("## Sibling parents' recent experience (Atelier group evolution)")
    lines.append("")
    lines.append(
        f"You are evolving harness `{focus_node_id}` (score={focus_s}). "
        f"{len(siblings)} sibling parent(s) were also picked for this "
        f"iteration. Use their evidence as a hint about what works "
        f"and what regresses. **Prefer modifications that compose with "
        f"their successes; AVOID re-trying patterns they flagged as "
        f"MODIFIED.**"
    )
    lines.append("")

    for s in siblings:
        surfaces = (
            ", ".join(s.modified_surfaces) if s.modified_surfaces
            else "(no semantic analysis on record)"
        )
        lines.append(
            f"### Sibling `{s.node_id}` (score={s.score:.3f}, "
            f"verdict={s.verdict_decision}, probes={s.n_probe_pass}P/"
            f"{s.n_probe_fail}F)"
        )
        lines.append("")
        lines.append(f"- **modified surfaces**: {surfaces}")
        if s.improved_tasks:
            lines.append(
                f"- **newly solved tasks** (last commit's improvement): "
                f"{', '.join(s.improved_tasks)}"
            )
        if s.regressed_tasks:
            lines.append(
                f"- **regressed tasks** (avoid breaking these again): "
                f"`{', '.join(s.regressed_tasks)}`"
            )
        if s.ltm_lessons:
            lines.append("- **lessons this sibling persisted to LTM**:")
            for lesson in s.ltm_lessons:
                lines.append(f"  - {lesson}")
        if s.sample_failing_trial:
            lines.append("- **a failing trial they reasoned about**:")
            lines.append("  ```")
            for line in s.sample_failing_trial.splitlines()[:6]:
                lines.append(f"  {line}")
            lines.append("  ```")
        lines.append("")

    lines.append(
        "**Synthesis hint**: if a sibling added a bundled skill / "
        "sub-agent, your fix should target an ORTHOGONAL surface, "
        "not the same one. If a sibling regressed by modifying a "
        "core control-loop file, do not repeat that experiment."
    )
    lines.append("")
    return "\n".join(lines)
