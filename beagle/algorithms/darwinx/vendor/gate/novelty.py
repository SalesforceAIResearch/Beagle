"""GEA-style novelty over the archive's binary task-success vectors.

Adopts the parent-group selection mechanism from Group-Evolving Agents
(arXiv:2602.04837 §3.1, Algorithm 1). Each archived node is represented
as a binary vector ``z ∈ {0,1}^D`` over the configured task universe
(``z[t] = 1`` iff the node solved task ``t``). Novelty of a node is the
average cosine distance to its ``M`` nearest neighbors in that space —
intuitively, "how different are this node's solved-task pattern from
the closest other lineages in the archive?".

Used by ``self_evolve.parent_selection.PerformanceNoveltyStrategy`` to
rank parents by ``score(i) = node.score × sqrt(novelty(i))``, which
biases the picker toward parents whose solved-task fingerprint is
*both* high-scoring *and* exploratorily different from the rest of the
archive.

Why we need this: the previous ``mixed_high_score`` picker selected
purely by score, so promising-but-undertested branches died off. The
GEA paper directly addresses this — see paper §5.2, Figure 4: DGM
(score-only) consolidated 5/9 tool innovations into its best agent;
GEA (PN selection + sibling pooling) consolidated 8/9.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


__all__ = [
    "TaskVector",
    "NoveltyScore",
    "cosine_distance",
    "knn_novelty",
    "score_node_pn",
    "rank_by_pn",
]


_LOG = logging.getLogger("atelier.novelty")


@dataclass(frozen=True)
class TaskVector:
    """A node's binary task-success vector over a fixed task universe.

    Stored as a frozenset of solved task ids — fast set arithmetic
    + small memory footprint vs a dense bitvector for the usual
    ~50-100 task universes.
    """

    node_id: str
    solved: frozenset[str]
    """Tasks the node passed at its final-eval. Empty when the node
    didn't reach final-eval (e.g., MODIFIED by the gate)."""


@dataclass(frozen=True)
class NoveltyScore:
    """Per-node novelty + its supporting neighbor distances.

    Kept as a record so we can emit a per-node sidecar for auditing /
    later visualization (which neighbors drove the novelty score).
    """

    node_id: str
    novelty: float
    """Average cosine distance to the M nearest neighbors. ∈ [0, 1].
    1.0 means the node's solved-task pattern is maximally different
    from its M closest peers; 0.0 means it's identical."""

    neighbors: tuple[tuple[str, float], ...]
    """``(neighbor_node_id, cosine_distance)`` pairs, ordered by
    distance ascending. Length ≤ M. Used by the diversity-telemetry
    sidecar."""


# ─── Pure math ──────────────────────────────────────────────────────────────


def cosine_distance(a: frozenset[str], b: frozenset[str]) -> float:
    """Cosine distance between two binary task-success vectors.

    ``d(a, b) = 1 - (|a ∩ b| / sqrt(|a| × |b|))``  (with ε-guard).

    Returns 1.0 (maximally distant) when either set is empty. This
    handles the edge case where a node didn't reach final-eval —
    we don't want such nodes to dominate as "novel" in the picker,
    so the caller MUST filter out empty-solved nodes before ranking.
    """
    if not a or not b:
        return 1.0
    if a == b:
        # Exact identity short-circuit avoids the ε-guard introducing
        # a fractional float (e.g., 3.3e-13) that would surprise the
        # caller / tests asserting exact 0.0.
        return 0.0
    inter = len(a & b)
    denom = math.sqrt(len(a) * len(b)) + 1e-12
    cos_sim = inter / denom
    # Clamp; floating-point noise can push this fractionally over 1.0
    # which would flip the sign of (1 - cos_sim).
    cos_sim = max(0.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def knn_novelty(
    target: TaskVector,
    pool: Sequence[TaskVector],
    *,
    m: int = 4,
) -> NoveltyScore:
    """Average cosine distance from ``target`` to its M nearest
    neighbors in ``pool`` (target itself excluded).

    When ``pool`` has fewer than M eligible neighbors (after self-
    exclusion), uses whatever's available. When the pool is empty
    after exclusion (e.g., target is the only solved node), returns
    novelty=1.0 — the picker treats it as maximally diverse, which
    is honest: there's nothing to compare against, so we shouldn't
    penalize it.
    """
    others = [v for v in pool if v.node_id != target.node_id]
    if not others:
        return NoveltyScore(
            node_id=target.node_id, novelty=1.0, neighbors=(),
        )

    distances: list[tuple[str, float]] = []
    for other in others:
        d = cosine_distance(target.solved, other.solved)
        distances.append((other.node_id, d))

    distances.sort(key=lambda item: item[1])
    k = min(m, len(distances))
    nearest = tuple(distances[:k])
    avg = sum(d for _, d in nearest) / k
    return NoveltyScore(
        node_id=target.node_id, novelty=avg, neighbors=nearest,
    )


def score_node_pn(*, score: float | None, novelty: float) -> float:
    """Combined Performance-Novelty score per GEA Eq. 3:
    ``score(i) = α_i × sqrt(novelty(i))``.

    Returns 0.0 when ``score`` is None (un-scored node — not eligible
    as a parent). The ``sqrt`` moderates novelty so a maximally
    diverse but mediocre node doesn't outrank a high-scoring one
    with modest novelty.
    """
    if score is None:
        return 0.0
    s = max(0.0, float(score))
    n = max(0.0, min(1.0, float(novelty)))
    return s * math.sqrt(n)


# ─── Ranking helper ─────────────────────────────────────────────────────────


def rank_by_pn(
    *,
    nodes: Sequence[TaskVector],
    scores: Mapping[str, float | None],
    m: int = 4,
) -> list[tuple[str, float, NoveltyScore]]:
    """Compute PN score for every node and return a ranked list.

    Returns ``[(node_id, combined_pn_score, NoveltyScore), ...]``
    sorted by combined PN score descending. Stable on ties via
    secondary sort by node_id (deterministic for tests + reruns).
    """
    if not nodes:
        return []
    out: list[tuple[str, float, NoveltyScore]] = []
    for tv in nodes:
        nov = knn_novelty(tv, nodes, m=m)
        pn = score_node_pn(score=scores.get(tv.node_id), novelty=nov.novelty)
        out.append((tv.node_id, pn, nov))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def task_vectors_from_solved_lists(
    *, solved_by_node: Mapping[str, Iterable[str]],
) -> list[TaskVector]:
    """Construct TaskVectors from a ``{node_id: [task_id, ...]}`` map.

    Convenience for callers that pull ``solved_tasks_json`` straight
    out of the campaign tree DB. Empty solved lists are kept — the
    caller decides whether to filter them out before ranking (we
    don't filter here because the patch-record + diversity sidecar
    consumers may want the full set).
    """
    return [
        TaskVector(node_id=str(nid), solved=frozenset(str(t) for t in tasks))
        for nid, tasks in solved_by_node.items()
    ]
