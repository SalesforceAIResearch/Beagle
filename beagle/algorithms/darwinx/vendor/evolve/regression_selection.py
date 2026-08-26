"""Candidate selection for regression resolver workers."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

from . import tree


FULL_SCORE_WEIGHT = 2.0
FULL_SCORE_GAP_PENALTY = 0.30
SCORE_GAIN_WEIGHT = 0.75
REPAIRABILITY_WEIGHT = 0.15
IMPROVED_TASK_WEIGHT = 0.01
REGRESSED_TASK_PENALTY = 0.03
PRESERVATION_RISK_WEIGHT = 0.75
EXISTING_RESOLVER_PENALTY = 0.05
SCORE_GAIN_EPSILON = 1e-9


@dataclass(frozen=True)
class RegressionCandidate:
    node: tree.Node
    parent: tree.Node
    root: tree.Node | None
    full_score: float
    score_gain: float
    existing_resolvers: int
    selection_score: float
    repairability_score: float = 0.0
    preservation_risk: float = 0.0
    full_score_gap_from_best: float = 0.0
    ordered_regressed_tasks: tuple[str, ...] = ()
    claimable_regressed_tasks: tuple[str, ...] = ()
    active_claimed_regressed_count: int = 0
    selection_method: str = "rule"


@dataclass(frozen=True)
class RegressionDiagnostics:
    total_nodes: int
    eligible: int
    rejection_counts: dict[str, int]
    top_candidates: list[dict]


def eligible_candidates(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    min_score_gain: float,
    exclude_node_ids: set[str] | None = None,
) -> list[RegressionCandidate]:
    exclude_node_ids = exclude_node_ids or set()
    contexts, root = _eligible_candidate_contexts(
        conn,
        campaign=campaign,
        subset=subset,
        min_score_gain=min_score_gain,
        exclude_node_ids=exclude_node_ids,
    )
    best_full_score = _best_full_score(contexts)
    out: list[RegressionCandidate] = []
    for ctx in contexts:
        n = ctx.node
        parent = ctx.parent
        full_score = ctx.score
        full_score_gap_from_best = max(0.0, best_full_score - full_score)
        selection_score = _selection_score(
            full_score=full_score,
            full_score_gap_from_best=full_score_gap_from_best,
            score_gain=ctx.score_gain,
            repairability=ctx.repairability_score,
            improved_count=len(ctx.improved_tasks),
            claimable_regressed_count=len(ctx.claimable_regressed_tasks),
            preservation_risk=ctx.preservation_risk,
            existing_resolvers=ctx.existing_resolvers,
        )
        out.append(RegressionCandidate(
            node=n,
            parent=parent,
            root=root,
            full_score=full_score,
            score_gain=ctx.score_gain,
            existing_resolvers=ctx.existing_resolvers,
            selection_score=selection_score,
            repairability_score=ctx.repairability_score,
            preservation_risk=ctx.preservation_risk,
            full_score_gap_from_best=full_score_gap_from_best,
            ordered_regressed_tasks=ctx.ordered_regressed_tasks,
            claimable_regressed_tasks=ctx.ordered_regressed_tasks,
            active_claimed_regressed_count=(
                len(ctx.regressed_tasks) - len(ctx.claimable_regressed_tasks)
            ),
        ))
    return out


def diagnostics(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    min_score_gain: float,
    exclude_node_ids: set[str] | None = None,
    limit: int = 5,
) -> RegressionDiagnostics:
    exclude_node_ids = exclude_node_ids or set()
    nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    rejection_counts: dict[str, int] = {}
    contexts, _root = _eligible_candidate_contexts(
        conn,
        campaign=campaign,
        subset=subset,
        min_score_gain=min_score_gain,
        exclude_node_ids=exclude_node_ids,
        rejection_counts=rejection_counts,
    )
    best_full_score = _best_full_score(contexts)
    candidates: list[dict] = []

    for ctx in contexts:
        n = ctx.node
        parent = ctx.parent
        full_score = ctx.score
        full_score_gap_from_best = max(0.0, best_full_score - full_score)
        selection_score = _selection_score(
            full_score=full_score,
            full_score_gap_from_best=full_score_gap_from_best,
            score_gain=ctx.score_gain,
            repairability=ctx.repairability_score,
            improved_count=len(ctx.improved_tasks),
            claimable_regressed_count=len(ctx.claimable_regressed_tasks),
            preservation_risk=ctx.preservation_risk,
            existing_resolvers=ctx.existing_resolvers,
        )
        candidates.append({
            "node": n.id,
            "score": ctx.score,
            "parent": parent.id,
            "parent_score": ctx.parent_score,
            "selection_score": round(selection_score, 4),
            "score_gain": round(ctx.score_gain, 4),
            "full_score_gap_from_best": round(full_score_gap_from_best, 4),
            "claimable_regressed_tasks": ctx.claimable_regressed_tasks,
            "active_claimed_regressed_count": (
                len(ctx.regressed_tasks) - len(ctx.claimable_regressed_tasks)
            ),
            "regressed_tasks": list(ctx.regressed_tasks),
            "improved_tasks": list(ctx.improved_tasks),
            "repairability_score": round(ctx.repairability_score, 4),
            "preservation_risk": round(ctx.preservation_risk, 4),
            "existing_resolvers": ctx.existing_resolvers,
        })
    candidates.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["score"]),
            -float(row["repairability_score"]),
            float(row["preservation_risk"]),
            -float(row["score_gain"]),
            row["node"],
        )
    )
    return RegressionDiagnostics(
        total_nodes=len(nodes),
        eligible=len(candidates),
        rejection_counts=rejection_counts,
        top_candidates=candidates[:limit],
    )


def select_candidate(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    pipeline_id: str,
    min_score_gain: float,
    exclude_node_ids: set[str] | None = None,
) -> RegressionCandidate | None:
    candidates = eligible_candidates(
        conn,
        campaign=campaign,
        subset=subset,
        min_score_gain=min_score_gain,
        exclude_node_ids=exclude_node_ids,
    )
    if not candidates:
        return None
    candidates.sort(
        key=lambda c: (
            -c.selection_score,
            -c.full_score,
            -c.repairability_score,
            c.preservation_risk,
            -c.score_gain,
            c.existing_resolvers,
            _tiebreak(pipeline_id, c.node.id),
        )
    )
    return candidates[0]


def claim_regression_tasks(
    conn: sqlite3.Connection,
    *,
    candidate: RegressionCandidate,
    campaign: str,
    subset: str,
    pipeline_id: str,
    max_claim: int | None = None,
) -> list[str]:
    """Claim regressed tasks for a resolver pipeline.

    By default (`max_claim=None` or `max_claim<=0`) claims ALL of the
    candidate's regressed tasks so a single resolver owns the full
    regression set; sibling resolvers spawned later (when
    `_has_active_resolver_child` clears) get whatever is still active and
    unclaimed at that time. Pass a positive integer to cap claim count
    (operator safety knob — see `--regression-max-claim`).
    """
    node_eval = tree.node_search_eval(
        conn, campaign=campaign, node_id=candidate.node.id,
    )
    candidates = list(
        candidate.ordered_regressed_tasks
        or tuple(node_eval.regressed_tasks if node_eval else candidate.node.regressed_tasks)
    )
    k = len(candidates) if not max_claim or max_claim <= 0 else min(max_claim, len(candidates))
    return tree.try_claim_tasks(
        conn,
        campaign=campaign,
        subset=subset,
        candidate_tasks=candidates,
        k=k,
        pipeline_id=pipeline_id,
        parent_id=candidate.node.id,
        claim_kind="regression_resolve",
    )


def _has_active_resolver_child(
    conn: sqlite3.Connection, *, campaign: str, node_id: str,
) -> bool:
    edges = tree.list_node_edges(
        conn, campaign=campaign, parent_id=node_id, edge_type="regression_resolve",
    )
    for edge in edges:
        child = tree.get_node(conn, edge.child_id)
        if child and child.status == "in_progress":
            return True
    return False


def _resolver_child_count(
    conn: sqlite3.Connection, *, campaign: str, node_id: str,
) -> int:
    return len(tree.list_node_edges(
        conn, campaign=campaign, parent_id=node_id, edge_type="regression_resolve",
    ))


@dataclass(frozen=True)
class _CandidateScoringContext:
    node: tree.Node
    parent: tree.Node
    node_eval: tree.NodeEval | None
    parent_eval: tree.NodeEval | None
    root_eval: tree.NodeEval | None
    existing_resolvers: int
    score_gain: float
    score: float
    parent_score: float
    improved_tasks: tuple[str, ...]
    regressed_tasks: tuple[str, ...]
    solved_tasks: tuple[str, ...]
    claimable_regressed_tasks: list[str]
    repairability_score: float
    preservation_risk: float
    ordered_regressed_tasks: tuple[str, ...]


def _eligible_candidate_contexts(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    min_score_gain: float,
    exclude_node_ids: set[str],
    rejection_counts: dict[str, int] | None = None,
) -> tuple[list[_CandidateScoringContext], tree.Node | None]:
    nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    by_id = {n.id: n for n in nodes}
    search_evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    root = next(
        (
            n for n in nodes
            if n.parent_id is None and (n.id in search_evals or n.score is not None)
        ),
        None,
    )
    root_eval = search_evals.get(root.id) if root else None
    task_stats = tree.task_outcome_stats(conn, campaign=campaign, subset=subset)
    active_claims = {
        c.failure_task for c in tree.list_active_claims(
            conn, campaign=campaign, subset=subset,
            claim_kind="regression_resolve",
        )
    }
    contexts: list[_CandidateScoringContext] = []

    def reject(reason: str) -> None:
        if rejection_counts is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for n in nodes:
        if n.id in exclude_node_ids:
            reject("reserved")
            continue
        n_eval = search_evals.get(n.id)
        n_score = n_eval.score if n_eval else n.score
        if n.status not in {"completed", "no_change"} or n_score is None:
            reject("unscored_or_ineligible_status")
            continue
        if not n.parent_id:
            reject("root")
            continue
        regressed_tasks = tuple(n_eval.regressed_tasks if n_eval else n.regressed_tasks)
        improved_tasks = tuple(n_eval.improved_tasks if n_eval else n.improved_tasks)
        solved_tasks = tuple(n_eval.solved_tasks if n_eval else n.solved_tasks)
        if not regressed_tasks:
            reject("no_regressed_tasks")
            continue
        parent = by_id.get(n.parent_id)
        parent_eval = search_evals.get(parent.id) if parent else None
        parent_score = parent_eval.score if parent_eval else (parent.score if parent else None)
        root_score = root_eval.score if root_eval else (root.score if root else None)
        if parent is None or parent_score is None:
            reject("missing_scored_parent")
            continue
        if _has_active_resolver_child(conn, campaign=campaign, node_id=n.id):
            reject("active_resolver_child")
            continue
        existing = _resolver_child_count(conn, campaign=campaign, node_id=n.id)
        if existing >= 2:
            reject("resolver_attempt_limit")
            continue
        score_gain = max(
            float(n_score or 0.0) - float(parent_score or 0.0),
            float(n_score or 0.0) - float(root_score or 0.0) if root_score is not None else 0.0,
        )
        if score_gain + SCORE_GAIN_EPSILON < min_score_gain:
            reject("below_min_score_gain")
            continue
        claimable_regressed = [t for t in regressed_tasks if t not in active_claims]
        if not claimable_regressed:
            reject("regressed_tasks_already_claimed")
            continue
        repairability = _repairability_score(claimable_regressed, task_stats)
        preservation_risk = _preservation_risk_for_tasks(
            solved_tasks=solved_tasks,
            improved_tasks=improved_tasks,
            regressed_tasks=regressed_tasks,
            task_stats=task_stats,
        )
        ordered_regressed = tuple(sorted(
            claimable_regressed,
            key=lambda t: (-_task_repair_value(t, task_stats), t),
        ))
        contexts.append(_CandidateScoringContext(
            node=n,
            parent=parent,
            node_eval=n_eval,
            parent_eval=parent_eval,
            root_eval=root_eval,
            existing_resolvers=existing,
            score_gain=score_gain,
            score=float(n_score or 0.0),
            parent_score=float(parent_score or 0.0),
            improved_tasks=improved_tasks,
            regressed_tasks=regressed_tasks,
            solved_tasks=solved_tasks,
            claimable_regressed_tasks=claimable_regressed,
            repairability_score=repairability,
            preservation_risk=preservation_risk,
            ordered_regressed_tasks=ordered_regressed,
        ))

    return contexts, root


def _best_full_score(contexts: list[_CandidateScoringContext]) -> float:
    return max((float(ctx.score or 0.0) for ctx in contexts), default=0.0)


def _repairability_score(
    tasks: list[str],
    task_stats: dict[str, tree.TaskOutcomeStats],
) -> float:
    return sum(_task_repair_value(t, task_stats) for t in tasks)


def _selection_score(
    *,
    full_score: float,
    full_score_gap_from_best: float,
    score_gain: float,
    repairability: float,
    improved_count: int,
    claimable_regressed_count: int,
    preservation_risk: float,
    existing_resolvers: int,
) -> float:
    """Composite resolver value.

    Full final score gets an explicit leading weight: a high-scoring node has
    more solved benchmark surface to preserve, so repairing even a few of its
    regressions is often more valuable than repairing a lower-scoring branch.
    The other terms still let an obviously easier, safer repair outrank a
    slightly higher-scoring node.
    """
    return (
        FULL_SCORE_WEIGHT * max(0.0, min(1.0, full_score))
        - FULL_SCORE_GAP_PENALTY * full_score_gap_from_best
        + SCORE_GAIN_WEIGHT * score_gain
        + REPAIRABILITY_WEIGHT * repairability
        + IMPROVED_TASK_WEIGHT * improved_count
        - REGRESSED_TASK_PENALTY * claimable_regressed_count
        - PRESERVATION_RISK_WEIGHT * preservation_risk
        - EXISTING_RESOLVER_PENALTY * existing_resolvers
    )


def _task_repair_value(
    task: str,
    task_stats: dict[str, tree.TaskOutcomeStats],
) -> float:
    stats = task_stats.get(task)
    if stats is None:
        return 1.0
    prior_resolver_penalty = 0.35 * stats.regression_resolver_failures
    fragility_penalty = 0.25 * stats.failure_rate + 0.25 * stats.regression_rate
    success_bonus = 0.75 * stats.resolver_success_rate
    return max(0.05, 1.0 + success_bonus - prior_resolver_penalty - fragility_penalty)


def _preservation_risk(
    node: tree.Node,
    task_stats: dict[str, tree.TaskOutcomeStats],
) -> float:
    return _preservation_risk_for_tasks(
        solved_tasks=tuple(node.solved_tasks),
        improved_tasks=tuple(node.improved_tasks),
        regressed_tasks=tuple(node.regressed_tasks),
        task_stats=task_stats,
    )


def _preservation_risk_for_tasks(
    *,
    solved_tasks: tuple[str, ...],
    improved_tasks: tuple[str, ...],
    regressed_tasks: tuple[str, ...],
    task_stats: dict[str, tree.TaskOutcomeStats],
) -> float:
    solved = set(solved_tasks)
    regressed = set(regressed_tasks)
    preserved = solved - regressed
    fragile_preserved = 0.0
    for task in preserved:
        stats = task_stats.get(task)
        if stats is None:
            continue
        fragile_preserved += 0.02 * stats.failure_rate + 0.04 * stats.regression_rate
    return (
        0.02 * len(improved_tasks)
        + 0.03 * len(regressed_tasks)
        + fragile_preserved
    )


def _tiebreak(pipeline_id: str, node_id: str) -> str:
    return hashlib.sha256(f"{pipeline_id}:regression:{node_id}".encode()).hexdigest()


__all__ = [
    "RegressionCandidate",
    "RegressionDiagnostics",
    "claim_regression_tasks",
    "diagnostics",
    "eligible_candidates",
    "select_candidate",
]
