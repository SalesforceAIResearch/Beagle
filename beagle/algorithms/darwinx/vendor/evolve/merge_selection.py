"""Candidate selection for self-evolve node mergers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import cursor_agent, meta_agent, run_config, tree


@dataclass(frozen=True)
class MergeCandidate:
    id: str
    branch_name: str
    commit_sha: str | None
    score: float
    solved_tasks: list[str]
    unsolved_tasks: list[str]
    created_at: str


@dataclass(frozen=True)
class MergePair:
    primary: tree.Node
    secondary: tree.Node
    selection_method: str


@dataclass(frozen=True)
class MergePairFeatures:
    primary_unique_wins: list[str]
    secondary_unique_wins: list[str]
    complementary_wins: list[str]
    shared_failures: list[str]
    fragile_parent_wins: list[str]
    parent_regression_surface: int
    prior_failed_similar_merges: int
    diversity_gain: int
    solved_union_size: int
    union_gain_over_best_parent: int
    minimum_new_wins_needed: int
    acceptability_risk: float
    coverage_value: float
    expected_utility: float
    observed_task_count: int
    task_value: float
    normalized_expected_utility: float
    union_gain_rate: float
    fragile_parent_win_rate: float
    parent_regression_rate: float
    shared_failure_rate: float
    parent_score_ceiling: float


GateKind = Literal["fixed", "auto"]

# Safety floors used by the auto gate. Match the historical fixed-gate
# defaults so a small / uniform-low-risk candidate population gets at
# least as permissive a threshold as today's fixed gate would have.
AUTO_SAFETY_FLOOR_PARENT_REGRESSION_RATE = 0.25
AUTO_SAFETY_FLOOR_FRAGILE_PARENT_WIN_RATE = 0.30
AUTO_SAFETY_FLOOR_SHARED_FAILURE_RATE = 0.50
# Minimum candidate-pair population before p25/p75 derivation activates;
# below this the auto gate falls back to absolute floors. 4 pairs is the
# smallest population where a 25th and 75th percentile are meaningfully
# distinguishable on a per-bucket basis.
AUTO_TUNE_MIN_PAIRS = 4
# Hard absolute floor for floor_new_wins when there are not enough pairs
# to derive p25(union_gain_over_best_parent). `max(2, ceil(N * 0.005))`
# gives 2 up to N=400, 5 at N=1000, 10 at N=2000.
AUTO_ABSOLUTE_FLOOR_NEW_WINS_PCT = 0.005
AUTO_FLOOR_NEW_WINS_MIN = 2


@dataclass(frozen=True)
class MergeAutoTuneStats:
    """Per-campaign distributional thresholds derived from the live merge-
    candidate population. Computed once per supervisor cycle and stuffed
    into `MergeGateConfig.auto_tune` so the gate uses the campaign's own
    behavior (not a hand-tuned constant) to decide which pairs pass.

    `floor_new_wins` is `max(MIN, floor(p25(union_gain_over_best_parent)))`
    when at least `AUTO_TUNE_MIN_PAIRS` pairs are available; otherwise it
    falls back to `max(MIN, ceil(N * 0.005))` where N is the observed task
    count. The three p75 fields are effective thresholds (`max(safety,
    p75)`), so the gate can compare directly without re-applying the floor.
    `pair_count` records the size of the population the stats were derived
    from — `pair_count < AUTO_TUNE_MIN_PAIRS` means the stats are
    fallback-only and the gate effectively matches today's fixed gate.
    """
    floor_new_wins: int
    floor_new_wins_source: str  # "p25" or "abs_fallback"
    p25_union_gain_raw: float
    abs_fallback_floor_new_wins: int
    effective_parent_regression_rate: float
    effective_fragile_parent_win_rate: float
    effective_shared_failure_rate: float
    raw_p75_parent_regression_rate: float
    raw_p75_fragile_parent_win_rate: float
    raw_p75_shared_failure_rate: float
    pair_count: int
    observed_task_count: int


@dataclass(frozen=True)
class MergeGateConfig:
    # Legacy / "fixed" gate parameters. Used directly when `gate_kind ==
    # "fixed"`. The "auto" gate ignores them (uses `auto_tune` instead).
    min_normalized_utility: float = 0.0
    max_parent_regression_rate: float = 0.25
    max_fragile_parent_win_rate: float = 0.30
    min_union_gain_rate_over_risk: float = 0.0
    allow_required_new_wins: bool = False
    # New: gate variant. Default "auto" derives thresholds from the live
    # candidate population (see `MergeAutoTuneStats`). Pass "fixed" to
    # restore the legacy rate-vs-rate gate with the fields above.
    gate_kind: GateKind = "auto"
    # New: explicit override for the auto gate's absolute "union must add
    # at least N new wins" floor. None means "let the auto-tuner derive
    # it from the population (p25 with absolute fallback)".
    floor_new_wins: int | None = None
    # Campaign-derived population stats. None when `gate_kind == "fixed"`
    # or when the caller hasn't supplied them yet (the diagnostics path
    # builds and caches the stats once per cycle).
    auto_tune: MergeAutoTuneStats | None = None


@dataclass(frozen=True)
class MergePairDiagnostics:
    total_pairs: int
    gated_pairs: int
    rejection_counts: dict[str, int]
    top_pairs: list[dict]
    auto_tune: MergeAutoTuneStats | None = None


def _eval_for(
    node: tree.Node,
    node_evals: dict[str, tree.NodeEval] | None,
) -> tree.NodeEval | None:
    return (node_evals or {}).get(node.id)


def _score(
    node: tree.Node,
    node_evals: dict[str, tree.NodeEval] | None,
) -> float | None:
    ev = _eval_for(node, node_evals)
    return ev.score if ev else node.score


def _solved(
    node: tree.Node,
    node_evals: dict[str, tree.NodeEval] | None,
) -> list[str]:
    ev = _eval_for(node, node_evals)
    return ev.solved_tasks if ev else node.solved_tasks


def _failed(
    node: tree.Node,
    node_evals: dict[str, tree.NodeEval] | None,
) -> list[str]:
    ev = _eval_for(node, node_evals)
    return ev.failed_tasks if ev else node.failed_tasks


def _partial(
    node: tree.Node,
    node_evals: dict[str, tree.NodeEval] | None,
) -> list[str]:
    ev = _eval_for(node, node_evals)
    return ev.partially_solved_tasks if ev else node.partially_solved_tasks


def _improved(
    node: tree.Node,
    node_evals: dict[str, tree.NodeEval] | None,
) -> list[str]:
    ev = _eval_for(node, node_evals)
    return ev.improved_tasks if ev else node.improved_tasks


def _regressed(
    node: tree.Node,
    node_evals: dict[str, tree.NodeEval] | None,
) -> list[str]:
    ev = _eval_for(node, node_evals)
    return ev.regressed_tasks if ev else node.regressed_tasks


def eligible_candidates(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    score_slack: float | None = None,
    eligible_node_ids: set[str] | None = None,
) -> list[tree.Node]:
    nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    root = next(
        (
            n for n in nodes
            if n.parent_id is None and (n.id in evals or n.score is not None)
        ),
        None,
    )
    if root is None:
        return []
    root_score = _score(root, evals)
    merge_children = {
        edge.child_id
        for edge in tree.list_node_edges(conn, campaign=campaign, edge_type="merge")
    }
    scored = [
        n for n in nodes
        if n.parent_id is not None
        # REDESIGN (2026-06-30): recombination must be able to combine COMPLEMENTARY
        # specialists, not only nodes that already beat root. A node that solves a
        # task the others don't is merge fodder even if its own total score <= root
        # (two additive specialists can union into a child that beats both parents,
        # and the downstream union-gain + preservation gate is the real validator).
        # So admit archived stepping-stones and DROP the strict > root_score filter.
        and n.status in {"completed", "no_change", "archived"}
        and not (n.status == "no_change" and n.id in merge_children)
        and _score(n, evals) is not None
        and (eligible_node_ids is None or n.id in eligible_node_ids)
    ]
    if len(scored) < 2:
        return []
    best_score = max(_score(n, evals) or 0.0 for n in scored)
    best_node = max(scored, key=lambda n: ((_score(n, evals) or 0.0), n.created_at, n.id))
    if score_slack is None:
        task_count = max(
            (len(_solved(n, evals)) + len(_failed(n, evals)) for n in scored),
            default=0,
        )
        score_slack = (1.0 / task_count) if task_count else 0.02
    threshold = best_score - score_slack
    near_best = [n for n in scored if (_score(n, evals) or 0.0) >= threshold]

    # Large campaigns often produce specialist nodes that are below the normal
    # score band but solve tasks the current best node does NOT. Keep those
    # available as secondaries; pair scoring still downranks risky or low-value
    # combinations later.
    #
    # Complementarity must be measured GLOBALLY, not against the best node's
    # batch-local failures: under cluster-batch evolution each node is mini-eval'd
    # on its own disjoint task batch, so `best_node.failed_tasks` only spans the
    # best node's own batch and misses cross-cluster winners entirely (a node that
    # solves sanitize-git/extract-elf/chess is invisible to a best node whose batch
    # was sqlite/regex-log/query-optimize). The correct, batch-agnostic test is
    # whether the candidate adds at least one solved task beyond `best_solved` —
    # i.e. real recombination potential. The downstream merge gate
    # (union_gain_over_best_parent) still validates that the union actually gains.
    best_solved = set(_solved(best_node, evals))
    near_best_ids = {n.id for n in near_best}
    specialists = [
        n for n in scored
        if n.id not in near_best_ids and (set(_solved(n, evals)) - best_solved)
    ]
    out: list[tree.Node] = []
    seen: set[str] = set()
    for node in [*near_best, *specialists]:
        if node.id in seen:
            continue
        seen.add(node.id)
        out.append(node)
    return out


def best_complementary_group(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    max_n: int = 4,
    eligible_node_ids: set[str] | None = None,
    seed_primary_id: str | None = None,
) -> tuple[tree.Node, list[tree.Node]] | None:
    """Greedily build the maximal COMPLEMENTARY group for an N-way merge: start
    from the highest-scoring eligible candidate (primary), then repeatedly add the
    candidate that contributes the most NEW solved/improved tasks beyond the group's
    union, until no member adds a unique win or ``max_n`` is reached. This is the
    lightweight union-screen's candidate SELECTION (ranked by predicted union gain,
    no eval); the merged child is then confirmed by the downstream avg@k eval. Picks
    by predicted average union, never best-of-k any-pass. Returns (primary, [secondaries])
    or None if no complementary pair exists."""
    cands = eligible_candidates(
        conn, campaign=campaign, subset=subset, eligible_node_ids=eligible_node_ids,
    )
    if len(cands) < 2:
        return None
    evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)

    def _wins(n: tree.Node) -> set[str]:
        return set(_solved(n, evals)) | set(_improved(n, evals))

    primary = None
    if seed_primary_id is not None:
        primary = next((c for c in cands if c.id == seed_primary_id), None)
    if primary is None:
        primary = max(cands, key=lambda n: ((_score(n, evals) or 0.0), n.created_at, n.id))
    group = [primary]
    union = set(_wins(primary))
    pool = [c for c in cands if c.id != primary.id]
    while len(group) < max(2, max_n):
        best = None
        best_gain = 0
        for c in pool:
            gain = len(_wins(c) - union)
            if gain > best_gain:
                best_gain = gain
                best = c
        if best is None or best_gain < 1:
            break
        group.append(best)
        union |= _wins(best)
        pool.remove(best)
    if len(group) < 2:
        return None
    return group[0], group[1:]


def eligible_pairs(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    score_slack: float | None = None,
    exclude_pair_keys: set[tuple[str, str]] | None = None,
    eligible_node_ids: set[str] | None = None,
    gate_config: MergeGateConfig | None = None,
) -> list[tuple[tree.Node, tree.Node]]:
    """Return the candidate pairs that pass `gate_config` (or all candidate
    pairs if `gate_config is None`).

    When `gate_config.gate_kind == "auto"` and `gate_config.auto_tune is
    None`, this function derives the auto-tune stats from the live
    pre-gate candidate population FIRST, then applies the gate. So the
    gate's thresholds always reflect the campaign's own distribution at
    the moment of decision, not a stale snapshot.
    """
    pairs_with_features = _collect_pair_features(
        conn,
        campaign=campaign,
        subset=subset,
        score_slack=score_slack,
        exclude_pair_keys=exclude_pair_keys,
        eligible_node_ids=eligible_node_ids,
    )
    if gate_config is None:
        return [(a, b) for a, b, _f_ab, _f_ba in pairs_with_features]
    effective_gate = _ensure_auto_tune(gate_config, pairs_with_features)
    out: list[tuple[tree.Node, tree.Node]] = []
    for a, b, f_ab, f_ba in pairs_with_features:
        reasons_ab = _gate_reasons(f_ab, effective_gate)
        reasons_ba = _gate_reasons(f_ba, effective_gate)
        if reasons_ab and reasons_ba:
            continue
        out.append((a, b))
    return out


def _collect_pair_features(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    score_slack: float | None = None,
    exclude_pair_keys: set[tuple[str, str]] | None = None,
    eligible_node_ids: set[str] | None = None,
) -> list[tuple[tree.Node, tree.Node, MergePairFeatures, MergePairFeatures]]:
    """One-pass enumeration: gather every candidate pair plus the
    forward+reverse `pair_features` for it. Centralizes the per-cycle
    cost of computing features so both gating and auto-tune derivation
    consume the same memoized values (no double work).
    """
    candidates = eligible_candidates(
        conn,
        campaign=campaign,
        subset=subset,
        score_slack=score_slack,
        eligible_node_ids=eligible_node_ids,
    )
    node_evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    attempted = set(tree.merge_attempted_pair_keys(conn, campaign=campaign))
    attempted.update(exclude_pair_keys or set())
    ancestors_by_node = _ancestor_map(conn, campaign=campaign)
    task_stats = tree.task_outcome_stats(conn, campaign=campaign, subset=subset)
    reserved_solved_tasks = _reserved_solved_tasks(
        conn, campaign=campaign, pair_keys=exclude_pair_keys or set(),
    )
    prior_failed = _failed_merge_task_signatures(conn, campaign=campaign)
    out: list[tuple[tree.Node, tree.Node, MergePairFeatures, MergePairFeatures]] = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if tuple(sorted((a.id, b.id))) in attempted:
                continue
            if a.id in ancestors_by_node.get(b.id, set()):
                continue
            if b.id in ancestors_by_node.get(a.id, set()):
                continue
            if not _has_complementary_solved_tasks(a, b, node_evals=node_evals):
                continue
            f_ab = pair_features(
                a, b,
                task_stats=task_stats,
                reserved_solved_tasks=reserved_solved_tasks,
                prior_failed_task_signatures=prior_failed,
                node_evals=node_evals,
            )
            f_ba = pair_features(
                b, a,
                task_stats=task_stats,
                reserved_solved_tasks=reserved_solved_tasks,
                prior_failed_task_signatures=prior_failed,
                node_evals=node_evals,
            )
            out.append((a, b, f_ab, f_ba))
    return out


def _ensure_auto_tune(
    gate_config: MergeGateConfig,
    pairs_with_features: list[tuple[tree.Node, tree.Node, MergePairFeatures, MergePairFeatures]],
) -> MergeGateConfig:
    """If `gate_config` is the auto kind and has no auto_tune yet, derive
    it from `pairs_with_features` and return a new config with the stats
    attached. Otherwise return the input unchanged.
    """
    if gate_config.gate_kind != "auto" or gate_config.auto_tune is not None:
        return gate_config
    all_features: list[MergePairFeatures] = []
    for _a, _b, f_ab, f_ba in pairs_with_features:
        all_features.append(f_ab)
        all_features.append(f_ba)
    auto = derive_auto_tune_stats(all_features)
    return dataclasses.replace(gate_config, auto_tune=auto)


def select_probe_pair(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    pipeline_id: str,
    score_slack: float | None = None,
    exclude_pair_keys: set[tuple[str, str]] | None = None,
    eligible_node_ids: set[str] | None = None,
) -> MergePair | None:
    """Pick the highest-utility candidate pair regardless of any gate.

    Used by the supervisor's "merge probe" escape valve: after warmup,
    if every pair is rejected by `MergeGateConfig`, the probe valve
    fires ONE pair to seed the campaign with empirical merge evidence
    that the auto-tuner can then incorporate. Without this valve, an
    overly strict gate at startup can deadlock the merge pool for the
    entire campaign.

    Returns None when there are no candidate pairs (no merge work
    available), in which case the supervisor falls through and skips
    the probe.
    """
    pairs_with_features = _collect_pair_features(
        conn,
        campaign=campaign,
        subset=subset,
        score_slack=score_slack,
        exclude_pair_keys=exclude_pair_keys,
        eligible_node_ids=eligible_node_ids,
    )
    if not pairs_with_features:
        return None
    pairs = [(a, b) for a, b, _f_ab, _f_ba in pairs_with_features]
    reserved_solved_tasks = _reserved_solved_tasks(
        conn, campaign=campaign, pair_keys=exclude_pair_keys or set(),
    )
    task_stats = tree.task_outcome_stats(conn, campaign=campaign, subset=subset)
    prior_failed = _failed_merge_task_signatures(conn, campaign=campaign)
    node_evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    chosen = _select_rule_from_pairs(
        pairs,
        pipeline_id=pipeline_id,
        reserved_solved_tasks=reserved_solved_tasks,
        task_stats=task_stats,
        prior_failed_task_signatures=prior_failed,
        node_evals=node_evals,
    )
    return MergePair(chosen.primary, chosen.secondary, "probe")


def select_pair_rule_based(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    pipeline_id: str,
    score_slack: float | None = None,
    exclude_pair_keys: set[tuple[str, str]] | None = None,
    eligible_node_ids: set[str] | None = None,
    gate_config: MergeGateConfig | None = None,
) -> MergePair | None:
    pairs = eligible_pairs(
        conn,
        campaign=campaign,
        subset=subset,
        score_slack=score_slack,
        exclude_pair_keys=exclude_pair_keys,
        eligible_node_ids=eligible_node_ids,
        gate_config=gate_config,
    )
    if not pairs:
        return None
    reserved_solved_tasks = _reserved_solved_tasks(
        conn, campaign=campaign, pair_keys=exclude_pair_keys or set(),
    )
    task_stats = tree.task_outcome_stats(conn, campaign=campaign, subset=subset)
    prior_failed = _failed_merge_task_signatures(conn, campaign=campaign)
    node_evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    return _select_rule_from_pairs(
        pairs,
        pipeline_id=pipeline_id,
        reserved_solved_tasks=reserved_solved_tasks,
        task_stats=task_stats,
        prior_failed_task_signatures=prior_failed,
        node_evals=node_evals,
    )


def _select_rule_from_pairs(
    pairs: list[tuple[tree.Node, tree.Node]], *,
    pipeline_id: str,
    reserved_solved_tasks: set[str] | None = None,
    task_stats: dict[str, tree.TaskOutcomeStats] | None = None,
    prior_failed_task_signatures: dict[frozenset[str], int] | None = None,
    node_evals: dict[str, tree.NodeEval] | None = None,
) -> MergePair:
    reserved_solved_tasks = set(reserved_solved_tasks or set())
    task_stats = task_stats or {}
    prior_failed_task_signatures = prior_failed_task_signatures or {}
    scored: list[tuple[float, float, float, float, float, float, tree.Node, tree.Node]] = []
    for a, b in pairs:
        orientations = ((a, b), (b, a))
        for primary, secondary in orientations:
            features = pair_features(
                primary,
                secondary,
                task_stats=task_stats,
                reserved_solved_tasks=reserved_solved_tasks,
                prior_failed_task_signatures=prior_failed_task_signatures,
                node_evals=node_evals,
            )
            shared_failures = len(features.shared_failures)
            combined_score = (_score(primary, node_evals) or 0.0) + (
                _score(secondary, node_evals) or 0.0
            )
            scored.append((
                -features.expected_utility,
                _orientation_risk(primary, secondary, node_evals=node_evals),
                _validation_cost(primary, secondary, node_evals=node_evals),
                -features.diversity_gain,
                -combined_score,
                shared_failures,
                primary,
                secondary,
            ))
    scored.sort(key=lambda item: item[:6])
    best_key = scored[0][:6]
    tied = [item for item in scored if item[:6] == best_key]
    chosen = _sample_tied_pair(tied, pipeline_id=pipeline_id)
    return MergePair(chosen[6], chosen[7], "rule")


def select_pair(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    pipeline_id: str,
    score_slack: float | None = None,
    gate_config: MergeGateConfig | None = None,
    workspace: Path | None = None,
    prompt_dir: Path | None = None,
    cursor_log_path: Path | None = None,
    cursor_model: str | None = None,
    cursor_timeout_s: int = cursor_agent.DEFAULT_TIMEOUT_S,
    meta_model: str | None = None,
    meta_effort: str | None = None,
    exclude_pair_keys: set[tuple[str, str]] | None = None,
    eligible_node_ids: set[str] | None = None,
) -> MergePair | None:
    pairs = eligible_pairs(
        conn,
        campaign=campaign,
        subset=subset,
        score_slack=score_slack,
        exclude_pair_keys=exclude_pair_keys,
        eligible_node_ids=eligible_node_ids,
        gate_config=gate_config,
    )
    if not pairs:
        return None
    reserved_solved_tasks = _reserved_solved_tasks(
        conn, campaign=campaign, pair_keys=exclude_pair_keys or set(),
    )
    task_stats = tree.task_outcome_stats(conn, campaign=campaign, subset=subset)
    prior_failed = _failed_merge_task_signatures(conn, campaign=campaign)
    node_evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    if _coin_agent(pipeline_id) and workspace and prompt_dir and cursor_log_path:
        chosen = select_pair_with_agent(
            pairs=pairs,
            pipeline_id=pipeline_id,
            reserved_solved_tasks=reserved_solved_tasks,
            task_stats=task_stats,
            prior_failed_task_signatures=prior_failed,
            node_evals=node_evals,
            workspace=workspace,
            prompt_dir=prompt_dir,
            cursor_log_path=cursor_log_path,
            cursor_model=cursor_model or run_config.load_cursor_model_from_config(),
            cursor_timeout_s=cursor_timeout_s,
            meta_model=meta_model,
            meta_effort=meta_effort,
        )
        if chosen is not None:
            return chosen
    return _select_rule_from_pairs(
        pairs,
        pipeline_id=pipeline_id,
        reserved_solved_tasks=reserved_solved_tasks,
        task_stats=task_stats,
        prior_failed_task_signatures=prior_failed,
        node_evals=node_evals,
    )


def select_pair_with_agent(
    *,
    pairs: list[tuple[tree.Node, tree.Node]],
    pipeline_id: str,
    workspace: Path,
    prompt_dir: Path,
    cursor_log_path: Path,
    cursor_model: str,
    cursor_timeout_s: int,
    meta_model: str | None = None,
    meta_effort: str | None = None,
    reserved_solved_tasks: set[str] | None = None,
    task_stats: dict[str, tree.TaskOutcomeStats] | None = None,
    prior_failed_task_signatures: dict[frozenset[str], int] | None = None,
    node_evals: dict[str, tree.NodeEval] | None = None,
) -> MergePair | None:
    candidate_ids = {n.id: n for pair in pairs for n in pair}
    task_stats = task_stats or {}
    prior_failed_task_signatures = prior_failed_task_signatures or {}
    prompt_dir.mkdir(parents=True, exist_ok=True)
    compact = [
        {
            "id": n.id,
            "branch": n.branch_name,
            "score": _score(n, node_evals),
            "solved_tasks": _solved(n, node_evals),
            "unsolved_tasks": _failed(n, node_evals),
            "partially_solved_tasks": _partial(n, node_evals),
            "improved_tasks": _improved(n, node_evals),
            "regressed_tasks": _regressed(n, node_evals),
            "works_md_path": n.works_md_path,
            "effort_md_path": n.effort_md_path,
            "created_at": n.created_at,
        }
        for n in candidate_ids.values()
    ]
    allowed_pairs = [tuple(sorted((a.id, b.id))) for a, b in pairs]
    pair_summaries = [
        _pair_summary_json(
            a,
            b,
            task_stats=task_stats,
            reserved_solved_tasks=reserved_solved_tasks or set(),
            prior_failed_task_signatures=prior_failed_task_signatures,
            node_evals=node_evals,
        )
        for a, b in pairs
    ]
    prompt = cursor_agent.render_prompt(
        Path(__file__).resolve().parent / "prompts" / "merge_select.md",
        {
            "pipeline_id": pipeline_id,
            "candidates_json": json.dumps(compact, indent=2, sort_keys=True),
            "allowed_pairs_json": json.dumps(allowed_pairs, indent=2),
            "pair_summaries_json": json.dumps(pair_summaries, indent=2, sort_keys=True),
            "reserved_solved_tasks_json": json.dumps(
                sorted(reserved_solved_tasks or set()),
                indent=2,
            ),
        },
    )
    (prompt_dir / "merge_select.md").write_text(prompt)
    # Route through the active proposer backend (META_AGENT). meta_model/meta_effort
    # are resolved upstream by the supervisor; with the default cursor backend they
    # are None, so this falls back to cursor_model with no effort flag (unchanged).
    result = meta_agent.run(
        prompt,
        workspace=workspace,
        log_path=cursor_log_path,
        model=meta_model or cursor_model,
        plan_mode=True,
        timeout_s=cursor_timeout_s,
        reasoning_effort=meta_effort,
    )
    if result.error:
        return None
    parsed = parse_agent_pair(result.text or "")
    if parsed is None:
        return None
    key = tuple(sorted(parsed))
    if key not in set(allowed_pairs):
        return None
    a = candidate_ids[parsed[0]]
    b = candidate_ids[parsed[1]]
    return MergePair(a, b, "agent")


def parse_agent_pair(text: str) -> tuple[str, str] | None:
    start = text.find("<<<MERGE_PAIR>>>")
    if start < 0:
        return None
    tail = text[start + len("<<<MERGE_PAIR>>>"):].strip().splitlines()
    if not tail:
        return None
    parts = tail[0].strip().replace(",", " ").split()
    if len(parts) < 2:
        return None
    if parts[0] == parts[1]:
        return None
    return parts[0], parts[1]


def _ancestor_map(conn: sqlite3.Connection, *, campaign: str) -> dict[str, set[str]]:
    parents_by_child: dict[str, set[str]] = {}
    nodes = tree.list_nodes(conn, campaign=campaign)
    for node in nodes:
        if node.parent_id:
            parents_by_child.setdefault(node.id, set()).add(node.parent_id)
    for edge in tree.list_node_edges(conn, campaign=campaign):
        parents_by_child.setdefault(edge.child_id, set()).add(edge.parent_id)

    cache: dict[str, set[str]] = {}

    def ancestors(node_id: str, seen: set[str] | None = None) -> set[str]:
        if node_id in cache:
            return set(cache[node_id])
        seen = set(seen or set())
        if node_id in seen:
            return set()
        seen.add(node_id)
        out: set[str] = set()
        for parent_id in parents_by_child.get(node_id, set()):
            out.add(parent_id)
            out.update(ancestors(parent_id, seen))
        cache[node_id] = set(out)
        return out

    return {node.id: ancestors(node.id) for node in nodes}


def _has_complementary_solved_tasks(
    a: tree.Node,
    b: tree.Node,
    *,
    node_evals: dict[str, tree.NodeEval] | None = None,
) -> bool:
    solved_a = set(_solved(a, node_evals))
    solved_b = set(_solved(b, node_evals))
    return bool(solved_a - solved_b) and bool(solved_b - solved_a)


def _gate_reasons(
    features: MergePairFeatures,
    gate_config: MergeGateConfig,
) -> list[str]:
    """Per-pair gate decision. Dispatches by `gate_config.gate_kind`:
    "auto" uses the population-derived `auto_tune` stats (or absolute
    fallbacks when `auto_tune` is None / `pair_count < min`), "fixed"
    uses the legacy hand-tuned rates.

    Both return the list of REJECTION reasons; an empty list means the
    pair passes the gate.
    """
    if gate_config.gate_kind == "auto":
        reasons = _gate_reasons_auto(features, gate_config)
    else:
        reasons = _gate_reasons_fixed(features, gate_config)
    # Specialist contract (DARWINX_GATE_SPECIALIST_CONTRACT): fragile PERIPHERY wins are
    # tradeable, not a veto — the net-positive-on-periphery rule is enforced at
    # eval time by preserve_extend.specialist_preserved, and the contract already
    # surfaced these tasks to the merge proposer as deliberately sacrificeable. So
    # drop the periphery-driven rejection reasons here. CORE regressions are NOT
    # relaxed: they still surface via parent_regression_rate / low_union_gain and
    # the eval-time core-loss hard reject. Default OFF => reasons unchanged.
    if reasons and os.environ.get("DARWINX_GATE_SPECIALIST_CONTRACT", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        _periphery_only = {"high_fragile_parent_win_rate", "parents_too_fragile"}
        reasons = [r for r in reasons if r not in _periphery_only]
    return reasons


def _gate_reasons_fixed(
    features: MergePairFeatures,
    gate_config: MergeGateConfig,
) -> list[str]:
    """Legacy gate (pre-auto-tune). Kept for backward compatibility when
    the user explicitly opts in with `--merge-gate-kind=fixed`."""
    reasons: list[str] = []
    if features.normalized_expected_utility < gate_config.min_normalized_utility:
        reasons.append("low_normalized_utility")
    if (
        not gate_config.allow_required_new_wins
        and features.minimum_new_wins_needed > 0
    ):
        reasons.append("requires_new_wins")
    if features.parent_regression_rate > gate_config.max_parent_regression_rate:
        reasons.append("high_parent_regression_rate")
    if features.fragile_parent_win_rate > gate_config.max_fragile_parent_win_rate:
        reasons.append("high_fragile_parent_win_rate")
    risk_rate = max(features.parent_regression_rate, features.fragile_parent_win_rate)
    if (
        features.union_gain_rate + gate_config.min_union_gain_rate_over_risk
        <= risk_rate
    ):
        reasons.append("union_gain_not_above_risk")
    return reasons


def _gate_reasons_auto(
    features: MergePairFeatures,
    gate_config: MergeGateConfig,
) -> list[str]:
    """Auto gate: population-derived thresholds + absolute union-gain floor.

    All threshold comparisons are STRICT `>` (rejected) / `<` (rejected)
    so a pair lying exactly at the boundary still passes. This is what
    keeps the auto gate at-least-as-permissive as the fixed gate when
    the candidate population is small (and the safety floor dominates).
    """
    floor_new_wins = _resolve_auto_floor_new_wins(features, gate_config)
    auto = gate_config.auto_tune
    if auto is not None:
        effective_regress = auto.effective_parent_regression_rate
        effective_fragile = auto.effective_fragile_parent_win_rate
        effective_shared = auto.effective_shared_failure_rate
    else:
        effective_regress = AUTO_SAFETY_FLOOR_PARENT_REGRESSION_RATE
        effective_fragile = AUTO_SAFETY_FLOOR_FRAGILE_PARENT_WIN_RATE
        effective_shared = AUTO_SAFETY_FLOOR_SHARED_FAILURE_RATE
    reasons: list[str] = []
    if features.union_gain_over_best_parent < floor_new_wins:
        reasons.append("low_union_gain")
    if features.shared_failure_rate > effective_shared:
        reasons.append("parents_solve_same_problems")
    if features.parent_regression_rate > effective_regress:
        reasons.append("parents_too_regressive")
    if features.fragile_parent_win_rate > effective_fragile:
        reasons.append("parents_too_fragile")
    if features.expected_utility < 0:
        reasons.append("negative_expected_utility")
    return reasons


def _resolve_auto_floor_new_wins(
    features: MergePairFeatures,
    gate_config: MergeGateConfig,
) -> int:
    """Pick the absolute "this pair must add at least N new wins" floor.

    Precedence:
      1. User-provided `gate_config.floor_new_wins` if set (overrides
         everything).
      2. The auto-tuner's `floor_new_wins` if population stats are
         available.
      3. The conservative absolute fallback computed from N.
    """
    if gate_config.floor_new_wins is not None:
        return max(0, int(gate_config.floor_new_wins))
    if gate_config.auto_tune is not None:
        return gate_config.auto_tune.floor_new_wins
    return _absolute_floor_new_wins(features.observed_task_count)


def _absolute_floor_new_wins(observed_task_count: int) -> int:
    """Conservative absolute fallback when the population is too small
    for p25(union_gain_over_best_parent) to mean anything.

    `max(AUTO_FLOOR_NEW_WINS_MIN, ceil(N * AUTO_ABSOLUTE_FLOOR_NEW_WINS_PCT))`
    gives 2 for N <= 400, 5 for N=1000, 10 for N=2000. Adapts to the
    task pool size without requiring any population evidence.
    """
    abs_floor = math.ceil(
        max(0, observed_task_count) * AUTO_ABSOLUTE_FLOOR_NEW_WINS_PCT
    )
    return max(AUTO_FLOOR_NEW_WINS_MIN, abs_floor)


def _percentile(values: list[float], p: float) -> float:
    """Plain linear-interpolation percentile (no numpy)."""
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return float(vs[0])
    idx = (len(vs) - 1) * max(0.0, min(1.0, p))
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(vs[lo])
    return float(vs[lo] + (vs[hi] - vs[lo]) * (idx - lo))


def derive_auto_tune_stats(
    pair_features_list: list[MergePairFeatures],
) -> MergeAutoTuneStats:
    """Compute the campaign-derived gate thresholds from the live
    candidate-pair population.

    Empty / tiny populations (< `AUTO_TUNE_MIN_PAIRS`) fall back to the
    safety floors so the auto gate degrades gracefully into "behaves
    like the historical fixed gate" instead of producing wild p25/p75
    values from 1-2 samples. The caller is expected to call
    `pair_features` once per pair and pass the resulting list here.
    """
    observed_task_count = max(
        (f.observed_task_count for f in pair_features_list),
        default=0,
    )
    abs_floor = _absolute_floor_new_wins(observed_task_count)
    if len(pair_features_list) < AUTO_TUNE_MIN_PAIRS:
        return MergeAutoTuneStats(
            floor_new_wins=abs_floor,
            floor_new_wins_source="abs_fallback",
            p25_union_gain_raw=0.0,
            abs_fallback_floor_new_wins=abs_floor,
            effective_parent_regression_rate=AUTO_SAFETY_FLOOR_PARENT_REGRESSION_RATE,
            effective_fragile_parent_win_rate=AUTO_SAFETY_FLOOR_FRAGILE_PARENT_WIN_RATE,
            effective_shared_failure_rate=AUTO_SAFETY_FLOOR_SHARED_FAILURE_RATE,
            raw_p75_parent_regression_rate=0.0,
            raw_p75_fragile_parent_win_rate=0.0,
            raw_p75_shared_failure_rate=0.0,
            pair_count=len(pair_features_list),
            observed_task_count=observed_task_count,
        )
    union_gains = [float(f.union_gain_over_best_parent) for f in pair_features_list]
    parent_regs = [float(f.parent_regression_rate) for f in pair_features_list]
    fragiles = [float(f.fragile_parent_win_rate) for f in pair_features_list]
    shareds = [float(f.shared_failure_rate) for f in pair_features_list]
    p25_union = _percentile(union_gains, 0.25)
    p75_reg = _percentile(parent_regs, 0.75)
    p75_frag = _percentile(fragiles, 0.75)
    p75_shared = _percentile(shareds, 0.75)
    pop_floor = max(AUTO_FLOOR_NEW_WINS_MIN, int(math.floor(p25_union)))
    return MergeAutoTuneStats(
        floor_new_wins=pop_floor,
        floor_new_wins_source="p25",
        p25_union_gain_raw=p25_union,
        abs_fallback_floor_new_wins=abs_floor,
        effective_parent_regression_rate=max(
            AUTO_SAFETY_FLOOR_PARENT_REGRESSION_RATE, p75_reg,
        ),
        effective_fragile_parent_win_rate=max(
            AUTO_SAFETY_FLOOR_FRAGILE_PARENT_WIN_RATE, p75_frag,
        ),
        effective_shared_failure_rate=max(
            AUTO_SAFETY_FLOOR_SHARED_FAILURE_RATE, p75_shared,
        ),
        raw_p75_parent_regression_rate=p75_reg,
        raw_p75_fragile_parent_win_rate=p75_frag,
        raw_p75_shared_failure_rate=p75_shared,
        pair_count=len(pair_features_list),
        observed_task_count=observed_task_count,
    )


def pair_allowed_by_gate(
    primary: tree.Node,
    secondary: tree.Node,
    *,
    gate_config: MergeGateConfig,
    task_stats: dict[str, tree.TaskOutcomeStats] | None = None,
    reserved_solved_tasks: set[str] | None = None,
    prior_failed_task_signatures: dict[frozenset[str], int] | None = None,
) -> tuple[bool, list[str], MergePairFeatures]:
    features = pair_features(
        primary,
        secondary,
        task_stats=task_stats or {},
        reserved_solved_tasks=reserved_solved_tasks or set(),
        prior_failed_task_signatures=prior_failed_task_signatures or {},
    )
    reasons = _gate_reasons(features, gate_config)
    return not reasons, reasons, features


def pair_diagnostics(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    score_slack: float | None = None,
    exclude_pair_keys: set[tuple[str, str]] | None = None,
    eligible_node_ids: set[str] | None = None,
    gate_config: MergeGateConfig | None = None,
    limit: int = 5,
) -> MergePairDiagnostics:
    """Per-cycle gate diagnostics. Derives auto-tune stats from the
    pre-gate population so the user can see the effective thresholds in
    the supervisor log alongside per-pair rejections.
    """
    pairs_with_features = _collect_pair_features(
        conn,
        campaign=campaign,
        subset=subset,
        score_slack=score_slack,
        exclude_pair_keys=exclude_pair_keys,
        eligible_node_ids=eligible_node_ids,
    )
    gate_config = gate_config or MergeGateConfig()
    effective_gate = _ensure_auto_tune(gate_config, pairs_with_features)
    rejection_counts: dict[str, int] = {}
    top_pairs: list[dict] = []
    gated_pairs = 0
    for a, b, f_ab, f_ba in pairs_with_features:
        orientation_rows = []
        pair_allowed = False
        for primary, secondary, features in ((a, b, f_ab), (b, a, f_ba)):
            reasons = _gate_reasons(features, effective_gate)
            allowed = not reasons
            pair_allowed = pair_allowed or allowed
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            orientation_rows.append({
                "primary": primary.id,
                "secondary": secondary.id,
                "allowed": allowed,
                "reasons": reasons,
                "expected_utility": round(features.expected_utility, 4),
                "normalized_expected_utility": round(features.normalized_expected_utility, 4),
                "union_gain_rate": round(features.union_gain_rate, 4),
                "union_gain_over_best_parent": features.union_gain_over_best_parent,
                "fragile_parent_win_rate": round(features.fragile_parent_win_rate, 4),
                "parent_regression_rate": round(features.parent_regression_rate, 4),
                "shared_failure_rate": round(features.shared_failure_rate, 4),
                "minimum_new_wins_needed": features.minimum_new_wins_needed,
                "solved_union_size": features.solved_union_size,
                "observed_task_count": features.observed_task_count,
            })
        if pair_allowed:
            gated_pairs += 1
        top_pairs.extend(orientation_rows)
    top_pairs.sort(
        key=lambda row: (
            not row["allowed"],
            -float(row["normalized_expected_utility"]),
            -float(row["union_gain_rate"]),
        )
    )
    return MergePairDiagnostics(
        total_pairs=len(pairs_with_features),
        gated_pairs=gated_pairs,
        rejection_counts=rejection_counts,
        top_pairs=top_pairs[:limit],
        auto_tune=effective_gate.auto_tune,
    )


def _reserved_solved_tasks(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    pair_keys: set[tuple[str, str]],
) -> set[str]:
    if not pair_keys:
        return set()
    node_ids = {node_id for pair in pair_keys for node_id in pair}
    nodes = {n.id: n for n in tree.list_nodes(conn, campaign=campaign)}
    node_evals = tree.search_eval_by_node(conn, campaign=campaign)
    solved: set[str] = set()
    for node_id in node_ids:
        node = nodes.get(node_id)
        if node is not None:
            solved.update(_solved(node, node_evals))
    return solved


def pair_features(
    primary: tree.Node,
    secondary: tree.Node,
    *,
    task_stats: dict[str, tree.TaskOutcomeStats] | None = None,
    reserved_solved_tasks: set[str] | None = None,
    prior_failed_task_signatures: dict[frozenset[str], int] | None = None,
    node_evals: dict[str, tree.NodeEval] | None = None,
) -> MergePairFeatures:
    task_stats = task_stats or {}
    reserved_solved_tasks = set(reserved_solved_tasks or set())
    prior_failed_task_signatures = prior_failed_task_signatures or {}
    primary_solved = set(_solved(primary, node_evals))
    secondary_solved = set(_solved(secondary, node_evals))
    primary_unsolved = set(_failed(primary, node_evals))
    secondary_unsolved = set(_failed(secondary, node_evals))
    solved_union = primary_solved | secondary_solved
    primary_unique = primary_solved - secondary_solved
    secondary_unique = secondary_solved - primary_solved
    complementary = sorted(
        (primary_unique & secondary_unsolved) |
        (secondary_unique & primary_unsolved)
    )
    shared_failures = sorted(primary_unsolved & secondary_unsolved)
    fragile = sorted(
        task for task in solved_union
        if _task_fragility(task_stats.get(task)) > 0.35
    )
    parent_regression_surface = len(_regressed(primary, node_evals)) + len(_regressed(secondary, node_evals))
    prior_failed = prior_failed_task_signatures.get(frozenset(solved_union), 0)
    diversity_gain = len(solved_union - reserved_solved_tasks)
    union_gain_over_best_parent = len(solved_union) - max(len(primary_solved), len(secondary_solved))
    observed_task_count = len(solved_union | primary_unsolved | secondary_unsolved)
    task_value = 1.0 / observed_task_count if observed_task_count else 0.0
    parent_score_ceiling = max(
        _score(primary, node_evals) or 0.0,
        _score(secondary, node_evals) or 0.0,
    )
    minimum_new_wins_needed = max(
        0,
        int(parent_score_ceiling * observed_task_count) + 1 - len(solved_union),
    ) if observed_task_count else 0
    acceptability_risk = (
        0.80 * max(0, minimum_new_wins_needed)
        + 0.45 * max(0, 1 - union_gain_over_best_parent)
        + 0.20 * len(fragile)
    )
    coverage_value = _coverage_value(solved_union, task_stats)
    validation_cost = max(0, len(solved_union) - 12) * 0.05
    risk = (
        sum(_task_fragility(task_stats.get(task)) for task in solved_union)
        + 0.35 * len(shared_failures)
        + 0.20 * parent_regression_surface
        + 1.30 * prior_failed
        + validation_cost
        + acceptability_risk
    )
    expected_utility = (
        len(solved_union)
        + 0.75 * len(complementary)
        + 0.20 * diversity_gain
        + 0.15 * coverage_value
        - risk
    )
    normalized_expected_utility = (
        expected_utility / observed_task_count if observed_task_count else 0.0
    )
    solved_union_size = len(solved_union)
    union_gain_rate = (
        union_gain_over_best_parent / observed_task_count
        if observed_task_count else 0.0
    )
    fragile_parent_win_rate = (
        len(fragile) / solved_union_size if solved_union_size else 0.0
    )
    parent_regression_rate = (
        parent_regression_surface / solved_union_size if solved_union_size else 0.0
    )
    shared_failure_rate = (
        len(shared_failures) / observed_task_count if observed_task_count else 0.0
    )
    return MergePairFeatures(
        primary_unique_wins=sorted(primary_unique),
        secondary_unique_wins=sorted(secondary_unique),
        complementary_wins=complementary,
        shared_failures=shared_failures,
        fragile_parent_wins=fragile,
        parent_regression_surface=parent_regression_surface,
        prior_failed_similar_merges=prior_failed,
        diversity_gain=diversity_gain,
        solved_union_size=solved_union_size,
        union_gain_over_best_parent=union_gain_over_best_parent,
        minimum_new_wins_needed=minimum_new_wins_needed,
        acceptability_risk=acceptability_risk,
        coverage_value=coverage_value,
        expected_utility=expected_utility,
        observed_task_count=observed_task_count,
        task_value=task_value,
        normalized_expected_utility=normalized_expected_utility,
        union_gain_rate=union_gain_rate,
        fragile_parent_win_rate=fragile_parent_win_rate,
        parent_regression_rate=parent_regression_rate,
        shared_failure_rate=shared_failure_rate,
        parent_score_ceiling=parent_score_ceiling,
    )


def _task_fragility(stats: tree.TaskOutcomeStats | None) -> float:
    if stats is None:
        return 0.0
    return (
        0.30 * stats.failure_rate
        + 0.50 * stats.regression_rate
        + 0.15 * stats.merge_failures
        + 0.10 * stats.regression_resolver_failures
    )


def _coverage_value(
    tasks: set[str],
    task_stats: dict[str, tree.TaskOutcomeStats],
) -> float:
    if not tasks:
        return 0.0
    families = {_task_family(t) for t in tasks}
    undercovered = 0.0
    for task in tasks:
        stats = task_stats.get(task)
        undercovered += 1.0 / (1.0 + (stats.total_evals if stats else 0))
    return undercovered + 0.25 * len(families)


def _task_family(task: str) -> str:
    return str(task).split("-", 1)[0] or str(task)


def _orientation_risk(
    primary: tree.Node,
    secondary: tree.Node,
    *,
    node_evals: dict[str, tree.NodeEval] | None = None,
) -> float:
    primary_regressions = len(_regressed(primary, node_evals))
    secondary_regressions = len(_regressed(secondary, node_evals))
    primary_unique = len(set(_solved(primary, node_evals)) - set(_solved(secondary, node_evals)))
    secondary_unique = len(set(_solved(secondary, node_evals)) - set(_solved(primary, node_evals)))
    score_gap = (_score(secondary, node_evals) or 0.0) - (_score(primary, node_evals) or 0.0)
    return (
        0.50 * primary_regressions
        + 0.10 * max(0.0, score_gap)
        + 0.03 * max(0, primary_unique - secondary_unique)
        - 0.10 * secondary_regressions
    )


def _validation_cost(
    primary: tree.Node,
    secondary: tree.Node,
    *,
    node_evals: dict[str, tree.NodeEval] | None = None,
) -> float:
    return float(len(set(_solved(primary, node_evals)) | set(_solved(secondary, node_evals))))


def _pair_summary_json(
    primary: tree.Node,
    secondary: tree.Node,
    *,
    task_stats: dict[str, tree.TaskOutcomeStats],
    reserved_solved_tasks: set[str],
    prior_failed_task_signatures: dict[frozenset[str], int],
    node_evals: dict[str, tree.NodeEval] | None = None,
) -> dict:
    features = pair_features(
        primary,
        secondary,
        task_stats=task_stats,
        reserved_solved_tasks=reserved_solved_tasks,
        prior_failed_task_signatures=prior_failed_task_signatures,
        node_evals=node_evals,
    )
    return {
        "pair": sorted([primary.id, secondary.id]),
        "primary": primary.id,
        "secondary": secondary.id,
        "orientation_risk": round(_orientation_risk(primary, secondary, node_evals=node_evals), 4),
        "reverse_orientation_risk": round(_orientation_risk(secondary, primary, node_evals=node_evals), 4),
        "expected_utility": round(features.expected_utility, 4),
        "normalized_expected_utility": round(features.normalized_expected_utility, 4),
        "observed_task_count": features.observed_task_count,
        "task_value": round(features.task_value, 6),
        "primary_unique_wins": features.primary_unique_wins,
        "secondary_unique_wins": features.secondary_unique_wins,
        "complementary_wins": features.complementary_wins,
        "shared_failures": features.shared_failures,
        "fragile_parent_wins": features.fragile_parent_wins,
        "fragile_parent_win_rate": round(features.fragile_parent_win_rate, 4),
        "parent_regression_surface": features.parent_regression_surface,
        "parent_regression_rate": round(features.parent_regression_rate, 4),
        "shared_failure_rate": round(features.shared_failure_rate, 4),
        "prior_failed_similar_merges": features.prior_failed_similar_merges,
        "diversity_gain": features.diversity_gain,
        "union_gain_over_best_parent": features.union_gain_over_best_parent,
        "union_gain_rate": round(features.union_gain_rate, 4),
        "minimum_new_wins_needed": features.minimum_new_wins_needed,
        "parent_score_ceiling": round(features.parent_score_ceiling, 4),
        "acceptability_risk": round(features.acceptability_risk, 4),
        "coverage_value": round(features.coverage_value, 4),
        "validation_cost_tasks": int(_validation_cost(primary, secondary, node_evals=node_evals)),
    }


def _failed_merge_task_signatures(
    conn: sqlite3.Connection, *, campaign: str,
) -> dict[frozenset[str], int]:
    nodes = {n.id: n for n in tree.list_nodes(conn, campaign=campaign)}
    by_child: dict[str, list[str]] = {}
    for edge in tree.list_node_edges(conn, campaign=campaign, edge_type="merge"):
        by_child.setdefault(edge.child_id, []).append(edge.parent_id)
    out: dict[frozenset[str], int] = {}
    for child_id, parent_ids in by_child.items():
        child = nodes.get(child_id)
        parents = [nodes[p] for p in set(parent_ids) if p in nodes]
        if child is None or len(parents) < 2:
            continue
        parent_best = max((p.score or 0.0) for p in parents)
        if child.score is not None and child.score > parent_best and child.status == "completed":
            continue
        signature = frozenset(task for p in parents for task in p.solved_tasks)
        if signature:
            out[signature] = out.get(signature, 0) + 1
    return out


def _sample_tied_pair(
    tied: list[tuple[float, float, float, float, float, float, tree.Node, tree.Node]],
    *,
    pipeline_id: str,
) -> tuple[float, float, float, float, float, float, tree.Node, tree.Node]:
    if len(tied) == 1:
        return tied[0]
    pair_keys = [
        ",".join(sorted((primary.id, secondary.id)))
        for _, _, _, _, _, _, primary, secondary in tied
    ]
    seed = hashlib.sha256(
        f"{pipeline_id}:{'|'.join(sorted(pair_keys))}".encode()
    ).hexdigest()
    return random.Random(seed).choice(tied)


def _coin_agent(pipeline_id: str) -> bool:
    h = hashlib.sha256(f"{pipeline_id}:merge-agent-coin".encode()).hexdigest()
    return (int(h[0], 16) & 1) == 0


def _tiebreak_hash(pipeline_id: str, a: str, b: str) -> str:
    left, right = sorted((a, b))
    return hashlib.sha256(f"{pipeline_id}:{left}:{right}".encode()).hexdigest()


__all__ = [
    "MergeCandidate",
    "MergePair",
    "MergePairFeatures",
    "MergeGateConfig",
    "MergeAutoTuneStats",
    "MergePairDiagnostics",
    "AUTO_SAFETY_FLOOR_PARENT_REGRESSION_RATE",
    "AUTO_SAFETY_FLOOR_FRAGILE_PARENT_WIN_RATE",
    "AUTO_SAFETY_FLOOR_SHARED_FAILURE_RATE",
    "AUTO_TUNE_MIN_PAIRS",
    "AUTO_FLOOR_NEW_WINS_MIN",
    "derive_auto_tune_stats",
    "eligible_candidates",
    "eligible_pairs",
    "best_complementary_group",
    "pair_features",
    "pair_allowed_by_gate",
    "pair_diagnostics",
    "select_pair_rule_based",
    "select_pair",
    "select_pair_with_agent",
    "select_probe_pair",
    "parse_agent_pair",
]
