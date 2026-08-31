"""Shared unsolved-task pool for a `(campaign, subset)`.

This module replaces the per-parent failure_selection module. Every pipeline
in a campaign reads from a single shared pool whose membership is derived
from the **current best-scoring node**'s `failed_tasks`:

    pool = best_node.failed_tasks ∩ subset_filter  −  active_evolve_claims

Properties this gives us:
  - Concurrent evolvers never claim the same task (unique active claim
    per `(campaign, subset, claim_kind, failure_task)` enforced by the schema).
    Regression resolvers use their own claim kind and therefore do not drain
    this evolver pool.
  - When a pipeline finishes, `tree.release_claims(pipeline_id=…)` releases
    all of its claims. Tasks that the pipeline actually resolved drop out
    of the pool automatically because the child becomes the new best node
    (or an ancestor's failures shrank). Tasks that are still failing
    re-enter the pool the next time a worker queries it.
  - As soon as a child surpasses its parent's score, the pool's view of
    "what's still unsolved" follows along — no separate bookkeeping.

There is no explicit "return k tasks to the pool" step on pipeline exit:
the release-on-exit semantics + best-node-derived membership give the same
effect for free.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3

from . import tree


def best_node(
    conn: sqlite3.Connection, *, campaign: str, subset: str,
) -> tree.Node | None:
    """Return the campaign+subset's highest-scoring eligible node.

    Eligibility: `status in {completed, no_change}` AND `score IS NOT NULL`.
    Ties broken by `created_at ASC` so a deterministic earlier node wins.

    Returns None when no node has scored yet for this `(campaign, subset)` —
    the supervisor's bootstrap precursor will normally have scored the root
    before fan-out, so this is rare under healthy operation.
    """
    pair = tree.best_node_by_search_eval(
        conn, campaign=campaign, subset=subset,
    )
    if pair is not None:
        return pair[0]

    # Unit-test fallback for synthetic legacy nodes that have not been given
    # node_evals. Fresh campaigns should always take the branch above.
    nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    merge_children = {
        edge.child_id
        for edge in tree.list_node_edges(conn, campaign=campaign, edge_type="merge")
    }
    eligible = [
        n for n in nodes
        if n.status in {"completed", "no_change"} and n.score is not None
        and not (n.status == "no_change" and n.id in merge_children)
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda n: (-(n.score or 0.0), n.created_at))
    return eligible[0]


def unresolved_tasks(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    subset_filter: list[str] | None = None,
) -> list[str]:
    """Read-only preview of what `pick_and_claim` would be choosing from.

    Returns the list of task-only names that are
      1. Still failing under the current best node (per `best_node.failed_tasks`).
      2. Inside `subset_filter` when provided.
      3. Not currently held by an active evolver claim in this `(campaign, subset)`.

    Used by the supervisor preflight and the orchestrator's
    `_preflight_pool_is_empty` short-circuit.
    """
    best = best_node(conn, campaign=campaign, subset=subset)
    if best is None:
        return []
    best_eval = tree.node_search_eval(conn, campaign=campaign, node_id=best.id)
    failed = list(best_eval.failed_tasks if best_eval else best.failed_tasks)
    # FLAKY/recoverable tasks (0<avg@5<1) are the real headroom, but the
    # best-of-2 bootstrap marks them "solved" (any-pass), so they're ABSENT from
    # failed_tasks and would never be claimed. Union in the operator-supplied,
    # avg@5-derived variance list (DARWINX_EVAL_VARIANCE_TASKS) so the campaign
    # consolidates flaky->reliable (the smooth-gradient headroom) AND still
    # grinds the 0/k walls (failed_tasks) for chance gains. Variance first.
    variance = _variance_tasks()
    # The root's failures, minus what the best node has been measured to solve. Without this the pool
    # is only as wide as the best node's eval, and a campaign that scores nodes on a shared panel
    # (DARWINX_GATE_FIXED_EVAL_PANEL) would narrow to the panel's handful of failures the moment the first
    # child became best -- proposing against 7 tasks for the rest of the run while the subset it is
    # meant to be evolving against holds ~84. No log line would say so: claims keep succeeding.
    #
    # Self-limiting, so it needs no knob. If the best node's eval covers the whole subset then every
    # root failure is already in its failures or its solves, the union adds nothing, and this is
    # exactly the behaviour it replaces.
    root_failed: list[str] = []
    try:
        root_eval = tree.root_search_eval(conn, campaign=campaign)
        if root_eval is not None and root_eval.node_id != (best.id if best else None):
            solved_now = set(best_eval.solved_tasks if best_eval else best.solved_tasks)
            root_failed = [t for t in (root_eval.failed_tasks or []) if t not in solved_now]
    except Exception:  # noqa: BLE001 -- a wider pool is a nicety; never break claiming for it
        root_failed = []
    candidates = list(dict.fromkeys(list(variance) + list(failed) + root_failed))
    if subset_filter is not None:
        allowed = set(subset_filter)
        candidates = [t for t in candidates if t in allowed]
    # Operator denylist (DARWINX_EVAL_EXCLUDE_TASKS): hard-remove tasks that can
    # never pass under the current harness (e.g. the monet↔gateway tool-role-400
    # message-format bug that spuriously fails extract-moves-from-video) so the
    # campaign never burns a node on them. Always applied (claim-only; does not
    # touch eval/scoring). Substring match, mirroring the priority logic.
    excluded = _excluded_substrings()
    if excluded:
        candidates = [t for t in candidates if not _matches_priority(t, excluded)]
    if not candidates:
        return []
    claimed = {
        c.failure_task for c in tree.list_active_claims(
            conn, campaign=campaign, subset=subset, claim_kind="evolve",
        )
    }
    candidates = [t for t in candidates if t not in claimed]
    return candidates


def pick_and_claim(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    k: int,
    pipeline_id: str,
    subset_filter: list[str] | None = None,
    parent_id: str | None = None,
) -> list[str]:
    """Atomically claim up to `k` unresolved tasks for `pipeline_id`.

    Args:
        campaign / subset: the pool's scope; the unique active-claim index
            enforces "at most one evolver per (campaign, subset, task)".
        k: max number of tasks to claim per call.
        pipeline_id: claim owner (recorded on each row).
        subset_filter: optional task-name allow-list. When None, every task
            in the best node's `failed_tasks` is eligible.
        parent_id: informational column — which node the worker will branch
            from. Not part of any uniqueness constraint.

    Returns the list of tasks actually claimed in deterministic order
    (hash(pipeline_id, task)). Possibly empty if every candidate is held
    by a sibling pipeline or the pool is dry.
    """
    candidates = unresolved_tasks(
        conn, campaign=campaign, subset=subset, subset_filter=subset_filter,
    )
    if not candidates:
        return []
    # Curriculum-aware deterministic ordering: FLAKY/recoverable tasks (the
    # avg@5 variance band — "make a sometimes-pass reliable", the smooth-gradient
    # headroom) FIRST, then the harder 0/k walls (chance gains). Within each
    # bucket, keep the existing coverage/fragility/hash spread.
    variance = set(_variance_tasks())
    best = best_node(conn, campaign=campaign, subset=subset)
    best_eval = (
        tree.node_search_eval(conn, campaign=campaign, node_id=best.id)
        if best is not None else None
    )
    partial = set(
        best_eval.partially_solved_tasks if best_eval else best.partially_solved_tasks
    ) if best is not None else set()
    stats = tree.task_outcome_stats(conn, campaign=campaign, subset=subset)
    # Cluster-targeted seeding (self-evolve v2): an operator can steer the
    # campaign at the failing capability clusters (ML-distributed, bio
    # assembly/fitting, long-horizon graphics, low-level systems) by setting
    # DARWINX_EVAL_PRIORITY_TASKS to a comma-separated list of task-name
    # substrings (e.g. "torch,fasttext,gpt2,dna,protein,raman,path-tracing,
    # video,extract-elf,db-wal,mips"). Matching tasks sort first. Empty
    # (default) preserves the prior curriculum ordering exactly.
    priority = _priority_substrings()
    candidates.sort(
        key=lambda t: (
            0 if t in variance else 1,        # FLAKY (recoverable) first — curriculum
            0 if (priority and _matches_priority(t, priority)) else 1,
            0 if t in partial else 1,
            *_coverage_key(pipeline_id, t, stats.get(t)),
        )
    )
    # Cluster-batch claiming (DARWINX_EVAL_CLUSTER_CLAIM=1): draw the whole batch
    # from ONE capability cluster so a single coherent patch can address related
    # failure modes (a batch spanning unrelated tasks couples wins with
    # collateral). Parallel workers rotate across clusters (deterministic on
    # pipeline_id) for tree breadth. Falls back to the flat flaky-first order
    # when off or when no cluster has a claimable task.
    if _cluster_claim_on() and candidates:
        candidates = _restrict_to_one_cluster(candidates, pipeline_id)
    return tree.try_claim_tasks(
        conn,
        campaign=campaign,
        subset=subset,
        candidate_tasks=candidates,
        k=k,
        pipeline_id=pipeline_id,
        parent_id=parent_id,
        claim_kind="evolve",
    )


def _cluster_claim_on() -> bool:
    """Whether to draw each claimed batch from a single capability cluster
    (cluster-batch evolution). Gated by ``DARWINX_EVAL_CLUSTER_CLAIM``."""
    return os.environ.get("DARWINX_EVAL_CLUSTER_CLAIM", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# Capability clusters for cluster-batch claiming. A claimed batch is drawn from
# ONE cluster so one patch addresses related failure modes. Override with
# ``DARWINX_EVAL_CLUSTERS`` (JSON {name: [substrings]}). Tasks matching no cluster
# fall in "misc".
_DEFAULT_CLUSTERS: dict[str, list[str]] = {
    "ml-numerical": ["torch", "fasttext", "gpt2", "caffe", "cifar", "mteb",
                      "sam-cell", "model-extraction", "mcmc", "raman", "relu",
                      "logits", "leaderboard"],
    "systems-lowlevel": ["mips", "doom", "elf", "corewars", "compcert", "qemu",
                          "-gc", "kernel", "assembler"],
    "bio-assembly": ["dna", "protein", "assembly"],
    "parse-text-tools": ["html", "gcode", "regex", "sanitize", "hbox", "chess",
                         "compressor", "video", "json", "metacircular", "scheme"],
    "db-data": ["db-wal", "db-", "sql", "parquet", "query-optimize", "wal"],
}


def _capability_clusters() -> dict[str, list[str]]:
    raw = os.environ.get("DARWINX_EVAL_CLUSTERS", "").strip()
    if raw:
        try:
            import json
            d = json.loads(raw)
            if isinstance(d, dict) and d:
                return {k: [str(s).lower() for s in v] for k, v in d.items()}
        except Exception:  # noqa: BLE001 — bad override falls back to defaults
            pass
    return _DEFAULT_CLUSTERS


def _task_cluster(task: str, clusters: dict[str, list[str]]) -> str:
    t = task.lower()
    for name, subs in clusters.items():
        if any(s in t for s in subs):
            return name
    return "misc"


def _restrict_to_one_cluster(candidates: list[str], pipeline_id: str) -> list[str]:
    """Return only the candidates belonging to a single chosen cluster, keeping
    the incoming (flaky-first) order. The cluster is chosen by rotating on
    ``pipeline_id`` across clusters that have >=1 candidate, so parallel workers
    spread across clusters. Preserves order within the cluster."""
    clusters = _capability_clusters()
    grouped: dict[str, list[str]] = {}
    for t in candidates:
        grouped.setdefault(_task_cluster(t, clusters), []).append(t)
    names = [c for c in grouped if grouped[c]]
    if not names:
        return candidates
    names.sort()  # stable order before rotation
    idx = int(hashlib.sha256(pipeline_id.encode()).hexdigest(), 16) % len(names)
    chosen = names[idx]
    return grouped[chosen]


def _priority_substrings() -> list[str]:
    """Operator-supplied capability-cluster substrings for targeted seeding.

    Read from ``DARWINX_EVAL_PRIORITY_TASKS`` (comma-separated). Empty/unset
    disables cluster targeting (prior curriculum ordering is used verbatim).
    """
    raw = os.environ.get("DARWINX_EVAL_PRIORITY_TASKS", "")
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def _variance_tasks() -> list[str]:
    """Operator-supplied FLAKY/recoverable task list (exact task-base names),
    derived from the baseline's avg@5 per-task pass-rates (0<rate<1) — NOT the
    best-of-2 bootstrap, which hides flaky tasks as "solved". Read from
    ``DARWINX_EVAL_VARIANCE_TASKS`` (comma-separated). These are unioned into the
    claim pool (so they're claimable even though best-of-2 marks them solved) and
    sorted FIRST (consolidate sometimes-pass -> reliable: the smooth-gradient
    headroom). Empty/unset = no variance targeting (walls-only, legacy)."""
    raw = os.environ.get("DARWINX_EVAL_VARIANCE_TASKS", "")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _matches_priority(task: str, priority: list[str]) -> bool:
    t = task.lower()
    return any(sub in t for sub in priority)


def _excluded_substrings() -> list[str]:
    """Operator claim-only denylist substrings.

    Read from ``DARWINX_EVAL_EXCLUDE_TASKS`` (comma-separated, substring match).
    Tasks matching any substring are removed from the claim pool entirely.
    Empty/unset disables exclusion. Unlike the eval-level ``harbor.exclude_tasks``
    this does NOT change the scoring denominator — it only stops the campaign
    from CLAIMING the task.
    """
    raw = os.environ.get("DARWINX_EVAL_EXCLUDE_TASKS", "")
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def _variance_band_enabled() -> bool:
    """Whether to restrict claiming to the recoverable-variance (partial) band.

    Gated by ``DARWINX_EVAL_CLAIM_VARIANCE_BAND`` (``1``/``true``/``yes`` → on).
    Default off so shared callers and other campaigns are unaffected.
    """
    return os.environ.get("DARWINX_EVAL_CLAIM_VARIANCE_BAND", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _hash(pipeline_id: str, task: str) -> str:
    return hashlib.sha256(f"{pipeline_id}|{task}".encode()).hexdigest()


def _coverage_key(
    pipeline_id: str,
    task: str,
    stats: tree.TaskOutcomeStats | None,
) -> tuple[float, float, str]:
    total = stats.total_evals if stats else 0
    fragility = (
        (stats.failure_rate if stats else 1.0)
        + (stats.regression_rate if stats else 0.0)
        + (0.25 * (stats.merge_failures if stats else 0))
    )
    return (float(total), -float(fragility), _hash(pipeline_id, task))


__all__ = ["best_node", "unresolved_tasks", "pick_and_claim"]
