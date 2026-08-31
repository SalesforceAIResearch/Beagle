"""Pure helpers used by `scripts/self_evolve_report.py` and the visualizer.

Kept LLM-free so the same building blocks (digest, leaderboard, token cost)
work in unit tests and in the report CLI without spending tokens.

Public surface:

    build_campaign_digest(conn, campaign, reports_root, *, top_n=10) -> dict
    compute_token_cost(campaign, reports_root) -> dict
    top_n_nodes(conn, campaign, n=10, *, subset=None) -> list[dict]
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import cursor_agent, tree


def top_n_nodes(
    conn: sqlite3.Connection,
    campaign: str,
    n: int = 10,
    *,
    subset: str | None = None,
) -> list[dict[str, Any]]:
    """Return the top-N nodes by `(score DESC, delta_vs_parent DESC, created_at ASC)`.

    Each row carries the fields the visualizer's Leaderboard view + the
    REPORT.md table need: branch, commit, score, parent_score, delta,
    delta_vs_root, resolved_tasks, status, pr_url (if recorded), node_id.
    """
    nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    search_evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    edges = tree.list_node_edges(conn, campaign=campaign)
    if not nodes:
        return []

    by_id = {nd.id: nd for nd in nodes}
    primary_parent_by_child = {
        e.child_id: e.parent_id
        for e in edges
        if e.edge_type == "merge" and e.parent_role == "primary"
    }

    # Root score for delta-vs-root: pick the campaign's root (parent_id IS NULL).
    root_score = None
    for nd in nodes:
        ev = search_evals.get(nd.id)
        if nd.parent_id is None and (ev is not None or nd.score is not None):
            root_score = ev.score if ev else nd.score
            break

    rows: list[dict[str, Any]] = []
    for nd in nodes:
        ev = search_evals.get(nd.id)
        if ev is None and nd.score is None:
            continue
        if nd.status not in {"completed", "no_change"}:
            continue
        score = ev.score if ev else (nd.score or 0.0)
        solved_tasks = ev.solved_tasks if ev else nd.solved_tasks
        unsolved_tasks = ev.unsolved_tasks if ev else nd.unsolved_tasks
        partial_tasks = ev.partially_solved_tasks if ev else nd.partially_solved_tasks
        improved_tasks = ev.improved_tasks if ev else nd.improved_tasks
        regressed_tasks = ev.regressed_tasks if ev else nd.regressed_tasks
        job_log_path = ev.job_log_path if ev else nd.job_log_path
        parent_id = primary_parent_by_child.get(nd.id, nd.parent_id)
        parent = by_id.get(parent_id) if parent_id else None
        parent_ev = search_evals.get(parent.id) if parent else None
        parent_score = parent_ev.score if parent_ev else (parent.score if parent else None)
        delta = (score - parent_score) if parent_score is not None else None
        delta_root = (score - root_score) if root_score is not None else None
        fullset = tree.node_fullset_eval(conn, campaign=campaign, node_id=nd.id)
        rows.append({
            "node_id": nd.id,
            "branch": nd.branch_name,
            "commit": nd.commit_sha,
            "score": score,
            "subset_score": score,
            "fullset_score": fullset.score if fullset else None,
            "subset": nd.subset,
            "parent_id": parent_id,
            "parent_score": parent_score,
            "delta_vs_parent": delta,
            "delta_vs_root": delta_root,
            "resolved_tasks": nd.resolved_tasks,
            "solved_tasks": solved_tasks,
            "unsolved_tasks": unsolved_tasks,
            "partially_solved_tasks": partial_tasks,
            "improved_tasks": improved_tasks,
            "regressed_tasks": regressed_tasks,
            "solved_count": len(solved_tasks),
            "unsolved_count": len(unsolved_tasks),
            "partially_solved_count": len(partial_tasks),
            "improved_count": len(improved_tasks),
            "regressed_count": len(regressed_tasks),
            "status": nd.status,
            "created_at": nd.created_at,
            "pr_url": _resolved_pr_url(nd),
            "job_log_path": job_log_path,
        })

    # Score desc, then delta desc (None last), then created_at asc.
    def _key(row: dict) -> tuple:
        d = row["delta_vs_parent"] if row["delta_vs_parent"] is not None else float("-inf")
        return (-(row["score"] or 0.0), -d, row["created_at"] or "")

    rows.sort(key=_key)
    return rows[:n]


def _resolved_pr_url(node: tree.Node) -> str | None:
    """Pull the PR URL out of resolved_tasks if any iteration recorded it."""
    for entry in node.resolved_tasks:
        if not isinstance(entry, dict):
            continue
        if entry.get("pr_url"):
            return entry["pr_url"]
        if entry.get("merge_pr_url"):
            return entry["merge_pr_url"]
        if isinstance(entry.get("merge_pr"), dict) and entry["merge_pr"].get("url"):
            return entry["merge_pr"]["url"]
    return None


def rejected_frontier_nodes(
    conn: sqlite3.Connection,
    campaign: str,
    *,
    subset: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    merge_children = {
        edge.child_id
        for edge in tree.list_node_edges(conn, campaign=campaign, edge_type="merge")
    }
    rows: list[dict[str, Any]] = []
    for nd in nodes:
        if nd.id not in merge_children or nd.status != "rejected" or nd.score is None:
            continue
        merge_delta = next(
            (
                item["merge_delta"]
                for item in nd.resolved_tasks
                if isinstance(item, dict) and isinstance(item.get("merge_delta"), dict)
            ),
            {},
        )
        rows.append({
            "node_id": nd.id,
            "branch": nd.branch_name,
            "commit": nd.commit_sha,
            "score": nd.score,
            "subset": nd.subset,
            "lost_parent_wins": merge_delta.get("lost_parent_wins", []),
            "new_child_wins": merge_delta.get("new_child_wins", []),
            "validation_only": bool(merge_delta.get("validation_only")),
            "job_log_path": nd.job_log_path,
            "pr_url": _resolved_pr_url(nd),
        })
    rows.sort(key=lambda r: (len(r["lost_parent_wins"]), -(r["score"] or 0.0), r["node_id"]))
    return rows[:limit]


def compute_token_cost(campaign: str, reports_root: Path | str) -> dict[str, int]:
    """Sum token usage across every cursor-agent stream-json log in the campaign.

    Returns {'input': ..., 'output': ..., 'cache_read': ..., 'cache_write': ...,
             'total': ..., 'n_calls': ..., 'duration_ms': ...}
    """
    totals = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total": 0,
        "n_calls": 0,
        "duration_ms": 0,
    }
    pipelines_dir = tree.campaign_root(reports_root, campaign) / "pipelines"
    if not pipelines_dir.is_dir():
        return totals
    for log_path in pipelines_dir.rglob("cursor/*.log"):
        parsed = cursor_agent.parse_log(log_path)
        usage = parsed.get("usage") or {}
        totals["input"] += int(usage.get("inputTokens") or 0)
        totals["output"] += int(usage.get("outputTokens") or 0)
        totals["cache_read"] += int(usage.get("cacheReadTokens") or 0)
        totals["cache_write"] += int(usage.get("cacheWriteTokens") or 0)
        totals["duration_ms"] += int(parsed.get("duration_ms") or 0)
        if parsed.get("usage"):
            totals["n_calls"] += 1
    totals["total"] = totals["input"] + totals["output"]
    return totals


def build_campaign_digest(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    reports_root: Path | str,
    top_n: int = 10,
) -> dict[str, Any]:
    """Assemble a structured campaign digest for the REPORT.md prompt.

    The digest is what the cursor-agent report run reads (alongside the
    per-node markdown files). Keep keys stable — the report prompt
    references several of them by name.
    """
    nodes = tree.list_nodes(conn, campaign=campaign)
    search_evals = tree.search_eval_by_node(conn, campaign=campaign)
    pipelines = tree.list_pipelines(conn, campaign=campaign)
    edges = tree.list_node_edges(conn, campaign=campaign)

    by_id = {n.id: n for n in nodes}

    # Per-node enriched dicts.
    node_dicts: list[dict[str, Any]] = []
    root_node = next((n for n in nodes if n.parent_id is None), None)
    root_eval = search_evals.get(root_node.id) if root_node else None
    root_score = root_eval.score if root_eval else None
    best_score = max((ev.score for ev in search_evals.values()), default=None)

    biggest_step = 0.0
    for n in nodes:
        ev = search_evals.get(n.id)
        if ev is None or n.parent_id is None:
            continue
        p = by_id.get(n.parent_id)
        p_ev = search_evals.get(p.id) if p else None
        if p_ev is not None:
            step = ev.score - p_ev.score
            if step > biggest_step:
                biggest_step = step

        node_dir = tree.node_dir(reports_root, campaign, n.id)
        fullset = tree.node_fullset_eval(conn, campaign=campaign, node_id=n.id)
        node_dicts.append({
            "id": n.id,
            "branch": n.branch_name,
            "commit": n.commit_sha,
            "parent_id": n.parent_id,
            "score": ev.score,
            "subset_score": ev.score,
            "fullset_score": fullset.score if fullset else None,
            "subset": n.subset,
            "status": n.status,
            "created_at": n.created_at,
            "updated_at": n.updated_at,
            "failed_tasks": ev.failed_tasks,
            "solved_tasks": ev.solved_tasks,
            "unsolved_tasks": ev.unsolved_tasks,
            "partially_solved_tasks": ev.partially_solved_tasks,
            "improved_tasks": ev.improved_tasks,
            "regressed_tasks": ev.regressed_tasks,
            "resolved_tasks": n.resolved_tasks,
            "works_md": str(node_dir / "works.md"),
            "effort_md": str(node_dir / "effort.md"),
            "learnings_md": str(node_dir / "learnings.md"),
            "pr_url": _resolved_pr_url(n),
        })
    # Also include the root if we skipped it above (parent_id None loop adds nothing).
    for n in nodes:
        if n.parent_id is None and not any(d["id"] == n.id for d in node_dicts):
            node_dir = tree.node_dir(reports_root, campaign, n.id)
            ev = search_evals.get(n.id)
            fullset = tree.node_fullset_eval(conn, campaign=campaign, node_id=n.id)
            node_dicts.append({
                "id": n.id,
                "branch": n.branch_name,
                "commit": n.commit_sha,
                "parent_id": None,
                "score": ev.score if ev else None,
                "subset_score": ev.score if ev else None,
                "fullset_score": fullset.score if fullset else None,
                "subset": n.subset,
                "status": n.status,
                "created_at": n.created_at,
                "updated_at": n.updated_at,
                "failed_tasks": ev.failed_tasks if ev else n.failed_tasks,
                "solved_tasks": ev.solved_tasks if ev else n.solved_tasks,
                "unsolved_tasks": ev.unsolved_tasks if ev else n.unsolved_tasks,
                "partially_solved_tasks": ev.partially_solved_tasks if ev else n.partially_solved_tasks,
                "improved_tasks": ev.improved_tasks if ev else n.improved_tasks,
                "regressed_tasks": ev.regressed_tasks if ev else n.regressed_tasks,
                "resolved_tasks": n.resolved_tasks,
                "works_md": str(node_dir / "works.md"),
                "effort_md": str(node_dir / "effort.md"),
                "learnings_md": str(node_dir / "learnings.md"),
                "pr_url": None,
            })
    node_dicts.sort(key=lambda d: d["created_at"] or "")

    pipeline_dicts = [
        {
            "id": p.id,
            "status": p.status,
            "parent_node_id": p.parent_node_id,
            "child_node_id": p.child_node_id,
            "selected_tasks": json.loads(p.selected_tasks_json or "[]"),
            "current_iteration": p.current_iteration,
            "started_at": p.started_at,
            "finished_at": p.finished_at,
            "log_path": p.log_path,
        }
        for p in pipelines
    ]

    tokens = compute_token_cost(campaign, reports_root)
    uplift = (best_score - root_score) if (best_score is not None and root_score is not None) else None
    wall_clock_ms = tokens["duration_ms"]

    return {
        "campaign": campaign,
        "report_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "stats": {
            "total_nodes": len(node_dicts),
            "completed": sum(1 for d in node_dicts if d["status"] == "completed"),
            "no_change": sum(1 for d in node_dicts if d["status"] == "no_change"),
            "failed": sum(1 for d in node_dicts if d["status"] == "failed"),
            "in_progress": sum(1 for d in node_dicts if d["status"] == "in_progress"),
            "root_score": root_score,
            "best_score": best_score,
            "uplift": uplift,
            "biggest_step": biggest_step,
            "total_tokens": tokens["total"],
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
            "cache_read_tokens": tokens["cache_read"],
            "cache_write_tokens": tokens["cache_write"],
            "n_cursor_calls": tokens["n_calls"],
            "wall_clock_ms": wall_clock_ms,
            "wall_clock_human": _ms_to_h(wall_clock_ms),
            "report_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "top_n": top_n,
        },
        "nodes": node_dicts,
        "edges": [
            {
                "id": e.id,
                "parent_id": e.parent_id,
                "child_id": e.child_id,
                "edge_type": e.edge_type,
                "parent_role": e.parent_role,
                "pipeline_id": e.pipeline_id,
                "created_at": e.created_at,
            }
            for e in edges
        ],
        "pipelines": pipeline_dicts,
        "top_nodes": top_n_nodes(conn, campaign, top_n),
        "rejected_frontier_nodes": rejected_frontier_nodes(conn, campaign, limit=top_n),
    }


def _ms_to_h(ms: int) -> str:
    if not ms:
        return "0h"
    s = ms / 1000.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h {m}m"


__all__ = [
    "top_n_nodes",
    "rejected_frontier_nodes",
    "compute_token_cost",
    "build_campaign_digest",
]
