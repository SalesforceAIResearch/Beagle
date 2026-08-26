"""Housekeeping primitives: orphan detection + GC of worktrees / branches /
artifacts. Each function returns a structured Action plan; the CLI wraps
these and (in --apply mode) executes the plans.

Each `detect_*` function is pure — given a DB connection (read-only) and
filters, it returns a list of `Action`s describing what *would* be done
without doing anything. The matching `apply_*` function takes an action
list and a writable DB connection and performs the side effect.

The split keeps things unit-testable and makes `--dry-run` cheap.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from . import tree, worktree


# ─── Action descriptor ───────────────────────────────────────────────────


@dataclass
class Action:
    """One thing that would be (or was) done by the cleanup tool."""

    kind: str                  # 'orphan_pipeline' / 'release_claim' / 'remove_worktree' /
                               # 'delete_branch' / 'remove_artifacts' / 'vacuum'
    target: str                # display name for the operator
    detail: dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Duration parsing ────────────────────────────────────────────────────


def parse_duration(s: str) -> timedelta:
    """Parse '30m' / '7d' / '36h' / '120s' into a timedelta. Required suffix."""
    if not s:
        raise ValueError("empty duration")
    unit = s[-1]
    try:
        val = int(s[:-1])
    except ValueError as e:
        raise ValueError(f"could not parse duration {s!r}: {e}") from None
    if unit == "s":
        return timedelta(seconds=val)
    if unit == "m":
        return timedelta(minutes=val)
    if unit == "h":
        return timedelta(hours=val)
    if unit == "d":
        return timedelta(days=val)
    raise ValueError(f"unknown duration suffix {unit!r} in {s!r} (use s/m/h/d)")


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


# ─── Layer 1: orphan pipelines ───────────────────────────────────────────


def detect_orphan_pipelines(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    stale_after: timedelta,
    pipeline_filter: str | None = None,
    now_fn=None,
) -> list[Action]:
    """Find non-terminal pipelines that have died (PID gone) or gone stale (no heartbeat)."""
    actions: list[Action] = []
    now = (now_fn or (lambda: datetime.now(timezone.utc)))()
    this_host = socket.gethostname()

    pipelines = tree.list_pipelines(conn, campaign=campaign)
    for p in pipelines:
        if pipeline_filter and p.id != pipeline_filter:
            continue
        if p.status not in tree.PIPELINE_NON_TERMINAL:
            continue

        is_local = (p.host or "") == this_host
        pid_dead = is_local and not _pid_alive(p.pid)
        last_beat = _parse_iso(p.heartbeat_at) or _parse_iso(p.started_at)
        stale = (last_beat is None) or ((now - last_beat) > stale_after)

        if not (pid_dead or stale):
            continue

        reason = []
        if pid_dead:
            reason.append(f"pid {p.pid} not alive on {this_host}")
        if stale:
            age = (now - last_beat) if last_beat else "no heartbeat"
            reason.append(f"stale heartbeat ({age})")
        actions.append(Action(
            kind="orphan_pipeline",
            target=f"{p.id} ({p.status})",
            detail={
                "pipeline_id": p.id,
                "child_node_id": p.child_node_id,
                "host": p.host,
                "pid": p.pid,
                "heartbeat_at": p.heartbeat_at,
                "reason": "; ".join(reason),
            },
        ))
    return actions


def apply_orphan_pipelines(conn: sqlite3.Connection, actions: list[Action]) -> None:
    for a in actions:
        if a.kind != "orphan_pipeline":
            continue
        try:
            pid_id = a.detail["pipeline_id"]
            child = a.detail.get("child_node_id")
            tree.update_pipeline(conn, pid_id, status="failed", finished_at=tree.utcnow_iso())
            if child:
                node = tree.get_node(conn, child)
                if node and node.status == "in_progress":
                    tree.update_node(conn, child, status="failed")
            tree.release_claims(conn, pipeline_id=pid_id)
            a.applied = True
        except Exception as e:  # don't let one bad row break the loop
            a.error = str(e)


# ─── Layer 2: leaked claims ──────────────────────────────────────────────


def detect_leaked_claims(
    conn: sqlite3.Connection,
    *,
    campaign: str,
) -> list[Action]:
    """Active claims whose owning pipeline is now in a terminal status."""
    actions: list[Action] = []
    pipelines = {p.id: p for p in tree.list_pipelines(conn, campaign=campaign)}
    claims = tree.list_active_claims(conn)
    for c in claims:
        p = pipelines.get(c.pipeline_id)
        if p is None or p.status not in tree.PIPELINE_TERMINAL:
            continue
        actions.append(Action(
            kind="release_claim",
            target=f"{c.pipeline_id}/{c.failure_task}",
            detail={
                "pipeline_id": c.pipeline_id,
                "parent_id": c.parent_id,
                "claim_kind": c.claim_kind,
                "failure_task": c.failure_task,
                "pipeline_status": p.status,
            },
        ))
    return actions


def apply_leaked_claims(conn: sqlite3.Connection, actions: list[Action]) -> None:
    pids = {a.detail["pipeline_id"] for a in actions if a.kind == "release_claim"}
    for pid in pids:
        try:
            tree.release_claims(conn, pipeline_id=pid)
        except Exception as e:
            for a in actions:
                if a.kind == "release_claim" and a.detail["pipeline_id"] == pid:
                    a.error = str(e)
                    break
        else:
            for a in actions:
                if a.kind == "release_claim" and a.detail["pipeline_id"] == pid:
                    a.applied = True


# ─── Layer 3: worktrees ──────────────────────────────────────────────────


def detect_worktrees(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    keep_for_pr: bool = True,
    repo_root: Path = worktree.REPO_ROOT,
) -> list[Action]:
    """Worktrees of terminal pipelines that exist on disk."""
    actions: list[Action] = []
    pipelines = tree.list_pipelines(conn, campaign=campaign)
    for p in pipelines:
        if p.status not in tree.PIPELINE_TERMINAL:
            continue
        if not p.worktree_path:
            continue
        wt_path = Path(p.worktree_path)
        if not wt_path.exists():
            continue
        # If --keep-for-pr is on, skip when the child node has an open PR.
        skip = False
        if keep_for_pr and p.child_node_id:
            node = tree.get_node(conn, p.child_node_id)
            if node and _has_open_pr(node):
                skip = True
        if skip:
            continue
        actions.append(Action(
            kind="remove_worktree",
            target=str(wt_path),
            detail={
                "pipeline_id": p.id,
                "worktree_path": str(wt_path),
                "repo_root": str(repo_root),
            },
        ))
    return actions


def apply_worktrees(conn: sqlite3.Connection, actions: list[Action]) -> None:
    for a in actions:
        if a.kind != "remove_worktree":
            continue
        try:
            ok = worktree.remove_eval_worktree(
                Path(a.detail["worktree_path"]),
                repo_root=Path(a.detail["repo_root"]),
            )
            a.applied = bool(ok)
            if not ok:
                a.error = "git worktree remove failed; tree was force-deleted from disk"
        except Exception as e:
            a.error = str(e)


def _has_open_pr(node: tree.Node) -> bool:
    pr_url = None
    for entry in node.resolved_tasks:
        if isinstance(entry, dict) and entry.get("pr_url"):
            pr_url = entry["pr_url"]
            break
    if not pr_url:
        return False
    # Best-effort: don't fail cleanup if `gh` isn't available — assume open.
    try:
        proc = subprocess.run(
            ["gh", "pr", "view", pr_url, "--json", "state", "-q", ".state"],
            capture_output=True, text=True, timeout=20,
        )
        return (proc.stdout or "").strip().upper() == "OPEN"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True


# ─── Layer 4: branches ──────────────────────────────────────────────────


def detect_branches(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    repo_root: Path = worktree.REPO_ROOT,
) -> list[Action]:
    """evolve/* branches that aren't worth keeping (no improvement, no PR, no children)."""
    actions: list[Action] = []
    nodes = tree.list_nodes(conn, campaign=campaign)
    nodes_by_branch = {n.branch_name: n for n in nodes}
    parent_branches = {nodes_by_branch[n.parent_id].branch_name
                       for n in nodes if n.parent_id and n.parent_id in nodes_by_branch}
    branches = worktree.list_evolve_branches(repo_root=repo_root)

    for b in branches:
        node = nodes_by_branch.get(b)
        keep_reason = None
        if node and node.status == "completed" and node.score is not None:
            parent = next((p for p in nodes if p.id == node.parent_id), None) if node.parent_id else None
            if parent and parent.score is not None and node.score > parent.score:
                keep_reason = "score-improving completed node"
        # QD specialists are archived with REAL unique wins (improved_tasks) and are
        # merge fuel — do NOT GC their branches or recombination loses its material.
        if node and node.status == "archived" and getattr(node, "improved_tasks", None):
            keep_reason = "QD specialist (archived w/ unique wins) — merge fuel"
        if node and _has_open_pr(node):
            keep_reason = "node has open PR"
        if b in parent_branches:
            keep_reason = "branch is parent of another node"
        if keep_reason:
            continue

        actions.append(Action(
            kind="delete_branch",
            target=b,
            detail={"branch": b, "repo_root": str(repo_root)},
        ))
    return actions


def apply_branches(conn: sqlite3.Connection, actions: list[Action]) -> None:
    for a in actions:
        if a.kind != "delete_branch":
            continue
        try:
            ok = worktree.delete_submodule_branch(
                a.detail["branch"], repo_root=Path(a.detail["repo_root"])
            )
            a.applied = ok
            if not ok:
                a.error = "branch -D failed (may not exist or be checked out elsewhere)"
        except Exception as e:
            a.error = str(e)


# ─── Layer 5: artifacts ──────────────────────────────────────────────────


def detect_artifacts(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    reports_root: Path,
    older_than: timedelta,
    purge: bool = False,
    now_fn=None,
) -> list[Action]:
    """Pipeline artifacts under campaign_root/pipelines/<pid>/."""
    actions: list[Action] = []
    now = (now_fn or (lambda: datetime.now(timezone.utc)))()

    pipelines_dir = tree.campaign_root(reports_root, campaign) / "pipelines"
    if not pipelines_dir.is_dir():
        return actions

    pipelines = {p.id: p for p in tree.list_pipelines(conn, campaign=campaign)}
    for sub in sorted(pipelines_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue  # skip _supervisor/ etc.
        pid = sub.name
        p = pipelines.get(pid)
        if p is None:
            # Pipeline row missing — orphan disk dir; clean up if it's old.
            mtime = datetime.fromtimestamp(sub.stat().st_mtime, tz=timezone.utc)
            if (now - mtime) <= older_than:
                continue
        else:
            if p.status not in tree.PIPELINE_TERMINAL:
                continue
            ts = _parse_iso(p.finished_at) or _parse_iso(p.started_at)
            if ts and (now - ts) <= older_than:
                continue
        actions.append(Action(
            kind="remove_artifacts",
            target=str(sub),
            detail={
                "pipeline_id": pid,
                "dir": str(sub),
                "purge": purge,
                "prompts_dir": str(tree.prompts_dir(reports_root, campaign, pid)),
            },
        ))
    return actions


def apply_artifacts(conn: sqlite3.Connection, actions: list[Action]) -> None:
    for a in actions:
        if a.kind != "remove_artifacts":
            continue
        try:
            d = Path(a.detail["dir"])
            cursor_dir = d / "cursor"
            if cursor_dir.is_dir():
                shutil.rmtree(cursor_dir, ignore_errors=True)
            if a.detail.get("purge"):
                # Nuke the whole pipeline dir + prompts dir.
                shutil.rmtree(d, ignore_errors=True)
                pdir = Path(a.detail["prompts_dir"])
                if pdir.is_dir():
                    shutil.rmtree(pdir, ignore_errors=True)
            a.applied = True
        except Exception as e:
            a.error = str(e)


# ─── Layer 6: VACUUM ─────────────────────────────────────────────────────


def vacuum(conn: sqlite3.Connection) -> Action:
    a = Action(kind="vacuum", target="state.db")
    try:
        # VACUUM cannot run inside a transaction.
        conn.execute("VACUUM")
        a.applied = True
    except Exception as e:
        a.error = str(e)
    return a


# ─── Aggregator ──────────────────────────────────────────────────────────


def detect_all(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    reports_root: Path,
    do_orphan_pipelines: bool = True,
    do_orphan_claims: bool = True,
    do_worktrees: bool = False,
    do_branches: bool = False,
    do_artifacts: bool = False,
    stale_after: timedelta = timedelta(minutes=30),
    older_than: timedelta = timedelta(days=7),
    keep_for_pr: bool = True,
    purge_artifacts: bool = False,
    repo_root: Path = worktree.REPO_ROOT,
) -> list[Action]:
    """Convenience: return all actions in execution order (orphans → claims → ...)."""
    actions: list[Action] = []
    if do_orphan_pipelines:
        actions += detect_orphan_pipelines(conn, campaign=campaign, stale_after=stale_after)
    if do_orphan_claims:
        actions += detect_leaked_claims(conn, campaign=campaign)
    if do_worktrees:
        actions += detect_worktrees(
            conn, campaign=campaign, keep_for_pr=keep_for_pr, repo_root=repo_root,
        )
    if do_branches:
        actions += detect_branches(conn, campaign=campaign, repo_root=repo_root)
    if do_artifacts:
        actions += detect_artifacts(
            conn, campaign=campaign, reports_root=reports_root,
            older_than=older_than, purge=purge_artifacts,
        )
    return actions


def apply_all(conn: sqlite3.Connection, actions: list[Action]) -> None:
    """Apply each action group. Idempotent — already-applied actions stay applied=True."""
    apply_orphan_pipelines(conn, actions)
    apply_leaked_claims(conn, actions)
    apply_worktrees(conn, actions)
    apply_branches(conn, actions)
    apply_artifacts(conn, actions)


__all__ = [
    "Action",
    "parse_duration",
    "detect_orphan_pipelines",
    "apply_orphan_pipelines",
    "detect_leaked_claims",
    "apply_leaked_claims",
    "detect_worktrees",
    "apply_worktrees",
    "detect_branches",
    "apply_branches",
    "detect_artifacts",
    "apply_artifacts",
    "vacuum",
    "detect_all",
    "apply_all",
]
