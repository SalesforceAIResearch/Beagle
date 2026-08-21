"""SQLite-backed campaign tree DAO.

The single source of truth for the self-evolve campaign state. Schema:

    nodes(id PK, campaign, branch_name, commit_sha, commits_json, parent_id,
          score, subset, job_log_path, failed_tasks_json,
          solved_tasks_json, unsolved_tasks_json,
          improved_tasks_json, regressed_tasks_json, resolved_tasks_json,
          status, works_md_path, effort_md_path, pipeline_id,
          created_at, updated_at)

    `commit_sha`   — the HEAD (most recent) commit on this node's branch as
                     of the last DAO write. Convenience pointer.
    `commits_json` — JSON array of every commit "associated with" this node,
                     in chronological order. For root nodes this is the
                     single baseline commit; for evolved children it's the
                     list of SHAs added in kept iterations (empty if none
                     of the agent's iterations survived guard rejection +
                     mini-eval). The invariant for non-empty arrays is
                     `commit_sha == commits[-1]`.

    claims(parent_id, failure_task, pipeline_id, claimed_at, released_at,
           PRIMARY KEY(parent_id, failure_task, pipeline_id))

    pipelines(id PK, campaign, parent_node_id, child_node_id,
              selected_tasks_json, status, current_iteration, log_path,
              host, pid, heartbeat_at, worktree_path, started_at, finished_at)

    iteration_outcomes(id PK, campaign, pipeline_id, node_id, iteration,
                       stage, outcome, reason, committed_shas_json,
                       reverted, reverted_shas_json, mini_eval_job_path,
                       mini_eval_score, claimed_rewards_json,
                       canary_rewards_json, failed_canaries_json, ...)

All writes happen inside `BEGIN IMMEDIATE` transactions so concurrent
pipelines serialize cleanly. Opened in WAL mode so readers (e.g. the
visualizer) never block writers.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Status values
# 'archived' = a preserved stepping-stone variant (kept with its OWN commit +
# score for DGM-style parent sampling / future merges) that is NOT eligible to
# be the shipped tip/best-node (so a no-effect or non-improving variant can't
# hijack best-node with an inflated subset score). Parent selection DOES sample
# archived nodes; best_node_by_search_eval does NOT.
NODE_STATUSES = {"in_progress", "completed", "no_change", "failed", "rejected", "archived"}
PIPELINE_NON_TERMINAL = {"preparing", "baseline", "evolving", "eval"}
PIPELINE_TERMINAL = {"done", "failed", "no_change"}
EDGE_TYPES = {"evolve", "merge", "regression_resolve"}
PARENT_ROLES = {"evolve", "primary", "secondary", "regression_target"}
CLAIM_KINDS = {"evolve", "regression_resolve"}
EVAL_KINDS = {
    "root_full",
    "subset_final",
    "fullset_final",
    "mini_eval",
    "merge_validation",
}
# 'improved'/'regressed' come from evals; 'rejected'/'poisoned' record the
# patterns of NON-promoted nodes (a regression/no-gain edit and a verifier-gaming
# edit) so the collective-knowledge digest spans failed + partial + regressed +
# poisoned attempts, not just measured task deltas.
EXPERIENCE_KINDS = {"improved", "regressed", "rejected", "poisoned"}


def utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 with trailing Z (matches harbor convention)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_id() -> str:
    """8-char URL-safe ID used for nodes and pipelines."""
    return uuid.uuid4().hex[:8]


# ─── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class Node:
    id: str
    campaign: str
    branch_name: str
    commit_sha: str | None
    commits_json: str  # JSON array of all commits associated with this node
    picked_commit_sha: str | None
    claimed_task_scores_json: str  # JSON object: task → reward
    parent_id: str | None
    score: float | None
    subset: str
    job_log_path: str | None
    failed_tasks_json: str  # JSON array
    resolved_tasks_json: str  # JSON array
    status: str
    works_md_path: str | None
    effort_md_path: str | None
    pipeline_id: str | None
    created_at: str
    updated_at: str
    solved_tasks_json: str = "[]"  # JSON array
    unsolved_tasks_json: str = "[]"  # JSON array
    partially_solved_tasks_json: str = "[]"  # JSON array
    improved_tasks_json: str = "[]"  # JSON array
    regressed_tasks_json: str = "[]"  # JSON array

    @property
    def failed_tasks(self) -> list[str]:
        # Backwards-compatible work-pool alias: partial tasks are still
        # repairable, but they must not be counted as fully solved.
        partial = self.partially_solved_tasks
        unsolved = self.unsolved_tasks
        failed = list(dict.fromkeys(partial + unsolved))
        return failed if failed else json.loads(self.failed_tasks_json or "[]")

    @property
    def solved_tasks(self) -> list[str]:
        return json.loads(self.solved_tasks_json or "[]")

    @property
    def unsolved_tasks(self) -> list[str]:
        return json.loads(self.unsolved_tasks_json or "[]")

    @property
    def partially_solved_tasks(self) -> list[str]:
        return json.loads(self.partially_solved_tasks_json or "[]")

    @property
    def improved_tasks(self) -> list[str]:
        return json.loads(self.improved_tasks_json or "[]")

    @property
    def regressed_tasks(self) -> list[str]:
        return json.loads(self.regressed_tasks_json or "[]")

    @property
    def resolved_tasks(self) -> list[dict]:
        return json.loads(self.resolved_tasks_json or "[]")

    @property
    def commits(self) -> list[str]:
        """All commits associated with this node, oldest → newest.

        Root: a single baseline commit. Evolved child: the SHAs of every
        kept-iteration commit (Layer 2 + Layer 3 guards passed). Empty
        when no iteration was kept.
        """
        return json.loads(self.commits_json or "[]")

    @property
    def claimed_task_scores(self) -> dict[str, float]:
        """Per-task reward map for the claimed tasks, from the final eval.

        Populated in `_finalize` after the picker chooses a commit and
        the final eval runs against that commit. Empty before the
        final eval (e.g. failed / in-progress nodes).
        """
        return json.loads(self.claimed_task_scores_json or "{}")


@dataclass
class Pipeline:
    id: str
    campaign: str
    parent_node_id: str | None
    child_node_id: str | None
    selected_tasks_json: str
    status: str
    current_iteration: int
    log_path: str | None
    host: str | None
    pid: int | None
    heartbeat_at: str | None
    worktree_path: str | None
    started_at: str
    finished_at: str | None
    pipeline_kind: str = "evolve"
    target_node_id: str | None = None


@dataclass
class IterationOutcomeRow:
    id: str
    campaign: str
    pipeline_id: str
    node_id: str | None
    iteration: int
    stage: str
    outcome: str
    reason: str
    committed_shas_json: str
    reverted: bool
    reverted_shas_json: str
    mini_eval_job_path: str | None
    mini_eval_score: float | None
    mini_eval_n_trials: int | None
    mini_eval_n_errors: int | None
    claimed_rewards_json: str
    canary_rewards_json: str
    canary_tasks_json: str
    failed_canaries_json: str
    review_duration_ms: int | None
    review_error: str | None
    created_at: str
    updated_at: str

    @property
    def committed_shas(self) -> list[str]:
        return json.loads(self.committed_shas_json or "[]")

    @property
    def reverted_shas(self) -> list[str]:
        return json.loads(self.reverted_shas_json or "[]")

    @property
    def claimed_rewards(self) -> dict[str, float]:
        return json.loads(self.claimed_rewards_json or "{}")

    @property
    def canary_rewards(self) -> dict[str, float]:
        return json.loads(self.canary_rewards_json or "{}")

    @property
    def canary_tasks(self) -> list[str]:
        return json.loads(self.canary_tasks_json or "[]")

    @property
    def failed_canaries(self) -> list[str]:
        return json.loads(self.failed_canaries_json or "[]")


@dataclass
class Claim:
    # `campaign`/`subset`/`claim_kind` is the claim scope. campaign/subset
    # may be NULL on legacy rows whose migration backfill couldn't resolve
    # `parent_id` (e.g. parent already deleted) — callers should treat that
    # case as "stale".
    campaign: str | None
    subset: str | None
    claim_kind: str
    parent_id: str | None  # informational, no longer in uniqueness constraint
    failure_task: str
    pipeline_id: str
    claimed_at: str
    released_at: str | None


@dataclass
class NodeEval:
    id: str
    campaign: str
    node_id: str
    eval_kind: str
    subset_label: str
    task_names_json: str
    n_trials: int
    n_errors: int
    score: float
    job_log_path: str | None
    solved_tasks_json: str
    unsolved_tasks_json: str
    partially_solved_tasks_json: str
    task_rewards_json: str
    improved_tasks_json: str
    regressed_tasks_json: str
    source_pipeline_id: str | None
    created_at: str
    metadata_json: str

    @property
    def task_names(self) -> list[str]:
        return json.loads(self.task_names_json or "[]")

    @property
    def solved_tasks(self) -> list[str]:
        return json.loads(self.solved_tasks_json or "[]")

    @property
    def unsolved_tasks(self) -> list[str]:
        return json.loads(self.unsolved_tasks_json or "[]")

    @property
    def partially_solved_tasks(self) -> list[str]:
        return json.loads(self.partially_solved_tasks_json or "[]")

    @property
    def failed_tasks(self) -> list[str]:
        return list(dict.fromkeys(self.partially_solved_tasks + self.unsolved_tasks))

    @property
    def task_rewards(self) -> dict[str, float]:
        return json.loads(self.task_rewards_json or "{}")

    @property
    def improved_tasks(self) -> list[str]:
        return json.loads(self.improved_tasks_json or "[]")

    @property
    def regressed_tasks(self) -> list[str]:
        return json.loads(self.regressed_tasks_json or "[]")

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json or "{}")


@dataclass
class TaskExperience:
    id: str
    campaign: str
    task: str
    node_id: str | None
    pipeline_id: str | None
    worker_kind: str
    commit_sha: str | None
    commit_number: int | None
    experience_kind: str
    eval_kind: str | None
    before_reward: float | None
    after_reward: float | None
    analysis: str
    code_change_summary: str
    artifact_paths_json: str
    log_excerpt: str
    confidence: float
    created_at: str
    metadata_json: str

    @property
    def artifact_paths(self) -> list[str]:
        return json.loads(self.artifact_paths_json or "[]")

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json or "{}")


@dataclass
class NodeEdge:
    id: str
    campaign: str
    parent_id: str
    child_id: str
    edge_type: str
    parent_role: str
    pipeline_id: str | None
    created_at: str


@dataclass(frozen=True)
class TaskOutcomeStats:
    """Campaign-local history for one task base."""

    task: str
    passes: int = 0
    failures: int = 0
    improvements: int = 0
    regressions: int = 0
    regression_resolver_attempts: int = 0
    regression_resolver_successes: int = 0
    regression_resolver_failures: int = 0
    merge_attempts: int = 0
    merge_successes: int = 0
    merge_failures: int = 0

    @property
    def total_evals(self) -> int:
        return self.passes + self.failures

    @property
    def failure_rate(self) -> float:
        return self.failures / max(1, self.total_evals)

    @property
    def regression_rate(self) -> float:
        return self.regressions / max(1, self.total_evals)

    @property
    def resolver_success_rate(self) -> float:
        return self.regression_resolver_successes / max(1, self.regression_resolver_attempts)


@dataclass(frozen=True)
class NodeOutcomeFeatures:
    """Compact task/outcome features used by selectors."""

    node_id: str
    solved: frozenset[str]
    unsolved: frozenset[str]
    partially_solved: frozenset[str]
    improved: frozenset[str]
    regressed: frozenset[str]
    score: float | None


# ─── Schema + connection management ───────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id                        TEXT PRIMARY KEY,
    campaign                  TEXT NOT NULL,
    branch_name               TEXT NOT NULL,
    commit_sha                TEXT,
    commits_json              TEXT NOT NULL DEFAULT '[]',
    picked_commit_sha         TEXT,
    claimed_task_scores_json  TEXT NOT NULL DEFAULT '{}',
    parent_id                 TEXT REFERENCES nodes(id),
    score                     REAL,
    subset                    TEXT NOT NULL DEFAULT 'full',
    job_log_path              TEXT,
    failed_tasks_json         TEXT NOT NULL DEFAULT '[]',
    solved_tasks_json         TEXT NOT NULL DEFAULT '[]',
    unsolved_tasks_json       TEXT NOT NULL DEFAULT '[]',
    partially_solved_tasks_json TEXT NOT NULL DEFAULT '[]',
    improved_tasks_json       TEXT NOT NULL DEFAULT '[]',
    regressed_tasks_json      TEXT NOT NULL DEFAULT '[]',
    resolved_tasks_json       TEXT NOT NULL DEFAULT '[]',
    status                    TEXT NOT NULL,
    works_md_path             TEXT,
    effort_md_path            TEXT,
    pipeline_id               TEXT,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_nodes_campaign ON nodes(campaign);
CREATE INDEX IF NOT EXISTS ix_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS ix_nodes_subset ON nodes(campaign, subset);

CREATE TABLE IF NOT EXISTS node_evals (
    id                          TEXT PRIMARY KEY,
    campaign                    TEXT NOT NULL,
    node_id                     TEXT NOT NULL REFERENCES nodes(id),
    eval_kind                   TEXT NOT NULL CHECK(eval_kind IN (
                                    'root_full',
                                    'subset_final',
                                    'fullset_final',
                                    'mini_eval',
                                    'merge_validation'
                                  )),
    subset_label                TEXT NOT NULL,
    task_names_json             TEXT NOT NULL DEFAULT '[]',
    n_trials                    INTEGER NOT NULL DEFAULT 0,
    n_errors                    INTEGER NOT NULL DEFAULT 0,
    score                       REAL NOT NULL DEFAULT 0.0,
    job_log_path                TEXT,
    solved_tasks_json           TEXT NOT NULL DEFAULT '[]',
    unsolved_tasks_json         TEXT NOT NULL DEFAULT '[]',
    partially_solved_tasks_json TEXT NOT NULL DEFAULT '[]',
    task_rewards_json           TEXT NOT NULL DEFAULT '{}',
    improved_tasks_json         TEXT NOT NULL DEFAULT '[]',
    regressed_tasks_json        TEXT NOT NULL DEFAULT '[]',
    source_pipeline_id          TEXT,
    created_at                  TEXT NOT NULL,
    metadata_json               TEXT NOT NULL DEFAULT '{}',
    UNIQUE(campaign, node_id, eval_kind, source_pipeline_id)
);

CREATE INDEX IF NOT EXISTS ix_node_evals_campaign_kind
    ON node_evals(campaign, eval_kind, created_at);
CREATE INDEX IF NOT EXISTS ix_node_evals_node_kind
    ON node_evals(campaign, node_id, eval_kind, created_at);

CREATE TABLE IF NOT EXISTS task_experiences (
    id                    TEXT PRIMARY KEY,
    campaign              TEXT NOT NULL,
    task                  TEXT NOT NULL,
    node_id               TEXT,
    pipeline_id           TEXT,
    worker_kind           TEXT NOT NULL,
    commit_sha            TEXT,
    commit_number         INTEGER,
    experience_kind       TEXT NOT NULL CHECK(experience_kind IN ('improved', 'regressed', 'rejected', 'poisoned')),
    eval_kind             TEXT,
    before_reward         REAL,
    after_reward          REAL,
    analysis              TEXT NOT NULL DEFAULT '',
    code_change_summary   TEXT NOT NULL DEFAULT '',
    artifact_paths_json   TEXT NOT NULL DEFAULT '[]',
    log_excerpt           TEXT NOT NULL DEFAULT '',
    confidence            REAL NOT NULL DEFAULT 0.5,
    created_at            TEXT NOT NULL,
    metadata_json         TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_task_experiences_task
    ON task_experiences(campaign, task, created_at);
CREATE INDEX IF NOT EXISTS ix_task_experiences_node
    ON task_experiences(campaign, node_id, created_at);

CREATE TABLE IF NOT EXISTS full_eval_rounds (
    campaign              TEXT NOT NULL,
    round_index           INTEGER NOT NULL,
    triggered_at_count    INTEGER NOT NULL,
    created_at            TEXT NOT NULL,
    PRIMARY KEY(campaign, round_index)
);

CREATE TABLE IF NOT EXISTS node_eval_claims (
    campaign              TEXT NOT NULL,
    node_id               TEXT NOT NULL,
    eval_kind             TEXT NOT NULL,
    pipeline_id           TEXT NOT NULL,
    round_index           INTEGER,
    claimed_at            TEXT NOT NULL,
    released_at           TEXT,
    PRIMARY KEY(campaign, node_id, eval_kind, pipeline_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_node_eval_claims_active
ON node_eval_claims(campaign, node_id, eval_kind) WHERE released_at IS NULL;

CREATE TABLE IF NOT EXISTS node_edges (
    id           TEXT PRIMARY KEY,
    campaign     TEXT NOT NULL,
    parent_id    TEXT NOT NULL REFERENCES nodes(id),
    child_id     TEXT NOT NULL REFERENCES nodes(id),
    edge_type    TEXT NOT NULL CHECK(edge_type IN ('evolve', 'merge', 'regression_resolve')),
    parent_role  TEXT NOT NULL,
    pipeline_id  TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE(parent_id, child_id, edge_type, parent_role)
);

CREATE INDEX IF NOT EXISTS ix_node_edges_campaign ON node_edges(campaign);
CREATE INDEX IF NOT EXISTS ix_node_edges_parent ON node_edges(parent_id);
CREATE INDEX IF NOT EXISTS ix_node_edges_child ON node_edges(child_id);
CREATE INDEX IF NOT EXISTS ix_node_edges_type ON node_edges(campaign, edge_type);

CREATE TABLE IF NOT EXISTS claims (
    -- (campaign, subset, claim_kind) is the claim's scope. Evolvers and
    -- regression resolvers intentionally use separate pools, while workers
    -- of the same kind still never race on the same failing task.
    campaign     TEXT,
    subset       TEXT,
    claim_kind   TEXT NOT NULL DEFAULT 'evolve'
                 CHECK(claim_kind IN ('evolve', 'regression_resolve')),
    -- `parent_id` is now informational only: which node the worker chose
    -- to branch from. NOT part of any uniqueness constraint.
    parent_id    TEXT,
    failure_task TEXT NOT NULL,
    pipeline_id  TEXT NOT NULL,
    claimed_at   TEXT NOT NULL,
    released_at  TEXT,
    PRIMARY KEY (campaign, subset, claim_kind, failure_task, pipeline_id)
);

CREATE INDEX IF NOT EXISTS ix_claims_pipeline ON claims(pipeline_id);
CREATE INDEX IF NOT EXISTS ix_claims_active
    ON claims(campaign, subset, claim_kind, failure_task, released_at);

-- Hard guarantee against double-claim: at most one ACTIVE claim per
-- (campaign, subset, claim_kind, failure_task). Released claims
-- (released_at IS NOT NULL) are excluded so a task can be re-claimed after
-- the previous pipeline releases it (which happens automatically on pipeline
-- exit). Cross-kind claims are allowed so evolvers and regression resolvers
-- can work on the same task simultaneously.
CREATE UNIQUE INDEX IF NOT EXISTS uq_claims_active_one_per_task
ON claims(campaign, subset, claim_kind, failure_task) WHERE released_at IS NULL;

CREATE TABLE IF NOT EXISTS pipelines (
    id                   TEXT PRIMARY KEY,
    campaign             TEXT NOT NULL,
    parent_node_id       TEXT,
    child_node_id        TEXT,
    selected_tasks_json  TEXT NOT NULL DEFAULT '[]',
    status               TEXT NOT NULL,
    current_iteration    INTEGER NOT NULL DEFAULT 0,
    log_path             TEXT,
    host                 TEXT,
    pid                  INTEGER,
    heartbeat_at         TEXT,
    worktree_path        TEXT,
    pipeline_kind        TEXT NOT NULL DEFAULT 'evolve',
    target_node_id       TEXT,
    started_at           TEXT NOT NULL,
    finished_at          TEXT
);

CREATE INDEX IF NOT EXISTS ix_pipelines_campaign ON pipelines(campaign);
CREATE INDEX IF NOT EXISTS ix_pipelines_status ON pipelines(campaign, status);

CREATE TABLE IF NOT EXISTS iteration_outcomes (
    id                     TEXT PRIMARY KEY,
    campaign               TEXT NOT NULL,
    pipeline_id            TEXT NOT NULL,
    node_id                TEXT,
    iteration              INTEGER NOT NULL,
    stage                  TEXT NOT NULL DEFAULT '',
    outcome                TEXT NOT NULL DEFAULT '',
    reason                 TEXT NOT NULL DEFAULT '',
    committed_shas_json    TEXT NOT NULL DEFAULT '[]',
    reverted               INTEGER NOT NULL DEFAULT 0,
    reverted_shas_json     TEXT NOT NULL DEFAULT '[]',
    mini_eval_job_path     TEXT,
    mini_eval_score        REAL,
    mini_eval_n_trials     INTEGER,
    mini_eval_n_errors     INTEGER,
    claimed_rewards_json   TEXT NOT NULL DEFAULT '{}',
    canary_rewards_json    TEXT NOT NULL DEFAULT '{}',
    canary_tasks_json      TEXT NOT NULL DEFAULT '[]',
    failed_canaries_json   TEXT NOT NULL DEFAULT '[]',
    review_duration_ms     INTEGER,
    review_error           TEXT,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    UNIQUE(campaign, pipeline_id, iteration)
);

CREATE INDEX IF NOT EXISTS ix_iteration_outcomes_node
    ON iteration_outcomes(campaign, node_id, iteration);
CREATE INDEX IF NOT EXISTS ix_iteration_outcomes_pipeline
    ON iteration_outcomes(campaign, pipeline_id, iteration);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open the campaign DB in WAL mode with sane defaults.

    Creates the file (and parent dirs) and applies the schema if the file
    is new. Safe to call from many processes simultaneously — WAL mode lets
    readers proceed while a writer holds the IMMEDIATE lock.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        # Allow timeouts up to 30s when waiting for a write lock; concurrent
        # IMMEDIATE transactions otherwise would raise immediately.
        timeout=30.0,
        isolation_level=None,  # we manage BEGIN/COMMIT explicitly
    )
    conn.row_factory = sqlite3.Row

    # WAL + foreign keys + busy timeout in ms. WAL persists across closes.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA synchronous = NORMAL")  # safe with WAL, much faster

    # Apply schema (idempotent — uses IF NOT EXISTS).
    conn.executescript(SCHEMA)

    # Forward migrations for legacy DBs created before later columns landed.
    # Cheap (PRAGMA + maybe one ALTER + one UPDATE per missing column) so
    # safe to run on every connect.
    _apply_forward_migrations(conn)
    return conn


def _apply_forward_migrations(conn: sqlite3.Connection) -> None:
    """Apply any schema additions not yet on an older campaign DB.

    SQLite's `ALTER TABLE ... ADD COLUMN` is O(1) (header-only), so each
    migration is cheap. Backfills are scoped (the WHERE clause keeps them
    cheap on already-migrated DBs) and idempotent.
    """
    _ensure_node_edges_support_regression_resolve(conn)

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}

    # `commits_json` (added when we generalised the single-commit `commit_sha`
    # to a list of all commits associated with a node).
    if "commits_json" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN commits_json TEXT NOT NULL DEFAULT '[]'"
        )
        # Backfill is best-effort — the legacy schema doesn't store per-
        # iteration history, so we recover what we can from `commit_sha`:
        #
        #   - Roots (parent_id IS NULL) with a commit_sha: seed
        #     `commits = [commit_sha]` (root represents that one baseline).
        #   - Children whose `commit_sha` equals the parent's `commit_sha`:
        #     no iteration was kept (the HEAD never advanced). Leave
        #     `commits = []`.
        #   - Children whose `commit_sha` differs from the parent's:
        #     at least one iteration was kept; we only know the final tip,
        #     so seed with that single SHA. New runs against this node will
        #     `append_commits` correctly going forward.
        conn.execute(
            "UPDATE nodes SET commits_json = json_array(commit_sha) "
            "WHERE commits_json = '[]' "
            "  AND commit_sha IS NOT NULL "
            "  AND parent_id IS NULL"
        )
        conn.execute(
            """
            UPDATE nodes AS c
               SET commits_json = json_array(c.commit_sha)
             WHERE c.commits_json = '[]'
               AND c.commit_sha IS NOT NULL
               AND c.parent_id IS NOT NULL
               AND c.commit_sha != (
                   SELECT p.commit_sha FROM nodes p WHERE p.id = c.parent_id
               )
            """
        )

    # `picked_commit_sha` + `claimed_task_scores_json` (added when we taught
    # the loop to ask cursor-agent to review the kept iterations and pick
    # the best commit before the final eval).
    if "picked_commit_sha" not in cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN picked_commit_sha TEXT")
    if "claimed_task_scores_json" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN claimed_task_scores_json "
            "TEXT NOT NULL DEFAULT '{}'"
        )

    # Explicit solved/unsolved task lists. `failed_tasks_json` remains as a
    # compatibility column for existing pool/report code, but new code writes
    # `unsolved_tasks_json` as the canonical list.
    if "solved_tasks_json" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN solved_tasks_json "
            "TEXT NOT NULL DEFAULT '[]'"
        )
    if "unsolved_tasks_json" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN unsolved_tasks_json "
            "TEXT NOT NULL DEFAULT '[]'"
        )
        conn.execute(
            "UPDATE nodes SET unsolved_tasks_json = failed_tasks_json "
            "WHERE unsolved_tasks_json = '[]' AND failed_tasks_json != '[]'"
        )
    if "partially_solved_tasks_json" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN partially_solved_tasks_json "
            "TEXT NOT NULL DEFAULT '[]'"
        )
    _backfill_partially_solved_tasks(conn)
    if "improved_tasks_json" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN improved_tasks_json "
            "TEXT NOT NULL DEFAULT '[]'"
        )
    if "regressed_tasks_json" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN regressed_tasks_json "
            "TEXT NOT NULL DEFAULT '[]'"
        )
    _backfill_task_deltas(conn)

    # `claims.campaign` / `claims.subset` / `claims.claim_kind` (added as
    # the scheduler moved from per-parent claims to campaign-wide pools, then
    # split evolver and regression-resolver pools). campaign/subset remain
    # nullable to keep legacy ALTERs cheap; new try_claim_tasks always writes
    # them. Existing rows are backfilled by joining against `nodes` and
    # `node_edges`.
    claim_cols = {
        r["name"] for r in conn.execute("PRAGMA table_info(claims)").fetchall()
    }
    if claim_cols and "campaign" not in claim_cols:
        conn.execute("ALTER TABLE claims ADD COLUMN campaign TEXT")
        conn.execute(
            "UPDATE claims SET campaign = ("
            "  SELECT n.campaign FROM nodes n WHERE n.id = claims.parent_id"
            ") WHERE campaign IS NULL"
        )
    if claim_cols and "subset" not in claim_cols:
        conn.execute("ALTER TABLE claims ADD COLUMN subset TEXT")
        conn.execute(
            "UPDATE claims SET subset = ("
            "  SELECT n.subset FROM nodes n WHERE n.id = claims.parent_id"
            ") WHERE subset IS NULL"
        )
    if claim_cols and "claim_kind" not in claim_cols:
        conn.execute(
            "ALTER TABLE claims ADD COLUMN claim_kind TEXT NOT NULL DEFAULT 'evolve'"
        )
        conn.execute(
            "UPDATE claims SET claim_kind = 'regression_resolve' "
            "WHERE EXISTS ("
            "  SELECT 1 FROM node_edges e "
            "  WHERE e.campaign = claims.campaign "
            "    AND e.pipeline_id = claims.pipeline_id "
            "    AND e.edge_type = 'regression_resolve'"
            ")"
        )
        claim_cols.add("claim_kind")
    # Re-point the partial unique index at
    # (campaign, subset, claim_kind, failure_task).
    # `IF EXISTS` is necessary because the legacy DB's index was on
    # (parent_id, failure_task); we DROP it first, then create the new one.
    # CREATE INDEX in SCHEMA is idempotent (`IF NOT EXISTS`) so it's safe to
    # run on a fresh DB too. We only do the swap when both new columns are
    # present (i.e. either fresh schema or migration above just ran).
    if claim_cols and ("campaign" not in claim_cols or "subset" not in claim_cols):
        # In the same connect cycle we just ALTERed; the swap below applies.
        pass
    if claim_cols:
        # Check the index definition; if it still points at parent_id or
        # does not include claim_kind, drop & recreate it. Fresh DBs already
        # have the new definition from `SCHEMA` so this is a no-op there.
        idx_row = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='index' AND name='uq_claims_active_one_per_task'"
        ).fetchone()
        idx_sql = idx_row["sql"] or "" if idx_row else ""
        if idx_row and ("parent_id" in idx_sql or "claim_kind" not in idx_sql):
            conn.execute("DROP INDEX IF EXISTS uq_claims_active_one_per_task")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_claims_active_one_per_task "
                "ON claims(campaign, subset, claim_kind, failure_task) "
                "WHERE released_at IS NULL"
            )
        # Same story for the helper index: legacy points at parent_id or
        # lacks claim_kind; new should point at
        # (campaign, subset, claim_kind, failure_task, released_at).
        idx_row2 = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='index' AND name='ix_claims_active'"
        ).fetchone()
        idx_sql2 = idx_row2["sql"] or "" if idx_row2 else ""
        if idx_row2 and ("parent_id" in idx_sql2 or "claim_kind" not in idx_sql2):
            conn.execute("DROP INDEX IF EXISTS ix_claims_active")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_claims_active "
                "ON claims(campaign, subset, claim_kind, failure_task, released_at)"
            )

    pipeline_cols = {
        r["name"] for r in conn.execute("PRAGMA table_info(pipelines)").fetchall()
    }
    if pipeline_cols and "pipeline_kind" not in pipeline_cols:
        conn.execute(
            "ALTER TABLE pipelines ADD COLUMN pipeline_kind TEXT NOT NULL DEFAULT 'evolve'"
        )
    if pipeline_cols and "target_node_id" not in pipeline_cols:
        conn.execute("ALTER TABLE pipelines ADD COLUMN target_node_id TEXT")

    # Backfill typed evolve edges for legacy rows whose only relationship
    # record is `nodes.parent_id`.
    conn.execute(
        """
        INSERT OR IGNORE INTO node_edges (
            id, campaign, parent_id, child_id, edge_type, parent_role,
            pipeline_id, created_at
        )
        SELECT lower(hex(randomblob(4))), campaign, parent_id, id,
               'evolve', 'evolve', pipeline_id, created_at
          FROM nodes
         WHERE parent_id IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM node_edges e
                WHERE e.campaign = nodes.campaign
                  AND e.child_id = nodes.id
           )
        """
    )


def _ensure_node_edges_support_regression_resolve(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'node_edges'"
    ).fetchone()
    sql = row["sql"] if row else ""
    if "regression_resolve" in sql:
        return

    with transaction(conn):
        conn.execute("ALTER TABLE node_edges RENAME TO node_edges_legacy")
        conn.execute(
            """
            CREATE TABLE node_edges (
                id           TEXT PRIMARY KEY,
                campaign     TEXT NOT NULL,
                parent_id    TEXT NOT NULL REFERENCES nodes(id),
                child_id     TEXT NOT NULL REFERENCES nodes(id),
                edge_type    TEXT NOT NULL CHECK(edge_type IN ('evolve', 'merge', 'regression_resolve')),
                parent_role  TEXT NOT NULL,
                pipeline_id  TEXT,
                created_at   TEXT NOT NULL,
                UNIQUE(parent_id, child_id, edge_type, parent_role)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO node_edges (
                id, campaign, parent_id, child_id, edge_type, parent_role,
                pipeline_id, created_at
            )
            SELECT id, campaign, parent_id, child_id, edge_type, parent_role,
                   pipeline_id, created_at
              FROM node_edges_legacy
            """
        )
        conn.execute("DROP TABLE node_edges_legacy")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_node_edges_campaign ON node_edges(campaign)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_node_edges_parent ON node_edges(parent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_node_edges_child ON node_edges(child_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_node_edges_type ON node_edges(campaign, edge_type)")


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(x) for x in value] if isinstance(value, list) else []


def _backfill_partially_solved_tasks(conn: sqlite3.Connection) -> None:
    """Repair legacy multi-trial rows that stored mixed tasks in both lists."""
    rows = conn.execute(
        """
        SELECT id, solved_tasks_json, unsolved_tasks_json,
               failed_tasks_json, partially_solved_tasks_json
          FROM nodes
        """
    ).fetchall()
    for row in rows:
        if _json_list(row["partially_solved_tasks_json"]):
            continue
        solved = _json_list(row["solved_tasks_json"])
        failed = _json_list(row["unsolved_tasks_json"]) or _json_list(row["failed_tasks_json"])
        overlap = set(solved) & set(failed)
        if not overlap:
            continue
        partial = [t for t in failed if t in overlap]
        solved_clean = [t for t in solved if t not in overlap]
        unsolved_clean = [t for t in failed if t not in overlap]
        failed_clean = list(dict.fromkeys(partial + unsolved_clean))
        conn.execute(
            """
            UPDATE nodes
               SET solved_tasks_json = ?,
                   unsolved_tasks_json = ?,
                   partially_solved_tasks_json = ?,
                   failed_tasks_json = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (
                json.dumps(solved_clean),
                json.dumps(unsolved_clean),
                json.dumps(partial),
                json.dumps(failed_clean),
                utcnow_iso(),
                row["id"],
            ),
        )


def task_deltas(
    *,
    parent_solved: list[str],
    parent_unsolved: list[str],
    child_solved: list[str],
    child_unsolved: list[str],
) -> tuple[list[str], list[str]]:
    """Return tasks that improved/regressed from parent to child.

    Here "unsolved" means "not fully solved" and may include partial tasks.
    A task improves when the parent did not fully solve it and the child did.
    A task regresses when the parent solved it and the child no longer does.
    Ordering follows the child's eval lists so the visualizer mirrors the
    latest run's task order.
    """
    parent_solved_set = set(parent_solved)
    parent_unsolved_set = set(parent_unsolved)
    improved = [t for t in child_solved if t in parent_unsolved_set]
    regressed = [t for t in child_unsolved if t in parent_solved_set]
    return improved, regressed


def node_outcome_features(node: Node) -> NodeOutcomeFeatures:
    """Return hashable pass/fail/delta sets for selector scoring."""
    return NodeOutcomeFeatures(
        node_id=node.id,
        solved=frozenset(node.solved_tasks),
        unsolved=frozenset(node.failed_tasks),
        partially_solved=frozenset(node.partially_solved_tasks),
        improved=frozenset(node.improved_tasks),
        regressed=frozenset(node.regressed_tasks),
        score=node.score,
    )


def task_outcome_stats(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str | None = None,
) -> dict[str, TaskOutcomeStats]:
    """Summarize campaign-local task fragility and repair history.

    The result is derived entirely from existing node and edge state. Selectors
    use it as a prior: tasks that frequently regress, frequently fail, or have
    repeated unsuccessful repair attempts should be treated as riskier than a
    plain solved/unsolved bitset implies.
    """
    nodes = list_nodes(conn, campaign=campaign, subset=subset)
    by_id = {n.id: n for n in nodes}
    search_evals = search_eval_by_node(conn, campaign=campaign, subset=subset)
    mutable: dict[str, dict[str, int]] = {}

    def bucket(task: str) -> dict[str, int]:
        return mutable.setdefault(task, {
            "passes": 0,
            "failures": 0,
            "improvements": 0,
            "regressions": 0,
            "regression_resolver_attempts": 0,
            "regression_resolver_successes": 0,
            "regression_resolver_failures": 0,
            "merge_attempts": 0,
            "merge_successes": 0,
            "merge_failures": 0,
        })

    for n in nodes:
        ev = search_evals.get(n.id)
        if n.status not in {"completed", "no_change"}:
            continue
        if ev is None and n.score is None:
            continue
        solved_tasks = ev.solved_tasks if ev else n.solved_tasks
        failed_tasks = ev.failed_tasks if ev else n.failed_tasks
        improved_tasks = ev.improved_tasks if ev else n.improved_tasks
        regressed_tasks = ev.regressed_tasks if ev else n.regressed_tasks
        for task in solved_tasks:
            bucket(task)["passes"] += 1
        for task in failed_tasks:
            bucket(task)["failures"] += 1
        for task in improved_tasks:
            bucket(task)["improvements"] += 1
        for task in regressed_tasks:
            bucket(task)["regressions"] += 1

    for edge in list_node_edges(conn, campaign=campaign):
        child = by_id.get(edge.child_id)
        parent = by_id.get(edge.parent_id)
        child_ev = search_evals.get(edge.child_id)
        parent_ev = search_evals.get(edge.parent_id)
        if child is None or parent is None:
            continue
        if child_ev is None and child.score is None:
            continue
        if edge.edge_type == "regression_resolve":
            parent_regressed = parent_ev.regressed_tasks if parent_ev else parent.regressed_tasks
            child_solved = set(child_ev.solved_tasks if child_ev else child.solved_tasks)
            child_score = child_ev.score if child_ev else (child.score or 0.0)
            parent_score = parent_ev.score if parent_ev else (parent.score or 0.0)
            for task in parent_regressed:
                b = bucket(task)
                b["regression_resolver_attempts"] += 1
                if task in child_solved and child_score > parent_score:
                    b["regression_resolver_successes"] += 1
                else:
                    b["regression_resolver_failures"] += 1
        elif edge.edge_type == "merge":
            validation = _merge_validation_metadata(child)
            validation_only = bool(
                validation
                and any(
                    isinstance(item, dict)
                    and isinstance(item.get("merge_delta"), dict)
                    and item["merge_delta"].get("validation_only")
                    for item in child.resolved_tasks
                )
            )
            validated_tasks = set(validation.get("validated_tasks") or []) if validation else set()
            lost_validated = set(
                validation.get("lost_validated_parent_wins")
                or validation.get("lost_parent_wins")
                or []
            ) if validation else set()
            parent_solved = parent_ev.solved_tasks if parent_ev else parent.solved_tasks
            child_solved = set(child_ev.solved_tasks if child_ev else child.solved_tasks)
            for task in parent_solved:
                if validation_only and task not in validated_tasks:
                    continue
                b = bucket(task)
                b["merge_attempts"] += 1
                if validation_only:
                    passed = task not in lost_validated
                else:
                    passed = task in child_solved
                if passed:
                    b["merge_successes"] += 1
                else:
                    b["merge_failures"] += 1

    return {
        task: TaskOutcomeStats(task=task, **values)
        for task, values in mutable.items()
    }


def _merge_validation_metadata(node: Node) -> dict[str, Any] | None:
    for item in node.resolved_tasks:
        if isinstance(item, dict) and isinstance(item.get("merge_validation"), dict):
            return item["merge_validation"]
    return None


def fragile_tasks_from_stats(
    stats: dict[str, TaskOutcomeStats],
    *,
    min_failure_rate: float = 0.35,
) -> set[str]:
    """Tasks whose campaign history suggests extra preservation coverage."""
    fragile: set[str] = set()
    for task, s in stats.items():
        if s.regressions > 0 or s.failure_rate >= min_failure_rate or s.merge_failures > 0:
            fragile.add(task)
    return fragile


def _backfill_task_deltas(conn: sqlite3.Connection) -> None:
    """Populate improved/regressed lists for older rows when possible."""
    rows = conn.execute(
        """
        SELECT
            c.id,
            c.solved_tasks_json AS child_solved,
            c.unsolved_tasks_json AS child_unsolved,
            c.partially_solved_tasks_json AS child_partial,
            c.failed_tasks_json AS child_failed,
            c.improved_tasks_json,
            c.regressed_tasks_json,
            p.solved_tasks_json AS parent_solved,
            p.unsolved_tasks_json AS parent_unsolved,
            p.partially_solved_tasks_json AS parent_partial,
            p.failed_tasks_json AS parent_failed
          FROM nodes c
          JOIN nodes p ON p.id = c.parent_id
         WHERE c.parent_id IS NOT NULL
           AND c.improved_tasks_json = '[]'
           AND c.regressed_tasks_json = '[]'
        """
    ).fetchall()
    for row in rows:
        parent_solved = _json_list(row["parent_solved"])
        parent_unsolved = list(dict.fromkeys(
            _json_list(row["parent_partial"]) +
            (_json_list(row["parent_unsolved"]) or _json_list(row["parent_failed"]))
        ))
        child_solved = _json_list(row["child_solved"])
        child_unsolved = list(dict.fromkeys(
            _json_list(row["child_partial"]) +
            (_json_list(row["child_unsolved"]) or _json_list(row["child_failed"]))
        ))
        improved, regressed = task_deltas(
            parent_solved=parent_solved,
            parent_unsolved=parent_unsolved,
            child_solved=child_solved,
            child_unsolved=child_unsolved,
        )
        if improved or regressed:
            conn.execute(
                """
                UPDATE nodes
                   SET improved_tasks_json = ?,
                       regressed_tasks_json = ?
                 WHERE id = ?
                """,
                (json.dumps(improved), json.dumps(regressed), row["id"]),
            )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Wrap a write in BEGIN IMMEDIATE … COMMIT (rolls back on exception).

    IMMEDIATE acquires the reserved lock immediately, serializing concurrent
    writers without a blind retry loop. Combined with WAL mode this yields
    clean serialization across processes.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ─── Node CRUD ────────────────────────────────────────────────────────────


def insert_node(
    conn: sqlite3.Connection,
    *,
    id: str,
    campaign: str,
    branch_name: str,
    commit_sha: str | None,
    parent_id: str | None,
    subset: str,
    status: str,
    pipeline_id: str | None,
    score: float | None = None,
    job_log_path: str | None = None,
    failed_tasks: list[str] | None = None,
    solved_tasks: list[str] | None = None,
    unsolved_tasks: list[str] | None = None,
    partially_solved_tasks: list[str] | None = None,
    improved_tasks: list[str] | None = None,
    regressed_tasks: list[str] | None = None,
    works_md_path: str | None = None,
    effort_md_path: str | None = None,
    commits: list[str] | None = None,
) -> Node:
    """Insert a fresh node row.

    `commits` is the list of commit SHAs "associated with" this node
    (in chronological order). Callers must decide what to put here:
      - Root nodes: typically `[commit_sha]` (the baseline commit).
      - Evolved children at creation time: `[]` (no iterations kept yet).
        Subsequent kept iterations grow the list via `append_commits`.
    """
    now = utcnow_iso()
    canonical_unsolved = unsolved_tasks if unsolved_tasks is not None else (failed_tasks or [])
    canonical_partial = partially_solved_tasks or []
    canonical_failed = (
        failed_tasks
        if failed_tasks is not None
        else list(dict.fromkeys(canonical_partial + canonical_unsolved))
    )
    failed_json = json.dumps(canonical_failed)
    solved_json = json.dumps(solved_tasks or [])
    unsolved_json = json.dumps(canonical_unsolved)
    partial_json = json.dumps(canonical_partial)
    improved_json = json.dumps(improved_tasks or [])
    regressed_json = json.dumps(regressed_tasks or [])
    commits_json = json.dumps(commits or [])
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO nodes (
                id, campaign, branch_name, commit_sha, commits_json, parent_id, score,
                subset, job_log_path, failed_tasks_json, solved_tasks_json,
                unsolved_tasks_json, partially_solved_tasks_json,
                improved_tasks_json, regressed_tasks_json,
                resolved_tasks_json,
                status, works_md_path, effort_md_path, pipeline_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?)
            """,
            (
                id, campaign, branch_name, commit_sha, commits_json, parent_id, score,
                subset, job_log_path, failed_json, solved_json, unsolved_json,
                partial_json, improved_json, regressed_json,
                status, works_md_path, effort_md_path, pipeline_id,
                now, now,
            ),
        )
    return get_node(conn, id)  # type: ignore[return-value]


def bootstrap_root_if_absent(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    branch_name: str,
    commit_sha: str,
    pipeline_id: str | None,
) -> tuple[str, bool]:
    """Atomic check-then-insert: ensure exactly one root exists for (campaign, subset).

    Returns ``(root_id, created_now)``. ``created_now`` is True iff THIS call
    inserted the root (i.e. caller is responsible for scoring it). Safe to call
    from many parallel pipelines — only one will get ``created_now=True``.
    """
    now = utcnow_iso()
    with transaction(conn):
        # Re-check inside the lock; this is the bug-fix for the race where two
        # parallel workers would both observe an empty tree and both insert.
        existing = conn.execute(
            "SELECT id FROM nodes WHERE campaign = ? AND subset = ? AND parent_id IS NULL "
            "ORDER BY created_at ASC LIMIT 1",
            (campaign, subset),
        ).fetchone()
        if existing:
            return existing["id"], False
        new_id_ = new_id()
        # Root nodes are "associated with" exactly one commit: the baseline
        # they snapshot. Seed commits_json with that single SHA so callers
        # don't need a separate update.
        commits_json = json.dumps([commit_sha]) if commit_sha else "[]"
        conn.execute(
            """
            INSERT INTO nodes (
                id, campaign, branch_name, commit_sha, commits_json, parent_id, score,
                subset, job_log_path, failed_tasks_json, solved_tasks_json,
                unsolved_tasks_json, partially_solved_tasks_json,
                improved_tasks_json, regressed_tasks_json,
                resolved_tasks_json,
                status, works_md_path, effort_md_path, pipeline_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, '[]', '[]', '[]', '[]', '[]', '[]', '[]',
                      'in_progress', NULL, NULL, ?, ?, ?)
            """,
            (
                new_id_, campaign, branch_name, commit_sha, commits_json,
                subset, pipeline_id,
                now, now,
            ),
        )
        return new_id_, True


def update_node(conn: sqlite3.Connection, id: str, **fields: Any) -> None:
    """Patch a node row. JSON fields accept lists/dicts and are encoded."""
    if not fields:
        return
    fields["updated_at"] = utcnow_iso()
    # JSON-encode list/dict values for *_json columns.
    for k in list(fields):
        if k.endswith("_json") and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])
    cols = ", ".join(f"{k} = ?" for k in fields)
    with transaction(conn):
        conn.execute(f"UPDATE nodes SET {cols} WHERE id = ?", (*fields.values(), id))


def get_node(conn: sqlite3.Connection, id: str) -> Node | None:
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (id,)).fetchone()
    return _row_to_node(row) if row else None


def list_nodes(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str | None = None,
    status: str | None = None,
) -> list[Node]:
    sql = "SELECT * FROM nodes WHERE campaign = ?"
    params: list[Any] = [campaign]
    if subset is not None:
        sql += " AND subset = ?"
        params.append(subset)
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY created_at ASC"
    return [_row_to_node(r) for r in conn.execute(sql, params).fetchall()]


def child_count(conn: sqlite3.Connection, node_id: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT child_id) AS n
          FROM (
                SELECT child_id FROM node_edges WHERE parent_id = ?
                UNION
                SELECT id AS child_id FROM nodes WHERE parent_id = ?
          )
        """,
        (node_id, node_id),
    ).fetchone()
    return int(row["n"])


def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        id=row["id"],
        campaign=row["campaign"],
        branch_name=row["branch_name"],
        commit_sha=row["commit_sha"],
        commits_json=row["commits_json"] or "[]",
        picked_commit_sha=row["picked_commit_sha"],
        claimed_task_scores_json=row["claimed_task_scores_json"] or "{}",
        parent_id=row["parent_id"],
        score=row["score"],
        subset=row["subset"],
        job_log_path=row["job_log_path"],
        failed_tasks_json=row["failed_tasks_json"] or "[]",
        solved_tasks_json=row["solved_tasks_json"] or "[]",
        unsolved_tasks_json=row["unsolved_tasks_json"] or "[]",
        partially_solved_tasks_json=row["partially_solved_tasks_json"] or "[]",
        improved_tasks_json=row["improved_tasks_json"] or "[]",
        regressed_tasks_json=row["regressed_tasks_json"] or "[]",
        resolved_tasks_json=row["resolved_tasks_json"] or "[]",
        status=row["status"],
        works_md_path=row["works_md_path"],
        effort_md_path=row["effort_md_path"],
        pipeline_id=row["pipeline_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def append_commits(
    conn: sqlite3.Connection, node_id: str, new_shas: list[str],
) -> list[str]:
    """Atomically append SHAs to a node's commits_json + bump commit_sha.

    Reads the current `commits_json`, appends `new_shas` (skipping any
    SHA already present so retries are idempotent), and writes both
    `commits_json` and `commit_sha` (= the new tail) in one transaction.

    Returns the full updated list. No-op if `new_shas` is empty.
    """
    if not new_shas:
        n = get_node(conn, node_id)
        return n.commits if n else []
    with transaction(conn):
        row = conn.execute(
            "SELECT commits_json FROM nodes WHERE id = ?", (node_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"node {node_id!r} not found")
        existing = json.loads(row["commits_json"] or "[]")
        seen = set(existing)
        for sha in new_shas:
            if sha not in seen:
                existing.append(sha)
                seen.add(sha)
        conn.execute(
            "UPDATE nodes SET commits_json = ?, commit_sha = ?, updated_at = ? "
            "WHERE id = ?",
            (json.dumps(existing), existing[-1], utcnow_iso(), node_id),
        )
    return existing


# ─── Typed DAG edge CRUD ──────────────────────────────────────────────────


def insert_node_edge(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    parent_id: str,
    child_id: str,
    edge_type: str,
    parent_role: str,
    pipeline_id: str | None = None,
) -> NodeEdge:
    """Insert a typed parent→child edge and return the persisted row."""
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"unknown edge_type {edge_type!r}")
    if parent_role not in PARENT_ROLES:
        raise ValueError(f"unknown parent_role {parent_role!r}")
    edge_id = new_id()
    now = utcnow_iso()
    with transaction(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO node_edges (
                id, campaign, parent_id, child_id, edge_type, parent_role,
                pipeline_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id, campaign, parent_id, child_id, edge_type,
                parent_role, pipeline_id, now,
            ),
        )
    edges = list_node_edges(
        conn,
        campaign=campaign,
        parent_id=parent_id,
        child_id=child_id,
        edge_type=edge_type,
        parent_role=parent_role,
    )
    if not edges:
        raise RuntimeError("node edge insert failed")
    return edges[0]


def list_node_edges(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    parent_id: str | None = None,
    child_id: str | None = None,
    edge_type: str | None = None,
    parent_role: str | None = None,
) -> list[NodeEdge]:
    sql = "SELECT * FROM node_edges WHERE campaign = ?"
    params: list[Any] = [campaign]
    if parent_id is not None:
        sql += " AND parent_id = ?"
        params.append(parent_id)
    if child_id is not None:
        sql += " AND child_id = ?"
        params.append(child_id)
    if edge_type is not None:
        sql += " AND edge_type = ?"
        params.append(edge_type)
    if parent_role is not None:
        sql += " AND parent_role = ?"
        params.append(parent_role)
    sql += " ORDER BY created_at ASC"
    return [_row_to_node_edge(r) for r in conn.execute(sql, params).fetchall()]


def parents_for_node(
    conn: sqlite3.Connection, *, campaign: str, child_id: str,
) -> list[NodeEdge]:
    return list_node_edges(conn, campaign=campaign, child_id=child_id)


def merge_attempted_pair_keys(conn: sqlite3.Connection, *, campaign: str) -> set[tuple[str, str]]:
    """Return unordered parent-id pairs for every attempted merge child."""
    rows = conn.execute(
        """
        SELECT child_id, parent_id
          FROM node_edges
         WHERE campaign = ? AND edge_type = 'merge'
         ORDER BY child_id, parent_role
        """,
        (campaign,),
    ).fetchall()
    by_child: dict[str, list[str]] = {}
    for r in rows:
        by_child.setdefault(r["child_id"], []).append(r["parent_id"])
    out: set[tuple[str, str]] = set()
    for parents in by_child.values():
        uniq = sorted(set(parents))
        if len(uniq) >= 2:
            out.add((uniq[0], uniq[1]))
    return out


def merged_pair_exists(
    conn: sqlite3.Connection, *, campaign: str, parent_a: str, parent_b: str,
) -> bool:
    key = tuple(sorted((parent_a, parent_b)))
    return key in merge_attempted_pair_keys(conn, campaign=campaign)


def _row_to_node_edge(row: sqlite3.Row) -> NodeEdge:
    return NodeEdge(
        id=row["id"],
        campaign=row["campaign"],
        parent_id=row["parent_id"],
        child_id=row["child_id"],
        edge_type=row["edge_type"],
        parent_role=row["parent_role"],
        pipeline_id=row["pipeline_id"],
        created_at=row["created_at"],
    )


# ─── Pipeline CRUD + heartbeat ────────────────────────────────────────────


def insert_pipeline(
    conn: sqlite3.Connection,
    *,
    id: str,
    campaign: str,
    parent_node_id: str | None,
    log_path: str | None,
    worktree_path: str | None,
    pid: int,
    host: str | None = None,
    pipeline_kind: str = "evolve",
    target_node_id: str | None = None,
) -> Pipeline:
    host = host or socket.gethostname()
    now = utcnow_iso()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO pipelines (
                id, campaign, parent_node_id, child_node_id,
                selected_tasks_json, status, current_iteration, log_path,
                host, pid, heartbeat_at, worktree_path, pipeline_kind, target_node_id,
                started_at, finished_at
            ) VALUES (?, ?, ?, NULL, '[]', 'preparing', 0, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                id, campaign, parent_node_id, log_path, host, pid, now,
                worktree_path, pipeline_kind, target_node_id, now,
            ),
        )
    return get_pipeline(conn, id)  # type: ignore[return-value]


def update_pipeline(conn: sqlite3.Connection, id: str, **fields: Any) -> None:
    if not fields:
        return
    for k in list(fields):
        if k.endswith("_json") and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])
    cols = ", ".join(f"{k} = ?" for k in fields)
    with transaction(conn):
        conn.execute(f"UPDATE pipelines SET {cols} WHERE id = ?", (*fields.values(), id))


def heartbeat(conn: sqlite3.Connection, pipeline_id: str) -> None:
    """Update `heartbeat_at` so the cleanup tool can spot live pipelines."""
    update_pipeline(conn, pipeline_id, heartbeat_at=utcnow_iso())


def get_pipeline(conn: sqlite3.Connection, id: str) -> Pipeline | None:
    row = conn.execute("SELECT * FROM pipelines WHERE id = ?", (id,)).fetchone()
    return _row_to_pipeline(row) if row else None


def list_pipelines(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    status: str | None = None,
) -> list[Pipeline]:
    sql = "SELECT * FROM pipelines WHERE campaign = ?"
    params: list[Any] = [campaign]
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY started_at DESC"
    return [_row_to_pipeline(r) for r in conn.execute(sql, params).fetchall()]


def _row_to_pipeline(row: sqlite3.Row) -> Pipeline:
    return Pipeline(
        id=row["id"],
        campaign=row["campaign"],
        parent_node_id=row["parent_node_id"],
        child_node_id=row["child_node_id"],
        selected_tasks_json=row["selected_tasks_json"] or "[]",
        status=row["status"],
        current_iteration=row["current_iteration"] or 0,
        log_path=row["log_path"],
        host=row["host"],
        pid=row["pid"],
        heartbeat_at=row["heartbeat_at"],
        worktree_path=row["worktree_path"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        pipeline_kind=row["pipeline_kind"] or "evolve",
        target_node_id=row["target_node_id"],
    )


# ─── Iteration outcome CRUD ───────────────────────────────────────────────


def upsert_iteration_outcome(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    pipeline_id: str,
    node_id: str | None,
    iteration: int,
    stage: str | None = None,
    outcome: str | None = None,
    reason: str | None = None,
    committed_shas: list[str] | None = None,
    reverted: bool | None = None,
    reverted_shas: list[str] | None = None,
    mini_eval_job_path: str | None = None,
    mini_eval_score: float | None = None,
    mini_eval_n_trials: int | None = None,
    mini_eval_n_errors: int | None = None,
    claimed_rewards: dict[str, float] | None = None,
    canary_rewards: dict[str, float] | None = None,
    canary_tasks: list[str] | None = None,
    failed_canaries: list[str] | None = None,
    review_duration_ms: int | None = None,
    review_error: str | None = None,
) -> IterationOutcomeRow:
    """Create/update a structured record for one pipeline iteration.

    Every argument is optional except identity fields. ``None`` means "leave
    the previous value alone" on update, which lets the pipeline write cheap
    progress records at stage boundaries and fill in result fields later.
    """
    now = utcnow_iso()
    row = conn.execute(
        """
        SELECT id, created_at FROM iteration_outcomes
         WHERE campaign = ? AND pipeline_id = ? AND iteration = ?
        """,
        (campaign, pipeline_id, iteration),
    ).fetchone()
    outcome_id = row["id"] if row else new_id()
    created_at = row["created_at"] if row else now

    def encoded(value: Any, default: str) -> str:
        if value is None:
            return default
        return json.dumps(value)

    if row is None:
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO iteration_outcomes (
                    id, campaign, pipeline_id, node_id, iteration, stage,
                    outcome, reason, committed_shas_json, reverted,
                    reverted_shas_json, mini_eval_job_path, mini_eval_score,
                    mini_eval_n_trials, mini_eval_n_errors,
                    claimed_rewards_json, canary_rewards_json,
                    canary_tasks_json, failed_canaries_json,
                    review_duration_ms, review_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome_id, campaign, pipeline_id, node_id, iteration,
                    stage or "", outcome or "", reason or "",
                    encoded(committed_shas, "[]"),
                    1 if reverted else 0,
                    encoded(reverted_shas, "[]"),
                    mini_eval_job_path, mini_eval_score,
                    mini_eval_n_trials, mini_eval_n_errors,
                    encoded(claimed_rewards, "{}"),
                    encoded(canary_rewards, "{}"),
                    encoded(canary_tasks, "[]"),
                    encoded(failed_canaries, "[]"),
                    review_duration_ms, review_error,
                    created_at, now,
                ),
            )
    else:
        fields: dict[str, Any] = {"node_id": node_id, "updated_at": now}
        if stage is not None:
            fields["stage"] = stage
        if outcome is not None:
            fields["outcome"] = outcome
        if reason is not None:
            fields["reason"] = reason
        if committed_shas is not None:
            fields["committed_shas_json"] = json.dumps(committed_shas)
        if reverted is not None:
            fields["reverted"] = 1 if reverted else 0
        if reverted_shas is not None:
            fields["reverted_shas_json"] = json.dumps(reverted_shas)
        if mini_eval_job_path is not None:
            fields["mini_eval_job_path"] = mini_eval_job_path
        if mini_eval_score is not None:
            fields["mini_eval_score"] = mini_eval_score
        if mini_eval_n_trials is not None:
            fields["mini_eval_n_trials"] = mini_eval_n_trials
        if mini_eval_n_errors is not None:
            fields["mini_eval_n_errors"] = mini_eval_n_errors
        if claimed_rewards is not None:
            fields["claimed_rewards_json"] = json.dumps(claimed_rewards)
        if canary_rewards is not None:
            fields["canary_rewards_json"] = json.dumps(canary_rewards)
        if canary_tasks is not None:
            fields["canary_tasks_json"] = json.dumps(canary_tasks)
        if failed_canaries is not None:
            fields["failed_canaries_json"] = json.dumps(failed_canaries)
        if review_duration_ms is not None:
            fields["review_duration_ms"] = review_duration_ms
        if review_error is not None:
            fields["review_error"] = review_error
        cols = ", ".join(f"{k} = ?" for k in fields)
        with transaction(conn):
            conn.execute(
                f"UPDATE iteration_outcomes SET {cols} WHERE id = ?",
                (*fields.values(), outcome_id),
            )
    return get_iteration_outcome(
        conn, campaign=campaign, pipeline_id=pipeline_id, iteration=iteration,
    )  # type: ignore[return-value]


def get_iteration_outcome(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    pipeline_id: str,
    iteration: int,
) -> IterationOutcomeRow | None:
    row = conn.execute(
        """
        SELECT * FROM iteration_outcomes
         WHERE campaign = ? AND pipeline_id = ? AND iteration = ?
        """,
        (campaign, pipeline_id, iteration),
    ).fetchone()
    return _row_to_iteration_outcome(row) if row else None


def list_iteration_outcomes(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    node_id: str | None = None,
    pipeline_id: str | None = None,
) -> list[IterationOutcomeRow]:
    sql = "SELECT * FROM iteration_outcomes WHERE campaign = ?"
    params: list[Any] = [campaign]
    if node_id is not None:
        sql += " AND node_id = ?"
        params.append(node_id)
    if pipeline_id is not None:
        sql += " AND pipeline_id = ?"
        params.append(pipeline_id)
    sql += " ORDER BY iteration ASC"
    return [_row_to_iteration_outcome(r) for r in conn.execute(sql, params).fetchall()]


def _row_to_iteration_outcome(row: sqlite3.Row) -> IterationOutcomeRow:
    return IterationOutcomeRow(
        id=row["id"],
        campaign=row["campaign"],
        pipeline_id=row["pipeline_id"],
        node_id=row["node_id"],
        iteration=row["iteration"],
        stage=row["stage"] or "",
        outcome=row["outcome"] or "",
        reason=row["reason"] or "",
        committed_shas_json=row["committed_shas_json"] or "[]",
        reverted=bool(row["reverted"]),
        reverted_shas_json=row["reverted_shas_json"] or "[]",
        mini_eval_job_path=row["mini_eval_job_path"],
        mini_eval_score=row["mini_eval_score"],
        mini_eval_n_trials=row["mini_eval_n_trials"],
        mini_eval_n_errors=row["mini_eval_n_errors"],
        claimed_rewards_json=row["claimed_rewards_json"] or "{}",
        canary_rewards_json=row["canary_rewards_json"] or "{}",
        canary_tasks_json=row["canary_tasks_json"] or "[]",
        failed_canaries_json=row["failed_canaries_json"] or "[]",
        review_duration_ms=row["review_duration_ms"],
        review_error=row["review_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ─── Node eval CRUD ───────────────────────────────────────────────────────


def upsert_node_eval(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    node_id: str,
    eval_kind: str,
    subset_label: str,
    task_names: list[str] | None = None,
    n_trials: int = 0,
    n_errors: int = 0,
    score: float = 0.0,
    job_log_path: str | None = None,
    solved_tasks: list[str] | None = None,
    unsolved_tasks: list[str] | None = None,
    partially_solved_tasks: list[str] | None = None,
    task_rewards: dict[str, float] | None = None,
    improved_tasks: list[str] | None = None,
    regressed_tasks: list[str] | None = None,
    source_pipeline_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> NodeEval:
    if eval_kind not in EVAL_KINDS:
        raise ValueError(f"unknown eval_kind {eval_kind!r}")
    now = created_at or utcnow_iso()
    row = conn.execute(
        """
        SELECT id FROM node_evals
         WHERE campaign = ?
           AND node_id = ?
           AND eval_kind = ?
           AND (
                source_pipeline_id = ?
                OR (source_pipeline_id IS NULL AND ? IS NULL)
           )
        """,
        (campaign, node_id, eval_kind, source_pipeline_id, source_pipeline_id),
    ).fetchone()
    eval_id = row["id"] if row else new_id()
    payload = {
        "campaign": campaign,
        "node_id": node_id,
        "eval_kind": eval_kind,
        "subset_label": subset_label,
        "task_names_json": json.dumps(task_names or []),
        "n_trials": int(n_trials or 0),
        "n_errors": int(n_errors or 0),
        "score": float(score or 0.0),
        "job_log_path": job_log_path,
        "solved_tasks_json": json.dumps(solved_tasks or []),
        "unsolved_tasks_json": json.dumps(unsolved_tasks or []),
        "partially_solved_tasks_json": json.dumps(partially_solved_tasks or []),
        "task_rewards_json": json.dumps(task_rewards or {}),
        "improved_tasks_json": json.dumps(improved_tasks or []),
        "regressed_tasks_json": json.dumps(regressed_tasks or []),
        "source_pipeline_id": source_pipeline_id,
        "created_at": now,
        "metadata_json": json.dumps(metadata or {}),
    }
    with transaction(conn):
        if row:
            cols = ", ".join(f"{k} = ?" for k in payload)
            conn.execute(
                f"UPDATE node_evals SET {cols} WHERE id = ?",
                (*payload.values(), eval_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO node_evals (
                    id, campaign, node_id, eval_kind, subset_label,
                    task_names_json, n_trials, n_errors, score, job_log_path,
                    solved_tasks_json, unsolved_tasks_json,
                    partially_solved_tasks_json, task_rewards_json,
                    improved_tasks_json, regressed_tasks_json,
                    source_pipeline_id, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (eval_id, *payload.values()),
            )
    out = get_node_eval(conn, eval_id)
    assert out is not None
    return out


def get_node_eval(conn: sqlite3.Connection, eval_id: str) -> NodeEval | None:
    row = conn.execute("SELECT * FROM node_evals WHERE id = ?", (eval_id,)).fetchone()
    return _row_to_node_eval(row) if row else None


def list_node_evals(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    node_id: str | None = None,
    eval_kind: str | None = None,
) -> list[NodeEval]:
    if eval_kind is not None and eval_kind not in EVAL_KINDS:
        raise ValueError(f"unknown eval_kind {eval_kind!r}")
    sql = "SELECT * FROM node_evals WHERE campaign = ?"
    params: list[Any] = [campaign]
    if node_id is not None:
        sql += " AND node_id = ?"
        params.append(node_id)
    if eval_kind is not None:
        sql += " AND eval_kind = ?"
        params.append(eval_kind)
    sql += " ORDER BY created_at ASC, id ASC"
    return [_row_to_node_eval(r) for r in conn.execute(sql, params).fetchall()]


def latest_node_eval(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    node_id: str,
    eval_kind: str,
) -> NodeEval | None:
    if eval_kind not in EVAL_KINDS:
        raise ValueError(f"unknown eval_kind {eval_kind!r}")
    row = conn.execute(
        """
        SELECT * FROM node_evals
         WHERE campaign = ? AND node_id = ? AND eval_kind = ?
         ORDER BY created_at DESC, id DESC
         LIMIT 1
        """,
        (campaign, node_id, eval_kind),
    ).fetchone()
    return _row_to_node_eval(row) if row else None


def node_search_eval(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    node_id: str,
) -> NodeEval | None:
    return latest_node_eval(
        conn, campaign=campaign, node_id=node_id, eval_kind="subset_final",
    )


def node_fullset_eval(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    node_id: str,
) -> NodeEval | None:
    return latest_node_eval(
        conn, campaign=campaign, node_id=node_id, eval_kind="fullset_final",
    )


def root_full_eval(conn: sqlite3.Connection, *, campaign: str) -> NodeEval | None:
    row = conn.execute(
        """
        SELECT e.* FROM node_evals e
        JOIN nodes n ON n.id = e.node_id
         WHERE e.campaign = ? AND e.eval_kind = 'root_full' AND n.parent_id IS NULL
         ORDER BY e.created_at ASC, e.id ASC
         LIMIT 1
        """,
        (campaign,),
    ).fetchone()
    return _row_to_node_eval(row) if row else None


def search_eval_by_node(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str | None = None,
) -> dict[str, NodeEval]:
    sql = """
        SELECT e.* FROM node_evals e
        JOIN (
            SELECT campaign, node_id, MAX(created_at) AS created_at
              FROM node_evals
             WHERE campaign = ? AND eval_kind = 'subset_final'
             GROUP BY campaign, node_id
        ) latest
          ON latest.campaign = e.campaign
         AND latest.node_id = e.node_id
         AND latest.created_at = e.created_at
         AND e.eval_kind = 'subset_final'
    """
    params: list[Any] = [campaign]
    if subset is not None:
        sql += " WHERE e.subset_label = ?"
        params.append(subset)
    out: dict[str, NodeEval] = {}
    for row in conn.execute(sql, params).fetchall():
        ev = _row_to_node_eval(row)
        out[ev.node_id] = ev
    # Read-only fallback for synthetic tests and manually seeded old-style DBs.
    # Fresh campaigns write node_evals and should never rely on this branch.
    for node in list_nodes(conn, campaign=campaign, subset=subset):
        if node.id in out or node.score is None:
            continue
        out[node.id] = NodeEval(
            id=f"legacy:{node.id}",
            campaign=node.campaign,
            node_id=node.id,
            eval_kind="subset_final",
            subset_label=node.subset,
            task_names_json=json.dumps(list(dict.fromkeys(node.solved_tasks + node.failed_tasks))),
            n_trials=0,
            n_errors=0,
            score=float(node.score or 0.0),
            job_log_path=node.job_log_path,
            solved_tasks_json=node.solved_tasks_json or "[]",
            unsolved_tasks_json=node.unsolved_tasks_json or node.failed_tasks_json or "[]",
            partially_solved_tasks_json=node.partially_solved_tasks_json or "[]",
            task_rewards_json=node.claimed_task_scores_json or "{}",
            improved_tasks_json=node.improved_tasks_json or "[]",
            regressed_tasks_json=node.regressed_tasks_json or "[]",
            source_pipeline_id=node.pipeline_id,
            created_at=node.updated_at,
            metadata_json=json.dumps({"basis": "legacy_node_columns"}),
        )
    return out


def best_node_by_search_eval(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    exclude_no_change_merge_children: bool = True,
) -> tuple[Node, NodeEval] | None:
    nodes = list_nodes(conn, campaign=campaign, subset=subset)
    evals = search_eval_by_node(conn, campaign=campaign, subset=subset)
    merge_children = {
        edge.child_id
        for edge in list_node_edges(conn, campaign=campaign, edge_type="merge")
    } if exclude_no_change_merge_children else set()
    # A ``no_change`` node only re-measures its claimed subset (it made no
    # progress over its parent), so its ``failed_tasks`` reflects that tiny
    # subset — not a real frontier. If such a node "aces" its small claimed
    # set (score 1.0, failed=[]) and is allowed to win the best-node slot, the
    # shared pool (derived from ``best.failed_tasks``) collapses to empty and
    # the supervisor early-stops on a phantom "no work" signal. Prefer
    # ``completed`` nodes — whose eval reflects real search progress — and only
    # fall back to ``no_change`` nodes when no completed node has scored yet
    # (synthetic/legacy campaigns). The root is always ``completed``, so under
    # healthy operation a no_change phantom can never hijack the pool.
    completed: list[tuple[Node, NodeEval]] = []
    no_change: list[tuple[Node, NodeEval]] = []
    for n in nodes:
        ev = evals.get(n.id)
        if ev is None:
            continue
        if n.status == "completed":
            completed.append((n, ev))
        elif n.status == "no_change":
            if n.id in merge_children:
                continue
            no_change.append((n, ev))
    eligible = completed or no_change
    if not eligible:
        return None
    eligible.sort(key=lambda pair: (-pair[1].score, pair[0].created_at))
    return eligible[0]


def copy_latest_node_eval(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    source_node_id: str,
    target_node_id: str,
    eval_kind: str,
    source_pipeline_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> NodeEval | None:
    ev = latest_node_eval(
        conn, campaign=campaign, node_id=source_node_id, eval_kind=eval_kind,
    )
    if ev is None:
        return None
    copied_meta = dict(ev.metadata)
    copied_meta.update(metadata or {})
    copied_meta["copied_from_node_id"] = source_node_id
    copied_meta["copied_from_eval_id"] = ev.id
    return upsert_node_eval(
        conn,
        campaign=campaign,
        node_id=target_node_id,
        eval_kind=eval_kind,
        subset_label=ev.subset_label,
        task_names=ev.task_names,
        n_trials=ev.n_trials,
        n_errors=ev.n_errors,
        score=ev.score,
        job_log_path=ev.job_log_path,
        solved_tasks=ev.solved_tasks,
        unsolved_tasks=ev.unsolved_tasks,
        partially_solved_tasks=ev.partially_solved_tasks,
        task_rewards=ev.task_rewards,
        improved_tasks=ev.improved_tasks,
        regressed_tasks=ev.regressed_tasks,
        source_pipeline_id=source_pipeline_id,
        metadata=copied_meta,
    )


def _row_to_node_eval(row: sqlite3.Row) -> NodeEval:
    return NodeEval(
        id=row["id"],
        campaign=row["campaign"],
        node_id=row["node_id"],
        eval_kind=row["eval_kind"],
        subset_label=row["subset_label"],
        task_names_json=row["task_names_json"] or "[]",
        n_trials=row["n_trials"] or 0,
        n_errors=row["n_errors"] or 0,
        score=float(row["score"] or 0.0),
        job_log_path=row["job_log_path"],
        solved_tasks_json=row["solved_tasks_json"] or "[]",
        unsolved_tasks_json=row["unsolved_tasks_json"] or "[]",
        partially_solved_tasks_json=row["partially_solved_tasks_json"] or "[]",
        task_rewards_json=row["task_rewards_json"] or "{}",
        improved_tasks_json=row["improved_tasks_json"] or "[]",
        regressed_tasks_json=row["regressed_tasks_json"] or "[]",
        source_pipeline_id=row["source_pipeline_id"],
        created_at=row["created_at"],
        metadata_json=row["metadata_json"] or "{}",
    )


# ─── Shared experience CRUD ───────────────────────────────────────────────


def insert_task_experience(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    task: str,
    experience_kind: str,
    worker_kind: str,
    node_id: str | None = None,
    pipeline_id: str | None = None,
    commit_sha: str | None = None,
    commit_number: int | None = None,
    eval_kind: str | None = None,
    before_reward: float | None = None,
    after_reward: float | None = None,
    analysis: str = "",
    code_change_summary: str = "",
    artifact_paths: list[str] | None = None,
    log_excerpt: str = "",
    confidence: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> TaskExperience:
    if experience_kind not in EXPERIENCE_KINDS:
        raise ValueError(f"unknown experience_kind {experience_kind!r}")
    exp_id = new_id()
    now = utcnow_iso()
    # Back-compat: a campaign DB created before 'rejected'/'poisoned' were added
    # has a CHECK constraint that only allows 'improved'/'regressed'. Rather than
    # crash the loop, fall back to 'regressed' (still routed to the "avoid"
    # bucket of the digest) and stash the true kind in metadata.
    legacy_map = {"rejected": "regressed", "poisoned": "regressed"}
    stored_kind = experience_kind
    meta = dict(metadata or {})

    def _do_insert(kind: str) -> None:
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO task_experiences (
                    id, campaign, task, node_id, pipeline_id, worker_kind,
                    commit_sha, commit_number, experience_kind, eval_kind,
                    before_reward, after_reward, analysis, code_change_summary,
                    artifact_paths_json, log_excerpt, confidence, created_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id, campaign, task, node_id, pipeline_id, worker_kind,
                    commit_sha, commit_number, kind, eval_kind,
                    before_reward, after_reward, analysis, code_change_summary,
                    json.dumps(artifact_paths or []), log_excerpt,
                    float(confidence), now, json.dumps(meta),
                ),
            )

    try:
        _do_insert(stored_kind)
    except sqlite3.IntegrityError:
        fallback = legacy_map.get(experience_kind)
        if fallback is None:
            raise
        meta["true_experience_kind"] = experience_kind
        stored_kind = fallback
        _do_insert(stored_kind)
    row = conn.execute("SELECT * FROM task_experiences WHERE id = ?", (exp_id,)).fetchone()
    assert row is not None
    return _row_to_task_experience(row)


def list_task_experiences(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> list[TaskExperience]:
    sql = "SELECT * FROM task_experiences WHERE campaign = ?"
    params: list[Any] = [campaign]
    if tasks:
        placeholders = ",".join("?" for _ in tasks)
        sql += f" AND task IN ({placeholders})"
        params.extend(tasks)
    sql += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [_row_to_task_experience(r) for r in conn.execute(sql, params).fetchall()]


def experience_summary_for_tasks(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    tasks: list[str],
    per_task_limit: int = 3,
) -> str:
    if not tasks:
        return ""
    experiences = list_task_experiences(conn, campaign=campaign, tasks=tasks)
    by_task: dict[str, list[TaskExperience]] = {task: [] for task in tasks}
    for exp in experiences:
        bucket = by_task.setdefault(exp.task, [])
        if len(bucket) < per_task_limit:
            bucket.append(exp)
    if not any(by_task.values()):
        return ""
    lines = [
        "Shared experience is advisory. It may be incomplete or inaccurate; inspect original logs when needed.",
    ]
    for task in tasks:
        rows = by_task.get(task) or []
        if not rows:
            continue
        lines.append(f"- {task}:")
        for exp in rows:
            summary = exp.code_change_summary or exp.analysis or "(no summary)"
            lines.append(
                f"  - {exp.experience_kind} by {exp.worker_kind}"
                f" commit={exp.commit_sha or '?'} eval={exp.eval_kind or '?'}"
                f" confidence={exp.confidence:.2f}: {summary}"
            )
            if exp.artifact_paths:
                lines.append(f"    evidence: {', '.join(exp.artifact_paths[:3])}")
    return "\n".join(lines)


def collective_knowledge_summary(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    max_worked: int = 14,
    max_avoid: int = 10,
    min_confidence: float = 0.0,
) -> str:
    """Campaign-wide COLLECTIVE knowledge digest across ALL nodes (not just the
    current pipeline's claimed tasks).

    The per-task ``experience_summary_for_tasks`` only shows a node its own
    lineage's lessons; this aggregates the whole tree's task_experiences so every
    proposer evolves with the campaign's *collective* memory — what changes have
    WORKED (improved tasks) and what has REGRESSED/failed (to avoid) — ordered
    most-recent-first (temporal). Deduplicated by (task, kind, summary) so a
    repeated lesson doesn't crowd the digest. Returns "" when the LTM is empty.
    Advisory only; never raises shape errors at the call site.
    """
    exps = list_task_experiences(conn, campaign=campaign, tasks=None)  # all, created_at DESC
    worked: list[str] = []
    avoid: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for exp in exps:
        if exp.confidence is not None and exp.confidence < min_confidence:
            continue
        summary = (exp.code_change_summary or exp.analysis or "").strip()
        if not summary:
            continue
        key = (exp.task, exp.experience_kind or "", summary[:80])
        if key in seen:
            continue
        seen.add(key)
        line = (f"  - {exp.task}: {summary}"
                f"  (node={(exp.node_id or '?')[:8]}, conf={exp.confidence:.2f})")
        kind = (exp.experience_kind or "").lower()
        if "poison" in kind:
            # Verifier-gaming anti-pattern — surface it prominently in "avoid".
            if len(avoid) < max_avoid:
                avoid.append(line + "  [POISON/anti-pattern: gamed the verifier]")
        elif "regress" in kind or "reject" in kind or "fail" in kind:
            if len(avoid) < max_avoid:
                avoid.append(line)
        elif "improv" in kind or "resolved" in kind or "keep" in kind:
            if len(worked) < max_worked:
                worked.append(line)
        if len(worked) >= max_worked and len(avoid) >= max_avoid:
            break
    if not worked and not avoid:
        return ""
    out = [
        "## Collective knowledge from this campaign so far (advisory, most-recent first)",
        "Lessons aggregated across ALL evolved nodes \u2014 build on what worked, avoid repeating what regressed. Inspect original logs before trusting any single line.",
    ]
    if worked:
        out.append("\nWhat has WORKED (changes that improved tasks):")
        out.extend(worked)
    if avoid:
        out.append("\nWhat has REGRESSED / failed (avoid repeating):")
        out.extend(avoid)
    return "\n".join(out)


def _row_to_task_experience(row: sqlite3.Row) -> TaskExperience:
    return TaskExperience(
        id=row["id"],
        campaign=row["campaign"],
        task=row["task"],
        node_id=row["node_id"],
        pipeline_id=row["pipeline_id"],
        worker_kind=row["worker_kind"],
        commit_sha=row["commit_sha"],
        commit_number=row["commit_number"],
        experience_kind=row["experience_kind"],
        eval_kind=row["eval_kind"],
        before_reward=row["before_reward"],
        after_reward=row["after_reward"],
        analysis=row["analysis"] or "",
        code_change_summary=row["code_change_summary"] or "",
        artifact_paths_json=row["artifact_paths_json"] or "[]",
        log_excerpt=row["log_excerpt"] or "",
        confidence=float(row["confidence"] or 0.5),
        created_at=row["created_at"],
        metadata_json=row["metadata_json"] or "{}",
    )


# ─── Full-set promotion bookkeeping ───────────────────────────────────────


def record_full_eval_round(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    round_index: int,
    triggered_at_count: int,
) -> bool:
    now = utcnow_iso()
    with transaction(conn):
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO full_eval_rounds (
                campaign, round_index, triggered_at_count, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (campaign, round_index, triggered_at_count, now),
        )
        return (cur.rowcount or 0) > 0


def completed_full_eval_rounds(conn: sqlite3.Connection, *, campaign: str) -> set[int]:
    rows = conn.execute(
        "SELECT round_index FROM full_eval_rounds WHERE campaign = ?",
        (campaign,),
    ).fetchall()
    return {int(r["round_index"]) for r in rows}


def try_claim_node_eval(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    node_id: str,
    eval_kind: str,
    pipeline_id: str,
    round_index: int | None = None,
) -> bool:
    if eval_kind not in EVAL_KINDS:
        raise ValueError(f"unknown eval_kind {eval_kind!r}")
    now = utcnow_iso()
    with transaction(conn):
        try:
            conn.execute(
                """
                INSERT INTO node_eval_claims (
                    campaign, node_id, eval_kind, pipeline_id, round_index,
                    claimed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (campaign, node_id, eval_kind, pipeline_id, round_index, now),
            )
        except sqlite3.IntegrityError:
            return False
    return True


def release_node_eval_claims(conn: sqlite3.Connection, *, pipeline_id: str) -> int:
    now = utcnow_iso()
    with transaction(conn):
        cur = conn.execute(
            "UPDATE node_eval_claims SET released_at = ? "
            "WHERE pipeline_id = ? AND released_at IS NULL",
            (now, pipeline_id),
        )
        return cur.rowcount or 0


def active_node_eval_claims(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    eval_kind: str | None = None,
) -> set[str]:
    sql = "SELECT node_id FROM node_eval_claims WHERE campaign = ? AND released_at IS NULL"
    params: list[Any] = [campaign]
    if eval_kind is not None:
        sql += " AND eval_kind = ?"
        params.append(eval_kind)
    return {r["node_id"] for r in conn.execute(sql, params).fetchall()}


# ─── Claim acquisition ────────────────────────────────────────────────────


class ClaimConflict(RuntimeError):
    """Raised when a (parent, task) pair is already claimed by another pipeline."""


def try_claim_tasks(
    conn: sqlite3.Connection,
    *,
    campaign: str,
    subset: str,
    candidate_tasks: list[str],
    k: int,
    pipeline_id: str,
    parent_id: str | None = None,
    claim_kind: str = "evolve",
) -> list[str]:
    """Atomically claim up to `k` tasks from `candidate_tasks` for `pipeline_id`.

    Claims are scoped to `(campaign, subset, claim_kind, failure_task)` — at
    most one pipeline of the same kind can hold an active claim on a given
    failing task per campaign + subset, regardless of which parent node they
    branch from. Evolvers and regression resolvers use different claim kinds,
    so they may work on the same task concurrently.

    `parent_id` is informational (which node the worker chose to branch
    from); not part of any uniqueness constraint.

    Two layers of protection against parallel-pipeline double-claim:

      1. Best-effort: SELECT active claims first, skip tasks already taken.
         Reduces wasted INSERT attempts under low contention.
      2. Hard guarantee: a partial UNIQUE index
         ``(campaign, subset, claim_kind, failure_task) WHERE released_at IS
         NULL`` makes the INSERT itself fail with IntegrityError if a sibling
         pipeline of the same kind raced us. Required because in WAL mode the
         SELECT inside an IMMEDIATE transaction can occasionally miss a
         just-committed sibling write (observed at --max-parallel-workers 4 in
         stress tests).
    """
    if claim_kind not in CLAIM_KINDS:
        raise ValueError(f"unknown claim_kind {claim_kind!r}")
    claimed: list[str] = []
    now = utcnow_iso()
    with transaction(conn):
        # Active claims in this campaign+subset+kind scope.
        rows = conn.execute(
            "SELECT failure_task FROM claims "
            "WHERE campaign = ? AND subset = ? AND claim_kind = ? "
            "AND released_at IS NULL",
            (campaign, subset, claim_kind),
        ).fetchall()
        active = {r["failure_task"] for r in rows}
        for task in candidate_tasks:
            if len(claimed) >= k:
                break
            if task in active:
                continue
            try:
                conn.execute(
                    "INSERT INTO claims (campaign, subset, claim_kind, parent_id, "
                    "failure_task, pipeline_id, claimed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (campaign, subset, claim_kind, parent_id, task, pipeline_id, now),
                )
            except sqlite3.IntegrityError:
                # A sibling pipeline already holds the partial-unique active
                # claim for this (campaign, subset, kind, task). Layer-2 of
                # the double-claim defense kicked in. Skip and keep going.
                continue
            claimed.append(task)
            active.add(task)
    return claimed


def release_claims(conn: sqlite3.Connection, *, pipeline_id: str) -> int:
    """Mark all of pipeline_id's active claims as released. Returns count."""
    now = utcnow_iso()
    with transaction(conn):
        cur = conn.execute(
            "UPDATE claims SET released_at = ? "
            "WHERE pipeline_id = ? AND released_at IS NULL",
            (now, pipeline_id),
        )
        return cur.rowcount or 0


def list_active_claims(
    conn: sqlite3.Connection,
    *,
    campaign: str | None = None,
    subset: str | None = None,
    claim_kind: str | None = None,
    parent_id: str | None = None,
    pipeline_id: str | None = None,
) -> list[Claim]:
    """List currently-active claims, optionally filtered by scope.

    `(campaign, subset, claim_kind)` is the active uniqueness scope;
    `parent_id` is retained as an informational filter for the legacy "which
    claims branched off this node?" query (still used in a few read paths).
    """
    if claim_kind is not None and claim_kind not in CLAIM_KINDS:
        raise ValueError(f"unknown claim_kind {claim_kind!r}")
    sql = "SELECT * FROM claims WHERE released_at IS NULL"
    params: list[Any] = []
    if campaign:
        sql += " AND campaign = ?"
        params.append(campaign)
    if subset:
        sql += " AND subset = ?"
        params.append(subset)
    if claim_kind:
        sql += " AND claim_kind = ?"
        params.append(claim_kind)
    if parent_id:
        sql += " AND parent_id = ?"
        params.append(parent_id)
    if pipeline_id:
        sql += " AND pipeline_id = ?"
        params.append(pipeline_id)
    sql += " ORDER BY claimed_at ASC"
    return [
        Claim(
            campaign=r["campaign"],
            subset=r["subset"],
            claim_kind=r["claim_kind"],
            parent_id=r["parent_id"],
            failure_task=r["failure_task"],
            pipeline_id=r["pipeline_id"],
            claimed_at=r["claimed_at"],
            released_at=r["released_at"],
        )
        for r in conn.execute(sql, params).fetchall()
    ]


# ─── Path helpers ─────────────────────────────────────────────────────────


def campaign_root(reports_root: Path | str, campaign: str) -> Path:
    return Path(reports_root) / "self_evolve" / campaign


def db_path_for(reports_root: Path | str, campaign: str) -> Path:
    return campaign_root(reports_root, campaign) / "state.db"


def node_dir(reports_root: Path | str, campaign: str, node_id: str) -> Path:
    return campaign_root(reports_root, campaign) / "nodes" / node_id


def pipeline_dir(reports_root: Path | str, campaign: str, pipeline_id: str) -> Path:
    return campaign_root(reports_root, campaign) / "pipelines" / pipeline_id


def prompts_dir(reports_root: Path | str, campaign: str, pipeline_id: str) -> Path:
    return campaign_root(reports_root, campaign) / "prompts" / pipeline_id


# Default reports root. Discovered from the package location so any clone
# works out of the box: coding-bench layout <repo>/self_evolve/tree.py →
# parents[1] is the repo root, whose sibling `reports/` dir (repo_root.parent)
# holds per-campaign state.
#
# Override with the env var MONET_EVAL_REPORTS_ROOT or the --reports-root flag
# on every entry script.
def _default_reports_root() -> Path:
    env = os.environ.get("MONET_EVAL_REPORTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1].parent / "reports"


DEFAULT_REPORTS_ROOT = _default_reports_root()


__all__ = [
    "Node",
    "Pipeline",
    "IterationOutcomeRow",
    "Claim",
    "NodeEval",
    "TaskExperience",
    "NodeEdge",
    "TaskOutcomeStats",
    "NodeOutcomeFeatures",
    "ClaimConflict",
    "NODE_STATUSES",
    "PIPELINE_NON_TERMINAL",
    "PIPELINE_TERMINAL",
    "CLAIM_KINDS",
    "EVAL_KINDS",
    "EXPERIENCE_KINDS",
    "DEFAULT_REPORTS_ROOT",
    "connect",
    "transaction",
    "utcnow_iso",
    "new_id",
    "task_deltas",
    "node_outcome_features",
    "task_outcome_stats",
    "fragile_tasks_from_stats",
    "insert_node",
    "bootstrap_root_if_absent",
    "update_node",
    "append_commits",
    "get_node",
    "list_nodes",
    "child_count",
    "insert_node_edge",
    "list_node_edges",
    "parents_for_node",
    "merge_attempted_pair_keys",
    "merged_pair_exists",
    "insert_pipeline",
    "update_pipeline",
    "heartbeat",
    "get_pipeline",
    "list_pipelines",
    "upsert_iteration_outcome",
    "get_iteration_outcome",
    "list_iteration_outcomes",
    "upsert_node_eval",
    "get_node_eval",
    "list_node_evals",
    "latest_node_eval",
    "node_search_eval",
    "node_fullset_eval",
    "root_full_eval",
    "search_eval_by_node",
    "best_node_by_search_eval",
    "copy_latest_node_eval",
    "insert_task_experience",
    "list_task_experiences",
    "experience_summary_for_tasks",
    "collective_knowledge_summary",
    "record_full_eval_round",
    "completed_full_eval_rounds",
    "try_claim_node_eval",
    "release_node_eval_claims",
    "active_node_eval_claims",
    "try_claim_tasks",
    "release_claims",
    "list_active_claims",
    "campaign_root",
    "db_path_for",
    "node_dir",
    "pipeline_dir",
    "prompts_dir",
]
