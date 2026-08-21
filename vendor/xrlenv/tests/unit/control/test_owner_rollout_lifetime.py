"""Unit tests for the owner_rollout_lifetime durable scoreboard feature.

Tests cover:

StateStore (both InMemory and Sqlite backends):
1. owner_rollout_lifetime() returns {} on an empty store.
2. After pruning terminal rows, lifetime holds exact per-owner/status counts;
   live raw_rollouts keeps non-expired rows; active rows never pruned nor counted.
3. No double-count / idempotency: pruning twice leaves lifetime unchanged and
   sum(live) + sum(lifetime) == total ever inserted.
4. Multiple owners AND multiple statuses accumulate independently.
5. Active (acquiring/running) rows are never included in lifetime even when old.

Sqlite-only:
6. Batching: batch_size smaller than expiring rows still accumulates every row
   exactly once (no loss, no dup).
7. Persistence across reopen: lifetime table survives close+reopen.
8. Legacy DB migration: a database that predates owner_rollout_lifetime migrates
   cleanly (table is created; existing rows unaffected).

Admin server (_users_blocking):
9. all-time totals = live + lifetime; active comes only from live rows.
10. per-owner total / released / success_pct computed correctly with pruned rows.
11. paced (capacity_rejected) still excluded from total even after pruning.
12. Multiple owners accumulate independently in the merged scoreboard.
13. _users_blocking on empty store returns empty rows with zeroed totals.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from xrlenv.admin.server import AdminServerConfig, _users_blocking
from xrlenv.control.state import (
    InMemoryStateStore,
    RawRolloutRecord,
    SqliteStateStore,
    StateStore,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

_OLD = time.time() - 200 * 86400  # 200 days ago — past any retention window
_RECENT = time.time()              # now — inside any retention window


def _raw(
    rollout_id: str,
    status: str,
    *,
    owner: str = "default",
    created_at: float | None = None,
    finished_at: float | None = None,
) -> RawRolloutRecord:
    """Convenience factory for RawRolloutRecord.

    Uses _OLD timestamps for both created_at and finished_at by default so rows
    are expired (past the 14-day retention window when prune_expired is called at
    `now=time.time()`).
    """
    ca = created_at if created_at is not None else _OLD
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status=status,  # type: ignore[arg-type]
        image="busybox:1",
        owner_id=owner,
        created_at=ca,
        finished_at=finished_at if finished_at is not None else (
            ca if status not in ("acquiring", "running") else None
        ),
    )


def _prune(store: StateStore) -> dict[str, int]:
    """Run prune_expired with a 14-day window evaluated at now."""
    return store.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Parametrized fixture — runs each test against BOTH backends
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["inmem", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StateStore]:
    if request.param == "inmem":
        s: StateStore = InMemoryStateStore()
        yield s
    else:
        s = SqliteStateStore(tmp_path / "state.db")
        try:
            yield s
        finally:
            s.close()  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Empty store returns {}
# ──────────────────────────────────────────────────────────────────────────────


def test_owner_rollout_lifetime_empty_store_returns_empty_dict(
    store: StateStore,
) -> None:
    """owner_rollout_lifetime() must return {} when no pruning has occurred."""
    assert store.owner_rollout_lifetime() == {}


# ──────────────────────────────────────────────────────────────────────────────
# 2. After pruning: lifetime holds exact counts; live keeps unexpired rows;
#    active rows are never pruned and never in lifetime.
# ──────────────────────────────────────────────────────────────────────────────


def test_pruned_terminal_rows_accumulate_in_lifetime(store: StateStore) -> None:
    """Terminal rows past retention land in lifetime; non-expired stay live."""
    # Two expired terminal rows for alice.
    store.record_raw_rollout(_raw("r1", "released", owner="alice"))
    store.record_raw_rollout(_raw("r2", "failed", owner="alice"))
    # One fresh terminal row — not expired, must stay live.
    store.record_raw_rollout(
        _raw("r3", "released", owner="alice", created_at=_RECENT, finished_at=_RECENT)
    )

    counts = _prune(store)
    assert counts["raw_rollouts"] == 2

    lifetime = store.owner_rollout_lifetime()
    assert lifetime == {"alice": {"released": 1, "failed": 1}}

    # Non-expired row stays in live raw_rollouts.
    assert store.get_raw_rollout("r3") is not None
    assert store.get_raw_rollout("r1") is None
    assert store.get_raw_rollout("r2") is None


def test_active_rows_never_pruned_never_in_lifetime(store: StateStore) -> None:
    """acquiring/running rows are never GC'd and never count in lifetime,
    even when they are old (stale active rows)."""
    store.record_raw_rollout(_raw("run1", "running", owner="bob"))
    store.record_raw_rollout(_raw("acq1", "acquiring", owner="bob"))
    # One expired terminal to trigger a GC pass.
    store.record_raw_rollout(_raw("done1", "released", owner="bob"))

    counts = _prune(store)
    assert counts["raw_rollouts"] == 1  # only the terminal row pruned

    # Active rows survive.
    assert store.get_raw_rollout("run1") is not None
    assert store.get_raw_rollout("acq1") is not None

    # Lifetime only has the terminal row; acquiring/running never appear.
    lifetime = store.owner_rollout_lifetime()
    assert "running" not in lifetime.get("bob", {})
    assert "acquiring" not in lifetime.get("bob", {})
    assert lifetime == {"bob": {"released": 1}}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Idempotency: pruning twice leaves lifetime unchanged; no double-count
# ──────────────────────────────────────────────────────────────────────────────


def test_second_prune_is_noop_lifetime_unchanged(store: StateStore) -> None:
    """After all expired rows are pruned, a second prune() call is a no-op and
    must NOT add duplicate entries to lifetime."""
    store.record_raw_rollout(_raw("r1", "released", owner="carol"))
    store.record_raw_rollout(_raw("r2", "cancelled", owner="carol"))

    _prune(store)
    after_first = store.owner_rollout_lifetime()

    # Second call — nothing left to prune.
    counts2 = _prune(store)
    assert counts2["raw_rollouts"] == 0

    after_second = store.owner_rollout_lifetime()
    assert after_second == after_first, (
        f"Second prune changed lifetime: {after_first!r} → {after_second!r}"
    )


def test_live_plus_lifetime_equals_total_ever_inserted(store: StateStore) -> None:
    """sum(live aggregate counts) + sum(lifetime counts) == total rows inserted.

    This is the fundamental no-double-count invariant: a row is in exactly one
    source — raw_rollouts (while alive) or owner_rollout_lifetime (after GC).
    """
    # 3 old terminal rows (will be pruned).
    store.record_raw_rollout(_raw("e1", "released", owner="dan"))
    store.record_raw_rollout(_raw("e2", "released", owner="dan"))
    store.record_raw_rollout(_raw("e3", "failed", owner="dan"))
    # 2 fresh terminal rows (will stay live).
    store.record_raw_rollout(
        _raw("f1", "released", owner="dan", created_at=_RECENT, finished_at=_RECENT)
    )
    store.record_raw_rollout(
        _raw("f2", "cancelled", owner="dan", created_at=_RECENT, finished_at=_RECENT)
    )
    total_inserted = 5

    _prune(store)

    # Live rows contribute via the aggregate.
    live_agg = store.aggregate_raw_rollouts_by_owner_status()
    live_sum = sum(sum(by_s.values()) for by_s in live_agg.values())

    # Pruned rows contribute via lifetime.
    lifetime = store.owner_rollout_lifetime()
    lifetime_sum = sum(sum(by_s.values()) for by_s in lifetime.values())

    assert live_sum + lifetime_sum == total_inserted, (
        f"Expected {total_inserted} total; "
        f"live={live_sum}, lifetime={lifetime_sum}, sum={live_sum + lifetime_sum}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Multiple owners AND multiple statuses accumulate independently
# ──────────────────────────────────────────────────────────────────────────────


def test_multiple_owners_and_statuses_accumulate_independently(
    store: StateStore,
) -> None:
    """Each (owner, status) bucket must be independent — cross-owner contamination
    and status conflation must not occur."""
    inserts = [
        ("alice", "released"),
        ("alice", "released"),
        ("alice", "failed"),
        ("bob", "released"),
        ("bob", "cancelled"),
        ("bob", "reaped"),
    ]
    for i, (owner, status) in enumerate(inserts):
        store.record_raw_rollout(_raw(f"r{i}", status, owner=owner))

    _prune(store)

    lifetime = store.owner_rollout_lifetime()
    assert set(lifetime.keys()) == {"alice", "bob"}

    assert lifetime["alice"] == {"released": 2, "failed": 1}
    assert lifetime["bob"] == {"released": 1, "cancelled": 1, "reaped": 1}


def test_incremental_prune_accumulates_across_multiple_prune_calls(
    store: StateStore,
) -> None:
    """Accumulated counts grow correctly across multiple separate prune calls
    (each call prunes a different cohort of rows)."""
    # First cohort — prune at time 0 + ε.
    store.record_raw_rollout(_raw("r1", "released", owner="eve"))

    _prune(store)
    after_first = store.owner_rollout_lifetime()
    assert after_first == {"eve": {"released": 1}}

    # Second cohort — add a new expired row, prune again.
    store.record_raw_rollout(_raw("r2", "released", owner="eve"))
    _prune(store)

    after_second = store.owner_rollout_lifetime()
    assert after_second == {"eve": {"released": 2}}, (
        f"Expected {{'eve': {{'released': 2}}}}, got {after_second!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. All terminal statuses accumulate (not just released)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", [
    "released", "cancelled", "failed", "reaped", "capacity_rejected",
])
def test_all_terminal_statuses_accumulated_by_prune(
    status: str, store: StateStore,
) -> None:
    """Every terminal status is folded into lifetime on prune."""
    store.record_raw_rollout(_raw("r1", status, owner="frank"))
    _prune(store)
    lifetime = store.owner_rollout_lifetime()
    assert lifetime == {"frank": {status: 1}}


# ──────────────────────────────────────────────────────────────────────────────
# 6. Sqlite-only: batching — batch_size smaller than expiring rows
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_small_batch_size_accumulates_all_rows(tmp_path: Path) -> None:
    """With batch_size=2 and 7 expiring rows, all 7 must be counted in lifetime
    (no loss, no duplication despite multiple batch iterations)."""
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        n = 7
        for i in range(n):
            s.record_raw_rollout(_raw(f"r{i}", "released", owner="grace"))

        counts = s.prune_expired(
            now=time.time(),
            audit_retention_days=None,
            events_retention_days=None,
            raw_rollout_retention_days=14,
            batch_size=2,  # forces 4 iterations: 2+2+2+1
        )
        assert counts["raw_rollouts"] == n

        lifetime = s.owner_rollout_lifetime()
        assert lifetime == {"grace": {"released": n}}, (
            f"Expected all {n} rows in lifetime; got {lifetime!r}"
        )

        # Nothing left in live raw_rollouts.
        assert s.list_raw_rollouts() == []
    finally:
        s.close()


def test_sqlite_batch_size_exact_multiple(tmp_path: Path) -> None:
    """batch_size exactly divides the row count (6 rows, batch_size=3):
    forces exactly 2 full batches + the sentinel empty batch that ends the loop.
    All rows still accumulated exactly once."""
    s = SqliteStateStore(tmp_path / "state.db")
    try:
        n = 6
        for i in range(n):
            s.record_raw_rollout(_raw(f"r{i}", "failed", owner="heidi"))

        counts = s.prune_expired(
            now=time.time(),
            audit_retention_days=None,
            events_retention_days=None,
            raw_rollout_retention_days=14,
            batch_size=3,
        )
        assert counts["raw_rollouts"] == n
        assert s.owner_rollout_lifetime() == {"heidi": {"failed": n}}
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────────────
# 7. Sqlite-only: persistence across close + reopen
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_lifetime_persists_across_close_reopen(tmp_path: Path) -> None:
    """owner_rollout_lifetime rows must survive a close+reopen — they live in
    the durable owner_rollout_lifetime table, not in RAM."""
    db = tmp_path / "state.db"
    s1 = SqliteStateStore(db)
    s1.record_raw_rollout(_raw("r1", "released", owner="igor"))
    s1.record_raw_rollout(_raw("r2", "failed", owner="igor"))
    s1.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )
    s1.close()

    s2 = SqliteStateStore(db)
    try:
        lifetime = s2.owner_rollout_lifetime()
        assert lifetime == {"igor": {"released": 1, "failed": 1}}, (
            f"Lifetime not preserved across reopen: {lifetime!r}"
        )
    finally:
        s2.close()


# ──────────────────────────────────────────────────────────────────────────────
# 8. Sqlite-only: legacy DB migration creates owner_rollout_lifetime table
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_migration_creates_owner_rollout_lifetime_table(
    tmp_path: Path,
) -> None:
    """A database that predates owner_rollout_lifetime (lacks the table) migrates
    cleanly: the table is created, existing raw_rollouts rows are unaffected,
    and owner_rollout_lifetime() returns {} on the fresh table.

    We simulate a "pre-feature" DB by creating the full raw_rollouts table
    (with all expected columns) but deliberately omitting owner_rollout_lifetime,
    then verifying SqliteStateStore.__init__'s _SCHEMA CREATE TABLE IF NOT EXISTS
    adds it transparently.
    """
    db = tmp_path / "legacy.db"

    # Build a pre-existing DB that has raw_rollouts (with the full current column
    # set) but lacks the owner_rollout_lifetime table — the state of a database
    # produced before this feature landed.
    con = sqlite3.connect(str(db))
    try:
        con.executescript("""
            PRAGMA journal_mode = WAL;
            CREATE TABLE raw_rollouts (
                rollout_id               TEXT PRIMARY KEY,
                status                   TEXT NOT NULL,
                image                    TEXT NOT NULL,
                node_id                  TEXT,
                container_id             TEXT,
                container_name           TEXT,
                artifact_path            TEXT,
                displayed_name           TEXT,
                task_key                 TEXT,
                group_id                 TEXT,
                fleet_id                 TEXT,
                owner_id                 TEXT NOT NULL DEFAULT 'default',
                created_at               REAL NOT NULL,
                finished_at              REAL,
                error                    TEXT,
                deadline_at              REAL,
                effective_resources_json TEXT
            );
            INSERT INTO raw_rollouts (rollout_id, status, image, owner_id, created_at)
            VALUES ('legacy-r1', 'released', 'img:1', 'legacy-owner', 0.0);
        """)
        con.commit()
    finally:
        con.close()

    # Verify the table does NOT yet exist before SqliteStateStore opens the DB.
    raw_check = sqlite3.connect(str(db))
    tables_before = {
        r[0] for r in raw_check.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    raw_check.close()
    assert "owner_rollout_lifetime" not in tables_before, (
        "Test precondition violated: owner_rollout_lifetime should not exist yet"
    )

    # Opening via SqliteStateStore runs _SCHEMA which has
    # CREATE TABLE IF NOT EXISTS owner_rollout_lifetime → table is created.
    s = SqliteStateStore(db)
    try:
        # Table now exists and is empty.
        assert s.owner_rollout_lifetime() == {}

        # The legacy raw_rollout row is untouched.
        rr = s.get_raw_rollout("legacy-r1")
        assert rr is not None
        assert rr.owner_id == "legacy-owner"
        assert rr.status == "released"
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────────────
# Admin server (_users_blocking) tests
# These use SqliteStateStore only (the function opens a SqliteStateStore).
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    r = tmp_path / "runs"
    r.mkdir()
    return r


def _cfg(state_db: Path, runs_root: Path) -> AdminServerConfig:
    return AdminServerConfig(state_db=state_db, runs_root=runs_root)


# 9. all-time totals = live + lifetime; active only from live rows


def test_users_blocking_merges_live_and_lifetime(
    state_db: Path, runs_root: Path,
) -> None:
    """_users_blocking totals must be live + lifetime (all-time scoreboard).

    Setup: 2 old terminal rows (will be pruned → go to lifetime) + 1 fresh live
    terminal row + 1 active (running) live row.

    After pruning, expected for alice:
      - released = 3 (2 from lifetime + 1 live)
      - active = 1 (the running row, from live only)
      - total = released(3) + active(1) = 4  (active is included in total;
        only capacity_rejected is excluded)
      - success_pct = released(3) / total(4) = 75%
    """
    s = SqliteStateStore(state_db)
    # Old terminal → will end up in lifetime.
    s.record_raw_rollout(_raw("old1", "released", owner="alice"))
    s.record_raw_rollout(_raw("old2", "released", owner="alice"))
    # Fresh terminal → stays live.
    s.record_raw_rollout(
        _raw("new1", "released", owner="alice", created_at=_RECENT, finished_at=_RECENT)
    )
    # Active row → stays live; contributes to active AND total (not excluded).
    s.record_raw_rollout(_raw("act1", "running", owner="alice"))

    s.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )
    s.close()

    data = _users_blocking(_cfg(state_db, runs_root))

    rows = {r["owner"]: r for r in data["rows"]}
    assert "alice" in rows, f"alice missing from rows: {data['rows']!r}"

    alice = rows["alice"]
    assert alice["released"] == 3, f"released={alice['released']!r}, expected 3"
    # active is part of total (only capacity_rejected excluded from total).
    assert alice["active"] == 1, f"active={alice['active']!r}, expected 1"
    assert alice["total"] == 4, f"total={alice['total']!r}, expected 4"
    # success_pct = released(3) / total(4) = 75%
    assert alice["success_pct"] == pytest.approx(75.0, rel=1e-3)


def test_users_blocking_active_comes_only_from_live_rows(
    state_db: Path, runs_root: Path,
) -> None:
    """active counts only acquiring+running rows in live raw_rollouts.

    total = sum(all statuses) - paced (capacity_rejected). Active rows ARE
    included in total — only capacity_rejected is excluded. This test verifies
    that the active counter reflects only live rows (never pruned ones).
    """
    s = SqliteStateStore(state_db)
    # Pruned terminal row → goes to lifetime.
    s.record_raw_rollout(_raw("t1", "released", owner="bob"))
    # Two active live rows → stay live; contribute to active AND total.
    s.record_raw_rollout(_raw("a1", "running", owner="bob"))
    s.record_raw_rollout(_raw("a2", "acquiring", owner="bob"))
    s.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )
    s.close()

    data = _users_blocking(_cfg(state_db, runs_root))
    (bob,) = [r for r in data["rows"] if r["owner"] == "bob"]

    # active = 2 (from live rows only).
    assert bob["active"] == 2
    # total = released(1, from lifetime) + running(1) + acquiring(1) = 3.
    assert bob["total"] == 3
    # released from lifetime shows in the scoreboard.
    assert bob["released"] == 1


# 10. per-owner total / released / success_pct correct with pruned rows


def test_users_blocking_success_pct_correct_with_pruned_rows(
    state_db: Path, runs_root: Path,
) -> None:
    """success_pct = released / total where total includes pruned rows.

    Alice: 3 released (2 pruned + 1 live) + 1 failed (pruned) = total 4.
    success_pct = 3/4 = 75%.
    """
    s = SqliteStateStore(state_db)
    s.record_raw_rollout(_raw("r1", "released", owner="alice"))
    s.record_raw_rollout(_raw("r2", "released", owner="alice"))
    s.record_raw_rollout(_raw("r3", "failed", owner="alice"))
    # Fresh live released.
    s.record_raw_rollout(
        _raw("r4", "released", owner="alice", created_at=_RECENT, finished_at=_RECENT)
    )
    s.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )
    s.close()

    data = _users_blocking(_cfg(state_db, runs_root))
    (alice,) = [r for r in data["rows"] if r["owner"] == "alice"]

    assert alice["total"] == 4
    assert alice["released"] == 3
    assert alice["failed"] == 1
    assert alice["success_pct"] == pytest.approx(75.0, rel=1e-3)


# 11. paced (capacity_rejected) excluded from total even with pruned rows


def test_users_blocking_paced_excluded_from_total_with_lifetime(
    state_db: Path, runs_root: Path,
) -> None:
    """capacity_rejected is excluded from total whether it's live or in lifetime."""
    s = SqliteStateStore(state_db)
    # Two old capacity_rejected rows → go to lifetime.
    s.record_raw_rollout(_raw("p1", "capacity_rejected", owner="charlie"))
    s.record_raw_rollout(_raw("p2", "capacity_rejected", owner="charlie"))
    # One released → also pruned → lifetime.
    s.record_raw_rollout(_raw("r1", "released", owner="charlie"))
    s.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )
    s.close()

    data = _users_blocking(_cfg(state_db, runs_root))
    (charlie,) = [r for r in data["rows"] if r["owner"] == "charlie"]

    # total excludes paced even from lifetime.
    assert charlie["total"] == 1, f"total={charlie['total']!r}, expected 1"
    assert charlie["paced"] == 2, f"paced={charlie['paced']!r}, expected 2"
    assert charlie["released"] == 1
    assert charlie["success_pct"] == pytest.approx(100.0)


# 12. Multiple owners accumulate independently in the merged scoreboard


def test_users_blocking_multiple_owners_independent(
    state_db: Path, runs_root: Path,
) -> None:
    """Lifetime + live merging for multiple owners must not cross-contaminate."""
    s = SqliteStateStore(state_db)
    # Alice: 2 pruned released.
    s.record_raw_rollout(_raw("a1", "released", owner="alice"))
    s.record_raw_rollout(_raw("a2", "released", owner="alice"))
    # Bob: 1 pruned released + 1 pruned failed.
    s.record_raw_rollout(_raw("b1", "released", owner="bob"))
    s.record_raw_rollout(_raw("b2", "failed", owner="bob"))
    s.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )
    s.close()

    data = _users_blocking(_cfg(state_db, runs_root))
    rows_by_owner = {r["owner"]: r for r in data["rows"]}

    alice = rows_by_owner["alice"]
    assert alice["total"] == 2
    assert alice["released"] == 2
    assert alice["failed"] == 0

    bob = rows_by_owner["bob"]
    assert bob["total"] == 2
    assert bob["released"] == 1
    assert bob["failed"] == 1
    assert bob["success_pct"] == pytest.approx(50.0, rel=1e-3)


# 13. _users_blocking on empty / non-existent db returns empty rows


def test_users_blocking_empty_store_returns_empty_rows(
    state_db: Path, runs_root: Path,
) -> None:
    """_users_blocking with no state_db (file doesn't exist) returns a valid empty
    response with no rows and zeroed totals."""
    # Do NOT create the DB file — exercise the cfg.state_db.exists() == False path.
    data = _users_blocking(_cfg(state_db, runs_root))

    assert data["rows"] == []
    assert data["totals"]["total"] == 0
    assert data["totals"]["released"] == 0
    assert data["totals"]["active"] == 0
    assert data["totals"]["paced"] == 0
    assert data["totals"]["success_pct"] is None
    assert data["span_start"] is None
    assert data["span_end"] is None


def test_users_blocking_global_totals_across_all_owners(
    state_db: Path, runs_root: Path,
) -> None:
    """The totals row at data['totals'] sums across all owners including lifetime."""
    s = SqliteStateStore(state_db)
    # Alice: 2 old released → lifetime.
    s.record_raw_rollout(_raw("a1", "released", owner="alice"))
    s.record_raw_rollout(_raw("a2", "released", owner="alice"))
    # Bob: 1 old released → lifetime; 1 fresh released → live.
    s.record_raw_rollout(_raw("b1", "released", owner="bob"))
    s.record_raw_rollout(
        _raw("b2", "released", owner="bob", created_at=_RECENT, finished_at=_RECENT)
    )
    s.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )
    s.close()

    data = _users_blocking(_cfg(state_db, runs_root))

    # total = 2 (alice lifetime) + 2 (bob: 1 lifetime + 1 live) = 4
    assert data["totals"]["total"] == 4
    assert data["totals"]["released"] == 4
    assert data["totals"]["success_pct"] == pytest.approx(100.0)
