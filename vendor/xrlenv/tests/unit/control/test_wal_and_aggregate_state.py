"""Unit tests for the WAL checkpoint method and new bounded aggregate methods
on SqliteStateStore, and for the raw_rollouts index regression.

Sections
--------
1. checkpoint_wal — basic tuple return, WAL file shrinks after TRUNCATE,
   calling after close() returns (0,0,0) and does not raise.
2. Bounded aggregate methods — count_raw_rollouts_by_status,
   count_raw_rollouts_finished_since, count_raw_rollouts_created_since,
   active_raw_node_ids, running_raw_counts_by_node, list_long_running_raw:
   each is seeded with a variety of rows spanning statuses, nodes, and
   timestamps; assertions cover correct counts/sets, boundary conditions
   (finished_at exactly at cutoff, node_id=None exclusion), and empty-store
   baseline.
3. Index regression — a fresh SqliteStateStore has all five expected indexes on
   raw_rollouts; opening over a hand-crafted *legacy* raw_rollouts table
   (missing owner_id/task_key/group_id) succeeds, backfills owner_id='default',
   and ends up with all five indexes. Style mirrors test_state_owner_id.py.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from xrlenv.control.state import RawRolloutRecord, SqliteStateStore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ALL_FIVE_INDEXES = {
    "raw_rollouts_owner_id_status_idx",
    "raw_rollouts_task_key_idx",
    "raw_rollouts_group_id_idx",
    "raw_rollouts_status_created_at_idx",
    "raw_rollouts_status_finished_at_idx",
}


def _raw(
    rollout_id: str,
    *,
    status: str = "released",
    node_id: str | None = "node-1",
    created_at: float | None = None,
    finished_at: float | None = None,
    owner_id: str = "default",
    task_key: str | None = None,
    group_id: str | None = None,
) -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status=status,  # type: ignore[arg-type]
        image="busybox:1",
        node_id=node_id,
        owner_id=owner_id,
        task_key=task_key,
        group_id=group_id,
        created_at=created_at if created_at is not None else time.time(),
        finished_at=finished_at,
    )


def _index_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("PRAGMA index_list('raw_rollouts')").fetchall()
        return {str(r[1]) for r in rows}
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# 1. checkpoint_wal
# ──────────────────────────────────────────────────────────────────────────────


def test_checkpoint_wal_returns_three_int_tuple(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        result = store.checkpoint_wal(truncate=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)
    finally:
        store.close()


def test_checkpoint_wal_passive_returns_three_int_tuple(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        result = store.checkpoint_wal(truncate=False)
        assert isinstance(result, tuple)
        assert len(result) == 3
    finally:
        store.close()


def test_checkpoint_wal_after_writes_does_not_raise(tmp_path: Path) -> None:
    """After some writes, a TRUNCATE checkpoint must succeed."""
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        for i in range(5):
            store.record_raw_rollout(
                _raw(f"r-{i}", status="running", node_id="node-1")
            )
        busy, log_frames, checkpointed = store.checkpoint_wal(truncate=True)
        # busy should be 0 (no other readers in this test process)
        assert busy == 0
        assert isinstance(log_frames, int)
        assert isinstance(checkpointed, int)
    finally:
        store.close()


def test_checkpoint_wal_truncate_does_not_raise_and_wal_stays_bounded(
    tmp_path: Path,
) -> None:
    """After many writes, a TRUNCATE checkpoint must:
    - not raise,
    - return busy==0 (no competing readers in this process),
    - leave the WAL file absent or at its smallest size (SQLite may
      auto-checkpoint before we call it, so we only assert bounded, not
      a specific bytes-checkpointed count).
    """
    db_path = tmp_path / "s.db"
    wal_path = tmp_path / "s.db-wal"
    store = SqliteStateStore(db_path)
    try:
        # Write enough rows to generate WAL frames.
        for i in range(50):
            store.record_raw_rollout(
                _raw(f"r-{i}", status="running", node_id="node-1")
            )

        busy, log_frames, checkpointed = store.checkpoint_wal(truncate=True)
        # No competing readers → busy must be 0.
        assert busy == 0
        # All three return values must be non-negative ints.
        assert log_frames >= 0
        assert checkpointed >= 0
        # WAL file, if it still exists, must be at its minimum header size
        # (32 bytes) or gone entirely — TRUNCATE resets it.
        wal_size_after = wal_path.stat().st_size if wal_path.exists() else 0
        # 32 bytes is the SQLite WAL header (written on creation, truncated to
        # exactly that on a successful TRUNCATE checkpoint). We allow 0 too
        # (some builds unlink the file).
        assert wal_size_after <= 32, (
            f"WAL not truncated; size after checkpoint: {wal_size_after} bytes"
        )
    finally:
        store.close()


def test_checkpoint_wal_after_close_returns_zeros(tmp_path: Path) -> None:
    """Calling checkpoint_wal after close() must return (0,0,0) and not raise."""
    store = SqliteStateStore(tmp_path / "s.db")
    store.close()
    result = store.checkpoint_wal(truncate=True)
    assert result == (0, 0, 0)


def test_checkpoint_wal_after_close_does_not_raise_on_passive(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    store.close()
    result = store.checkpoint_wal(truncate=False)
    assert result == (0, 0, 0)


def test_checkpoint_interleaved_with_writes_stays_consistent(tmp_path: Path) -> None:
    """checkpoint_wal runs on a DEDICATED connection, not the shared writer
    connection under the store lock (audit residual, 2026-07-16). So it must
    coexist with live writes on the shared connection: interleaving writes and
    TRUNCATE checkpoints must persist every row and keep the WAL bounded, and
    the store must stay writable throughout. This is the regression guard for
    'a large inherited WAL blocks heartbeat writers behind the checkpoint'."""
    db = tmp_path / "s.db"
    wal = db.with_name(db.name + "-wal")
    store = SqliteStateStore(db)
    try:
        now = time.time()
        for batch in range(5):
            for i in range(50):
                store.record_raw_rollout(
                    _raw(f"r{batch}-{i}", status="running", created_at=now)
                )
            # Checkpoint on the dedicated connection while the shared
            # connection is fully live — must not raise, must truncate cleanly
            # (busy==0 since no other connection pins the WAL here), and the
            # store must remain writable for the next batch.
            busy, _log, _ckpt = store.checkpoint_wal(truncate=True)
            assert busy == 0
            assert not wal.exists() or wal.stat().st_size <= 32
        # All 250 rows survived the interleaved checkpoints.
        assert store.count_raw_rollouts_by_status().get("running") == 250
        # Store is still writable after the final checkpoint.
        store.record_raw_rollout(_raw("final", status="running", created_at=now))
        assert store.get_raw_rollout("final") is not None
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2a. count_raw_rollouts_by_status
# ──────────────────────────────────────────────────────────────────────────────


def test_count_by_status_empty_store(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        result = store.count_raw_rollouts_by_status()
        assert result == {}
    finally:
        store.close()


def test_count_by_status_all_statuses(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(_raw("r-acq-1", status="acquiring", node_id=None))
        store.record_raw_rollout(_raw("r-acq-2", status="acquiring", node_id=None))
        store.record_raw_rollout(_raw("r-run-1", status="running", node_id="node-1"))
        store.record_raw_rollout(_raw("r-rel-1", status="released", node_id="node-1"))
        store.record_raw_rollout(_raw("r-rel-2", status="released", node_id="node-2"))
        store.record_raw_rollout(_raw("r-rel-3", status="released", node_id="node-1"))
        store.record_raw_rollout(_raw("r-fail-1", status="failed", node_id="node-1"))

        counts = store.count_raw_rollouts_by_status()
        assert counts["acquiring"] == 2
        assert counts["running"] == 1
        assert counts["released"] == 3
        assert counts["failed"] == 1
        assert "cancelled" not in counts
    finally:
        store.close()


def test_count_by_status_single_status(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(_raw("r-1", status="reaped"))
        counts = store.count_raw_rollouts_by_status()
        assert counts == {"reaped": 1}
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2b. count_raw_rollouts_finished_since
# ──────────────────────────────────────────────────────────────────────────────


def test_finished_since_empty_statuses_returns_zero(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(_raw("r-1", status="released", finished_at=time.time()))
        assert store.count_raw_rollouts_finished_since(0.0, []) == 0
    finally:
        store.close()


def test_finished_since_counts_within_window(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 1_000_000.0
        # Before cutoff — should NOT be counted.
        store.record_raw_rollout(
            _raw("r-old", status="released", finished_at=cutoff - 1.0)
        )
        # At cutoff (inclusive boundary) — should be counted.
        store.record_raw_rollout(
            _raw("r-at", status="released", finished_at=cutoff)
        )
        # After cutoff — should be counted.
        store.record_raw_rollout(
            _raw("r-after", status="failed", finished_at=cutoff + 100.0)
        )
        # Running (no finished_at) — should NOT be counted.
        store.record_raw_rollout(
            _raw("r-running", status="running", finished_at=None)
        )

        n = store.count_raw_rollouts_finished_since(cutoff, ["released", "failed"])
        assert n == 2
    finally:
        store.close()


def test_finished_since_filters_by_status(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 1_000_000.0
        store.record_raw_rollout(
            _raw("r-rel", status="released", finished_at=cutoff + 1.0)
        )
        store.record_raw_rollout(
            _raw("r-fail", status="failed", finished_at=cutoff + 1.0)
        )
        store.record_raw_rollout(
            _raw("r-canc", status="cancelled", finished_at=cutoff + 1.0)
        )

        # Only count "released" and "failed".
        n = store.count_raw_rollouts_finished_since(cutoff, ["released", "failed"])
        assert n == 2

        # Only count "cancelled".
        n = store.count_raw_rollouts_finished_since(cutoff, ["cancelled"])
        assert n == 1
    finally:
        store.close()


def test_finished_since_exactly_at_cutoff_is_included(tmp_path: Path) -> None:
    """finished_at == since_ts is >= so it must be counted."""
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 1_234_567.89
        store.record_raw_rollout(
            _raw("r-exact", status="released", finished_at=cutoff)
        )
        assert store.count_raw_rollouts_finished_since(cutoff, ["released"]) == 1
    finally:
        store.close()


def test_finished_since_just_before_cutoff_is_excluded(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 1_234_567.89
        store.record_raw_rollout(
            _raw("r-before", status="released", finished_at=cutoff - 0.001)
        )
        assert store.count_raw_rollouts_finished_since(cutoff, ["released"]) == 0
    finally:
        store.close()


def test_finished_since_empty_table(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert store.count_raw_rollouts_finished_since(0.0, ["released"]) == 0
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2c. count_raw_rollouts_created_since
# ──────────────────────────────────────────────────────────────────────────────


def test_created_since_empty_table(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert store.count_raw_rollouts_created_since(0.0) == 0
    finally:
        store.close()


def test_created_since_counts_all_statuses(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 2_000_000.0
        store.record_raw_rollout(_raw("r-old", created_at=cutoff - 10.0))
        store.record_raw_rollout(_raw("r-at", created_at=cutoff))
        store.record_raw_rollout(_raw("r-new1", status="running", created_at=cutoff + 1.0))
        store.record_raw_rollout(_raw("r-new2", status="failed", created_at=cutoff + 5.0))

        assert store.count_raw_rollouts_created_since(cutoff) == 3
    finally:
        store.close()


def test_created_since_boundary_inclusive(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 9_000_000.0
        store.record_raw_rollout(_raw("r-exact", created_at=cutoff))
        assert store.count_raw_rollouts_created_since(cutoff) == 1
    finally:
        store.close()


def test_created_since_none_in_window(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(_raw("r-old", created_at=1000.0))
        assert store.count_raw_rollouts_created_since(9_999_999.0) == 0
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2d. active_raw_node_ids
# ──────────────────────────────────────────────────────────────────────────────


def test_active_raw_node_ids_empty(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert store.active_raw_node_ids() == set()
    finally:
        store.close()


def test_active_raw_node_ids_only_active_statuses(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        # acquiring and running are active.
        store.record_raw_rollout(_raw("r-acq", status="acquiring", node_id="node-A"))
        store.record_raw_rollout(_raw("r-run", status="running", node_id="node-B"))
        # Terminal — must NOT appear.
        store.record_raw_rollout(_raw("r-rel", status="released", node_id="node-C"))
        store.record_raw_rollout(_raw("r-fail", status="failed", node_id="node-D"))

        ids = store.active_raw_node_ids()
        assert ids == {"node-A", "node-B"}
    finally:
        store.close()


def test_active_raw_node_ids_excludes_none_node(tmp_path: Path) -> None:
    """Rows with node_id=None (still in acquiring with no scheduler decision)
    must NOT appear in the active-node set."""
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(_raw("r-acq-no-node", status="acquiring", node_id=None))
        store.record_raw_rollout(_raw("r-run", status="running", node_id="node-X"))

        ids = store.active_raw_node_ids()
        assert ids == {"node-X"}
    finally:
        store.close()


def test_active_raw_node_ids_deduplicates(tmp_path: Path) -> None:
    """Multiple rows on the same node should not duplicate it in the set."""
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(_raw("r-1", status="running", node_id="node-A"))
        store.record_raw_rollout(_raw("r-2", status="running", node_id="node-A"))
        store.record_raw_rollout(_raw("r-3", status="acquiring", node_id="node-A"))

        ids = store.active_raw_node_ids()
        assert ids == {"node-A"}
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2e. running_raw_counts_by_node
# ──────────────────────────────────────────────────────────────────────────────


def test_running_raw_counts_by_node_empty(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert store.running_raw_counts_by_node() == {}
    finally:
        store.close()


def test_running_raw_counts_by_node_only_running(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        # Only "running" rows count; "acquiring" does not.
        store.record_raw_rollout(_raw("r-run-a1", status="running", node_id="node-A"))
        store.record_raw_rollout(_raw("r-run-a2", status="running", node_id="node-A"))
        store.record_raw_rollout(_raw("r-run-b1", status="running", node_id="node-B"))
        store.record_raw_rollout(_raw("r-acq-a", status="acquiring", node_id="node-A"))
        store.record_raw_rollout(_raw("r-rel", status="released", node_id="node-A"))

        counts = store.running_raw_counts_by_node()
        assert counts == {"node-A": 2, "node-B": 1}
    finally:
        store.close()


def test_running_raw_counts_by_node_excludes_none_node(tmp_path: Path) -> None:
    """Rows with node_id=NULL are excluded (WHERE node_id IS NOT NULL)."""
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(_raw("r-run-null", status="running", node_id=None))
        store.record_raw_rollout(_raw("r-run-x", status="running", node_id="node-X"))

        counts = store.running_raw_counts_by_node()
        assert counts == {"node-X": 1}
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2f. list_long_running_raw
# ──────────────────────────────────────────────────────────────────────────────


def test_list_long_running_empty(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert store.list_long_running_raw(older_than_ts=time.time()) == []
    finally:
        store.close()


def test_list_long_running_returns_only_active_and_old(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 3_000_000.0

        # Old + acquiring → included.
        r_acq_old = _raw("r-acq-old", status="acquiring", node_id=None, created_at=cutoff - 100.0)
        # Old + running → included.
        r_run_old = _raw("r-run-old", status="running", node_id="node-A", created_at=cutoff - 50.0)
        # New + running → excluded (created_at >= cutoff).
        r_run_new = _raw("r-run-new", status="running", node_id="node-B", created_at=cutoff + 1.0)
        # Old + released → excluded (terminal).
        r_rel_old = _raw("r-rel-old", status="released", node_id="node-A", created_at=cutoff - 200.0)

        for r in [r_acq_old, r_run_old, r_run_new, r_rel_old]:
            store.record_raw_rollout(r)

        rows = store.list_long_running_raw(older_than_ts=cutoff)
        ids = {r.rollout_id for r in rows}
        assert ids == {"r-acq-old", "r-run-old"}
    finally:
        store.close()


def test_list_long_running_older_than_is_exclusive(tmp_path: Path) -> None:
    """created_at < older_than_ts is strict; a row created exactly at the
    cutoff is not included."""
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 5_000_000.0
        store.record_raw_rollout(
            _raw("r-exact", status="running", node_id="node-A", created_at=cutoff)
        )
        store.record_raw_rollout(
            _raw("r-before", status="running", node_id="node-A", created_at=cutoff - 1.0)
        )
        rows = store.list_long_running_raw(older_than_ts=cutoff)
        assert {r.rollout_id for r in rows} == {"r-before"}
    finally:
        store.close()


def test_list_long_running_returns_rawrolloutrecord_instances(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        store.record_raw_rollout(
            _raw("r-1", status="running", node_id="node-A", created_at=1.0)
        )
        rows = store.list_long_running_raw(older_than_ts=time.time() + 1e9)
        assert len(rows) == 1
        assert isinstance(rows[0], RawRolloutRecord)
    finally:
        store.close()


def test_list_long_running_ordered_by_created_at(tmp_path: Path) -> None:
    """Results must be oldest-first (ORDER BY created_at)."""
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        cutoff = 1e9
        store.record_raw_rollout(_raw("r-new", status="running", node_id="n", created_at=cutoff - 1.0))
        store.record_raw_rollout(_raw("r-old", status="running", node_id="n", created_at=cutoff - 100.0))

        rows = store.list_long_running_raw(older_than_ts=cutoff)
        assert rows[0].rollout_id == "r-old"
        assert rows[1].rollout_id == "r-new"
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Index presence — fresh DB and legacy DB
# ──────────────────────────────────────────────────────────────────────────────


def test_fresh_db_has_all_five_indexes(tmp_path: Path) -> None:
    """A newly created SqliteStateStore must have all five indexes on raw_rollouts."""
    db = tmp_path / "fresh.db"
    store = SqliteStateStore(db)
    store.close()

    names = _index_names(db)
    for idx in _ALL_FIVE_INDEXES:
        assert idx in names, f"missing index: {idx} (found: {names})"


def test_legacy_db_migration_adds_all_five_indexes(tmp_path: Path) -> None:
    """Opening a SqliteStateStore over a legacy raw_rollouts table (without
    owner_id/task_key/group_id columns, plus one pre-existing row) must:
    - not raise
    - backfill owner_id='default' on the legacy row
    - end up with all five expected indexes on raw_rollouts
    """
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    try:
        # Legacy raw_rollouts: no owner_id, task_key, group_id.
        conn.execute(
            """
            CREATE TABLE raw_rollouts (
                rollout_id      TEXT PRIMARY KEY,
                status          TEXT NOT NULL,
                image           TEXT NOT NULL,
                node_id         TEXT,
                container_id    TEXT,
                container_name  TEXT,
                artifact_path   TEXT,
                displayed_name  TEXT,
                created_at      REAL NOT NULL,
                finished_at     REAL,
                error           TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO raw_rollouts (rollout_id, status, image, created_at) "
            "VALUES ('r-legacy', 'released', 'busybox:1', ?)",
            (time.time(),),
        )
        conn.commit()
    finally:
        conn.close()

    # Opening through SqliteStateStore runs _migrate() which must handle this.
    store = SqliteStateStore(db)
    try:
        got = store.get_raw_rollout("r-legacy")
        assert got is not None, "legacy row must survive migration"
        assert got.owner_id == "default", (
            f"owner_id was not backfilled; got {got.owner_id!r}"
        )
    finally:
        store.close()

    names = _index_names(db)
    for idx in _ALL_FIVE_INDEXES:
        assert idx in names, f"missing index after migration: {idx} (found: {names})"


def test_legacy_db_with_task_key_group_id_but_no_owner_id(tmp_path: Path) -> None:
    """A DB that has task_key and group_id but lacks owner_id (intermediate
    migration state) should still migrate cleanly and gain all five indexes."""
    db = tmp_path / "semi-legacy.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            CREATE TABLE raw_rollouts (
                rollout_id      TEXT PRIMARY KEY,
                status          TEXT NOT NULL,
                image           TEXT NOT NULL,
                node_id         TEXT,
                container_id    TEXT,
                container_name  TEXT,
                artifact_path   TEXT,
                displayed_name  TEXT,
                task_key        TEXT,
                group_id        TEXT,
                created_at      REAL NOT NULL,
                finished_at     REAL,
                error           TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO raw_rollouts "
            "(rollout_id, status, image, task_key, created_at) "
            "VALUES ('r-semi', 'running', 'img:1', 'tk-1', ?)",
            (time.time(),),
        )
        conn.commit()
    finally:
        conn.close()

    store = SqliteStateStore(db)
    try:
        got = store.get_raw_rollout("r-semi")
        assert got is not None
        assert got.task_key == "tk-1"
        assert got.owner_id == "default"
    finally:
        store.close()

    names = _index_names(db)
    for idx in _ALL_FIVE_INDEXES:
        assert idx in names, f"missing index: {idx} (found: {names})"


def test_fresh_db_reopens_with_all_indexes(tmp_path: Path) -> None:
    """Closing and reopening a fresh DB must retain all five indexes (no duplication
    or loss from repeated IF NOT EXISTS calls)."""
    db = tmp_path / "reopen.db"
    store = SqliteStateStore(db)
    store.close()

    # Reopen — _migrate() runs again, all IF NOT EXISTS guards are hit.
    store2 = SqliteStateStore(db)
    store2.close()

    names = _index_names(db)
    for idx in _ALL_FIVE_INDEXES:
        assert idx in names, f"missing index after reopen: {idx}"
