"""Multi-user (Slice B) — ``owner_id`` filter + additive migration on the
StateStore.

Covers the server-authoritative tenant column that lands on both the gym/step
``rollouts`` table and the case-2/3 ``raw_rollouts`` table:

- ``list_raw_rollouts(owner_id=...)`` / ``count_raw_rollouts(owner_id=...)``
  scope to one tenant; ``owner_id=None`` returns all owners.
- ``list_rollouts_page(owner_id=...)`` scopes the gym/step page the same way.
- The additive ALTER-TABLE migration: a fresh db has the ``owner_id`` column
  on ``raw_rollouts``, and a default-owner record round-trips as ``"default"``.

All dbs live under ``tmp_path`` so the suite stays hermetic.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from xrlenv.control.state import (
    RawRolloutRecord,
    RolloutRecord,
    SqliteStateStore,
)
from xrlenv.types import RolloutStatus


def _raw(rollout_id: str, owner_id: str) -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status="released",
        image="busybox:1",
        displayed_name=f"inst-{rollout_id}",
        owner_id=owner_id,
        created_at=time.time(),
    )


# ── raw_rollouts owner filter ─────────────────────────────────────────────────


def test_list_raw_rollouts_scopes_to_owner(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    store.record_raw_rollout(_raw("r-alice-1", "alice"))
    store.record_raw_rollout(_raw("r-alice-2", "alice"))
    store.record_raw_rollout(_raw("r-bob-1", "bob"))

    alice_rows = store.list_raw_rollouts(owner_id="alice")
    assert {r.rollout_id for r in alice_rows} == {"r-alice-1", "r-alice-2"}
    assert all(r.owner_id == "alice" for r in alice_rows)

    bob_rows = store.list_raw_rollouts(owner_id="bob")
    assert {r.rollout_id for r in bob_rows} == {"r-bob-1"}


def test_list_raw_rollouts_none_owner_returns_all(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    store.record_raw_rollout(_raw("r-alice-1", "alice"))
    store.record_raw_rollout(_raw("r-bob-1", "bob"))

    # Default (no owner_id kwarg) and explicit None both mean "all tenants".
    assert {r.rollout_id for r in store.list_raw_rollouts()} == {
        "r-alice-1", "r-bob-1",
    }
    assert {r.rollout_id for r in store.list_raw_rollouts(owner_id=None)} == {
        "r-alice-1", "r-bob-1",
    }


def test_count_raw_rollouts_scopes_to_owner(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    store.record_raw_rollout(_raw("r-alice-1", "alice"))
    store.record_raw_rollout(_raw("r-alice-2", "alice"))
    store.record_raw_rollout(_raw("r-bob-1", "bob"))

    assert store.count_raw_rollouts(owner_id="bob") == 1
    assert store.count_raw_rollouts(owner_id="alice") == 2
    assert store.count_raw_rollouts() == 3
    assert store.count_raw_rollouts(owner_id="nobody") == 0


# ── gym/step rollouts owner filter ────────────────────────────────────────────


def _gym(rollout_id: str, owner_id: str) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=rollout_id,
        template="hello-shell",
        status=RolloutStatus.FINISHED,
        owner_id=owner_id,
    )


def test_list_rollouts_page_scopes_to_owner(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    store.insert_rollout(_gym("g-alice-1", "alice"))
    store.insert_rollout(_gym("g-alice-2", "alice"))
    store.insert_rollout(_gym("g-bob-1", "bob"))

    alice_rows, has_next = store.list_rollouts_page(owner_id="alice", limit=10)
    assert {r.rollout_id for r in alice_rows} == {"g-alice-1", "g-alice-2"}
    assert has_next is False

    bob_rows, _ = store.list_rollouts_page(owner_id="bob", limit=10)
    assert {r.rollout_id for r in bob_rows} == {"g-bob-1"}


def test_list_rollouts_page_none_owner_returns_all(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    store.insert_rollout(_gym("g-alice-1", "alice"))
    store.insert_rollout(_gym("g-bob-1", "bob"))

    rows, _ = store.list_rollouts_page(owner_id=None, limit=10)
    assert {r.rollout_id for r in rows} == {"g-alice-1", "g-bob-1"}


# ── additive migration + default round-trip ───────────────────────────────────


def test_fresh_db_has_owner_id_column_on_raw_rollouts(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    SqliteStateStore(db)  # creates schema

    conn = sqlite3.connect(db)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(raw_rollouts)")}
    finally:
        conn.close()
    assert "owner_id" in cols


def test_default_owner_round_trips_as_default(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "s.db")
    # RawRolloutRecord defaults owner_id to "default" when unset.
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-default",
        status="released",
        image="busybox:1",
        created_at=time.time(),
    ))
    got = store.get_raw_rollout("r-default")
    assert got is not None
    assert got.owner_id == "default"
    # And it shows up under the explicit "default" tenant filter.
    assert store.count_raw_rollouts(owner_id="default") == 1


def test_additive_migration_backfills_owner_on_legacy_db(tmp_path: Path) -> None:
    """A db created before the owner_id column existed must gain it via the
    additive ALTER TABLE, with pre-existing rows defaulting to ``"default"``.

    We simulate the legacy shape by creating a raw_rollouts table WITHOUT
    owner_id, inserting a row, then reopening through SqliteStateStore (which
    runs the migration on connect).
    """
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    try:
        # Pre-owner_id raw_rollouts shape: the current CREATE minus the
        # owner_id column the migration adds. (task_key/group_id/error are
        # already present by this slice's baseline.)
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
            "INSERT INTO raw_rollouts (rollout_id, status, image, created_at) "
            "VALUES ('r-legacy', 'released', 'busybox:1', ?)",
            (time.time(),),
        )
        conn.commit()
    finally:
        conn.close()

    # Reopen through the store — migration runs and adds owner_id.
    store = SqliteStateStore(db)
    got = store.get_raw_rollout("r-legacy")
    assert got is not None
    assert got.owner_id == "default"
    assert store.count_raw_rollouts(owner_id="default") == 1
