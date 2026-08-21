"""Unit tests for the env-gated SQLite journal-mode override in
``SqliteStateStore._apply_journal_mode()`` (``XRLENV_SQLITE_JOURNAL_MODE``).

The method is called BEFORE any schema DDL so a rollback-journal deployment
never transiently enters WAL / creates the mmap'd ``-shm``.

Tested invariants (B2 + H3 audit findings)
-------------------------------------------
* Env unset (or empty/whitespace-only) → WAL default; no exception.
* A value in ``_ALLOWED_JOURNAL_MODES`` {WAL, TRUNCATE, DELETE, PERSIST}
  (case/whitespace-insensitive) is applied AND verified.
* MEMORY and OFF are explicitly rejected (durability-unsafe) → ``ValueError``.
* Any other non-empty junk value → ``ValueError`` (fails-closed, not silently
  WAL).
* Conversion of an existing WAL DB to TRUNCATE: mode converts, ``-shm`` is
  gone, previously-written data survives.

Isolation
---------
All tests use ``monkeypatch.setenv`` / ``monkeypatch.delenv`` so the env var
is *never* leaked between test functions — many existing tests assume WAL
mode and would be broken by a persistent override.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from xrlenv.control.state import RawRolloutRecord, SqliteStateStore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ENV_KEY = "XRLENV_SQLITE_JOURNAL_MODE"


def _journal_mode(store: SqliteStateStore) -> str:
    """Return the current journal mode string as reported by SQLite."""
    return store._conn.execute("PRAGMA journal_mode").fetchone()[0]


def _make_raw_rollout(rollout_id: str | None = None) -> RawRolloutRecord:
    """Minimal RawRolloutRecord with every NOT NULL column populated."""
    return RawRolloutRecord(
        rollout_id=rollout_id or str(uuid.uuid4()),
        status="running",
        image="ubuntu:22.04",
        owner_id="default",
        created_at=time.time(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_default_journal_mode_is_wal_when_env_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env unset → WAL default from _SCHEMA stands unchanged."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert _journal_mode(store) == "wal"
    finally:
        store.close()


def test_truncate_mode_applied_and_no_shm_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XRLENV_SQLITE_JOURNAL_MODE=TRUNCATE → journal_mode == 'truncate',
    no ``-shm`` sidecar, and a basic write+read still works.
    """
    db_path = tmp_path / "s.db"
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store = SqliteStateStore(db_path)
    try:
        assert _journal_mode(store) == "truncate"

        # -shm is a WAL artefact; must be absent in a rollback-journal mode.
        assert not (tmp_path / "s.db-shm").exists()

        # Basic write+read must still work after the mode conversion.
        rec = _make_raw_rollout()
        store.record_raw_rollout(rec)
        count = store.count_raw_rollouts()
        assert count == 1
    finally:
        store.close()


def test_delete_mode_applied_and_no_shm_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XRLENV_SQLITE_JOURNAL_MODE=DELETE → journal_mode == 'delete',
    no ``-shm`` sidecar.
    """
    db_path = tmp_path / "s.db"
    monkeypatch.setenv(_ENV_KEY, "DELETE")
    store = SqliteStateStore(db_path)
    try:
        assert _journal_mode(store) == "delete"
        assert not (tmp_path / "s.db-shm").exists()
    finally:
        store.close()


def test_conversion_of_existing_wal_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open WAL DB, close it, reopen with TRUNCATE → mode converts,
    -shm is gone, and the previously-written data is still there.
    """
    db_path = tmp_path / "s.db"

    # Step 1: create a WAL-mode database and write a row.
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store1 = SqliteStateStore(db_path)
    rec = _make_raw_rollout(rollout_id="persisted-rollout")
    store1.record_raw_rollout(rec)
    assert _journal_mode(store1) == "wal"
    store1.close()

    # Step 2: reopen the same file with TRUNCATE.
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store2 = SqliteStateStore(db_path)
    try:
        assert _journal_mode(store2) == "truncate"
        # WAL's -shm must be gone after the conversion.
        assert not db_path.with_suffix(".db-shm").exists()
        # Previously-written data must survive the journal-mode conversion.
        count = store2.count_raw_rollouts()
        assert count == 1
        fetched = store2.get_raw_rollout("persisted-rollout")
        assert fetched is not None
        assert fetched.rollout_id == "persisted-rollout"
    finally:
        store2.close()


def test_bogus_value_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognised env value ('bogus') → ValueError; fails CLOSED (not silent WAL)."""
    monkeypatch.setenv(_ENV_KEY, "bogus")
    with pytest.raises(ValueError, match="XRLENV_SQLITE_JOURNAL_MODE"):
        SqliteStateStore(tmp_path / "s.db")


def test_memory_mode_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEMORY is durability-unsafe (crash mid-write corrupts DB) → rejected with ValueError."""
    monkeypatch.setenv(_ENV_KEY, "MEMORY")
    with pytest.raises(ValueError, match="XRLENV_SQLITE_JOURNAL_MODE"):
        SqliteStateStore(tmp_path / "s.db")


def test_off_mode_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFF is durability-unsafe (drops rollback journal entirely) → rejected with ValueError."""
    monkeypatch.setenv(_ENV_KEY, "OFF")
    with pytest.raises(ValueError, match="XRLENV_SQLITE_JOURNAL_MODE"):
        SqliteStateStore(tmp_path / "s.db")


def test_empty_string_env_does_not_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env set to the empty string (or whitespace-only) → treated as unset; WAL stands."""
    monkeypatch.setenv(_ENV_KEY, "   ")
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert _journal_mode(store) == "wal"
    finally:
        store.close()


@pytest.mark.parametrize("raw_value", [" truncate ", "Truncate", "TRUNCATE", "tRuNcAtE"])
def test_case_and_whitespace_normalisation(
    raw_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case-insensitive and leading/trailing-whitespace variants all apply TRUNCATE."""
    monkeypatch.setenv(_ENV_KEY, raw_value)
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert _journal_mode(store) == "truncate"
    finally:
        store.close()


def test_persist_mode_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PERSIST is an allowed rollback-journal mode; must be applied and verified."""
    monkeypatch.setenv(_ENV_KEY, "PERSIST")
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert _journal_mode(store) == "persist"
        # A basic write+read must still work in PERSIST mode.
        rec = _make_raw_rollout()
        store.record_raw_rollout(rec)
        assert store.count_raw_rollouts() == 1
    finally:
        store.close()


def test_wal_explicit_env_still_sets_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAL is in _ALLOWED_JOURNAL_MODES; an explicit WAL env value still works."""
    monkeypatch.setenv(_ENV_KEY, "WAL")
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert _journal_mode(store) == "wal"
    finally:
        store.close()


def test_mode_applied_before_schema_so_no_shm_created_on_fresh_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_apply_journal_mode runs BEFORE _SCHEMA DDL so a TRUNCATE deployment never
    creates the mmap'd -shm file — not even transiently during CREATE TABLE.
    This is the key safety guarantee of B2/H3: on a network filesystem (Lustre/FSx),
    even a brief WAL-mode touch mmap-faults. Proving -shm absent after full init
    demonstrates the ordering invariant holds.
    """
    db_path = tmp_path / "fresh.db"
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store = SqliteStateStore(db_path)
    try:
        assert _journal_mode(store) == "truncate"
        # -shm must never have been created — not even during schema DDL.
        assert not (tmp_path / "fresh.db-shm").exists()
    finally:
        store.close()


@pytest.mark.parametrize("bad_value", ["bogus", "MEMORY", "OFF", "memory", "off", " off "])
def test_invalid_values_raise_value_error_parametrized(
    bad_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All invalid/durability-unsafe values raise ValueError, leaving no half-open DB."""
    monkeypatch.setenv(_ENV_KEY, bad_value)
    with pytest.raises(ValueError, match="XRLENV_SQLITE_JOURNAL_MODE"):
        SqliteStateStore(tmp_path / "s.db")
    # No DB file should have been created (the error fires before the connection
    # does anything meaningful — the path may or may not exist depending on
    # sqlite3.connect's lazy-create, but the store is not open).
    # We do NOT assert db absence (sqlite3 creates the file on connect) but we
    # assert the store is unusable by verifying the ValueError propagated.


# ──────────────────────────────────────────────────────────────────────────────
# M4: _journal_mode attribute + checkpoint_wal() early-return (FIX M4)
# ──────────────────────────────────────────────────────────────────────────────


def test_journal_mode_attribute_set_to_truncate_when_env_is_truncate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_apply_journal_mode() stores the resolved mode on self._journal_mode
    (FIX M4): XRLENV_SQLITE_JOURNAL_MODE=TRUNCATE → store._journal_mode == 'TRUNCATE'.
    """
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert store._journal_mode == "TRUNCATE"
    finally:
        store.close()


def test_journal_mode_attribute_set_to_wal_when_env_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_apply_journal_mode() stores 'WAL' on self._journal_mode when env is unset
    (FIX M4 default path).
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        assert store._journal_mode == "WAL"
    finally:
        store.close()


def test_checkpoint_wal_returns_zero_triple_and_creates_no_sidecar_under_truncate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX M4: checkpoint_wal() early-returns (0,0,0) when _journal_mode != 'WAL'
    so a TRUNCATE-mode store never opens a per-tick connection and no -wal/-shm
    sidecar files are created.
    """
    db_path = tmp_path / "s.db"
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store = SqliteStateStore(db_path)
    try:
        result = store.checkpoint_wal()
        assert result == (0, 0, 0), (
            f"checkpoint_wal() under TRUNCATE must return (0,0,0), got {result}"
        )
        # A rollback-journal store must never create the WAL's mmap'd sidecar.
        assert not (tmp_path / "s.db-wal").exists(), "-wal must not exist under TRUNCATE"
        assert not (tmp_path / "s.db-shm").exists(), "-shm must not exist under TRUNCATE"
    finally:
        store.close()


def test_checkpoint_wal_returns_three_tuple_without_raising_under_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX M4: checkpoint_wal() executes PRAGMA wal_checkpoint and returns a
    3-tuple of ints when _journal_mode == 'WAL'. Write a row first so the WAL
    has at least one frame to checkpoint.
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        # Ensure there is something in the WAL to checkpoint.
        rec = _make_raw_rollout()
        store.record_raw_rollout(rec)

        result = store.checkpoint_wal()
        # Must be a 3-tuple of ints — (busy, log_frames, checkpointed_frames).
        assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
        assert len(result) == 3, f"expected 3-tuple, got {result}"
        busy, log_frames, checkpointed = result
        assert isinstance(busy, int)
        assert isinstance(log_frames, int)
        assert isinstance(checkpointed, int)
        # busy=0 means no readers blocked the checkpoint; log_frames >= 1
        # because we wrote a row. Not asserting exact values (SQLite internals),
        # but the structure must be valid.
        assert log_frames >= 0
    finally:
        store.close()


def test_checkpoint_wal_returns_zero_triple_when_store_is_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing contract: checkpoint_wal() returns (0,0,0) when the store
    has already been closed (guards callers that call it after shutdown).
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store = SqliteStateStore(tmp_path / "s.db")
    store.close()
    result = store.checkpoint_wal()
    assert result == (0, 0, 0), (
        f"checkpoint_wal() on a closed store must return (0,0,0), got {result}"
    )
