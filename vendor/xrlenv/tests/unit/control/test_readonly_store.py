"""Read-only SqliteStateStore open (audit H2 follow-up).

A pure-read consumer (the CLI `nodes` command, the deploy liveness probe) must
NOT mutate the live DB. Critically it must never run `PRAGMA journal_mode`
(env-unset defaults to WAL and would flip a control plane's TRUNCATE DB back to
WAL, recreating the -shm mmap SIGBUS exposure on a network filesystem), never
create -wal/-shm sidecars, and never write. `SqliteStateStore(path,
read_only=True)` opens `file:...?mode=ro` and skips journal-mode/schema/stamp.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from xrlenv.control.state import SqliteStateStore

_ENV_KEY = "XRLENV_SQLITE_JOURNAL_MODE"


def _write_version(db: Path) -> int:
    """SQLite header byte 18: 1 = rollback journal (TRUNCATE/DELETE), 2 = WAL."""
    with open(db, "rb") as f:
        return f.read(20)[18]


def _make_truncate_db(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a DB in TRUNCATE mode (as the control plane does) and close it."""
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    SqliteStateStore(db).close()
    monkeypatch.delenv(_ENV_KEY, raising=False)


def test_read_only_open_does_not_flip_truncate_to_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)
    assert _write_version(db) == 1  # sanity: TRUNCATE

    # A read-only open with the env UNSET (a login-user CLI) must not flip it.
    store = SqliteStateStore(db, read_only=True)
    try:
        assert store.list_nodes() == []  # read works
    finally:
        store.close()
    assert _write_version(db) == 1, "read-only open flipped the journal mode"


def test_read_only_open_creates_no_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)
    SqliteStateStore(db, read_only=True).close()
    assert not (tmp_path / "s.db-wal").exists()
    assert not (tmp_path / "s.db-shm").exists()


def test_mutating_open_env_unset_would_flip_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the DEFAULT (read-write, env-unset) open DOES flip a
    TRUNCATE DB to WAL — which is exactly why read commands must use read_only."""
    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)
    SqliteStateStore(db).close()  # read-write, env unset -> WAL
    assert _write_version(db) == 2


def test_read_only_store_write_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)
    store = SqliteStateStore(db, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match=r"readonly|read-only"):
            store._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('x', 'y')"
            )
    finally:
        store.close()


def test_read_only_reads_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows written by a read-write store are visible to a later read-only open."""
    db = tmp_path / "s.db"
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    rw = SqliteStateStore(db)
    try:
        rw._conn.execute(
            "INSERT INTO raw_rollouts (rollout_id, status, image, owner_id, created_at) "
            "VALUES ('r0', 'released', 'img', 'alice', 1.0)"
        )
        rw._conn.commit()
    finally:
        rw.close()
    monkeypatch.delenv(_ENV_KEY, raising=False)

    ro = SqliteStateStore(db, read_only=True)
    try:
        got = ro.aggregate_raw_rollouts_by_owner_status()
    finally:
        ro.close()
    assert got == {"alice": {"released": 1}}
    assert _write_version(db) == 1  # still TRUNCATE


def test_read_only_missing_file_raises(tmp_path: Path) -> None:
    """A missing/unreadable DB raises on read-only open — the correct
    fail-closed signal for a liveness probe (not a silent empty result)."""
    with pytest.raises(sqlite3.OperationalError):
        SqliteStateStore(tmp_path / "does-not-exist.db", read_only=True)


@pytest.mark.parametrize("cmd_name", ["cmd_nodes", "cmd_rollouts", "cmd_events",
                                      "cmd_audit", "cmd_fairshare_show"])
def test_pure_read_cli_command_does_not_flip_journal_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cmd_name: str
) -> None:
    """COMMAND-LEVEL guard (audit H5): running a pure-read CLI command against a
    control plane's TRUNCATE DB with the env UNSET (a login-user invocation) must
    NOT flip it to WAL. `_open_state` defaults to read_only=True so a new reader
    can't inherit the mutating open."""
    import io

    import xrlenv.cli.commands as cli

    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)  # also clears the env
    getattr(cli, cmd_name)(state_db=db, out=io.StringIO())
    assert _write_version(db) == 1, f"{cmd_name} flipped the journal mode to WAL"


def test_write_cli_command_still_works_read_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: a WRITE command (db prune) opts into read_only=False and
    still succeeds (a read-only open would raise 'attempt to write a readonly
    database')."""
    import io

    import xrlenv.cli.commands as cli

    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)
    assert cli.cmd_db_prune(state_db=db, out=io.StringIO()) == 0
