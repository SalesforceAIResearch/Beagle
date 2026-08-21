"""Unit tests for audit "missing state-path coverage" gaps (GAP 1, 2, 4, 5).

GAP 1 — WAL→TRUNCATE recovery with real data + live -wal sidecar.
GAP 2 — failed journal conversion raises RuntimeError (held-connection scenario).
GAP 4 — concurrent GC vs /users: no double/under-count across a prune.
GAP 5 — cmd_build_status is a read-only command and must not flip journal mode.

See also:
  tests/unit/admin/test_missing_state_path_coverage.py  (GAP 3 — /users H4 banner)

Isolation
---------
All tests use ``monkeypatch.setenv`` / ``monkeypatch.delenv`` so
``XRLENV_SQLITE_JOURNAL_MODE`` never leaks between test functions.
"""

from __future__ import annotations

import io
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from xrlenv.control.state import (
    InMemoryStateStore,
    RawRolloutRecord,
    SqliteStateStore,
    StateStore,
)

_ENV_KEY = "XRLENV_SQLITE_JOURNAL_MODE"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (mirror helpers from test_readonly_store.py / test_journal_mode_override.py)
# ──────────────────────────────────────────────────────────────────────────────


def _write_version(db: Path) -> int:
    """SQLite header byte 18: 1 = rollback journal (TRUNCATE/DELETE), 2 = WAL."""
    with open(db, "rb") as f:
        return f.read(20)[18]


def _raw(
    rollout_id: str | None = None,
    *,
    status: str = "released",
    owner: str = "alice",
    created_at: float | None = None,
    finished_at: float | None = None,
) -> RawRolloutRecord:
    rid = rollout_id or str(uuid.uuid4())
    ca = created_at if created_at is not None else time.time() - 200 * 86400
    return RawRolloutRecord(
        rollout_id=rid,
        status=status,  # type: ignore[arg-type]
        image="busybox:1",
        owner_id=owner,
        created_at=ca,
        finished_at=finished_at if finished_at is not None else ca,
    )


def _prune(store: StateStore) -> dict[str, int]:
    return store.prune_expired(
        now=time.time(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=14,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GAP 1 — WAL→TRUNCATE recovery with real data and live -wal sidecar
#
# Scenario: existing WAL-mode production DB with a real -wal file.
# Reopen with XRLENV_SQLITE_JOURNAL_MODE=TRUNCATE → SQLite converts in-place.
# Postconditions:
#   a) SQLite header write_version == 1  (rollback-journal marker)
#   b) No -wal or -shm sidecar files remain
#   c) Previously-written data is intact (read it back)
# ──────────────────────────────────────────────────────────────────────────────


def test_wal_to_truncate_recovery_header_write_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After WAL→TRUNCATE conversion the SQLite header write_version is 1
    (rollback-journal marker), confirming the mode switch is durable even if
    the process dies immediately after."""
    db = tmp_path / "s.db"

    # Step 1: create a WAL-mode DB and write a real row so there is a -wal.
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store1 = SqliteStateStore(db)
    store1.record_raw_rollout(_raw("wal-row-1"))
    # Close without checkpoint so -wal stays around (or may be checkpointed;
    # either way SQLite will handle it on the next open).
    store1.close()

    # Sanity: the DB should be in WAL mode.
    assert _write_version(db) == 2, "precondition: DB must be WAL before conversion"

    # Step 2: reopen with TRUNCATE.
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store2 = SqliteStateStore(db)
    store2.close()

    assert _write_version(db) == 1, (
        "WAL→TRUNCATE conversion must set SQLite header write_version to 1"
    )


def test_wal_to_truncate_recovery_no_sidecar_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After WAL→TRUNCATE conversion no -wal or -shm sidecar files remain."""
    db = tmp_path / "s.db"

    monkeypatch.delenv(_ENV_KEY, raising=False)
    store1 = SqliteStateStore(db)
    store1.record_raw_rollout(_raw("wal-sidecar-row"))
    store1.close()

    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store2 = SqliteStateStore(db)
    store2.close()

    assert not (tmp_path / "s.db-wal").exists(), "-wal must not exist after TRUNCATE conversion"
    assert not (tmp_path / "s.db-shm").exists(), "-shm must not exist after TRUNCATE conversion"


def test_wal_to_truncate_recovery_data_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Previously-written data survives the WAL→TRUNCATE conversion.

    This is the 'existing WAL prod DB converts cleanly to TRUNCATE' path.
    We write multiple rows across two tables (raw_rollouts + a connected node)
    to exercise that diverse live data is preserved, not just the first row.
    """
    db = tmp_path / "s.db"

    # Step 1: WAL mode — write raw_rollouts and a connected node record.
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store1 = SqliteStateStore(db)
    store1.record_raw_rollout(_raw("persistent-rollout-1"))
    store1.record_raw_rollout(_raw("persistent-rollout-2", status="failed"))
    # Use record_node_connected to seed a node row.
    store1.record_node_connected("node-001")
    store1.close()

    assert _write_version(db) == 2, "precondition: WAL DB"

    # Step 2: reopen with TRUNCATE — _apply_journal_mode runs BEFORE any DDL.
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store2 = SqliteStateStore(db)
    try:
        # Both raw_rollouts rows must survive the journal-mode conversion.
        fetched1 = store2.get_raw_rollout("persistent-rollout-1")
        assert fetched1 is not None, "raw_rollout row 1 lost after WAL→TRUNCATE"
        assert fetched1.rollout_id == "persistent-rollout-1"
        assert fetched1.owner_id == "alice"

        fetched2 = store2.get_raw_rollout("persistent-rollout-2")
        assert fetched2 is not None, "raw_rollout row 2 lost after WAL→TRUNCATE"
        assert fetched2.status == "failed"

        # The node record must also survive.
        nodes = store2.list_nodes()
        assert any(n.node_id == "node-001" for n in nodes), (
            "node record lost after WAL→TRUNCATE conversion"
        )

        # Confirm the journal mode actually took.
        got = store2._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert got == "truncate", f"Expected truncate, got {got!r}"

        # Total row count must be correct.
        assert store2.count_raw_rollouts() == 2
    finally:
        store2.close()

    # Header must be rollback-journal after close.
    assert _write_version(db) == 1


def test_wal_to_truncate_recovery_with_live_uncheckpointed_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The realest crash-leftover case: a WAL DB whose committed row is still
    sitting in an UNCHECKPOINTED ``-wal`` frame at conversion time.

    The other GAP-1 tests ``close()`` the source store first, which — on this
    SQLite build — checkpoints and removes the sidecars, so the subsequent
    TRUNCATE open sees a clean main DB and never actually exercises ``-wal``
    recovery. Here we force a faithful mid-flight crash image instead:

      1. disable ``wal_autocheckpoint`` so the committed row stays in ``-wal``,
         not the main DB;
      2. copy the main ``.db`` + the live ``.db-wal`` aside *while the frame is
         uncheckpointed* (a crashed process leaves exactly those two — SQLite
         rebuilds ``-shm`` from ``-wal`` on recovery, so we deliberately do NOT
         copy ``-shm``);
      3. open the copy with ``TRUNCATE``.

    If the WAL→TRUNCATE conversion discarded ``-wal`` instead of recovering it,
    the row would be lost — so a surviving row is proof of real recovery.
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)
    src = tmp_path / "src"
    src.mkdir()
    db = src / "s.db"

    store1 = SqliteStateStore(db)
    # Disable autocheckpoint so the committed row stays in -wal, not the main DB.
    store1._conn.execute("PRAGMA wal_autocheckpoint=0")
    store1.record_raw_rollout(_raw("uncheckpointed-row", owner="dave"))

    wal = Path(f"{db}-wal")
    assert wal.exists() and wal.stat().st_size > 0, (
        "precondition: the committed row must still be in a live -wal frame"
    )

    # Snapshot the crash image (main + -wal, NOT -shm) while the connection is
    # still open and the frame is uncheckpointed.
    dst = tmp_path / "dst"
    dst.mkdir()
    db2 = dst / "s.db"
    shutil.copy(db, db2)
    shutil.copy(wal, Path(f"{db2}-wal"))
    store1.close()  # original store — irrelevant to the copied crash image

    assert Path(f"{db2}-wal").stat().st_size > 0, "copied -wal must be non-empty"

    # Convert the copied crash image to TRUNCATE — recovery must apply the frame.
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store2 = SqliteStateStore(db2)
    try:
        row = store2.get_raw_rollout("uncheckpointed-row")
        assert row is not None, (
            "the uncheckpointed -wal frame was NOT recovered during the "
            "WAL→TRUNCATE conversion — the committed row was lost"
        )
        assert row.owner_id == "dave"
        got = store2._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert got == "truncate", f"Expected truncate, got {got!r}"
    finally:
        store2.close()

    assert _write_version(db2) == 1
    assert not Path(f"{db2}-wal").exists(), "-wal must be gone after TRUNCATE conversion"
    assert not Path(f"{db2}-shm").exists(), "-shm must be gone after TRUNCATE conversion"


# ──────────────────────────────────────────────────────────────────────────────
# GAP 2 — failed journal conversion raises RuntimeError
#
# The switch WAL↔rollback fails if another connection holds the DB open.
# Test: create a WAL-mode DB; open a first raw sqlite3.connect on it and keep
# it open; then attempt SqliteStateStore(db) with TRUNCATE and assert it raises
# RuntimeError (could not adopt the requested mode).
#
# If the real held-connection scenario doesn't fail on this SQLite build (some
# builds allow the switch), we fall back to forcing the failure by monkeypatching
# _apply_journal_mode to return the wrong mode.
# ──────────────────────────────────────────────────────────────────────────────


def test_failed_journal_conversion_raises_on_held_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holding a WAL-mode connection open and trying to switch to TRUNCATE via a
    second SqliteStateStore open raises an exception.

    GAP 2 finding: on this SQLite build (CPython C extension), a held connection
    causes `PRAGMA journal_mode=TRUNCATE` to raise sqlite3.OperationalError
    ("database is locked") before _apply_journal_mode's verification step.
    Either OperationalError or RuntimeError (the documented "could not adopt")
    is acceptable — the important contracts are:
      (a) an exception IS raised (not silent WAL fallback)
      (b) the connection IS closed (no FD leak)

    The forced-failure test below specifically exercises the RuntimeError branch
    of _apply_journal_mode by making SQLite lie about the mode it adopted.
    """
    db = tmp_path / "held.db"

    # Step 1: create a WAL-mode DB.
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store1 = SqliteStateStore(db)
    store1.record_raw_rollout(_raw("held-row"))
    store1.close()

    assert _write_version(db) == 2  # WAL

    # Step 2: open a raw connection that "holds" the WAL DB in WAL mode.
    held_conn = sqlite3.connect(str(db), check_same_thread=False)
    # Run a read to register as a reader.
    held_conn.execute("SELECT COUNT(*) FROM raw_rollouts").fetchone()

    try:
        monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
        # Either RuntimeError (mode mismatch detected by _apply_journal_mode)
        # or OperationalError ("database is locked" from the PRAGMA itself)
        # must be raised — NOT a silent success in an unexpected mode.
        with pytest.raises((RuntimeError, sqlite3.OperationalError)):
            store2 = SqliteStateStore(db)
            # If no exception, something unexpected happened — ensure we close cleanly.
            store2.close()
            pytest.fail(
                "Expected RuntimeError or OperationalError when attempting WAL→TRUNCATE "
                "conversion with a held connection, but SqliteStateStore constructed "
                "without error. The journal_mode postcondition should be checked."
            )
    finally:
        held_conn.close()


def test_failed_journal_conversion_raises_runtime_error_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the RuntimeError branch of _apply_journal_mode by making SQLite's
    PRAGMA journal_mode appear to return a different mode than requested.

    This exercises the "got = self._conn.execute(...).fetchone()[0]; if str(got).upper() != mode"
    check that raises RuntimeError("could not set SQLite journal_mode=...").

    Approach: subclass _TrackingConnection (from test_sqlite_conn_leak_on_bad_init.py)
    to override execute() and return a lying cursor for the journal_mode PRAGMA.
    sqlite3.Connection.execute is an immutable C extension type and cannot be monkeypatched
    directly — the wrapper approach is the only portable solution.

    Also asserts the connection is closed on failure (no FD leak).
    """
    from unittest.mock import patch

    from tests.unit.control.test_sqlite_conn_leak_on_bad_init import (
        _TrackingConnection,
    )

    class _LyingTrackingConnection(_TrackingConnection):
        """Wraps a real connection but lies about journal_mode PRAGMA results."""

        def execute(self, sql: str, *args, **kwargs):  # type: ignore[override]
            # Intercept the journal_mode PRAGMA and lie: say it's still 'wal'.
            if sql.strip().upper().startswith("PRAGMA JOURNAL_MODE="):
                class _FakeRow:
                    def __getitem__(self_inner, idx: int) -> str:
                        return "wal"

                class _FakeCursor:
                    def fetchone(self_inner):  # type: ignore[override]
                        return _FakeRow()

                return _FakeCursor()
            # All other SQL goes to the real connection.
            return self._real.execute(sql, *args, **kwargs)

    db = tmp_path / "forced.db"
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")

    # Build a lying wrapper manually (can't reuse _make_connect_wrapper's factory
    # because we need the _LyingTrackingConnection subclass).
    real_conn = sqlite3.connect(str(db), check_same_thread=False)
    real_conn.row_factory = sqlite3.Row
    lying_wrapper = _LyingTrackingConnection(real_conn)

    def _patched_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return lying_wrapper

    with (
        patch("xrlenv.control.state.sqlite3.connect", side_effect=_patched_connect),
        pytest.raises(RuntimeError, match="could not set SQLite journal_mode"),
    ):
        SqliteStateStore(db)

    # The failed init must close the connection (no FD leak).
    assert lying_wrapper.close_call_count == 1, (
        f"Expected close() called exactly once after RuntimeError from _apply_journal_mode; "
        f"got close_call_count={lying_wrapper.close_call_count}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# GAP 4 — concurrent GC vs /users: no double/under-count across a prune
#
# aggregate_raw_rollouts_all_time_by_owner_status() reads live + lifetime in ONE
# statement to avoid the race window. Add a test that frames the GC race:
#   1. Insert rollouts.
#   2. Snapshot all-time aggregate.
#   3. Run prune_expired (moves some rows live→lifetime).
#   4. Snapshot again.
#   5. Both snapshots must be EQUAL (a row moved live→lifetime is still counted
#      exactly once — never double- or under-counted).
#
# Both InMemory and SQLite backends.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["inmem", "sqlite"])
def both_backends(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StateStore]:
    if request.param == "inmem":
        s: StateStore = InMemoryStateStore()
        yield s
    else:
        s = SqliteStateStore(tmp_path / "gc_race.db")
        try:
            yield s
        finally:
            s.close()  # type: ignore[attr-defined]


def test_all_time_aggregate_equal_before_and_after_prune(
    both_backends: StateStore,
) -> None:
    """Snapshot all-time aggregate before prune == after prune.

    This proves that prune_expired (which moves rows from live raw_rollouts into
    owner_rollout_lifetime) does NOT change the all-time total: each row is counted
    exactly once in the all-time view regardless of which table it currently lives in.
    """
    store = both_backends
    _OLD = time.time() - 200 * 86400  # 200 days ago — expired
    _RECENT = time.time()              # now — unexpired

    # Insert a mix of expired terminal rows (will be pruned) and fresh rows (stay live).
    store.record_raw_rollout(_raw("exp-1", status="released", owner="alice", created_at=_OLD))
    store.record_raw_rollout(_raw("exp-2", status="failed", owner="alice", created_at=_OLD))
    store.record_raw_rollout(_raw("exp-3", status="released", owner="bob", created_at=_OLD))
    store.record_raw_rollout(_raw("fresh-1", status="released", owner="alice",
                                  created_at=_RECENT, finished_at=_RECENT))
    store.record_raw_rollout(_raw("fresh-2", status="cancelled", owner="bob",
                                  created_at=_RECENT, finished_at=_RECENT))

    # Snapshot 1: all-time aggregate BEFORE prune (all rows live).
    before = store.aggregate_raw_rollouts_all_time_by_owner_status()
    total_before = sum(
        sum(by_status.values()) for by_status in before.values()
    )

    # Prune: moves expired rows to owner_rollout_lifetime.
    counts = _prune(store)
    assert counts["raw_rollouts"] == 3, (
        f"Expected 3 expired rows pruned, got {counts['raw_rollouts']}"
    )

    # Snapshot 2: all-time aggregate AFTER prune (expired rows now in lifetime).
    after = store.aggregate_raw_rollouts_all_time_by_owner_status()
    total_after = sum(
        sum(by_status.values()) for by_status in after.values()
    )

    assert before == after, (
        f"All-time aggregate changed across prune — "
        f"possible double/under-count.\n"
        f"  Before: {before}\n"
        f"  After:  {after}"
    )
    assert total_before == total_after == 5, (
        f"Total row count must be 5 (all inserted rows); "
        f"before={total_before}, after={total_after}"
    )


def test_all_time_aggregate_no_double_count_per_owner_status(
    both_backends: StateStore,
) -> None:
    """Per-owner, per-status counts are invariant across prune — no double-count."""
    store = both_backends
    _OLD = time.time() - 200 * 86400
    _RECENT = time.time()

    # Alice: 2 expired released, 1 expired failed, 1 fresh released.
    store.record_raw_rollout(_raw("a1", status="released", owner="alice", created_at=_OLD))
    store.record_raw_rollout(_raw("a2", status="released", owner="alice", created_at=_OLD))
    store.record_raw_rollout(_raw("a3", status="failed", owner="alice", created_at=_OLD))
    store.record_raw_rollout(_raw("a4", status="released", owner="alice",
                                  created_at=_RECENT, finished_at=_RECENT))

    # Bob: 1 expired cancelled.
    store.record_raw_rollout(_raw("b1", status="cancelled", owner="bob", created_at=_OLD))

    before = store.aggregate_raw_rollouts_all_time_by_owner_status()

    _prune(store)

    after = store.aggregate_raw_rollouts_all_time_by_owner_status()

    # Exact per-owner buckets must be unchanged.
    assert after.get("alice", {}).get("released", 0) == 3, (
        f"alice/released must be 3 after prune; got {after.get('alice', {})}"
    )
    assert after.get("alice", {}).get("failed", 0) == 1, (
        f"alice/failed must be 1 after prune; got {after.get('alice', {})}"
    )
    assert after.get("bob", {}).get("cancelled", 0) == 1, (
        f"bob/cancelled must be 1 after prune; got {after.get('bob', {})}"
    )
    assert before == after, (
        f"Before/after mismatch — a row was double- or under-counted.\n"
        f"  Before: {before}\n"
        f"  After:  {after}"
    )


def test_all_time_aggregate_active_rows_not_pruned_not_moved(
    both_backends: StateStore,
) -> None:
    """Active (acquiring/running) rows are never pruned, so the all-time aggregate
    is unchanged — they stay live and are counted once both before and after prune."""
    store = both_backends
    _OLD = time.time() - 200 * 86400

    # An active row (old — but active rows are exempt from pruning).
    store.record_raw_rollout(_raw("run1", status="running", owner="carol",
                                  created_at=_OLD, finished_at=None))
    # An expired terminal row (will be pruned).
    store.record_raw_rollout(_raw("done1", status="released", owner="carol",
                                  created_at=_OLD))

    before = store.aggregate_raw_rollouts_all_time_by_owner_status()

    counts = _prune(store)
    # Only the terminal row should be pruned.
    assert counts["raw_rollouts"] == 1

    after = store.aggregate_raw_rollouts_all_time_by_owner_status()

    assert before == after, (
        f"All-time aggregate must be unchanged — active rows are not pruned.\n"
        f"  Before: {before}\n"
        f"  After:  {after}"
    )
    # The running row is still in the live table.
    assert isinstance(store, (InMemoryStateStore, SqliteStateStore))
    if hasattr(store, "get_raw_rollout"):
        assert store.get_raw_rollout("run1") is not None


def test_all_time_aggregate_stable_under_concurrent_prune(
    both_backends: StateStore,
) -> None:
    """A GENUINELY threaded reader hammering the all-time aggregate while the
    janitor prunes concurrently must never observe an over- or under-count.

    This is the race the sequential before/after tests above cannot catch. The
    janitor's ``accumulate-then-delete`` runs as one transaction on the store's
    shared connection: mid-transaction (lifetime row inserted, ``raw_rollouts``
    row not yet deleted) the row exists in BOTH tables. If the aggregate read
    the connection without ``self._lock`` it could see that intermediate state
    and over-count. ``batch_size=1`` makes the SQLite janitor commit one such
    transaction PER expired row, maximizing the number of mid-transaction
    windows the reader is exposed to; a ``threading.Barrier`` starts both sides
    together so the reads genuinely overlap the prune.

    The all-time total is invariant at every committed point (a row is counted
    in ``owner_rollout_lifetime`` the instant it leaves ``raw_rollouts``), so
    every one of the reader's samples must equal the constant EXPECTED.
    """
    store = both_backends
    _OLD = time.time() - 200 * 86400   # expired terminal → will be pruned
    _RECENT = time.time()              # fresh terminal → stays live

    for i in range(40):
        store.record_raw_rollout(
            _raw(f"old-{i}", status="released", owner="alice", created_at=_OLD)
        )
    for i in range(20):
        store.record_raw_rollout(
            _raw(f"new-{i}", status="released", owner="bob",
                 created_at=_RECENT, finished_at=_RECENT)
        )
    EXPECTED = 60  # 40 move live→lifetime, 20 stay live — all-time is constant 60

    totals: list[int] = []
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def reader() -> None:
        try:
            start.wait(timeout=10)
            for _ in range(200):
                agg = store.aggregate_raw_rollouts_all_time_by_owner_status()
                totals.append(sum(sum(v.values()) for v in agg.values()))
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=reader)
    t.start()
    try:
        start.wait(timeout=10)
        store.prune_expired(
            now=time.time(),
            audit_retention_days=None,
            events_retention_days=None,
            raw_rollout_retention_days=14,
            batch_size=1,  # one accumulate-then-delete txn per expired row
        )
    finally:
        t.join(15)

    assert not errors, f"reader thread raised under concurrent prune: {errors!r}"
    assert len(totals) == 200, f"reader did not complete its run: {len(totals)} samples"
    bad = sorted({x for x in totals if x != EXPECTED})
    assert not bad, (
        f"all-time total wobbled under concurrent prune (over/under-count): "
        f"saw {bad} across {len(totals)} reads, expected constant {EXPECTED}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# GAP 5 — cmd_build_status is a read-only command; must NOT flip journal mode
#
# Extends the pattern from test_readonly_store.py test_pure_read_cli_command_does_not_flip.
# cmd_build_status opens SqliteStateStore(state_db, read_only=True).
# With the env UNSET (login-user invocation against a TRUNCATE prod DB), the
# read-only open must not flip write_version to 2 (WAL).
# ──────────────────────────────────────────────────────────────────────────────


def _make_truncate_db(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a TRUNCATE-mode DB (as the control plane does) and close it."""
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    SqliteStateStore(db).close()
    monkeypatch.delenv(_ENV_KEY, raising=False)


def test_cmd_build_status_does_not_flip_journal_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_build_status (build plan read) must not flip a TRUNCATE DB to WAL.

    It opens SqliteStateStore(state_db, read_only=True), so a login-user
    invocation with XRLENV_SQLITE_JOURNAL_MODE unset cannot run PRAGMA journal_mode
    and cannot flip the mode.
    """
    import xrlenv.cli.commands as cli

    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)  # also clears the env

    assert _write_version(db) == 1, "precondition: TRUNCATE DB"

    ret = cli.cmd_build_status(plan_id=None, state_db=db, out=io.StringIO())
    # No build plans → returns 0 with "no build plans applied yet".
    assert ret == 0

    assert _write_version(db) == 1, (
        "cmd_build_status flipped the journal mode to WAL — "
        "it must open read_only=True and skip PRAGMA journal_mode"
    )


def test_cmd_build_status_no_flip_writes_expected_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_build_status prints 'no build plans applied yet' when the DB is empty
    and has no build plans — confirming the read-only open still works end-to-end."""
    import xrlenv.cli.commands as cli

    db = tmp_path / "s.db"
    _make_truncate_db(db, monkeypatch)

    buf = io.StringIO()
    ret = cli.cmd_build_status(plan_id=None, state_db=db, out=buf)
    assert ret == 0
    assert "no build plans applied yet" in buf.getvalue()


def test_cmd_build_status_missing_db_returns_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_build_status with a non-existent state.db prints an error and returns 2
    without attempting to open a SqliteStateStore (no journal flip possible)."""
    import xrlenv.cli.commands as cli

    monkeypatch.delenv(_ENV_KEY, raising=False)
    db = tmp_path / "does-not-exist.db"

    buf = io.StringIO()
    ret = cli.cmd_build_status(plan_id=None, state_db=db, out=buf)
    assert ret == 2
    assert "not found" in buf.getvalue() or "error" in buf.getvalue().lower()
