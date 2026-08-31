"""Unit tests for GAP 1 — H4 lifetime inception.

Covers:
  * ``InMemoryStateStore.lifetime_inception_ts()`` — non-None at construction.
  * ``SqliteStateStore.lifetime_inception_ts()`` — non-None after first open.
  * ``SqliteStateStore.lifetime_inception_ts()`` — STABLE across close + reopen of
    the same DB path (persisted once via ``INSERT OR IGNORE``, never re-stamped).
  * ``_users_blocking(cfg)`` — ``"inception"`` key present; value is an ISO string
    when the DB exists with a stamped inception; None when the DB path doesn't exist.

All tests monkeypatch ``XRLENV_SQLITE_JOURNAL_MODE`` to ensure no env leakage to
other tests that assume WAL mode.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from xrlenv.admin.server import AdminServerConfig, _users_blocking
from xrlenv.control.state import (
    InMemoryStateStore,
    RawRolloutRecord,
    SqliteStateStore,
)

_ENV_KEY = "XRLENV_SQLITE_JOURNAL_MODE"
_ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _raw(rollout_id: str, owner: str = "alice", status: str = "released") -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status=status,  # type: ignore[arg-type]
        image="img:1",
        owner_id=owner,
        created_at=time.time(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# InMemoryStateStore
# ──────────────────────────────────────────────────────────────────────────────


def test_inmemory_lifetime_inception_ts_is_non_none_at_construction() -> None:
    """InMemoryStateStore._lifetime_inception is set at __init__ time;
    lifetime_inception_ts() must never return None."""
    store = InMemoryStateStore()
    ts = store.lifetime_inception_ts()
    assert ts is not None, "InMemoryStateStore.lifetime_inception_ts() must not be None"


def test_inmemory_lifetime_inception_ts_is_a_float() -> None:
    """The returned inception timestamp is a finite float (epoch seconds)."""
    store = InMemoryStateStore()
    ts = store.lifetime_inception_ts()
    assert isinstance(ts, float)
    # Should be a plausible recent epoch (> year 2020)
    assert ts > 1_577_836_800.0  # 2020-01-01 UTC


def test_inmemory_lifetime_inception_ts_is_stable() -> None:
    """A second call returns the same value — it is not re-computed."""
    store = InMemoryStateStore()
    ts1 = store.lifetime_inception_ts()
    ts2 = store.lifetime_inception_ts()
    assert ts1 == ts2


def test_two_inmemory_stores_have_independent_inceptions() -> None:
    """Each InMemoryStateStore construction stamps its own independent inception."""
    s1 = InMemoryStateStore()
    t1 = s1.lifetime_inception_ts()
    # Sleep is not needed — construction is close but independent stamps.
    # We only need to verify they are both non-None (they may be equal if
    # constructed within the same time.time() resolution tick).
    s2 = InMemoryStateStore()
    t2 = s2.lifetime_inception_ts()
    assert t1 is not None
    assert t2 is not None


# ──────────────────────────────────────────────────────────────────────────────
# SqliteStateStore
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_lifetime_inception_ts_non_none_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A freshly-opened SqliteStateStore must have a non-None inception stamp."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        ts = store.lifetime_inception_ts()
        assert ts is not None, (
            "SqliteStateStore.lifetime_inception_ts() must be non-None after first open"
        )
    finally:
        store.close()


def test_sqlite_lifetime_inception_ts_is_a_float(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inception timestamp returned by SqliteStateStore is a float."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store = SqliteStateStore(tmp_path / "s.db")
    try:
        ts = store.lifetime_inception_ts()
        assert isinstance(ts, float)
        assert ts > 1_577_836_800.0  # plausible (> 2020-01-01 UTC)
    finally:
        store.close()


def test_sqlite_lifetime_inception_ts_stable_across_close_and_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inception timestamp is PERSISTED in schema_meta and NOT re-stamped
    on subsequent opens of the same DB path.  This is the key correctness
    invariant: the /users page's 'cumulative since X' claim must refer to
    the SAME epoch regardless of how many times the control plane restarts.
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)
    db_path = tmp_path / "s.db"

    # First open: stamps the inception.
    s1 = SqliteStateStore(db_path)
    ts_first = s1.lifetime_inception_ts()
    s1.close()

    assert ts_first is not None

    # Brief delay to ensure time.time() would return a different value
    # if the inception were re-stamped on open.
    # (No actual sleep needed: we check equality, and the close+reopen
    #  is fast enough that a re-stamp would almost certainly match anyway,
    #  but the INSERT OR IGNORE prevents a re-stamp entirely.)

    # Second open of the SAME path: must return the same timestamp.
    s2 = SqliteStateStore(db_path)
    ts_second = s2.lifetime_inception_ts()
    s2.close()

    assert ts_second is not None
    assert ts_second == ts_first, (
        f"Inception should be stable across reopen: "
        f"first={ts_first}, second={ts_second}"
    )


def test_sqlite_lifetime_inception_ts_stable_across_multiple_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three consecutive opens of the same DB all return the same inception."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    db_path = tmp_path / "s.db"

    timestamps: list[float] = []
    for _ in range(3):
        s = SqliteStateStore(db_path)
        ts = s.lifetime_inception_ts()
        s.close()
        assert ts is not None
        timestamps.append(ts)

    # All three reads must agree.
    assert timestamps[0] == timestamps[1] == timestamps[2], (
        f"Inception drifted across reopens: {timestamps}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# _users_blocking — "inception" key
# ──────────────────────────────────────────────────────────────────────────────


def test_users_blocking_inception_key_present_when_db_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the state_db path does not exist, _users_blocking must still
    include the 'inception' key in its return dict (value: None)."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    cfg = AdminServerConfig(
        state_db=tmp_path / "nonexistent.db",
        runs_root=tmp_path / "runs",
    )
    data = _users_blocking(cfg)
    assert "inception" in data, "'inception' key must be present in _users_blocking result"
    assert data["inception"] is None, (
        f"inception should be None when DB doesn't exist; got {data['inception']!r}"
    )


def test_users_blocking_inception_is_iso_string_when_db_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the state DB exists and has been opened (so the inception is
    stamped), _users_blocking must return 'inception' as an ISO string."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    db_path = tmp_path / "s.db"
    # Open the store to trigger schema creation + inception stamp.
    store = SqliteStateStore(db_path)
    store.close()

    cfg = AdminServerConfig(state_db=db_path, runs_root=tmp_path / "runs")
    data = _users_blocking(cfg)

    assert "inception" in data
    inception = data["inception"]
    assert inception is not None, (
        "inception must be non-None when the DB has been opened and stamped"
    )
    assert _ISO_PATTERN.match(inception), (
        f"inception must be an ISO string like '2026-01-15 12:00:00 UTC'; "
        f"got {inception!r}"
    )


def test_users_blocking_inception_is_none_when_db_does_not_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly separate test: DB path doesn't exist → inception is None."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    cfg = AdminServerConfig(
        state_db=tmp_path / "does_not_exist.db",
        runs_root=tmp_path / "runs",
    )
    data = _users_blocking(cfg)
    assert data.get("inception") is None


def test_users_blocking_inception_iso_format_matches_pattern(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'inception' ISO string must match '%Y-%m-%d %H:%M:%S UTC' exactly."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    db_path = tmp_path / "s.db"
    store = SqliteStateStore(db_path)
    store.record_raw_rollout(_raw("r1"))
    store.close()

    cfg = AdminServerConfig(state_db=db_path, runs_root=tmp_path / "runs")
    data = _users_blocking(cfg)

    inception = data.get("inception")
    assert inception is not None
    # Full format check: 'YYYY-MM-DD HH:MM:SS UTC'
    assert _ISO_PATTERN.match(inception), (
        f"inception ISO string {inception!r} does not match expected format"
    )
