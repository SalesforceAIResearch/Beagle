"""Unit tests for GAP 3 — conn-leak on failed SqliteStateStore.__init__.

When ``XRLENV_SQLITE_JOURNAL_MODE`` is set to an invalid value (e.g. "bogus"),
``_apply_journal_mode()`` raises ``ValueError``.  The ``__init__`` wraps the
entire initialisation block in ``try / except BaseException: self._conn.close(); raise``
so the open sqlite3 connection is closed before the exception propagates.

This test file asserts that the connection is closed on the failed-init path
via two complementary approaches:

1. **Explicit spy** (preferred): monkeypatch ``sqlite3.connect`` to return a
   wrapper whose ``close()`` call is tracked; assert ``close()`` was called
   exactly once when construction raises ``ValueError``.

2. **ResourceWarning** (fallback cross-check): verify that no
   ``ResourceWarning`` is emitted by Python's GC after a failed construction,
   using ``warnings.catch_warnings`` + ``gc.collect()``.  This is a
   belt-and-suspenders check and is expected to pass whether or not the
   explicit-spy test is healthy.

All tests use ``monkeypatch`` so the env var never leaks.
"""

from __future__ import annotations

import contextlib
import gc
import sqlite3
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from xrlenv.control.state import SqliteStateStore

_ENV_KEY = "XRLENV_SQLITE_JOURNAL_MODE"


# ──────────────────────────────────────────────────────────────────────────────
# Spy wrapper around sqlite3.Connection
# ──────────────────────────────────────────────────────────────────────────────


class _TrackingConnection:
    """Thin wrapper around a real sqlite3 connection that tracks ``close()`` calls."""

    def __init__(self, real_conn: sqlite3.Connection) -> None:
        self._real = real_conn
        self.close_call_count = 0

    def close(self) -> None:
        self.close_call_count += 1
        self._real.close()

    # Proxy every other attribute through to the real connection so the
    # SqliteStateStore's __init__ can proceed normally up to the point where
    # _apply_journal_mode() raises.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_real", "close_call_count"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._real, name, value)


def _make_connect_wrapper(
    db_path: Path,
) -> tuple[_TrackingConnection, Any]:
    """Return (tracking_wrapper, patched_connect_fn).

    The patched connect returns the wrapper for any path; subsequent sqlite3
    calls inside SqliteStateStore.__init__ go through the real connection via
    the proxy.
    """
    real_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    real_conn.row_factory = sqlite3.Row
    wrapper = _TrackingConnection(real_conn)

    def _patched_connect(*args: Any, **kwargs: Any) -> _TrackingConnection:
        return wrapper

    return wrapper, _patched_connect


# ──────────────────────────────────────────────────────────────────────────────
# Approach 1: Explicit spy — assert close() called on failed init
# ──────────────────────────────────────────────────────────────────────────────


def test_failed_init_closes_connection_on_invalid_journal_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid XRLENV_SQLITE_JOURNAL_MODE raises ValueError; the __init__
    must call close() on the sqlite3 connection exactly once so no descriptor
    is leaked to GC.
    """
    monkeypatch.setenv(_ENV_KEY, "bogus_mode")

    db_path = tmp_path / "s.db"
    wrapper, patched_connect = _make_connect_wrapper(db_path)

    with (
        patch("xrlenv.control.state.sqlite3.connect", side_effect=patched_connect),
        pytest.raises(ValueError, match="XRLENV_SQLITE_JOURNAL_MODE"),
    ):
        SqliteStateStore(db_path)

    assert wrapper.close_call_count == 1, (
        f"Expected close() called exactly once after failed init; "
        f"got close_call_count={wrapper.close_call_count}"
    )


def test_failed_init_closes_connection_for_memory_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract for MEMORY mode (also durability-unsafe → ValueError)."""
    monkeypatch.setenv(_ENV_KEY, "MEMORY")

    db_path = tmp_path / "s.db"
    wrapper, patched_connect = _make_connect_wrapper(db_path)

    with (
        patch("xrlenv.control.state.sqlite3.connect", side_effect=patched_connect),
        pytest.raises(ValueError, match="XRLENV_SQLITE_JOURNAL_MODE"),
    ):
        SqliteStateStore(db_path)

    assert wrapper.close_call_count == 1, (
        f"close() must be called once for MEMORY-mode failure; "
        f"got {wrapper.close_call_count}"
    )


def test_failed_init_closes_connection_for_off_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract for OFF mode."""
    monkeypatch.setenv(_ENV_KEY, "OFF")

    db_path = tmp_path / "s.db"
    wrapper, patched_connect = _make_connect_wrapper(db_path)

    with (
        patch("xrlenv.control.state.sqlite3.connect", side_effect=patched_connect),
        pytest.raises(ValueError, match="XRLENV_SQLITE_JOURNAL_MODE"),
    ):
        SqliteStateStore(db_path)

    assert wrapper.close_call_count == 1, (
        f"close() must be called once for OFF-mode failure; "
        f"got {wrapper.close_call_count}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Approach 2: ResourceWarning — belt-and-suspenders cross-check
# ──────────────────────────────────────────────────────────────────────────────


def test_failed_init_does_not_leak_connection_resource_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python's GC emits a ResourceWarning when a file descriptor is garbage-
    collected without being explicitly closed.  After a failed SqliteStateStore
    construction, there must be no ResourceWarning — the __init__ must have
    closed the connection in its BaseException handler.

    This is a belt-and-suspenders check; the explicit-spy test above is
    more precise.
    """
    monkeypatch.setenv(_ENV_KEY, "bogus_resource_check")

    db_path = tmp_path / "s_rc.db"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        with contextlib.suppress(ValueError):
            SqliteStateStore(db_path)
        # Force GC to collect any un-closed objects.
        gc.collect()

    resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert len(resource_warnings) == 0, (
        f"ResourceWarning(s) emitted after failed init — connection leaked: "
        f"{[str(w.message) for w in resource_warnings]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Positive control: successful construction does NOT close connection on init
# ──────────────────────────────────────────────────────────────────────────────


def test_successful_init_does_not_close_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BaseException handler only fires on failure.  A successful init
    must NOT close the connection during __init__ — only close() should do that.
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)

    db_path = tmp_path / "ok.db"
    wrapper, patched_connect = _make_connect_wrapper(db_path)

    with patch("xrlenv.control.state.sqlite3.connect", side_effect=patched_connect):
        store = SqliteStateStore(db_path)

    try:
        # Connection must still be alive — no close() called during __init__.
        assert wrapper.close_call_count == 0, (
            f"close() must not be called during a successful __init__; "
            f"got {wrapper.close_call_count}"
        )
    finally:
        store.close()
