"""Unit tests for GAP 2 — M4 runtime WalCheckpointer gate.

The gate in ``build_distributed_runtime`` constructs a ``WalCheckpointer``
only when the state store is a ``SqliteStateStore`` AND its ``_journal_mode``
is ``"WAL"``.  Under any rollback-journal mode (e.g. ``TRUNCATE``), or when
using ``InMemoryStateStore``, ``wal_checkpointer`` must be ``None``.

Feasibility note
----------------
``build_distributed_runtime`` is a heavyweight async factory that spins up a
gRPC server, admin server (if requested), reconcilers, and an asyncio event
loop.  Calling it in a unit test requires:
  - a free port (``socket``-allocated),
  - ``tmp_path`` for ``runs_root`` and the state DB,
  - ``await runtime.shutdown()`` in a ``finally`` block to release the gRPC
    server socket.

That is practical here because there are already integration-level tests
(``tests/unit/control/test_distributed_runtime.py``) that do exactly this.
We follow the same pattern and assert the full-runtime observable
(``runtime.wal_checkpointer is None`` vs ``is not None``).

The tests deliberately pass ``gc_reconcile_interval_s=None`` and omit
``admin_port`` to keep startup lightweight (no GC tasks, no HTTP server).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from xrlenv.control.distributed_runtime import build_distributed_runtime
from xrlenv.control.state import InMemoryStateStore, SqliteStateStore

_ENV_KEY = "XRLENV_SQLITE_JOURNAL_MODE"


def _free_port() -> int:
    """Allocate a free TCP port without holding a listening socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ──────────────────────────────────────────────────────────────────────────────
# Narrow observable: _journal_mode attribute on the store
# (fast, no runtime startup needed)
# ──────────────────────────────────────────────────────────────────────────────


def test_sqlite_store_journal_mode_attribute_is_wal_when_env_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SqliteStateStore._journal_mode == 'WAL' when env var is unset.

    This is the attribute the WAL-checkpointer gate checks
    (``getattr(state, '_journal_mode', 'WAL') == 'WAL'``).
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)
    store = SqliteStateStore(tmp_path / "wal.db")
    try:
        assert getattr(store, "_journal_mode", None) == "WAL"
    finally:
        store.close()


def test_sqlite_store_journal_mode_attribute_is_truncate_when_env_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SqliteStateStore._journal_mode == 'TRUNCATE' when env var is TRUNCATE.

    The WAL-checkpointer gate uses this attribute to decide whether to skip
    scheduling a per-tick checkpoint.  Under TRUNCATE there is no -wal file,
    so ``wal_checkpointer`` must be ``None``.
    """
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store = SqliteStateStore(tmp_path / "truncate.db")
    try:
        assert getattr(store, "_journal_mode", None) == "TRUNCATE"
    finally:
        store.close()


def test_sqlite_store_checkpoint_wal_returns_zero_triple_under_truncate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """checkpoint_wal() on a TRUNCATE store returns (0, 0, 0) immediately.

    This is the observable that proved the early-return gate works (M4 fix).
    The full-runtime assertion for TRUNCATE (below) proves the WalCheckpointer
    is not even constructed; this test proves the store-level gate alone.
    """
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")
    store = SqliteStateStore(tmp_path / "truncate.db")
    try:
        result = store.checkpoint_wal()
        assert result == (0, 0, 0), (
            f"checkpoint_wal() on a TRUNCATE store must return (0,0,0); got {result}"
        )
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# Full-runtime assertion: wal_checkpointer is None under TRUNCATE
# ──────────────────────────────────────────────────────────────────────────────


async def test_build_distributed_runtime_wal_checkpointer_none_under_truncate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full runtime's wal_checkpointer must be None when the SqliteStateStore
    is opened in TRUNCATE mode (no WAL, no -wal file to checkpoint).

    Passes a real SqliteStateStore so the isinstance() + _journal_mode checks
    inside build_distributed_runtime are exercised against the real object.
    """
    monkeypatch.setenv(_ENV_KEY, "TRUNCATE")

    db_path = tmp_path / "state.db"
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Build a SqliteStateStore in TRUNCATE mode first, then pass it to the
    # factory as ``state=`` so the factory doesn't create its own.
    state = SqliteStateStore(db_path)

    runtime = await build_distributed_runtime(
        grpc_port=_free_port(),
        runs_root=runs_root,
        state=state,
        state_db_path=db_path,
        gc_reconcile_interval_s=None,  # no background tasks
        run_dir_retention_days=None,
    )
    try:
        assert runtime.wal_checkpointer is None, (
            "wal_checkpointer must be None when the store is in TRUNCATE mode "
            "(no WAL to checkpoint)"
        )
    finally:
        await runtime.shutdown()


async def test_build_distributed_runtime_wal_checkpointer_not_none_under_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full runtime's wal_checkpointer must be non-None when the
    SqliteStateStore is in WAL mode (the default).
    """
    monkeypatch.delenv(_ENV_KEY, raising=False)

    db_path = tmp_path / "state.db"
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    state = SqliteStateStore(db_path)

    runtime = await build_distributed_runtime(
        grpc_port=_free_port(),
        runs_root=runs_root,
        state=state,
        state_db_path=db_path,
        gc_reconcile_interval_s=None,  # no background tasks
        run_dir_retention_days=None,
    )
    try:
        assert runtime.wal_checkpointer is not None, (
            "wal_checkpointer must be non-None when the store is in WAL mode"
        )
    finally:
        await runtime.shutdown()


async def test_build_distributed_runtime_wal_checkpointer_none_with_inmemory_store(
    tmp_path: Path,
) -> None:
    """The wal_checkpointer must be None when the state is InMemoryStateStore
    (not a SqliteStateStore at all — isinstance() guard fails).
    """
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    runtime = await build_distributed_runtime(
        grpc_port=_free_port(),
        runs_root=runs_root,
        state=InMemoryStateStore(),
        state_db_path=tmp_path / "state.db",
        gc_reconcile_interval_s=None,
        run_dir_retention_days=None,
    )
    try:
        assert runtime.wal_checkpointer is None, (
            "wal_checkpointer must be None when state is InMemoryStateStore"
        )
    finally:
        await runtime.shutdown()
