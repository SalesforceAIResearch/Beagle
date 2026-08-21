"""Unit tests for WalCheckpointer (xrlenv/control/wal_checkpointer.py).

Covers:
- start/shutdown lifecycle: task created on start, cancelled on shutdown,
  no leaked tasks after shutdown.
- checkpoint_once delegates to the store's checkpoint_wal method and returns
  the (busy, log_frames, checkpointed) tuple.
- _consecutive_busy counter increments when busy>0, resets when busy==0.
- Warning is emitted at consecutive_busy==3 and above, not before.
- An exception raised by the store's checkpoint_wal is caught by _loop
  and does not kill the background task.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from xrlenv.control.wal_checkpointer import WalCheckpointer

# ──────────────────────────────────────────────────────────────────────────────
# Fake store
# ──────────────────────────────────────────────────────────────────────────────


class FakeStore:
    """Minimal fake that satisfies the _Checkpointable protocol."""

    def __init__(self, return_values: list[tuple[int, int, int]] | None = None) -> None:
        self._returns = iter(return_values or [(0, 0, 0)])
        self.calls: list[dict[str, Any]] = []

    def checkpoint_wal(self, *, truncate: bool = True) -> tuple[int, int, int]:
        self.calls.append({"truncate": truncate})
        return next(self._returns)

    def set_return(self, value: tuple[int, int, int]) -> None:
        self._returns = iter([value])

    def set_returns(self, values: list[tuple[int, int, int]]) -> None:
        self._returns = iter(values)


class RaisingStore:
    """Store whose checkpoint_wal always raises."""

    def checkpoint_wal(self, *, truncate: bool = True) -> tuple[int, int, int]:
        raise RuntimeError("db is on fire")


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_creates_task() -> None:
    store = FakeStore([(0, 0, 0)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    assert cp._task is None
    await cp.start()
    assert cp._task is not None
    assert not cp._task.done()
    await cp.shutdown()


@pytest.mark.asyncio
async def test_start_idempotent() -> None:
    """Calling start() twice must not create a second task."""
    store = FakeStore()
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    await cp.start()
    task_first = cp._task
    await cp.start()
    assert cp._task is task_first  # same object
    await cp.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancels_task_and_clears_it() -> None:
    store = FakeStore()
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    await cp.start()
    task = cp._task
    assert task is not None
    await cp.shutdown()
    # Task must be done and the handle cleared.
    assert task.done()
    assert cp._task is None


@pytest.mark.asyncio
async def test_shutdown_before_start_does_not_raise() -> None:
    """Shutdown on a never-started checkpointer must be a safe no-op."""
    store = FakeStore()
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    await cp.shutdown()  # should not raise


@pytest.mark.asyncio
async def test_shutdown_twice_does_not_raise() -> None:
    store = FakeStore()
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    await cp.start()
    await cp.shutdown()
    await cp.shutdown()  # second call is a no-op


# ──────────────────────────────────────────────────────────────────────────────
# checkpoint_once delegates to the store
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_once_returns_store_tuple() -> None:
    store = FakeStore([(1, 42, 0)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    result = await cp.checkpoint_once()
    assert result == (1, 42, 0)
    assert len(store.calls) == 1
    assert store.calls[0]["truncate"] is True


@pytest.mark.asyncio
async def test_checkpoint_once_clean_returns_correct_tuple() -> None:
    store = FakeStore([(0, 10, 10)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    result = await cp.checkpoint_once()
    assert result == (0, 10, 10)


# ──────────────────────────────────────────────────────────────────────────────
# consecutive_busy counter
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consecutive_busy_increments_on_busy() -> None:
    store = FakeStore([(1, 5, 0), (2, 8, 0), (1, 3, 0)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    await cp.checkpoint_once()
    assert cp._consecutive_busy == 1
    await cp.checkpoint_once()
    assert cp._consecutive_busy == 2
    await cp.checkpoint_once()
    assert cp._consecutive_busy == 3


@pytest.mark.asyncio
async def test_consecutive_busy_resets_on_clean() -> None:
    store = FakeStore([(1, 5, 0), (1, 5, 0), (0, 0, 5)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    await cp.checkpoint_once()
    await cp.checkpoint_once()
    assert cp._consecutive_busy == 2
    await cp.checkpoint_once()
    assert cp._consecutive_busy == 0


@pytest.mark.asyncio
async def test_consecutive_busy_stays_zero_when_always_clean() -> None:
    store = FakeStore([(0, 0, 10), (0, 0, 10)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    await cp.checkpoint_once()
    await cp.checkpoint_once()
    assert cp._consecutive_busy == 0


# ──────────────────────────────────────────────────────────────────────────────
# Warning threshold
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_warning_below_three_consecutive_busy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeStore([(1, 5, 0), (1, 5, 0)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    with caplog.at_level(logging.WARNING, logger="xrlenv.control.wal_checkpointer"):
        await cp.checkpoint_once()
        await cp.checkpoint_once()
    assert not any("consecutive" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_warning_at_three_consecutive_busy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeStore([(1, 5, 0), (1, 5, 0), (1, 10, 0)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    with caplog.at_level(logging.WARNING, logger="xrlenv.control.wal_checkpointer"):
        await cp.checkpoint_once()
        await cp.checkpoint_once()
        await cp.checkpoint_once()
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "3" in warns[0].message


@pytest.mark.asyncio
async def test_warning_continues_above_three(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Each tick above the threshold emits another warning.
    store = FakeStore([(1, 1, 0)] * 5)
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    with caplog.at_level(logging.WARNING, logger="xrlenv.control.wal_checkpointer"):
        for _ in range(5):
            await cp.checkpoint_once()
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    # Ticks 3, 4, 5 all cross the >=3 threshold.
    assert len(warns) == 3


@pytest.mark.asyncio
async def test_info_logged_on_recovery_after_busy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two busy ticks, then clean — should log info that WAL truncate recovered.
    store = FakeStore([(1, 5, 0), (1, 5, 0), (0, 0, 7)])
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    with caplog.at_level(logging.INFO, logger="xrlenv.control.wal_checkpointer"):
        await cp.checkpoint_once()
        await cp.checkpoint_once()
        await cp.checkpoint_once()
    info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("recovered" in r.message for r in info_msgs)


# ──────────────────────────────────────────────────────────────────────────────
# Exception safety: store exception must not kill the _loop
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_continues_after_store_exception() -> None:
    """Exceptions raised by checkpoint_wal must be caught and logged; the loop
    must keep running so a transient DB error doesn't kill the background task."""
    # Use a very short interval so the loop fires quickly.
    store = RaisingStore()
    cp = WalCheckpointer(state=store, interval_s=0.02)
    await cp.start()
    # Give the loop time to fire at least once and NOT crash.
    await asyncio.sleep(0.08)
    assert cp._task is not None
    assert not cp._task.done(), "task must still be alive after store exception"
    await cp.shutdown()


@pytest.mark.asyncio
async def test_checkpoint_once_propagates_exception_to_caller() -> None:
    """checkpoint_once itself propagates exceptions — only _loop swallows them."""
    store = RaisingStore()
    cp = WalCheckpointer(state=store, interval_s=9999.0)
    with pytest.raises(RuntimeError, match="db is on fire"):
        await cp.checkpoint_once()


@pytest.mark.asyncio
async def test_loop_exception_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = RaisingStore()
    cp = WalCheckpointer(state=store, interval_s=0.02)
    await cp.start()
    await asyncio.sleep(0.08)
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("checkpoint tick raised" in r.message for r in error_records)
    await cp.shutdown()
