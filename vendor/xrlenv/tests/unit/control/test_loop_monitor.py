"""Unit tests for xrlenv.control.loop_monitor.LoopLagMonitor (2026-08-21)."""

from __future__ import annotations

import asyncio
import time

import pytest
from xrlenv.control.loop_monitor import LoopLagMonitor, StallRecord

# ──────────────────────────────────────────────────────────────────────────────
# start / shutdown lifecycle
# ──────────────────────────────────────────────────────────────────────────────


async def test_start_creates_background_task() -> None:
    monitor = LoopLagMonitor(interval_s=10.0)
    await monitor.start()
    assert monitor._task is not None
    assert not monitor._task.done()
    await monitor.shutdown()


async def test_start_is_idempotent() -> None:
    """Calling start() twice does not create a second task."""
    monitor = LoopLagMonitor(interval_s=10.0)
    await monitor.start()
    task_first = monitor._task
    await monitor.start()
    assert monitor._task is task_first
    await monitor.shutdown()


async def test_shutdown_cancels_task_and_clears_reference() -> None:
    monitor = LoopLagMonitor(interval_s=10.0)
    await monitor.start()
    task = monitor._task
    assert task is not None
    await monitor.shutdown()
    assert monitor._task is None
    assert task.done()


async def test_shutdown_before_start_is_safe() -> None:
    """shutdown() without a prior start() must not raise."""
    monitor = LoopLagMonitor(interval_s=10.0)
    await monitor.shutdown()  # must not raise


async def test_double_shutdown_is_idempotent() -> None:
    monitor = LoopLagMonitor(interval_s=10.0)
    await monitor.start()
    await monitor.shutdown()
    await monitor.shutdown()  # second call must not raise


# ──────────────────────────────────────────────────────────────────────────────
# _record_stall: counter, max, last_stall, hook
# ──────────────────────────────────────────────────────────────────────────────


def test_record_stall_increments_stall_count() -> None:
    monitor = LoopLagMonitor(warn_lag_s=1.0)
    assert monitor.stall_count == 0
    monitor._record_stall(1.5)
    assert monitor.stall_count == 1
    monitor._record_stall(2.0)
    assert monitor.stall_count == 2


def test_record_stall_updates_max_lag_keeps_maximum() -> None:
    monitor = LoopLagMonitor(warn_lag_s=1.0)
    monitor._record_stall(2.5)
    assert monitor.max_lag_s == pytest.approx(2.5)
    monitor._record_stall(1.2)
    # max stays at 2.5, not replaced by the smaller value
    assert monitor.max_lag_s == pytest.approx(2.5)
    monitor._record_stall(5.0)
    assert monitor.max_lag_s == pytest.approx(5.0)


def test_record_stall_sets_last_stall() -> None:
    monitor = LoopLagMonitor(warn_lag_s=1.0)
    assert monitor.last_stall is None
    before = time.time()
    monitor._record_stall(3.7)
    after = time.time()
    assert monitor.last_stall is not None
    assert isinstance(monitor.last_stall, StallRecord)
    assert monitor.last_stall.lag_s == pytest.approx(3.7)
    assert before <= monitor.last_stall.at_wall <= after


def test_record_stall_invokes_on_stall_hook_with_lag() -> None:
    calls: list[float] = []
    monitor = LoopLagMonitor(warn_lag_s=1.0, on_stall=calls.append)
    monitor._record_stall(4.2)
    assert calls == [pytest.approx(4.2)]


def test_record_stall_multiple_calls_accumulate() -> None:
    calls: list[float] = []
    monitor = LoopLagMonitor(warn_lag_s=1.0, on_stall=calls.append)
    monitor._record_stall(1.1)
    monitor._record_stall(2.2)
    monitor._record_stall(3.3)
    assert len(calls) == 3
    assert monitor.stall_count == 3


def test_record_stall_raising_hook_does_not_propagate() -> None:
    """An on_stall hook that raises must be swallowed."""
    def _bad_hook(lag: float) -> None:
        raise RuntimeError("hook exploded")

    monitor = LoopLagMonitor(warn_lag_s=1.0, on_stall=_bad_hook)
    # Must not raise; stall is still recorded
    monitor._record_stall(2.0)
    assert monitor.stall_count == 1


def test_record_stall_no_hook_is_fine() -> None:
    monitor = LoopLagMonitor(warn_lag_s=1.0, on_stall=None)
    monitor._record_stall(1.5)  # must not raise
    assert monitor.stall_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# False-positive guard: healthy loop records ZERO stalls
# ──────────────────────────────────────────────────────────────────────────────


async def test_healthy_loop_records_no_stalls() -> None:
    """A short real run with no blocking records zero stalls.

    The warn threshold is 1.0 s (default), and the interval is 0.02 s.
    A healthy sleep overshoots by milliseconds, well under 1 s, so no
    stall should fire.
    """
    monitor = LoopLagMonitor(interval_s=0.02, warn_lag_s=1.0)
    await monitor.start()
    await asyncio.sleep(0.08)  # 4 intervals — no blocking
    await monitor.shutdown()
    assert monitor.stall_count == 0, (
        f"Expected zero stalls on a healthy loop; got stall_count="
        f"{monitor.stall_count}, last_stall={monitor.last_stall}"
    )
