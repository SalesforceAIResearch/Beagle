"""Event-loop lag monitor (2026-08-21).

A tiny background task that sleeps a fixed interval and measures how much
*later* than expected it actually woke. On a healthy asyncio control plane the
lag is a few milliseconds; a large lag means the event-loop thread was blocked
— a synchronous-I/O stall (e.g. a network-filesystem hiccup on a loop-thread
write), a runaway CPU-bound section, or a GC pause. Such a stall is exactly
what let the heartbeat watchdog false-mark the whole fleet lost during the
2026-08-21 outage: the loop froze for ~31 min, no heartbeats were processed,
and on thaw every node looked stale at once.

This monitor's job is *detection*, not repair: it emits a loud, timestamped
WARNING/ERROR the instant a stall is observed so an operator (or an alerting
rule scraping the log / the admin overview) learns about a freeze in seconds
rather than hours. The stall-aware watchdog (``NodeRegistry``) independently
refuses to mass-evict after a blackout; this monitor makes the blackout
*visible*.

It also keeps a small in-memory record (``last_stall``, ``max_lag_s``,
``stall_count``) the admin overview surfaces, and calls an optional
``on_stall`` hook so callers can bump a metric.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 1.0
# Warn above this lag; escalate to ERROR above ``_ERROR_MULTIPLIER`` x it. The
# threshold is deliberately well above normal scheduler jitter so it only fires
# on a genuine stall, not a momentarily busy loop.
_DEFAULT_WARN_LAG_S = 1.0
_ERROR_MULTIPLIER = 5.0


@dataclass
class StallRecord:
    """The most recent observed stall (monotonic-based lag in seconds, plus a
    wall-clock stamp for operator display)."""

    lag_s: float
    at_wall: float


class LoopLagMonitor:
    """Background task that flags event-loop stalls. Same start/shutdown shape
    as the other control-plane watchdogs."""

    def __init__(
        self,
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
        warn_lag_s: float = _DEFAULT_WARN_LAG_S,
        on_stall: Callable[[float], None] | None = None,
    ) -> None:
        self._interval_s = interval_s
        self._warn_lag_s = warn_lag_s
        self._on_stall = on_stall
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Observability snapshot (read by the admin overview).
        self.stall_count: int = 0
        self.max_lag_s: float = 0.0
        self.last_stall: StallRecord | None = None

    async def start(self) -> None:
        """Schedule the monitor task. Idempotent."""
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name="loop-lag-monitor")

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                expected = time.monotonic() + self._interval_s
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._interval_s,
                    )
                if self._stop.is_set():
                    return
                # Lag = how much later than ``expected`` we actually resumed.
                # A healthy sleep overshoots by ms; a stall by seconds/minutes.
                lag = time.monotonic() - expected
                if lag > self._warn_lag_s:
                    self._record_stall(lag)
        except asyncio.CancelledError:
            return

    def _record_stall(self, lag: float) -> None:
        self.stall_count += 1
        self.max_lag_s = max(self.max_lag_s, lag)
        self.last_stall = StallRecord(lag_s=lag, at_wall=time.time())
        level = logging.ERROR if lag > self._warn_lag_s * _ERROR_MULTIPLIER else logging.WARNING
        LOGGER.log(
            level,
            "event-loop STALL: the loop was blocked for %.1fs (>%.1fs threshold) "
            "— synchronous I/O on the loop thread, a CPU-bound section, or a GC "
            "pause. While blocked, node heartbeats are NOT processed; a long "
            "stall can trip the heartbeat watchdog. Investigate blocking calls "
            "on the event loop.",
            lag,
            self._warn_lag_s,
        )
        if self._on_stall is not None:
            try:
                self._on_stall(lag)
            except Exception:
                LOGGER.exception("loop-lag monitor: on_stall hook raised")


__all__ = ["LoopLagMonitor", "StallRecord"]
