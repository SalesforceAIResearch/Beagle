"""IdleTtlWatcher (spec 02 §"Consumer-side heartbeat" + §"Idle TTL").

Catches abandoned rollouts: a consumer process that hangs after starting a
rollout but stops calling ``step`` / ``heartbeat`` will, after
``idle_ttl_s`` of silence, have its rollout reaped.

Design mirrors :class:`xrlenv.control.deadlines.DeadlineWatcher` — one
asyncio.Task per rollout. The difference is that the timer *resets* on
every touch (step, explicit heartbeat) rather than firing on a fixed wall
clock. We model that with an asyncio.Event the watcher waits on with a
timeout; touch sets-then-clears the event so the wait short-circuits and
restarts the next sleep.

Per spec 02 the reap action is to seal the rollout as
``status=truncated`` / ``reason=idle_ttl`` and destroy the sandbox
through the standard terminate path. The coordinator wires this via the
:class:`TruncationCallback`-shaped ``on_idle_ttl`` constructor argument.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

LOGGER = logging.getLogger(__name__)


IdleTtlCallback = Callable[[str, str], Awaitable[None]]
"""``async def on_idle_ttl(rollout_id, reason) -> None`` — invoked when a
rollout has been silent for longer than ``idle_ttl_s``. The reason is
always ``"idle_ttl"`` per spec 02; carried as a parameter so the
coordinator's ``_terminate(..., reason=reason)`` plumbing stays uniform
with the deadline watcher."""


class IdleTtlWatcher:
    """Per-rollout idle-TTL reaper.

    Lifecycle:
        watcher = IdleTtlWatcher(on_idle_ttl=callback)
        watcher.watch(rollout_id, idle_ttl_s=120)   # at start_rollout
        watcher.touch(rollout_id)                   # on step() and heartbeat()
        watcher.cancel(rollout_id)                  # on terminate
        await watcher.shutdown()                    # on runtime shutdown
    """

    def __init__(self, on_idle_ttl: IdleTtlCallback) -> None:
        self._on_idle_ttl = on_idle_ttl
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._wakeups: dict[str, asyncio.Event] = {}

    # ── Per-rollout API ──────────────────────────────────────────────────────

    def watch(self, rollout_id: str, idle_ttl_s: float) -> None:
        """Arm the reaper. Idempotent — second call for the same rollout
        keeps the original timer (touch the existing one if you want to
        reset it).
        """
        if rollout_id in self._tasks:
            return
        self._wakeups[rollout_id] = asyncio.Event()
        self._tasks[rollout_id] = asyncio.create_task(
            self._fire(rollout_id, idle_ttl_s),
            name=f"idle-ttl-{rollout_id[:8]}",
        )

    def touch(self, rollout_id: str) -> None:
        """Reset the timer. Called on every step + explicit heartbeat."""
        event = self._wakeups.get(rollout_id)
        if event is not None:
            event.set()

    def cancel(self, rollout_id: str) -> None:
        """Disarm — called when the rollout terminates by any path."""
        task = self._tasks.pop(rollout_id, None)
        if task is not None and not task.done():
            task.cancel()
        self._wakeups.pop(rollout_id, None)

    def has_watcher(self, rollout_id: str) -> bool:
        return rollout_id in self._tasks

    # ── Shutdown ─────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError):
                await t
        self._tasks.clear()
        self._wakeups.clear()

    # ── Internals ────────────────────────────────────────────────────────────

    async def _fire(self, rollout_id: str, idle_ttl_s: float) -> None:
        """Wait up to ``idle_ttl_s`` for a touch; if none arrives, reap.

        On touch we clear the wakeup and loop back into another wait. The
        cycle exits when (a) the watch is cancelled (CancelledError raised
        by ``cancel(rollout_id)``), or (b) the timeout fires with no
        intervening touch — in which case we invoke the reap callback.
        """
        try:
            while True:
                event = self._wakeups.get(rollout_id)
                if event is None:
                    return  # cancelled out from under us; exit cleanly
                try:
                    await asyncio.wait_for(event.wait(), timeout=idle_ttl_s)
                except TimeoutError:
                    break
                event.clear()
        except asyncio.CancelledError:
            return

        # Idle window expired. Drop the watcher's bookkeeping first so a
        # subsequent terminate path's ``cancel(rollout_id)`` is a clean
        # no-op rather than racing the in-flight callback.
        self._tasks.pop(rollout_id, None)
        self._wakeups.pop(rollout_id, None)
        try:
            await self._on_idle_ttl(rollout_id, "idle_ttl")
        except Exception:
            LOGGER.exception(
                "idle-ttl callback failed for rollout=%s", rollout_id
            )


__all__ = ["IdleTtlCallback", "IdleTtlWatcher"]
