"""Deadline enforcement (spec 02 §"Deadline semantics").

One asyncio.Task per rollout. The watcher sleeps until ``hard_s``, then sets a
per-rollout truncate event the coordinator's :py:meth:`step` races against
``node.env_step()`` so an in-flight step raises ``RolloutTruncated`` rather
than waiting for the env to return. The watcher *also* invokes a coordinator
hook that destroys the sandbox (containers are killed at the runtime layer;
we never call ``env_teardown`` on truncation since the EnvAdapter's pinned
worker thread may be mid-step and can't be safely preempted — see
spec 14 / spec 02).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DeadlineWatcher — owns one asyncio.Task per rollout
# ──────────────────────────────────────────────────────────────────────────────


TruncationCallback = Callable[[str, str], Awaitable[None]]
"""``async def truncate(rollout_id, reason) -> None`` — invoked when hard_s fires."""


class DeadlineWatcher:
    """Per-rollout deadline timers.

    Use is symmetric: ``watch`` at start_rollout, ``cancel`` on terminate (the
    coordinator already runs every terminal-status path through ``_terminate``
    so cancellation always pairs with creation). The truncate event is
    exposed via :py:meth:`event_for` so the coordinator's step() can race it.
    """

    def __init__(self, on_truncate: TruncationCallback) -> None:
        self._on_truncate = on_truncate
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._events: dict[str, asyncio.Event] = {}

    # ── Per-rollout API ──────────────────────────────────────────────────────

    def watch(self, rollout_id: str, hard_s: float) -> None:
        """Start the watcher for ``rollout_id``. Idempotent."""
        if rollout_id in self._tasks:
            return
        self._events[rollout_id] = asyncio.Event()
        self._tasks[rollout_id] = asyncio.create_task(
            self._fire(rollout_id, hard_s),
            name=f"deadline-{rollout_id[:8]}",
        )

    def cancel(self, rollout_id: str) -> None:
        """Cancel the watcher (e.g. the rollout finished or was cancelled)."""
        task = self._tasks.pop(rollout_id, None)
        if task is not None and not task.done():
            task.cancel()
        self._events.pop(rollout_id, None)

    def event_for(self, rollout_id: str) -> asyncio.Event | None:
        """Return the per-rollout truncate event (or ``None`` if no watcher)."""
        return self._events.get(rollout_id)

    def has_watcher(self, rollout_id: str) -> bool:
        return rollout_id in self._tasks

    # ── Shutdown ─────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Cancel all in-flight watchers and await their teardown."""
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError):
                await t
        self._tasks.clear()
        self._events.clear()

    # ── Internals ────────────────────────────────────────────────────────────

    async def _fire(self, rollout_id: str, hard_s: float) -> None:
        try:
            await asyncio.sleep(hard_s)
        except asyncio.CancelledError:
            return
        # 1) Wake any in-flight step racing this event so it raises promptly.
        event = self._events.get(rollout_id)
        if event is not None:
            event.set()
        # 2) Drive the coordinator-side truncation (destroy sandbox + seal).
        try:
            await self._on_truncate(rollout_id, "hard_deadline")
        except Exception:
            LOGGER.exception(
                "deadline watcher: truncate callback failed for rollout=%s",
                rollout_id,
            )
        finally:
            # We pop here so cancel() called later from _terminate is a no-op.
            self._tasks.pop(rollout_id, None)
            self._events.pop(rollout_id, None)


__all__ = ["DeadlineWatcher", "TruncationCallback"]
