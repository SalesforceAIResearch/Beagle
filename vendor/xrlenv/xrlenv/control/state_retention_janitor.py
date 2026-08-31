"""State-store retention janitor (spec 20 Retention/GC matrix).

The control plane's ``state.db`` carries several append-only / terminal tables
that otherwise grow without bound — the spec-19 ``audit`` trail dominates (one
row per authenticated RPC; observed at ~96% of a multi-week ``state.db``), plus
the generic ``events`` log and terminal ``raw_rollouts`` metadata. Spec 20's
Retention/GC matrix prescribes per-table windows; this janitor enforces them.

It calls :meth:`StateStore.prune_expired` once at startup and once every 24 h
thereafter, hard-deleting rows past their retention window in batches. This is
a sibling to :class:`RunDirJanitor` (which prunes on-disk run-dir artifacts) —
same 24 h loop shape, different owner in the retention matrix.

Deletes free pages for REUSE (which bounds ``state.db`` growth) but do NOT
shrink the file on disk. Run ``VACUUM`` once (``xrlenv db vacuum``) to reclaim
already-accumulated space.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

from xrlenv.control.state import StateStore

LOGGER = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 24 * 60 * 60  # 24 h


class StateRetentionJanitor:
    """Background task that prunes expired ``state.db`` rows per the spec-20
    retention matrix. A ``None`` window disables retention for that table."""

    def __init__(
        self,
        state: StateStore,
        *,
        audit_retention_days: int | None = 30,
        events_retention_days: int | None = 14,
        raw_rollout_retention_days: int | None = 14,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ) -> None:
        for name, val in (
            ("audit_retention_days", audit_retention_days),
            ("events_retention_days", events_retention_days),
            ("raw_rollout_retention_days", raw_rollout_retention_days),
        ):
            if val is not None and val < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {val}")
        self._state = state
        self._audit_days = audit_retention_days
        self._events_days = events_retention_days
        self._raw_days = raw_rollout_retention_days
        self._interval_s = interval_s
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Schedule the janitor task. Idempotent."""
        if self._task is None:
            self._task = asyncio.create_task(
                self._loop(), name="state-retention-janitor"
            )

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def kick(self) -> None:
        """Wake the janitor so it sweeps now instead of waiting for the timer."""
        self._wakeup.set()

    async def sweep_once(self) -> dict[str, int]:
        """Run one prune synchronously; return ``{table: rows_deleted}``.

        Public for tests + operator flows. The prune runs in a worker thread
        (via ``asyncio.to_thread``) so a large first purge — potentially
        millions of audit rows — doesn't stall the event loop.
        """
        return await asyncio.to_thread(self._prune_blocking)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _prune_blocking(self) -> dict[str, int]:
        return self._state.prune_expired(
            now=time.time(),
            audit_retention_days=self._audit_days,
            events_retention_days=self._events_days,
            raw_rollout_retention_days=self._raw_days,
        )

    async def _loop(self) -> None:
        while True:
            try:
                pruned = await self.sweep_once()
            except Exception:
                LOGGER.exception(
                    "state-retention janitor: sweep failed; continuing on next tick"
                )
            else:
                total = sum(pruned.values())
                if total:
                    LOGGER.info(
                        "state-retention janitor: pruned %d expired row(s): %s",
                        total,
                        pruned,
                    )

            with suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._interval_s)
            self._wakeup.clear()


__all__ = ["StateRetentionJanitor"]
