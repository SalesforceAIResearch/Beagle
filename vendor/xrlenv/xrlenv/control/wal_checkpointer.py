"""Periodic WAL checkpointer for the SQLite state store.

Only relevant when the state store is in WAL mode. Under a rollback-journal
mode (``XRLENV_SQLITE_JOURNAL_MODE=TRUNCATE`` on a network filesystem — spec 20)
there is no ``-wal`` to check-point, so ``build_distributed_runtime`` does not
even construct/schedule this task, and ``checkpoint_wal`` is a no-op there.

In WAL mode: SQLite's built-in auto-checkpoint is
*passive* and cannot reclaim WAL frames that an open reader still needs,
so a steady reader (the admin panel polls several DB-backed tabs)
alongside the control plane's steady write stream (per-heartbeat node
mirrors + rollout-lifecycle rows) lets the ``-wal`` file grow without
bound between process restarts — the only other time the WAL is
checkpointed. A 51 GiB WAL on the shared filesystem was observed to stall
the control-plane event loop badly enough that every node blew its
heartbeat grace at once and was mass-marked ``lost``.

This task forces a ``PRAGMA wal_checkpoint(TRUNCATE)`` on a fixed interval
from the store's writer connection, folding frames into the main DB and
resetting the ``-wal`` file, so the WAL stays bounded regardless of
reader/writer load. Lifecycle mirrors
:class:`~xrlenv.control.gc_reconciler.GCReconciler`: ``await
checkpointer.start()`` from runtime startup, ``await
checkpointer.shutdown()`` from runtime shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class _Checkpointable(Protocol):
    """The single method the checkpointer needs. A store without it
    (e.g. the in-memory store used in tests) is simply never wrapped."""

    def checkpoint_wal(self, *, truncate: bool = True) -> tuple[int, int, int]: ...


class WalCheckpointer:
    """Runs one background task that checkpoints the state-store WAL every
    ``interval_s`` seconds.

    The checkpoint runs in a worker thread (``asyncio.to_thread``) because
    a ``TRUNCATE`` checkpoint of a large WAL on a network filesystem can
    take seconds — running it inline would block the event loop, which is
    the very failure this task exists to prevent. Any exception is caught
    and logged so a transient DB error can't kill the loop.
    """

    def __init__(
        self,
        *,
        state: _Checkpointable,
        interval_s: float = 60.0,
    ) -> None:
        self._state = state
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Count consecutive busy (non-truncating) checkpoints so a reader
        # that persistently pins the WAL surfaces as a single escalating
        # warning instead of silence or per-tick spam.
        self._consecutive_busy = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._loop(), name="wal-checkpointer",
            )

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._interval_s,
                    )
                    return  # stop fired during sleep
                except TimeoutError:
                    pass
                try:
                    await self.checkpoint_once()
                except Exception:
                    LOGGER.exception(
                        "wal-checkpointer: checkpoint tick raised; continuing",
                    )
        except asyncio.CancelledError:
            raise

    async def checkpoint_once(self) -> tuple[int, int, int]:
        """Run one checkpoint and log the outcome. Returns SQLite's
        ``(busy, log_frames, checkpointed_frames)``."""
        busy, log_frames, checkpointed = await asyncio.to_thread(
            self._state.checkpoint_wal, truncate=True,
        )
        if busy:
            self._consecutive_busy += 1
            # A reader (almost always the admin panel mid-query) held the
            # WAL open, so TRUNCATE could not reset the file this round.
            # One passive checkpoint still folds what it can; warn only
            # once contention persists so the WAL can't creep unnoticed.
            if self._consecutive_busy >= 3:
                LOGGER.warning(
                    "wal-checkpointer: WAL truncate blocked %d consecutive "
                    "ticks (a reader is pinning the WAL); %d frames pending. "
                    "The -wal file cannot shrink until the reader releases.",
                    self._consecutive_busy, log_frames,
                )
        else:
            if self._consecutive_busy:
                LOGGER.info(
                    "wal-checkpointer: WAL truncate recovered after %d "
                    "blocked tick(s)", self._consecutive_busy,
                )
            self._consecutive_busy = 0
            if checkpointed:
                LOGGER.debug(
                    "wal-checkpointer: checkpointed %d WAL frame(s); "
                    "-wal reset", checkpointed,
                )
        return (busy, log_frames, checkpointed)
