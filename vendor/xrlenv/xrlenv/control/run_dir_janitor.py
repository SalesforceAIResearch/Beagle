"""Run-dir rotation janitor (spec 09 GC layer 4).

Phase 0 stores per-rollout artifacts under
``~/.xrlenv/runs/<YYYY-MM-DD>/<rollout_id>/`` (spec 20 layout). After a few
weeks of training those directories accumulate to GBs even with modest
rollout volume — the trajectory body alone for an SWE-bench Lite run is
typically 5-50 MB, and the per-rollout coordinator/node/stub/env_adapter
log files add another 1-10 MB.

This janitor sweeps date directories older than ``retention_days`` (default
14) once at startup and once every 24 h thereafter. It walks
``runs_root.iterdir()`` looking for ``YYYY-MM-DD`` directories and deletes
in-place (no archival to object storage in phase 0 — that's spec-20 phase 2
work).

The sweep runs in an asyncio task driven by an asyncio.Event so callers can
nudge it to run sooner (used in tests + `xrlenv reload`-style operator
flows). All disk work goes through ``asyncio.to_thread`` so the event loop
stays responsive even when a date directory holds thousands of rollout
subdirs.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_DATE_FORMAT = "%Y-%m-%d"
_DEFAULT_INTERVAL_S = 24 * 60 * 60  # 24 h


class RunDirJanitor:
    """Background task that prunes run-dir date directories older than
    ``retention_days``.
    """

    def __init__(
        self,
        runs_root: Path,
        *,
        retention_days: int = 14,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ) -> None:
        if retention_days < 1:
            raise ValueError(
                f"retention_days must be >= 1, got {retention_days}"
            )
        self._runs_root = runs_root
        self._retention_days = retention_days
        self._interval_s = interval_s
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def runs_root(self) -> Path:
        return self._runs_root

    @property
    def retention_days(self) -> int:
        return self._retention_days

    async def start(self) -> None:
        """Schedule the janitor task. Idempotent."""
        if self._task is None:
            self._task = asyncio.create_task(
                self._loop(), name="run-dir-janitor"
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

    async def sweep_once(self) -> list[Path]:
        """Run one sweep synchronously; return the list of pruned date dirs.

        Public for tests + the operator CLI's ``xrlenv reload`` (later
        slice). Calling this from inside the loop body is also fine —
        sweeps are idempotent.
        """
        return await asyncio.to_thread(self._sweep_blocking)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while True:
            try:
                pruned = await self.sweep_once()
            except Exception:
                LOGGER.exception(
                    "run-dir janitor: sweep failed; continuing on next tick"
                )
            else:
                if pruned:
                    LOGGER.info(
                        "run-dir janitor: pruned %d date dir(s): %s",
                        len(pruned),
                        [str(p.name) for p in pruned],
                    )

            with suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._interval_s)
            self._wakeup.clear()

    def _sweep_blocking(self) -> list[Path]:
        """Walk the runs root and delete date dirs older than retention.

        Runs in a thread (called via ``asyncio.to_thread``) so a slow
        ``shutil.rmtree`` over thousands of rollout subdirs doesn't stall
        the event loop.
        """
        if not self._runs_root.exists():
            return []
        cutoff = datetime.now(UTC).date() - timedelta(days=self._retention_days)
        pruned: list[Path] = []
        for entry in sorted(self._runs_root.iterdir()):
            if not entry.is_dir():
                continue
            try:
                entry_date = datetime.strptime(entry.name, _DATE_FORMAT).date()
            except ValueError:
                # Not a YYYY-MM-DD date dir — leave it alone (could be
                # operator-managed cache or a sibling file like state.db).
                continue
            if entry_date >= cutoff:
                continue
            try:
                shutil.rmtree(entry)
                pruned.append(entry)
            except OSError:
                LOGGER.exception(
                    "run-dir janitor: failed to rmtree %s; will retry next tick",
                    entry,
                )
        return pruned


__all__ = ["RunDirJanitor"]
