"""Node-local adaptive image-pull concurrency (AIMD).

Replaces the static two-lane (runtime / prefetch) pull semaphores with a
single resizable limiter whose bound an AIMD controller moves between a
floor and a ceiling based on the node's own load:

* **busy** node (running rollouts above a threshold) → multiplicative
  decrease (halve toward the floor), so cold pulls never starve live,
  time-sensitive agent containers;
* **calm** node → additive increase (slow ramp toward the ceiling), so an
  idle cluster (e.g. ``xrlenv build apply``) saturates the pull pipe.

This mirrors the control-plane admission AIMD (``HealthAimdController`` in
``xrlenv.control.capacity``) but runs node-locally with no new wire
protocol — the only input is the node's in-use container count, which the
``ImageCacheManager`` already tracks.
"""

from __future__ import annotations

import asyncio


class AdjustableSemaphore:
    """An async concurrency limiter whose bound can change at runtime.

    ``asyncio.Semaphore`` fixes its bound at construction; the AIMD
    controller needs to raise/lower it live, so this re-implements the
    acquire / release accounting against a mutable ``limit``. Lowering the
    limit never cancels in-flight holders — they drain naturally, and new
    acquirers simply wait until ``in_flight`` falls below the new bound.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._in_flight = 0
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def set_limit(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        async with self._cond:
            raised = limit > self._limit
            self._limit = limit
            if raised:
                # Slots opened up — wake parked acquirers to re-check.
                self._cond.notify_all()

    async def acquire(self) -> None:
        async with self._cond:
            while self._in_flight >= self._limit:
                await self._cond.wait()
            self._in_flight += 1

    async def release(self) -> None:
        async with self._cond:
            if self._in_flight > 0:
                self._in_flight -= 1
            self._cond.notify_all()

    async def __aenter__(self) -> AdjustableSemaphore:
        await self.acquire()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()


class PullAimdController:
    """Pure AIMD controller for the pull-concurrency limit.

    ``observe(busy=...)`` is called once per tick:

    * ``busy``  → multiplicative decrease: ``max(floor, limit // 2)``
    * ``calm``  → additive increase: ``min(ceiling, limit + step)``

    The controller is pure (no I/O, no clock) so it is trivially unit
    testable; the :class:`AdjustableSemaphore` and the loop that drives
    it live in the node agent.
    """

    def __init__(
        self,
        *,
        floor: int,
        ceiling: int,
        initial: int,
        additive_step: int = 2,
    ) -> None:
        if floor < 1:
            raise ValueError("floor must be >= 1")
        if ceiling < floor:
            raise ValueError("ceiling must be >= floor")
        if additive_step < 1:
            raise ValueError("additive_step must be >= 1")
        self._floor = floor
        self._ceiling = ceiling
        self._step = additive_step
        self._limit = min(max(initial, floor), ceiling)

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def floor(self) -> int:
        return self._floor

    @property
    def ceiling(self) -> int:
        return self._ceiling

    def set_ceiling(self, ceiling: int) -> None:
        """Update the ceiling at runtime (e.g. a disk-bounded cap that
        scales with free disk). Never below the floor; clamps the current
        limit down if it now exceeds the new ceiling."""
        self._ceiling = max(self._floor, ceiling)
        if self._limit > self._ceiling:
            self._limit = self._ceiling

    def observe(self, *, busy: bool) -> int:
        if busy:
            self._limit = max(self._floor, self._limit // 2)
        else:
            self._limit = min(self._ceiling, self._limit + self._step)
        return self._limit
