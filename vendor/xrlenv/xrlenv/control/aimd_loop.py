"""Stage-3 — the AIMD admission control loop.

Periodically feeds the Stage-1 per-node health signals + current load
into the pure :class:`~xrlenv.control.capacity.HealthAimdController`, so
its per-node admission limits track docker-daemon health. The scheduler
reads those limits to gate placement. See
``notes/admission-stage-3-aimd-controller.md``.

The loop is the only I/O / async piece of pillar P1; the controller it
drives is pure. Lifecycle mirrors the GC reconciler: ``await start()``
from runtime startup, ``await shutdown()`` from runtime shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from xrlenv.control.capacity import HealthAimdController, NodeHealthInput

LOGGER = logging.getLogger(__name__)

_TICK_INTERVAL_S: float = 15.0


def _health_input_from_dict(raw: Any) -> NodeHealthInput | None:
    """Map a transport's stashed Stage-1 ``_last_health`` dict to the
    controller's input type. ``None`` / malformed → ``None`` — the
    controller treats that as 'unknown' and holds the node's limit."""
    if not isinstance(raw, dict):
        return None
    try:
        return NodeHealthInput(
            create_p95_ms=float(raw.get("create_p95_ms", 0.0)),
            docker_error_count=int(raw.get("docker_error_count", 0)),
            docker_timeout_count=int(raw.get("docker_timeout_count", 0)),
        )
    except (TypeError, ValueError):
        return None


class AimdControlLoop:
    """Background loop that ticks a :class:`HealthAimdController`."""

    def __init__(
        self,
        *,
        controller: HealthAimdController,
        registry: Any,   # NodeRegistry — duck-typed: .node_ids, .get()
        scheduler: Any,   # Scheduler — duck-typed: .node_load_snapshot()
        state: Any = None,    # StateStore — for the admin-panel mirror
        metrics: Any = None,  # MetricsRegistry — for the per-node gauge
        interval_s: float = _TICK_INTERVAL_S,
    ) -> None:
        self._controller = controller
        self._registry = registry
        self._scheduler = scheduler
        self._state = state
        self._metrics = metrics
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._loop(), name="aimd-control-loop",
            )

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def tick(self) -> None:
        """One control round — gather per-node health + load, run the
        AIMD step. Synchronous + cheap (in-memory reads + arithmetic);
        separated from the loop so it is unit-testable on its own."""
        health: dict[str, NodeHealthInput | None] = {}
        for node_id in self._registry.node_ids:
            transport = self._registry.get(node_id)
            raw = getattr(transport, "_last_health", None)
            health[node_id] = _health_input_from_dict(raw)
        load = self._scheduler.node_load_snapshot()
        self._controller.step(health=health, load=load)
        # Stage-3 (3c) — mirror each node's limit to the state store so
        # the admin "Cluster health" page can show it out-of-process,
        # and emit the per-node gauge for graphing AIMD's sawtooth.
        for node_id in load:
            limit = self._controller.limit_for(node_id)
            if self._state is not None:
                with contextlib.suppress(Exception):
                    self._state.update_node_aimd_limit(node_id, limit)
            if self._metrics is not None:
                with contextlib.suppress(Exception):
                    self._metrics.set_node_admission_limit(
                        node_id, float(limit),
                    )

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._interval_s,
                    )
                if self._stop.is_set():
                    return
                try:
                    self.tick()
                except Exception:
                    LOGGER.exception(
                        "aimd-control-loop: tick raised; will retry "
                        "next interval",
                    )
        except asyncio.CancelledError:
            return


__all__ = ["AimdControlLoop"]
