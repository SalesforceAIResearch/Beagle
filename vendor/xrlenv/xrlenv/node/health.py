"""Node-side health-signal collector.

Stage 1 of the admission/capacity design
(``notes/admission-stage-1-observability.md``). Records cheap
per-operation samples on the hot path; :meth:`NodeHealthCollector.snapshot`
folds them into a value object the heartbeat carries to the control
plane. Pure in-memory — no docker round trip — so it is safe to read on
the heartbeat path (issue #18 kept that path daemon-independent).

These signals are the instrument panel for the future health-derived
admission controller (Stage 3) and feed the admin "Cluster health" page
(Stage 1, P3.3).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeHealthSnapshot:
    """Point-in-time per-node health signals.

    ``create_*`` latency is the *smooth* node-saturation signal (a
    docker daemon answering ``docker run`` slowly); the docker
    error/timeout counts are the *emergency* signal. ``create_inflight``
    / ``create_queued`` expose the per-node create-gate contention.
    """

    window_s: float
    create_p50_ms: float
    create_p95_ms: float
    create_count: int
    docker_error_count: int
    docker_timeout_count: int
    create_inflight: int
    create_queued: int


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list; ``0.0`` empty."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


class NodeHealthCollector:
    """Rolling-window collector of node-side docker-operation health.

    ``record_*`` is O(1) and runs on the hot path. ``snapshot`` runs once
    per heartbeat (~5 s) and is the only place that sorts — a handful of
    microseconds over a ~120 s window.
    """

    def __init__(self, *, window_s: float = 120.0) -> None:
        self._window_s = window_s
        # (monotonic_ts, duration_s) for completed ``docker run`` calls.
        self._creates: deque[tuple[float, float]] = deque()
        # monotonic_ts of node-side docker errors / timeouts (timeouts ⊆
        # errors). Fed from the create path today; destroy / exec sites
        # feed the same counters as they gain error translation.
        self._errors: deque[float] = deque()
        self._timeouts: deque[float] = deque()

    def record_create(self, duration_s: float) -> None:
        """Record one completed ``docker run`` call's wall duration."""
        self._creates.append((time.monotonic(), duration_s))

    def record_docker_error(self, *, is_timeout: bool) -> None:
        """Record one node-side docker-operation failure."""
        now = time.monotonic()
        self._errors.append(now)
        if is_timeout:
            self._timeouts.append(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._creates and self._creates[0][0] < cutoff:
            self._creates.popleft()
        while self._errors and self._errors[0] < cutoff:
            self._errors.popleft()
        while self._timeouts and self._timeouts[0] < cutoff:
            self._timeouts.popleft()

    def snapshot(
        self, *, create_inflight: int = 0, create_queued: int = 0,
    ) -> NodeHealthSnapshot:
        """Evict the stale window, fold the rest into a snapshot.

        ``create_inflight`` / ``create_queued`` are passed in by the
        caller (read from the create-gate semaphore) — the collector
        owns only time-series samples, not live gate state.
        """
        now = time.monotonic()
        self._evict(now)
        durations = sorted(d for _, d in self._creates)
        return NodeHealthSnapshot(
            window_s=self._window_s,
            create_p50_ms=_percentile(durations, 0.50) * 1000.0,
            create_p95_ms=_percentile(durations, 0.95) * 1000.0,
            create_count=len(durations),
            docker_error_count=len(self._errors),
            docker_timeout_count=len(self._timeouts),
            create_inflight=max(0, create_inflight),
            create_queued=max(0, create_queued),
        )
