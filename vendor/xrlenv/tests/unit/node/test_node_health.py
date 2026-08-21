"""Stage 1 — tests for the node-side health collector
(``xrlenv/node/health.py``)."""

from __future__ import annotations

import time

from xrlenv.node.health import NodeHealthCollector


def test_empty_snapshot_is_all_zero() -> None:
    snap = NodeHealthCollector().snapshot()
    assert snap.create_count == 0
    assert snap.create_p50_ms == 0.0
    assert snap.create_p95_ms == 0.0
    assert snap.docker_error_count == 0
    assert snap.docker_timeout_count == 0


def test_create_percentiles_nearest_rank() -> None:
    """p50/p95 are nearest-rank over the create-duration window."""
    c = NodeHealthCollector()
    for d in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        c.record_create(d)
    snap = c.snapshot()
    assert snap.create_count == 10
    # nearest-rank: p50 -> idx int(0.5*10)=5 -> 0.6s; p95 -> idx 9 -> 1.0s.
    assert snap.create_p50_ms == 600.0
    assert snap.create_p95_ms == 1000.0


def test_docker_error_and_timeout_counts() -> None:
    """Timeouts are a subset of errors."""
    c = NodeHealthCollector()
    c.record_docker_error(is_timeout=True)
    c.record_docker_error(is_timeout=False)
    c.record_docker_error(is_timeout=True)
    snap = c.snapshot()
    assert snap.docker_error_count == 3
    assert snap.docker_timeout_count == 2


def test_window_evicts_stale_samples() -> None:
    """Samples older than ``window_s`` drop out of the snapshot."""
    c = NodeHealthCollector(window_s=0.05)
    c.record_create(0.9)
    c.record_docker_error(is_timeout=True)
    time.sleep(0.08)  # both samples now older than the 50ms window
    c.record_create(0.1)
    snap = c.snapshot()
    assert snap.create_count == 1  # only the fresh create survives
    assert snap.create_p95_ms == 100.0
    assert snap.docker_error_count == 0  # stale error evicted
    assert snap.docker_timeout_count == 0


def test_snapshot_carries_gate_depth() -> None:
    """``create_inflight`` / ``create_queued`` are passed through verbatim."""
    snap = NodeHealthCollector().snapshot(create_inflight=3, create_queued=7)
    assert snap.create_inflight == 3
    assert snap.create_queued == 7
