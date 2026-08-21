"""Run identity — config hash (order-independent, raw-dict) + human-readable run id."""

from __future__ import annotations

from datetime import datetime, timezone

from beagle.config import RunConfig
from beagle.rollout.run_id import build_run_id, compute_config_hash


def test_config_hash_is_order_independent_and_prefixed() -> None:
    a = compute_config_hash({"model": {"name": "x"}, "parallelism": 2})
    b = compute_config_hash({"parallelism": 2, "model": {"name": "x"}})  # keys reordered
    assert a == b and a.startswith("sha256:") and len(a) == len("sha256:") + 64
    assert compute_config_hash({"model": {"name": "y"}}) != a  # content-sensitive


def test_build_run_id_shape_and_sanitization() -> None:
    cfg = RunConfig.from_dict({
        "model": {"name": "org/gpt-5.5"},  # slug takes the last path segment
        "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t"]},
    })
    h = compute_config_hash({"any": "thing"})
    rid = build_run_id(cfg, h, timestamp=datetime(2026, 8, 4, 5, 30, 0, tzinfo=timezone.utc))
    assert rid == f"2026-08-04T05-30-00Z__gpt-5.5__monet__terminal_bench_2_1__{h.split(':')[1][:8]}"
