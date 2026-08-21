"""Tests for the Slice 7b control-plane trajectory cache (spec 17)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xrlenv.control.trajectory_cache import (
    TrajectoryCache,
    TrajectoryCacheConfig,
    cache_size_bytes,
)
from xrlenv.types import RolloutStatus, Step, Trajectory


def _traj(rollout_id: str, *, n_steps: int = 3) -> Trajectory:
    return Trajectory(
        rollout_id=rollout_id,
        template="t",
        steps=[
            Step(
                index=i, action={"a": i}, obs={"o": i}, reward=0.0,
                done=(i == n_steps - 1), truncated=False,
                info={}, ts=float(i),
            )
            for i in range(n_steps)
        ],
        status=RolloutStatus.FINISHED,
        final_reward=0.5,
        metadata={},
    )


@pytest.fixture
def cache(tmp_path: Path) -> TrajectoryCache:
    return TrajectoryCache(
        TrajectoryCacheConfig(
            cache_root=tmp_path / "admin-cache",
            max_bytes=10 * 1024,  # tight so eviction tests fire
            ttl_s=3600.0,
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# get / fetch_fn
# ──────────────────────────────────────────────────────────────────────────────


async def test_cache_miss_invokes_fetch_fn_then_caches(cache: TrajectoryCache) -> None:
    calls = 0

    async def fetch(rid: str) -> Trajectory:
        nonlocal calls
        calls += 1
        return _traj(rid)

    t1 = await cache.get("r-1", fetch)
    assert t1.rollout_id == "r-1"
    assert calls == 1

    # Second call hits the on-disk cache; fetch_fn not invoked.
    t2 = await cache.get("r-1", fetch)
    assert t2.rollout_id == "r-1"
    assert calls == 1


async def test_concurrent_misses_coalesce_into_single_fetch(
    cache: TrajectoryCache,
) -> None:
    calls = 0

    async def slow_fetch(rid: str) -> Trajectory:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return _traj(rid)

    results = await asyncio.gather(*(cache.get("r-x", slow_fetch) for _ in range(5)))
    assert all(t.rollout_id == "r-x" for t in results)
    assert calls == 1


async def test_fetch_fn_failure_propagates_and_does_not_cache(
    cache: TrajectoryCache,
) -> None:
    calls = 0

    async def boom(rid: str) -> Trajectory:
        nonlocal calls
        calls += 1
        raise FileNotFoundError(f"no run dir for {rid}")

    with pytest.raises(FileNotFoundError):
        await cache.get("r-bad", boom)
    # Second call retries (no negative caching).
    with pytest.raises(FileNotFoundError):
        await cache.get("r-bad", boom)
    assert calls == 2


# ──────────────────────────────────────────────────────────────────────────────
# TTL + LRU
# ──────────────────────────────────────────────────────────────────────────────


async def test_expired_cache_entry_triggers_refetch(tmp_path: Path) -> None:
    cache = TrajectoryCache(
        TrajectoryCacheConfig(
            cache_root=tmp_path / "ac",
            max_bytes=10 * 1024**3,
            ttl_s=0.05,
        ),
    )
    calls = 0

    async def fetch(rid: str) -> Trajectory:
        nonlocal calls
        calls += 1
        return _traj(rid)

    await cache.get("r-ttl", fetch)
    assert calls == 1
    await asyncio.sleep(0.1)  # past the 50ms TTL
    await cache.get("r-ttl", fetch)
    assert calls == 2


async def test_sweep_expired_prunes_old_entries(tmp_path: Path) -> None:
    cache_root = tmp_path / "ac"
    cache = TrajectoryCache(
        TrajectoryCacheConfig(cache_root=cache_root, ttl_s=0.05),
    )

    async def fetch(rid: str) -> Trajectory:
        return _traj(rid)

    await cache.get("r-old", fetch)
    await asyncio.sleep(0.1)
    pruned = cache.sweep_expired()
    assert pruned == 1
    assert cache_size_bytes(cache_root) == 0


async def test_lru_eviction_removes_oldest_when_over_budget(tmp_path: Path) -> None:
    cache_root = tmp_path / "ac"

    async def fetch(rid: str) -> Trajectory:
        return _traj(rid, n_steps=10)

    # Size the budget so two entries fit but a third forces eviction.
    one_size = len(_traj("r-x", n_steps=10).model_dump_json())
    cache = TrajectoryCache(
        TrajectoryCacheConfig(
            cache_root=cache_root,
            max_bytes=int(one_size * 2.5),  # room for ~2 entries
            ttl_s=3600.0,
        ),
    )

    await cache.get("r-1", fetch)
    await asyncio.sleep(0.01)
    await cache.get("r-2", fetch)
    await asyncio.sleep(0.01)
    await cache.get("r-3", fetch)

    files = sorted((cache_root).glob("*.json"))
    # Three writes, budget for ~2 → at least one was evicted.
    assert len(files) <= 2
    # r-1 (oldest) is the eviction target.
    assert not (cache_root / "r-1.json").exists()
    # r-3 (newest) is still resident.
    assert (cache_root / "r-3.json").exists()


async def test_invalidate_drops_specific_entry(cache: TrajectoryCache) -> None:
    async def fetch(rid: str) -> Trajectory:
        return _traj(rid)

    await cache.get("r-inv", fetch)
    cache.invalidate("r-inv")
    # Subsequent get triggers a fresh fetch.
    fetch_calls = 0

    async def counting_fetch(rid: str) -> Trajectory:
        nonlocal fetch_calls
        fetch_calls += 1
        return _traj(rid)

    await cache.get("r-inv", counting_fetch)
    assert fetch_calls == 1


# ──────────────────────────────────────────────────────────────────────────────
# Disk corruption resilience
# ──────────────────────────────────────────────────────────────────────────────


async def test_corrupt_cache_file_triggers_refetch(tmp_path: Path) -> None:
    cache_root = tmp_path / "ac"
    cache_root.mkdir()
    (cache_root / "r-corrupt.json").write_text("not-valid-json{")
    cache = TrajectoryCache(
        TrajectoryCacheConfig(cache_root=cache_root, ttl_s=3600.0),
    )

    async def fetch(rid: str) -> Trajectory:
        return _traj(rid)

    fresh = await cache.get("r-corrupt", fetch)
    # Cache returned a freshly-fetched trajectory and replaced the file.
    assert fresh.rollout_id == "r-corrupt"
    assert (cache_root / "r-corrupt.json").exists()


# ──────────────────────────────────────────────────────────────────────────────
# Default config
# ──────────────────────────────────────────────────────────────────────────────


def test_default_config_matches_spec_17() -> None:
    cfg = TrajectoryCacheConfig()
    assert cfg.max_bytes == 5 * 1024**3
    assert cfg.ttl_s == 3600.0
