"""Regression tests for the P1 disk-exclusion deadlock (notes/audit.md).

The control-plane scheduler and the node image cache independently make
disk decisions. If the cache maintains free disk at/below the
scheduler's placement-admit floor, a node the cache calls *healthy* is
simultaneously scheduler-*excluded* → gets no work → generates no
eviction pressure → stays pinned excluded forever (observed on
``aws-node-host``, 2026-07-01: 24 GiB free / 4.8% on a 500 GiB disk,
connected but zero placements for hours).

``xrlenv.disk_policy`` is the single source of truth that both planes
consume, and the invariant these tests lock in is: **the cache keeps
free disk strictly ABOVE the scheduler's admit floor, on every disk
size** — so a cache-maintained node is always admittable.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from xrlenv.disk_policy import (
    CACHE_EVICT_START_MARGIN,
    CACHE_EVICT_TARGET_MARGIN,
    DISK_PRESSURE_FREE_BYTES_FLOOR,
    cache_evict_floor_bytes,
    disk_admit_free_floor_bytes,
)

GIB = 1024**3
# Span small → very large data-roots; the deadlock was disk-size
# dependent (equal on 500 GiB, worse on ≥1 TiB), so the invariant must
# hold across the range, not at one hand-picked size.
_DISK_SIZES = [50 * GIB, 100 * GIB, 500 * GIB, 1000 * GIB, 2000 * GIB]


def test_admit_floor_is_fraction_on_large_disks() -> None:
    assert disk_admit_free_floor_bytes(500 * GIB) == 25 * GIB  # 5%
    assert disk_admit_free_floor_bytes(1000 * GIB) == 50 * GIB  # 5%


def test_admit_floor_is_absolute_on_small_disks() -> None:
    # 50 GiB disk: 5% = 2.5 GiB < the 5 GiB absolute floor → floor wins.
    assert disk_admit_free_floor_bytes(50 * GIB) == DISK_PRESSURE_FREE_BYTES_FLOOR
    assert disk_admit_free_floor_bytes(0) == DISK_PRESSURE_FREE_BYTES_FLOOR
    assert disk_admit_free_floor_bytes(-1) == DISK_PRESSURE_FREE_BYTES_FLOOR


@pytest.mark.parametrize("total", _DISK_SIZES)
def test_cache_band_sits_strictly_above_admit_floor(total: int) -> None:
    """THE deadlock regression. A node the cache maintains at its evict
    band must be strictly ABOVE the scheduler's admit floor on every
    disk size. Pre-fix the cache floor was the absolute 15/25 GiB, which
    EQUALS the 5% admit floor on a 500 GiB disk and is BELOW it on
    ≥1 TiB — exactly the collision that deadlocked 174-9."""
    admit = disk_admit_free_floor_bytes(total)
    start = cache_evict_floor_bytes(total, CACHE_EVICT_START_MARGIN)
    target = cache_evict_floor_bytes(total, CACHE_EVICT_TARGET_MARGIN)
    assert start > admit, f"start {start} must exceed admit {admit}"
    assert target > start, "target must exceed start (hysteresis gap)"


@pytest.mark.parametrize("total", _DISK_SIZES)
def test_node_at_cache_target_is_not_scheduler_excluded(total: int) -> None:
    """Executable cross-plane invariant through the REAL scheduler gate:
    a node whose free disk equals the cache's evict target is NOT
    disk-pressured, so eviction can never pin it excluded. Conversely a
    node exactly at the admit floor IS excluded (the gate still bites)."""
    from xrlenv.control.scheduler import _is_disk_pressured

    @dataclass
    class _Node:
        _state: tuple[int, int]

        def disk_state(self) -> tuple[int, int]:
            return self._state

    target_free = cache_evict_floor_bytes(total, CACHE_EVICT_TARGET_MARGIN)
    assert _is_disk_pressured(_Node((target_free, total))) is False
    # And the gate still excludes a node sitting at the admit floor.
    admit = disk_admit_free_floor_bytes(total)
    assert _is_disk_pressured(_Node((admit, total))) is True


def test_image_cache_effective_thresholds_respect_admit_floor() -> None:
    """The real ImageCacheManager accessors (which the eviction sweep and
    the WS2 disk guard consume) enforce the floor once a disk total is
    known — so the sweep evicts to keep the node admittable."""
    from xrlenv.node.image_cache import ImageCacheManager

    cache = ImageCacheManager(backend=MagicMock())
    cache._last_total_disk_bytes = 500 * GIB  # scheduler admit floor = 25 GiB

    admit = disk_admit_free_floor_bytes(500 * GIB)
    assert cache.effective_evict_threshold() > admit
    assert cache.effective_evict_target() > cache.effective_evict_threshold()
    # On a 500 GiB disk the 2x target margin (50 GiB) dominates the
    # default 25 GiB adaptive target floor.
    assert cache.effective_evict_target() >= cache_evict_floor_bytes(
        500 * GIB, CACHE_EVICT_TARGET_MARGIN,
    )


def test_unknown_total_falls_back_to_absolute_floor() -> None:
    """Before the first disk sample (``_last_total_disk_bytes == 0``) the
    floor is based on the absolute admit floor, not zero — so a fresh
    node still gets a sane, non-degenerate eviction band."""
    from xrlenv.node.image_cache import ImageCacheManager

    cache = ImageCacheManager(backend=MagicMock())
    assert cache._last_total_disk_bytes == 0
    # cache_evict_floor(0, margin) = margin x 5 GiB absolute floor.
    assert cache.effective_evict_threshold() >= int(
        CACHE_EVICT_START_MARGIN * DISK_PRESSURE_FREE_BYTES_FLOOR,
    )
