"""Shared disk-pressure policy — the single source of truth for the
free-disk thresholds the control-plane scheduler and the node image
cache must agree on.

Two planes make disk decisions and MUST NOT hardcode disagreeing
thresholds. When they did, prod nodes deadlocked permanently excluded
(a production deadlock observed on a worker node on 2026-07-01):

- the scheduler's placement gate (control plane, ``_is_disk_pressured``)
  refuses to place work on a node whose free disk is at/below the
  *admit floor* — ``max(absolute floor, fraction x total)``;
- the node image cache (data plane) evicts cold images to keep free disk
  inside its evict band.

The bug was that the cache's evict floors (absolute 15/25 GiB) sat at or
below the scheduler's admit floor (5% of total = 25 GiB on a 500 GiB
disk), so a node the cache considered *healthy* was simultaneously
*scheduler-excluded*, got no work, generated no eviction pressure, and
stayed pinned there forever.

The invariant this module makes enforceable: the cache keeps free disk
strictly ABOVE the scheduler's admit floor, with a hysteresis gap
(``TARGET_MARGIN > START_MARGIN > 1``). Both the admit floor and the
cache margins are defined ONCE here and scale with the disk
(fraction-based), so a large data-root can't silently re-break the
relationship the way an absolute floor did.
"""

from __future__ import annotations

# Scheduler admit floor: refuse placement when free <= max(absolute,
# fraction x total). The fraction catches large-disk nodes where the
# absolute floor is generous.
DISK_PRESSURE_FREE_BYTES_FLOOR: int = 5 * 1024**3
DISK_PRESSURE_FREE_FRACTION_FLOOR: float = 0.05

# The image cache holds free disk this many multiples of the admit floor
# above it, so a cache-maintained node stays comfortably admittable:
#   start evicting  when free  <  START_MARGIN  x admit_floor
#   stop  evicting  when free  >= TARGET_MARGIN x admit_floor
# TARGET_MARGIN > START_MARGIN > 1 guarantees the whole evict band sits
# ABOVE the scheduler's exclusion boundary, with hysteresis so eviction
# doesn't immediately re-trigger.
CACHE_EVICT_START_MARGIN: float = 1.5
CACHE_EVICT_TARGET_MARGIN: float = 2.0


def disk_admit_free_floor_bytes(total_bytes: int) -> int:
    """Free-disk level at/below which the scheduler refuses placement.

    ``max(absolute floor, fraction x total)``. A node with ``free <=``
    this is disk-pressured (excluded).
    """
    frac = int(DISK_PRESSURE_FREE_FRACTION_FLOOR * max(0, total_bytes))
    return max(DISK_PRESSURE_FREE_BYTES_FLOOR, frac)


def cache_evict_floor_bytes(total_bytes: int, margin: float) -> int:
    """Free-disk floor the image cache holds to stay ``margin`` x above
    the scheduler's admit floor. Pass ``CACHE_EVICT_START_MARGIN`` for
    the eviction-start level, ``CACHE_EVICT_TARGET_MARGIN`` for the
    eviction-stop level.
    """
    return int(margin * disk_admit_free_floor_bytes(total_bytes))


__all__ = [
    "CACHE_EVICT_START_MARGIN",
    "CACHE_EVICT_TARGET_MARGIN",
    "DISK_PRESSURE_FREE_BYTES_FLOOR",
    "DISK_PRESSURE_FREE_FRACTION_FLOOR",
    "cache_evict_floor_bytes",
    "disk_admit_free_floor_bytes",
]
