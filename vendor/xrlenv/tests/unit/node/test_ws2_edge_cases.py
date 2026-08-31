"""WS2 edge cases — additional disk guard and image refcount coverage.

Supplements test_disk_guard.py and test_raw_container_disk.py with:

 * ``build_disk_guard`` returns None when the agent has no
   ``raw_container_manager`` method (no docker backend).
 * ``build_disk_guard`` returns None when the agent has no
   ``image_cache`` attribute.
 * ``force_destroy`` of a TRUE orphan (no in-memory record, not tracked
   by the cache) does NOT call ``image_cache.release`` — the cache never
   acquired the orphan.
 * Image refcount: double destroy (benign race) releases the cache only
   ONCE (the first destroy does the release; a second attempt fails the
   ownership check before touching the cache).
 * Guard: exactly-at-recovery stops killing (``projected_free >=
   recovery`` exits immediately, sparing the next offender).
 * ``evictable_image_bytes`` returns 0 when there are no cached images
   (the cache has an empty listing) — no ZeroDivision / negative result.
"""

from __future__ import annotations

from typing import Any

import pytest
from xrlenv.node.disk_guard import DiskGuardConfig, DiskPressureGuard, build_disk_guard
from xrlenv.node.raw_container import RawContainerDiskUsage, RawContainerManager

# ── helpers ───────────────────────────────────────────────────────────────────


class _FakeCache:
    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.released: list[str] = []

    async def ensure_present(
        self, image: str, *, deadline_s: float | None = None,
    ) -> None:
        pass

    def acquire(self, image: str) -> None:
        self.acquired.append(image)

    def release(self, image: str) -> None:
        self.released.append(image)


class _NotFound(Exception):
    pass


class _FakeContainer:
    def __init__(self, cid: str, name: str, image: str, labels: dict[str, str]) -> None:
        self.id = cid
        self.name = name
        self.labels = labels
        self.removed = False
        self.attrs: dict[str, Any] = {
            "HostConfig": {"CpusetCpus": ""},
            "Config": {"Labels": dict(labels), "Image": image},
            "SizeRw": 0,
        }

    def remove(self, *, force: bool = False) -> None:
        self.removed = True


class _FakeContainers:
    def __init__(self) -> None:
        self._n = 0
        self._reg: dict[str, _FakeContainer] = {}

    def list(
        self, *, filters: dict[str, str] | None = None,
        all: bool = False, sparse: bool = False, size: bool = False,
    ) -> list[_FakeContainer]:
        return [
            c for c in self._reg.values()
            if not c.removed
            and c.labels.get("xrlenv.session_kind") == "raw"
        ]

    def run(
        self, *, image: str, detach: bool, labels: dict[str, str],
        command: list[str] | None = None, name: str | None = None,
        **_extra: Any,
    ) -> _FakeContainer:
        self._n += 1
        cid = f"c-{self._n:04d}"
        c = _FakeContainer(cid, name or cid, image, labels)
        self._reg[cid] = c
        return c

    def get(self, cid: str) -> _FakeContainer:
        c = self._reg.get(cid)
        if c is None or c.removed:
            raise _NotFound(cid)
        return c


class _FakeImages:
    def get(self, image: str) -> Any:
        return object()


class _FakeClient:
    def __init__(self) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers()
        self.api = None


# ── build_disk_guard wiring ───────────────────────────────────────────────────


def test_build_disk_guard_returns_none_without_raw_manager() -> None:
    """An agent with an image_cache but no raw_container_manager should
    return None from build_disk_guard (guard is not started)."""

    class _AgentNoManager:
        image_cache = object()  # not None, but no raw_container_manager

    assert build_disk_guard(_AgentNoManager()) is None


def test_build_disk_guard_returns_none_without_image_cache() -> None:
    """An agent with a raw_container_manager but no image_cache should
    return None (the guard needs the cache for adaptive thresholds)."""

    class _AgentNoCache:
        image_cache = None

        def raw_container_manager(self, backend: str) -> object:
            return object()

    assert build_disk_guard(_AgentNoCache()) is None


def test_build_disk_guard_returns_none_when_manager_getter_returns_none() -> None:
    """If raw_container_manager("docker") returns None, build_disk_guard
    should return None (no docker-backed manager wired)."""

    class _AgentNullManager:
        image_cache = object()

        def raw_container_manager(self, backend: str) -> None:
            return None

    assert build_disk_guard(_AgentNullManager()) is None


# ── force_destroy true-orphan: no release ────────────────────────────────────


@pytest.mark.asyncio
async def test_force_destroy_untracked_orphan_does_not_release_cache() -> None:
    """A container force-destroyed that was never acquired through this
    manager (a true CP-restart orphan with no in-memory record) must NOT
    call image_cache.release — the cache never saw an acquire() for it."""
    client = _FakeClient()
    cache = _FakeCache()
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    # Inject a container directly into the docker fake (simulating an
    # orphan that predates this manager instance, e.g. a CP restart).
    orphan = client.containers.run(
        image="orphan:img",
        detach=True,
        labels={"xrlenv.session_kind": "raw", "xrlenv.rollout_id": "old-r"},
    )
    # _records is empty — manager has no knowledge of this container.
    assert orphan.id not in mgr._records

    await mgr.force_destroy(container_id=orphan.id)

    assert orphan.removed is True
    # The cache should never have been touched for an untracked orphan.
    assert cache.released == [], (
        "force_destroy of an untracked orphan must not release the cache"
    )


# ── image refcount: benign race (already gone) releases exactly once ──────────


@pytest.mark.asyncio
async def test_destroy_already_removed_releases_cache_once() -> None:
    """When the container is gone from docker before destroy reaches the
    docker call, the benign-race branch still calls image_cache.release
    exactly once. The container being gone does not prevent the release."""
    client = _FakeClient()
    cache = _FakeCache()
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    rec = await mgr.acquire(rollout_id="r", image="myimg:1")
    # Container vanishes out from under us.
    client.containers._reg[rec.container_id].removed = True

    await mgr.destroy(rollout_id="r", container_id=rec.container_id)

    assert cache.released == ["myimg:1"]
    # Container is no longer tracked.
    assert rec.container_id not in mgr._records


# ── guard: stops at exact recovery ───────────────────────────────────────────


def _usage(cid: str, size: int) -> RawContainerDiskUsage:
    return RawContainerDiskUsage(
        container_id=cid, rollout_id="r", image="img:1", size_rw_bytes=size,
    )


@pytest.mark.asyncio
async def test_guard_stops_exactly_at_recovery_spares_next_offender() -> None:
    """Kill the first offender takes projected_free to exactly ``recovery``.
    The loop checks ``projected_free >= recovery`` at the START of each
    iteration, so the second offender must be spared."""
    killed: list[str] = []

    async def sample_disk() -> tuple[int, int]:
        return (5, 500)

    async def list_offenders() -> list[RawContainerDiskUsage]:
        return [_usage("big", 15), _usage("small", 5)]

    async def kill(off: RawContainerDiskUsage) -> None:
        killed.append(off.container_id)

    # free=5, recovery=20, evictable=0 → shortfall=15.
    # Kill "big" (15 bytes) → projected_free=20 == recovery → stop.
    guard = DiskPressureGuard(
        sample_disk=sample_disk,
        critical_threshold=lambda: 10,
        recovery_target=lambda: 20,
        evictable_image_bytes=lambda: 0,
        list_offenders=list_offenders,
        kill=kill,
        cfg=DiskGuardConfig(interval_s=0.01),
    )
    result = await guard.check_once()

    assert [o.container_id for o in result] == ["big"]
    assert killed == ["big"]  # "small" must be spared


@pytest.mark.asyncio
async def test_guard_all_offenders_zero_size_logs_error_no_kill(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All offenders have size_rw_bytes == 0 — none are killable —
    the guard should log an error about 'no killable offender' and
    return an empty list."""
    killed: list[str] = []

    async def sample_disk() -> tuple[int, int]:
        return (2, 500)

    async def list_offenders() -> list[RawContainerDiskUsage]:
        return [_usage("z1", 0), _usage("z2", 0)]

    async def kill(off: RawContainerDiskUsage) -> None:  # pragma: no cover
        killed.append(off.container_id)

    guard = DiskPressureGuard(
        sample_disk=sample_disk,
        critical_threshold=lambda: 10,
        recovery_target=lambda: 20,
        evictable_image_bytes=lambda: 0,
        list_offenders=list_offenders,
        kill=kill,
        cfg=DiskGuardConfig(interval_s=0.01),
    )
    with caplog.at_level("ERROR", logger="xrlenv.node.disk_guard"):
        result = await guard.check_once()

    assert result == []
    assert killed == []
    assert any("no killable" in r.message or "manual" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_guard_evictable_exactly_equals_shortfall_defers() -> None:
    """When evictable_image_bytes == shortfall (not strictly greater),
    the guard should still defer to image eviction (>= condition)."""
    killed: list[str] = []
    scanned: list[bool] = []

    async def sample_disk() -> tuple[int, int]:
        return (30, 500)  # free=30 < critical=40

    async def list_offenders() -> list[RawContainerDiskUsage]:
        scanned.append(True)
        return [_usage("a", 100)]

    async def kill(off: RawContainerDiskUsage) -> None:  # pragma: no cover
        killed.append(off.container_id)

    # shortfall = recovery(60) - free(30) = 30 == evictable(30) → defer
    guard = DiskPressureGuard(
        sample_disk=sample_disk,
        critical_threshold=lambda: 40,
        recovery_target=lambda: 60,
        evictable_image_bytes=lambda: 30,  # exactly equal to shortfall
        list_offenders=list_offenders,
        kill=kill,
        cfg=DiskGuardConfig(interval_s=0.01),
    )
    result = await guard.check_once()

    assert result == []
    assert killed == []
    assert scanned == [], "should NOT scan offenders when deferring to image eviction"
