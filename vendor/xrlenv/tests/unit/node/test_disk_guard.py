"""WS2 — DiskPressureGuard policy tests.

The guard's collaborators are injected callables, so the kill policy is
exercised here with no docker / image cache — just numbers. The
production wiring (image cache + raw manager) is covered by
test_raw_container_disk.py and the module's ``build_disk_guard``.
"""

from __future__ import annotations

import pytest
from xrlenv.node.disk_guard import DiskGuardConfig, DiskPressureGuard
from xrlenv.node.raw_container import RawContainerDiskUsage


def _usage(cid: str, size: int, *, rollout: str = "r", image: str = "img:1"):
    return RawContainerDiskUsage(
        container_id=cid, rollout_id=rollout, image=image, size_rw_bytes=size,
    )


def _make_guard(
    *,
    free: int,
    total: int,
    critical: int,
    recovery: int,
    evictable: int,
    offenders: list[RawContainerDiskUsage],
    killed: list[str],
    fail_on: set[str] | None = None,
    scanned: list[bool] | None = None,
) -> DiskPressureGuard:
    fail = fail_on or set()

    async def sample_disk() -> tuple[int, int]:
        return (free, total)

    async def list_offenders() -> list[RawContainerDiskUsage]:
        if scanned is not None:
            scanned.append(True)
        return list(offenders)

    async def kill(off: RawContainerDiskUsage) -> None:
        if off.container_id in fail:
            raise RuntimeError(f"simulated kill failure for {off.container_id}")
        killed.append(off.container_id)

    return DiskPressureGuard(
        sample_disk=sample_disk,
        critical_threshold=lambda: critical,
        recovery_target=lambda: recovery,
        evictable_image_bytes=lambda: evictable,
        list_offenders=list_offenders,
        kill=kill,
        cfg=DiskGuardConfig(interval_s=0.01),
    )


@pytest.mark.asyncio
async def test_healthy_disk_no_kill_no_scan() -> None:
    """Free ≥ critical → no offender scan, no kills (the cheap hot path
    must not pay for the expensive ``docker ps -s``)."""
    killed: list[str] = []
    scanned: list[bool] = []
    guard = _make_guard(
        free=100, total=500, critical=50, recovery=80, evictable=0,
        offenders=[_usage("a", 999)], killed=killed, scanned=scanned,
    )
    assert await guard.check_once() == []
    assert killed == []
    assert scanned == []  # never scanned offenders


@pytest.mark.asyncio
async def test_unknown_disk_sample_is_treated_as_healthy() -> None:
    """The ``(0, 0)`` 'sample failed' sentinel must not trigger kills on
    a blind reading."""
    killed: list[str] = []
    guard = _make_guard(
        free=0, total=0, critical=50, recovery=80, evictable=0,
        offenders=[_usage("a", 999)], killed=killed,
    )
    assert await guard.check_once() == []
    assert killed == []


@pytest.mark.asyncio
async def test_pressure_defers_when_image_eviction_can_relieve() -> None:
    """Under pressure but with enough evictable image space to cover the
    shortfall, the guard defers to the image-eviction sweep — no kills."""
    killed: list[str] = []
    scanned: list[bool] = []
    # shortfall = recovery(80) - free(40) = 40; evictable 50 ≥ 40 → defer.
    guard = _make_guard(
        free=40, total=500, critical=50, recovery=80, evictable=50,
        offenders=[_usage("a", 999)], killed=killed, scanned=scanned,
    )
    assert await guard.check_once() == []
    assert killed == []
    assert scanned == []  # deferred before the expensive scan


@pytest.mark.asyncio
async def test_pressure_kills_largest_until_recovery() -> None:
    """When image eviction can't relieve it, kill largest-writable first
    and stop once projected free reaches the recovery target — the
    smallest offender is spared."""
    killed: list[str] = []
    # free=5, recovery=20, evictable=0 → shortfall 15, must reclaim ≥15.
    # offenders 8,7,1 (desc): kill 8 → 13, kill 7 → 20 (stop); 1 spared.
    guard = _make_guard(
        free=5, total=500, critical=10, recovery=20, evictable=0,
        offenders=[_usage("small", 1), _usage("big", 8), _usage("mid", 7)],
        killed=killed,
    )
    result = await guard.check_once()
    assert [o.container_id for o in result] == ["big", "mid"]
    assert killed == ["big", "mid"]  # largest-first, stopped at recovery


@pytest.mark.asyncio
async def test_pressure_with_no_offenders_logs_and_no_kill(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pressure with nothing reclaimable from images AND no raw-container
    offender → ERROR (manual-intervention signal), no kills."""
    killed: list[str] = []
    guard = _make_guard(
        free=5, total=500, critical=10, recovery=20, evictable=0,
        offenders=[], killed=killed,
    )
    with caplog.at_level("ERROR", logger="xrlenv.node.disk_guard"):
        assert await guard.check_once() == []
    assert killed == []
    assert any(
        "manual intervention" in r.message for r in caplog.records
    )


@pytest.mark.asyncio
async def test_kill_failure_continues_to_next_offender() -> None:
    """A kill that raises is logged and skipped; the guard proceeds to
    the next offender rather than aborting the whole pass."""
    killed: list[str] = []
    # largest 'bad' fails; guard should still kill the next-largest.
    guard = _make_guard(
        free=5, total=500, critical=10, recovery=20, evictable=0,
        offenders=[_usage("bad", 30), _usage("good", 30)],
        killed=killed, fail_on={"bad"},
    )
    result = await guard.check_once()
    assert [o.container_id for o in result] == ["good"]
    assert killed == ["good"]


@pytest.mark.asyncio
async def test_zero_size_offenders_skipped() -> None:
    """Containers reporting 0 writable bytes aren't worth killing — they
    free nothing — so they're skipped even under pressure."""
    killed: list[str] = []
    guard = _make_guard(
        free=5, total=500, critical=10, recovery=20, evictable=0,
        offenders=[_usage("zero", 0), _usage("real", 50)],
        killed=killed,
    )
    result = await guard.check_once()
    assert [o.container_id for o in result] == ["real"]
    assert "zero" not in killed


@pytest.mark.asyncio
async def test_disabled_guard_run_loop_returns_immediately() -> None:
    """A disabled guard's run_loop is a no-op (no polling)."""
    killed: list[str] = []
    guard = _make_guard(
        free=5, total=500, critical=10, recovery=20, evictable=0,
        offenders=[_usage("a", 50)], killed=killed,
    )
    guard._cfg = DiskGuardConfig(enabled=False, interval_s=0.01)
    await guard.run_loop()  # returns at once; would loop forever if enabled
    assert killed == []


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_DISK_GUARD_INTERVAL_S", "7.5")
    monkeypatch.setenv("XRLENV_DISK_GUARD_ENABLED", "false")
    cfg = DiskGuardConfig.from_env()
    assert cfg.interval_s == 7.5
    assert cfg.enabled is False
