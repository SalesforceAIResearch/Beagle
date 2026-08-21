"""Tests for the node-local disk-I/O saturation sampler (#1).

The sampler turns the kernel's per-device ``io_ticks`` into a hysteretic
``saturated()`` signal the pull controller uses to back off when the
data-root volume is pegged. Tests inject a fake clock + io_ticks reader so
they never touch ``/sys`` and are fully deterministic.
"""

from __future__ import annotations

import pytest
from xrlenv.node.disk_io import DiskIoSampler


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _Ticks:
    """Monotonic io_ticks counter (ms the device had I/O in flight)."""

    def __init__(self) -> None:
        self.v = 0

    def __call__(self) -> int:
        return self.v

    def add(self, n: int) -> None:
        self.v += n


# ── utilization math ──────────────────────────────────────────────────────────


def test_first_sample_returns_none() -> None:
    s = DiskIoSampler(
        path_provider=lambda: None, min_interval_s=0.0,
        clock=_Clock(), io_ticks_reader=_Ticks(),
    )
    assert s.utilization() is None  # no delta yet
    assert s.last_utilization is None


def test_utilization_is_io_ticks_delta_over_elapsed() -> None:
    clk, tk = _Clock(), _Ticks()
    s = DiskIoSampler(
        path_provider=lambda: None, min_interval_s=0.0,
        clock=clk, io_ticks_reader=tk,
    )
    assert s.utilization() is None  # prime
    clk.advance(1.0)
    tk.add(500)  # busy 500ms of 1000ms
    assert s.utilization() == pytest.approx(0.5)
    clk.advance(1.0)
    tk.add(1000)  # busy the whole interval
    assert s.utilization() == pytest.approx(1.0)


def test_utilization_clamps_to_one() -> None:
    clk, tk = _Clock(), _Ticks()
    s = DiskIoSampler(
        path_provider=lambda: None, min_interval_s=0.0,
        clock=clk, io_ticks_reader=tk,
    )
    s.utilization()
    clk.advance(1.0)
    tk.add(5000)  # impossibly busy (rounding/multiqueue) → clamp 1.0
    assert s.utilization() == pytest.approx(1.0)


def test_counter_wrap_or_reset_yields_none() -> None:
    clk = _Clock()
    vals = iter([1000, 500])  # io_ticks decreases (reset/wrap)
    s = DiskIoSampler(
        path_provider=lambda: None, min_interval_s=0.0,
        clock=clk, io_ticks_reader=lambda: next(vals),
    )
    assert s.utilization() is None  # first read = 1000
    clk.advance(1.0)
    assert s.utilization() is None  # negative delta → discard interval


def test_min_interval_caches_between_samples() -> None:
    clk, tk = _Clock(), _Ticks()
    s = DiskIoSampler(
        path_provider=lambda: None, min_interval_s=2.0,
        clock=clk, io_ticks_reader=tk,
    )
    s.utilization()  # sample at t=0 (ticks=0)
    clk.advance(1.0)
    tk.add(1000)
    assert s.utilization() is None  # 1s < 2s → cached (still None)
    clk.advance(1.5)  # t=2.5, ≥ 2s since last sample
    assert s.utilization() == pytest.approx(1000 / (2.5 * 1000.0))


# ── hysteresis ────────────────────────────────────────────────────────────────


def test_saturated_hysteresis() -> None:
    clk, tk = _Clock(), _Ticks()
    s = DiskIoSampler(
        path_provider=lambda: None, high=0.9, low=0.7,
        min_interval_s=0.0, clock=clk, io_ticks_reader=tk,
    )

    def sat_after(busy_ms: int) -> bool:
        clk.advance(1.0)  # 1s window
        tk.add(busy_ms)   # busy_ms of 1000ms -> util = busy_ms/1000
        return s.saturated()

    assert s.saturated() is False        # no signal yet
    assert sat_after(950) is True        # util 0.95 >= high -> saturated
    assert sat_after(800) is True        # util 0.80 in band -> stays True
    assert sat_after(600) is False       # util 0.60 <= low  -> clears
    assert sat_after(850) is False       # util 0.85 in band -> stays False
    assert sat_after(920) is True        # util 0.92 >= high -> saturated again


def test_unreadable_sample_clears_saturation_latch() -> None:
    # Audit M1: once saturated, a fail-open (None) sample must clear the
    # latch — otherwise after the counter re-primes, an in-band utilization
    # (between low and high) re-reports saturated without re-crossing high,
    # keeping pulls throttled after a transient /sys read failure.
    clk = _Clock()
    # cumulative io_ticks reads, one per saturated() call below:
    reads = iter([0, 950, None, 2000, 2800])
    s = DiskIoSampler(
        path_provider=lambda: None, high=0.9, low=0.7,
        min_interval_s=0.0, clock=clk, io_ticks_reader=lambda: next(reads),
    )
    assert s.saturated() is False  # prime (u None)
    clk.advance(1.0)
    assert s.saturated() is True   # util 0.95 >= high -> latched
    clk.advance(1.0)
    assert s.saturated() is False  # unreadable sample -> latch cleared
    clk.advance(1.0)
    assert s.saturated() is False  # readable re-prime (delta baseline, u None)
    clk.advance(1.0)
    # util 0.80 is in-band; with the latch cleared it must stay False
    # (pre-fix this returned True from the stale latch).
    assert s.saturated() is False


# ── fail-open ─────────────────────────────────────────────────────────────────


def test_fail_open_when_reader_returns_none() -> None:
    clk = _Clock()
    s = DiskIoSampler(
        path_provider=lambda: None, min_interval_s=0.0,
        clock=clk, io_ticks_reader=lambda: None,
    )
    assert s.utilization() is None
    clk.advance(1.0)
    assert s.utilization() is None
    assert s.saturated() is False


def test_fail_open_when_reader_raises() -> None:
    def _boom() -> int:
        raise OSError("sys read failed")

    s = DiskIoSampler(
        path_provider=lambda: None, min_interval_s=0.0,
        io_ticks_reader=_boom,
    )
    assert s.utilization() is None
    assert s.saturated() is False


def test_fail_open_when_device_unresolvable() -> None:
    # No reader injected + provider yields no path → /sys resolution can't
    # bind a device → utilization None → never throttles.
    s = DiskIoSampler(path_provider=lambda: None, min_interval_s=0.0)
    assert s.utilization() is None
    assert s.saturated() is False


def test_provider_none_is_retried_not_sticky() -> None:
    # Provider returns None (daemon not up) on first call, then a path on
    # the next. The sampler must NOT permanently disable after the first
    # None — it retries resolution. We can't bind /sys deterministically in
    # CI, so assert it stays fail-open without raising across calls.
    calls = {"n": 0}

    def _provider() -> str | None:
        calls["n"] += 1
        return None if calls["n"] == 1 else "/nonexistent/path"

    s = DiskIoSampler(path_provider=_provider, min_interval_s=0.0)
    assert s.utilization() is None  # provider None → retry later
    assert s.utilization() is None  # provider path but resolve fails → fail-open
    assert calls["n"] >= 2  # was retried, not stuck after the first None


# ── validation ────────────────────────────────────────────────────────────────


def test_invalid_watermarks_raise() -> None:
    with pytest.raises(ValueError):
        DiskIoSampler(path_provider=lambda: None, high=0.5, low=0.8)  # low>high
    with pytest.raises(ValueError):
        DiskIoSampler(path_provider=lambda: None, high=1.5, low=0.5)  # >1
    with pytest.raises(ValueError):
        DiskIoSampler(path_provider=lambda: None, high=0.9, low=0.0)  # low<=0


# ── real /sys smoke (best-effort, environment-tolerant) ───────────────────────


def test_real_sys_path_smoke() -> None:
    # Against a real path the sampler must never raise and must return a
    # float in [0,1] or None (fail-open), and a bool from saturated().
    s = DiskIoSampler(path_provider=lambda: "/", min_interval_s=0.0)
    assert s.utilization() is None  # first sample
    u = s.utilization()
    assert u is None or (0.0 <= u <= 1.0)
    assert isinstance(s.saturated(), bool)
