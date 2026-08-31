"""Tests for the per-rollout idle-TTL watcher (Slice 4 / spec 02)."""

from __future__ import annotations

import asyncio

import pytest
from xrlenv.control.idle_ttl import IdleTtlWatcher

# ──────────────────────────────────────────────────────────────────────────────
# Watcher fundamentals
# ──────────────────────────────────────────────────────────────────────────────


async def test_watcher_fires_after_idle_ttl_expires() -> None:
    fired: list[tuple[str, str]] = []

    async def on_idle(rid: str, reason: str) -> None:
        fired.append((rid, reason))

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.watch("r-1", idle_ttl_s=0.05)
    await asyncio.sleep(0.15)
    assert fired == [("r-1", "idle_ttl")]
    assert w.has_watcher("r-1") is False


async def test_touch_resets_the_clock() -> None:
    fired: list[str] = []

    async def on_idle(rid: str, reason: str) -> None:
        fired.append(rid)

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.watch("r-2", idle_ttl_s=0.1)
    # Touch every 30ms for 250ms — never lets the 100ms idle window elapse.
    for _ in range(8):
        await asyncio.sleep(0.03)
        w.touch("r-2")
    assert fired == []
    assert w.has_watcher("r-2") is True
    # Now stop touching and let it fire.
    await asyncio.sleep(0.2)
    assert fired == ["r-2"]


async def test_cancel_prevents_fire() -> None:
    fired: list[str] = []

    async def on_idle(rid: str, reason: str) -> None:
        fired.append(rid)

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.watch("r-3", idle_ttl_s=0.5)
    await asyncio.sleep(0.05)
    w.cancel("r-3")
    await asyncio.sleep(0.6)
    assert fired == []
    assert w.has_watcher("r-3") is False


async def test_watch_is_idempotent() -> None:
    fired: list[str] = []

    async def on_idle(rid: str, reason: str) -> None:
        fired.append(rid)

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.watch("r-4", idle_ttl_s=0.05)
    w.watch("r-4", idle_ttl_s=10.0)  # second call ignored — original timer wins
    await asyncio.sleep(0.15)
    assert fired == ["r-4"]


async def test_touch_unknown_rollout_is_noop() -> None:
    async def on_idle(rid: str, reason: str) -> None:
        return None

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.touch("never-watched")  # must not raise


async def test_cancel_unknown_rollout_is_noop() -> None:
    async def on_idle(rid: str, reason: str) -> None:
        return None

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.cancel("never-watched")  # must not raise


async def test_shutdown_cancels_all_watchers() -> None:
    fired: list[str] = []

    async def on_idle(rid: str, reason: str) -> None:
        fired.append(rid)

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.watch("a", idle_ttl_s=0.5)
    w.watch("b", idle_ttl_s=0.5)
    await w.shutdown()
    await asyncio.sleep(0.6)
    assert fired == []
    assert w.has_watcher("a") is False
    assert w.has_watcher("b") is False


# ──────────────────────────────────────────────────────────────────────────────
# Failure isolation
# ──────────────────────────────────────────────────────────────────────────────


async def test_callback_exception_does_not_crash_watcher() -> None:
    """A buggy callback shouldn't break subsequent watcher invocations."""
    fired: list[str] = []

    async def on_idle(rid: str, reason: str) -> None:
        fired.append(rid)
        raise RuntimeError("callback exploded")

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.watch("r-bad", idle_ttl_s=0.05)
    await asyncio.sleep(0.15)
    assert fired == ["r-bad"]
    assert w.has_watcher("r-bad") is False
    # Watcher table is consistent — register another and confirm it works.
    w.watch("r-good", idle_ttl_s=0.05)
    await asyncio.sleep(0.15)
    assert "r-good" in fired


@pytest.mark.parametrize("ttl_s", [0.0, 0.001])
async def test_zero_or_tiny_ttl_fires_immediately(ttl_s: float) -> None:
    fired = asyncio.Event()

    async def on_idle(rid: str, reason: str) -> None:
        fired.set()

    w = IdleTtlWatcher(on_idle_ttl=on_idle)
    w.watch("r", idle_ttl_s=ttl_s)
    await asyncio.wait_for(fired.wait(), timeout=0.5)
