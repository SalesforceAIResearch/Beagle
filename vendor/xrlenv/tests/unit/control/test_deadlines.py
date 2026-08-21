"""Tests for the DeadlineWatcher (spec 02 §"Deadline semantics")."""

from __future__ import annotations

import asyncio

import pytest
from xrlenv.control.deadlines import DeadlineWatcher

# ──────────────────────────────────────────────────────────────────────────────
# Fundamentals: arm, fire, cancel
# ──────────────────────────────────────────────────────────────────────────────


async def test_watcher_fires_after_hard_s_expires() -> None:
    fired: list[tuple[str, str]] = []

    async def on_truncate(rid: str, reason: str) -> None:
        fired.append((rid, reason))

    w = DeadlineWatcher(on_truncate)
    w.watch("r-1", hard_s=0.05)
    await asyncio.sleep(0.15)
    assert fired == [("r-1", "hard_deadline")]
    # Watcher should have cleaned up after firing.
    assert w.has_watcher("r-1") is False
    assert w.event_for("r-1") is None


async def test_event_set_when_watcher_fires() -> None:
    async def on_truncate(rid: str, reason: str) -> None:
        return None

    w = DeadlineWatcher(on_truncate)
    w.watch("r-2", hard_s=0.05)
    event = w.event_for("r-2")
    assert event is not None
    assert event.is_set() is False
    await asyncio.sleep(0.15)
    # After fire, the event lookup returns None (cleaned up), but the local
    # reference is set.
    assert event.is_set() is True


async def test_cancel_prevents_fire() -> None:
    fired: list[str] = []

    async def on_truncate(rid: str, reason: str) -> None:
        fired.append(rid)

    w = DeadlineWatcher(on_truncate)
    w.watch("r-3", hard_s=0.5)
    await asyncio.sleep(0.05)
    w.cancel("r-3")
    await asyncio.sleep(0.6)
    assert fired == []
    assert w.has_watcher("r-3") is False


async def test_cancel_unknown_rollout_is_noop() -> None:
    async def on_truncate(rid: str, reason: str) -> None:
        return None

    w = DeadlineWatcher(on_truncate)
    w.cancel("never-watched")  # must not raise


async def test_watch_is_idempotent() -> None:
    fired: list[str] = []

    async def on_truncate(rid: str, reason: str) -> None:
        fired.append(rid)

    w = DeadlineWatcher(on_truncate)
    w.watch("r-4", hard_s=0.05)
    w.watch("r-4", hard_s=10.0)  # second call ignored (still 0.05)
    await asyncio.sleep(0.15)
    assert fired == ["r-4"]


# ──────────────────────────────────────────────────────────────────────────────
# Multi-rollout + shutdown
# ──────────────────────────────────────────────────────────────────────────────


async def test_independent_rollouts_fire_independently() -> None:
    fired: list[tuple[str, str]] = []

    async def on_truncate(rid: str, reason: str) -> None:
        fired.append((rid, reason))

    w = DeadlineWatcher(on_truncate)
    w.watch("a", hard_s=0.05)
    w.watch("b", hard_s=0.15)
    await asyncio.sleep(0.25)
    assert [r for r, _ in fired] == ["a", "b"]


async def test_shutdown_cancels_all_watchers() -> None:
    fired: list[str] = []

    async def on_truncate(rid: str, reason: str) -> None:
        fired.append(rid)

    w = DeadlineWatcher(on_truncate)
    w.watch("a", hard_s=0.5)
    w.watch("b", hard_s=0.5)
    await w.shutdown()
    await asyncio.sleep(0.6)
    assert fired == []
    assert w.has_watcher("a") is False
    assert w.has_watcher("b") is False


# ──────────────────────────────────────────────────────────────────────────────
# Truncate-callback failure isolation
# ──────────────────────────────────────────────────────────────────────────────


async def test_truncate_callback_exception_does_not_crash_loop() -> None:
    fired: list[str] = []

    async def on_truncate(rid: str, reason: str) -> None:
        fired.append(rid)
        raise RuntimeError("callback boom")

    w = DeadlineWatcher(on_truncate)
    w.watch("a", hard_s=0.05)
    await asyncio.sleep(0.15)
    # The watcher fired and the exception was swallowed by the watcher's
    # finally clause; the watcher table is still consistent.
    assert fired == ["a"]
    assert w.has_watcher("a") is False


# ──────────────────────────────────────────────────────────────────────────────
# event_for / has_watcher API surface
# ──────────────────────────────────────────────────────────────────────────────


def test_event_for_unknown_returns_none() -> None:
    async def on_truncate(rid: str, reason: str) -> None:
        return None

    w = DeadlineWatcher(on_truncate)
    assert w.event_for("never") is None
    assert w.has_watcher("never") is False


@pytest.mark.parametrize("hard_s", [0.0, 0.001])
async def test_zero_or_tiny_deadline_fires_immediately(hard_s: float) -> None:
    fired = asyncio.Event()

    async def on_truncate(rid: str, reason: str) -> None:
        fired.set()

    w = DeadlineWatcher(on_truncate)
    w.watch("r", hard_s=hard_s)
    await asyncio.wait_for(fired.wait(), timeout=0.5)
