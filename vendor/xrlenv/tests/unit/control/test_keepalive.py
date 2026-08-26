"""Unit tests for xrlenv.control.keepalive.ControlKeepaliveLoop (2026-08-21)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from xrlenv.control.keepalive import ControlKeepaliveLoop

# ──────────────────────────────────────────────────────────────────────────────
# Fake registry helpers
# ──────────────────────────────────────────────────────────────────────────────


class _FakeRegistry:
    """Minimal duck-typed NodeRegistry for keepalive tests."""

    def __init__(self, nodes: dict[str, object]) -> None:
        self._nodes = nodes

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def get(self, node_id: str) -> object | None:
        return self._nodes.get(node_id)


def _transport_with_keepalive() -> MagicMock:
    t = MagicMock()
    # Ensure send_keepalive is callable
    t.send_keepalive = MagicMock(return_value=None)
    return t


def _transport_without_keepalive() -> object:
    """A plain object that has NO send_keepalive attribute."""
    return SimpleNamespace(node_id="no-keepalive-node")


# ──────────────────────────────────────────────────────────────────────────────
# _beat_once
# ──────────────────────────────────────────────────────────────────────────────


def test_beat_once_calls_send_keepalive_on_each_transport() -> None:
    t1 = _transport_with_keepalive()
    t2 = _transport_with_keepalive()
    registry = _FakeRegistry({"n1": t1, "n2": t2})
    loop = ControlKeepaliveLoop(registry, interval_s=5.0)
    loop._beat_once()
    t1.send_keepalive.assert_called_once_with()
    t2.send_keepalive.assert_called_once_with()


def test_beat_once_skips_transport_without_send_keepalive() -> None:
    """A transport lacking send_keepalive must not cause an AttributeError."""
    t_good = _transport_with_keepalive()
    t_bad = _transport_without_keepalive()
    registry = _FakeRegistry({"n1": t_good, "n2": t_bad})
    loop = ControlKeepaliveLoop(registry, interval_s=5.0)
    loop._beat_once()  # must not raise
    t_good.send_keepalive.assert_called_once_with()


def test_beat_once_all_transports_lack_keepalive_no_crash() -> None:
    """All-missing send_keepalive scenario is silently safe."""
    t_bad1 = _transport_without_keepalive()
    t_bad2 = _transport_without_keepalive()
    registry = _FakeRegistry({"n1": t_bad1, "n2": t_bad2})
    loop = ControlKeepaliveLoop(registry, interval_s=5.0)
    loop._beat_once()  # must not raise


def test_beat_once_empty_registry_no_crash() -> None:
    registry = _FakeRegistry({})
    loop = ControlKeepaliveLoop(registry, interval_s=5.0)
    loop._beat_once()  # must not raise


def test_beat_once_raising_send_keepalive_is_swallowed() -> None:
    """A transport whose send_keepalive raises must not propagate; other
    transports in the same pass still get their beat."""
    t_bad = MagicMock()
    t_bad.send_keepalive = MagicMock(side_effect=OSError("network gone"))
    t_good = _transport_with_keepalive()
    registry = _FakeRegistry({"n_bad": t_bad, "n_good": t_good})
    loop = ControlKeepaliveLoop(registry, interval_s=5.0)
    loop._beat_once()  # must not raise
    t_good.send_keepalive.assert_called_once_with()


def test_beat_once_raising_first_does_not_skip_rest() -> None:
    """Order-independent: even if n1 raises, n2 still gets beaten."""
    t1 = MagicMock()
    t1.send_keepalive = MagicMock(side_effect=RuntimeError("boom"))
    t2 = _transport_with_keepalive()
    # Dict preserves insertion order in CPython 3.7+.
    registry = _FakeRegistry({"n1": t1, "n2": t2})
    loop = ControlKeepaliveLoop(registry, interval_s=5.0)
    loop._beat_once()
    t2.send_keepalive.assert_called_once_with()


# ──────────────────────────────────────────────────────────────────────────────
# start / shutdown lifecycle
# ──────────────────────────────────────────────────────────────────────────────


async def test_start_creates_background_task() -> None:
    registry = _FakeRegistry({})
    loop = ControlKeepaliveLoop(registry, interval_s=60.0)
    await loop.start()
    assert loop._task is not None
    assert not loop._task.done()
    await loop.shutdown()


async def test_start_is_idempotent() -> None:
    registry = _FakeRegistry({})
    loop = ControlKeepaliveLoop(registry, interval_s=60.0)
    await loop.start()
    first_task = loop._task
    await loop.start()
    assert loop._task is first_task
    await loop.shutdown()


async def test_shutdown_cancels_task_and_clears_reference() -> None:
    registry = _FakeRegistry({})
    loop = ControlKeepaliveLoop(registry, interval_s=60.0)
    await loop.start()
    task = loop._task
    assert task is not None
    await loop.shutdown()
    assert loop._task is None
    assert task.done()


async def test_shutdown_before_start_is_safe() -> None:
    registry = _FakeRegistry({})
    loop = ControlKeepaliveLoop(registry, interval_s=60.0)
    await loop.shutdown()  # must not raise


async def test_double_shutdown_is_idempotent() -> None:
    registry = _FakeRegistry({})
    loop = ControlKeepaliveLoop(registry, interval_s=60.0)
    await loop.start()
    await loop.shutdown()
    await loop.shutdown()  # second call must not raise
