"""Unit tests for NodeGrpcLink._server_liveness_loop (2026-08-21).

The loop watches for control-plane silence and redials when the CP has gone
quiet past the server_silence_deadline_s. We test it in isolation by binding
the coroutine via types.MethodType onto a minimal SimpleNamespace so we don't
have to construct a full NodeGrpcLink with a live gRPC channel.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from xrlenv.node.grpc_link import NodeGrpcLink

# ──────────────────────────────────────────────────────────────────────────────
# Minimal fake self that satisfies _server_liveness_loop
# ──────────────────────────────────────────────────────────────────────────────


def _make_link_ns(
    deadline_s: float,
    stop_event: asyncio.Event | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace with the minimal attributes consumed by
    _server_liveness_loop:
      - _server_silence_deadline_s: float
      - _stop: asyncio.Event
      - _agent.node_id: str
      - _control_addr: str
    """
    agent = SimpleNamespace(node_id="test-node")
    return SimpleNamespace(
        _server_silence_deadline_s=deadline_s,
        _stop=stop_event if stop_event is not None else asyncio.Event(),
        _agent=agent,
        _control_addr="localhost:9001",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stale last_ctrl_at: loop must cancel call and set redial
# ──────────────────────────────────────────────────────────────────────────────


async def test_stale_last_ctrl_at_calls_cancel_and_sets_redial() -> None:
    """When last_ctrl_at is older than the deadline, the loop calls
    call.cancel() exactly once and sets redial[0] = True, then returns.

    Use deadline=0.15 s so check_interval = max(1.0, 0.15/3) = max(1.0, 0.05)
    = 1.0 s. That would be slow. Instead set last_ctrl_at to a timestamp
    well in the past so the first check (after check_interval sleep) fires
    immediately.

    Actually max(1.0, deadline/3) means we need deadline ≥ 3.0 to get a
    sub-1-second check_interval. Use deadline=0.15 gives check_interval=1.0 s.

    Work-around: pass a namespace whose deadline makes check_interval ~0.06 s.
    check_interval = max(1.0, deadline/3) → need deadline > 3 for <1s checks.
    Use deadline=0.18 → check_interval=max(1.0, 0.06)=1.0. Still 1 s.

    The simplest approach: use a deadline of 0.18 s (check_interval=1.0 s) but
    set last_ctrl_at to be very stale. The first check after the sleep fires
    immediately after 1 s. That means the test takes ~1 s. Acceptable for a CI
    test.

    To keep it faster, monkey-patch check_interval by injecting a custom
    deadline that gives a smaller check_interval. But max(1.0, x) clamps at 1.0
    for any x < 1.0. So the minimum check_interval is 1.0 s by design.

    Alternative: run the coroutine in isolation and force the sleep to complete
    instantly by patching asyncio.sleep. This is cleaner.
    """
    deadline = 0.15
    ns = _make_link_ns(deadline_s=deadline)
    call = MagicMock()
    call.cancel = MagicMock()
    # Make last_ctrl_at stale: well beyond the deadline.
    last_ctrl_at = [time.monotonic() - deadline - 1.0]
    redial = [False]

    bound = types.MethodType(NodeGrpcLink._server_liveness_loop, ns)

    # Patch asyncio.sleep to be instant so the check_interval wait is skipped.
    original_sleep = asyncio.sleep

    async def _instant_sleep(delay: float) -> None:
        await original_sleep(0)  # yield to the event loop without real delay

    import unittest.mock
    with unittest.mock.patch("asyncio.sleep", side_effect=_instant_sleep):
        await asyncio.wait_for(bound(call, last_ctrl_at, redial), timeout=2.0)

    call.cancel.assert_called_once()
    assert redial[0] is True


async def test_stale_loop_returns_after_cancel() -> None:
    """The loop must return (not keep looping) after calling cancel."""
    deadline = 0.15
    ns = _make_link_ns(deadline_s=deadline)
    call = MagicMock()
    call.cancel = MagicMock()
    last_ctrl_at = [time.monotonic() - deadline - 1.0]
    redial = [False]

    bound = types.MethodType(NodeGrpcLink._server_liveness_loop, ns)

    import unittest.mock
    original_sleep = asyncio.sleep

    async def _instant_sleep(delay: float) -> None:
        await original_sleep(0)

    with unittest.mock.patch("asyncio.sleep", side_effect=_instant_sleep):
        # If the loop continued after cancel the wait_for would time out.
        await asyncio.wait_for(bound(call, last_ctrl_at, redial), timeout=2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Fresh last_ctrl_at: loop must NOT cancel within a short window
# ──────────────────────────────────────────────────────────────────────────────


async def test_fresh_last_ctrl_at_does_not_cancel() -> None:
    """When last_ctrl_at is fresh (well within the deadline), the loop must
    NOT call call.cancel(). We run one check cycle with instant sleep and then
    cancel the task."""
    deadline = 60.0  # huge — last_ctrl_at will never expire
    ns = _make_link_ns(deadline_s=deadline)
    call = MagicMock()
    call.cancel = MagicMock()
    last_ctrl_at = [time.monotonic()]  # fresh
    redial = [False]

    bound = types.MethodType(NodeGrpcLink._server_liveness_loop, ns)

    import unittest.mock
    original_sleep = asyncio.sleep
    checks_done = [0]

    async def _instant_then_stop(delay: float) -> None:
        checks_done[0] += 1
        if checks_done[0] >= 2:
            # After two check cycles, abort via stop event.
            ns._stop.set()
        await original_sleep(0)

    # wait_for with a generous timeout; the loop exits via _stop.set()
    with (
        unittest.mock.patch("asyncio.sleep", side_effect=_instant_then_stop),
        contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError),
    ):
        await asyncio.wait_for(bound(call, last_ctrl_at, redial), timeout=2.0)

    call.cancel.assert_not_called()
    assert redial[0] is False


# ──────────────────────────────────────────────────────────────────────────────
# Stop event: loop exits cleanly without cancelling call
# ──────────────────────────────────────────────────────────────────────────────


async def test_stop_event_exits_loop_without_cancelling_call() -> None:
    """Setting _stop makes the loop exit on the next iteration without
    calling call.cancel()."""
    deadline = 0.15
    stop = asyncio.Event()
    ns = _make_link_ns(deadline_s=deadline, stop_event=stop)
    call = MagicMock()
    call.cancel = MagicMock()
    last_ctrl_at = [time.monotonic()]  # fresh
    redial = [False]

    bound = types.MethodType(NodeGrpcLink._server_liveness_loop, ns)

    import unittest.mock
    original_sleep = asyncio.sleep

    async def _instant_then_stop(delay: float) -> None:
        stop.set()  # set stop before returning from sleep
        await original_sleep(0)

    with unittest.mock.patch("asyncio.sleep", side_effect=_instant_then_stop):
        await asyncio.wait_for(bound(call, last_ctrl_at, redial), timeout=2.0)

    call.cancel.assert_not_called()
    assert redial[0] is False


# ──────────────────────────────────────────────────────────────────────────────
# CancelledError: loop exits gracefully
# ──────────────────────────────────────────────────────────────────────────────


async def test_cancelled_error_exits_loop_gracefully() -> None:
    """Cancelling the task from outside causes the loop to exit via the
    CancelledError handler. call.cancel must never have been called."""
    deadline = 60.0
    ns = _make_link_ns(deadline_s=deadline)
    call = MagicMock()
    call.cancel = MagicMock()
    last_ctrl_at = [time.monotonic()]
    redial = [False]

    bound = types.MethodType(NodeGrpcLink._server_liveness_loop, ns)
    task = asyncio.create_task(bound(call, last_ctrl_at, redial))
    await asyncio.sleep(0.01)
    task.cancel()
    # The coroutine catches CancelledError and returns normally;
    # suppress any residual CancelledError from the task wrapper.
    with contextlib.suppress(asyncio.CancelledError):
        await task

    call.cancel.assert_not_called()
    assert redial[0] is False
