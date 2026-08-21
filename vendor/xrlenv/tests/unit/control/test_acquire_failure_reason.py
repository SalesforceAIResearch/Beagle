"""Unit tests for ``_acquire_cancel_reason`` — the factual ``error``
string a cleanly-cancelled raw acquire records (prod "CancelledError with
no reason" fix).

A bare ``asyncio.CancelledError`` stringifies to ``""``, so the old
``f"{type(exc).__name__}: {exc}"`` recorded the useless ``"CancelledError: "``.
The helper instead records the teardown facts (how long it waited, whether
it had been placed) plus the canceller's own message if one was set. It
does NOT speculate about *who* issued the cancel (a CancelledError carries
no origin). The acquire-path status logic (cancel-success -> cancelled,
cancel-teardown-failure -> failed) is exercised in
``test_raw_container_coordinator.py``.
"""

from __future__ import annotations

import asyncio

from xrlenv.control.raw_container_service import _acquire_cancel_reason


def test_cancelled_while_queued_is_descriptive() -> None:
    reason = _acquire_cancel_reason(
        asyncio.CancelledError(), queue_wait_s=43.2, placed=False,
    )
    assert "cancelled" in reason.lower()
    assert "never placed" in reason
    assert "43.2s" in reason
    # Not the useless bare form.
    assert reason.strip() not in ("CancelledError:", "CancelledError")


def test_cancelled_after_placement_is_descriptive() -> None:
    reason = _acquire_cancel_reason(
        asyncio.CancelledError(), queue_wait_s=0.4, placed=True,
    )
    assert "after placement" in reason
    assert "0.4s" in reason


def test_reason_does_not_speculate_about_who_issued_the_cancel() -> None:
    """The reason states the outcome (cancelled, unwound cleanly), not a
    guess about the origin — that was the 'sloppy wording' complaint."""
    reason = _acquire_cancel_reason(
        asyncio.CancelledError(), queue_wait_s=1.0, placed=False,
    )
    lowered = reason.lower()
    assert "unwound cleanly" in lowered
    # No speculation about the caller.
    assert "abandoned" not in lowered
    assert "disconnect" not in lowered


def test_cancel_message_is_surfaced_verbatim() -> None:
    """If the canceller supplied a reason (task.cancel("...") /
    deadline-watcher), record it verbatim — that IS the origin, when known."""
    reason = _acquire_cancel_reason(
        asyncio.CancelledError("session deadline 600s exceeded"),
        queue_wait_s=0.0, placed=True,
    )
    assert "session deadline 600s exceeded" in reason
