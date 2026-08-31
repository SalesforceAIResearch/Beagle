"""Stage 2 — consumer-side admission-queue poller.

`GrpcClientTransport._poll_queue_status` narrates a blocked acquire's
admission-queue position so a queued request is visible instead of a
silent wait. It only logs — it never raises into the acquire.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest
from xrlenv.client import transport as transport_mod
from xrlenv.client.transport import GrpcClientTransport


def _bare_transport(stub: object) -> GrpcClientTransport:
    """A ``GrpcClientTransport`` with only the attributes
    ``_poll_queue_status`` touches — no real gRPC channel."""
    t = object.__new__(GrpcClientTransport)
    t._token = None  # type: ignore[attr-defined]
    t._stub = stub  # type: ignore[attr-defined]
    return t


class _QueuedStub:
    def __init__(self, *, state: str) -> None:
        self._state = state
        self.calls = 0

    async def QueueStatus(
        self, req: object, metadata: object = None,
    ) -> object:
        self.calls += 1
        return SimpleNamespace(position=4, queue_depth=9, state=self._state)


async def _run_poller_briefly(transport: GrpcClientTransport) -> None:
    task = asyncio.create_task(transport._poll_queue_status("r-1"))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if transport._stub.calls >= 1:  # type: ignore[attr-defined]
            break
    await asyncio.sleep(0.02)  # let the post-poll log branch run
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_poller_logs_live_queue_position(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A queued request gets a live 'position N of M' log line."""
    monkeypatch.setattr(transport_mod, "_QUEUE_POLL_INTERVAL_S", 0.01)
    transport = _bare_transport(_QueuedStub(state="queued"))
    with caplog.at_level("INFO", logger="xrlenv.client.transport"):
        await _run_poller_briefly(transport)
    assert any(
        "position 4 of 9" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_poller_silent_when_not_queued(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A request that is not queued (fast-path placement) is polled but
    produces no 'admission queue' log line."""
    monkeypatch.setattr(transport_mod, "_QUEUE_POLL_INTERVAL_S", 0.01)
    transport = _bare_transport(_QueuedStub(state="not_in_queue"))
    with caplog.at_level("INFO", logger="xrlenv.client.transport"):
        await _run_poller_briefly(transport)
    assert transport._stub.calls >= 1  # type: ignore[attr-defined]
    assert not any(
        "admission queue" in r.getMessage() for r in caplog.records
    )
