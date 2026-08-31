"""Teardown-hang fix (2026-07-07) — client-side gRPC deadlines on the
raw-container teardown / transfer RPCs.

Without a per-call ``timeout=`` these RPCs block forever when the control plane
is slow to answer (degraded sysbox node, wedged dockerd, or a CP whose per-node
command queue is backed up behind large whole-directory archive transfers).
harbor awaits every trial in one ``asyncio.TaskGroup`` with no per-trial
wall-clock, so a single hung teardown stalls a whole benchmark sweep. These
tests pin: (1) every teardown/transfer RPC carries a bounded deadline sized
above the CP's own ceilings, and (2) a client-side ``DEADLINE_EXCEEDED`` is
surfaced as a clean, retryable ``NodeCommandTimeout`` rather than a hang.
"""

from __future__ import annotations

from types import SimpleNamespace

import grpc
import pytest
from xrlenv.client import transport as transport_mod
from xrlenv.client.transport import (
    GrpcClientTransport,
    _classify_unmarked_rpc_error,
)
from xrlenv.errors import ControlPlaneLost, NodeCommandTimeout


def _bare_transport(stub: object) -> GrpcClientTransport:
    """A ``GrpcClientTransport`` with only the attributes the raw-container
    RPC methods touch — no real gRPC channel."""
    t = object.__new__(GrpcClientTransport)
    t._token = None  # type: ignore[attr-defined]
    t._stub = stub  # type: ignore[attr-defined]
    return t


class _RecordingStub:
    """Captures the ``timeout=`` each teardown/transfer RPC was called with."""

    def __init__(self) -> None:
        self.timeouts: dict[str, float | None] = {}

    async def DestroyContainer(
        self, req: object, metadata: object = None, timeout: float | None = None,
    ) -> object:
        self.timeouts["destroy"] = timeout
        return SimpleNamespace()

    async def ContainerPutArchive(
        self, req: object, metadata: object = None, timeout: float | None = None,
    ) -> object:
        # Unary fallback path (old CP). New clients prefer ContainerPutArchiveStream.
        self.timeouts["put_archive"] = timeout
        return SimpleNamespace()

    async def ContainerPutArchiveStream(
        self, request_iterator: object, metadata: object = None,
        timeout: float | None = None,
    ) -> object:
        # WS1 client-streaming upload — drain the chunk iterator + record the deadline.
        self.timeouts["put_archive"] = timeout
        async for _ in request_iterator:  # type: ignore[attr-defined]
            pass
        return SimpleNamespace()

    def ContainerGetArchiveStream(
        self, req: object, metadata: object = None, timeout: float | None = None,
    ) -> object:
        self.timeouts["get_archive"] = timeout

        async def _gen() -> object:
            yield SimpleNamespace(data=b"tarbytes", done=True)

        return _gen()

    async def ContainerExec(
        self, req: object, metadata: object = None, timeout: float | None = None,
    ) -> object:
        self.timeouts["exec"] = timeout
        return SimpleNamespace(
            stdout=b"out", stderr=b"", exit_code=0, timed_out=False,
        )

    def ContainerExecStream(
        self, req: object, metadata: object = None, timeout: float | None = None,
    ) -> object:
        self.timeouts["exec_stream"] = timeout

        async def _gen() -> object:
            yield SimpleNamespace(
                stdout=b"out", stderr=b"", done=True, exit_code=0,
                timed_out=False,
            )

        return _gen()


# ──────────────────────────────────────────────────────────────────────────────
# Deadlines are threaded onto every teardown / transfer RPC
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_destroy_container_carries_deadline() -> None:
    stub = _RecordingStub()
    t = _bare_transport(stub)
    await t.destroy_container(rollout_id="r1", container_id="c1")
    assert stub.timeouts["destroy"] == transport_mod._DESTROY_DEADLINE_S
    # Backstop must sit ABOVE the CP's own 300 s destroy ceiling.
    assert stub.timeouts["destroy"] > 300.0


@pytest.mark.asyncio
async def test_put_archive_carries_deadline() -> None:
    stub = _RecordingStub()
    t = _bare_transport(stub)
    await t.container_put_archive(
        rollout_id="r1", container_id="c1", target_dir="/logs", tarball=b"x",
    )
    assert stub.timeouts["put_archive"] == transport_mod._ARCHIVE_DEADLINE_S
    assert stub.timeouts["put_archive"] > 300.0


@pytest.mark.asyncio
async def test_get_archive_stream_carries_deadline() -> None:
    stub = _RecordingStub()
    t = _bare_transport(stub)
    data = await t.container_get_archive(
        rollout_id="r1", container_id="c1", source_path="/logs",
    )
    assert data == b"tarbytes"
    assert stub.timeouts["get_archive"] == transport_mod._ARCHIVE_DEADLINE_S
    assert stub.timeouts["get_archive"] > 300.0


@pytest.mark.asyncio
async def test_exec_carries_deadline_above_server_timeout() -> None:
    stub = _RecordingStub()
    t = _bare_transport(stub)
    await t.container_exec(
        rollout_id="r1", container_id="c1", cmd=["true"], timeout_s=120.0,
    )
    # Client deadline = server-side exec timeout + the margin.
    assert stub.timeouts["exec"] == 120.0 + transport_mod._EXEC_DEADLINE_MARGIN_S
    assert stub.timeouts["exec"] > 120.0


@pytest.mark.asyncio
async def test_exec_stream_carries_deadline_above_server_timeout() -> None:
    stub = _RecordingStub()
    t = _bare_transport(stub)
    chunks = [c async for c in t.container_exec_stream(
        rollout_id="r1", container_id="c1", cmd=["true"], timeout_s=1800.0,
    )]
    assert chunks  # the stream yielded
    assert stub.timeouts["exec_stream"] == (
        1800.0 + transport_mod._EXEC_DEADLINE_MARGIN_S
    )
    # A legitimate long (1800 s) exec is NOT cut off — the deadline is above it.
    assert stub.timeouts["exec_stream"] > 1800.0


# ──────────────────────────────────────────────────────────────────────────────
# DEADLINE_EXCEEDED → clean, retryable NodeCommandTimeout (not a hang)
# ──────────────────────────────────────────────────────────────────────────────


def test_deadline_exceeded_maps_to_node_command_timeout() -> None:
    err = SimpleNamespace(
        code=lambda: grpc.StatusCode.DEADLINE_EXCEEDED,
        details=lambda: "context deadline exceeded",
    )
    out = _classify_unmarked_rpc_error(err)  # type: ignore[arg-type]
    assert isinstance(out, NodeCommandTimeout)
    assert out.retryable is True
    assert "deadline exceeded" in str(out)


def test_unavailable_still_maps_to_control_plane_lost() -> None:
    # Regression guard: the new DEADLINE_EXCEEDED branch must not disturb the
    # existing UNAVAILABLE → ControlPlaneLost mapping.
    err = SimpleNamespace(
        code=lambda: grpc.StatusCode.UNAVAILABLE,
        details=lambda: "channel closed",
    )
    out = _classify_unmarked_rpc_error(err)  # type: ignore[arg-type]
    assert isinstance(out, ControlPlaneLost)
