"""WS1 edge cases — additional coverage for chunked archive and guard.

Supplements test_get_archive_chunking.py with cases not covered there:

 * Exact-multiple-of-chunk-size tarball: range() loop emits N full data
   chunks + exactly one terminator (no leftover slice).
 * Concurrent archive streams keyed by different command_ids don't
   cross-talk: two in-flight collects receive only their own chunks.
 * ``_send_and_collect_archive`` on a closed transport raises immediately
   without enqueuing anything.
 * ``_guard_outbound_message`` at exactly the guard boundary (==, not >):
   a message serialised to exactly ``guard_bytes`` is NOT swapped.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.control.grpc_endpoint import RemoteNodeTransport
from xrlenv.control.grpc_endpoint import _MonotonicCounter as _CtrlCounter
from xrlenv.errors import XRLEnvError
from xrlenv.node.grpc_link import NodeGrpcLink, _guard_outbound_message
from xrlenv.node.grpc_link import _MonotonicCounter as _NodeCounter
from xrlenv.node.hw_probe import HardwareInfo

# ── shared helpers ────────────────────────────────────────────────────────────


class _FakeAgent:
    def __init__(self, *, tarball: bytes = b"") -> None:
        self.node_id = "test-node"
        self._tarball = tarball

    async def container_get_archive(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> bytes:
        return self._tarball

    def container_get_archive_stream(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> Any:
        """Streaming get_archive stand-in: yields the tarball as one
        chunk (the node dispatcher re-slices at ARCHIVE_CHUNK_BYTES)."""
        tarball = self._tarball

        async def _gen() -> Any:
            if tarball:
                yield tarball

        return _gen()


def _make_link(agent: Any) -> NodeGrpcLink:
    return NodeGrpcLink(agent, control_addr="addr:1", backends=["docker"])


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _make_transport() -> RemoteNodeTransport:
    return RemoteNodeTransport(
        node_id="test-node",
        backends=["docker"],
        hardware=_hw(),
        outbox=asyncio.Queue(),
        stream_epoch="test-epoch",
        control_instance_id="ctrl-1",
        control_seq=_CtrlCounter(),
    )


def _drain(q: asyncio.Queue[pb.NodeMsg]) -> list[pb.NodeMsg]:
    out: list[pb.NodeMsg] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def _command_id_from_outbox(transport: RemoteNodeTransport) -> str:
    for _ in range(200):
        if not transport._outbox.empty():
            sent = transport._outbox.get_nowait()
            return sent.container_get_archive.header.command_id
        await asyncio.sleep(0)
    raise AssertionError("transport never enqueued the get_archive command")


# ── node-side: exact-multiple-of-chunk-size tarball ──────────────────────────


@pytest.mark.asyncio
async def test_node_exact_multiple_chunk_size_has_single_terminator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tarball whose length is an exact multiple of the chunk size must
    still emit exactly one done=True terminator (and no spurious empty
    data chunk before it). The range-loop ``for start in range(0,
    len(tarball), chunk)`` over an exact multiple yields no leftover
    slice, so all data chunks are full and the terminator is the only
    message with ``done=True``."""
    monkeypatch.setattr("xrlenv.node.grpc_link.ARCHIVE_CHUNK_BYTES", 4)
    tarball = b"12345678"  # 8 bytes == 2 * 4 — exact multiple
    link = _make_link(_FakeAgent(tarball=tarball))
    outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
    cmd = pb.ContainerGetArchiveCommand(
        header=pb.CommandHeader(command_id="exact-mult"),
        rollout_id="r", container_id="c", source_path="/x",
    )

    await link._dispatch_stream_container_get_archive(
        cmd, cmd.header, outbox, "epoch", _NodeCounter(),
    )

    msgs = _drain(outbox)
    # 2 data chunks + 1 terminator = 3 total
    assert len(msgs) == 3, f"expected 3 msgs (2 data + terminator), got {len(msgs)}"
    terminators = [m for m in msgs if m.reply.container_get_archive_chunk.done]
    assert len(terminators) == 1
    assert msgs[-1] is terminators[0], "terminator must be the last message"
    # Terminator carries no data
    assert bytes(terminators[0].reply.container_get_archive_chunk.data) == b""
    # Data chunks are each exactly chunk_size and together equal the tarball
    data_msgs = msgs[:-1]
    assert all(
        len(m.reply.container_get_archive_chunk.data) == 4 for m in data_msgs
    )
    reassembled = b"".join(
        m.reply.container_get_archive_chunk.data for m in msgs
    )
    assert reassembled == tarball


@pytest.mark.asyncio
async def test_node_single_byte_tarball(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1-byte tarball (less than chunk size) emits one data chunk + one
    terminator (not just a terminator)."""
    monkeypatch.setattr("xrlenv.node.grpc_link.ARCHIVE_CHUNK_BYTES", 4)
    tarball = b"X"
    link = _make_link(_FakeAgent(tarball=tarball))
    outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
    cmd = pb.ContainerGetArchiveCommand(
        header=pb.CommandHeader(command_id="one-byte"),
        rollout_id="r", container_id="c", source_path="/x",
    )

    await link._dispatch_stream_container_get_archive(
        cmd, cmd.header, outbox, "epoch", _NodeCounter(),
    )

    msgs = _drain(outbox)
    assert len(msgs) == 2  # 1 data chunk + 1 terminator
    assert bytes(msgs[0].reply.container_get_archive_chunk.data) == b"X"
    assert msgs[0].reply.container_get_archive_chunk.done is False
    assert msgs[1].reply.container_get_archive_chunk.done is True


# ── control-side: concurrent streams don't cross-talk ────────────────────────


@pytest.mark.asyncio
async def test_two_concurrent_archive_streams_no_crosstalk() -> None:
    """Two simultaneously in-flight container_get_archive calls on the
    same transport use distinct _pending_streams queue keys (command_ids).
    Replies delivered for one stream must NOT appear in the other."""
    transport = _make_transport()

    task_a = asyncio.create_task(
        transport.container_get_archive(
            rollout_id="r-A", container_id="a" * 12, source_path="/a",
        ),
    )
    task_b = asyncio.create_task(
        transport.container_get_archive(
            rollout_id="r-B", container_id="b" * 12, source_path="/b",
        ),
    )

    # Drain both command_ids off the outbox.
    cids: list[str] = []
    for _ in range(400):
        if not transport._outbox.empty():
            cids.append(
                transport._outbox.get_nowait().container_get_archive.header.command_id
            )
        if len(cids) == 2:
            break
        await asyncio.sleep(0)
    assert len(cids) == 2, "expected two outbox entries for two streams"
    cid_a, cid_b = cids

    # Deliver only stream-A's reply (one chunk + terminator).
    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid_a, status=pb.ReplyStatus.OK,
            container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                data=b"payload-A", done=False,
            ),
        ),
    )
    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid_a, status=pb.ReplyStatus.OK,
            container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                data=b"", done=True,
            ),
        ),
    )

    result_a = await asyncio.wait_for(task_a, timeout=2.0)
    assert result_a == b"payload-A"

    # Stream B is still pending — its task should still be running.
    assert not task_b.done(), "stream-B must still be pending (no replies delivered)"
    assert cid_b in transport._pending_streams

    # Clean up by delivering stream-B's terminator.
    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid_b, status=pb.ReplyStatus.OK,
            container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                data=b"payload-B", done=False,
            ),
        ),
    )
    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid_b, status=pb.ReplyStatus.OK,
            container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                data=b"", done=True,
            ),
        ),
    )
    result_b = await asyncio.wait_for(task_b, timeout=2.0)
    assert result_b == b"payload-B"
    assert cid_a not in transport._pending_streams
    assert cid_b not in transport._pending_streams


# ── control-side: closed transport raises immediately ─────────────────────────


@pytest.mark.asyncio
async def test_send_and_collect_archive_raises_on_closed_transport() -> None:
    """A closed transport raises XRLEnvError immediately without any
    attempt to enqueue a command or await chunks."""
    transport = _make_transport()
    transport.close()

    msg = pb.ControlMsg(stream_epoch="e", seq=1)
    with pytest.raises(XRLEnvError, match="disconnected"):
        await transport._send_and_collect_archive(
            msg, command_id="c-closed", stream_timeout_s=5.0,
        )
    # Nothing went on the outbox.
    assert transport._outbox.empty()
    assert "c-closed" not in transport._pending_streams


# ── guard boundary conditions ─────────────────────────────────────────────────


def test_guard_at_exactly_boundary_is_not_swapped() -> None:
    """``_guard_outbound_message`` swaps when ``ByteSize() > guard_bytes``
    (strict greater-than). A message serialised to exactly ``guard_bytes``
    is within budget and must pass through unchanged."""
    # Build a reply whose ByteSize == chosen guard, then set the guard
    # to that exact size. Easiest: build a big payload, measure it, use
    # that size as the guard.
    big = pb.NodeMsg(
        stream_epoch="e", seq=3,
        reply=pb.CommandReply(
            command_id="x", status=pb.ReplyStatus.OK,
            exec=pb.ExecReply(stdout=b"A" * 100),
        ),
    )
    exact_size = big.ByteSize()
    # At exactly the boundary the message passes through.
    out = _guard_outbound_message(big, guard_bytes=exact_size)
    assert out is big  # same object — not swapped

    # One byte above should swap.
    out_over = _guard_outbound_message(big, guard_bytes=exact_size - 1)
    assert out_over is not big
    assert out_over.reply.status == pb.ReplyStatus.FAILED


def test_guard_preserves_seq_and_epoch_on_swap() -> None:
    """When a reply is swapped for FAILED, the outer NodeMsg envelope's
    ``seq`` and ``stream_epoch`` are copied verbatim so the receiver's
    monotonic-seq tracker stays valid."""
    big = pb.NodeMsg(
        stream_epoch="epoch-xyz", seq=42,
        reply=pb.CommandReply(
            command_id="oversized-cmd", status=pb.ReplyStatus.OK,
            exec=pb.ExecReply(stdout=b"B" * 5000),
        ),
    )
    out = _guard_outbound_message(big, guard_bytes=10)
    assert out.seq == 42
    assert out.stream_epoch == "epoch-xyz"
    assert out.reply.command_id == "oversized-cmd"
    assert out.reply.status == pb.ReplyStatus.FAILED
    assert out.reply.error_kind == "ReplyTooLarge"
