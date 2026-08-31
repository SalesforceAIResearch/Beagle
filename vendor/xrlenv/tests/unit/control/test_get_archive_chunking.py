"""WS1 — chunked ``container_get_archive`` (node-lost oversized-reply fix).

A single ``ContainerGetArchiveReply`` carrying the whole tarball was
the prod failure that severed the heartbeat stream and marked nodes
"lost": a >128 MiB archive trips gRPC's send ceiling
(``RESOURCE_EXHAUSTED``) and a >2 GiB one trips protobuf's serialize
limit (``EncodeError``); either tears down the bidi stream the
heartbeat shares. These tests cover the three halves of the fix:

  * **node side** — ``container_get_archive`` forks to a multi-reply
    chunk stream (``_dispatch_stream_container_get_archive``);
  * **control side** — ``_send_and_collect_archive`` reassembles the
    chunks (and still accepts an old node's single whole-tarball reply);
  * **transport safety net** — ``_guard_outbound_message`` converts any
    oversized reply into a clean FAILED instead of severing the stream;
  * **client hop** — ``ContainerGetArchiveStream`` re-chunks the
    reassembled tarball to the CONSUMER so a large archive doesn't trip
    the 128 MiB unary message limit; the client reassembles and falls
    back to the unary ``ContainerGetArchive`` against an old control
    plane.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import grpc
import pytest
from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.api.constants import ARCHIVE_CHUNK_BYTES, MAX_OUTBOUND_MESSAGE_GUARD_BYTES
from xrlenv.client.transport import GrpcClientTransport
from xrlenv.control.grpc_endpoint import RemoteNodeTransport
from xrlenv.control.grpc_endpoint import _MonotonicCounter as _CtrlCounter
from xrlenv.control.rollout_endpoint import RolloutControlServicer
from xrlenv.errors import NodeCommandTimeout, XRLEnvError
from xrlenv.node.grpc_link import NodeGrpcLink, _guard_outbound_message
from xrlenv.node.grpc_link import _MonotonicCounter as _NodeCounter
from xrlenv.node.hw_probe import HardwareInfo

# ──────────────────────────────────────────────────────────────────────────────
# Fakes / helpers
# ──────────────────────────────────────────────────────────────────────────────


class _FakeAgent:
    """Minimal NodeAgent stand-in for the node-side dispatcher: only
    ``container_get_archive`` is exercised."""

    def __init__(
        self, *, tarball: bytes = b"", raise_exc: Exception | None = None,
    ) -> None:
        self.node_id = "test-node"
        self._tarball = tarball
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def container_get_archive(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> bytes:
        self.calls.append(
            {
                "rollout_id": rollout_id,
                "container_id": container_id,
                "source_path": source_path,
            },
        )
        if self._raise is not None:
            raise self._raise
        return self._tarball

    def container_get_archive_stream(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> Any:
        """Mirror the real agent's streaming get_archive: a sync method
        returning an async generator that yields the tarball (as a
        single chunk here; the node dispatcher re-slices at
        ``ARCHIVE_CHUNK_BYTES``) or raises during iteration."""
        self.calls.append(
            {
                "rollout_id": rollout_id,
                "container_id": container_id,
                "source_path": source_path,
            },
        )
        raise_exc = self._raise
        tarball = self._tarball

        async def _gen() -> Any:
            if raise_exc is not None:
                raise raise_exc
            if tarball:
                yield tarball

        return _gen()


def _make_link(agent: Any) -> NodeGrpcLink:
    # backends passed explicitly so __init__ never calls
    # agent.supported_backends().
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


# ──────────────────────────────────────────────────────────────────────────────
# Node side — _dispatch_stream_container_get_archive
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_chunks_archive_into_bounded_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tarball larger than the chunk size is split into several
    bounded ``ContainerGetArchiveChunk`` replies plus exactly one
    terminator, all sharing the command_id; reassembly is lossless."""
    monkeypatch.setattr("xrlenv.node.grpc_link.ARCHIVE_CHUNK_BYTES", 4)
    tarball = b"0123456789"  # 10 bytes → chunks of 4,4,2 + terminator
    link = _make_link(_FakeAgent(tarball=tarball))
    outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
    cmd = pb.ContainerGetArchiveCommand(
        header=pb.CommandHeader(command_id="cid-1"),
        rollout_id="r-1", container_id="c-1", source_path="/x",
    )

    await link._dispatch_stream_container_get_archive(
        cmd, cmd.header, outbox, "epoch-1", _NodeCounter(),
    )

    msgs = _drain(outbox)
    # All replies: same command_id, OK status, chunk payload.
    assert all(m.reply.command_id == "cid-1" for m in msgs)
    assert all(m.reply.status == pb.ReplyStatus.OK for m in msgs)
    assert all(
        m.reply.WhichOneof("payload") == "container_get_archive_chunk"
        for m in msgs
    )
    # Exactly one terminator, and it is the LAST message.
    terminators = [m for m in msgs if m.reply.container_get_archive_chunk.done]
    assert len(terminators) == 1
    assert msgs[-1] is terminators[0]
    # Reassembly is lossless.
    reassembled = b"".join(
        m.reply.container_get_archive_chunk.data for m in msgs
    )
    assert reassembled == tarball
    # No single chunk exceeds the (monkeypatched) chunk bound.
    assert all(
        len(m.reply.container_get_archive_chunk.data) <= 4 for m in msgs
    )
    # seq is strictly monotonic across the multi-reply stream.
    seqs = [m.seq for m in msgs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


@pytest.mark.asyncio
async def test_node_empty_archive_emits_single_terminator() -> None:
    """An empty tarball still yields exactly one terminator (done=True,
    empty data) so the consumer always sees an end-of-stream marker."""
    link = _make_link(_FakeAgent(tarball=b""))
    outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
    cmd = pb.ContainerGetArchiveCommand(
        header=pb.CommandHeader(command_id="cid-empty"),
        rollout_id="r", container_id="c", source_path="/x",
    )

    await link._dispatch_stream_container_get_archive(
        cmd, cmd.header, outbox, "epoch", _NodeCounter(),
    )

    msgs = _drain(outbox)
    assert len(msgs) == 1
    chunk = msgs[0].reply.container_get_archive_chunk
    assert chunk.done is True
    assert bytes(chunk.data) == b""


@pytest.mark.asyncio
async def test_node_archive_error_emits_single_failed_reply() -> None:
    """If the agent's get_archive raises, the dispatcher emits one
    FAILED reply (carrying the exception kind/message) and no chunks —
    the control-side collector aborts on FAILED."""
    link = _make_link(_FakeAgent(raise_exc=ValueError("boom")))
    outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
    cmd = pb.ContainerGetArchiveCommand(
        header=pb.CommandHeader(command_id="cid-err"),
        rollout_id="r", container_id="c", source_path="/x",
    )

    await link._dispatch_stream_container_get_archive(
        cmd, cmd.header, outbox, "epoch", _NodeCounter(),
    )

    msgs = _drain(outbox)
    assert len(msgs) == 1
    assert msgs[0].reply.status == pb.ReplyStatus.FAILED
    assert msgs[0].reply.error_kind == "ValueError"
    assert "boom" in msgs[0].reply.error_message


# ──────────────────────────────────────────────────────────────────────────────
# Control side — _send_and_collect_archive / container_get_archive
# ──────────────────────────────────────────────────────────────────────────────


async def _command_id_from_outbox(transport: RemoteNodeTransport) -> str:
    """Pull the get_archive ControlMsg off the outbox and return its
    command_id (the transport puts it there before awaiting chunks)."""
    for _ in range(100):
        if not transport._outbox.empty():
            sent = transport._outbox.get_nowait()
            return sent.container_get_archive.header.command_id
        await asyncio.sleep(0)
    raise AssertionError("transport never enqueued the get_archive command")


@pytest.mark.asyncio
async def test_control_reassembles_chunked_replies() -> None:
    """The transport reassembles a sequence of chunk replies (delivered
    via the reader hook) into the full tarball."""
    transport = _make_transport()
    task = asyncio.create_task(
        transport.container_get_archive(
            rollout_id="r-1", container_id="c" * 12, source_path="/x",
        ),
    )
    cid = await _command_id_from_outbox(transport)

    for part in (b"aa", b"bb", b"cc"):
        transport.deliver_reply(
            pb.CommandReply(
                command_id=cid, status=pb.ReplyStatus.OK,
                container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                    data=part, done=False,
                ),
            ),
        )
    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid, status=pb.ReplyStatus.OK,
            container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                data=b"", done=True,
            ),
        ),
    )

    out = await asyncio.wait_for(task, timeout=2.0)
    assert out == b"aabbcc"
    # Stream registration cleaned up after collection.
    assert cid not in transport._pending_streams


@pytest.mark.asyncio
async def test_control_accepts_old_single_reply_node() -> None:
    """Backward-compat: an older node answers with a single
    ``ContainerGetArchiveReply`` (whole tarball). The collector returns
    it as-is rather than waiting for chunks that never come."""
    transport = _make_transport()
    task = asyncio.create_task(
        transport.container_get_archive(
            rollout_id="r-1", container_id="c" * 12, source_path="/x",
        ),
    )
    cid = await _command_id_from_outbox(transport)

    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid, status=pb.ReplyStatus.OK,
            container_get_archive=pb.ContainerGetArchiveReply(
                tarball=b"whole-tarball",
            ),
        ),
    )

    out = await asyncio.wait_for(task, timeout=2.0)
    assert out == b"whole-tarball"


@pytest.mark.asyncio
async def test_control_failed_chunk_reply_raises() -> None:
    """A FAILED reply mid-stream surfaces as an XRLEnvError naming the
    node + the remote error."""
    transport = _make_transport()
    task = asyncio.create_task(
        transport.container_get_archive(
            rollout_id="r-1", container_id="c" * 12, source_path="/x",
        ),
    )
    cid = await _command_id_from_outbox(transport)

    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid, status=pb.ReplyStatus.FAILED,
            error_kind="NotFound", error_message="no such path",
        ),
    )

    with pytest.raises(XRLEnvError, match="NotFound"):
        await asyncio.wait_for(task, timeout=2.0)
    assert cid not in transport._pending_streams


@pytest.mark.asyncio
async def test_control_archive_timeout_records_node_health() -> None:
    """A collection that never receives a terminator times out, raising
    ``NodeCommandTimeout`` and recording the degraded-node signal the
    scheduler placement gate reads (parity with ``_send_and_wait``)."""
    transport = _make_transport()
    assert transport.seconds_since_last_command_timeout() is None
    assert transport._command_timeout_total == 0

    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)
    with pytest.raises(NodeCommandTimeout):
        await transport._send_and_collect_archive(
            msg, command_id="stuck", stream_timeout_s=0.05,
        )

    assert transport._command_timeout_total == 1
    assert transport.seconds_since_last_command_timeout() is not None
    assert "stuck" not in transport._pending_streams


@pytest.mark.asyncio
async def test_control_get_archive_uses_300s_stream_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``container_get_archive`` hands ``_send_and_collect_archive`` the
    300s ceiling (lock-step with ``container_put_archive``)."""
    transport = _make_transport()
    captured: dict[str, Any] = {}

    async def _fake_collect(
        msg: Any, command_id: str, *, stream_timeout_s: float,
    ) -> bytes:
        captured["stream_timeout_s"] = stream_timeout_s
        return b"<tar>"

    monkeypatch.setattr(
        transport, "_send_and_collect_archive", _fake_collect,
    )
    out = await transport.container_get_archive(
        rollout_id="r-1", container_id="c" * 12, source_path="/x",
    )
    assert captured["stream_timeout_s"] == 300.0
    assert out == b"<tar>"


# ──────────────────────────────────────────────────────────────────────────────
# Round-trip — node chunks → control reassembles (no live gRPC)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_round_trip_node_chunks_to_control_reassembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through the real chunk/reassembly code: the node-side
    dispatcher's replies, fed verbatim into the control-side collector,
    reproduce the original tarball."""
    monkeypatch.setattr("xrlenv.node.grpc_link.ARCHIVE_CHUNK_BYTES", 8)
    tarball = bytes(range(256)) * 5  # 1280 bytes → 160 chunks of 8
    link = _make_link(_FakeAgent(tarball=tarball))
    node_outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
    cmd = pb.ContainerGetArchiveCommand(
        header=pb.CommandHeader(command_id="rt-1"),
        rollout_id="r", container_id="c", source_path="/x",
    )
    await link._dispatch_stream_container_get_archive(
        cmd, cmd.header, node_outbox, "epoch", _NodeCounter(),
    )
    node_msgs = _drain(node_outbox)

    transport = _make_transport()
    task = asyncio.create_task(
        transport.container_get_archive(
            rollout_id="r", container_id="c" * 12, source_path="/x",
        ),
    )
    cid = await _command_id_from_outbox(transport)
    # Re-key the node's replies onto this transport's command_id and
    # feed them through the reader hook in order.
    for m in node_msgs:
        reply = pb.CommandReply()
        reply.CopyFrom(m.reply)
        reply.command_id = cid
        transport.deliver_reply(reply)

    out = await asyncio.wait_for(task, timeout=2.0)
    assert out == tarball


# ──────────────────────────────────────────────────────────────────────────────
# Transport safety net — _guard_outbound_message
# ──────────────────────────────────────────────────────────────────────────────


def test_guard_passes_small_reply_unchanged() -> None:
    msg = pb.NodeMsg(
        stream_epoch="e", seq=5,
        reply=pb.CommandReply(command_id="x", status=pb.ReplyStatus.OK),
    )
    assert _guard_outbound_message(msg, guard_bytes=1000) is msg


def test_guard_swaps_oversized_reply_for_failed() -> None:
    big = pb.NodeMsg(
        stream_epoch="e", seq=7,
        reply=pb.CommandReply(
            command_id="x", status=pb.ReplyStatus.OK,
            exec=pb.ExecReply(stdout=b"A" * 5000),
        ),
    )
    out = _guard_outbound_message(big, guard_bytes=1000)
    assert out is not big
    # Envelope coords preserved so seq stays monotonic on the wire.
    assert out.seq == 7
    assert out.stream_epoch == "e"
    assert out.reply.command_id == "x"
    assert out.reply.status == pb.ReplyStatus.FAILED
    assert out.reply.error_kind == "ReplyTooLarge"
    # The substitute is small enough to actually send.
    assert out.ByteSize() <= 1000


def test_guard_passes_oversized_non_reply() -> None:
    """Only replies are ever swapped — a heartbeat (or any non-reply)
    above the guard passes through untouched (they're never large in
    practice, and a swapped heartbeat would be meaningless)."""
    hb = pb.NodeMsg(stream_epoch="e", seq=1, heartbeat=pb.Heartbeat())
    assert _guard_outbound_message(hb, guard_bytes=1) is hb


def test_guard_threshold_is_below_hard_cap() -> None:
    """The guard leaves headroom under the gRPC hard ceiling for HTTP/2
    framing that ByteSize() doesn't account for."""
    from xrlenv.api.constants import DEFAULT_MAX_MESSAGE_BYTES

    assert MAX_OUTBOUND_MESSAGE_GUARD_BYTES < DEFAULT_MAX_MESSAGE_BYTES


# ──────────────────────────────────────────────────────────────────────────────
# Client <-> control hop — ContainerGetArchiveStream (WS1 client-facing chunking)
# ──────────────────────────────────────────────────────────────────────────────


class _FakeContext:
    """gRPC ServicerContext stand-in: on the success path the endpoint only
    reaches the owner guard, which no-ops with no token store."""

    def invocation_metadata(self) -> tuple[()]:
        return ()


class _ArchiveService:
    """Minimal RolloutService for the endpoint: returns a fixed tarball and
    answers the owner guard (None -> single-tenant, guard passes)."""

    def __init__(self, tarball: bytes) -> None:
        self._tarball = tarball
        self.calls: list[tuple[str, str, str]] = []

    def raw_session_owner(self, rollout_id: str) -> str | None:
        return None

    async def container_get_archive(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> bytes:
        self.calls.append((rollout_id, container_id, source_path))
        return self._tarball


async def _collect(agen: Any) -> list[Any]:
    return [x async for x in agen]


@pytest.mark.asyncio
async def test_endpoint_stream_chunks_and_terminates() -> None:
    """The control plane re-chunks the reassembled tarball into
    <=ARCHIVE_CHUNK_BYTES client-facing chunks + exactly one terminator, and
    the concatenation is byte-exact."""
    tarball = os.urandom(ARCHIVE_CHUNK_BYTES + 1234)  # 2 data chunks
    svc = _ArchiveService(tarball)
    servicer = RolloutControlServicer(service=svc, token_store=None)

    chunks = await _collect(servicer.ContainerGetArchiveStream(
        rpb.ContainerGetArchiveRequest(
            rollout_id="r1", container_id="c1", source_path="/x",
        ),
        _FakeContext(),
    ))

    assert [c.done for c in chunks].count(True) == 1  # exactly one terminator
    assert chunks[-1].done is True and chunks[-1].data == b""
    for c in chunks[:-1]:
        assert c.done is False
        assert 0 < len(c.data) <= ARCHIVE_CHUNK_BYTES  # bounded
    assert b"".join(c.data for c in chunks) == tarball  # byte-exact
    assert svc.calls == [("r1", "c1", "/x")]


@pytest.mark.asyncio
async def test_endpoint_stream_empty_tarball_single_terminator() -> None:
    """An empty tarball still emits exactly one terminator chunk."""
    servicer = RolloutControlServicer(
        service=_ArchiveService(b""), token_store=None,
    )
    chunks = await _collect(servicer.ContainerGetArchiveStream(
        rpb.ContainerGetArchiveRequest(
            rollout_id="r", container_id="c", source_path="/x",
        ),
        _FakeContext(),
    ))
    assert len(chunks) == 1
    assert chunks[0].done is True and chunks[0].data == b""


def _bare_transport(stub: object) -> GrpcClientTransport:
    """A ``GrpcClientTransport`` with just the attrs container_get_archive
    touches — no real gRPC channel (mirrors test_queue_status_poller)."""
    t = object.__new__(GrpcClientTransport)
    t._token = None  # type: ignore[attr-defined]
    t._stub = stub  # type: ignore[attr-defined]
    return t


@pytest.mark.asyncio
async def test_transport_reassembles_streamed_archive() -> None:
    """The client transport reassembles the streamed chunks in order."""
    payload = os.urandom(ARCHIVE_CHUNK_BYTES + 99)  # 2 chunks

    class _StreamStub:
        def ContainerGetArchiveStream(
            self, req: Any, metadata: Any = None, timeout: Any = None,
        ) -> Any:
            async def _gen() -> Any:
                for start in range(0, len(payload), ARCHIVE_CHUNK_BYTES):
                    yield rpb.ContainerGetArchiveChunkResponse(
                        data=payload[start:start + ARCHIVE_CHUNK_BYTES],
                        done=False,
                    )
                yield rpb.ContainerGetArchiveChunkResponse(data=b"", done=True)
            return _gen()

    got = await _bare_transport(_StreamStub()).container_get_archive(
        rollout_id="r", container_id="c", source_path="/x",
    )
    assert got == payload


@pytest.mark.asyncio
async def test_transport_falls_back_to_unary_on_unimplemented() -> None:
    """Against an older control plane without the stream RPC (UNIMPLEMENTED),
    the client falls back to the unary ContainerGetArchive."""
    payload = b"legacy-tarball-bytes"

    class _OldStub:
        def __init__(self) -> None:
            self.unary_calls = 0

        def ContainerGetArchiveStream(
            self, req: Any, metadata: Any = None, timeout: Any = None,
        ) -> Any:
            async def _gen() -> Any:
                raise grpc.aio.AioRpcError(
                    grpc.StatusCode.UNIMPLEMENTED,
                    grpc.aio.Metadata(), grpc.aio.Metadata(), "no stream RPC",
                )
                yield  # pragma: no cover — makes _gen an async generator
            return _gen()

        async def ContainerGetArchive(
            self, req: Any, metadata: Any = None, timeout: Any = None,
        ) -> Any:
            self.unary_calls += 1
            return rpb.ContainerGetArchiveResponse(tarball=payload)

    stub = _OldStub()
    got = await _bare_transport(stub).container_get_archive(
        rollout_id="r", container_id="c", source_path="/x",
    )
    assert got == payload
    assert stub.unary_calls == 1
