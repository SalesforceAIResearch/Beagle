"""WS1 — chunked ``container_put_archive`` (the upload twin of the get_archive
chunking). Validates, with no cluster and no real gRPC channel:

- client-side slicing: byte-for-byte reassembly, routing metadata on the first frame
  only, exactly one ``done`` terminator, and that a >1-chunk tarball is actually sliced;
- the empty-tarball terminator;
- the UNIMPLEMENTED -> unary fallback (new client ↔ old control plane);
- node-side accumulation: frames reassemble in order into one ``put_archive`` + one OK
  reply (and a FAILED reply when the agent raises);
- the RESOURCE_EXHAUSTED "message larger than max" -> ArchiveTooLarge mapping (so an
  oversize payload is not mislabelled CapacityExhausted and does not burn infra retries).
"""
from __future__ import annotations

from types import SimpleNamespace

import grpc
import pytest
from xrlenv.api._pb2 import node_control_pb2 as npb
from xrlenv.api.constants import ARCHIVE_CHUNK_BYTES
from xrlenv.client.transport import (
    GrpcClientTransport,
    _classify_unmarked_rpc_error,
)
from xrlenv.errors import ArchiveTooLarge, CapacityExhausted
from xrlenv.node.grpc_link import NodeGrpcLink, _MonotonicCounter


def _bare_transport(stub: object) -> GrpcClientTransport:
    t = object.__new__(GrpcClientTransport)
    t._token = None  # type: ignore[attr-defined]
    t._stub = stub  # type: ignore[attr-defined]
    return t


# ── client-side slicing ───────────────────────────────────────────────────────


class _ReassemblingStub:
    """A ContainerPutArchiveStream that reassembles the client's chunk stream."""

    def __init__(self) -> None:
        self.reassembled: bytes | None = None
        self.meta: tuple[str, str, str] | None = None
        self.n_frames = 0
        self.n_done = 0
        self.meta_frames = 0  # frames that carried non-empty routing metadata

    async def ContainerPutArchiveStream(
        self, request_iterator: object, metadata: object = None,
        timeout: float | None = None,
    ) -> object:
        parts: list[bytes] = []
        async for ch in request_iterator:  # type: ignore[attr-defined]
            self.n_frames += 1
            if ch.rollout_id or ch.container_id or ch.target_dir:
                self.meta_frames += 1
                self.meta = (ch.rollout_id, ch.container_id, ch.target_dir)
            if ch.data:
                parts.append(bytes(ch.data))
            if ch.done:
                self.n_done += 1
        self.reassembled = b"".join(parts)
        return SimpleNamespace()


@pytest.mark.asyncio
async def test_client_chunks_large_tarball_roundtrip() -> None:
    stub = _ReassemblingStub()
    t = _bare_transport(stub)
    # > 2 chunks so slicing is genuinely exercised.
    tarball = bytes((i * 7) % 256 for i in range(ARCHIVE_CHUNK_BYTES * 2 + 12345))
    await t.container_put_archive(
        rollout_id="r1", container_id="c1", target_dir="/app", tarball=tarball,
    )
    assert stub.reassembled == tarball          # byte-for-byte lossless
    assert stub.meta == ("r1", "c1", "/app")    # metadata present
    assert stub.meta_frames == 1                 # metadata on the FIRST frame only
    assert stub.n_done == 1                      # exactly one terminator
    assert stub.n_frames >= 3                    # actually sliced into >2 frames


@pytest.mark.asyncio
async def test_client_empty_tarball_sends_one_terminator_with_meta() -> None:
    stub = _ReassemblingStub()
    t = _bare_transport(stub)
    await t.container_put_archive(
        rollout_id="r", container_id="c", target_dir="/x", tarball=b"",
    )
    assert stub.reassembled == b""
    assert stub.meta == ("r", "c", "/x")
    assert stub.n_frames == 1 and stub.n_done == 1


# ── UNIMPLEMENTED -> unary fallback ───────────────────────────────────────────


class _OldControlPlaneStub:
    """Stream RPC UNIMPLEMENTED (pre-WS1 CP); the unary path records the fallback."""

    def __init__(self) -> None:
        self.unary_tarball: bytes | None = None

    async def ContainerPutArchiveStream(self, *a: object, **k: object) -> object:
        raise grpc.aio.AioRpcError(
            grpc.StatusCode.UNIMPLEMENTED, None, None, details="no such method",
        )

    async def ContainerPutArchive(
        self, req: object, metadata: object = None, timeout: float | None = None,
    ) -> object:
        self.unary_tarball = bytes(req.tarball)  # type: ignore[attr-defined]
        return SimpleNamespace()


@pytest.mark.asyncio
async def test_client_falls_back_to_unary_on_unimplemented() -> None:
    stub = _OldControlPlaneStub()
    t = _bare_transport(stub)
    await t.container_put_archive(
        rollout_id="r", container_id="c", target_dir="/x", tarball=b"hello-world",
    )
    assert stub.unary_tarball == b"hello-world"


# ── node-side accumulation ────────────────────────────────────────────────────


class _RecordingAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, str, bytes]] = []
        self._fail = fail

    async def container_put_archive(
        self, *, rollout_id: str, container_id: str, target_dir: str, tarball: bytes,
    ) -> None:
        if self._fail:
            raise RuntimeError("docker put_archive boom")
        self.calls.append((rollout_id, container_id, target_dir, tarball))


def _chunk(cid: str, data: bytes, done: bool, *, meta: bool = False) -> npb.ContainerPutArchiveChunk:
    c = npb.ContainerPutArchiveChunk(
        header=npb.CommandHeader(command_id=cid), data=data, done=done,
    )
    if meta:
        c.rollout_id, c.container_id, c.target_dir = "r", "c", "/app"
    return c


async def _feed(link: NodeGrpcLink, chunks: list[npb.ContainerPutArchiveChunk]):
    import asyncio
    outbox: asyncio.Queue[npb.NodeMsg] = asyncio.Queue()
    seq = _MonotonicCounter()
    for ch in chunks:
        await link._accumulate_put_archive_chunk(
            ch, ch.header, outbox, "epoch-1", seq,
        )
    replies = []
    while not outbox.empty():
        replies.append(outbox.get_nowait())
    return replies


def _bare_link(agent: object) -> NodeGrpcLink:
    link = object.__new__(NodeGrpcLink)
    link._put_archive_chunks = {}  # type: ignore[attr-defined]
    link._agent = agent  # type: ignore[attr-defined]
    return link


@pytest.mark.asyncio
async def test_node_accumulates_in_order_and_replies_ok() -> None:
    agent = _RecordingAgent()
    link = _bare_link(agent)
    replies = await _feed(link, [
        _chunk("cmd1", b"AAAA", done=False, meta=True),
        _chunk("cmd1", b"BBBB", done=False),
        _chunk("cmd1", b"CCCC", done=True),
    ])
    # one put_archive with the frames reassembled IN ORDER
    assert agent.calls == [("r", "c", "/app", b"AAAABBBBCCCC")]
    # exactly one OK reply on the terminator; correlated by command_id
    assert len(replies) == 1
    assert replies[0].reply.command_id == "cmd1"
    assert replies[0].reply.status == npb.ReplyStatus.OK
    # buffer cleaned up
    assert link._put_archive_chunks == {}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_node_replies_failed_when_agent_raises() -> None:
    agent = _RecordingAgent(fail=True)
    link = _bare_link(agent)
    replies = await _feed(link, [
        _chunk("cmd2", b"x", done=False, meta=True),
        _chunk("cmd2", b"y", done=True),
    ])
    assert len(replies) == 1
    assert replies[0].reply.status == npb.ReplyStatus.FAILED
    assert "RuntimeError" in replies[0].reply.error_kind
    assert link._put_archive_chunks == {}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_node_intermediate_chunks_emit_no_reply() -> None:
    agent = _RecordingAgent()
    link = _bare_link(agent)
    # only non-done frames → no put_archive, no reply, buffer still holds the parts
    replies = await _feed(link, [
        _chunk("cmd3", b"AAAA", done=False, meta=True),
        _chunk("cmd3", b"BBBB", done=False),
    ])
    assert replies == []
    assert agent.calls == []
    assert link._put_archive_chunks["cmd3"]["parts"] == [b"AAAA", b"BBBB"]  # type: ignore[index]


# ── error mapping (oversize is not capacity) ──────────────────────────────────


def _rpc(code: grpc.StatusCode, details: str) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(code, None, None, details=details)


def test_oversize_resource_exhausted_maps_to_archive_too_large() -> None:
    err = _classify_unmarked_rpc_error(
        _rpc(grpc.StatusCode.RESOURCE_EXHAUSTED,
             "Sent message larger than max (639180917 vs. 134217728)"),
    )
    assert isinstance(err, ArchiveTooLarge)
    assert not isinstance(err, CapacityExhausted)


def test_genuine_resource_exhausted_still_maps_to_capacity() -> None:
    err = _classify_unmarked_rpc_error(
        _rpc(grpc.StatusCode.RESOURCE_EXHAUSTED, "admission queue full"),
    )
    assert isinstance(err, CapacityExhausted)
