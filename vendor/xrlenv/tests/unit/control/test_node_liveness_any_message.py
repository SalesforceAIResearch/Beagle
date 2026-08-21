"""Issue #18 — node liveness is refreshed by ANY inbound NodeMsg.

Before the fix the :class:`NodeRegistry` watchdog keyed liveness on
heartbeat messages alone (``RemoteNodeTransport.touch``). A node busy
enough to starve its heartbeat task still streams command *replies*;
counting only heartbeats false-flagged that working node ``lost`` —
the registry + state store recorded ``lost`` while the scheduler kept
routing acquires to it (operator-observed during the 2026-05-19
SWE-bench Pro grading run: a node showed ``lost`` in ``xrlenv nodes``
while its journal proved it was actively acquiring containers).

``RemoteNodeTransport.mark_seen`` now carries the liveness bump, and
the reader loop calls it for reply / ack messages too — not only
heartbeats.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.control.grpc_endpoint import (
    NodeControlServicer,
    RemoteNodeTransport,
    _MonotonicCounter,
)
from xrlenv.node.hw_probe import HardwareInfo


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
        control_seq=_MonotonicCounter(),
    )


class _MsgIter:
    """Async iterator yielding a fixed list of NodeMsgs then stopping —
    the shape ``_reader_loop`` consumes."""

    def __init__(self, msgs: list[pb.NodeMsg]) -> None:
        self._msgs = list(msgs)

    def __aiter__(self) -> AsyncIterator[pb.NodeMsg]:
        return self

    async def __anext__(self) -> pb.NodeMsg:
        if not self._msgs:
            raise StopAsyncIteration
        return self._msgs.pop(0)


async def test_mark_seen_advances_liveness_and_fires_callback() -> None:
    """``mark_seen`` bumps the watchdog's liveness clock and fires the
    state-store mirror callback the registry installs."""
    transport = _make_transport()
    seen: list[str] = []
    transport.set_on_heartbeat(seen.append)

    before = transport.last_heartbeat_at
    await asyncio.sleep(0.01)
    transport.mark_seen()

    assert transport.last_heartbeat_at > before
    assert seen == ["test-node"]


async def test_touch_still_advances_liveness() -> None:
    """Regression guard: ``touch`` delegates the liveness bump to
    ``mark_seen``, so heartbeats still refresh the watchdog clock."""
    transport = _make_transport()
    before = transport.last_heartbeat_at
    await asyncio.sleep(0.01)
    transport.touch(free_disk_bytes=10 * 1024**3, total_disk_bytes=200 * 1024**3)

    assert transport.last_heartbeat_at > before


async def test_reader_loop_reply_message_refreshes_liveness() -> None:
    """The li-4 scenario: a node whose only inbound traffic is command
    replies (heartbeat task starved) must stay alive to the watchdog.
    A ``reply`` NodeMsg bumps the transport's liveness clock."""
    servicer = NodeControlServicer(
        on_connected=lambda _t: None,
        on_disconnected=lambda _t: None,
        control_instance_id="test-ctrl",
    )
    transport = _make_transport()
    before = transport.last_heartbeat_at

    reply_msg = pb.NodeMsg(
        stream_epoch="test-epoch",
        seq=1,
        reply=pb.CommandReply(command_id="no-such-command"),
    )
    done = asyncio.Event()
    await asyncio.sleep(0.01)
    await servicer._reader_loop(_MsgIter([reply_msg]), transport, done)

    assert transport.last_heartbeat_at > before, (
        "a command reply must refresh node liveness — a busy node "
        "streaming replies is not lost"
    )
    assert done.is_set()


async def test_reader_loop_heartbeat_stashes_health_stats() -> None:
    """Stage 1: a heartbeat carrying NodeHealthStats lands on the
    transport as ``health_json`` for the registry to mirror to state."""
    servicer = NodeControlServicer(
        on_connected=lambda _t: None,
        on_disconnected=lambda _t: None,
        control_instance_id="test-ctrl",
    )
    transport = _make_transport()
    assert transport.health_json is None  # nothing reported yet

    hb = pb.NodeMsg(
        stream_epoch="test-epoch",
        seq=1,
        heartbeat=pb.Heartbeat(
            health=pb.NodeHealthStats(
                window_s=120,
                create_p95_ms=1234.0,
                create_count=7,
                docker_error_count=3,
                docker_timeout_count=2,
                create_inflight=4,
                create_queued=5,
            ),
        ),
    )
    done = asyncio.Event()
    await servicer._reader_loop(_MsgIter([hb]), transport, done)

    assert done.is_set()
    parsed = json.loads(transport.health_json)
    assert parsed["create_p95_ms"] == 1234.0
    assert parsed["create_count"] == 7
    assert parsed["docker_timeout_count"] == 2
    assert parsed["create_queued"] == 5
