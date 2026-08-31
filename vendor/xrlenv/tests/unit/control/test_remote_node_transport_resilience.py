"""Unit tests for RemoteNodeTransport.request_terminate and send_keepalive
(the 2026-08-21 resilience additions).

These focus on the NEW surface introduced in this branch; other transport
behaviour is tested in test_node_liveness_any_message.py and
test_node_control_stream.py.
"""

from __future__ import annotations

import asyncio

from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.control.grpc_endpoint import RemoteNodeTransport, _MonotonicCounter
from xrlenv.node.hw_probe import HardwareInfo

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=2, mem_bytes=8 * 1024**3, disk_bytes=100 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _make_transport(
    outbox: asyncio.Queue[pb.ControlMsg] | None = None,
) -> RemoteNodeTransport:
    return RemoteNodeTransport(
        node_id="test-node",
        backends=["docker"],
        hardware=_hw(),
        outbox=outbox if outbox is not None else asyncio.Queue(),
        stream_epoch="epoch-abc",
        control_instance_id="ctrl-xyz",
        control_seq=_MonotonicCounter(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# request_terminate
# ──────────────────────────────────────────────────────────────────────────────


def test_request_terminate_sets_terminate_event() -> None:
    t = _make_transport()
    assert not t.terminate_event.is_set()
    t.request_terminate("watchdog")
    assert t.terminate_event.is_set()


def test_request_terminate_records_reason() -> None:
    t = _make_transport()
    t.request_terminate("false-loss-recovery")
    assert t.terminate_reason == "false-loss-recovery"


def test_request_terminate_is_idempotent_second_call_no_op() -> None:
    """Second request_terminate must not overwrite the first reason or clear
    the event."""
    t = _make_transport()
    t.request_terminate("first-reason")
    t.request_terminate("second-reason")  # must be a no-op
    assert t.terminate_reason == "first-reason"
    assert t.terminate_event.is_set()


def test_request_terminate_empty_reason_accepted() -> None:
    t = _make_transport()
    t.request_terminate("")
    assert t.terminate_event.is_set()
    assert t.terminate_reason == ""


def test_request_terminate_event_is_same_object_as_property() -> None:
    """terminate_event property exposes the same event object that is set."""
    t = _make_transport()
    evt = t.terminate_event
    t.request_terminate("test")
    assert evt.is_set()


# ──────────────────────────────────────────────────────────────────────────────
# send_keepalive — normal path
# ──────────────────────────────────────────────────────────────────────────────


def test_send_keepalive_enqueues_exactly_one_message() -> None:
    outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()
    t = _make_transport(outbox=outbox)
    t.send_keepalive()
    assert outbox.qsize() == 1


def test_send_keepalive_message_has_empty_body() -> None:
    """The enqueued ControlMsg must have no command body (empty keepalive)."""
    outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()
    t = _make_transport(outbox=outbox)
    t.send_keepalive()
    msg = outbox.get_nowait()
    assert msg.WhichOneof("body") is None


def test_send_keepalive_message_carries_stream_epoch() -> None:
    outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()
    t = _make_transport(outbox=outbox)
    t.send_keepalive()
    msg = outbox.get_nowait()
    assert msg.stream_epoch == "epoch-abc"


def test_send_keepalive_message_carries_control_instance_id() -> None:
    outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()
    t = _make_transport(outbox=outbox)
    t.send_keepalive()
    msg = outbox.get_nowait()
    assert msg.control_instance_id == "ctrl-xyz"


# ──────────────────────────────────────────────────────────────────────────────
# send_keepalive — no-op paths
# ──────────────────────────────────────────────────────────────────────────────


def test_send_keepalive_noop_after_close() -> None:
    """send_keepalive after close() must not enqueue anything."""
    outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()
    t = _make_transport(outbox=outbox)
    t.close()
    t.send_keepalive()
    assert outbox.empty()


def test_send_keepalive_noop_after_request_terminate() -> None:
    """send_keepalive after request_terminate() must not enqueue anything."""
    outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()
    t = _make_transport(outbox=outbox)
    t.request_terminate("watchdog")
    t.send_keepalive()
    assert outbox.empty()


def test_send_keepalive_multiple_calls_each_enqueue_one_message() -> None:
    """Each call before close/terminate enqueues its own message."""
    outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()
    t = _make_transport(outbox=outbox)
    t.send_keepalive()
    t.send_keepalive()
    assert outbox.qsize() == 2
