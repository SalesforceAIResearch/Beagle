"""P6 step-2a — the control plane stores the node's CPU-isolation capability
(NodeHello) + live pinned-CPU accounting (heartbeat) on the node transport.

Behavior-neutral: this is reporting/accounting only. Nothing schedules on
``isolation_capable`` or ``pinned_cpu_state`` yet — the pinned-capacity
predicate + pending reservations land in later P6 steps.
"""

from __future__ import annotations

import asyncio

from xrlenv.control.grpc_endpoint import RemoteNodeTransport, _MonotonicCounter
from xrlenv.node.hw_probe import HardwareInfo


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=8, mem_bytes=32 * 1024**3, disk_bytes=500 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _make_transport(*, isolation_capable: bool = False) -> RemoteNodeTransport:
    return RemoteNodeTransport(
        node_id="test-node",
        backends=["docker"],
        hardware=_hw(),
        outbox=asyncio.Queue(),
        stream_epoch="test-epoch",
        control_instance_id="ctrl-1",
        control_seq=_MonotonicCounter(),
        isolation_capable=isolation_capable,
    )


def test_isolation_capable_defaults_false() -> None:
    """A node that never advertised the field (pre-2a agent) → False."""
    assert _make_transport().isolation_capable() is False


def test_isolation_capable_carries_advertised_value() -> None:
    """A node advertising isolation_capable=True (NodeHello) is stored on the
    transport and exposed to the scheduler-facing surface."""
    assert _make_transport(isolation_capable=True).isolation_capable() is True


def test_pinned_cpu_state_unknown_until_first_heartbeat() -> None:
    """Before any heartbeat the pinned-CPU state is the (0, 0) 'unknown'
    sentinel — the scheduler must not make a pinned-capacity decision on it."""
    assert _make_transport().pinned_cpu_state() == (0, 0)


def test_touch_records_pinned_cpu_state() -> None:
    """A heartbeat carrying pinnable-CPU counts updates pinned_cpu_state."""
    transport = _make_transport()
    transport.touch(pinned_cpus_free=6, pinned_cpus_total=8)
    assert transport.pinned_cpu_state() == (6, 8)


def test_touch_zero_total_keeps_last_known_pinned_state() -> None:
    """Same sentinel discipline as disk: a (0, 0) report (pre-field agent or a
    node with no core ledger) must NOT clobber the last real reading."""
    transport = _make_transport()
    transport.touch(pinned_cpus_free=5, pinned_cpus_total=8)
    # A later beat that failed to sample pinnable CPUs arrives as (0, 0).
    transport.touch(pinned_cpus_free=0, pinned_cpus_total=0)
    assert transport.pinned_cpu_state() == (5, 8)


def test_touch_pinned_state_independent_of_disk_state() -> None:
    """The pinned-CPU sentinel and the disk sentinel are tracked separately —
    a disk-only heartbeat doesn't reset pinned state and vice versa."""
    transport = _make_transport()
    transport.touch(free_disk_bytes=100, total_disk_bytes=200)
    assert transport.pinned_cpu_state() == (0, 0)  # never reported → unknown
    transport.touch(pinned_cpus_free=3, pinned_cpus_total=4)
    assert transport.disk_state() == (100, 200)     # disk unchanged
    assert transport.pinned_cpu_state() == (3, 4)
