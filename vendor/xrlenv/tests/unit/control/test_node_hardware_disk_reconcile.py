"""The control plane reconciles ``hardware.disk_bytes`` from the heartbeat.

A node reports its disk twice: ``NodeHello.hardware.disk_bytes`` (a
``probe_hardware`` statvfs) and ``Heartbeat.total_disk_bytes`` (the image
cache's statvfs of the runtime data-root). Only the second one measures the
volume sandboxes actually write into, so it is authoritative for capacity.

Regression guard for the cn-cluster collapse (2026-08-26): agents probing
``/`` on hosts whose data-root was a separate volume advertised 97 GiB
instead of 500 GiB, and the estimator capped each node at ~22 containers
instead of ~78. Reconciling here means such a fleet self-heals on its next
heartbeat, with no node redeploy.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from xrlenv.control.grpc_endpoint import RemoteNodeTransport, _MonotonicCounter
from xrlenv.node.hw_probe import HardwareInfo

_ROOT_FS = 97 * 1024**3        # what a "/"-probing agent advertises
_DATA_ROOT = 500 * 1024**3     # what the data-root actually holds


def _hw(disk_bytes: int = _ROOT_FS) -> HardwareInfo:
    return HardwareInfo(
        vcpus=192, mem_bytes=744 * 1024**3, disk_bytes=disk_bytes,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _make_transport(hardware: HardwareInfo | None = None) -> RemoteNodeTransport:
    return RemoteNodeTransport(
        node_id="test-node",
        backends=["docker"],
        hardware=hardware if hardware is not None else _hw(),
        outbox=asyncio.Queue(),
        stream_epoch="test-epoch",
        control_instance_id="ctrl-1",
        control_seq=_MonotonicCounter(),
    )


def test_heartbeat_total_overrides_a_wrong_hello_disk() -> None:
    transport = _make_transport()
    assert transport.hardware().disk_bytes == _ROOT_FS

    transport.touch(free_disk_bytes=200 * 1024**3, total_disk_bytes=_DATA_ROOT)

    assert transport.hardware().disk_bytes == _DATA_ROOT


def test_reconcile_leaves_the_other_hardware_fields_alone() -> None:
    """Only ``disk_bytes`` is heartbeat-derived; cpu/mem stay as advertised."""
    transport = _make_transport()

    transport.touch(free_disk_bytes=1, total_disk_bytes=_DATA_ROOT)
    hw = transport.hardware()

    assert (hw.vcpus, hw.mem_bytes) == (192, 744 * 1024**3)
    assert (hw.kernel_version, hw.platform) == ("0.0.0", "linux")


def test_unreported_disk_leaves_hardware_untouched() -> None:
    """``total == 0`` is the documented "node didn't report" sentinel — it
    must not zero out a good hello reading."""
    transport = _make_transport(_hw(_DATA_ROOT))

    transport.touch(free_disk_bytes=0, total_disk_bytes=0)

    assert transport.hardware().disk_bytes == _DATA_ROOT


def test_agreeing_agent_is_left_alone_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A correctly-probing agent (hello == heartbeat) is the quiet path."""
    transport = _make_transport(_hw(_DATA_ROOT))

    with caplog.at_level(logging.WARNING, logger="xrlenv.control.grpc_endpoint"):
        transport.touch(free_disk_bytes=1, total_disk_bytes=_DATA_ROOT)

    assert transport.hardware().disk_bytes == _DATA_ROOT
    assert "disk_bytes" not in caplog.text


def test_correction_warns_once_not_once_per_heartbeat(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One line per connection is a useful signal; one per beat is noise."""
    transport = _make_transport()

    with caplog.at_level(logging.WARNING, logger="xrlenv.control.grpc_endpoint"):
        for _ in range(5):
            transport.touch(free_disk_bytes=1, total_disk_bytes=_DATA_ROOT)

    assert len([r for r in caplog.records if "disk_bytes" in r.getMessage()]) == 1


def test_reconciled_disk_reaches_the_capacity_estimator() -> None:
    """End-to-end on the axis that actually broke: the corrected disk must
    change what the estimator admits.

    2 CPU / 8 GiB containers (the deepswe footprint) on a 192-vCPU /
    744 GiB node: against a ~97 GiB root fs the disk axis caps the node in
    the low twenties, while cpu (84) and mem (78) sit idle. Against the
    real 500 GiB data-root the disk axis rises to 123 and memory becomes
    the honest binding constraint.
    """
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.capacity import NodeProfile, StaticCapacityEstimator
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateManifest,
    )

    manifest = TemplateManifest(
        name="deepswe-like",
        version="0.1",
        digest="sha256:deepswe-like",
        image="im/deepswe-like:0.1",
        resources=ResourceSpec(
            cpu_request=2.0, cpu_limit=2.0,
            mem_request_bytes=8 * 1024**3, mem_limit_bytes=8 * 1024**3,
            disk_request_bytes=2 * 1024**3,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )
    transport = _make_transport()
    estimator = StaticCapacityEstimator()

    def _cap() -> tuple[int, str]:
        cell = estimator.capacity(
            NodeProfile(
                node_id="n", hardware=transport.hardware(), backends=("docker",),
            ),
            manifest,
        )
        return cell.max_concurrent, cell.binding_constraint

    before = _cap()
    transport.touch(free_disk_bytes=1, total_disk_bytes=_DATA_ROOT)
    after = _cap()

    assert before == (23, "disk:sandbox_writable"), before
    assert after == (78, "mem"), after
