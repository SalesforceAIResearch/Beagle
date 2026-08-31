"""Direct round-trip tests for ``xrlenv/api/converters.py``.

Existing coverage is transitive — most converters are exercised through
the gRPC RPC tests. This file pins a handful of pairs whose round-trip
contract was previously deferred (D4 from
``notes/deferred_audit_todos.md``) and could silently break a proto
field rename without surfacing a test failure elsewhere.

Pairs covered here:

- ``resource_usage_to_proto`` / ``_from_proto``
- ``mount_spec_to_proto`` / ``_from_proto``
"""

from __future__ import annotations

import pytest
from xrlenv.api import converters as conv
from xrlenv.api._pb2 import node_control_pb2 as npb
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.backends.base import (
    CpuIsolation,
    MountSpec,
    ResourceSpec,
    ResourceUsage,
    RuntimeLimits,
    effective_cpu_isolation,
)


def _rs(cpu_isolation: CpuIsolation = CpuIsolation.OFF) -> ResourceSpec:
    return ResourceSpec(
        cpu_request=0.5, cpu_limit=2.0, mem_request_bytes=1 << 30,
        mem_limit_bytes=1 << 30, disk_request_bytes=1 << 30,
        cpu_isolation=cpu_isolation,
    )


def test_resource_usage_round_trip() -> None:
    original = ResourceUsage(
        cpu_seconds=12.5,
        rss_bytes=1024 * 1024 * 512,
        disk_bytes=1024 * 1024 * 1024 * 8,
        rx_bytes=4096,
        tx_bytes=2048,
    )
    proto = conv.resource_usage_to_proto(original)
    restored = conv.resource_usage_from_proto(proto)
    assert restored == original


def test_resource_usage_zero_values_round_trip() -> None:
    """All-zero usage (fresh sandbox) round-trips cleanly — no coercion
    surprises around the int/float boundaries."""
    original = ResourceUsage(
        cpu_seconds=0.0,
        rss_bytes=0,
        disk_bytes=0,
        rx_bytes=0,
        tx_bytes=0,
    )
    restored = conv.resource_usage_from_proto(
        conv.resource_usage_to_proto(original),
    )
    assert restored == original


def test_mount_spec_round_trip_readonly_true() -> None:
    original = MountSpec(
        host_path="/host/data",
        sandbox_path="/in/sandbox/data",
        readonly=True,
    )
    proto = conv.mount_spec_to_proto(original)
    restored = conv.mount_spec_from_proto(proto)
    assert restored == original


def test_mount_spec_round_trip_readonly_false() -> None:
    """Read-write mount path — ``readonly=False`` is the bool-default
    case but the proto field is still sent on the wire."""
    original = MountSpec(
        host_path="/host/scratch",
        sandbox_path="/scratch",
        readonly=False,
    )
    restored = conv.mount_spec_from_proto(
        conv.mount_spec_to_proto(original),
    )
    assert restored == original


def test_runtime_limits_round_trip_with_cpu_pinning() -> None:
    original = RuntimeLimits(
        pids_limit=512,
        shm_size_bytes=64 * 1024 * 1024,
        tmpfs={"/tmp": "size=64m"},
        readonly_rootfs=True,
        cpu_pinning=True,
    )
    restored = conv.runtime_limits_from_proto(
        conv.runtime_limits_to_proto(original),
    )
    assert restored == original
    assert restored.cpu_pinning is True


def test_runtime_limits_from_proto_serves_rollout_control_proto() -> None:
    """Regression: the duck-typed ``runtime_limits_from_proto`` is called
    with BOTH ``node_control`` (node command) and ``rollout_control``
    (client→CP AcquireContainer request) RuntimeLimits messages. When
    ``cpu_pinning`` was added to node_control only, this path raised
    ``AttributeError: cpu_pinning`` on every acquire that set the flag —
    surfacing as ``gRPC error UNKNOWN`` and killing cpuset-opt-in tasks.
    Both protos must carry the field so the shared converter serves either."""
    req = rpb.AcquireContainerRequest()
    req.runtime_limits.cpu_pinning = True
    dc = conv.runtime_limits_from_proto(req.runtime_limits)
    assert dc.cpu_pinning is True


def test_both_runtime_limits_protos_expose_cpu_pinning() -> None:
    """Invariant guard: node_control and rollout_control RuntimeLimits must
    stay field-compatible (the converter is duck-typed across both). A field
    added to one but not the other silently breaks the acquire path."""
    assert npb.RuntimeLimits(cpu_pinning=True).cpu_pinning is True
    assert rpb.RuntimeLimits(cpu_pinning=True).cpu_pinning is True


# ── P6: ResourceSpec.cpu_isolation wire round-trip + derive-once ──────────────


@pytest.mark.parametrize(
    "iso", [CpuIsolation.OFF, CpuIsolation.BEST_EFFORT, CpuIsolation.REQUIRED],
)
def test_resource_spec_cpu_isolation_round_trip(iso: CpuIsolation) -> None:
    """Every isolation mode survives the node_control ResourceSpec wire — the
    HIGH gap the audit flagged (a `required` request must not silently become
    `off`)."""
    restored = conv.resource_spec_from_proto(conv.resource_spec_to_proto(_rs(iso)))
    assert restored.cpu_isolation is iso
    assert restored == _rs(iso)


def test_resource_spec_cpu_isolation_wire_serializes_str_value() -> None:
    # the wire carrier is the StrEnum value, not a repr
    assert conv.resource_spec_to_proto(_rs(CpuIsolation.REQUIRED)).cpu_isolation == "required"
    assert conv.resource_spec_to_proto(_rs()).cpu_isolation == "off"


def test_resource_spec_cpu_isolation_defaults_off() -> None:
    """A default ResourceSpec (older caller that never set the field) → OFF, and
    an empty/unknown wire string maps to OFF (safe default, never accidental
    isolation)."""
    default = ResourceSpec(
        cpu_request=1.0, cpu_limit=1.0, mem_request_bytes=1,
        mem_limit_bytes=1, disk_request_bytes=1,
    )
    assert default.cpu_isolation is CpuIsolation.OFF
    assert conv.resource_spec_from_proto(conv.resource_spec_to_proto(default)).cpu_isolation is CpuIsolation.OFF
    assert conv._cpu_isolation_from_wire("") is CpuIsolation.OFF
    assert conv._cpu_isolation_from_wire("mode_from_a_newer_peer") is CpuIsolation.OFF


def test_rollout_and_node_resource_spec_both_carry_cpu_isolation() -> None:
    """Both proto surfaces expose the field (parity — the audit flagged both)."""
    assert npb.ResourceSpec(cpu_isolation="required").cpu_isolation == "required"
    assert rpb.ResourceSpec(cpu_isolation="required").cpu_isolation == "required"


def test_effective_cpu_isolation_derivation() -> None:
    """Derive-once precedence: explicit cpu_isolation wins; else legacy
    cpu_pinning=True → BEST_EFFORT; else OFF."""
    # legacy alias
    assert effective_cpu_isolation(_rs(), RuntimeLimits(cpu_pinning=True)) is CpuIsolation.BEST_EFFORT
    assert effective_cpu_isolation(_rs(), RuntimeLimits(cpu_pinning=False)) is CpuIsolation.OFF
    assert effective_cpu_isolation(_rs(), None) is CpuIsolation.OFF
    # explicit
    assert effective_cpu_isolation(_rs(CpuIsolation.REQUIRED), None) is CpuIsolation.REQUIRED
    # explicit REQUIRED wins over the legacy best_effort alias
    assert effective_cpu_isolation(
        _rs(CpuIsolation.REQUIRED), RuntimeLimits(cpu_pinning=True),
    ) is CpuIsolation.REQUIRED
    # .pins helper
    assert CpuIsolation.OFF.pins is False
    assert CpuIsolation.BEST_EFFORT.pins is True and CpuIsolation.REQUIRED.pins is True
