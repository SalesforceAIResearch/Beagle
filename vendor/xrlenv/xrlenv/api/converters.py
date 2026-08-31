"""Bidirectional converters between pydantic models and protobuf messages.

Keeps wire-format glue out of the data models. Each pair is named
``<X>_to_proto`` / ``<X>_from_proto`` so the call sites are greppable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.api._pb2 import rollout_control_pb2 as rpb

if TYPE_CHECKING:
    from xrlenv.control.state import NodeRecord
from xrlenv.backends.base import (
    CpuIsolation,
    MountSpec,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    RuntimeLimits,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.types import (
    CancelGroupReport,
    Deadline,
    RolloutStatus,
    Step,
    StepResult,
    TerminateRawGroupReport,
    Trajectory,
)

# ──────────────────────────────────────────────────────────────────────────────
# TemplateRef
# ──────────────────────────────────────────────────────────────────────────────


def template_ref_to_proto(ref: TemplateRef) -> pb.TemplateRef:
    return pb.TemplateRef(name=ref.name, image=ref.image, digest=ref.digest or "")


def template_ref_from_proto(p: pb.TemplateRef) -> TemplateRef:
    return TemplateRef(
        name=p.name,
        image=p.image,
        digest=p.digest or None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MountSpec / ResourceSpec
# ──────────────────────────────────────────────────────────────────────────────


def mount_spec_to_proto(m: MountSpec) -> pb.MountSpec:
    return pb.MountSpec(
        host_path=m.host_path,
        sandbox_path=m.sandbox_path,
        readonly=m.readonly,
    )


def mount_spec_from_proto(p: pb.MountSpec) -> MountSpec:
    return MountSpec(
        host_path=p.host_path,
        sandbox_path=p.sandbox_path,
        readonly=p.readonly,
    )


def cpu_isolation_from_wire(s: str) -> CpuIsolation:
    """Map the ``cpu_isolation`` wire string to :class:`CpuIsolation`. Empty
    (proto3 default) → ``OFF``; an unrecognized value from a newer peer also →
    ``OFF`` (the safe default — never accidentally isolate on a garbage value).

    Public so the control-plane ingress can decode ``request.resources.
    cpu_isolation`` at the derive-once point without pulling in the whole
    ResourceSpec converter (the rollout-control ResourceSpec is a thin
    scheduling-input message, not a full spec)."""
    if not s:
        return CpuIsolation.OFF
    try:
        return CpuIsolation(s)
    except ValueError:
        return CpuIsolation.OFF


# Back-compat alias — earlier P6 slices + tests reference the private name.
_cpu_isolation_from_wire = cpu_isolation_from_wire


def resource_spec_to_proto(r: ResourceSpec) -> pb.ResourceSpec:
    return pb.ResourceSpec(
        cpu_request=r.cpu_request,
        cpu_limit=r.cpu_limit,
        mem_request_bytes=r.mem_request_bytes,
        mem_limit_bytes=r.mem_limit_bytes,
        disk_request_bytes=r.disk_request_bytes,
        gpu_required=r.gpu_required,
        mounts=[mount_spec_to_proto(m) for m in r.mounts],
        cpu_isolation=str(r.cpu_isolation),  # P6: StrEnum value; OFF → "off"
    )


def resource_spec_from_proto(p: pb.ResourceSpec) -> ResourceSpec:
    return ResourceSpec(
        cpu_request=p.cpu_request,
        cpu_limit=p.cpu_limit,
        mem_request_bytes=p.mem_request_bytes,
        mem_limit_bytes=p.mem_limit_bytes,
        disk_request_bytes=p.disk_request_bytes,
        gpu_required=p.gpu_required,
        mounts=tuple(mount_spec_from_proto(m) for m in p.mounts),
        cpu_isolation=_cpu_isolation_from_wire(p.cpu_isolation),
    )


# P0b — RuntimeLimits. ``pb`` and ``rpb`` declare a structurally
# identical RuntimeLimits message; these converters accept either
# (duck-typed on field names) so the same helper serves the
# control-plane RPC and the node command.

def runtime_limits_to_proto(r: RuntimeLimits) -> pb.RuntimeLimits:
    return pb.RuntimeLimits(
        # proto3 scalars have no "unset" — 0 is the sentinel both
        # directions read as "harness did not specify".
        pids_limit=r.pids_limit or 0,
        shm_size_bytes=r.shm_size_bytes or 0,
        tmpfs=dict(r.tmpfs),
        readonly_rootfs=r.readonly_rootfs,
        cpu_pinning=r.cpu_pinning,
    )


def runtime_limits_from_proto(p: pb.RuntimeLimits) -> RuntimeLimits:
    return RuntimeLimits(
        pids_limit=p.pids_limit or None,
        shm_size_bytes=p.shm_size_bytes or None,
        tmpfs=dict(p.tmpfs),
        readonly_rootfs=p.readonly_rootfs,
        cpu_pinning=p.cpu_pinning,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SandboxHandle
# ──────────────────────────────────────────────────────────────────────────────


def sandbox_handle_to_proto(h: SandboxHandle) -> pb.SandboxHandle:
    return pb.SandboxHandle(
        id=h.id,
        backend=h.backend,
        backend_ref=h.backend_ref,
        stub_endpoint=h.stub_endpoint,
    )


def sandbox_handle_from_proto(p: pb.SandboxHandle) -> SandboxHandle:
    return SandboxHandle(
        id=p.id,
        backend=p.backend,
        backend_ref=p.backend_ref,
        stub_endpoint=p.stub_endpoint,
    )


# ──────────────────────────────────────────────────────────────────────────────
# HardwareInfo
# ──────────────────────────────────────────────────────────────────────────────


def hardware_info_to_proto(h: HardwareInfo) -> pb.HardwareInfo:
    return pb.HardwareInfo(
        vcpus=h.vcpus,
        mem_bytes=h.mem_bytes,
        disk_bytes=h.disk_bytes,
        has_kvm=h.has_kvm,
        has_gpu=h.has_gpu,
        gpu_model=h.gpu_model or "",
        kernel_version=h.kernel_version,
        platform=h.platform,
    )


def hardware_info_from_proto(p: pb.HardwareInfo) -> HardwareInfo:
    return HardwareInfo(
        vcpus=p.vcpus,
        mem_bytes=p.mem_bytes,
        disk_bytes=p.disk_bytes,
        has_kvm=p.has_kvm,
        has_gpu=p.has_gpu,
        gpu_model=p.gpu_model or None,
        kernel_version=p.kernel_version,
        platform=p.platform,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ResourceUsage (Slice 3.5 — StatsReply)
# ──────────────────────────────────────────────────────────────────────────────


def resource_usage_to_proto(u: ResourceUsage) -> pb.ResourceUsage:
    return pb.ResourceUsage(
        cpu_seconds=u.cpu_seconds,
        rss_bytes=u.rss_bytes,
        disk_bytes=u.disk_bytes,
        rx_bytes=u.rx_bytes,
        tx_bytes=u.tx_bytes,
    )


def resource_usage_from_proto(p: pb.ResourceUsage) -> ResourceUsage:
    return ResourceUsage(
        cpu_seconds=p.cpu_seconds,
        rss_bytes=p.rss_bytes,
        disk_bytes=p.disk_bytes,
        rx_bytes=p.rx_bytes,
        tx_bytes=p.tx_bytes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rollout-control wire format (spec 05) — mostly JSON-over-bytes for inner
# free-form payloads (init / action / obs / metadata / info). Top-level
# scalar/typed fields stay first-class on the proto.
# ──────────────────────────────────────────────────────────────────────────────


# Reward-mode <-> proto enum (spec 02). Kept narrow on purpose so a
# typo in either direction fails fast on the wire instead of silently
# falling back to env_step.
_REWARD_MODE_TO_PROTO: dict[str, rpb.RewardMode.ValueType] = {
    "env_step": rpb.REWARD_MODE_ENV_STEP,
    "in_sandbox_final": rpb.REWARD_MODE_IN_SANDBOX_FINAL,
    "trainer_final": rpb.REWARD_MODE_TRAINER_FINAL,
    "consumer_final": rpb.REWARD_MODE_CONSUMER_FINAL,
}
_REWARD_MODE_FROM_PROTO: dict[rpb.RewardMode.ValueType, str] = {
    v: k for k, v in _REWARD_MODE_TO_PROTO.items()
}


def reward_mode_to_proto(mode: str) -> rpb.RewardMode.ValueType:
    return _REWARD_MODE_TO_PROTO.get(mode, rpb.REWARD_MODE_UNSPECIFIED)


def reward_mode_from_proto(p: rpb.RewardMode.ValueType) -> str:
    return _REWARD_MODE_FROM_PROTO.get(p, "env_step")


# RolloutStatus <-> proto enum.
_STATUS_TO_PROTO: dict[RolloutStatus, rpb.RolloutStatus.ValueType] = {
    RolloutStatus.QUEUED: rpb.ROLLOUT_STATUS_QUEUED,
    RolloutStatus.STARTING: rpb.ROLLOUT_STATUS_STARTING,
    RolloutStatus.RUNNING: rpb.ROLLOUT_STATUS_RUNNING,
    RolloutStatus.CANCELLING: rpb.ROLLOUT_STATUS_CANCELLING,
    RolloutStatus.FINISHING: rpb.ROLLOUT_STATUS_FINISHING,
    RolloutStatus.FINISHED: rpb.ROLLOUT_STATUS_FINISHED,
    RolloutStatus.TRUNCATED: rpb.ROLLOUT_STATUS_TRUNCATED,
    RolloutStatus.CANCELLED: rpb.ROLLOUT_STATUS_CANCELLED,
    RolloutStatus.FAILED: rpb.ROLLOUT_STATUS_FAILED,
}
_STATUS_FROM_PROTO: dict[rpb.RolloutStatus.ValueType, RolloutStatus] = {
    v: k for k, v in _STATUS_TO_PROTO.items()
}


def rollout_status_to_proto(s: RolloutStatus) -> rpb.RolloutStatus.ValueType:
    return _STATUS_TO_PROTO.get(s, rpb.ROLLOUT_STATUS_UNSPECIFIED)


def rollout_status_from_proto(p: rpb.RolloutStatus.ValueType) -> RolloutStatus:
    return _STATUS_FROM_PROTO.get(p, RolloutStatus.QUEUED)


def json_dump(value: Any) -> bytes:
    """Serialise arbitrary JSON-able payloads. ``None`` → empty bytes so
    the receiver can distinguish "explicitly null" from "missing"."""
    if value is None:
        return b""
    return json.dumps(value, default=str).encode("utf-8")


def json_load(blob: bytes) -> Any:
    """Inverse of :func:`json_dump`. Empty bytes → ``None``."""
    if not blob:
        return None
    return json.loads(blob.decode("utf-8"))


# Deadline (optional fields, all double seconds).
def deadline_to_proto(d: Deadline | None) -> rpb.Deadline | None:
    if d is None:
        return None
    msg = rpb.Deadline()
    if d.hard_s is not None:
        msg.hard_s = d.hard_s
    if d.queue_timeout_s is not None:
        msg.queue_timeout_s = d.queue_timeout_s
    if d.setup_timeout_s is not None:
        msg.setup_timeout_s = d.setup_timeout_s
    if d.step_timeout_s is not None:
        msg.step_timeout_s = d.step_timeout_s
    if d.idle_ttl_s is not None:
        msg.idle_ttl_s = d.idle_ttl_s
    return msg


def deadline_from_proto(p: rpb.Deadline | None) -> Deadline | None:
    if p is None:
        return None
    kwargs: dict[str, Any] = {}
    if p.HasField("hard_s"):
        kwargs["hard_s"] = p.hard_s
    if p.HasField("queue_timeout_s"):
        kwargs["queue_timeout_s"] = p.queue_timeout_s
    if p.HasField("setup_timeout_s"):
        kwargs["setup_timeout_s"] = p.setup_timeout_s
    if p.HasField("step_timeout_s"):
        kwargs["step_timeout_s"] = p.step_timeout_s
    if p.HasField("idle_ttl_s"):
        kwargs["idle_ttl_s"] = p.idle_ttl_s
    return Deadline(**kwargs) if kwargs else None


# StartRolloutRequest <-> proto.
def start_rollout_request_to_proto(req: StartRolloutRequest) -> rpb.StartRolloutRequest:
    msg = rpb.StartRolloutRequest(
        template=req.template,
        init_json=json_dump(req.init),
    )
    if req.request_id is not None:
        msg.request_id = req.request_id
    if req.task_key is not None:
        msg.task_key = req.task_key
    if req.group_id is not None:
        msg.group_id = req.group_id
    if req.backend is not None:
        msg.backend = req.backend
    if req.network is not None:
        msg.network = req.network
    deadline_msg = deadline_to_proto(req.deadline)
    if deadline_msg is not None:
        msg.deadline.CopyFrom(deadline_msg)
    return msg


def start_rollout_request_from_proto(p: rpb.StartRolloutRequest) -> StartRolloutRequest:
    # Wire ``network`` is a plain string; pydantic validates against
    # the ``NetworkPolicy`` literal at construction so a bogus value
    # from a non-Python consumer raises ValidationError here rather
    # than fail-opening to bridge networking inside the Docker backend.
    network = cast(NetworkPolicy, p.network) if p.HasField("network") else None
    return StartRolloutRequest(
        template=p.template,
        init=json_load(p.init_json) or {},
        request_id=p.request_id if p.HasField("request_id") else None,
        task_key=p.task_key if p.HasField("task_key") else None,
        group_id=p.group_id if p.HasField("group_id") else None,
        deadline=deadline_from_proto(p.deadline) if p.HasField("deadline") else None,
        backend=p.backend if p.HasField("backend") else None,
        network=network,
    )


def start_rollout_response_to_proto(
    resp: StartRolloutResponse,
) -> rpb.StartRolloutResponse:
    return rpb.StartRolloutResponse(
        rollout_id=resp.rollout_id,
        init_obs_json=json_dump(resp.init_obs),
        reward_mode=reward_mode_to_proto(resp.reward_mode),
    )


def start_rollout_response_from_proto(
    p: rpb.StartRolloutResponse,
) -> StartRolloutResponse:
    return StartRolloutResponse(
        rollout_id=p.rollout_id,
        init_obs=json_load(p.init_obs_json),
        reward_mode=reward_mode_from_proto(p.reward_mode),
    )


# StepResult <-> proto StepResponse.
def step_result_to_proto(r: StepResult) -> rpb.StepResponse:
    return rpb.StepResponse(
        obs_json=json_dump(r.obs),
        reward=r.reward,
        done=r.done,
        truncated=r.truncated,
        info_json=json_dump(r.info),
    )


def step_result_from_proto(p: rpb.StepResponse) -> StepResult:
    return StepResult(
        obs=json_load(p.obs_json),
        reward=p.reward,
        done=p.done,
        truncated=p.truncated,
        info=json_load(p.info_json) or {},
    )


# Step <-> proto.
def step_to_proto(s: Step) -> rpb.Step:
    return rpb.Step(
        index=s.index,
        action_json=json_dump(s.action),
        obs_json=json_dump(s.obs),
        reward=s.reward,
        done=s.done,
        truncated=s.truncated,
        info_json=json_dump(s.info),
        ts=s.ts,
    )


def step_from_proto(p: rpb.Step) -> Step:
    return Step(
        index=p.index,
        action=json_load(p.action_json),
        obs=json_load(p.obs_json),
        reward=p.reward,
        done=p.done,
        truncated=p.truncated,
        info=json_load(p.info_json) or {},
        ts=p.ts,
    )


# Trajectory <-> proto.
def trajectory_to_proto(t: Trajectory) -> rpb.Trajectory:
    msg = rpb.Trajectory(
        rollout_id=t.rollout_id,
        template=t.template,
        steps=[step_to_proto(s) for s in t.steps],
        status=rollout_status_to_proto(t.status),
        final_reward=t.final_reward,
        metadata_json=json_dump(t.metadata),
    )
    if t.reason is not None:
        msg.reason = t.reason
    return msg


def trajectory_from_proto(p: rpb.Trajectory) -> Trajectory:
    return Trajectory(
        rollout_id=p.rollout_id,
        template=p.template,
        steps=[step_from_proto(s) for s in p.steps],
        status=rollout_status_from_proto(p.status),
        reason=p.reason if p.HasField("reason") else None,
        final_reward=p.final_reward,
        metadata=json_load(p.metadata_json) or {},
    )


# CancelGroupReport <-> proto.
def cancel_group_report_to_proto(r: CancelGroupReport) -> rpb.CancelGroupReport:
    return rpb.CancelGroupReport(
        group_id=r.group_id,
        cancelled=list(r.cancelled),
        already_terminal=list(r.already_terminal),
    )


def cancel_group_report_from_proto(p: rpb.CancelGroupReport) -> CancelGroupReport:
    return CancelGroupReport(
        group_id=p.group_id,
        cancelled=tuple(p.cancelled),
        already_terminal=tuple(p.already_terminal),
    )


# TerminateRawGroupReport <-> proto.
def terminate_raw_group_report_to_proto(
    r: TerminateRawGroupReport,
) -> rpb.TerminateRawGroupReport:
    return rpb.TerminateRawGroupReport(
        group_id=r.group_id,
        terminated=list(r.terminated),
        already_terminal=list(r.already_terminal),
    )


def terminate_raw_group_report_from_proto(
    p: rpb.TerminateRawGroupReport,
) -> TerminateRawGroupReport:
    return TerminateRawGroupReport(
        group_id=p.group_id,
        terminated=tuple(p.terminated),
        already_terminal=tuple(p.already_terminal),
    )


# D21 — NodeInfo <-> NodeRecord. The wire-side ``NodeInfo`` matches
# the consumer-relevant subset of ``state.NodeRecord``; we don't ship
# the registry's full row to the consumer SDK (e.g. internal
# bookkeeping like ``backend_versions`` if those land later stays
# server-side).
def node_info_to_proto(rec: NodeRecord) -> rpb.NodeInfo:
    return rpb.NodeInfo(
        node_id=rec.node_id,
        status=rec.status,
        backends=list(rec.backends),
        connected_at=rec.connected_at,
        last_seen_at=rec.last_seen_at,
        stream_epoch=rec.stream_epoch or "",
        instance_id=rec.instance_id or "",
    )


def node_info_from_proto(p: rpb.NodeInfo) -> NodeRecord:
    from xrlenv.control.state import NodeRecord

    return NodeRecord(
        node_id=p.node_id,
        status=p.status,
        backends=list(p.backends),
        connected_at=p.connected_at,
        last_seen_at=p.last_seen_at,
        stream_epoch=p.stream_epoch or None,
        instance_id=p.instance_id or None,
    )


__all__ = [
    "cancel_group_report_from_proto",
    "cancel_group_report_to_proto",
    "deadline_from_proto",
    "deadline_to_proto",
    "hardware_info_from_proto",
    "hardware_info_to_proto",
    "json_dump",
    "json_load",
    "mount_spec_from_proto",
    "mount_spec_to_proto",
    "node_info_from_proto",
    "node_info_to_proto",
    "resource_spec_from_proto",
    "resource_spec_to_proto",
    "resource_usage_from_proto",
    "resource_usage_to_proto",
    "reward_mode_from_proto",
    "reward_mode_to_proto",
    "rollout_status_from_proto",
    "rollout_status_to_proto",
    "sandbox_handle_from_proto",
    "sandbox_handle_to_proto",
    "start_rollout_request_from_proto",
    "start_rollout_request_to_proto",
    "start_rollout_response_from_proto",
    "start_rollout_response_to_proto",
    "step_from_proto",
    "step_result_from_proto",
    "step_result_to_proto",
    "step_to_proto",
    "template_ref_from_proto",
    "template_ref_to_proto",
    "terminate_raw_group_report_from_proto",
    "terminate_raw_group_report_to_proto",
    "trajectory_from_proto",
    "trajectory_to_proto",
]
