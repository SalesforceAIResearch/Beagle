"""Transport seam for the consumer SDK.

The :class:`ClientTransport` Protocol is what :class:`xrlenv.client.Client`
talks to. Two implementations:

- :class:`InProcessTransport` (Slice 1): direct in-process bridge from
  :class:`Client` → :class:`RolloutService`. Used by
  ``Client.in_process(...)`` for laptop-only / single-process deployments.
- :class:`GrpcClientTransport` (this file): unary gRPC RPCs to
  :class:`xrlenv.control.rollout_endpoint.RolloutControlServicer`. Used
  by ``Client.grpc(host, port, token)`` so a separate trainer process
  can drive a live ``xrlenv up`` control plane.

Both implement the same Protocol so the :class:`Client` never branches
on transport.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any, Protocol

import grpc

from xrlenv.api import converters as conv
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.api._pb2 import rollout_control_pb2_grpc as rpb_grpc
from xrlenv.api.constants import (
    ARCHIVE_CHUNK_BYTES,
    GRPC_CHANNEL_OPTIONS,
    HEARTBEAT_RPC_TIMEOUT_S,
)
from xrlenv.backends.base import CpuIsolation, ResourceSpec, RuntimeLimits
from xrlenv.backends.egress import EgressAllowlist
from xrlenv.control.admission import DEFAULT_QUEUE_TIMEOUT_S
from xrlenv.control.raw_container_service import RawComposeAcquireResult
from xrlenv.control.service import (
    RawAcquireResult,
    RawExecChunk,
    RawExecResult,
    RolloutService,
    StartRolloutRequest,
    StartRolloutResponse,
)
from xrlenv.errors import (
    ArchiveTooLarge,
    AuthDenied,
    BackendCapabilityMissing,
    CapacityExhausted,
    ControlPlaneLost,
    MountDenied,
    NodeCommandTimeout,
    NodeLost,
    PinCapacityExhausted,
    ReplayUnavailable,
    RewardFnRequired,
    RolloutCancelled,
    RolloutFailed,
    RolloutTruncated,
    SessionReaped,
    TemplateUnknown,
    XRLEnvError,
)
from xrlenv.types import (
    Action,
    CancelGroupReport,
    StepResult,
    TerminateRawGroupReport,
    Trajectory,
)

LOGGER = logging.getLogger(__name__)

# Stage-2 — how often a blocked acquire polls QueueStatus for its
# admission-queue position (notes/admission-stage-2-queue-clocks.md).
_QUEUE_POLL_INTERVAL_S: float = 5.0

# Client-side gRPC deadlines for the raw-container teardown / transfer RPCs
# (teardown-hang fix, 2026-07-07). Without a per-call ``timeout=`` these RPCs
# block *forever* when the control plane is slow to answer — e.g. a degraded
# sysbox node whose docker daemon is wedged, or a CP whose per-node command
# queue is backed up behind large whole-directory archive transfers on the
# shared bidi stream. A single hung teardown then stalls a whole benchmark
# sweep (harbor awaits every trial in one asyncio.TaskGroup with no per-trial
# wall-clock). These deadlines are BACKSTOPS: sized ABOVE the control plane's
# own bounded ceilings (destroy / put_archive / get_archive all cap at 300 s;
# exec caps at the request's ``timeout_s`` + 30 s) so the CP's graceful
# semantics fire first (e.g. destroy → defer removal to raw-GC on a node
# timeout). The client deadline only trips when the CP itself is unresponsive,
# converting an infinite hang into a bounded ``DEADLINE_EXCEEDED`` that the
# caller (harbor's teardown ``except Exception``) already handles → the trial
# reaps and the sweep proceeds. Motivating incident: a TerminalWorld sweep hung
# ~40 min on one trial's post-agent log-salvage + destroy against a degraded
# sysbox node, with no per-trial wall-clock in harbor to reap it.
_DESTROY_DEADLINE_S: float = 330.0   # CP destroy ceiling is 300 s
_ARCHIVE_DEADLINE_S: float = 330.0   # CP get/put-archive ceiling is 300 s
# exec: bound the whole streaming call at the server-side exec ``timeout_s``
# plus this margin (the CP already extends the node stream by +30 s).
_EXEC_DEADLINE_MARGIN_S: float = 60.0


class ClientTransport(Protocol):
    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse: ...
    async def step(self, rollout_id: str, action: Action) -> StepResult: ...
    async def finish(self, rollout_id: str) -> Trajectory: ...
    async def cancel(self, rollout_id: str, reason: str) -> Trajectory: ...
    async def cancel_group(self, group_id: str, reason: str) -> CancelGroupReport: ...
    async def terminate_raw_group(
        self, group_id: str, reason: str,
    ) -> TerminateRawGroupReport: ...
    async def replay(self, rollout_id: str) -> Trajectory: ...
    async def heartbeat(self, rollout_id: str) -> None: ...
    async def heartbeat_many(self, rollout_ids: list[str]) -> None: ...
    async def set_final_reward(self, rollout_id: str, final_reward: float) -> None: ...
    async def list_nodes(self) -> list[Any]: ...
    async def close(self) -> None: ...

    # P1.7.A.1 + P1.7.B.2 — raw container session.
    async def acquire_container(
        self,
        *,
        image: str,
        command: list[str] | None = None,
        entrypoint: list[str] | None = None,
        user: str | None = None,
        cap_add: list[str] | None = None,
        devices: list[str] | None = None,
        privileged: bool = False,
        network_mode: str | None = None,
        binds: list[str] | None = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        task_key: str | None = None,
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        cpu_limit: float | None = None,
        mem_limit_bytes: int | None = None,
        cpu_isolation: CpuIsolation = CpuIsolation.OFF,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawAcquireResult: ...

    async def container_exec(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> RawExecResult: ...

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
    ) -> None: ...

    async def destroy_container(
        self, *, rollout_id: str, container_id: str, force: bool = True,
    ) -> None: ...

    # P1.7.C.2 — multi-service compose project. The footprint arrives as scalar
    # cpu/mem (the whole-stack reserve) so the consumer surface stays ResourceSpec-
    # free, symmetric with acquire_container's cpu_limit/mem_limit_bytes; each
    # transport builds the ResourceSpec / proto footprint at its own boundary.
    async def acquire_compose_project(
        self,
        *,
        compose_yaml: str,
        images: list[str],
        footprint_cpu: float,
        footprint_mem_bytes: int,
        main_service: str = "main",
        project_name: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        labels: dict[str, str] | None = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        up_timeout_s: float | None = None,
    ) -> RawComposeAcquireResult: ...

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str | None = None,
    ) -> None: ...

    async def container_put_archive(
        self, *, rollout_id: str, container_id: str,
        target_dir: str, tarball: bytes,
    ) -> None: ...

    async def container_get_archive(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> bytes: ...

    def container_exec_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> Any: ...  # AsyncIterator[RawExecChunk]


class InProcessTransport:
    """Direct in-process bridge from :class:`Client` → :class:`RolloutService`."""

    def __init__(self, service: RolloutService) -> None:
        self._service = service

    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
        return await self._service.start_rollout(req)

    async def step(self, rollout_id: str, action: Action) -> StepResult:
        return await self._service.step(rollout_id, action)

    async def finish(self, rollout_id: str) -> Trajectory:
        return await self._service.finish(rollout_id)

    async def cancel(self, rollout_id: str, reason: str) -> Trajectory:
        return await self._service.cancel(rollout_id, reason)

    async def cancel_group(self, group_id: str, reason: str) -> CancelGroupReport:
        return await self._service.cancel_group(group_id, reason)

    async def terminate_raw_group(
        self, group_id: str, reason: str,
    ) -> TerminateRawGroupReport:
        return await self._service.terminate_raw_group(group_id, reason)

    async def replay(self, rollout_id: str) -> Trajectory:
        return await self._service.replay(rollout_id)

    async def heartbeat(self, rollout_id: str) -> None:
        await self._service.heartbeat(rollout_id)

    async def heartbeat_many(self, rollout_ids: list[str]) -> None:
        await self._service.heartbeat_many(rollout_ids)

    async def set_final_reward(self, rollout_id: str, final_reward: float) -> None:
        await self._service.set_final_reward(rollout_id, final_reward)

    async def list_nodes(self) -> list[Any]:
        return await self._service.list_nodes()

    async def close(self) -> None:
        return None

    # ── P1.7.A.1 + P1.7.B.2 — raw container session ─────────────────────────

    async def acquire_container(
        self,
        *,
        image: str,
        command: list[str] | None = None,
        entrypoint: list[str] | None = None,
        user: str | None = None,
        cap_add: list[str] | None = None,
        devices: list[str] | None = None,
        privileged: bool = False,
        network_mode: str | None = None,
        binds: list[str] | None = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        task_key: str | None = None,
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        cpu_limit: float | None = None,
        mem_limit_bytes: int | None = None,
        cpu_isolation: CpuIsolation = CpuIsolation.OFF,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawAcquireResult:
        return await self._service.acquire_container(
            image=image, command=command, entrypoint=entrypoint,
            user=user, cap_add=cap_add, devices=devices,
            privileged=privileged, network_mode=network_mode, binds=binds,
            name=name,
            labels=labels, environment=environment,
            task_key=task_key,
            ensure_image_present=ensure_image_present,
            userns_mode=userns_mode,
            acquire_timeout_s=acquire_timeout_s,
            cpu_limit=cpu_limit,
            mem_limit_bytes=mem_limit_bytes,
            cpu_isolation=cpu_isolation,
            runtime_limits=runtime_limits,
            container_runtime=container_runtime,
            # Stage-2: ``None`` means "use the cluster default" — the
            # 24h backstop, same as the gRPC path. Audit M1 fix: this
            # was a hardcoded 3600s that bypassed DEFAULT_QUEUE_TIMEOUT_S
            # for in-process / local clients.
            queue_timeout_s=(
                queue_timeout_s
                if queue_timeout_s is not None
                else DEFAULT_QUEUE_TIMEOUT_S
            ),
            session_deadline_s=session_deadline_s,
        )

    async def container_exec(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> RawExecResult:
        return await self._service.container_exec(
            rollout_id=rollout_id, container_id=container_id,
            cmd=cmd, timeout_s=timeout_s, cwd=cwd, env=env, user=user,
        )

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
    ) -> None:
        await self._service.apply_egress(
            rollout_id=rollout_id, container_id=container_id,
            allowlist=allowlist, dns_resolver=dns_resolver,
        )

    async def destroy_container(
        self, *, rollout_id: str, container_id: str, force: bool = True,
    ) -> None:
        await self._service.destroy_container(
            rollout_id=rollout_id, container_id=container_id, force=force,
        )

    async def acquire_compose_project(
        self,
        *,
        compose_yaml: str,
        images: list[str],
        footprint_cpu: float,
        footprint_mem_bytes: int,
        main_service: str = "main",
        project_name: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        labels: dict[str, str] | None = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        up_timeout_s: float | None = None,
    ) -> RawComposeAcquireResult:
        return await self._service.acquire_compose_project(
            compose_yaml=compose_yaml,
            images=images,
            footprint=ResourceSpec(
                cpu_request=footprint_cpu, cpu_limit=footprint_cpu,
                mem_request_bytes=footprint_mem_bytes,
                mem_limit_bytes=footprint_mem_bytes,
                disk_request_bytes=0,
            ),
            main_service=main_service,
            project_name=project_name,
            task_key=task_key,
            group_id=group_id,
            labels=labels,
            # ``None`` → the cluster default, same as the gRPC path (0.0 sentinel).
            queue_timeout_s=(
                queue_timeout_s
                if queue_timeout_s is not None
                else DEFAULT_QUEUE_TIMEOUT_S
            ),
            session_deadline_s=session_deadline_s,
            up_timeout_s=up_timeout_s,
        )

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str | None = None,
    ) -> None:
        await self._service.destroy_compose_project(
            rollout_id=rollout_id, project_name=project_name,
        )

    async def container_put_archive(
        self, *, rollout_id: str, container_id: str,
        target_dir: str, tarball: bytes,
    ) -> None:
        await self._service.container_put_archive(
            rollout_id=rollout_id, container_id=container_id,
            target_dir=target_dir, tarball=tarball,
        )

    async def container_get_archive(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> bytes:
        return await self._service.container_get_archive(
            rollout_id=rollout_id, container_id=container_id,
            source_path=source_path,
        )

    def container_exec_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> Any:
        # Async-generator: returns the service's iterator
        # directly. The caller does ``async for chunk in
        # transport.container_exec_stream(...)``.
        return self._service.container_exec_stream(
            rollout_id=rollout_id, container_id=container_id,
            cmd=cmd, timeout_s=timeout_s, cwd=cwd, env=env, user=user,
        )


# ──────────────────────────────────────────────────────────────────────────────
# gRPC transport — server lives in xrlenv.control.rollout_endpoint
# ──────────────────────────────────────────────────────────────────────────────


# Trailing-metadata keys shared with the server-side ``rollout_endpoint``.
# Kept duplicated here on purpose so the client doesn't import the server
# module — the wire contract is the proto + these metadata keys.
_KIND_META_KEY = "xrlenv-error-kind"
_REASON_META_KEY = "xrlenv-error-reason"
_PARTIAL_META_KEY = "xrlenv-error-partial-bin"


# Mirror of ``rollout_endpoint._EXC_TO_CODE``: each known error kind maps
# to a class the client uses to rehydrate the exception. Anything we
# don't recognise becomes a generic :class:`XRLEnvError`.
_KIND_TO_EXC: dict[str, type[XRLEnvError]] = {
    "TemplateUnknown":          TemplateUnknown,
    "RewardFnRequired":         RewardFnRequired,
    "BackendCapabilityMissing": BackendCapabilityMissing,
    "MountDenied":              MountDenied,
    "AuthDenied":               AuthDenied,
    "CapacityExhausted":        CapacityExhausted,
    # Node-side REQUIRED-pin exhaustion (a stale-heartbeat / ledger race, not global pool
    # exhaustion). Reconstruct the concrete subclass so a type-keyed classifier (the CP
    # re-admit path, the compat infra-evidence recorder) sees it as itself, not a bare
    # XRLEnvError (audit M8).
    "PinCapacityExhausted":     PinCapacityExhausted,
    "ControlPlaneLost":         ControlPlaneLost,
    "NodeLost":                 NodeLost,
    # Platform-initiated session teardown — the liveness quarantine, but also
    # the wall-clock deadline sweep and node-side orphan seals; read ``reason``
    # for which. Rehydrated as its own type so a harness can classify it as
    # infra-transient rather than reading a bare XRLEnvError that looks like a
    # stale handle.
    "SessionReaped":            SessionReaped,
    # A node reply-timeout (create/exec/destroy) surfaced through the CP's
    # _abort_with_xrlenv_error, which stamps the kind. Without this entry it fell
    # through to _classify_unmarked_rpc_error → a bare XRLEnvError, invisible to
    # any classifier keying on the type (2026-07-08: conc-32 create-timeouts read
    # as "gRPC error UNKNOWN" instead of NodeCommandTimeout).
    "NodeCommandTimeout":       NodeCommandTimeout,
    "RolloutTruncated":         RolloutTruncated,
    "RolloutCancelled":         RolloutCancelled,
    "RolloutFailed":            RolloutFailed,
    "ReplayUnavailable":        ReplayUnavailable,
}


def _rehydrate_xrlenv_error(rpc_error: grpc.aio.AioRpcError) -> XRLEnvError:
    """Rebuild an xrlenv exception from a gRPC error's trailing metadata.

    Falls back to a generic :class:`XRLEnvError` when the server didn't
    set our metadata keys (e.g. unrelated gRPC failures: connection
    refused, deadline at the channel layer, etc.).
    """
    raw_meta = rpc_error.trailing_metadata() or ()
    meta: dict[str, str | bytes] = {}
    for entry in raw_meta:
        # grpc-aio yields either ``Metadatum`` objects (newer versions
        # have ``.key`` / ``.value`` attrs) or plain ``(key, value)``
        # tuples — handle both shapes so the test fakes and real
        # channels both work.
        if hasattr(entry, "key"):
            meta[entry.key] = entry.value
        else:
            tup: tuple[str, str | bytes] = entry  # type: ignore[assignment]
            meta[tup[0]] = tup[1]
    kind_raw = meta.get(_KIND_META_KEY) or ""
    kind = kind_raw if isinstance(kind_raw, str) else kind_raw.decode("utf-8", "replace")
    reason_raw = meta.get(_REASON_META_KEY)
    reason: str | None
    if reason_raw is None:
        reason = None
    elif isinstance(reason_raw, str):
        reason = reason_raw
    else:
        reason = reason_raw.decode("utf-8", "replace")
    partial_bytes = meta.get(_PARTIAL_META_KEY)
    partial: Trajectory | None = None
    if partial_bytes is not None:
        try:
            payload = (
                partial_bytes if isinstance(partial_bytes, bytes)
                else partial_bytes.encode("latin-1")
            )
            traj_proto = rpb.Trajectory()
            traj_proto.ParseFromString(payload)
            partial = conv.trajectory_from_proto(traj_proto)
        except Exception:
            LOGGER.exception(
                "GrpcClientTransport: failed to deserialise partial trajectory; "
                "raising the carrier exception without a partial",
            )
            partial = None

    cls = _KIND_TO_EXC.get(kind)
    msg = rpc_error.details() or str(rpc_error)
    if cls is RolloutFailed:
        return RolloutFailed(msg, reason=reason or "unknown", partial=partial)
    if cls in (RolloutTruncated, RolloutCancelled):
        return cls(msg, partial=partial)  # type: ignore[call-arg]
    if cls is SessionReaped:
        # SessionReaped.__init__ requires ``reason`` (no default), so the bare
        # ``cls(msg)`` fallback below would raise TypeError instead of rehydrating
        # the exception. (``RolloutFailed`` is the only other entry that requires
        # ``reason``; it is intercepted above and never reaches the fallback.)
        # reason rides the existing _REASON_META_KEY; reaped_at has no metadata
        # key of its own and does NOT survive the wire round-trip (always None
        # on the client side).
        return SessionReaped(msg, reason=reason or "session reaped")
    if cls is not None:
        return cls(msg)
    # No structured metadata — surface the underlying transport problem.
    return _classify_unmarked_rpc_error(rpc_error)


def _classify_unmarked_rpc_error(rpc_error: grpc.aio.AioRpcError) -> XRLEnvError:
    """When the server didn't set xrlenv error metadata, classify by gRPC code.

    These are *transport* failures (channel down, timeout at the gRPC
    layer, the server dying mid-call). The trainer SDK's Session
    bookkeeping treats them differently from structured xrlenv errors,
    so we return a class hint that's reasonable for the network case.
    """
    code = rpc_error.code()
    msg = rpc_error.details() or str(rpc_error)
    if code is grpc.StatusCode.UNAUTHENTICATED:
        return AuthDenied(msg)
    if code is grpc.StatusCode.PERMISSION_DENIED:
        return AuthDenied(msg)
    if code is grpc.StatusCode.DEADLINE_EXCEEDED:
        # A client-side deadline tripped (teardown-hang backstop): the CP
        # didn't answer within our per-call ceiling. Retryable, and shaped
        # like a command timeout so downstream classifiers treat it as one.
        return NodeCommandTimeout(msg)
    if code is grpc.StatusCode.UNAVAILABLE:
        return ControlPlaneLost(msg)
    if code is grpc.StatusCode.CANCELLED:
        # A transport-level RPC abort — typically the channel/peer dropped MID-STREAM (a
        # streaming ``/eval.sh`` exec that lost its connection). A deliberate client-side
        # ``future.cancel()`` raises ``asyncio.CancelledError``, not an AioRpcError, so this
        # only fires for a real transport drop. Preserve it as a concrete infra kind so a
        # swallowed streaming disconnect is retryable, not a bare ``XRLEnvError`` (audit M8).
        return ControlPlaneLost(msg)
    if code is grpc.StatusCode.RESOURCE_EXHAUSTED:
        # A client/server-side "Sent message larger than max" is an oversized PAYLOAD,
        # not node capacity — mapping it to CapacityExhausted (which is in the infra-retry
        # set) would burn retries on a deterministic failure and mislabel the cause. Only
        # the genuine admission/capacity RESOURCE_EXHAUSTED is CapacityExhausted.
        if "larger than max" in msg or "Sent message larger" in msg:
            return ArchiveTooLarge(msg)
        return CapacityExhausted(msg)
    if code is grpc.StatusCode.NOT_FOUND:
        return ReplayUnavailable(msg)
    return XRLEnvError(f"gRPC error {code.name}: {msg}")


class _BearerAuthMetadataPlugin(grpc.AuthMetadataPlugin):
    """Attach ``authorization: Bearer <token>`` to every RPC."""

    def __init__(self, token: str) -> None:
        self._token = f"Bearer {token}"

    def __call__(
        self,
        context: grpc.AuthMetadataContext,
        callback: grpc.AuthMetadataPluginCallback,
    ) -> None:
        callback((("authorization", self._token),), None)


class GrpcClientTransport:
    """gRPC implementation of :class:`ClientTransport`.

    Built by :py:meth:`xrlenv.client.Client.grpc`. Holds a single
    :class:`grpc.aio.Channel` for the process lifetime and uses one
    :class:`RolloutControlStub` against it. Auth is via a bearer token
    in metadata (spec 19): the token must have the ``consumer`` role
    or the server returns ``UNAUTHENTICATED`` /
    ``PERMISSION_DENIED``, both of which we surface as
    :class:`AuthDenied`.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str | None,
        secure: bool = False,
        channel_options: list[tuple[str, Any]] | None = None,
    ) -> None:
        target = f"{host}:{port}"
        # Audit M1: layer the platform's GRPC_CHANNEL_OPTIONS (covers
        # max_*_message_length so trajectory/replay payloads above the
        # 4 MB gRPC default still flow) under any caller-supplied
        # overrides. ``opts`` last lets advanced callers still tighten
        # or relax specific options.
        opts: list[tuple[str, Any]] = list(GRPC_CHANNEL_OPTIONS)
        if channel_options:
            opts.extend(channel_options)
        # Setting up channel credentials. Phase-0 supports plaintext
        # (loopback / SSH-tunnel topology) and TLS with a per-call
        # bearer token. mTLS lands later if/when the operator demands
        # it; the CallCredentials plumbing here is forward-compatible.
        if secure:
            channel_creds = grpc.ssl_channel_credentials()
            if token is not None:
                call_creds = grpc.metadata_call_credentials(
                    _BearerAuthMetadataPlugin(token),
                )
                composed = grpc.composite_channel_credentials(
                    channel_creds, call_creds,
                )
                self._channel = grpc.aio.secure_channel(target, composed, options=opts)
            else:
                self._channel = grpc.aio.secure_channel(
                    target, channel_creds, options=opts,
                )
        else:
            # Plaintext channel; bearer token (if any) is attached via
            # an interceptor so every unary call carries it.
            self._channel = grpc.aio.insecure_channel(target, options=opts)

        self._token = token
        self._stub = rpb_grpc.RolloutControlStub(self._channel)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _metadata(self) -> list[tuple[str, str]]:
        """Bearer token in metadata. Empty list when no token configured
        (operator-trusted local channels)."""
        if self._token is None:
            return []
        return [("authorization", f"Bearer {self._token}")]

    # ── Protocol surface ─────────────────────────────────────────────────────

    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
        try:
            resp = await self._stub.StartRollout(
                conv.start_rollout_request_to_proto(req),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return conv.start_rollout_response_from_proto(resp)

    async def step(self, rollout_id: str, action: Action) -> StepResult:
        try:
            resp = await self._stub.Step(
                rpb.StepRequest(
                    rollout_id=rollout_id, action_json=conv.json_dump(action),
                ),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return conv.step_result_from_proto(resp)

    async def finish(self, rollout_id: str) -> Trajectory:
        try:
            resp = await self._stub.Finish(
                rpb.FinishRequest(rollout_id=rollout_id),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return conv.trajectory_from_proto(resp.trajectory)

    async def cancel(self, rollout_id: str, reason: str) -> Trajectory:
        try:
            resp = await self._stub.Cancel(
                rpb.CancelRequest(rollout_id=rollout_id, reason=reason),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return conv.trajectory_from_proto(resp.trajectory)

    async def cancel_group(self, group_id: str, reason: str) -> CancelGroupReport:
        try:
            resp = await self._stub.CancelGroup(
                rpb.CancelGroupRequest(group_id=group_id, reason=reason),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return conv.cancel_group_report_from_proto(resp.report)

    async def terminate_raw_group(
        self, group_id: str, reason: str,
    ) -> TerminateRawGroupReport:
        try:
            resp = await self._stub.TerminateRawGroup(
                rpb.TerminateRawGroupRequest(group_id=group_id, reason=reason),
                metadata=self._metadata(),
                # Teardown fans out to N nodes; give it the same ceiling as destroy_container.
                timeout=_DESTROY_DEADLINE_S,
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return conv.terminate_raw_group_report_from_proto(resp.report)

    async def replay(self, rollout_id: str) -> Trajectory:
        try:
            resp = await self._stub.Replay(
                rpb.ReplayRequest(rollout_id=rollout_id),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return conv.trajectory_from_proto(resp.trajectory)

    async def heartbeat(self, rollout_id: str) -> None:
        try:
            await self._stub.Heartbeat(
                rpb.HeartbeatRequest(rollout_id=rollout_id),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc

    async def heartbeat_many(self, rollout_ids: list[str]) -> None:
        # Batched keepalive: one RPC for all of this process's live raw
        # sessions (overhead scales with processes, not sessions).
        #
        # DEADLINE, not best-effort. The caller swallows failures so a blip
        # can't tear the loop down — but a HUNG beat is not a failure, it just
        # never returns, and the loop is single-threaded: one wedged RPC
        # silences this process's keepalive forever and every quiet session it
        # owns is destroyed at the horizon. A timeout well under the beat
        # cadence turns that hang into an ordinary swallowed error the next
        # beat retries.
        try:
            await self._stub.Heartbeat(
                rpb.HeartbeatRequest(rollout_ids=rollout_ids),
                metadata=self._metadata(),
                timeout=HEARTBEAT_RPC_TIMEOUT_S,
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc

    async def set_final_reward(self, rollout_id: str, final_reward: float) -> None:
        try:
            await self._stub.SetFinalReward(
                rpb.SetFinalRewardRequest(
                    rollout_id=rollout_id, final_reward=final_reward,
                ),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc

    async def list_nodes(self) -> list[Any]:
        try:
            resp = await self._stub.ListNodes(
                rpb.ListNodesRequest(),
                metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return [conv.node_info_from_proto(p) for p in resp.nodes]

    async def close(self) -> None:
        await self._channel.close()

    # ── P1.7.A.1 + P1.7.B.2 — raw container session ─────────────────────────

    async def acquire_container(
        self,
        *,
        image: str,
        command: list[str] | None = None,
        entrypoint: list[str] | None = None,
        user: str | None = None,
        cap_add: list[str] | None = None,
        devices: list[str] | None = None,
        privileged: bool = False,
        network_mode: str | None = None,
        binds: list[str] | None = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        task_key: str | None = None,
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        cpu_limit: float | None = None,
        mem_limit_bytes: int | None = None,
        cpu_isolation: CpuIsolation = CpuIsolation.OFF,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawAcquireResult:
        req = rpb.AcquireContainerRequest(
            image=image,
            # Negative-form on the wire; proto3 default-false means
            # "ensure runs on the chosen node" (the new default UX).
            strict_image_check=not ensure_image_present,
            userns_mode=userns_mode if userns_mode != "host" else "",
            # Issue #12 — pass-through override; 0.0 is proto3 default
            # and means "use the server's default" on the receiving
            # side (currently 600 s).
            acquire_timeout_s=(
                acquire_timeout_s if acquire_timeout_s is not None else 0.0
            ),
            # Issue #18 (Ask #1) — admission-queue wait bound; 0.0 is
            # the proto3 default meaning "use the server default"
            # (3600 s).
            queue_timeout_s=(
                queue_timeout_s if queue_timeout_s is not None else 0.0
            ),
            # Issue #18 — session lifetime cap; 0.0 = server default.
            session_deadline_s=(
                session_deadline_s
                if session_deadline_s is not None
                else 0.0
            ),
        )
        if command:
            req.command.extend(command)
        if entrypoint is not None:
            # ``[""]`` ("clear the entrypoint") MUST flow as a single-
            # element list, not be silently dropped — that's the whole
            # docker CLI ``--entrypoint ""`` semantic that consumers
            # rely on for benchmark images. ``None`` is "leave at image
            # default" and stays off the wire.
            req.entrypoint.extend(entrypoint)
        if user:
            req.user = user
        if cap_add:
            req.cap_add.extend(cap_add)
        if devices:
            req.devices.extend(devices)
        if privileged:
            req.privileged = True
        if network_mode:
            req.network_mode = network_mode
        if binds:
            req.binds.extend(binds)
        if name is not None:
            req.name = name
        if labels:
            for k, v in labels.items():
                req.labels[k] = v
        if environment:
            for k, v in environment.items():
                req.environment[k] = v
        if task_key is not None:
            req.task_key = task_key
        # P0a — carry the harness's effective CPU/memory request as an
        # explicit scheduling input. Non-zero fields are harness
        # overrides; the control plane merges them with the
        # raw-container default budget. cpu_request is set == cpu_limit
        # (raw containers pack request==limit until P1 — see
        # cluster-resource-isolation-plan P3).
        if cpu_limit is not None or mem_limit_bytes is not None:
            if cpu_limit is not None:
                req.resources.cpu_request = cpu_limit
                req.resources.cpu_limit = cpu_limit
            if mem_limit_bytes is not None:
                req.resources.mem_request_bytes = mem_limit_bytes
                req.resources.mem_limit_bytes = mem_limit_bytes
        # P6 — explicit CPU-isolation contract. proto3 default-empty on the
        # receiver maps to OFF, so only non-OFF modes need to ride the wire.
        if cpu_isolation is not CpuIsolation.OFF:
            req.resources.cpu_isolation = str(cpu_isolation)
        # P0b — carry the harness's container-shape RuntimeLimits.
        # proto3 scalars: 0 is the "unset" sentinel the receiver reads.
        if runtime_limits is not None and not runtime_limits.is_empty():
            req.runtime_limits.pids_limit = runtime_limits.pids_limit or 0
            req.runtime_limits.shm_size_bytes = (
                runtime_limits.shm_size_bytes or 0
            )
            req.runtime_limits.readonly_rootfs = runtime_limits.readonly_rootfs
            # cpuset opt-in (mirrors node_control.RuntimeLimits field 5). Without
            # this the flag is dropped client-side and never reaches the node.
            req.runtime_limits.cpu_pinning = runtime_limits.cpu_pinning
            for path, opts in runtime_limits.tmpfs.items():
                req.runtime_limits.tmpfs[path] = opts
        # §5.1 — OCI runtime selector. Empty string is the proto3 "unset"
        # sentinel the receiver reads back as None.
        if container_runtime:
            req.container_runtime = container_runtime
        # Stage-2 — tag the request so a concurrent poller can ask the
        # control plane for *this* acquire's admission-queue position
        # while the (blocking) AcquireContainer RPC waits for capacity.
        request_id = uuid.uuid4().hex
        req.request_id = request_id
        poller = asyncio.create_task(self._poll_queue_status(request_id))
        try:
            resp = await self._stub.AcquireContainer(
                req, metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        finally:
            poller.cancel()
            with suppress(asyncio.CancelledError):
                await poller
        return RawAcquireResult(
            rollout_id=resp.rollout_id,
            container_id=resp.container_id,
            container_name=resp.container_name,
            node_id=resp.node_id,
            queue_wait_s=resp.queue_wait_s,
        )

    async def _poll_queue_status(self, request_id: str) -> None:
        """Stage-2 — while an acquire blocks in the admission queue,
        poll QueueStatus and log this request's live FIFO position so a
        queued acquire is visible instead of a silent wait.

        Waiting in the queue is not an error; this only logs and never
        raises into the acquire it is narrating. The first poll is one
        interval in, so a fast (immediately-placed) acquire logs
        nothing — the poller is cancelled before it fires.
        """
        while True:
            await asyncio.sleep(_QUEUE_POLL_INTERVAL_S)
            try:
                st = await self._stub.QueueStatus(
                    rpb.QueueStatusRequest(request_id=request_id),
                    metadata=self._metadata(),
                )
            except Exception:
                # A transient QueueStatus failure must never disturb the
                # acquire this only narrates — stop polling quietly.
                return
            if st.state != "queued":
                continue
            LOGGER.info(
                "acquire queued — position %d of %d in the cluster "
                "admission queue; waiting for capacity (not an error)",
                st.position, st.queue_depth,
            )

    async def container_exec(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> RawExecResult:
        req = rpb.ContainerExecRequest(
            rollout_id=rollout_id, container_id=container_id,
            cmd=cmd, timeout_s=timeout_s,
        )
        if cwd is not None:
            req.cwd = cwd
        if env:
            for k, v in env.items():
                req.env[k] = v
        if user is not None:
            req.user = user
        try:
            resp = await self._stub.ContainerExec(
                req, metadata=self._metadata(),
                # Backstop above the server-side exec timeout — bounds the
                # sysbox log-salvage exec (base64 tarball) on a wedged node.
                timeout=timeout_s + _EXEC_DEADLINE_MARGIN_S,
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        return RawExecResult(
            exit_code=resp.exit_code,
            stdout=bytes(resp.stdout),
            stderr=bytes(resp.stderr),
            timed_out=resp.timed_out,
        )

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
    ) -> None:
        req = rpb.ApplyEgressRequest(
            rollout_id=rollout_id,
            container_id=container_id,
            allow=[
                rpb.EgressAllowEntry(cidr=r.cidr, ports=list(r.ports or ()))
                for r in allowlist.rules
            ],
        )
        if dns_resolver is not None:
            req.dns_resolver = dns_resolver
        try:
            await self._stub.ApplyEgress(req, metadata=self._metadata())
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc

    async def destroy_container(
        self, *, rollout_id: str, container_id: str, force: bool = True,
    ) -> None:
        try:
            await self._stub.DestroyContainer(
                rpb.DestroyContainerRequest(
                    rollout_id=rollout_id,
                    container_id=container_id,
                    force=force,
                ),
                metadata=self._metadata(),
                # Backstop deadline — see _DESTROY_DEADLINE_S. Without it a
                # wedged node makes teardown hang the whole sweep.
                timeout=_DESTROY_DEADLINE_S,
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc

    async def acquire_compose_project(
        self,
        *,
        compose_yaml: str,
        images: list[str],
        footprint_cpu: float,
        footprint_mem_bytes: int,
        main_service: str = "main",
        project_name: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        labels: dict[str, str] | None = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        up_timeout_s: float | None = None,
    ) -> RawComposeAcquireResult:
        req = rpb.AcquireComposeProjectRequest(
            compose_yaml=compose_yaml,
            main_service=main_service,
            # proto3 ``0.0`` = "use the server default" on the receiving side
            # (mirrors AcquireContainer's sentinels).
            queue_timeout_s=(
                queue_timeout_s if queue_timeout_s is not None else 0.0
            ),
            session_deadline_s=(
                session_deadline_s if session_deadline_s is not None else 0.0
            ),
            up_timeout_s=up_timeout_s if up_timeout_s is not None else 0.0,
        )
        # Whole-stack footprint (scheduler reserve): request==limit like the
        # raw-container path packs cpu/mem until P1.
        req.footprint.cpu_request = footprint_cpu
        req.footprint.cpu_limit = footprint_cpu
        req.footprint.mem_request_bytes = footprint_mem_bytes
        req.footprint.mem_limit_bytes = footprint_mem_bytes
        if images:
            req.images.extend(images)
        if project_name is not None:
            req.project_name = project_name
        if task_key is not None:
            req.task_key = task_key
        if group_id is not None:
            req.group_id = group_id
        if labels:
            for k, v in labels.items():
                req.labels[k] = v
        # Stage-2 — tag the request so a concurrent poller narrates this
        # acquire's admission-queue position while it blocks for capacity
        # (identical to AcquireContainer). No client-side wire deadline: the
        # server bounds it by queue_timeout_s and the node by up_timeout_s, so a
        # legitimate cold-pull + ``up --wait`` never trips a client timeout.
        request_id = uuid.uuid4().hex
        req.request_id = request_id
        poller = asyncio.create_task(self._poll_queue_status(request_id))
        try:
            resp = await self._stub.AcquireComposeProject(
                req, metadata=self._metadata(),
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc
        finally:
            poller.cancel()
            with suppress(asyncio.CancelledError):
                await poller
        return RawComposeAcquireResult(
            rollout_id=resp.rollout_id,
            node_id=resp.node_id,
            main_container_id=resp.main_container_id,
            main_container_name=resp.main_container_name,
            project_name=resp.project_name,
            service_container_ids=dict(resp.service_container_ids),
            queue_wait_s=resp.queue_wait_s,
        )

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str | None = None,
    ) -> None:
        req = rpb.DestroyComposeProjectRequest(rollout_id=rollout_id)
        if project_name is not None:
            req.project_name = project_name
        try:
            await self._stub.DestroyComposeProject(
                req,
                metadata=self._metadata(),
                # Same teardown backstop as destroy_container.
                timeout=_DESTROY_DEADLINE_S,
            )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc

    async def container_put_archive(
        self, *, rollout_id: str, container_id: str,
        target_dir: str, tarball: bytes,
    ) -> None:
        # WS1 (upload twin) — prefer the client-streaming RPC so a large tarball is sent
        # as bounded ``ARCHIVE_CHUNK_BYTES`` slices and no single message hits the 128 MiB
        # gRPC limit. The FIRST frame carries the routing metadata + first slice; the rest
        # carry ``data``; a final ``done=true`` terminator is always sent (even for an empty
        # tarball). Fall back to the unary RPC against an older control plane
        # (UNIMPLEMENTED) — that path keeps the pre-WS1 128 MiB cap. Mirrors
        # ``container_get_archive``'s stream-then-unary pattern.
        async def _chunks() -> AsyncIterator[Any]:
            first = True
            for start in range(0, len(tarball), ARCHIVE_CHUNK_BYTES):
                data = tarball[start:start + ARCHIVE_CHUNK_BYTES]
                last = start + ARCHIVE_CHUNK_BYTES >= len(tarball)
                if first:
                    yield rpb.ContainerPutArchiveChunkRequest(
                        rollout_id=rollout_id, container_id=container_id,
                        target_dir=target_dir, data=data, done=last,
                    )
                    first = False
                else:
                    yield rpb.ContainerPutArchiveChunkRequest(data=data, done=last)
            if first:
                # empty tarball: still emit one terminator carrying the metadata.
                yield rpb.ContainerPutArchiveChunkRequest(
                    rollout_id=rollout_id, container_id=container_id,
                    target_dir=target_dir, done=True,
                )

        try:
            await self._stub.ContainerPutArchiveStream(
                _chunks(), metadata=self._metadata(), timeout=_ARCHIVE_DEADLINE_S,
            )
        except grpc.aio.AioRpcError as exc:
            if exc.code() is grpc.StatusCode.UNIMPLEMENTED:
                try:
                    await self._stub.ContainerPutArchive(
                        rpb.ContainerPutArchiveRequest(
                            rollout_id=rollout_id,
                            container_id=container_id,
                            target_dir=target_dir,
                            tarball=tarball,
                        ),
                        metadata=self._metadata(),
                        timeout=_ARCHIVE_DEADLINE_S,
                    )
                except grpc.aio.AioRpcError as exc2:
                    raise _rehydrate_xrlenv_error(exc2) from exc2
                return
            raise _rehydrate_xrlenv_error(exc) from exc

    async def container_get_archive(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> bytes:
        req = rpb.ContainerGetArchiveRequest(
            rollout_id=rollout_id,
            container_id=container_id,
            source_path=source_path,
        )
        # WS1 — prefer the server-streaming RPC so a large archive arrives as
        # bounded chunks and no single message hits the 128 MiB gRPC limit.
        # Reassemble in stream order (terminator = chunk with ``done=true``).
        # Fall back to the unary RPC against an older control plane that
        # doesn't implement the stream (UNIMPLEMENTED) — that path keeps the
        # pre-WS1 128 MiB cap, which is the best an old server can do.
        try:
            parts: list[bytes] = []
            # ``timeout`` bounds the WHOLE stream (grpc.aio applies it as the
            # call deadline), so a wedged node can't keep the archive stream
            # open forever during log-salvage.
            async for chunk in self._stub.ContainerGetArchiveStream(
                req, metadata=self._metadata(), timeout=_ARCHIVE_DEADLINE_S,
            ):
                if chunk.data:
                    parts.append(bytes(chunk.data))
                if chunk.done:
                    break
            return b"".join(parts)
        except grpc.aio.AioRpcError as exc:
            if exc.code() is grpc.StatusCode.UNIMPLEMENTED:
                try:
                    resp = await self._stub.ContainerGetArchive(
                        req, metadata=self._metadata(),
                        timeout=_ARCHIVE_DEADLINE_S,
                    )
                except grpc.aio.AioRpcError as exc2:
                    raise _rehydrate_xrlenv_error(exc2) from exc2
                return bytes(resp.tarball)
            raise _rehydrate_xrlenv_error(exc) from exc

    async def container_exec_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> AsyncIterator[RawExecChunk]:
        req = rpb.ContainerExecStreamRequest(
            rollout_id=rollout_id, container_id=container_id,
            cmd=cmd, timeout_s=timeout_s,
        )
        if cwd is not None:
            req.cwd = cwd
        if env:
            for k, v in env.items():
                req.env[k] = v
        if user is not None:
            req.user = user
        # gRPC server-streaming: iterate until the stream ends.
        # ``ContainerExecStream`` returns an async iterator of
        # ``ContainerExecChunkResponse`` directly. ``timeout`` bounds the whole
        # call at the server-side exec ``timeout_s`` plus a margin (the CP
        # already extends the node stream by +30 s), so a legitimate long exec
        # isn't cut off but a wedged stream can't hang forever.
        try:
            async for resp in self._stub.ContainerExecStream(
                req, metadata=self._metadata(),
                timeout=timeout_s + _EXEC_DEADLINE_MARGIN_S,
            ):
                yield RawExecChunk(
                    stdout=bytes(resp.stdout),
                    stderr=bytes(resp.stderr),
                    done=bool(resp.done),
                    exit_code=int(resp.exit_code),
                    timed_out=bool(resp.timed_out),
                )
        except grpc.aio.AioRpcError as exc:
            raise _rehydrate_xrlenv_error(exc) from exc


__all__ = ["ClientTransport", "GrpcClientTransport", "InProcessTransport"]
