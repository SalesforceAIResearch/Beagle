"""gRPC servicer that exposes :class:`RolloutService` over the wire (spec 05).

Wraps the existing in-process :class:`RolloutService` Protocol and
translates each unary RPC into the matching method call. Errors round-
trip through gRPC trailing metadata (``xrlenv-error-kind`` /
``xrlenv-error-reason`` / ``xrlenv-error-partial``) so the client-side
:class:`GrpcClientTransport` can rehydrate the original exception
(``RolloutTruncated``, ``RolloutCancelled``, ``RolloutFailed``,
``CapacityExhausted``, etc.) — including the carrier exceptions'
partial trajectories.

Auth is layered on by :class:`BearerScopeInterceptor` (spec 19) when
the servicer is added to the gRPC server in
:py:func:`build_distributed_runtime`. Each RPC declares the
``consumer.rollout`` scope so trainers/smokes need a ``consumer``-role
bearer token to drive a rollout.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import grpc

from xrlenv.api import converters as conv
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.api._pb2 import rollout_control_pb2_grpc as rpb_grpc
from xrlenv.api.constants import ARCHIVE_CHUNK_BYTES
from xrlenv.backends.base import ResourceSpec, resolve_cpu_isolation
from xrlenv.backends.egress import EgressAllowlist, EgressRule
from xrlenv.control.admission import DEFAULT_QUEUE_TIMEOUT_S
from xrlenv.control.service import RolloutService
from xrlenv.errors import (
    AuthDenied,
    BackendCapabilityMissing,
    CapacityExhausted,
    ControlPlaneLost,
    MountDenied,
    NodeCommandTimeout,
    NodeLost,
    ReplayUnavailable,
    RewardFnRequired,
    RolloutCancelled,
    RolloutFailed,
    RolloutTruncated,
    SessionReaped,
    TemplateUnknown,
    XRLEnvError,
)

LOGGER = logging.getLogger(__name__)


# Trailing-metadata keys used to carry structured error details across
# the wire. The client transport reads these and rehydrates the right
# xrlenv exception subclass.
_KIND_META_KEY = "xrlenv-error-kind"
_REASON_META_KEY = "xrlenv-error-reason"
# Partial trajectory is a binary proto payload. Per gRPC convention the
# metadata key for binary values must end in `-bin` so middleware
# doesn't try to interpret it as text.
_PARTIAL_META_KEY = "xrlenv-error-partial-bin"


# Map xrlenv exception class -> gRPC StatusCode. Mirrors the spec-05
# §"Error model" table: caller-side input errors map to
# INVALID_ARGUMENT, capacity-exhausted to RESOURCE_EXHAUSTED, etc.
# Carrier exceptions (RolloutTruncated/RolloutCancelled/RolloutFailed)
# use DEADLINE_EXCEEDED/CANCELLED/ABORTED with structured trailing
# metadata that carries the partial trajectory; spec 05 §"Error model —
# gRPC wire shape" blesses the structured-error-over-gRPC model so non-
# Python consumers can branch on the status code AND read the partial
# via the ``xrlenv-error-partial-bin`` metadata key.
_EXC_TO_CODE: dict[type[XRLEnvError], grpc.StatusCode] = {
    TemplateUnknown:           grpc.StatusCode.INVALID_ARGUMENT,
    RewardFnRequired:          grpc.StatusCode.INVALID_ARGUMENT,
    BackendCapabilityMissing:  grpc.StatusCode.FAILED_PRECONDITION,
    MountDenied:               grpc.StatusCode.PERMISSION_DENIED,
    AuthDenied:                grpc.StatusCode.UNAUTHENTICATED,
    CapacityExhausted:         grpc.StatusCode.RESOURCE_EXHAUSTED,
    ControlPlaneLost:          grpc.StatusCode.UNAVAILABLE,
    NodeLost:                  grpc.StatusCode.UNAVAILABLE,
    # A node reply-timeout. DEADLINE_EXCEEDED so the client maps it to
    # NodeCommandTimeout even on the unmarked-fallback path (see
    # _classify_unmarked_rpc_error) — not the UNKNOWN default that read as a bare
    # XRLEnvError before.
    NodeCommandTimeout:        grpc.StatusCode.DEADLINE_EXCEEDED,
    RolloutTruncated:          grpc.StatusCode.DEADLINE_EXCEEDED,
    RolloutCancelled:          grpc.StatusCode.CANCELLED,
    RolloutFailed:             grpc.StatusCode.ABORTED,
    ReplayUnavailable:         grpc.StatusCode.NOT_FOUND,
    # A session the platform tore down (liveness quarantine, wall-clock deadline,
    # or a node-side orphan seal). FAILED_PRECONDITION, not NOT_FOUND:
    # the id is real and the platform knows exactly what happened to it, so the
    # consumer can tell a reap apart from a stale/unknown handle and retry.
    SessionReaped:             grpc.StatusCode.FAILED_PRECONDITION,
}


async def _abort_with_xrlenv_error(
    context: grpc.aio.ServicerContext[Any, Any],
    exc: XRLEnvError,
) -> None:
    """Translate an xrlenv exception into a gRPC abort with structured
    trailing metadata. Always raises (via the awaited ``context.abort``);
    annotated as returning ``None`` to satisfy the static type checker
    at call sites.
    """
    code = _EXC_TO_CODE.get(type(exc), grpc.StatusCode.UNKNOWN)
    md: list[tuple[str, str | bytes]] = [
        (_KIND_META_KEY, type(exc).__name__),
    ]
    reason = getattr(exc, "reason", None)
    if isinstance(reason, str):
        md.append((_REASON_META_KEY, reason))
    partial = getattr(exc, "partial", None)
    if partial is not None:
        try:
            md.append(
                (_PARTIAL_META_KEY, conv.trajectory_to_proto(partial).SerializeToString())
            )
        except Exception:
            LOGGER.exception(
                "RolloutControl: failed to serialise partial trajectory for %s",
                type(exc).__name__,
            )
    context.set_trailing_metadata(md)
    # ``await context.abort(...)`` is how grpc-aio propagates the status
    # to the channel. The call raises ``AbortError`` so the function
    # never returns normally; the ``raise`` after each call site is
    # belt-and-suspenders for mypy.
    await context.abort(code, str(exc))


class RolloutControlServicer(rpb_grpc.RolloutControlServicer):
    """Wrap a :class:`RolloutService` and serve it over gRPC.

    The servicer holds a reference to the in-process service (the same
    one ``Client.in_process`` uses) and translates each RPC into the
    matching method call. There's no extra state on the servicer
    itself — the service object is the source of truth for rollout
    state, idempotency, and the in-flight session map.
    """

    def __init__(
        self,
        service: RolloutService,
        admission: Any = None,
        *,
        token_store: Any = None,
    ) -> None:
        self._service = service
        # Stage-2 — the AdmissionQueue, for the QueueStatus RPC. Optional
        # so single-node setups without a queue still construct cleanly;
        # QueueStatus then reports an empty queue.
        self._admission = admission
        # Multi-user (Slice B) — the TokenStore used to resolve the caller's
        # owner_id from the verified bearer so the server stamps every
        # rollout / raw acquire with the authenticated tenant. ``None`` (or an
        # empty store) means single-tenant: owner stays ``"default"``. The
        # auth interceptor has already verified + scoped the bearer by the
        # time a method runs, so this resolve never rejects — it only reads
        # the owner off the same token.
        self._token_store = token_store

    def _owner_from_context(
        self, context: grpc.aio.ServicerContext[Any, Any],
    ) -> str:
        """Resolve the caller's ``owner_id`` from the request's bearer token.

        Server-authoritative: the value comes from the verified token, never
        from a client-supplied request field. Falls back to ``"default"`` when
        no TokenStore is wired or the store is empty (single-tenant / embedded
        / no-auth smoke), and defensively when the token can't be resolved.
        """
        store = self._token_store
        if store is None or store.is_empty:
            return "default"
        token: str | None = None
        for key, value in (context.invocation_metadata() or ()):
            if key.lower() != "authorization":
                continue
            raw = (
                value.decode("utf-8", "replace")
                if isinstance(value, bytes) else str(value)
            )
            raw = raw.strip()
            token = (
                raw[len("Bearer "):].strip()
                if raw.startswith("Bearer ") else raw
            )
            break
        identity = store.verify(token)
        return identity.owner_id if identity is not None else "default"

    # ── Owner enforcement on follow-up RPCs (audit M2) ───────────────────────

    def _scoped_owner(self, context: grpc.aio.ServicerContext[Any, Any]) -> str | None:
        """The caller's owner when multi-tenant auth is active, else ``None``.

        ``None`` means "no cross-owner enforcement" — the single-tenant /
        embedded / no-auth path (empty TokenStore), where every caller is the
        implicit owner. With auth on, returns the verified tenant.
        """
        store = self._token_store
        if store is None or store.is_empty:
            return None
        return self._owner_from_context(context)

    async def _guard_owner(
        self,
        context: grpc.aio.ServicerContext[Any, Any],
        stored_owner: str | None,
    ) -> None:
        """Abort PERMISSION_DENIED if the caller doesn't own ``stored_owner``.

        No-op when auth is off (single-tenant) or the id is unknown (``None``
        stored owner — the underlying method then produces its own NotFound).
        """
        caller = self._scoped_owner(context)
        if caller is None or stored_owner is None:
            return
        if caller != stored_owner:
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "rollout or session belongs to a different owner",
            )

    def _rollout_owner(self, rollout_id: str) -> str | None:
        getter = getattr(self._service, "rollout_owner", None)
        return getter(rollout_id) if getter is not None else None

    def _raw_owner(self, rollout_id: str) -> str | None:
        getter = getattr(self._service, "raw_session_owner", None)
        return getter(rollout_id) if getter is not None else None

    async def StartRollout(
        self,
        request: rpb.StartRolloutRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.StartRolloutResponse:
        try:
            req = conv.start_rollout_request_from_proto(request)
            # Server-stamp the authenticated owner; ignore any value the
            # proto/client may have carried (the proto has no owner field
            # today, but this keeps the boundary explicit).
            req = req.model_copy(update={"owner_id": self._owner_from_context(context)})
            resp = await self._service.start_rollout(req)
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise  # unreachable; abort() raises
        return conv.start_rollout_response_to_proto(resp)

    async def Step(
        self,
        request: rpb.StepRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.StepResponse:
        await self._guard_owner(context, self._rollout_owner(request.rollout_id))
        try:
            action: Any = conv.json_load(request.action_json)
            result = await self._service.step(request.rollout_id, action)
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return conv.step_result_to_proto(result)

    async def Finish(
        self,
        request: rpb.FinishRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.FinishResponse:
        await self._guard_owner(context, self._rollout_owner(request.rollout_id))
        try:
            traj = await self._service.finish(request.rollout_id)
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.FinishResponse(trajectory=conv.trajectory_to_proto(traj))

    async def Cancel(
        self,
        request: rpb.CancelRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.CancelResponse:
        await self._guard_owner(context, self._rollout_owner(request.rollout_id))
        try:
            traj = await self._service.cancel(request.rollout_id, request.reason)
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        except KeyError:
            # Cancel must be idempotent. ``service.cancel`` resolves the
            # rollout via ``state.get_rollout``, which raises ``KeyError``
            # for an id that was never placed / already reaped / belongs to
            # the raw-container path (not the gym/step ``rollouts`` table).
            # That is exactly what a consumer cancelling its queue-timeout
            # / deadline victims hits — and without this catch the bare
            # KeyError escaped as an unhandled servicer crash (the prod
            # "Unexpected [KeyError] raised by servicer method … /Cancel"
            # spam). Report it as a clean NOT_FOUND: there is nothing left
            # to cancel.
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"rollout {request.rollout_id!r} not found or already "
                "terminal — nothing to cancel",
            )
            raise
        return rpb.CancelResponse(trajectory=conv.trajectory_to_proto(traj))

    async def CancelGroup(
        self,
        request: rpb.CancelGroupRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.CancelGroupResponse:
        try:
            # Owner-scope the group cancel (audit M2): a scoped caller cancels
            # only their own rollouts in the group, never another tenant's.
            report = await self._service.cancel_group(
                request.group_id, request.reason,
                owner_id=self._scoped_owner(context),
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.CancelGroupResponse(
            report=conv.cancel_group_report_to_proto(report),
        )

    async def TerminateRawGroup(
        self,
        request: rpb.TerminateRawGroupRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.TerminateRawGroupResponse:
        try:
            # Owner-scope the raw-group teardown (audit M2): a scoped caller tears
            # down only their own raw containers in the group, never another tenant's.
            report = await self._service.terminate_raw_group(
                request.group_id, request.reason,
                owner_id=self._scoped_owner(context),
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.TerminateRawGroupResponse(
            report=conv.terminate_raw_group_report_to_proto(report),
        )

    async def Replay(
        self,
        request: rpb.ReplayRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.ReplayResponse:
        await self._guard_owner(context, self._rollout_owner(request.rollout_id))
        try:
            traj = await self._service.replay(request.rollout_id)
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.ReplayResponse(trajectory=conv.trajectory_to_proto(traj))

    async def Heartbeat(
        self,
        request: rpb.HeartbeatRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.HeartbeatResponse:
        # Batched form (``rollout_ids``) supersedes the single ``rollout_id``;
        # fall back to the single id for older clients / gym/step heartbeats.
        ids = list(request.rollout_ids)
        if not ids and request.rollout_id:
            ids = [request.rollout_id]
        # Owner-scope (audit M2): a scoped caller may only heartbeat their own
        # ids — drop any owned by another tenant so they can't keep a stranger's
        # session alive. Unknown ids (owner None) stay; they're harmless no-ops.
        caller = self._scoped_owner(context)
        if caller is not None:
            ids = [
                rid for rid in ids
                if (self._rollout_owner(rid) or self._raw_owner(rid)) in (None, caller)
            ]
        try:
            await self._service.heartbeat_many(ids)
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.HeartbeatResponse()

    async def SetFinalReward(
        self,
        request: rpb.SetFinalRewardRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.SetFinalRewardResponse:
        await self._guard_owner(context, self._rollout_owner(request.rollout_id))
        try:
            await self._service.set_final_reward(
                request.rollout_id, request.final_reward,
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.SetFinalRewardResponse()

    async def ListNodes(
        self,
        request: rpb.ListNodesRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.ListNodesResponse:
        # D21 — cluster-status query for the Consumer SDK. Reads
        # ``state.list_nodes()`` (the NodeRegistry's persistent mirror)
        # via the service, converts each row to the wire ``NodeInfo``,
        # and returns. No side effects + no XRLEnvError sources today,
        # but the try/except matches the surrounding RPC shape so a
        # future state-store error class lands cleanly.
        try:
            records = await self._service.list_nodes()
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.ListNodesResponse(
            nodes=[conv.node_info_to_proto(r) for r in records],
        )

    # ── P1.7.A.1 raw-container RPCs ─────────────────────────────────────────

    async def AcquireContainer(
        self,
        request: rpb.AcquireContainerRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.AcquireContainerResponse:
        # P0b — container-shape RuntimeLimits. Unset proto message → None;
        # the node then applies no constraint. Built once here so the P6
        # cpu_isolation derivation below can read its legacy cpu_pinning alias
        # without re-decoding the proto.
        runtime_limits = (
            conv.runtime_limits_from_proto(request.runtime_limits)
            if request.HasField("runtime_limits")
            else None
        )
        # P6 derive-once: the effective CPU-isolation contract is resolved a
        # single time, here at the control-plane ingress, from the explicit
        # wire field (preferred) falling back to the legacy cpu_pinning bool.
        # The resulting mode is threaded down (service → coordinator → stamped
        # on the effective ResourceSpec → node command) — never re-derived.
        cpu_isolation = resolve_cpu_isolation(
            conv.cpu_isolation_from_wire(request.resources.cpu_isolation),
            cpu_pinning=bool(runtime_limits is not None and runtime_limits.cpu_pinning),
        )
        try:
            result = await self._service.acquire_container(
                image=request.image,
                command=list(request.command) if request.command else None,
                # proto3 ``repeated`` can't distinguish "unset" from
                # "empty list"; we treat anything non-empty as "set"
                # (so ``[""]`` — the docker CLI ``--entrypoint ""``
                # idiom — flows through). Unset proto field arrives
                # as an empty list which we map back to None.
                entrypoint=(
                    list(request.entrypoint) if request.entrypoint else None
                ),
                user=request.user or None,
                cap_add=list(request.cap_add) if request.cap_add else None,
                devices=list(request.devices) if request.devices else None,
                privileged=request.privileged,
                network_mode=request.network_mode or None,
                binds=list(request.binds) if request.binds else None,
                name=request.name if request.HasField("name") else None,
                labels=dict(request.labels) if request.labels else None,
                environment=(
                    dict(request.environment) if request.environment else None
                ),
                task_key=(
                    request.task_key if request.HasField("task_key") else None
                ),
                # Stage-2 — the consumer-supplied request_id; lets the
                # consumer poll QueueStatus for this acquire's FIFO
                # position while it waits. Empty (pre-Stage-2 consumer)
                # → None; the request is then simply not pollable.
                request_id=(
                    request.request_id
                    if request.HasField("request_id")
                    else None
                ),
                # Multi-user (Slice B): server-stamp the authenticated owner.
                owner_id=self._owner_from_context(context),
                # Wire field is negative-form; SDK kwarg is
                # positive-form (the new default UX).
                ensure_image_present=not request.strict_image_check,
                # B5.4 — proto3 default-empty string → "host"; any
                # other value passes through (today: "remap" honors
                # the docker daemon's userns-remap config).
                userns_mode=request.userns_mode or "host",
                # Issue #12 — proto3 ``0.0`` is the "unset" sentinel
                # (use the server's default 600 s); positive values
                # override per-call.
                acquire_timeout_s=(
                    request.acquire_timeout_s
                    if request.acquire_timeout_s > 0
                    else None
                ),
                # Issue #18 (Ask #1) / Stage 2 — proto3 ``0.0`` means
                # "use the server default" (DEFAULT_QUEUE_TIMEOUT_S, 24 h);
                # a small positive value is the consumer's explicit
                # fail-fast opt-in on the admission-queue wait.
                queue_timeout_s=(
                    request.queue_timeout_s
                    if request.queue_timeout_s > 0
                    else DEFAULT_QUEUE_TIMEOUT_S
                ),
                # Issue #18 — proto3 ``0.0`` → server default cap;
                # a positive value caps this session's lifetime.
                session_deadline_s=(
                    request.session_deadline_s
                    if request.session_deadline_s > 0
                    else None
                ),
                # P0a — harness CPU/memory request as a scheduling
                # input. proto3 default-zero on a ResourceSpec field
                # means "harness didn't specify"; a positive value is
                # an explicit override the coordinator merges with the
                # raw-container default budget.
                cpu_limit=(
                    request.resources.cpu_limit
                    if request.resources.cpu_limit > 0
                    else None
                ),
                mem_limit_bytes=(
                    request.resources.mem_limit_bytes
                    if request.resources.mem_limit_bytes > 0
                    else None
                ),
                # P0b — container-shape RuntimeLimits (built once, above).
                runtime_limits=runtime_limits,
                # P6 — the derive-once CPU-isolation contract (resolved above).
                cpu_isolation=cpu_isolation,
                # §5.1 — OCI runtime selector. Empty proto3 string → None
                # (docker default runtime); gated by allowed_runtimes.
                container_runtime=request.container_runtime or None,
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise  # unreachable; abort() raises
        return rpb.AcquireContainerResponse(
            rollout_id=result.rollout_id,
            container_id=result.container_id,
            container_name=result.container_name,
            node_id=result.node_id,
            queue_wait_s=result.queue_wait_s,
        )

    async def QueueStatus(
        self,
        request: rpb.QueueStatusRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.QueueStatusResponse:
        # Stage-2 — report a request's admission-queue rank so a queued
        # consumer can poll its position. No admission queue wired
        # (single-node setups) → report an empty queue.
        if self._admission is None:
            return rpb.QueueStatusResponse(
                position=0, queue_depth=0, state="not_in_queue",
            )
        # Owner-scope the queue view (audit M1-residual): a scoped caller only
        # sees their own queued requests, so another tenant's request_id reads
        # not_in_queue and no cross-owner position/existence leaks.
        position, depth, state = self._admission.queue_status(
            request.request_id, self._scoped_owner(context),
        )
        return rpb.QueueStatusResponse(
            position=position, queue_depth=depth, state=state,
        )

    async def ContainerExec(
        self,
        request: rpb.ContainerExecRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.ContainerExecResponse:
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            result = await self._service.container_exec(
                rollout_id=request.rollout_id,
                container_id=request.container_id,
                cmd=list(request.cmd),
                timeout_s=request.timeout_s or 30.0,
                cwd=request.cwd if request.HasField("cwd") else None,
                env=dict(request.env) if request.env else None,
                user=request.user if request.HasField("user") else None,
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.ContainerExecResponse(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
        )

    async def ApplyEgress(
        self,
        request: rpb.ApplyEgressRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.ApplyEgressResponse:
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            try:
                allowlist = EgressAllowlist(
                    rules=tuple(
                        EgressRule(cidr=e.cidr, ports=tuple(e.ports) or None)
                        for e in request.allow
                    ),
                )
            except Exception as exc:  # malformed allowlist from the wire
                raise XRLEnvError(f"apply_egress: invalid allowlist: {exc}") from exc
            await self._service.apply_egress(
                rollout_id=request.rollout_id,
                container_id=request.container_id,
                allowlist=allowlist,
                dns_resolver=(
                    request.dns_resolver if request.HasField("dns_resolver") else None
                ),
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.ApplyEgressResponse()

    async def DestroyContainer(
        self,
        request: rpb.DestroyContainerRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.DestroyContainerResponse:
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            await self._service.destroy_container(
                rollout_id=request.rollout_id,
                container_id=request.container_id,
                force=request.force,
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.DestroyContainerResponse()

    # ── P1.7.C.2 — multi-service compose project ──────────────────────────────

    async def AcquireComposeProject(
        self,
        request: rpb.AcquireComposeProjectRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.AcquireComposeProjectResponse:
        try:
            result = await self._service.acquire_compose_project(
                compose_yaml=request.compose_yaml,
                images=list(request.images),
                # The plugin-computed whole-stack footprint (scheduler reserve).
                # Built inline: rollout_control's ResourceSpec is a distinct proto
                # type from node_control's (what conv.resource_spec_from_proto
                # expects), and carries no disk field.
                footprint=ResourceSpec(
                    cpu_request=request.footprint.cpu_request,
                    cpu_limit=(
                        request.footprint.cpu_limit
                        or request.footprint.cpu_request
                    ),
                    mem_request_bytes=request.footprint.mem_request_bytes,
                    mem_limit_bytes=(
                        request.footprint.mem_limit_bytes
                        or request.footprint.mem_request_bytes
                    ),
                    disk_request_bytes=0,
                ),
                main_service=request.main_service or "main",
                project_name=(
                    request.project_name if request.project_name else None
                ),
                task_key=(
                    request.task_key if request.HasField("task_key") else None
                ),
                group_id=(
                    request.group_id if request.HasField("group_id") else None
                ),
                request_id=(
                    request.request_id
                    if request.HasField("request_id")
                    else None
                ),
                labels=dict(request.labels) if request.labels else None,
                owner_id=self._owner_from_context(context),
                # proto3 ``0.0`` = "use the server default"; positive = explicit.
                queue_timeout_s=(
                    request.queue_timeout_s
                    if request.queue_timeout_s > 0
                    else DEFAULT_QUEUE_TIMEOUT_S
                ),
                session_deadline_s=(
                    request.session_deadline_s
                    if request.session_deadline_s > 0
                    else None
                ),
                up_timeout_s=(
                    request.up_timeout_s if request.up_timeout_s > 0 else None
                ),
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise  # unreachable; abort() raises
        return rpb.AcquireComposeProjectResponse(
            rollout_id=result.rollout_id,
            node_id=result.node_id,
            main_container_id=result.main_container_id,
            main_container_name=result.main_container_name,
            project_name=result.project_name,
            service_container_ids=dict(result.service_container_ids),
            queue_wait_s=result.queue_wait_s,
        )

    async def DestroyComposeProject(
        self,
        request: rpb.DestroyComposeProjectRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.DestroyComposeProjectResponse:
        # Multi-tenant owner scope (like DestroyContainer): a caller may only tear
        # down a compose project they own — the main session's owner is looked up
        # server-side, never trusted from the request.
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            await self._service.destroy_compose_project(
                rollout_id=request.rollout_id,
                project_name=request.project_name or None,
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.DestroyComposeProjectResponse()

    async def ContainerPutArchive(
        self,
        request: rpb.ContainerPutArchiveRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.ContainerPutArchiveResponse:
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            await self._service.container_put_archive(
                rollout_id=request.rollout_id,
                container_id=request.container_id,
                target_dir=request.target_dir,
                tarball=bytes(request.tarball),
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.ContainerPutArchiveResponse()

    async def ContainerGetArchive(
        self,
        request: rpb.ContainerGetArchiveRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.ContainerGetArchiveResponse:
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            tarball = await self._service.container_get_archive(
                rollout_id=request.rollout_id,
                container_id=request.container_id,
                source_path=request.source_path,
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.ContainerGetArchiveResponse(tarball=tarball)

    async def ContainerGetArchiveStream(
        self,
        request: rpb.ContainerGetArchiveRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> AsyncIterator[rpb.ContainerGetArchiveChunkResponse]:
        """WS1 server-streaming variant of :meth:`ContainerGetArchive`.

        The control plane already reassembles the whole tarball from the
        node's chunked node->control replies (that hop is what keeps the
        heartbeat stream intact under a large archive). This streams that
        tarball on to the CONSUMER in ``ARCHIVE_CHUNK_BYTES`` slices so no
        single client-facing message exceeds the gRPC limit — the 128 MiB
        cap that made large ``get_archive`` fetches fail with
        ``CapacityExhausted`` even after the node hop was fixed. A
        terminator chunk (``done=true``, empty ``data``) always ends the
        stream, even for an empty tarball, so the client sees exactly one
        terminator.

        The tarball is buffered once in control-plane memory here (as it
        already is for the unary path); archives are at most a few GiB and
        the control-plane box is provisioned for it. Errors surface as
        gRPC aborts via ``_abort_with_xrlenv_error`` before any chunk is
        sent (the ``raise`` is unreachable but mypy can't see that
        ``context.abort`` is NoReturn-shaped)."""
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            tarball = await self._service.container_get_archive(
                rollout_id=request.rollout_id,
                container_id=request.container_id,
                source_path=request.source_path,
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        for start in range(0, len(tarball), ARCHIVE_CHUNK_BYTES):
            yield rpb.ContainerGetArchiveChunkResponse(
                data=tarball[start:start + ARCHIVE_CHUNK_BYTES], done=False,
            )
        yield rpb.ContainerGetArchiveChunkResponse(data=b"", done=True)

    async def ContainerPutArchiveStream(
        self,
        request_iterator: AsyncIterator[rpb.ContainerPutArchiveChunkRequest],
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.ContainerPutArchiveResponse:
        """WS1 client-streaming variant of :meth:`ContainerPutArchive`.

        The client sends the tarball as bounded ``ARCHIVE_CHUNK_BYTES`` chunks so no
        single client->CP message exceeds the gRPC limit (the same 128 MiB cap that made
        a large ``solution/`` upload — e.g. an oracle bundling a 340 MB SDK — fail on this
        hop). The FIRST chunk carries the routing metadata. The CP reassembles the tarball
        here (buffered once in CP memory, as the unary path already is) and hands it to
        ``container_put_archive``, which re-chunks it onto the node->control heartbeat
        stream (that hop keeps the heartbeat intact under a large upload). A terminator
        (``done=true``) always ends the client stream, even for an empty tarball."""
        rollout_id = ""
        container_id = ""
        target_dir = ""
        parts: list[bytes] = []
        have_meta = False
        async for chunk in request_iterator:
            if not have_meta:
                rollout_id = chunk.rollout_id
                container_id = chunk.container_id
                target_dir = chunk.target_dir
                have_meta = True
                # ownership check as soon as we know the rollout (before any node work).
                await self._guard_owner(context, self._raw_owner(rollout_id))
            if chunk.data:
                parts.append(bytes(chunk.data))
            if chunk.done:
                break
        try:
            await self._service.container_put_archive(
                rollout_id=rollout_id,
                container_id=container_id,
                target_dir=target_dir,
                tarball=b"".join(parts),
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise
        return rpb.ContainerPutArchiveResponse()

    async def ContainerExecStream(
        self,
        request: rpb.ContainerExecStreamRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> AsyncIterator[rpb.ContainerExecChunkResponse]:
        """Server-streaming RPC. Iterates the service's
        ``container_exec_stream`` and yields one wire chunk per
        manager-side chunk. Errors mid-stream surface as gRPC
        aborts via the existing ``_abort_with_xrlenv_error`` shim
        (the ``raise`` is unreachable but mypy doesn't know
        ``context.abort`` is NoReturn-shaped)."""
        await self._guard_owner(context, self._raw_owner(request.rollout_id))
        try:
            async for chunk in self._service.container_exec_stream(
                rollout_id=request.rollout_id,
                container_id=request.container_id,
                cmd=list(request.cmd),
                timeout_s=request.timeout_s or 1800.0,
                cwd=request.cwd if request.HasField("cwd") else None,
                env=dict(request.env) if request.env else None,
                user=request.user if request.HasField("user") else None,
            ):
                yield rpb.ContainerExecChunkResponse(
                    stdout=chunk.stdout,
                    stderr=chunk.stderr,
                    done=chunk.done,
                    exit_code=chunk.exit_code,
                    timed_out=chunk.timed_out,
                )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise

    # ── P1.7.B.2 W7: operator-side FFD bin-packing ────────────────────────

    async def PlanImageDistribution(
        self,
        request: rpb.PlanImageDistributionRequest,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> rpb.PlanImageDistributionResponse:
        from xrlenv.control.image_planner import ImageToPlace

        rows = [
            ImageToPlace(
                image_ref=row.image_ref,
                size_bytes=int(row.size_bytes_hint),
                replication=int(row.replication) if row.replication > 0 else 1,
                benchmark="raw-image-plan",
            )
            for row in request.rows
        ]
        try:
            results = await self._service.plan_image_distribution(
                rows=rows, eager_prefetch=bool(request.eager_prefetch),
            )
        except XRLEnvError as exc:
            await _abort_with_xrlenv_error(context, exc)
            raise

        return rpb.PlanImageDistributionResponse(
            assignments=[
                rpb.ImagePlanResult(
                    image_ref=r.image_ref,
                    preferred_home_node=r.preferred_home_node,
                    status=r.status,
                    error=r.error or "",
                )
                for r in results
            ],
        )


__all__ = [
    "RolloutControlServicer",
]
