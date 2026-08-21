"""Consumer-facing rollout service surface (spec 05).

In Slice 1 the SDK calls this surface directly through an in-process transport
(see :mod:`xrlenv.client.transport`). In Slice 3 the same surface is exposed
over gRPC; the in-process and gRPC transports both implement the same shape
so the Client never branches on which is in play.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from xrlenv.backends.base import (
    CpuIsolation,
    NetworkPolicy,
    ResourceSpec,
    RuntimeLimits,
)
from xrlenv.backends.egress import EgressAllowlist
from xrlenv.control.admission import DEFAULT_QUEUE_TIMEOUT_S
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.raw_container_service import (
    RawComposeAcquireResult,
    RawContainerCoordinator,
)
from xrlenv.control.state import NodeRecord
from xrlenv.types import (
    Action,
    CancelGroupReport,
    Observation,
    StepResult,
    TerminateRawGroupReport,
    Trajectory,
)


class StartRolloutRequest(BaseModel):
    """Consumer-side request shape for ``RolloutService.start_rollout``."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    template: str
    init: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    task_key: str | None = None
    group_id: str | None = None
    owner_id: str = "default"
    """Tenant the caller acts as (multi-user). **Server-authoritative**: the
    gRPC servicer overwrites this from the verified bearer token before the
    request reaches the coordinator, so a client cannot spoof another owner.
    Stays ``"default"`` for the in-process / embedded path (single-tenant)."""
    deadline: Any = None  # xrlenv.types.Deadline; Any to keep service.py decoupled
    backend: str | None = None
    """User-side policy. The plug-in author does not know which
    sandbox runtime the operator wants — different VMs have different
    capabilities (KVM-or-not, etc.) — so the consumer supplies it
    here, sourced from the run-config (or per-rollout SDK kwarg).
    Coordinator falls back to :data:`~xrlenv.control.defaults.DEFAULT_BACKEND`
    when this is ``None``. An unknown backend is rejected later by the
    scheduler's node-capability check (no node advertises support), so
    backend doesn't need a literal type the way ``network`` does."""
    network: NetworkPolicy | None = None
    """User-side policy. Typed as
    :data:`xrlenv.backends.base.NetworkPolicy` so pydantic rejects typos
    at construction time — the Docker backend treats any unknown string
    as bridge networking, which would be a "fail-open" security
    footgun if a typo like ``"non"`` slipped through. Pattern A's
    resolver overrides this per-task in Slice 9b; for Pattern B / Simple
    the run-config supplies it. Coordinator falls back to
    :data:`~xrlenv.control.defaults.DEFAULT_NETWORK` when ``None``."""


class StartRolloutResponse(BaseModel):
    """Response shape: ``rollout_id`` plus the first observation + the
    template's ``reward.mode`` so the SDK can validate ``reward_fn``
    presence at call time per spec 05.
    """

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    init_obs: Observation = None
    reward_mode: str = "env_step"


class RolloutService(Protocol):
    """Consumer-facing async surface (matches SDK's transport interface)."""

    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse: ...
    async def step(self, rollout_id: str, action: Action) -> StepResult: ...
    async def finish(self, rollout_id: str) -> Trajectory: ...
    async def cancel(self, rollout_id: str, reason: str) -> Trajectory: ...
    async def cancel_group(
        self, group_id: str, reason: str, *, owner_id: str | None = None,
    ) -> CancelGroupReport: ...
    async def terminate_raw_group(
        self, group_id: str, reason: str, *, owner_id: str | None = None,
    ) -> TerminateRawGroupReport: ...
    async def replay(self, rollout_id: str) -> Trajectory: ...
    async def heartbeat(self, rollout_id: str) -> None: ...
    async def heartbeat_many(self, rollout_ids: list[str]) -> None: ...
    async def set_final_reward(self, rollout_id: str, final_reward: float) -> None: ...
    async def list_nodes(self) -> list[NodeRecord]: ...

    # P1.7.A.1 — raw container session for case 2/3 evaluation harnesses.
    # Implementations route to ``RawContainerCoordinator``; see
    # ``xrlenv/control/raw_container_service.py``.

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
        request_id: str | None = None,
        owner_id: str = "default",
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        session_deadline_s: float | None = None,
        cpu_limit: float | None = None,
        mem_limit_bytes: int | None = None,
        cpu_isolation: CpuIsolation = CpuIsolation.OFF,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawAcquireResult: ...

    # P1.7.B.2 — operator-side FFD bin-packing. NOT part of the
    # consumer-facing surface; invoked only by the ``xrlenv images
    # plan`` operator CLI.
    async def plan_image_distribution(
        self,
        *,
        rows: list[Any],
        eager_prefetch: bool = False,
    ) -> list[Any]: ...

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
        self,
        *,
        rollout_id: str,
        container_id: str,
        force: bool = True,
    ) -> None: ...

    async def acquire_compose_project(
        self,
        *,
        compose_yaml: str,
        images: list[str],
        footprint: ResourceSpec,
        main_service: str = "main",
        project_name: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        request_id: str | None = None,
        labels: dict[str, str] | None = None,
        owner_id: str = "default",
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        session_deadline_s: float | None = None,
        up_timeout_s: float | None = None,
    ) -> RawComposeAcquireResult: ...

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str | None = None,
    ) -> None: ...

    async def container_put_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        target_dir: str,
        tarball: bytes,
    ) -> None: ...

    async def container_get_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
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
    ) -> Any:  # AsyncIterator[RawExecChunk] — Any to match the
        # NodeTransport Protocol's lightweight typing.
        ...


@dataclass
class RawAcquireResult:
    """Wire-friendly subset of ``RawContainerSession`` returned to
    the consumer via the AcquireContainer RPC. Strips the live
    ``NodeTransport`` reference (transport-internal) and timestamp
    (operator-internal) — only the fields the consumer's docker-py
    drop-in needs."""

    rollout_id: str
    container_id: str
    container_name: str
    node_id: str
    queue_wait_s: float = 0.0
    """Issue #18 (Ask #1) — seconds spent in the admission queue
    before placement. ``0.0`` on the fast path. The docker-py
    drop-in surfaces this to the consumer as an over-request signal."""


@dataclass
class RawExecResult:
    """Wire-friendly exec result returned by ContainerExec.

    Mirrors the proto ``ContainerExecResponse`` shape; bytes for
    stdout/stderr because exec output is binary-by-nature (test
    output, diff bytes, etc.)."""

    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool


@dataclass
class RawExecChunk:
    """One chunk yielded by the streaming-exec iterator. Mirrors
    the proto ``ContainerExecChunk`` shape — bytes since the
    prior chunk plus a terminator flag. The terminator chunk
    (``done=True``) carries the final ``exit_code`` /
    ``timed_out``."""

    stdout: bytes
    stderr: bytes
    done: bool
    exit_code: int
    timed_out: bool


class CoordinatorRolloutService:
    """In-process implementation backed by a :class:`RolloutCoordinator`."""

    def __init__(
        self,
        coordinator: RolloutCoordinator,
        *,
        raw_container_coordinator: RawContainerCoordinator | None = None,
    ) -> None:
        self._coordinator = coordinator
        # P1.7.A.1 — optional raw-container coordinator. ``None`` is
        # the legacy single-mode shape (case 1 RL-training only); a
        # configured instance enables the AcquireContainer / Exec /
        # Destroy RPCs. ``LocalRuntime`` and ``DistributedRuntime``
        # construct + inject one wired to the same scheduler the
        # case-1 path uses.
        self._raw_container_coordinator = raw_container_coordinator

    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
        rollout_id, obs = await self._coordinator.start_rollout(
            template_name=req.template,
            init=req.init,
            request_id=req.request_id,
            task_key=req.task_key,
            group_id=req.group_id,
            owner_id=req.owner_id,
            deadline=req.deadline,
            backend=req.backend,
            network=req.network,
        )
        # Carry the template's reward.mode in the response so the SDK can
        # validate that consumer_final templates were called with reward_fn=
        # at start_rollout time (spec 05 §"Error model" — RewardFnRequired).
        manifest = self._coordinator.catalog.get(req.template)
        return StartRolloutResponse(
            rollout_id=rollout_id, init_obs=obs, reward_mode=manifest.reward.mode
        )

    async def step(self, rollout_id: str, action: Action) -> StepResult:
        return await self._coordinator.step(rollout_id, action)

    async def finish(self, rollout_id: str) -> Trajectory:
        return await self._coordinator.finish(rollout_id)

    async def cancel(self, rollout_id: str, reason: str) -> Trajectory:
        return await self._coordinator.cancel(rollout_id, reason)

    async def cancel_group(
        self, group_id: str, reason: str, *, owner_id: str | None = None,
    ) -> CancelGroupReport:
        return await self._coordinator.cancel_group(
            group_id, reason, owner_id=owner_id,
        )

    async def terminate_raw_group(
        self, group_id: str, reason: str, *, owner_id: str | None = None,
    ) -> TerminateRawGroupReport:
        coord = self._require_raw_coordinator()
        return await coord.terminate_raw_group(
            group_id, reason, owner_id=owner_id,
        )

    def rollout_owner(self, rollout_id: str) -> str | None:
        """Owner of a gym/step rollout by id (audit M2 — servicer scope check)."""
        return self._coordinator.get_rollout_owner(rollout_id)

    def raw_session_owner(self, rollout_id: str) -> str | None:
        """Owner of a raw (case-2/3) session by id (audit M2)."""
        coord = self._raw_container_coordinator
        if coord is None:
            return None
        getter = getattr(coord, "session_owner", None)
        return getter(rollout_id) if getter is not None else None

    async def replay(self, rollout_id: str) -> Trajectory:
        # Coordinator.replay is sync (sink read is local file IO); wrap so the
        # async surface is uniform across both in-proc and gRPC transports.
        return self._coordinator.replay(rollout_id)

    async def heartbeat(self, rollout_id: str) -> None:
        await self.heartbeat_many([rollout_id])

    async def heartbeat_many(self, rollout_ids: list[str]) -> None:
        # A heartbeat may target a raw-container session OR a gym/step rollout
        # (the SDK batches live raw-session ids here so heartbeat cost scales
        # with consumer processes, not sessions). Route to both: the raw
        # coordinator stamps the ids it owns (ignoring the rest), and the
        # gym/step idle-TTL ``touch`` is idempotent — a no-op for ids it
        # doesn't watch. So each id lands wherever it belongs, no filtering.
        if self._raw_container_coordinator is not None:
            self._raw_container_coordinator.mark_heartbeat(rollout_ids)
        for rollout_id in rollout_ids:
            await self._coordinator.touch(rollout_id)

    async def set_final_reward(self, rollout_id: str, final_reward: float) -> None:
        # Slice 4.5: consumer_final back-fill — the consumer-side reward_fn
        # ran on the SDK side after finish; we update the state-store row +
        # the trajectory sink's meta.json so future replay sees the
        # canonical value (spec 02 RewardContract consumer_final).
        await self._coordinator.set_final_reward(rollout_id, final_reward)

    async def list_nodes(self) -> list[NodeRecord]:
        # D21 — cluster-status query for the Consumer SDK. Reads the
        # NodeRegistry's persistent mirror in the state store; same data
        # the operator CLI's ``xrlenv nodes`` and the admin panel's
        # ``/nodes`` view consume. Sync state-store call wrapped to keep
        # the async surface uniform across in-process and gRPC transports.
        return self._coordinator.list_nodes()

    # ── Raw container session (P1.7.A.1) ──────────────────────────────────

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
        request_id: str | None = None,
        owner_id: str = "default",
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        session_deadline_s: float | None = None,
        cpu_limit: float | None = None,
        mem_limit_bytes: int | None = None,
        cpu_isolation: CpuIsolation = CpuIsolation.OFF,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawAcquireResult:
        coord = self._require_raw_coordinator()
        session = await coord.acquire(
            image=image,
            command=command,
            entrypoint=entrypoint,
            user=user,
            cap_add=cap_add,
            devices=devices,
            privileged=privileged,
            network_mode=network_mode,
            binds=binds,
            name=name,
            labels=labels,
            environment=environment,
            task_key=task_key,
            request_id=request_id,
            owner_id=owner_id,
            ensure_image_present=ensure_image_present,
            userns_mode=userns_mode,
            acquire_timeout_s=acquire_timeout_s,
            queue_timeout_s=queue_timeout_s,
            session_deadline_s=session_deadline_s,
            cpu_limit=cpu_limit,
            mem_limit_bytes=mem_limit_bytes,
            cpu_isolation=cpu_isolation,
            runtime_limits=runtime_limits,
            container_runtime=container_runtime,
        )
        return RawAcquireResult(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
            container_name=session.container_name,
            node_id=session.node_id,
            queue_wait_s=session.queue_wait_s,
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
        coord = self._require_raw_coordinator()
        result = await coord.exec(
            rollout_id=rollout_id,
            container_id=container_id,
            cmd=cmd,
            timeout_s=timeout_s,
            cwd=cwd,
            env=env,
            user=user,
        )
        return RawExecResult(
            exit_code=int(result.get("exit_code") or 0),
            stdout=bytes(result.get("stdout") or b""),
            stderr=bytes(result.get("stderr") or b""),
            timed_out=bool(result.get("timed_out") or False),
        )

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
    ) -> None:
        coord = self._require_raw_coordinator()
        await coord.apply_egress(
            rollout_id=rollout_id,
            container_id=container_id,
            allowlist=allowlist,
            dns_resolver=dns_resolver,
        )

    async def destroy_container(
        self,
        *,
        rollout_id: str,
        container_id: str,
        force: bool = True,
    ) -> None:
        coord = self._require_raw_coordinator()
        await coord.destroy(
            rollout_id=rollout_id,
            container_id=container_id,
            force=force,
        )

    async def acquire_compose_project(
        self,
        *,
        compose_yaml: str,
        images: list[str],
        footprint: ResourceSpec,
        main_service: str = "main",
        project_name: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        request_id: str | None = None,
        labels: dict[str, str] | None = None,
        owner_id: str = "default",
        queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        session_deadline_s: float | None = None,
        up_timeout_s: float | None = None,
    ) -> RawComposeAcquireResult:
        """P1.7.C.2 — bring up a multi-service compose project (the consumer-facing
        ``AcquireComposeProject`` RPC). Delegates to the raw-container coordinator."""
        coord = self._require_raw_coordinator()
        return await coord.acquire_compose_project(
            compose_yaml=compose_yaml,
            images=images,
            footprint=footprint,
            main_service=main_service,
            project_name=project_name,
            task_key=task_key,
            group_id=group_id,
            request_id=request_id,
            labels=labels,
            owner_id=owner_id,
            queue_timeout_s=queue_timeout_s,
            session_deadline_s=session_deadline_s,
            up_timeout_s=up_timeout_s,
        )

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str | None = None,
    ) -> None:
        """P1.7.C.2 — down the whole compose project (the ``DestroyComposeProject``
        RPC)."""
        coord = self._require_raw_coordinator()
        await coord.destroy_compose_project(
            rollout_id=rollout_id, project_name=project_name,
        )

    async def container_put_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        target_dir: str,
        tarball: bytes,
    ) -> None:
        coord = self._require_raw_coordinator()
        await coord.put_archive(
            rollout_id=rollout_id,
            container_id=container_id,
            target_dir=target_dir,
            tarball=tarball,
        )

    async def container_get_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
    ) -> bytes:
        coord = self._require_raw_coordinator()
        return await coord.get_archive(
            rollout_id=rollout_id,
            container_id=container_id,
            source_path=source_path,
        )

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
    ) -> Any:  # AsyncIterator[RawExecChunk]
        coord = self._require_raw_coordinator()
        async for chunk in coord.exec_stream(
            rollout_id=rollout_id,
            container_id=container_id,
            cmd=cmd,
            timeout_s=timeout_s,
            cwd=cwd,
            env=env,
            user=user,
        ):
            yield RawExecChunk(
                stdout=bytes(chunk.get("stdout") or b""),
                stderr=bytes(chunk.get("stderr") or b""),
                done=bool(chunk.get("done") or False),
                exit_code=int(chunk.get("exit_code") or 0),
                timed_out=bool(chunk.get("timed_out") or False),
            )

    # P1.7.B.2 — operator-side FFD bin-packing.
    async def plan_image_distribution(
        self,
        *,
        rows: list[Any],
        eager_prefetch: bool = False,
    ) -> list[Any]:
        coord = self._require_raw_coordinator()
        return await coord.plan_image_distribution(
            rows=rows, eager_prefetch=eager_prefetch,
        )

    def _require_raw_coordinator(self) -> RawContainerCoordinator:
        if self._raw_container_coordinator is None:
            from xrlenv.errors import XRLEnvError
            raise XRLEnvError(
                "raw-container session: this control plane was "
                "built without a RawContainerCoordinator. Use "
                "build_local_runtime / build_distributed_runtime "
                "(both wire one in by default 2026-05-06+).",
            )
        return self._raw_container_coordinator


__all__ = [
    "CoordinatorRolloutService",
    "RolloutService",
    "StartRolloutRequest",
    "StartRolloutResponse",
]
