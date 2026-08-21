"""``Client`` — the consumer-facing entry point (spec 05).

Phase 0 surface:

- ``Client.rollout(template, init, deadline, ...)`` async context manager
- ``Client.batch_rollout(...)`` *(Slice 2)*
- ``Client.replay(rollout_id)`` *(Slice 2)*

For Slice 1 the client is constructed from an in-process transport via
:py:meth:`Client.in_process`; the gRPC constructor lands in Slice 3.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from xrlenv.backends.base import CpuIsolation, RuntimeLimits
from xrlenv.client.container_session import (
    ClusterComposeSession,
    ClusterContainerSession,
)
from xrlenv.client.session import RewardFn, RolloutSession
from xrlenv.client.transport import ClientTransport, GrpcClientTransport, InProcessTransport
from xrlenv.control.run_config import load_run_config
from xrlenv.control.service import RolloutService, StartRolloutRequest
from xrlenv.errors import (
    RewardFnRequired,
    RolloutCancelled,
    RolloutFailed,
    RolloutTruncated,
)
from xrlenv.types import (
    Action,
    BatchRolloutResult,
    CancelGroupReport,
    Deadline,
    FailedRollout,
    Observation,
    TerminateRawGroupReport,
    Trajectory,
)

LOGGER = logging.getLogger(__name__)


PolicyFn = Callable[[Observation], Awaitable[Action]]
"""``async def policy(obs) -> action`` — what :py:meth:`Client.batch_rollout`
calls per step. Consumers wrap their policy/SGLang/vLLM engine in this shape;
the platform stays consumer-framework-agnostic per spec 05.
"""


# SDK keepalive cadence for raw-container sessions. The control plane reaps a
# raw session whose consumer stopped beating (XRLENV_RAW_LIVENESS_TTL_S, 120 s
# default); this is the beat interval, ~1/4 of that so a couple of dropped
# beats don't false-reap. ``0`` disables the keepalive (sessions then rely on
# the wall-clock deadline only). Env-tunable.
_RAW_HEARTBEAT_INTERVAL_S: float = float(
    os.environ.get("XRLENV_RAW_HEARTBEAT_INTERVAL_S", "30"),
)


class _RawSessionKeepalive:
    """Per-:class:`Client` background loop that batches heartbeats for live raw
    sessions.

    The raw-container path is all unary RPCs, so the control plane can't see a
    dead consumer. Each Client beats its live raw sessions so the server's
    liveness reaper can tell "alive" from "gone" without false-reaping a slow
    one. **One RPC per interval per process** (all live ids batched), so cost
    scales with consumer processes, not sessions. The loop runs on whatever
    event loop is active when the first session registers (the ``from_env``
    runner's loop, or the caller's) and is cancelled by :meth:`Client.close`.
    Heartbeat failures are swallowed so a transient blip never tears the loop
    down — which would let live sessions be reaped.
    """

    def __init__(self, transport: ClientTransport, *, interval_s: float) -> None:
        self._transport = transport
        self._interval_s = interval_s
        self._ids: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        # Set by register() to wake the loop for a prompt opt-in beat.
        self._wake = asyncio.Event()

    def register(self, rollout_id: str) -> None:
        self._ids.add(rollout_id)
        if self._interval_s <= 0:
            return
        if self._task is None or self._task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no running loop (shouldn't happen from async acquire)
            self._task = loop.create_task(self._run())
        # Wake the loop so the just-registered session sends its first beat
        # *now*, not after a full interval — otherwise a consumer killed within
        # the first interval after acquire never heartbeats, so it never opts
        # into the liveness reaper and falls back to the wall-clock deadline
        # (audit M1). Coalesced: a burst of acquires triggers one batched beat.
        self._wake.set()

    def unregister(self, rollout_id: str) -> None:
        self._ids.discard(rollout_id)

    async def _run(self) -> None:
        # Beat FIRST, then wait — so the first beat (the opt-in) is prompt.
        while True:
            # Clear the wake signal BEFORE snapshotting ids. A register() that
            # lands after this point either appears in the snapshot (beaten
            # now) or re-sets _wake while the beat is in flight — in which case
            # the wait below returns immediately and the next iteration beats
            # it. Clearing *after* the beat would instead drop a registration
            # made during the RPC, delaying its opt-in a full interval.
            self._wake.clear()
            ids = list(self._ids)
            if ids:
                try:
                    await self._transport.heartbeat_many(ids)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.debug(
                        "raw-session keepalive heartbeat failed; will retry",
                        exc_info=True,
                    )
            # Wait the interval, or wake early when a new session registers
            # (so its opt-in beat is prompt). Timeout = ordinary periodic tick;
            # CancelledError propagates to close().
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self._interval_s)

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._ids.clear()
        self._wake.clear()


class Template:
    """Built-in template name constants (spec 02 §"Concepts").

    Phase 0 ships the spine smoke template ``hello-shell``; benchmark
    plug-ins (terminal-bench-2 today; SWE-bench-Lite, OSWorld next)
    live under ``xrlenv_plugins/`` and register their template names
    automatically — pass them as plain strings to ``Client.rollout``.
    These constants are convenience only.
    """

    HELLO_SHELL = "hello-shell"
    TERMINAL_BENCH_2 = "terminal-bench-2"


class Client:
    """Async, transport-agnostic consumer SDK."""

    def __init__(
        self,
        transport: ClientTransport,
        *,
        run_config: str | Path | None = None,
    ) -> None:
        self._transport = transport
        self._run_config = (
            load_run_config(run_config) if run_config is not None else None
        )
        # Beats live raw-container sessions so the control plane can reap a
        # dead consumer's sessions (it can't see a unary-RPC consumer die).
        self._keepalive = _RawSessionKeepalive(
            transport, interval_s=_RAW_HEARTBEAT_INTERVAL_S,
        )

    @classmethod
    def in_process(
        cls,
        service: RolloutService,
        *,
        run_config: str | Path | None = None,
    ) -> Client:
        """Construct a client backed by an in-process service (Slice 1).

        Args:
            service: The control plane's RolloutService (typically
                ``runtime.service`` from a built local runtime).
            run_config: Optional path to a run-config YAML file. The
                run-config supplies per-template policy (deadlines,
                idle TTL, init_params) that the manifest doesn't carry.
                See :mod:`xrlenv.control.run_config`.
        """
        return cls(InProcessTransport(service), run_config=run_config)

    @classmethod
    def grpc(
        cls,
        host: str,
        port: int = 50051,
        *,
        token: str | None = None,
        secure: bool = False,
        channel_options: list[tuple[str, Any]] | None = None,
        run_config: str | Path | None = None,
    ) -> Client:
        """Construct a client that talks to a remote control plane over gRPC.

        Mirror of ``Client.in_process`` for the multi-process topology
        — a separate trainer / smoke driver dials a live ``xrlenv up``
        process and submits rollouts. The wire contract is
        spec 05 ``RolloutControl``;
        every ``ClientTransport`` method maps to one unary RPC.

        Args:
            host: Control-plane host (matches ``xrlenv up --grpc-host``).
            port: Control-plane gRPC port. Defaults to 50051.
            token: Bearer token issued by ``xrlenv tokens issue
                consumer`` on the control-plane host. The server
                requires it for spec-19 auth; pass ``None`` only for
                local trusted-channel testing where the server runs
                without auth.
            secure: When ``True``, use a TLS channel (server cert
                pinned by the system trust store + bearer-in-metadata).
                Phase-0 default is plaintext over loopback / SSH
                tunnel; flip to ``True`` for any direct
                public-internet deployment.
            channel_options: Forwarded to the underlying gRPC channel
                (e.g. ``[("grpc.max_send_message_length", -1)]`` for
                large trajectories).
        """
        return cls(
            GrpcClientTransport(
                host=host, port=port,
                token=token, secure=secure,
                channel_options=channel_options,
            ),
            run_config=run_config,
        )

    async def close(self) -> None:
        await self._keepalive.close()
        await self._transport.close()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # ── Single rollout ───────────────────────────────────────────────────────

    async def rollout(
        self,
        template: str,
        *,
        init: dict[str, Any] | None = None,
        deadline: Deadline | None = None,
        request_id: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        reward_fn: RewardFn | None = None,
    ) -> RolloutSession:
        """Start a new rollout and return its :class:`RolloutSession`.

        Use as an async context manager so the sandbox is reliably destroyed
        on exit::

            async with await client.rollout("hello-shell", init={...}) as s:
                while not s.done:
                    await s.step({"cmd": "echo hi"})
            print(s.trajectory)
        """
        # Layer-3 (per-rollout SDK kwargs) and layer-2 (run-config) merge
        # into the init dict. Layer 3 wins; run-config provides defaults
        # for keys the caller didn't set.
        policy = (
            self._run_config.policy_for(template)
            if self._run_config is not None else None
        )
        merged_init: dict[str, Any] = {}
        if policy is not None and policy.init_params:
            merged_init.update(policy.init_params)
        if init:
            merged_init.update(init)

        # Deadline: per-rollout kwarg wins; otherwise compose one from
        # the run-config's deadlines block (when it carries a hard_s,
        # which is the only Deadline field that is mandatory). Every
        # other run-config policy knob — idle_ttl_s, init_timeout_s,
        # teardown_timeout_s — flows through too so a run-config
        # advertising those values actually shapes the rollout instead
        # of being silently dropped (audit M1).
        effective_deadline = deadline
        if effective_deadline is None and policy is not None:
            d = policy.deadlines
            has_deadline_block = d is not None and any(
                v is not None
                for v in (
                    d.hard_s, d.step_timeout_s, d.setup_timeout_s,
                    d.teardown_timeout_s, d.init_timeout_s,
                )
            )
            has_idle_ttl = policy.idle_ttl_s is not None
            if has_deadline_block or has_idle_ttl:
                # Deadline.hard_s is mandatory in the type. If the
                # run-config carries any policy field we can honour but
                # not hard_s, surface that explicitly — silently
                # dropping idle_ttl_s "because there's no hard_s" is
                # exactly the bug the audit caught.
                if d is None or d.hard_s is None:
                    raise ValueError(
                        f"run-config for template {template!r} sets policy "
                        "fields (idle_ttl_s / init_timeout_s / step_timeout_s "
                        "/ etc.) without a deadlines.hard_s. Add "
                        "'deadlines.hard_s' to the run-config (it's the "
                        "rollout's hard envelope and must be explicit) or "
                        "drop the other fields."
                    )
                effective_deadline = Deadline(
                    hard_s=d.hard_s,
                    step_timeout_s=d.step_timeout_s,
                    setup_timeout_s=d.setup_timeout_s,
                    teardown_timeout_s=d.teardown_timeout_s,
                    init_timeout_s=d.init_timeout_s,
                    idle_ttl_s=policy.idle_ttl_s,
                )

        if effective_deadline is not None:
            merged_init.setdefault("step_timeout_s", effective_deadline.step_timeout_s)
            merged_init.setdefault("setup_timeout_s", effective_deadline.setup_timeout_s)

        # backend + network: user-side policy from the run-config.
        # Per-rollout SDK kwargs (when added) would override here.
        effective_backend = policy.backend if policy is not None else None
        effective_network = policy.network if policy is not None else None

        req = StartRolloutRequest(
            template=template,
            init=merged_init,
            request_id=request_id,
            task_key=task_key,
            group_id=group_id,
            deadline=effective_deadline,
            backend=effective_backend,
            network=effective_network,
        )
        resp = await self._transport.start_rollout(req)
        # Spec 05 §"Error model": validate consumer_final + reward_fn at call
        # time. If the template's reward_mode requires reward_fn and the
        # consumer didn't provide one, fail fast — and cancel the just-
        # started rollout so we don't leak the sandbox.
        if resp.reward_mode == "consumer_final" and reward_fn is None:
            with suppress(Exception):
                await self._transport.cancel(
                    resp.rollout_id, reason="reward_fn_missing"
                )
            raise RewardFnRequired(
                f"template {template!r} declares reward.mode=consumer_final but "
                "reward_fn was not passed to client.rollout(...)"
            )
        return RolloutSession(
            transport=self._transport,
            rollout_id=resp.rollout_id,
            initial_obs=resp.init_obs,
            template=template,
            reward_mode=resp.reward_mode,
            reward_fn=reward_fn,
        )

    async def replay(self, rollout_id: str) -> Trajectory:
        """Read back a sealed (or in-flight) trajectory by id (spec 05)."""
        return await self._transport.replay(rollout_id)

    # ── Cancellation primitives (spec 02 §"Group / batch rollouts") ──────────

    async def cancel_rollout(
        self, rollout_id: str, reason: str = "consumer_cancelled"
    ) -> Trajectory:
        """Cancel one rollout; returns its sealed (cancelled) trajectory."""
        return await self._transport.cancel(rollout_id, reason)

    async def cancel_group(
        self, group_id: str, reason: str = "group_cancelled"
    ) -> CancelGroupReport:
        """Cancel every still-running rollout that carries ``group_id``.

        Idempotent — rollouts already terminal are reported under
        ``already_terminal`` rather than re-cancelled.
        """
        return await self._transport.cancel_group(group_id, reason)

    async def terminate_raw_group(
        self, group_id: str, reason: str = "group_terminated"
    ) -> TerminateRawGroupReport:
        """Destroy every still-running RAW container that carries ``group_id`` — the
        raw-container analogue of :meth:`cancel_group` (which cancels gym/step rollouts).

        For a consumer driving raw containers (the ``docker.from_env`` drop-in /
        ``acquire_container``): call this to actively tear a run's containers down (e.g. on
        Ctrl-C) so capacity frees immediately, instead of leaving them for the raw-liveness
        reaper. Idempotent — rows already terminal (or without a container) are reported under
        ``already_terminal``.
        """
        return await self._transport.terminate_raw_group(group_id, reason)

    # ── Raw container session (P1.7.A.1) ────────────────────────────────────

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
        userns_mode: Literal["host", "remap"] = "host",
        acquire_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        cpu_limit: float | None = None,
        mem_limit_bytes: int | None = None,
        cpu_isolation: CpuIsolation = CpuIsolation.OFF,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> ClusterContainerSession:
        """Acquire a remote raw container scoped to a fresh rollout.

        The returned :class:`ClusterContainerSession` carries the
        ``rollout_id`` + ``container_id`` and exposes ``exec`` /
        ``destroy``. Use as an async context manager so destroy
        fires reliably on harness exception::

            async with await client.acquire_container(
                image="busybox:1",
                command=["sleep", "infinity"],
            ) as session:
                result = await session.exec(["echo", "hi"])
                # destroy fires automatically on context exit

        Image distribution (P1.7.B.2): the cluster routes the
        acquire to a scheduler-picked node via image-affinity
        scoring (`Scheduler.place(image_present=...)`), the same
        algorithm case-1 uses. The chosen node runs
        ``ImageCacheManager.ensure_present(image)`` to pull / build /
        no-op as appropriate. **Operators no longer need to pre-pull
        images on every node before consumers acquire** — the cluster
        handles distribution, caching, and eviction. For closed-set
        batch workloads (e.g., the full SWE-bench Verified set),
        operators can optionally run ``xrlenv images plan --refs <file>``
        to FFD-bin-pack the cluster's image bytes upfront.

        Args:
            image: Image ref (tag or digest).
            command: Optional CMD override; empty = image default.
                swebench's harness uses ``["tail", "-f", "/dev/null"]``
                so the container stays alive while ``exec_run`` does
                the actual work.
            name: Optional explicit container name (docker auto-names
                when unset). Useful for ``docker ps`` recognisability;
                platform identity is the returned ``container_id``.
            labels: Extra docker labels; merged on top of the
                platform's ``xrlenv.rollout_id`` +
                ``xrlenv.session_kind=raw`` reserved keys.
            environment: Container env vars (``docker -e``). Used by
                harnesses that configure the workload via env without
                baking it into the image (HF_TOKEN, GIT_AUTHOR, ...).
            task_key: Anti-affinity key for the scheduler. Spread
                parallel acquires for the same logical task across
                nodes by passing the same ``task_key``; the scheduler
                rejects nodes already running ``max_runs_per_task``
                rollouts with that key.
            ensure_image_present: defaults to True (the new UX). False
                opts back into the strict legacy contract — node
                refuses acquire if image isn't present locally. Pass
                False for deterministic-eval contexts that need "no
                surprise pulls during evaluation."
            userns_mode: ``"host"`` (default, existing behavior — the
                container runs with host UIDs even when the docker
                daemon has a ``userns-remap`` config) or ``"remap"``
                (opt in to the daemon's user-namespace remap; the
                container's UIDs are subordinate UIDs on the host).
                **Defense-in-depth**: if the container's processes
                escape (kernel bug, mount race, etc.), they land as
                an unprivileged subordinate UID on the host rather
                than host root.

                Requires daemon-level ``userns-remap`` configured in
                ``/etc/docker/daemon.json`` separately; xrlenv does
                not write that file. When the daemon has no remap
                configured, ``"remap"`` is a silent no-op (the
                daemon ignores the empty userns-mode override).
                Default ``"host"`` because lots of benchmark tasks
                need in-container root + write access to
                host-bind-mounted paths — opt in per-acquire when
                the task's image doesn't have that requirement.
            acquire_timeout_s (issue #12): wire-level deadline for
                the AcquireContainer round-trip. ``None`` uses the
                default 600 s (matches the node-side image-cache pull
                timeout, so a legitimate cold-pull surfaces as the
                node's "pull failed" rather than a wire timeout).
                Pass a larger value (e.g. 1800) when acquiring a
                known-huge image: SWE-bench Pro tags can be 5-15 GB
                and don't fit in 600 s on a slow link. Smaller values
                are accepted but rarely useful — they just trip the
                wire on legitimate work.
            queue_timeout_s (issue #18): how long the control-plane
                admission queue will hold this acquire waiting for
                cluster capacity before raising ``CapacityExhausted``.
                ``None`` uses the server default (3600 s). Raise it
                when deliberately over-requesting concurrency against
                a small cluster and you'd rather wait than fail;
                lower it when you want a fast failure signal that the
                cluster is saturated.
            session_deadline_s (issue #18): wall-clock cap on the
                acquired container's lifetime. The control plane
                force-destroys the container once the cap passes —
                a safety net so a consumer that dies mid-rollout
                (process killed, never reaching ``destroy``) can't
                leak the container and its capacity reservation.
                ``None`` uses the server default (a generous 4 h cap,
                well above any single grading task). Raise it for
                genuinely longer work.
            cpu_limit: Effective CPU limit (in cores) for the
                container, treated as a scheduling input — the cluster
                places the container on a node that can satisfy it.
                ``None`` uses the raw-container default budget. The
                docker-py drop-in derives this from the harness's
                ``host_config`` (``NanoCpus`` / ``CpuQuota``) so a
                benchmark that caps CPU in local Docker mode keeps
                that cap in cluster mode.
            mem_limit_bytes: Effective memory limit (bytes), same
                scheduling-input semantics as ``cpu_limit``. ``None``
                uses the raw-container default budget.
            cpu_isolation: CPU-isolation contract (P6). ``OFF`` (default)
                leaves CPU affinity to the OS; ``BEST_EFFORT`` pins the
                container onto disjoint whole cores when the node has spare
                capacity (scheduling-neutral — the legacy
                ``runtime_limits.cpu_pinning=True`` maps here); ``REQUIRED``
                additionally makes pinning a placement constraint (pin-or-fail;
                scheduling semantics land in a later P6 slice — currently
                treated as ``BEST_EFFORT`` end-to-end). The effective mode is
                derived once at the control-plane ingress from this field
                falling back to the ``cpu_pinning`` alias.
            runtime_limits: Container-shape limits (pids / shm / tmpfs /
                read-only rootfs) that do **not** affect scheduling.
                ``None`` applies no constraint (docker default). The
                docker-py drop-in derives this from the harness's
                ``host_config``.
        """
        result = await self._transport.acquire_container(
            image=image, command=command, entrypoint=entrypoint,
            user=user, cap_add=cap_add, devices=devices,
            privileged=privileged, network_mode=network_mode, binds=binds,
            name=name,
            labels=labels, environment=environment,
            task_key=task_key,
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
        # Start beating this session; stop when it's destroyed.
        self._keepalive.register(result.rollout_id)
        return ClusterContainerSession(
            self._transport, result, on_destroy=self._keepalive.unregister,
        )

    # ── Multi-service compose project (P1.7.C.2) ─────────────────────────────

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
    ) -> ClusterComposeSession:
        """Bring up a multi-service **compose project** scoped to a fresh
        rollout, returning a :class:`ClusterComposeSession`.

        The compose analog of :meth:`acquire_container`. Use for a task whose
        sandbox is a ``docker-compose.yaml`` stack (a ``main`` service plus
        sidecars — a db, a fake cloud endpoint, an app tier on a private
        network). The control plane vets the compose against its ``KwargsPolicy``,
        digest-pins ``main``, stamps the reserved ``xrlenv.*`` labels, reserves
        the whole-stack footprint, and runs ``docker compose up -d --wait`` on a
        scheduler-picked node; the returned session's ``exec`` / ``put_archive`` /
        ``get_archive`` / ``exec_stream`` / ``apply_egress`` all target the
        project's ``main`` service, and ``destroy`` downs the whole stack.

        Use as an async context manager so the project is torn down on harness
        exception::

            async with await client.acquire_compose_project(
                compose_yaml=rewritten_yaml,
                images=["ns/app@sha256:…", "postgres:14"],
                footprint_cpu=6.0,
                footprint_mem_bytes=8 * 1024**3,
            ) as project:
                await project.exec(["pytest", "-q"])
                # docker compose down fires automatically on context exit

        Args:
            compose_yaml: The **rewritten, image-ref-only** compose document —
                every service references a pullable image (no ``build:``). The
                caller (the benchmark plug-in) is responsible for pre-building +
                pushing per-service images and rewriting the document; the CP
                does not build. It vets the document against ``KwargsPolicy``
                and rejects (never strips) a disallowed privileged / host-bind /
                network-mode service.
            images: Image refs to ensure-present on the node before ``up``
                (``main`` + every sidecar), so a cold node pulls them via the
                image cache rather than failing ``up``.
            footprint_cpu: Whole-stack CPU reserve (cores) — ``main``'s declared
                cpu plus a sidecar allowance. The scheduler places the project on
                a node that can satisfy it; the CP can't derive ``main``'s
                declared size, so it arrives here.
            footprint_mem_bytes: Whole-stack memory reserve (bytes), same
                scheduling-input role as ``footprint_cpu``.
            main_service: The service the harness execs into; defaults to
                ``"main"``.
            project_name: Optional explicit compose project name; ``None`` lets
                the CP derive one from the session (sanitized to docker's
                ``[a-z0-9][a-z0-9_-]*``).
            task_key: Anti-affinity key for the scheduler, same semantics as
                :meth:`acquire_container`.
            group_id: Cancel-cohort tag, same semantics as elsewhere.
            labels: Extra docker labels; merged on top of the CP's reserved
                ``xrlenv.*`` keys (which the plug-in must not set itself).
            queue_timeout_s: Admission-queue wait bound before
                ``CapacityExhausted``; ``None`` uses the server default.
            session_deadline_s: Wall-clock cap on the project's lifetime; the CP
                force-reaps the whole stack past it. ``None`` uses the server
                default.
            up_timeout_s: ``docker compose up --wait`` ceiling forwarded to the
                node; ``None`` uses the node default.
        """
        result = await self._transport.acquire_compose_project(
            compose_yaml=compose_yaml,
            images=images,
            footprint_cpu=footprint_cpu,
            footprint_mem_bytes=footprint_mem_bytes,
            main_service=main_service,
            project_name=project_name,
            task_key=task_key,
            group_id=group_id,
            labels=labels,
            queue_timeout_s=queue_timeout_s,
            session_deadline_s=session_deadline_s,
            up_timeout_s=up_timeout_s,
        )
        # Beat the project's main session; stop when it's destroyed.
        self._keepalive.register(result.rollout_id)
        return ClusterComposeSession(
            self._transport, result, on_destroy=self._keepalive.unregister,
        )

    # ── Cluster status (D21) ─────────────────────────────────────────────────

    async def list_nodes(self) -> list[Any]:
        """Return the live cohort of attached nodes.

        Each entry is a :class:`xrlenv.control.state.NodeRecord` carrying
        ``node_id``, ``status`` (``connected`` / ``lost``), ``backends``,
        ``connected_at``, ``last_seen_at``. Same data the operator CLI's
        ``xrlenv nodes`` and the admin panel's ``/nodes`` view consume.
        """
        return await self._transport.list_nodes()

    async def wait_for_nodes(
        self,
        min_nodes: int = 1,
        *,
        timeout_s: float = 90.0,
        backend: str | None = None,
        poll_interval_s: float = 1.0,
    ) -> list[Any]:
        """Block until at least ``min_nodes`` nodes are attached
        (status=``connected``), then return the matching list.

        Useful right after restarting ``xrlenv up`` — the gRPC streams
        from cloud nodes can take seconds to reattach, and dispatching
        rollouts before they're back yields ``BackendCapabilityMissing``
        ("no node supports backend ...") on the consumer side. This
        helper polls ``list_nodes()`` and returns the moment the
        cluster is ready.

        - ``backend`` filters to nodes that advertise a specific backend
          (e.g. ``"docker"``); ``None`` accepts any.
        - ``timeout_s`` bounds the wait; raises ``TimeoutError`` on
          expiry. Default 90 s matches the smoke driver's
          ``--restart-grace`` default.
        - ``poll_interval_s`` is the gap between probes. Default 1 s;
          first poll is immediate.

        Replaces the connect-mode probe-and-retry hack
        (``_wait_for_nodes_via_probe``) in earlier smoke drivers — the
        SDK now has the cluster-status surface that hack approximated.
        """
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            nodes = await self.list_nodes()
            connected = [n for n in nodes if getattr(n, "status", None) == "connected"]
            if backend is not None:
                connected = [
                    n for n in connected if backend in getattr(n, "backends", [])
                ]
            if len(connected) >= min_nodes:
                return connected
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"only {len(connected)} of the required {min_nodes} "
                    f"node(s) attached within {timeout_s:.0f}s"
                    + (f" with backend={backend!r}" if backend else "")
                    + f" (saw {len(nodes)} total registry rows)"
                )
            await asyncio.sleep(min(poll_interval_s, remaining))

    # ── Batch (workhorse for consumers) ──────────────────────────────────────

    async def batch_rollout(
        self,
        template: str,
        inits: list[dict[str, Any]],
        policy: PolicyFn,
        *,
        deadline: Deadline | None = None,
        concurrency: int = 64,
        task_keys: list[str | None] | None = None,
        group_ids: list[str | None] | None = None,
        request_ids: list[str | None] | None = None,
        reward_fn: RewardFn | None = None,
    ) -> BatchRolloutResult:
        """Run ``len(inits)`` concurrent rollouts; bucket results by terminal
        status (spec 05 §"Batched").

        Returns a :class:`BatchRolloutResult` with three lists:
        ``finished`` / ``truncated`` / ``failed``. The platform guarantees:

        - All sandboxes are released (destroyed via the existing terminate
          path) before this call returns. No leaks.
        - ``deadline.hard_s`` is enforced per rollout by the coordinator's
          deadline watcher; one straggler does not delay the others.
        - ``concurrency`` bounds in-flight rollouts via an
          :class:`asyncio.Semaphore` — additional inits queue inside this
          call rather than overwhelming the admission queue.

        Per-rollout annotations (``task_keys`` / ``group_ids`` /
        ``request_ids``), if supplied, must be the same length as ``inits``.
        ``None`` entries in those lists leave the corresponding rollout
        unannotated.
        """
        n = len(inits)
        if task_keys is not None and len(task_keys) != n:
            raise ValueError(f"task_keys length {len(task_keys)} != inits length {n}")
        if group_ids is not None and len(group_ids) != n:
            raise ValueError(f"group_ids length {len(group_ids)} != inits length {n}")
        if request_ids is not None and len(request_ids) != n:
            raise ValueError(f"request_ids length {len(request_ids)} != inits length {n}")

        result = BatchRolloutResult()
        sem = asyncio.Semaphore(concurrency)

        async def _drive(idx: int) -> None:
            init_payload = inits[idx]
            tk = task_keys[idx] if task_keys else None
            gid = group_ids[idx] if group_ids else None
            rid = request_ids[idx] if request_ids else None
            async with sem:
                try:
                    session = await self.rollout(
                        template=template,
                        init=init_payload,
                        deadline=deadline,
                        request_id=rid,
                        task_key=tk,
                        group_id=gid,
                        reward_fn=reward_fn,
                    )
                except Exception as exc:  # admission / template-resolve failures
                    result.failed.append(
                        _failed_from_exc(rollout_id=None, exc=exc)
                    )
                    return

                async with session:
                    try:
                        while not session.done:
                            obs = session.observation
                            action = await policy(obs)
                            await session.step(action)
                    except RolloutTruncated as exc:
                        if exc.partial is not None:
                            result.truncated.append(exc.partial)
                        return
                    except RolloutCancelled as exc:
                        if exc.partial is not None:
                            result.truncated.append(exc.partial)
                        return
                    except RolloutFailed as exc:
                        result.failed.append(
                            FailedRollout(
                                rollout_id=session.rollout_id,
                                reason=exc.reason,
                                error_kind=type(exc).__name__,
                                error_message=str(exc),
                                partial=exc.partial,
                            )
                        )
                        return
                    except Exception as exc:  # unexpected — bucket as failed
                        LOGGER.exception(
                            "batch_rollout: rollout=%s drove unexpected error",
                            session.rollout_id,
                        )
                        result.failed.append(
                            _failed_from_exc(rollout_id=session.rollout_id, exc=exc)
                        )
                        return

                # Session exited cleanly; sealed trajectory is now available.
                trajectory = session.trajectory
                if trajectory.status.value == "truncated":
                    result.truncated.append(trajectory)
                elif trajectory.status.value == "failed":
                    result.failed.append(
                        FailedRollout(
                            rollout_id=trajectory.rollout_id,
                            reason=trajectory.reason or "unknown",
                            error_kind="RolloutFailed",
                            error_message=trajectory.reason or "unknown",
                            partial=trajectory,
                        )
                    )
                else:
                    result.finished.append(trajectory)

        await asyncio.gather(*(_drive(i) for i in range(n)))
        return result


def _failed_from_exc(*, rollout_id: str | None, exc: BaseException) -> FailedRollout:
    return FailedRollout(
        rollout_id=rollout_id,
        reason=getattr(exc, "reason", type(exc).__name__),
        error_kind=type(exc).__name__,
        error_message=str(exc),
        partial=getattr(exc, "partial", None),
    )


__all__ = ["Client", "Template"]
