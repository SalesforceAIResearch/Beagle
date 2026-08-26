"""Node-side outbound gRPC link to the control plane (spec 21).

The node initiates the bidi stream OUT to the control plane (invariant 7).
This module owns the long-lived ``NodeControlStream`` RPC, dispatches
incoming :class:`ControlMsg` commands to a local :class:`NodeAgent`, and ships
:class:`CommandReply` envelopes back. An idempotency cache keyed by
``idempotency_key`` returns cached replies for retried commands so a
reconnect-induced replay never double-executes.

Phase-0 essentials only — heartbeat, stats, drain, GC, image directives,
trajectory fetch, and snapshots are deferred to Slice 3.5+.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections import OrderedDict
from contextlib import aclosing, suppress
from typing import Any

import grpc

from xrlenv.api import converters as conv
from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.api._pb2 import node_control_pb2_grpc as pb_grpc
from xrlenv.api.constants import (
    ARCHIVE_CHUNK_BYTES,
    GRPC_CHANNEL_OPTIONS,
    MAX_OUTBOUND_MESSAGE_GUARD_BYTES,
)
from xrlenv.backends.egress import EgressAllowlist, EgressRule
from xrlenv.buildinfo import agent_identity
from xrlenv.errors import XRLEnvError
from xrlenv.node.agent import NodeAgent
from xrlenv.node.disk_guard import build_disk_guard
from xrlenv.node.health import NodeHealthSnapshot
from xrlenv.node.trajectory_reader import FetchRangeKind

LOGGER = logging.getLogger(__name__)

# NodeHello docker-ready gate (§5.3). Before advertising supported_runtimes we
# wait (bounded) for ``docker info`` to answer, so a node whose agent starts
# seconds after a docker restart doesn't enumerate runtimes against a not-yet-
# ready daemon and advertise a conservative {'runc'} for the whole connection.
# Docker usually answers within a few seconds of a restart; the timeout is a
# generous ceiling after which we proceed with whatever we have (unchanged
# conservative behaviour, logged loudly). After the first success the probe is
# cached, so reconnects skip the wait.
_DOCKER_READY_TIMEOUT_S = 60.0
_DOCKER_READY_INTERVAL_S = 2.0


def _health_to_proto(snap: NodeHealthSnapshot) -> pb.NodeHealthStats:
    """Stage-1 — node-side health snapshot → ``Heartbeat.health``."""
    return pb.NodeHealthStats(
        window_s=int(snap.window_s),
        create_p50_ms=snap.create_p50_ms,
        create_p95_ms=snap.create_p95_ms,
        create_count=snap.create_count,
        docker_error_count=snap.docker_error_count,
        docker_timeout_count=snap.docker_timeout_count,
        create_inflight=snap.create_inflight,
        create_queued=snap.create_queued,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Idempotency cache (per spec 21 §"Idempotency", node side)
# ──────────────────────────────────────────────────────────────────────────────


class _IdempotencyCache:
    """LRU keyed by ``idempotency_key`` → ``CommandReply``.

    Default size 4096 entries / TTL 300 s per spec 21. Same TTL as the
    control plane's request_id cache so retried rollout starts hit
    consistent state on either side.
    """

    def __init__(self, *, max_size: int = 4096, ttl_s: float = 300.0) -> None:
        self._max = max_size
        self._ttl = ttl_s
        self._items: OrderedDict[str, tuple[float, pb.CommandReply]] = OrderedDict()

    def get(self, key: str) -> pb.CommandReply | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        ts, reply = entry
        if time.monotonic() - ts > self._ttl:
            self._items.pop(key, None)
            return None
        # Refresh LRU order on hit.
        self._items.move_to_end(key)
        return reply

    def put(self, key: str, reply: pb.CommandReply) -> None:
        self._items[key] = (time.monotonic(), reply)
        self._items.move_to_end(key)
        while len(self._items) > self._max:
            self._items.popitem(last=False)


# ──────────────────────────────────────────────────────────────────────────────
# NodeGrpcLink
# ──────────────────────────────────────────────────────────────────────────────


def _guard_outbound_message(
    msg: pb.NodeMsg, *, guard_bytes: int,
) -> pb.NodeMsg:
    """Transport safety net against the node-lost root cause.

    A NodeMsg whose serialized size exceeds the gRPC send ceiling
    raises ``RESOURCE_EXHAUSTED`` on write and tears down the whole
    bidi stream — taking the heartbeat with it, so the control plane
    marks the node lost and seals every in-flight rollout there as
    ``node_lost``. Large payloads only ever come from command
    *replies* (``container_get_archive`` is now chunked, but a batched
    ``ExecReply`` with huge stdout could still balloon). If a reply
    exceeds ``guard_bytes`` we substitute a small FAILED CommandReply
    for the same command_id: that one command fails cleanly while the
    stream — and the heartbeat — survive. Non-reply frames and
    within-budget replies pass through unchanged.
    """
    if (
        msg.WhichOneof("body") == "reply"
        and msg.ByteSize() > guard_bytes
    ):
        oversized = msg.ByteSize()
        cid = msg.reply.command_id
        LOGGER.error(
            "node-link: dropping oversized reply command_id=%s "
            "(%d bytes > %d guard); failing the command instead of "
            "severing the stream",
            cid, oversized, guard_bytes,
        )
        return pb.NodeMsg(
            stream_epoch=msg.stream_epoch,
            seq=msg.seq,
            reply=pb.CommandReply(
                command_id=cid,
                status=pb.ReplyStatus.FAILED,
                error_kind="ReplyTooLarge",
                error_message=(
                    f"reply payload {oversized} bytes exceeds the "
                    f"{guard_bytes}-byte wire guard; the command produced "
                    f"too much data to return in one message"
                ),
            ),
        )
    return msg


class NodeGrpcLink:
    """Long-lived outbound bidi stream to the control plane.

    Lifecycle:
        link = NodeGrpcLink(agent, control_addr=..., node_id=...)
        await link.run_forever()  # connects + reconnects; blocks until cancelled

    Reconnect semantics: exponential backoff (1s → 30s, ±20% jitter) up to
    ``reconnect_max_s``. A fresh ``stream_epoch`` is minted per dial. The
    idempotency cache survives reconnects so replayed commands from the
    control plane return cached replies.
    """

    def __init__(
        self,
        agent: NodeAgent,
        *,
        control_addr: str,
        backends: list[str] | None = None,
        reconnect_max_s: float = 600.0,
        heartbeat_interval_s: float = 5.0,
        server_silence_deadline_s: float = 30.0,
        bearer_token: str | None = None,
    ) -> None:
        self._agent = agent
        self._control_addr = control_addr
        self._backends = backends or agent.supported_backends()
        self._reconnect_max_s = reconnect_max_s
        self._heartbeat_interval_s = heartbeat_interval_s
        # Defense-in-depth (2026-08-21): the control plane beats every connected
        # node with a periodic keepalive ControlMsg. If we go this long without
        # ANY message from the CP (command OR keepalive), the stream is
        # effectively dead to the CP — it deregistered us, or the stream went
        # half-open while HTTP/2 keepalive still answers — so proactively redial
        # and re-register instead of lingering ``lost`` forever. Must exceed the
        # CP keepalive cadence (5s) with margin; 30s ≈ 6 missed beats.
        self._server_silence_deadline_s = server_silence_deadline_s
        # Spec 19 §"API authz scopes": passed verbatim as
        # ``Authorization: Bearer <token>`` on every NodeControlStream
        # call. The control-plane interceptor verifies it against the
        # node-role token store (or accepts unauth when the store is empty,
        # phase-0 single-host smoke).
        self._bearer_token = bearer_token

        # Issue #18: cached disk-state, refreshed off the heartbeat
        # path by ``_disk_sample_loop``. The heartbeat loop reads this
        # instead of probing the docker daemon inline — see the
        # decoupling rationale in ``_disk_sample_loop``.
        self._disk_state: tuple[int, int] = (0, 0)

        self._cache = _IdempotencyCache()
        # WS1 — in-flight chunked put_archive uploads, keyed by command_id. Each entry is
        # ``{"rollout_id","container_id","target_dir","parts":[bytes,...]}``; the terminal
        # ``done`` frame assembles + runs one docker put_archive, then pops the entry.
        self._put_archive_chunks: dict[str, dict[str, Any]] = {}
        self._prior_epoch: str | None = None
        self._last_seen_control_instance: str | None = None
        self._last_command_seq_seen = 0
        self._last_reply_seq_acked = 0

        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        backoff_s = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_and_serve()
                # Clean disconnect — reset backoff for next reconnect.
                backoff_s = 1.0
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception(
                    "node link to %s failed; reconnecting in ~%.1fs",
                    self._control_addr,
                    backoff_s,
                )
            if self._stop.is_set():
                return
            jitter = backoff_s * 0.2 * (2 * random.random() - 1)
            await asyncio.sleep(max(0.0, backoff_s + jitter))
            backoff_s = min(backoff_s * 2, 30.0)

    async def _await_docker_ready(self) -> None:
        """Bounded wait for docker to answer a runtime probe before the first
        ``NodeHello``.

        A node whose agent starts seconds after a docker restart (the redeploy
        race) would otherwise enumerate ``supported_runtimes`` against a
        not-yet-ready daemon → conservative ``{'runc'}``, which is sent once at
        hello and never re-advertised on a persistent connection — so the
        scheduler's runtime filter can't place a sysbox acquire until the agent
        is manually restarted. We poll off-thread (the probe does a blocking
        ``docker info``) up to ``_DOCKER_READY_TIMEOUT_S``. If docker never
        answers we proceed with the conservative set (unchanged behaviour), logged
        loudly. Cached after the first success, so reconnects return immediately."""
        probe = getattr(self._agent, "probe_docker_runtimes_ready", None)
        if probe is None:
            return  # agent/test-double predating the probe — advertise as-is
        if await asyncio.to_thread(probe):
            return
        started = time.monotonic()
        while time.monotonic() - started < _DOCKER_READY_TIMEOUT_S:
            await asyncio.sleep(_DOCKER_READY_INTERVAL_S)
            if await asyncio.to_thread(probe):
                LOGGER.info(
                    "node link: docker became enumerable after %.0fs; advertising "
                    "the real runtime set on NodeHello",
                    time.monotonic() - started,
                )
                return
        LOGGER.error(
            "node link: docker not enumerable after %.0fs — advertising a "
            "conservative runtime set. Non-runc runtimes (e.g. sysbox-runc) will "
            "be invisible to the scheduler until docker recovers AND this agent "
            "reconnects/restarts.",
            _DOCKER_READY_TIMEOUT_S,
        )

    async def _connect_and_serve(self) -> None:
        # §5.3 — gate the hello on a ready docker so supported_runtimes reflects
        # the real runtime set, not a startup-race fallback.
        await self._await_docker_ready()
        stream_epoch = str(uuid.uuid4())
        outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
        node_seq = _MonotonicCounter()
        # Hold strong refs to per-command dispatch tasks until they finish so
        # the GC does not reap them mid-flight (RUF006).
        in_flight: set[asyncio.Task[None]] = set()
        heartbeat_task: asyncio.Task[None] | None = None
        disk_task: asyncio.Task[None] | None = None
        sweep_task: asyncio.Task[None] | None = None
        aimd_task: asyncio.Task[None] | None = None
        guard_task: asyncio.Task[None] | None = None
        silence_task: asyncio.Task[None] | None = None
        # Server-liveness (2026-08-21): last time ANY message (command or
        # keepalive) arrived from the control plane; the silence watchdog redials
        # if this goes stale. ``redial`` records that WE initiated the
        # disconnect (via ``call.cancel()``) so the reconnect isn't mistaken for
        # a fatal error or a shutdown.
        last_ctrl_at = [time.monotonic()]
        redial = [False]

        # Send NodeHello as the first NodeMsg.
        hello = pb.NodeHello(
            node_id=self._agent.node_id,
            backends=self._backends,
            hardware=conv.hardware_info_to_proto(self._agent.hardware()),
            prior_stream_epoch=self._prior_epoch or "",
            last_seen_control_instance_id=self._last_seen_control_instance or "",
            last_command_seq_seen=self._last_command_seq_seen,
            last_reply_seq_acked=self._last_reply_seq_acked,
            # Issue #18 (Ask #2) — let the control plane detect a
            # stale node-agent binary at connect time.
            agent_version=agent_identity(),
            # §5.3 — advertise the docker runtimes this node can run and
            # its daemon default runtime, so the scheduler filters by
            # runtime and the control plane can re-verify default==runc.
            # Defensive getattr: an agent/test-double predating these
            # methods advertises only runc (byte-compatible with a
            # pre-§5.3 node), never breaking the hello handshake.
            supported_runtimes=getattr(
                self._agent, "supported_runtimes", lambda: ["runc"],
            )(),
            default_runtime=getattr(
                self._agent, "default_runtime", lambda: "runc",
            )(),
            # P6 (§8.6) — whether this node can enforce the shared-parent
            # cpuset isolation scheme (cgroup v2 + passed the §8.5 self-test).
            # Defensive getattr: an agent/test-double predating this hook
            # advertises ``False`` (never presume unproven isolation).
            isolation_capable=getattr(
                self._agent, "isolation_capable", lambda: False,
            )(),
            # WS1 — this node-binary's grpc_link accumulates chunked
            # ContainerPutArchiveChunk commands (see _accumulate_put_archive_chunk), so
            # the control plane may send large uploads chunked instead of one >128 MiB
            # message. Always True for this binary; a pre-WS1 node omits the field
            # (proto default False) → the CP falls back to the unary put command.
            chunked_put_archive_capable=True,
        )
        await outbox.put(
            pb.NodeMsg(stream_epoch=stream_epoch, seq=node_seq.next(), hello=hello)
        )

        async def _outgoing() -> Any:
            while True:
                msg = await outbox.get()
                yield _guard_outbound_message(
                    msg, guard_bytes=MAX_OUTBOUND_MESSAGE_GUARD_BYTES,
                )

        # Audit M1: match the control-plane server's message-size caps. This is the
        # CLIENT list, so it also carries keepalive pings; the server permits them
        # via GRPC_SERVER_OPTIONS.
        # so verifier-asset tarballs above gRPC's 4 MB default still flow.
        async with grpc.aio.insecure_channel(
            self._control_addr, options=GRPC_CHANNEL_OPTIONS,
        ) as channel:
            stub = pb_grpc.NodeControlStub(channel)
            call_metadata: list[tuple[str, str]] = []
            if self._bearer_token:
                call_metadata.append(
                    ("authorization", f"Bearer {self._bearer_token}")
                )
            call = stub.NodeControlStream(
                _outgoing(),
                metadata=tuple(call_metadata) if call_metadata else None,
            )

            try:
                first_seen = False
                async for ctrl in call:
                    # Any message (command OR keepalive) proves the CP still
                    # holds this stream — refresh the silence watchdog's clock.
                    last_ctrl_at[0] = time.monotonic()
                    if not first_seen:
                        # First ControlMsg in every new epoch is ControlHello.
                        # Once we know the control plane is ready, start the
                        # periodic heartbeat task.
                        first_seen = True
                        self._last_seen_control_instance = ctrl.control_instance_id
                        LOGGER.info(
                            "node=%s connected to control_plane=%s (epoch=%s, instance=%s)",
                            self._agent.node_id,
                            self._control_addr,
                            stream_epoch,
                            ctrl.control_instance_id,
                        )
                        heartbeat_task = asyncio.create_task(
                            self._heartbeat_loop(outbox, stream_epoch, node_seq),
                            name=f"heartbeat-{self._agent.node_id}",
                        )
                        # Issue #18: disk sampling runs on its own
                        # task so a slow ``docker info`` can't delay
                        # the heartbeat above the watchdog grace.
                        disk_task = asyncio.create_task(
                            self._disk_sample_loop(),
                            name=f"disk-sample-{self._agent.node_id}",
                        )
                        # Issue #13: periodic image-cache eviction sweep.
                        # Closes the gap where the on-pull eviction trigger
                        # never fires during steady-state workloads on
                        # already-cached images while live containers'
                        # writable overlays quietly fill the disk.
                        cache = getattr(self._agent, "_image_cache", None)
                        if cache is not None:
                            sweep_task = asyncio.create_task(
                                cache.run_sweep_loop(),
                                name=f"image-sweep-{self._agent.node_id}",
                            )
                            # Node-local adaptive pull-concurrency (AIMD)
                            # loop. Idempotent — safe to (re)launch on
                            # each reconnect; duplicate calls return at
                            # once via the manager's running-guard. Tracked
                            # + cancelled in the finally below, like the
                            # heartbeat/disk/sweep tasks.
                            aimd_task = asyncio.create_task(
                                cache.run_pull_aimd_loop(),
                                name=f"image-pull-aimd-{self._agent.node_id}",
                            )
                        # WS2 — node-autonomous disk-pressure guard. Kills
                        # a runaway raw container whose writable layer is
                        # filling the data-root, before a full disk wedges
                        # the heartbeat and the control plane marks the
                        # node lost. None when no cache / docker raw
                        # manager is wired. Tracked + cancelled like the
                        # other per-connection loops.
                        guard = build_disk_guard(self._agent)
                        if guard is not None:
                            guard_task = asyncio.create_task(
                                guard.run_loop(),
                                name=f"disk-guard-{self._agent.node_id}",
                            )
                        # Server-liveness watchdog: redial if the CP goes silent
                        # past the deadline (it beats every connected node with
                        # keepalives, so prolonged silence means it dropped us or
                        # the stream is half-open). Cancelled in the finally.
                        silence_task = asyncio.create_task(
                            self._server_liveness_loop(call, last_ctrl_at, redial),
                            name=f"server-liveness-{self._agent.node_id}",
                        )
                        continue
                    if ctrl.WhichOneof("body") is None:
                        # Empty-body keepalive — liveness only (already refreshed
                        # ``last_ctrl_at`` above). Do NOT advance replay
                        # coordinates or dispatch it as a command.
                        continue
                    self._last_command_seq_seen = max(
                        self._last_command_seq_seen, ctrl.seq
                    )
                    task = asyncio.create_task(
                        self._dispatch(ctrl, outbox, stream_epoch, node_seq),
                        name=f"node-cmd-{ctrl.seq}",
                    )
                    in_flight.add(task)
                    task.add_done_callback(in_flight.discard)
            except (asyncio.CancelledError, grpc.aio.AioRpcError):
                # If WE triggered the disconnect via the server-liveness watchdog
                # (``call.cancel()`` on keepalive silence), swallow it so
                # ``run_forever`` treats this as a normal reconnect. Anything else
                # — a real shutdown task-cancel, or a genuine RPC error —
                # propagates unchanged (run_forever re-raises CancelledError and
                # reconnects on other errors, exactly as before).
                if not (redial[0] and not self._stop.is_set()):
                    raise
                LOGGER.info(
                    "node=%s redialing control_plane=%s after keepalive silence",
                    self._agent.node_id, self._control_addr,
                )
            finally:
                # Remember epoch so the next NodeHello can carry replay coords.
                self._prior_epoch = stream_epoch
                if silence_task is not None and not silence_task.done():
                    silence_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await silence_task
                if heartbeat_task is not None and not heartbeat_task.done():
                    heartbeat_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat_task
                if disk_task is not None and not disk_task.done():
                    disk_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await disk_task
                if sweep_task is not None and not sweep_task.done():
                    sweep_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await sweep_task
                if aimd_task is not None and not aimd_task.done():
                    aimd_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await aimd_task
                if guard_task is not None and not guard_task.done():
                    guard_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await guard_task

    async def _server_liveness_loop(
        self,
        call: Any,
        last_ctrl_at: list[float],
        redial: list[bool],
    ) -> None:
        """Redial if the control plane goes silent past the deadline.

        The CP beats every connected node with a periodic keepalive ControlMsg,
        so a prolonged gap in ANY inbound message (command or keepalive) means
        the CP no longer holds this stream — it deregistered us (the heartbeat
        watchdog), or the stream went half-open while HTTP/2 keepalive is still
        answered. Cancelling ``call`` breaks the reader's ``async for``;
        ``redial`` records that this was our own doing so ``run_forever``
        reconnects (fresh NodeHello → re-register) instead of erroring out.
        """
        check_interval = max(1.0, self._server_silence_deadline_s / 3.0)
        try:
            while not self._stop.is_set():
                await asyncio.sleep(check_interval)
                silent_for = time.monotonic() - last_ctrl_at[0]
                if silent_for > self._server_silence_deadline_s:
                    LOGGER.warning(
                        "node=%s: no message from control_plane=%s for %.0fs "
                        "(> %.0fs deadline) — stream looks dead to the CP; "
                        "redialing to re-register.",
                        self._agent.node_id,
                        self._control_addr,
                        silent_for,
                        self._server_silence_deadline_s,
                    )
                    redial[0] = True
                    call.cancel()
                    return
        except asyncio.CancelledError:
            return

    # ── Heartbeat ────────────────────────────────────────────────────────────

    async def _heartbeat_loop(
        self,
        outbox: asyncio.Queue[pb.NodeMsg],
        stream_epoch: str,
        node_seq: _MonotonicCounter,
    ) -> None:
        """Spec 21: periodic Heartbeat keeps the control plane's NodeRegistry
        from marking us dead. Default cadence 5s; the registry's grace is 60s
        so we have ~12 missed beats before being declared lost.

        Issue #18: the heartbeat reads the *cached* disk-state
        (``_disk_sample_loop`` refreshes it on its own task) rather
        than probing the docker daemon inline. A `docker info` round
        trip gets slow under heavy cold-pull load — exactly when the
        control plane most needs heartbeats to keep flowing — so the
        beat cadence must not depend on it.
        """
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval_s)
                free_disk, total_disk = self._disk_state
                # P6 (§8.6, R6) — live pinned-CPU accounting from the node's
                # core ledger; reporting only (no scheduling yet). Defensive
                # getattr: an agent predating this hook reports (0, 0) =
                # "unknown", which the control plane leaves out of any
                # pinned-capacity decision.
                # P6 step-4a — refresh the legacy-unpinned-runc gap first (async,
                # off this cadence) so the folded pinned_cpus_free below drains as
                # those containers exit. Best-effort: a failure leaves the last
                # value, never breaks the beat.
                _refresh = getattr(self._agent, "refresh_pinned_cpu_gap", None)
                if _refresh is not None:
                    try:
                        await _refresh()
                    except Exception:
                        LOGGER.debug(
                            "refresh_pinned_cpu_gap failed; using last value",
                            exc_info=True,
                        )
                _pinned_free, _pinned_total = getattr(
                    self._agent, "pinned_cpu_capacity", lambda: (0, 0),
                )()
                hb = pb.Heartbeat(
                    node_monotonic_s=time.monotonic(),
                    sandboxes_running=len(self._agent._sandboxes),
                    cpu_load_1m=0.0,
                    mem_used_bytes=0,
                    free_disk_bytes=free_disk,
                    total_disk_bytes=total_disk,
                    pinned_cpus_free=_pinned_free,
                    pinned_cpus_total=_pinned_total,
                )
                # Stage-1: attach per-node docker health. In-memory read,
                # no daemon round trip — keeps the issue-#18 guarantee
                # that the heartbeat cadence is daemon-independent.
                _hsnap = self._agent.health_snapshot()
                if _hsnap is not None:
                    hb.health.CopyFrom(_health_to_proto(_hsnap))
                await outbox.put(
                    pb.NodeMsg(
                        stream_epoch=stream_epoch,
                        seq=node_seq.next(),
                        heartbeat=hb,
                    )
                )
        except asyncio.CancelledError:
            return

    async def _disk_sample_loop(self) -> None:
        """Issue #18: refresh ``_disk_state`` on its own cadence, off the
        heartbeat path.

        ``_sample_disk_state`` makes two ``docker info`` calls; that
        round trip gets slow exactly when the node is under heavy
        cold-pull load (the docker daemon busy extracting multi-GB
        overlay layers). Sampling here keeps at most one probe in
        flight and lets the heartbeat loop send on a fixed cadence —
        a slow probe only ages the disk reading, it never delays a
        heartbeat or trips the control-plane watchdog.
        """
        try:
            while True:
                self._disk_state = await self._sample_disk_state()
                await asyncio.sleep(self._heartbeat_interval_s)
        except asyncio.CancelledError:
            return

    async def _sample_disk_state(self) -> tuple[int, int]:
        """Issue #14: read free/total disk from the image-cache backend
        on every heartbeat tick so the control plane has a live signal
        for the placement gate + admin pressure indicator.

        Returns ``(0, 0)`` when no image cache is wired (stripped test
        fixtures) or when the backend probe raises — the heartbeat must
        never be blocked by a transient daemon hiccup. ``(0, 0)`` is
        the documented "unknown" sentinel; the control-plane gate
        treats it as healthy until a real value arrives.
        """
        cache = getattr(self._agent, "_image_cache", None)
        if cache is None:
            return 0, 0
        backend = getattr(cache, "_backend", None)
        if backend is None:
            return 0, 0
        try:
            free = int(await backend.free_disk_bytes())
            total = int(await backend.total_disk_bytes())
        except Exception:
            LOGGER.debug(
                "heartbeat: disk-state sample failed; reporting unknown",
                exc_info=True,
            )
            return 0, 0
        return max(0, free), max(0, total)

    # ── Command dispatch ─────────────────────────────────────────────────────

    async def _dispatch(
        self,
        ctrl: pb.ControlMsg,
        outbox: asyncio.Queue[pb.NodeMsg],
        stream_epoch: str,
        node_seq: _MonotonicCounter,
    ) -> None:
        body_kind = ctrl.WhichOneof("body")
        if body_kind in ("hello", "ack"):
            return  # legitimate non-command frames; nothing to dispatch
        if body_kind is None:
            # Either a literally-empty ControlMsg (rare; would have
            # to be sent by a buggy control plane) or — much more
            # likely in practice — a NEW command kind whose proto
            # field tag this node-binary's bindings don't recognise.
            # The unknown bytes ride on the wire as an unknown
            # field, ``WhichOneof`` returns None, and without this
            # branch the dispatcher silently returned without ever
            # sending a CommandReply — leaving the control plane's
            # ``_send_and_wait`` to hang indefinitely. (Operator-
            # reported 2026-05-06 against a stale node binary +
            # the new ``AcquireContainerCommand`` field.)
            #
            # We can't reply with a structured FAILED status here
            # because we can't safely extract the inner
            # ``CommandHeader.command_id`` from the unknown bytes
            # (proto3 unknown-field parsing is fiddly). Best we
            # can do is log loudly so the operator sees the
            # version-mismatch in the node log, and rely on the
            # control plane's per-call timeout (added in the same
            # commit) to surface the consumer-side stall.
            LOGGER.error(
                "node-link: received ControlMsg with no recognised "
                "body field. The control plane likely has newer "
                "proto bindings than this node-agent. Redeploy the "
                "xrlenv-node binary on this host to enable the "
                "newer commands.",
            )
            return

        cmd_header = _command_header(ctrl)
        if cmd_header is None:
            LOGGER.warning("ignoring ControlMsg with no command header (kind=%s)", body_kind)
            return

        # Idempotency cache hit → return the cached reply without re-running.
        cached = self._cache.get(cmd_header.idempotency_key)
        if cached is not None:
            await outbox.put(
                pb.NodeMsg(
                    stream_epoch=stream_epoch,
                    seq=node_seq.next(),
                    reply=cached,
                )
            )
            return

        # P1.7.A.2 — streaming commands fork to a multi-reply
        # dispatch path. They emit N CommandReply messages all
        # sharing the same command_id; the terminator carries
        # ``done=true`` (signaled in the per-chunk payload). Not
        # cached — a streaming command is workload (replay would
        # mean replaying the entire output stream, which the
        # cache isn't designed for).
        if body_kind == "stream_container_exec":
            await self._dispatch_stream_container_exec(
                ctrl.stream_container_exec, cmd_header,
                outbox, stream_epoch, node_seq,
            )
            return

        # container_get_archive is likewise a multi-reply stream: the
        # tarball is sliced into bounded ContainerGetArchiveChunk
        # replies so a large verifier dir can never produce a single
        # oversized NodeMsg that severs the heartbeat stream. Not
        # cached for the same reason streaming exec isn't — and its
        # idempotency key already carries a fresh uuid per call, so
        # the cache never hit anyway.
        if body_kind == "container_get_archive":
            await self._dispatch_stream_container_get_archive(
                ctrl.container_get_archive, cmd_header,
                outbox, stream_epoch, node_seq,
            )
            return

        # container_put_archive_chunk is the UPLOAD twin: the control plane sends N
        # ControlMsg frames sharing this command_id, each a slice of the tarball. We
        # accumulate them IN FRAME ORDER — dispatch tasks start FIFO (per-seq create_task)
        # and the append in ``_accumulate_put_archive_chunk`` is synchronous, before any
        # await — and on the ``done`` frame run one docker put_archive + emit a single
        # reply. Not cached (fresh-uuid idempotency key per call; workload, like the get
        # stream).
        if body_kind == "container_put_archive_chunk":
            await self._accumulate_put_archive_chunk(
                ctrl.container_put_archive_chunk, cmd_header,
                outbox, stream_epoch, node_seq,
            )
            return

        try:
            reply = await self._execute(ctrl)
        except Exception as exc:
            LOGGER.exception(
                "command %s failed (kind=%s)", cmd_header.command_id, body_kind
            )
            reply = pb.CommandReply(
                command_id=cmd_header.command_id,
                status=pb.ReplyStatus.FAILED,
                error_kind=type(exc).__name__,
                error_message=str(exc),
            )

        self._cache.put(cmd_header.idempotency_key, reply)
        await outbox.put(
            pb.NodeMsg(
                stream_epoch=stream_epoch,
                seq=node_seq.next(),
                reply=reply,
            )
        )

    async def _dispatch_stream_container_exec(
        self,
        cmd: pb.StreamContainerExecCommand,
        cmd_header: pb.CommandHeader,
        outbox: asyncio.Queue[pb.NodeMsg],
        stream_epoch: str,
        node_seq: _MonotonicCounter,
    ) -> None:
        """Multi-reply dispatch for streaming exec.

        Iterates the manager's async generator; for each yielded
        chunk dict, builds a ``CommandReply`` carrying a
        ``ContainerExecChunk`` payload + the same ``command_id``;
        writes each to the outbox. The terminator chunk (the one
        the manager yields with ``done=True``) carries
        ``exit_code`` + ``timed_out``. On a generator-side
        exception, emits a single FAILED reply with the same
        command_id and stops the stream — the consumer-side
        reader closes the queue on FAILED status.
        """
        try:
            async for chunk in self._agent.container_exec_stream(
                rollout_id=cmd.rollout_id,
                container_id=cmd.container_id,
                cmd=list(cmd.cmd),
                timeout_s=cmd.timeout_s or 1800.0,
                cwd=cmd.cwd or None,
                env=dict(cmd.env) if cmd.env else None,
                user=cmd.user or None,
            ):
                reply = pb.CommandReply(
                    command_id=cmd_header.command_id,
                    status=pb.ReplyStatus.OK,
                    container_exec_chunk=pb.ContainerExecChunk(
                        stdout=chunk.get("stdout") or b"",
                        stderr=chunk.get("stderr") or b"",
                        done=bool(chunk.get("done") or False),
                        exit_code=int(chunk.get("exit_code") or 0),
                        timed_out=bool(chunk.get("timed_out") or False),
                    ),
                )
                await outbox.put(
                    pb.NodeMsg(
                        stream_epoch=stream_epoch,
                        seq=node_seq.next(),
                        reply=reply,
                    ),
                )
        except Exception as exc:
            LOGGER.exception(
                "stream_container_exec %s failed",
                cmd_header.command_id,
            )
            await outbox.put(
                pb.NodeMsg(
                    stream_epoch=stream_epoch,
                    seq=node_seq.next(),
                    reply=pb.CommandReply(
                        command_id=cmd_header.command_id,
                        status=pb.ReplyStatus.FAILED,
                        error_kind=type(exc).__name__,
                        error_message=str(exc),
                    ),
                ),
            )

    async def _dispatch_stream_container_get_archive(
        self,
        cmd: pb.ContainerGetArchiveCommand,
        cmd_header: pb.CommandHeader,
        outbox: asyncio.Queue[pb.NodeMsg],
        stream_epoch: str,
        node_seq: _MonotonicCounter,
    ) -> None:
        """Multi-reply dispatch for ``container_get_archive``.

        Streams ``source_path``'s tar back as a sequence of
        ``ContainerGetArchiveChunk`` replies, each at most
        ``ARCHIVE_CHUNK_BYTES``, all sharing ``command_id``. A final
        terminator chunk (``done=true``, empty ``data``) marks
        end-of-stream — sent even for an empty tarball so the consumer
        always sees exactly one terminator. On any error, emits a
        single FAILED reply with the same command_id (the control-plane
        reader closes the collect loop on FAILED).

        Node-lost fix: the tar is pulled off the docker socket one
        chunk at a time via ``container_get_archive_stream`` (each read
        is an ``asyncio.to_thread`` hop on the node), so a large
        ``/testbed`` copy never blocks the event loop and the heartbeat
        keeps flowing. We no longer buffer the whole tarball in node
        memory — each chunk is emitted as it is read. The node-side
        ``_archive_gate`` bounds how many such transfers run at once.
        """
        try:
            stream = self._agent.container_get_archive_stream(
                rollout_id=cmd.rollout_id,
                container_id=cmd.container_id,
                source_path=cmd.source_path,
            )
            async with aclosing(stream):
                async for chunk in stream:
                    # ``get_archive_stream`` already slices at
                    # ARCHIVE_CHUNK_BYTES, but re-slice defensively so a
                    # backend that hands back a larger buffer can still
                    # never produce an oversized NodeMsg.
                    for start in range(0, len(chunk), ARCHIVE_CHUNK_BYTES):
                        await outbox.put(
                            pb.NodeMsg(
                                stream_epoch=stream_epoch,
                                seq=node_seq.next(),
                                reply=pb.CommandReply(
                                    command_id=cmd_header.command_id,
                                    status=pb.ReplyStatus.OK,
                                    container_get_archive_chunk=(
                                        pb.ContainerGetArchiveChunk(
                                            data=chunk[
                                                start:start + ARCHIVE_CHUNK_BYTES
                                            ],
                                            done=False,
                                        )
                                    ),
                                ),
                            ),
                        )
            await outbox.put(
                pb.NodeMsg(
                    stream_epoch=stream_epoch,
                    seq=node_seq.next(),
                    reply=pb.CommandReply(
                        command_id=cmd_header.command_id,
                        status=pb.ReplyStatus.OK,
                        container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                            data=b"", done=True,
                        ),
                    ),
                ),
            )
        except Exception as exc:
            LOGGER.exception(
                "container_get_archive %s failed", cmd_header.command_id,
            )
            await outbox.put(
                pb.NodeMsg(
                    stream_epoch=stream_epoch,
                    seq=node_seq.next(),
                    reply=pb.CommandReply(
                        command_id=cmd_header.command_id,
                        status=pb.ReplyStatus.FAILED,
                        error_kind=type(exc).__name__,
                        error_message=str(exc),
                    ),
                ),
            )

    async def _accumulate_put_archive_chunk(
        self,
        chunk: pb.ContainerPutArchiveChunk,
        cmd_header: pb.CommandHeader,
        outbox: asyncio.Queue[pb.NodeMsg],
        stream_epoch: str,
        node_seq: _MonotonicCounter,
    ) -> None:
        """Accumulate one chunk of a chunked put_archive; run the upload on ``done``.

        The append below MUST stay synchronous (no ``await`` before it): dispatch tasks
        start FIFO in frame order, so a synchronous append preserves tarball byte order
        across the concurrent per-frame tasks. Only the terminal ``done`` frame awaits (the
        assembled docker put_archive + the single reply)."""
        cid = cmd_header.command_id
        entry = self._put_archive_chunks.get(cid)
        if entry is None:
            # First frame carries the routing metadata.
            entry = {
                "rollout_id": chunk.rollout_id,
                "container_id": chunk.container_id,
                "target_dir": chunk.target_dir,
                "parts": [],
            }
            self._put_archive_chunks[cid] = entry
        if chunk.data:
            entry["parts"].append(bytes(chunk.data))  # sync — before any await
        if not chunk.done:
            return

        # Terminal frame — assemble + run one docker put_archive, then reply once.
        self._put_archive_chunks.pop(cid, None)
        try:
            await self._agent.container_put_archive(
                rollout_id=entry["rollout_id"],
                container_id=entry["container_id"],
                target_dir=entry["target_dir"],
                tarball=b"".join(entry["parts"]),
            )
            reply = pb.CommandReply(
                command_id=cid,
                status=pb.ReplyStatus.OK,
                put_archive=pb.PutArchiveReply(),
            )
        except Exception as exc:
            LOGGER.exception("chunked put_archive %s failed", cid)
            reply = pb.CommandReply(
                command_id=cid,
                status=pb.ReplyStatus.FAILED,
                error_kind=type(exc).__name__,
                error_message=str(exc),
            )
        await outbox.put(
            pb.NodeMsg(stream_epoch=stream_epoch, seq=node_seq.next(), reply=reply),
        )

    async def _execute(self, ctrl: pb.ControlMsg) -> pb.CommandReply:
        kind = ctrl.WhichOneof("body")
        if kind == "create":
            return await self._exec_create(ctrl.create)
        if kind == "destroy":
            return await self._exec_destroy(ctrl.destroy)
        if kind == "env_setup":
            return await self._exec_env_setup(ctrl.env_setup)
        if kind == "env_step":
            return await self._exec_env_step(ctrl.env_step)
        if kind == "env_teardown":
            return await self._exec_env_teardown(ctrl.env_teardown)
        if kind == "run_in_sandbox":
            return await self._exec_run_in_sandbox(ctrl.run_in_sandbox)
        if kind == "put_archive":
            return await self._exec_put_archive(ctrl.put_archive)
        if kind == "stats_req":
            return await self._exec_stats(ctrl.stats_req)
        if kind == "fetch_trajectory":
            return await self._exec_fetch_trajectory(ctrl.fetch_trajectory)
        if kind == "list_sandboxes":
            return await self._exec_list_sandboxes(ctrl.list_sandboxes)
        if kind == "query_image":
            return await self._exec_query_image(ctrl.query_image)
        if kind == "report_images":
            return await self._exec_report_images(ctrl.report_images)
        if kind == "build_images":
            return await self._exec_build_images(ctrl.build_images)
        if kind == "ensure_present":
            return await self._exec_ensure_present(ctrl.ensure_present)
        if kind == "build_image":
            return await self._exec_build_image(ctrl.build_image)
        if kind == "cancel_build_image":
            return await self._exec_cancel_build_image(ctrl.cancel_build_image)
        if kind == "acquire_container":
            return await self._exec_acquire_container(ctrl.acquire_container)
        if kind == "acquire_compose_project":
            return await self._exec_acquire_compose_project(
                ctrl.acquire_compose_project,
            )
        if kind == "destroy_compose_project":
            return await self._exec_destroy_compose_project(
                ctrl.destroy_compose_project,
            )
        if kind == "container_exec":
            return await self._exec_container_exec(ctrl.container_exec)
        if kind == "destroy_container":
            return await self._exec_destroy_container(ctrl.destroy_container)
        if kind == "container_put_archive":
            return await self._exec_container_put_archive(ctrl.container_put_archive)
        if kind == "list_raw_containers":
            return await self._exec_list_raw_containers(ctrl.list_raw_containers)
        if kind == "force_destroy_container":
            return await self._exec_force_destroy_container(ctrl.force_destroy_container)
        if kind == "apply_egress":
            return await self._exec_apply_egress(ctrl.apply_egress)
        if kind == "evict_image":
            return await self._exec_evict_image(ctrl.evict_image)
        if kind == "register_scratch_source":
            return await self._exec_register_scratch_source(ctrl.register_scratch_source)
        raise XRLEnvError(f"node link: unknown command kind {kind!r}")

    async def _exec_create(self, cmd: pb.CreateSandboxCommand) -> pb.CommandReply:
        # A5 / D17 stage 1 (audit response): 0.0 sentinel from
        # ``stub_request_timeout_s`` means "unset"; pass None so the
        # NodeAgent falls back to NodeAgentConfig.stub_request_timeout_s.
        cap_override: float | None = (
            cmd.stub_request_timeout_s
            if cmd.stub_request_timeout_s and cmd.stub_request_timeout_s > 0
            else None
        )
        handle = await self._agent.create_sandbox(
            rollout_id=cmd.rollout_id,
            backend=cmd.backend,
            template=conv.template_ref_from_proto(cmd.template),
            resources=conv.resource_spec_from_proto(cmd.resources),
            network_policy=cmd.network_policy or "open",  # type: ignore[arg-type]
            stub_request_timeout_s=cap_override,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            create=pb.CreateSandboxReply(sandbox=conv.sandbox_handle_to_proto(handle)),
        )

    async def _exec_destroy(self, cmd: pb.DestroySandboxCommand) -> pb.CommandReply:
        await self._agent.destroy_sandbox(conv.sandbox_handle_from_proto(cmd.sandbox))
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            destroy=pb.DestroyReply(),
        )

    async def _exec_env_setup(self, cmd: pb.EnvSetupCommand) -> pb.CommandReply:
        body = await self._agent.env_setup(
            conv.sandbox_handle_from_proto(cmd.sandbox),
            adapter_module=cmd.adapter_module,
            adapter_class=cmd.adapter_class,
            init_params=json.loads(cmd.init_params_json or "{}"),
            request_timeout_s=_per_call_cap(cmd.request_timeout_s),
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            env=pb.EnvReply(body_json=json.dumps(body)),
        )

    async def _exec_env_step(self, cmd: pb.EnvStepCommand) -> pb.CommandReply:
        body = await self._agent.env_step(
            conv.sandbox_handle_from_proto(cmd.sandbox),
            json.loads(cmd.action_json or "null"),
            request_timeout_s=_per_call_cap(cmd.request_timeout_s),
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            env=pb.EnvReply(body_json=json.dumps(body)),
        )

    async def _exec_env_teardown(self, cmd: pb.EnvTeardownCommand) -> pb.CommandReply:
        body = await self._agent.env_teardown(
            conv.sandbox_handle_from_proto(cmd.sandbox),
            request_timeout_s=_per_call_cap(cmd.request_timeout_s),
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            env=pb.EnvReply(body_json=json.dumps(body)),
        )

    async def _exec_run_in_sandbox(self, cmd: pb.RunInSandboxCommand) -> pb.CommandReply:
        result = await self._agent.run_in_sandbox(
            conv.sandbox_handle_from_proto(cmd.sandbox),
            list(cmd.cmd),
            timeout_s=cmd.timeout_s or 30.0,
            cwd=cmd.cwd or None,
            env=dict(cmd.env) if cmd.env else None,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            exec=pb.ExecReply(
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
            ),
        )

    async def _exec_acquire_container(
        self, cmd: pb.AcquireContainerCommand,
    ) -> pb.CommandReply:
        record = await self._agent.acquire_container(
            rollout_id=cmd.rollout_id,
            backend="docker",  # P1.7.A.1: only docker backend in phase 1.
            image=cmd.image,
            command=list(cmd.command) if cmd.command else None,
            # proto3 ``repeated`` collapses unset and empty-list; we
            # accept anything non-empty as the caller's intent (so
            # the docker ``--entrypoint ""`` idiom flows through as
            # ``[""]``). See node_control.proto comments.
            entrypoint=list(cmd.entrypoint) if cmd.entrypoint else None,
            user=cmd.user or None,
            cap_add=list(cmd.cap_add) if cmd.cap_add else None,
            devices=list(cmd.devices) if cmd.devices else None,
            privileged=cmd.privileged,
            network_mode=cmd.network_mode or None,
            binds=list(cmd.binds) if cmd.binds else None,
            name=cmd.name or None,
            labels=dict(cmd.labels) if cmd.labels else None,
            environment=dict(cmd.environment) if cmd.environment else None,
            # Wire field is negative-form (strict_image_check) so
            # proto3 default-false aligns with the new default
            # (ensure_image_present=True). See node_control.proto
            # AcquireContainerCommand.strict_image_check.
            ensure_image_present=not cmd.strict_image_check,
            # B5.4 — proto3 default-empty maps to "host".
            userns_mode=cmd.userns_mode or "host",
            # Issue #12 — pull / build deadline override. Proto3
            # ``0.0`` is the "use node's default" sentinel; positive
            # values widen ``ImageCacheManager.ensure_present`` so
            # cold pulls for known-huge images don't fail at the
            # node while the wire patiently waits.
            acquire_timeout_s=(
                cmd.pull_deadline_s if cmd.pull_deadline_s > 0 else None
            ),
            # P1 — effective ResourceSpec for cgroup enforcement. Unset
            # (legacy control plane) → None; the manager then applies
            # its node-default cap, never an unbounded container.
            resources=(
                conv.resource_spec_from_proto(cmd.resources)
                if cmd.HasField("resources")
                else None
            ),
            # P0b — container-shape RuntimeLimits. Unset → None; the
            # manager then applies no pids/shm/tmpfs/read-only constraint.
            runtime_limits=(
                conv.runtime_limits_from_proto(cmd.runtime_limits)
                if cmd.HasField("runtime_limits")
                else None
            ),
            # §5.1 — OCI runtime selector. Empty proto3 string → None
            # (docker default runtime). The manager verifies it is a
            # registered runtime before ``containers.run`` (§5.5).
            container_runtime=cmd.container_runtime or None,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            acquire_container=pb.AcquireContainerReply(
                container_id=record.container_id,
                container_name=record.container_name,
            ),
        )

    async def _exec_acquire_compose_project(
        self, cmd: pb.AcquireComposeProjectCommand,
    ) -> pb.CommandReply:
        record = await self._agent.acquire_compose_project(
            rollout_id=cmd.rollout_id,
            project_name=cmd.project_name,
            compose_yaml=cmd.compose_yaml,
            images=list(cmd.images) if cmd.images else None,
            main_service=cmd.main_service or "main",
            # proto3 ``0.0`` = "use node default"; positive widens the up --wait.
            up_timeout_s=cmd.up_timeout_s if cmd.up_timeout_s > 0 else None,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            acquire_compose_project=pb.AcquireComposeProjectReply(
                main_container_id=record.main_container_id,
                main_container_name=record.main_container_name,
                project_name=record.project_name,
                project_dir=record.project_dir,
                service_container_ids=dict(record.service_container_ids),
            ),
        )

    async def _exec_destroy_compose_project(
        self, cmd: pb.DestroyComposeProjectCommand,
    ) -> pb.CommandReply:
        await self._agent.destroy_compose_project(
            rollout_id=cmd.rollout_id,
            project_name=cmd.project_name,
            force=cmd.force,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            destroy=pb.DestroyReply(),
        )

    async def _exec_container_exec(
        self, cmd: pb.ContainerExecCommand,
    ) -> pb.CommandReply:
        result = await self._agent.container_exec(
            rollout_id=cmd.rollout_id,
            container_id=cmd.container_id,
            cmd=list(cmd.cmd),
            timeout_s=cmd.timeout_s or 30.0,
            cwd=cmd.cwd or None,
            env=dict(cmd.env) if cmd.env else None,
            user=cmd.user or None,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            exec=pb.ExecReply(
                exit_code=int(result.get("exit_code") or 0),
                stdout=result.get("stdout") or b"",
                stderr=result.get("stderr") or b"",
                timed_out=bool(result.get("timed_out") or False),
            ),
        )

    async def _exec_apply_egress(
        self, cmd: pb.ApplyEgressCommand,
    ) -> pb.CommandReply:
        # Proto → typed allowlist at the wire boundary; the node compiles it.
        # An empty ``allow`` is valid (block all external egress).
        allowlist = EgressAllowlist(
            rules=tuple(
                EgressRule(cidr=e.cidr, ports=tuple(e.ports) or None)
                for e in cmd.allow
            ),
        )
        await self._agent.apply_egress(
            rollout_id=cmd.rollout_id,
            container_id=cmd.container_id,
            allowlist=allowlist,
            dns_resolver=cmd.dns_resolver or None,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            apply_egress=pb.ApplyEgressReply(),
        )

    async def _exec_destroy_container(
        self, cmd: pb.DestroyContainerCommand,
    ) -> pb.CommandReply:
        await self._agent.destroy_container(
            rollout_id=cmd.rollout_id,
            container_id=cmd.container_id,
            force=cmd.force,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            destroy=pb.DestroyReply(),
        )

    async def _exec_container_put_archive(
        self, cmd: pb.ContainerPutArchiveCommand,
    ) -> pb.CommandReply:
        await self._agent.container_put_archive(
            rollout_id=cmd.rollout_id,
            container_id=cmd.container_id,
            target_dir=cmd.target_dir,
            tarball=bytes(cmd.tarball),
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            put_archive=pb.PutArchiveReply(),
        )


    async def _exec_list_raw_containers(
        self, cmd: pb.ListRawContainersCommand,
    ) -> pb.CommandReply:
        backend = cmd.backend or "docker"
        # P1.7.C.2 — the info variant carries the correlation labels so the
        # reconciler can recognise/route compose mains. ``container_ids`` (the
        # legacy flat set the diff uses) is derived from it — one docker call.
        # ``container_ids`` MUST stay raw-only (the raw-GC orphan diff depends on
        # it — a compose sidecar there would be force-``docker rm``-ed, leaking
        # the rest of the project).
        info = await self._agent.list_raw_containers_info(backend=backend)
        container_ids = [cid for cid, _rid, _proj in info]
        if cmd.include_all_managed:
            # Audit H11 — the caller (readopt-on-connect) wants the BROADER set:
            # every managed container incl compose sidecars, each with its
            # session_kind, so a sidecar-only survivor can be detected. Separate
            # docker call (different label filter); ``container_ids`` unchanged.
            managed = await self._agent.list_managed_container_info(
                backend=backend,
            )
            containers_pb = [
                pb.RawContainerInfo(
                    container_id=cid, rollout_id=rid,
                    compose_project=proj, session_kind=kind,
                )
                for cid, rid, proj, kind in managed
            ]
        else:
            containers_pb = [
                pb.RawContainerInfo(
                    container_id=cid, rollout_id=rid, compose_project=proj,
                )
                for cid, rid, proj in info
            ]
        # Audit P3 — piggyback the node-autonomous reap reasons (disk
        # guard) so the reconciler can seal coordinator-only orphans with
        # the real cause. Best-effort: absent manager → empty map.
        reaped: dict[str, str] = {}
        mgr = self._agent.raw_container_manager(backend)
        if mgr is not None:
            reaped = mgr.disk_reaped_reasons()
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            list_raw_containers=pb.ListRawContainersReply(
                container_ids=container_ids,
                reaped_reasons=reaped,
                containers=containers_pb,
                # audit H11 — ACK the broad-inventory capability so the CP can tell a fully-
                # inventoried "clean" node from an old agent that returned raw-only.
                all_managed_supported=cmd.include_all_managed,
            ),
        )

    async def _exec_force_destroy_container(
        self, cmd: pb.ForceDestroyContainerCommand,
    ) -> pb.CommandReply:
        await self._agent.force_destroy_raw_container(
            container_id=cmd.container_id,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            destroy=pb.DestroyReply(),
        )

    async def _exec_put_archive(self, cmd: pb.PutArchiveCommand) -> pb.CommandReply:
        await self._agent.put_archive(
            conv.sandbox_handle_from_proto(cmd.sandbox),
            cmd.target_dir,
            cmd.tarball,
            clean_target=cmd.clean_target,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            put_archive=pb.PutArchiveReply(),
        )

    async def _exec_stats(self, cmd: pb.StatsRequest) -> pb.CommandReply:
        usage = await self._agent.stats(conv.sandbox_handle_from_proto(cmd.sandbox))
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            stats=pb.StatsReply(usage=conv.resource_usage_to_proto(usage)),
        )

    async def _exec_list_sandboxes(
        self, cmd: pb.ListSandboxesCommand,
    ) -> pb.CommandReply:
        ids = await self._agent.list_sandbox_ids(backend=cmd.backend or None)
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            list_sandboxes=pb.ListSandboxesReply(sandbox_ids=ids),
        )

    async def _exec_query_image(
        self, cmd: pb.QueryImageCommand,
    ) -> pb.CommandReply:
        result = await self._agent.query_image(cmd.image)
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            query_image=pb.QueryImageReply(
                present=result.present,
                digest=result.digest or "",
                last_used_at=result.last_used_at,
            ),
        )

    async def _exec_report_images(
        self, cmd: pb.ReportImagesCommand,
    ) -> pb.CommandReply:
        report = await self._agent.report_images(
            include_shared_size=cmd.include_shared_size,
        )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            report_images=pb.ReportImagesReply(
                images=[
                    pb.ImageStateEntry(
                        name=img.name,
                        tier=img.tier,
                        size_bytes=int(img.size_bytes),
                        in_use_count=int(img.in_use_count),
                        last_used_at=float(img.last_used_at or 0.0),
                        pinned=bool(img.pinned),
                        owner=img.owner,
                        shared_size_bytes=(
                            int(img.shared_size_bytes)
                            if img.shared_size_bytes is not None
                            else 0
                        ),
                        has_shared_size_bytes=(
                            img.shared_size_bytes is not None
                        ),
                        repo_digest=img.digest or "",
                    )
                    for img in report.images
                ],
                free_disk_bytes=int(report.free_disk_bytes),
                pinned=list(report.pinned),
            ),
        )

    async def _exec_evict_image(
        self, cmd: pb.EvictImageCommand,
    ) -> pb.CommandReply:
        """Operator-driven node-cache eviction (``xrlenv images evict``).

        Delegates to :py:meth:`ImageCacheManager.evict_ref`, which
        matches the ref registry-agnostically against the node's held
        tags and removes the matching image(s). The reply carries the
        per-node status the admin orchestrator aggregates. Never raises
        for the ordinary absent/in-use/daemon-error cases — those are
        surfaced in the reply's ``status``.
        """
        node_id = self._agent.node_id
        cache = self._agent.image_cache
        if cache is None:
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                evict_image=pb.EvictImageReply(
                    node_id=node_id,
                    image_ref=cmd.image_ref,
                    status="failed",
                    error="no ImageCacheManager wired on this node",
                ),
            )
        outcome = await cache.evict_ref(cmd.image_ref, force=cmd.force)
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            evict_image=pb.EvictImageReply(
                node_id=node_id,
                image_ref=cmd.image_ref,
                status=outcome.status,
                error=outcome.detail,
                reclaimed_bytes=int(outcome.reclaimed_bytes),
                removed=list(outcome.removed),
            ),
        )

    async def _exec_ensure_present(
        self, cmd: pb.EnsurePresentCommand,
    ) -> pb.CommandReply:
        """P1.6.g — async pre-fetch from the admission queue (F4=2).

        Calls the wired :class:`ImageCacheManager` ensure_present.
        Falls through to ``backend.pull_image`` for ordinary refs +
        the lazy-builder hook (step 1) for benchmark-internal refs.
        """
        cache = self._agent.image_cache
        if cache is None:
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                ensure_present=pb.EnsurePresentReply(
                    status="failed",
                    error="no ImageCacheManager wired on this node",
                ),
            )
        timeout_s = cmd.timeout_s if cmd.timeout_s > 0 else None
        try:
            # EnsurePresentCommand is the proactive pre-fetch path
            # (build apply / eager prefetch), so use the load-aware
            # prefetch lane — wide while idle, gentle while busy.
            await cache.ensure_present(
                cmd.image_ref, deadline_s=timeout_s, prefetch=True,
            )
        except Exception as exc:
            LOGGER.exception(
                "ensure_present failed for %s", cmd.image_ref,
            )
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                ensure_present=pb.EnsurePresentReply(
                    status="failed", error=str(exc),
                ),
            )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            ensure_present=pb.EnsurePresentReply(status="ok"),
        )

    async def _exec_build_image(
        self, cmd: pb.BuildImageCommand,
    ) -> pb.CommandReply:
        """Source-build dispatch — handle a per-image-ref build job.

        Branches on ``cmd.WhichOneof('source')``:
        - ``git`` → invoke :class:`GitSourceBuilder.build` with a
          ``GitSource`` reconstructed from the proto fields.
        - ``tarball`` → today's tarball-source dispatch is not yet
          shipped on the node side; returns a clear error.
        """
        from xrlenv.control.build_plan import GitSource, TarballSource
        from xrlenv.node.source_builder import GitSourceBuilder

        # The node-side source builder is owned by the NodeAgent so
        # the active-builds registry + persistent source-spec
        # registry are unified across the wire dispatch path here
        # AND the image cache's build-on-acquire hook
        # (``NodeAgent._lookup_image_producer`` consults the same
        # instance). Lazy-constructed on first access.
        get_sb = getattr(self._agent, "source_builder", None)
        if get_sb is not None:
            builder = get_sb()
        else:
            # Test fakes that don't subclass NodeAgent: construct
            # locally and stash on the link as before.
            builder = getattr(self, "_source_builder", None)
            if builder is None:
                builder = GitSourceBuilder()
                self._source_builder = builder

        kind = cmd.WhichOneof("source")
        source: GitSource | TarballSource
        if kind == "git":
            source = GitSource(
                repo=cmd.git.repo, ref=cmd.git.ref,
                subdir=cmd.git.subdir or ".",
                dockerfile=cmd.git.dockerfile or "Dockerfile",
            )
        elif kind == "tarball":
            # Re-encode wire bytes as base64 so the SourceBuilder's
            # public ``build()`` signature (which takes a TarballSource
            # with content_b64) is the single dispatch path. The
            # round-trip cost is one base64 encode + one decode on
            # ≤ 100 MB tarballs; trivial vs the docker build that
            # follows.
            import base64
            source = TarballSource(
                path="<wire>",
                dockerfile=cmd.tarball.dockerfile or "Dockerfile",
                content_b64=base64.b64encode(cmd.tarball.content).decode("ascii"),
            )
        else:
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                build_image=pb.BuildImageReply(
                    image_ref=cmd.image_ref, status="failed",
                    error=f"BuildImageCommand carried no recognized source ({kind!r})",
                ),
            )
        timeout_s = cmd.timeout_s if cmd.timeout_s > 0 else 1800.0
        repo_digest: str | None = None
        try:
            if cmd.push:
                # ``xrlenv build push`` — build AND push image_ref to the
                # registry it encodes, resolving the pushed digest. Registry-HEAD
                # skip (check_registry_first) makes a re-run cheap and keeps
                # overlapping dispatch from double-pushing (build-once fleet-wide).
                result = await builder.build_and_push(
                    image_ref=cmd.image_ref, source=source,
                    timeout_s=timeout_s, labels=dict(cmd.labels),
                    check_registry_first=True,
                )
                status, error, repo_digest = (
                    result.status, result.error, result.repo_digest,
                )
            else:
                status, error = await builder.build(
                    image_ref=cmd.image_ref, source=source,
                    timeout_s=timeout_s, labels=dict(cmd.labels),
                    skip_if_present=bool(cmd.skip_if_present),
                )
        except Exception as exc:
            LOGGER.exception(
                "build_image dispatch raised for %s", cmd.image_ref,
            )
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                build_image=pb.BuildImageReply(
                    image_ref=cmd.image_ref, status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            build_image=pb.BuildImageReply(
                image_ref=cmd.image_ref,
                status=status,
                error=error or "",
                repo_digest=repo_digest or "",
            ),
        )

    async def _exec_register_scratch_source(
        self, cmd: pb.RegisterScratchSourceCommand,
    ) -> pb.CommandReply:
        """Register a content-addressed scratch ref → build source on this
        node (scratch build-on-demand). No build happens here — the later
        ``ensure_present`` builds + pushes to the scratch registry lazily."""
        from xrlenv.control.build_plan import GitSource, TarballSource
        from xrlenv.node.source_builder import GitSourceBuilder

        get_sb = getattr(self._agent, "source_builder", None)
        if get_sb is not None:
            builder = get_sb()
        else:
            builder = getattr(self, "_source_builder", None)
            if builder is None:
                builder = GitSourceBuilder()
                self._source_builder = builder

        def _reply(status: str, error: str = "") -> pb.CommandReply:
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                register_scratch_source=pb.RegisterScratchSourceReply(
                    image_ref=cmd.image_ref, status=status, error=error,
                ),
            )

        kind = cmd.WhichOneof("source")
        source: GitSource | TarballSource
        if kind == "git":
            source = GitSource(
                repo=cmd.git.repo, ref=cmd.git.ref,
                subdir=cmd.git.subdir or ".",
                dockerfile=cmd.git.dockerfile or "Dockerfile",
            )
        elif kind == "tarball":
            import base64
            source = TarballSource(
                path="<wire>",
                dockerfile=cmd.tarball.dockerfile or "Dockerfile",
                content_b64=base64.b64encode(cmd.tarball.content).decode("ascii"),
            )
        else:
            return _reply(
                "failed",
                f"RegisterScratchSourceCommand carried no recognized "
                f"source ({kind!r})",
            )
        try:
            builder.register_scratch_source(
                cmd.image_ref, source, durable_to=cmd.durable_to or None,
            )
        except Exception as exc:
            LOGGER.exception(
                "register_scratch_source dispatch raised for %s", cmd.image_ref,
            )
            return _reply("failed", f"{type(exc).__name__}: {exc}")
        return _reply("ok")

    async def _exec_cancel_build_image(
        self, cmd: pb.CancelBuildImageCommand,
    ) -> pb.CommandReply:
        """Operator-driven mid-build cancel.

        Looks up the per-link ``GitSourceBuilder`` instance lazily
        constructed by ``_exec_build_image`` and asks it to cancel
        the named ``image_ref``. When the builder doesn't exist
        (e.g. cancel arrives at a node that never received a build),
        the reply is ``ok`` (idempotent: nothing to cancel = success).
        """
        # Find the same builder instance the build path uses —
        # NodeAgent owns it (sub-slice 2). Fallback to test-local
        # ``self._source_builder`` for FakeAgent-backed tests.
        get_sb = getattr(self._agent, "source_builder", None)
        builder = (
            get_sb() if get_sb is not None
            else getattr(self, "_source_builder", None)
        )
        if builder is None:
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                cancel_build_image=pb.CancelBuildImageReply(
                    image_ref=cmd.image_ref, status="ok",
                    error="",
                ),
            )
        try:
            status, error = await builder.cancel(cmd.image_ref)
        except Exception as exc:
            LOGGER.exception(
                "cancel_build_image dispatch raised for %s",
                cmd.image_ref,
            )
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.OK,
                cancel_build_image=pb.CancelBuildImageReply(
                    image_ref=cmd.image_ref, status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            cancel_build_image=pb.CancelBuildImageReply(
                image_ref=cmd.image_ref,
                status=status,
                error=error or "",
            ),
        )

    async def _exec_build_images(
        self, cmd: pb.BuildImagesCommand,
    ) -> pb.CommandReply:
        """P1.6.c — execute one node's slice of a build plan.

        Loads each per-benchmark builder via
        :func:`xrlenv.control.image_builder.load_image_builder`, runs the
        builds sequentially, returns one BuildResultEntry per assignment.
        Phase-A is batched (caller waits for the whole node's job to
        finish); a streaming variant is a phase-2 follow-on.
        """
        from xrlenv.control.image_builder import (
            ImageBuilderDecl,
            ImageBuilderImportError,
            load_image_builder,
        )
        from xrlenv.control.node_builder import BuilderRef

        cache: dict[tuple[str, str], object] = {}

        def _get_builder(module: str, class_name: str) -> object:
            key = (module, class_name)
            if key not in cache:
                decl = ImageBuilderDecl.model_validate({
                    "module": module, "class": class_name,
                })
                cache[key] = load_image_builder(decl)
            return cache[key]

        # P1.6.g — H3 lazy lifecycle: register every assignment's
        # builder mapping FIRST (even before kicking off any
        # synchronous build), so a later ``ensure_present`` call
        # for the same image_ref — fired by the agent's create
        # path or the new admission-time pre-fetch — can dispatch
        # to the right benchmark builder without a fresh control-
        # plane round-trip. The mapping is process-lifetime; cross-
        # restart recovery = re-apply the plan.
        # Audit P1.6.g-H1 fix (2026-05-05): include ``lazy_registrations``
        # too — those are deferred rows that didn't fit the budget at
        # apply time but need their builder mapping registered so the
        # lazy hook can produce them on first rollout.
        lazy_mapping: dict[str, tuple[BuilderRef, dict[str, str]]] = {}
        for assignment in (*cmd.assignments, *cmd.lazy_registrations):
            ref = cmd.builder_per_benchmark.get(assignment.benchmark)
            if ref is None or not ref.module:
                continue
            kw_entry = cmd.kwargs_per_benchmark.get(assignment.benchmark)
            lazy_mapping[assignment.image_ref] = (
                BuilderRef(module=ref.module, class_name=ref.class_name),
                dict(kw_entry.kv) if kw_entry is not None else {},
            )
        if lazy_mapping:
            self._agent.register_lazy_image_builders(lazy_mapping)

        results: list[pb.BuildResultEntry] = []
        for assignment in cmd.assignments:
            ref = cmd.builder_per_benchmark.get(assignment.benchmark)
            if ref is None or not ref.module:
                results.append(pb.BuildResultEntry(
                    image_ref=assignment.image_ref, status="failed",
                    error=(
                        f"no image_builder registered for benchmark "
                        f"{assignment.benchmark!r}"
                    ),
                ))
                continue
            kw_entry = cmd.kwargs_per_benchmark.get(assignment.benchmark)
            kwargs = dict(kw_entry.kv) if kw_entry is not None else {}
            try:
                builder = _get_builder(ref.module, ref.class_name)
            except ImageBuilderImportError as exc:
                results.append(pb.BuildResultEntry(
                    image_ref=assignment.image_ref, status="failed",
                    error=f"builder load failed: {exc}",
                ))
                continue
            try:
                result = await builder.build(  # type: ignore[attr-defined]
                    image_ref=assignment.image_ref,
                    kwargs=kwargs, force=bool(cmd.force),
                )
            except Exception as exc:
                LOGGER.exception(
                    "build_images: builder.build raised on %s",
                    assignment.image_ref,
                )
                results.append(pb.BuildResultEntry(
                    image_ref=assignment.image_ref, status="failed",
                    error=f"builder.build raised: {exc}",
                ))
                continue
            results.append(pb.BuildResultEntry(
                image_ref=result.image_ref,
                status=result.status,
                bytes_pulled=int(result.bytes_pulled),
                duration_s=float(result.duration_s),
                error=result.error or "",
            ))
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            build_images=pb.BuildImagesReply(results=results),
        )

    async def _exec_fetch_trajectory(
        self, cmd: pb.FetchTrajectoryCommand,
    ) -> pb.CommandReply:
        range_kind = _proto_range_kind_to_str(cmd.range)
        try:
            trajectory = await self._agent.fetch_trajectory(
                cmd.rollout_id,
                range_kind=range_kind,
                step_start=int(cmd.step_start),
                step_end=int(cmd.step_end) if cmd.step_end > 0 else None,
            )
        except FileNotFoundError as exc:
            # Spec-17 ReplayUnavailable: distinguish "no body on disk" from a
            # generic command failure so the cache layer maps it to a
            # rendered "no recording" page rather than a 500.
            return pb.CommandReply(
                command_id=cmd.header.command_id,
                status=pb.ReplyStatus.FAILED,
                error_kind="ReplayUnavailable",
                error_message=str(exc),
            )
        body_json = trajectory.model_dump_json()
        # Total step count belongs to the WHOLE-trajectory shape; for
        # SUMMARY_ONLY we surface the underlying ``step_count`` from
        # PlatformJsonlSink's meta.json so the viewer can render the count
        # without paying for the body.
        if range_kind == "summary_only":
            step_count = (trajectory.metadata or {}).get(
                "step_count", len(trajectory.steps),
            )
        else:
            step_count = len(trajectory.steps)
        return pb.CommandReply(
            command_id=cmd.header.command_id,
            status=pb.ReplyStatus.OK,
            trajectory=pb.TrajectoryReply(
                body_json=body_json,
                step_count=int(step_count),
                summary_only=(range_kind == "summary_only"),
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _MonotonicCounter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def _per_call_cap(raw: float) -> float | None:
    """Decode an env-command's ``request_timeout_s`` proto field.

    Proto3 doubles default to 0.0 when unset; the wire contract is
    "0.0 means no per-call override, fall back to the per-sandbox cap
    staged at create_sandbox time." Translate that to ``None`` so the
    NodeAgent's per-call kwarg semantics match its Protocol shape.
    Negative values are also treated as unset (defensive — the proto
    schema doesn't constrain sign, but a negative timeout would mean
    nothing useful here).
    """
    return raw if raw > 0 else None


def _command_header(ctrl: pb.ControlMsg) -> pb.CommandHeader | None:
    """Return the embedded :class:`CommandHeader` (if any)."""
    kind = ctrl.WhichOneof("body")
    body = getattr(ctrl, kind, None) if kind else None
    if body is None:
        return None
    header = getattr(body, "header", None)
    return header if isinstance(header, pb.CommandHeader) else None


def _proto_range_kind_to_str(kind: int) -> FetchRangeKind:
    """Spec-17 enum projection from the wire to the local string Literal."""
    if kind == pb.FetchRangeKind.SUMMARY_ONLY:
        return "summary_only"
    if kind == pb.FetchRangeKind.STEP_RANGE:
        return "step_range"
    return "whole"


__all__ = ["NodeGrpcLink"]
