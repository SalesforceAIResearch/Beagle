"""Control-plane side of the spec-21 bidi stream.

Two halves:

- :class:`NodeControlServicer` is the gRPC service the node connects out to.
  Per connected node it reads ``NodeHello``, replies with ``ControlHello``,
  spins a reader/writer pair, and binds the connection to a
  :class:`RemoteNodeTransport`.

- :class:`RemoteNodeTransport` implements :class:`NodeTransport` for the
  coordinator + scheduler. Each method generates a ``command_id``, builds a
  ``ControlMsg``, ships it on the per-node outbox, and awaits the matching
  ``CommandReply``. From the coordinator's POV it's indistinguishable from
  a local :class:`NodeAgent`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, cast

from xrlenv.api import converters as conv
from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.api._pb2 import node_control_pb2_grpc as pb_grpc
from xrlenv.api.constants import ARCHIVE_CHUNK_BYTES
from xrlenv.backends.base import (
    ExecResult,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    RuntimeLimits,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.backends.egress import EgressAllowlist
from xrlenv.buildinfo import agent_identity
from xrlenv.errors import (
    ManagedInventoryUnsupported,
    NodeCommandTimeout,
    XRLEnvError,
)
from xrlenv.node.hw_probe import HardwareInfo

if TYPE_CHECKING:
    from xrlenv.node.image_cache import (
        EvictOutcome,
        ImageQueryResult,
        NodeImageReport,
    )
from xrlenv.node.trajectory_reader import FetchRangeKind
from xrlenv.observability.tracing import get_tracer
from xrlenv.types import Trajectory

LOGGER = logging.getLogger(__name__)

NodeConnectedCallback = Callable[["RemoteNodeTransport"], None]
NodeDisconnectedCallback = Callable[["RemoteNodeTransport"], None]

# Issue #12 — wire timeout default for AcquireContainer. Set to match
# ``ImageCacheConfig.default_pull_timeout_s`` (600 s) so a legitimate
# cold-pull surfaces the node's own "pull failed" error rather than
# racing it with a "command timed out". Consumers pass
# ``acquire_timeout_s=N`` per-call to widen the deadline for known-
# huge images (SWE-bench Pro 5-15 GB tags, multi-GB GPU images).
DEFAULT_ACQUIRE_TIMEOUT_S: float = 600.0

# B7.6 — control-plane wait ceiling for ``report_images``. The node-side
# handler runs ``docker system df`` (SharedSize), whose cost grows with the
# node's image count, so the old admin-side 5 s ceiling tripped on every
# node once a real catalog landed. Giving the RPC its OWN timeout makes a
# slow node take ``_send_and_wait``'s clean timeout branch (pops pending +
# flags the command-timeout health marker) instead of being cancelled by
# the admin's outer ``wait_for`` — which would pop the future here and then
# log "reply for unknown command_id" when the node's late reply lands.
# Operator-tunable for very large per-node catalogs.
DEFAULT_REPORT_IMAGES_TIMEOUT_S: float = float(
    os.environ.get("XRLENV_REPORT_IMAGES_TIMEOUT_S", "60"),
)


# Inter-chunk (output-idle) ceiling for a STREAMING exec: how long the CP waits
# for the next stdout/stderr chunk before aborting the stream. The whole exec is
# already bounded by the request's ``timeout_s`` (node side) and the CP's
# ``stream_timeout_s = timeout_s + 30``, so this ceiling only exists to fail a
# stream faster than that whole-exec budget. The default is deliberately LARGE
# (an hour) so it effectively defers to the per-exec ``timeout_s``: a legitimately
# silent test/compile phase — common in benchmark verifiers, and slower under
# high per-node concurrency — is no longer killed at a tight idle window and then
# spuriously retried. A genuinely dead node is still caught by the heartbeat /
# stream-disconnect path, NOT this ceiling. Operator-tunable via
# ``XRLENV_EXEC_CHUNK_TIMEOUT_S`` (seconds).
_DEFAULT_EXEC_CHUNK_TIMEOUT_S: float = 3600.0


def _resolve_exec_chunk_timeout_s() -> float:
    """Read ``XRLENV_EXEC_CHUNK_TIMEOUT_S`` (seconds), defaulting to
    :data:`_DEFAULT_EXEC_CHUNK_TIMEOUT_S`. Fail-soft: a non-numeric or non-positive
    value logs a warning and falls back to the default — a ``<= 0`` ceiling would
    make ``asyncio.wait_for`` time out on the first idle tick and abort EVERY exec.
    """
    raw = os.environ.get("XRLENV_EXEC_CHUNK_TIMEOUT_S")
    if raw is None or raw.strip() == "":
        return _DEFAULT_EXEC_CHUNK_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        LOGGER.warning(
            "grpc-endpoint: ignoring non-numeric XRLENV_EXEC_CHUNK_TIMEOUT_S=%r; "
            "using %.0fs", raw, _DEFAULT_EXEC_CHUNK_TIMEOUT_S,
        )
        return _DEFAULT_EXEC_CHUNK_TIMEOUT_S
    if val <= 0:
        LOGGER.warning(
            "grpc-endpoint: XRLENV_EXEC_CHUNK_TIMEOUT_S=%r must be > 0; using %.0fs",
            raw, _DEFAULT_EXEC_CHUNK_TIMEOUT_S,
        )
        return _DEFAULT_EXEC_CHUNK_TIMEOUT_S
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Duck-typed record returned by RemoteNodeTransport.acquire_container.
# Same field names as ``xrlenv.node.raw_container.RawContainerRecord`` —
# tests + downstream code use attribute access, so the in-process
# pydantic model and this remote-side dataclass are interchangeable
# at the call site. Importing the pydantic class here would create a
# control→node import that the rest of grpc_endpoint avoids.
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _RemoteRawContainerRecord:
    rollout_id: str
    container_id: str
    container_name: str
    image: str


@dataclasses.dataclass(frozen=True)
class _RemoteComposeProjectRecord:
    """Duck-typed mirror of :class:`xrlenv.node.raw_compose.ComposeProjectRecord`
    (same field names) so the coordinator consumes the remote + in-process
    transports identically without importing the node module into the control
    plane."""

    project_name: str
    project_dir: str
    main_container_id: str
    main_container_name: str
    service_container_ids: dict[str, str]

    @property
    def member_container_ids(self) -> tuple[str, ...]:
        return tuple(self.service_container_ids.values())


# ──────────────────────────────────────────────────────────────────────────────
# RemoteNodeTransport — implements NodeTransport over a bidi gRPC stream
# ──────────────────────────────────────────────────────────────────────────────


class RemoteNodeTransport:
    """Per-connection :class:`NodeTransport` impl.

    Lifecycle is owned by :class:`NodeControlServicer`: the servicer creates
    one instance when a node's stream opens and discards it when the stream
    closes. Tracks an outbox of pending :class:`ControlMsg`-shaped messages
    plus a map of in-flight ``command_id`` → :class:`asyncio.Future` that the
    reader half resolves on receipt of the matching :class:`CommandReply`.
    """

    def __init__(
        self,
        *,
        node_id: str,
        backends: list[str],
        hardware: HardwareInfo,
        outbox: asyncio.Queue[pb.ControlMsg],
        stream_epoch: str,
        control_instance_id: str,
        control_seq: _MonotonicCounter,
        supported_runtimes: list[str] | None = None,
        default_runtime: str = "runc",
        isolation_capable: bool = False,
        chunked_put_archive_capable: bool = False,
    ) -> None:
        self.node_id = node_id
        self._backends = list(backends)
        # §5.3 — runtimes advertised on NodeHello. Empty (a pre-§5.3
        # agent) is normalized to ["runc"] so every node is at least
        # runc-capable and the normal path never sees an empty set.
        self._supported_runtimes = list(supported_runtimes or []) or ["runc"]
        self._default_runtime = default_runtime or "runc"
        # P6 (§8.6) — whether the node can enforce the shared-parent cpuset
        # scheme (from NodeHello). ``False`` on agents predating the field.
        self._isolation_capable = bool(isolation_capable)
        # WS1 — whether the node reassembles chunked ContainerPutArchiveChunk commands.
        # ``False`` on agents predating the field → the CP sends the unary command instead.
        self._chunked_put_archive_capable = bool(chunked_put_archive_capable)
        self._hardware = hardware
        self._outbox = outbox
        self._stream_epoch = stream_epoch
        self._control_instance_id = control_instance_id
        self._seq = control_seq
        self._pending: dict[str, asyncio.Future[pb.CommandReply]] = {}
        # P1.7.A.2 — streaming commands' multi-reply fan-out.
        # ``_send_and_stream`` registers a queue here keyed by
        # command_id; ``deliver_reply`` posts every chunk to the
        # queue. The consumer-side async iterator drains the
        # queue until a terminator (chunk.done=True) or a FAILED
        # reply arrives.
        self._pending_streams: dict[str, asyncio.Queue[pb.CommandReply]] = {}
        self._closed = False
        # Watchdog-initiated stream teardown (self-heal after a false loss).
        # Deregistering a node from the in-memory registry does NOT by itself
        # break the still-alive bidi stream: the servicer generator stays
        # parked reading from the node / on ``outbox.get()``, the node's
        # HTTP/2 keepalive pings keep being answered, so the node never sees
        # an RpcError and never redials — it stays ``lost`` forever with a
        # live socket (the 2026-08-21 outage). ``request_terminate`` sets
        # this event; the servicer main loop waits on it and, when set,
        # returns from the generator so the RPC completes and the node
        # reconnects (fresh NodeHello → re-register).
        self._terminate: asyncio.Event = asyncio.Event()
        self._terminate_reason: str = ""
        # Last-heartbeat-seen timestamp (control-plane wall clock). The
        # NodeRegistry watchdog reads this to decide when a node is dead.
        self._last_heartbeat_at: float = time.monotonic()
        # Optional per-heartbeat hook the registry sets to mirror the
        # heartbeat into the state-store (``nodes.last_seen_at``). Without
        # it, ``xrlenv nodes`` showed ever-growing "Xm ago" while the
        # in-memory watchdog (which reads ``_last_heartbeat_at``) said
        # the node was healthy.
        self._on_heartbeat: Callable[[str], None] | None = None
        # Issue #14 — last (free, total) disk bytes reported on the most
        # recent heartbeat. ``(0, 0)`` until the first heartbeat arrives;
        # the placement gate treats that sentinel as "unknown / healthy"
        # so freshly-connected nodes aren't refused work before reporting.
        self._last_free_disk_bytes: int = 0
        self._last_total_disk_bytes: int = 0
        # P6 (§8.6, R6) — last (free, total) pinnable-CPU counts reported on the
        # most recent heartbeat. ``(0, 0)`` until the first heartbeat / on an
        # agent predating the fields; callers treat ``total == 0`` as "unknown"
        # and make no pinned-capacity decision (mirrors the disk sentinel).
        self._last_pinned_cpus_free: int = 0
        self._last_pinned_cpus_total: int = 0
        # Stage-1 admission/capacity — last per-node health stats from the
        # heartbeat (notes/admission-stage-1-observability.md). ``None``
        # until a Stage-1 node-agent reports; stored as a plain dict so
        # the registry can mirror it to ``state.db`` without the proto.
        self._last_health: dict[str, Any] | None = None
        # Issue #18 (Ask #2) — node-health timeout tracking. When a
        # command reply-wait hits its ceiling in ``_send_and_wait``,
        # the node-agent is wedged or overloaded (it may still be
        # heartbeating — a stalled docker daemon doesn't stop the
        # heartbeat task). The scheduler reads
        # ``seconds_since_last_command_timeout`` and excludes the node
        # from placement while it's within a recent cooldown window,
        # so a degraded node stops attracting acquires. ``monotonic``
        # so it's immune to wall-clock adjustment; ``None`` until the
        # first timeout. ``_command_timeout_total`` is cumulative for
        # admin observability only — not consulted by the gate.
        self._last_command_timeout_monotonic: float | None = None
        self._command_timeout_total: int = 0
        # Audit P3 — rollout_id -> reason for rollouts the node reaped
        # autonomously (disk guard), refreshed on each
        # ``list_raw_container_ids`` sweep. The raw-GC reconciler reads
        # this to seal a coordinator-only orphan with the real cause.
        self._last_reaped_reasons: dict[str, str] = {}
        # P1.7.C.2 — per-container correlation labels from the last
        # ``list_raw_container_ids`` sweep (container_id -> (rollout_id,
        # compose_project)); the raw-GC reconciler reads it to recognise/route
        # compose-project mains.
        self._last_container_info: dict[str, tuple[str, str]] = {}

    # ── Public NodeTransport surface ─────────────────────────────────────────

    def supported_backends(self) -> list[str]:
        return list(self._backends)

    def supported_runtimes(self) -> list[str]:
        return list(self._supported_runtimes)

    def default_runtime(self) -> str:
        return self._default_runtime

    def isolation_capable(self) -> bool:
        """P6 (§8.6) — whether this node advertised that it can enforce the
        shared-parent cpuset isolation scheme (from NodeHello). ``False`` on
        agents predating the field / nodes that failed the self-test."""
        return self._isolation_capable

    def pinned_cpu_state(self) -> tuple[int, int]:
        """P6 (§8.6, R6) — most recent ``(free, total)`` pinnable-CPU counts
        reported via heartbeat. ``(0, 0)`` when the node hasn't reported yet;
        callers must treat ``total == 0`` as "unknown" (mirrors
        :py:meth:`disk_state`)."""
        return (self._last_pinned_cpus_free, self._last_pinned_cpus_total)

    def hardware(self) -> HardwareInfo:
        return self._hardware

    def disk_state(self) -> tuple[int, int]:
        """Issue #14 — most recent ``(free_bytes, total_bytes)`` reported
        via heartbeat. ``(0, 0)`` when the node hasn't reported yet (e.g.
        just connected, hasn't sent its first beat). Callers must treat
        ``total == 0`` as "unknown" rather than "no disk."
        """
        return (self._last_free_disk_bytes, self._last_total_disk_bytes)

    def seconds_since_last_command_timeout(self) -> float | None:
        """Issue #18 (Ask #2) — elapsed seconds since the last command
        reply-timeout, or ``None`` if the node has never timed out.
        See the field docstring in ``__init__`` for the placement-gate
        contract."""
        if self._last_command_timeout_monotonic is None:
            return None
        return time.monotonic() - self._last_command_timeout_monotonic

    async def create_sandbox(
        self,
        *,
        rollout_id: str,
        backend: str,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
        stub_request_timeout_s: float | None = None,
    ) -> SandboxHandle:
        header = self._fresh_header(idempotency_key=rollout_id)
        msg = self._control_msg(
            create=pb.CreateSandboxCommand(
                header=header,
                rollout_id=rollout_id,
                backend=backend,
                template=conv.template_ref_to_proto(template),
                resources=conv.resource_spec_to_proto(resources),
                network_policy=network_policy,
                # A5 / D17 stage 1: 0.0 sentinel = unset → node falls
                # back to its NodeAgentConfig default. Audit response.
                stub_request_timeout_s=(
                    stub_request_timeout_s
                    if stub_request_timeout_s is not None
                    else 0.0
                ),
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        return conv.sandbox_handle_from_proto(reply.create.sandbox)

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        header = self._fresh_header(idempotency_key=f"{sb.id}:destroy")
        msg = self._control_msg(
            destroy=pb.DestroySandboxCommand(
                header=header, sandbox=conv.sandbox_handle_to_proto(sb)
            )
        )
        await self._send_and_wait(msg, header.command_id)

    async def env_setup(
        self,
        sb: SandboxHandle,
        *,
        adapter_module: str,
        adapter_class: str,
        init_params: dict[str, Any],
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        header = self._fresh_header(idempotency_key=f"{sb.id}:env_setup")
        msg = self._control_msg(
            env_setup=pb.EnvSetupCommand(
                header=header,
                sandbox=conv.sandbox_handle_to_proto(sb),
                adapter_module=adapter_module,
                adapter_class=adapter_class,
                init_params_json=json.dumps(init_params),
                # Proto3 default for double is 0.0; ``None`` here means
                # "no per-call override, fall back to the per-sandbox
                # cap" — encode as 0.0 so old node-agents that don't
                # decode the field still get the safe fallback.
                request_timeout_s=request_timeout_s or 0.0,
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        body = json.loads(reply.env.body_json or "{}")
        return body if isinstance(body, dict) else {"value": body}

    async def env_step(
        self,
        sb: SandboxHandle,
        action: Any,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        # Each step gets a unique idempotency key — replays of a step would
        # double-execute the agent's action, which is a workload bug, not
        # something the cache should silently absorb. We append a UUID so the
        # cache doesn't conflate distinct steps.
        header = self._fresh_header(idempotency_key=f"{sb.id}:step:{uuid.uuid4().hex}")
        msg = self._control_msg(
            env_step=pb.EnvStepCommand(
                header=header,
                sandbox=conv.sandbox_handle_to_proto(sb),
                action_json=json.dumps(action),
                request_timeout_s=request_timeout_s or 0.0,
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        body = json.loads(reply.env.body_json or "{}")
        return body if isinstance(body, dict) else {"value": body}

    async def env_teardown(
        self,
        sb: SandboxHandle,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        header = self._fresh_header(idempotency_key=f"{sb.id}:env_teardown")
        msg = self._control_msg(
            env_teardown=pb.EnvTeardownCommand(
                header=header,
                sandbox=conv.sandbox_handle_to_proto(sb),
                request_timeout_s=request_timeout_s or 0.0,
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        body = json.loads(reply.env.body_json or "{}")
        return body if isinstance(body, dict) else {"value": body}

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        header = self._fresh_header(idempotency_key=f"{sb.id}:stats:{uuid.uuid4().hex}")
        msg = self._control_msg(
            stats_req=pb.StatsRequest(
                header=header, sandbox=conv.sandbox_handle_to_proto(sb)
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        return conv.resource_usage_from_proto(reply.stats.usage)

    async def list_sandbox_ids(self, *, backend: str | None = None) -> list[str]:
        # A3 / D15: per-call UUID idempotency key — the reconciler
        # fires periodically and must observe the *current* node-side
        # set, never a cached prior reply. Same pattern as ``stats``.
        header = self._fresh_header(
            idempotency_key=f"list_sandboxes:{uuid.uuid4().hex}"
        )
        msg = self._control_msg(
            list_sandboxes=pb.ListSandboxesCommand(
                header=header, backend=backend or "",
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        return list(reply.list_sandboxes.sandbox_ids)

    async def query_image(self, image: str) -> ImageQueryResult:
        # A1 / D18+D19: per-call UUID idempotency key — image presence
        # can flip between calls (eviction in-flight, warmup landing,
        # operator-initiated rmi). The cache must NOT absorb prior
        # replies; same per-call rationale as ``stats`` and
        # ``list_sandbox_ids``.
        from xrlenv.node.image_cache import ImageQueryResult

        header = self._fresh_header(
            idempotency_key=f"query_image:{uuid.uuid4().hex}",
        )
        msg = self._control_msg(
            query_image=pb.QueryImageCommand(header=header, image=image),
        )
        reply = await self._send_and_wait(msg, header.command_id)
        body = reply.query_image
        return ImageQueryResult(
            present=body.present,
            digest=body.digest or None,
            last_used_at=body.last_used_at,
        )

    async def report_images(
        self,
        *,
        timeout_s: float = DEFAULT_REPORT_IMAGES_TIMEOUT_S,
        include_shared_size: bool = False,
    ) -> NodeImageReport:
        # B7.6: per-call UUID — same rationale as query_image. The
        # cache state mutates (acquire / release / evict) faster than
        # the idempotency cache TTL, so a stale reply replay would
        # show the operator the wrong picture.
        from xrlenv.node.image_cache import (
            ImageStateRecord,
            NodeImageReport,
        )

        header = self._fresh_header(
            idempotency_key=f"report_images:{uuid.uuid4().hex}",
        )
        msg = self._control_msg(
            report_images=pb.ReportImagesCommand(
                header=header, include_shared_size=include_shared_size,
            ),
        )
        # Own timeout so a slow ``docker system df`` on a large catalog
        # surfaces a clean XRLEnvError (pending popped, node flagged
        # degraded) rather than being cancelled by an outer wait_for.
        cp_timeout = timeout_s if timeout_s > 0 else None
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=cp_timeout,
        )
        body = reply.report_images
        records = [
            ImageStateRecord(
                name=entry.name,
                tier=entry.tier,  # type: ignore[arg-type]
                size_bytes=int(entry.size_bytes),
                # SharedSize plumbing (2026-05-11): nodes on the
                # updated proto set ``has_shared_size_bytes=True`` so
                # the admin distinguishes "no sharing" (= 0 bytes)
                # from "no info" (older daemons / in-memory backend).
                # When unset, fall back to ``None`` so calibrate's
                # ``unique = size - shared`` reverts to ``size_bytes``
                # for backward compatibility.
                shared_size_bytes=(
                    int(entry.shared_size_bytes)
                    if entry.has_shared_size_bytes
                    else None
                ),
                in_use_count=int(entry.in_use_count),
                last_used_at=entry.last_used_at or None,
                pinned=bool(entry.pinned),
                # ``owner`` was added in the B7.6 admin-filter follow-on;
                # nodes still on the older proto leave it empty, which we
                # fall back to ``external`` so the default-on filter
                # doesn't drop their images entirely.
                owner=entry.owner or "external",  # type: ignore[arg-type]
                # Manifest digest (calibrate digest-match): nodes on the
                # updated proto set ``repo_digest`` from Docker RepoDigests;
                # older nodes leave it empty → ``None`` (calibrate falls back
                # to the tag/repo-path matchers, same as before).
                digest=entry.repo_digest or None,
            )
            for entry in body.images
        ]
        return NodeImageReport(
            images=records,
            free_disk_bytes=int(body.free_disk_bytes),
            pinned=tuple(body.pinned),
        )

    async def ensure_present(
        self, image_ref: str, *, timeout_s: float = 0.0,
    ) -> tuple[str, str]:
        """P1.6.g — admission-time async pre-fetch (F4=2).

        Returns ``(status, error)`` where status ∈ {"ok", "failed"}.
        Per-call UUID idempotency key: ensure_present is intent-safe
        to retry but the cache state is mutable, so a stale-reply
        replay would mislead the caller. Mirrors query_image's
        rationale.
        """
        header = self._fresh_header(
            idempotency_key=f"ensure_present:{uuid.uuid4().hex}",
        )
        msg = self._control_msg(
            ensure_present=pb.EnsurePresentCommand(
                header=header, image_ref=image_ref,
                timeout_s=float(timeout_s),
            ),
        )
        # Control-plane ceiling = node-side timeout + 60 s buffer so a
        # normal node-side timeout reply arrives before the control-plane
        # gives up.  Without this the control-plane waited forever if the
        # node silently dropped the command.
        cp_timeout = (timeout_s + 60.0) if timeout_s > 0 else None
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=cp_timeout,
        )
        body = reply.ensure_present
        return (str(body.status), str(body.error))

    async def build_image(
        self,
        *,
        image_ref: str,
        source: Any,   # GitSource | TarballSource (build_plan types)
        timeout_s: float,
        labels: dict[str, str] | None = None,
        skip_if_present: bool = False,
    ) -> tuple[str, str]:
        """Source-build dispatch (per-image-ref). Lowers a plan
        entry's ``GitSource`` (or ``TarballSource``) into a spec-21
        ``BuildImageCommand`` and waits for the node's
        ``BuildImageReply``.

        ``skip_if_present`` makes the node-side builder short-circuit
        if the image is already tagged locally — used by the
        operator-driven ``xrlenv build apply --skip-if-present`` flow
        for warm-cluster re-applies (post-calibrate, partial-failure
        recovery).

        Per-call UUID idempotency key: builds are not safe to absorb
        a stale reply (the docker daemon's view of the image is
        mutable; cache-hit replies on the wire would mislead the
        coordinator's status update).

        Returns ``(status, error)`` where status ∈ {"ok", "failed"}.
        """
        body = await self._dispatch_build_image(
            image_ref=image_ref, source=source, timeout_s=timeout_s,
            labels=labels, skip_if_present=skip_if_present, push=False,
        )
        return (str(body.status), str(body.error))

    async def build_and_push_image(
        self,
        *,
        image_ref: str,
        source: Any,   # GitSource | TarballSource (build_plan types)
        timeout_s: float,
        labels: dict[str, str] | None = None,
    ) -> tuple[str, str, str | None]:
        """Source-build-AND-push dispatch (``xrlenv build push``).

        Same source-lowering as :meth:`build_image` but sets ``push=true`` so
        the node builds ``image_ref`` and pushes it to the registry the
        (registry-qualified) ref encodes, resolving the pushed digest.
        Registry-HEAD skip is implied node-side, so a re-run is cheap and
        overlapping dispatch never double-pushes (build-once fleet-wide).

        Returns ``(status, error, repo_digest)`` — ``repo_digest`` is the
        pushed ``<repo>@sha256:...`` on success, ``None`` otherwise.
        """
        body = await self._dispatch_build_image(
            image_ref=image_ref, source=source, timeout_s=timeout_s,
            labels=labels, skip_if_present=False, push=True,
        )
        return (str(body.status), str(body.error), str(body.repo_digest) or None)

    async def _dispatch_build_image(
        self,
        *,
        image_ref: str,
        source: Any,
        timeout_s: float,
        labels: dict[str, str] | None,
        skip_if_present: bool,
        push: bool,
    ) -> Any:
        """Lower a build-plan source into a ``BuildImageCommand`` (carrying
        ``push``), send it, and return the node's ``BuildImageReply``. Shared
        by :meth:`build_image` and :meth:`build_and_push_image`."""
        from xrlenv.control.build_plan import GitSource, TarballSource

        header = self._fresh_header(
            idempotency_key=f"build_image:{uuid.uuid4().hex}",
        )
        cmd = pb.BuildImageCommand(
            header=header, image_ref=image_ref,
            timeout_s=float(timeout_s),
            labels=dict(labels or {}),
            skip_if_present=bool(skip_if_present),
            push=bool(push),
        )
        if isinstance(source, GitSource):
            cmd.git.repo = source.repo
            cmd.git.ref = source.ref
            cmd.git.subdir = source.subdir
            cmd.git.dockerfile = source.dockerfile
        elif isinstance(source, TarballSource):
            # Sub-slice 1.b: the operator's CLI populated
            # ``content_b64`` via ``resolve_tarball_sources`` before
            # the plan reached the coordinator. Decode + ship the
            # raw bytes on the wire.
            if source.content_b64 is None:
                raise TypeError(
                    f"build_image: tarball source for image_ref "
                    f"{image_ref!r} has no content_b64; the CLI's "
                    "resolve_tarball_sources helper must run before "
                    "dispatch",
                )
            import base64
            cmd.tarball.content = base64.b64decode(source.content_b64)
            cmd.tarball.dockerfile = source.dockerfile
        else:
            raise TypeError(
                f"build_image: unsupported source type "
                f"{type(source).__name__}",
            )
        msg = self._control_msg(build_image=cmd)
        cp_timeout = (timeout_s + 60.0) if timeout_s > 0 else None
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=cp_timeout,
        )
        return reply.build_image

    async def register_scratch_source(
        self,
        image_ref: str,
        source: Any,   # GitSource | TarballSource (build_plan types)
        *,
        durable_to: str | None = None,
    ) -> None:
        """Ship a spec-21 ``RegisterScratchSourceCommand`` so the node records
        the content-addressed scratch ref → build source (scratch build-on-
        demand). No build happens here — ``ensure_present`` builds + pushes to
        the scratch registry lazily. Raises :class:`XRLEnvError` if the node
        reports a failure. Called by the coordinator for scratch_build
        rollouts on the distributed transport."""
        from xrlenv.control.build_plan import GitSource, TarballSource

        header = self._fresh_header(
            idempotency_key=f"register_scratch_source:{uuid.uuid4().hex}",
        )
        cmd = pb.RegisterScratchSourceCommand(
            header=header, image_ref=image_ref, durable_to=durable_to or "",
        )
        if isinstance(source, GitSource):
            cmd.git.repo = source.repo
            cmd.git.ref = source.ref
            cmd.git.subdir = source.subdir
            cmd.git.dockerfile = source.dockerfile
        elif isinstance(source, TarballSource):
            if source.content_b64 is None:
                raise TypeError(
                    f"register_scratch_source: tarball source for image_ref "
                    f"{image_ref!r} has no content_b64",
                )
            import base64
            cmd.tarball.content = base64.b64decode(source.content_b64)
            cmd.tarball.dockerfile = source.dockerfile
        else:
            raise TypeError(
                f"register_scratch_source: unsupported source type "
                f"{type(source).__name__}",
            )
        msg = self._control_msg(register_scratch_source=cmd)
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=60.0,
        )
        body = reply.register_scratch_source
        if body.status != "ok":
            raise XRLEnvError(
                f"node failed to register scratch source for {image_ref!r}: "
                f"{body.error or 'unknown'}",
            )

    async def cancel_build_image(
        self, *, image_ref: str, timeout_s: float = 30.0,
    ) -> tuple[str, str]:
        """Operator-driven mid-build cancel (per-image-ref).

        Mirrors :meth:`build_image` — sends a :class:`CancelBuildImageCommand`
        to the node and waits for its reply. The node best-effort kills any
        running ``docker build`` container labeled ``xrlenv.cancel-key=
        <image_ref>`` and cancels the registered ``GitSourceBuilder`` task;
        the original ``BuildImageCommand`` command_id then completes
        asynchronously with a ``failed: cancelled by operator`` reply.

        ``timeout_s`` is the control-plane wait ceiling (default 30 s). Like
        :meth:`ensure_present` / :meth:`build_image`, a no-reply hit takes
        ``_send_and_wait``'s timeout branch — popping the pending entry and
        flagging the node's command-timeout health marker — instead of
        relying on an outer ``asyncio.wait_for`` cancelling the coroutine
        and leaking pending state.

        Returns ``(status, error)`` where status ∈ {"ok", "failed"}.
        ``ok`` includes the no-op case (no in-flight build for this
        image_ref on this node) — cancel is operator-idempotent.
        """
        header = self._fresh_header(
            idempotency_key=f"cancel_build_image:{uuid.uuid4().hex}",
        )
        cmd = pb.CancelBuildImageCommand(
            header=header, image_ref=image_ref,
        )
        msg = self._control_msg(cancel_build_image=cmd)
        cp_timeout = timeout_s if timeout_s > 0 else None
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=cp_timeout,
        )
        body = reply.cancel_build_image
        return (str(body.status), str(body.error))

    async def evict_image(
        self, *, image_ref: str, force: bool = False,
        timeout_s: float = 30.0,
    ) -> EvictOutcome:
        """Operator-driven node-cache eviction (``xrlenv images evict``).

        Mirrors :meth:`cancel_build_image` — sends an
        :class:`EvictImageCommand` and waits for the reply. Per-call
        UUID idempotency key: eviction mutates cache state, so a stale
        replayed reply would mislead the operator about what was
        actually removed.
        """
        from xrlenv.node.image_cache import EvictOutcome

        header = self._fresh_header(
            idempotency_key=f"evict_image:{uuid.uuid4().hex}",
        )
        cmd = pb.EvictImageCommand(
            header=header, image_ref=image_ref, force=force,
        )
        msg = self._control_msg(evict_image=cmd)
        cp_timeout = timeout_s if timeout_s > 0 else None
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=cp_timeout,
        )
        body = reply.evict_image
        return EvictOutcome(
            status=body.status or "failed",  # type: ignore[arg-type]
            reclaimed_bytes=int(body.reclaimed_bytes),
            removed=tuple(body.removed),
            detail=str(body.error),
        )

    async def build_images(
        self,
        *,
        assignments: list[Any],   # PlanAssignment; Any avoids cycle
        builder_per_benchmark: dict[str, Any],  # BuilderRef
        kwargs_per_benchmark: dict[str, dict[str, str]],
        force: bool,
        lazy_registrations: list[Any] | None = None,  # PlanAssignment
    ) -> list[Any]:  # list[BuildResult]
        """P1.6.c — dispatch one node's slice of a build plan via the
        spec-21 :class:`BuildImagesCommand`.

        Per-call UUID in the idempotency key: re-applying a plan after
        a partial failure should re-execute on the node, not absorb a
        cached "still building" reply. Idempotency at the plan level
        is the coordinator's job (layers 1+2); at the wire level each
        ``apply()`` is a fresh request.

        ``lazy_registrations`` carries refs that should *register* a
        builder mapping on the node (so a later ``ensure_present`` can
        dispatch lazily) but NOT build synchronously here. Used for
        opportunistic-mode deferred rows whose preferred_home is this
        node (audit P1.6.g-H1 fix, 2026-05-05).
        """
        from xrlenv.control.image_builder import BuildResult

        lazy_regs = lazy_registrations or []

        header = self._fresh_header(
            idempotency_key=f"build_images:{uuid.uuid4().hex}",
        )
        msg = self._control_msg(
            build_images=pb.BuildImagesCommand(
                header=header,
                assignments=[
                    pb.BuildAssignmentEntry(
                        image_ref=a.image_ref,
                        benchmark=a.benchmark,
                        size_bytes=int(a.size_bytes),
                    )
                    for a in assignments
                ],
                builder_per_benchmark={
                    name: pb.BuilderRefEntry(
                        module=ref.module, class_name=ref.class_name,
                    )
                    for name, ref in builder_per_benchmark.items()
                },
                kwargs_per_benchmark={
                    name: pb.BuildKwargsEntry(kv=kv)
                    for name, kv in kwargs_per_benchmark.items()
                },
                force=bool(force),
                lazy_registrations=[
                    pb.BuildAssignmentEntry(
                        image_ref=a.image_ref,
                        benchmark=a.benchmark,
                        size_bytes=int(a.size_bytes),
                    )
                    for a in lazy_regs
                ],
            ),
        )
        reply = await self._send_and_wait(msg, header.command_id)
        body = reply.build_images
        return [
            BuildResult(
                image_ref=str(r.image_ref),
                status=cast(Literal["done", "failed"], r.status),
                bytes_pulled=int(r.bytes_pulled),
                duration_s=float(r.duration_s),
                error=str(r.error) if r.error else None,
            )
            for r in body.results
        ]

    async def run_in_sandbox(
        self,
        sb: SandboxHandle,
        cmd: list[str],
        *,
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        # Per-call UUID in the idempotency key — re-running an arbitrary
        # command (init.cmd or otherwise) is a workload action the cache must
        # not silently absorb.
        header = self._fresh_header(idempotency_key=f"{sb.id}:run:{uuid.uuid4().hex}")
        msg = self._control_msg(
            run_in_sandbox=pb.RunInSandboxCommand(
                header=header,
                sandbox=conv.sandbox_handle_to_proto(sb),
                cmd=cmd,
                timeout_s=timeout_s,
                cwd=cwd or "",
                env=env or {},
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        return ExecResult(
            exit_code=reply.exec.exit_code,
            stdout=reply.exec.stdout,
            stderr=reply.exec.stderr,
            timed_out=reply.exec.timed_out,
        )

    async def put_archive(
        self,
        sb: SandboxHandle,
        target_dir: str,
        tarball: bytes,
        *,
        clean_target: bool = False,
    ) -> None:
        # D12 stage 1: per-call UUID in the idempotency key. Replaying a
        # put_archive could mask agent residue (a no-op on the second
        # call would leave the agent's mkdir+chmod in place); the cache
        # must therefore treat each call as workload, not infrastructure.
        header = self._fresh_header(
            idempotency_key=f"{sb.id}:put_archive:{uuid.uuid4().hex}"
        )
        msg = self._control_msg(
            put_archive=pb.PutArchiveCommand(
                header=header,
                sandbox=conv.sandbox_handle_to_proto(sb),
                target_dir=target_dir,
                tarball=tarball,
                clean_target=clean_target,
            )
        )
        await self._send_and_wait(msg, header.command_id)
        # PutArchiveReply is empty; success is OK status on the wrapping
        # CommandReply (raised inside _send_and_wait on FAILED).
        return None

    # ── P1.7.A.1 — raw container session ────────────────────────────────────

    async def acquire_container(
        self,
        *,
        rollout_id: str,
        backend: str,
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
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        resources: ResourceSpec | None = None,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> Any:
        # Per-call UUID — re-running an acquire is a workload action
        # (would create a duplicate container) the cache must not
        # silently absorb.
        header = self._fresh_header(
            idempotency_key=f"{rollout_id}:acquire_container:{uuid.uuid4().hex}",
        )
        cmd = pb.AcquireContainerCommand(
            header=header,
            rollout_id=rollout_id,
            image=image,
            # Negative-form on the wire so proto3 default-false
            # aligns with the new default behaviour (ensure runs).
            strict_image_check=not ensure_image_present,
            # B5.4 — proto3 default-empty maps to "host" on the
            # node; pass through for "remap" (or any future value).
            userns_mode=userns_mode if userns_mode != "host" else "",
        )
        if command:
            cmd.command.extend(command)
        if entrypoint is not None:
            # See rollout_endpoint.AcquireContainer for the
            # ``[""]`` vs ``None`` distinction.
            cmd.entrypoint.extend(entrypoint)
        if user:
            cmd.user = user
        if cap_add:
            cmd.cap_add.extend(cap_add)
        if devices:
            cmd.devices.extend(devices)
        if privileged:
            cmd.privileged = True
        if network_mode:
            cmd.network_mode = network_mode
        if binds:
            cmd.binds.extend(binds)
        if name:
            cmd.name = name
        if labels:
            for k, v in labels.items():
                cmd.labels[k] = v
        if environment:
            for k, v in environment.items():
                cmd.environment[k] = v
        # Issue #12 + audit P2 — stamp the *effective* acquire wire budget onto
        # ``pull_deadline_s`` so the node bounds BOTH its image pull/build AND its
        # create retry-with-backoff by the same deadline the control plane is
        # actually waiting on below (``_send_and_wait``). This is the caller's
        # explicit ``acquire_timeout_s`` when given, else ``DEFAULT_ACQUIRE_TIMEOUT_S``
        # (600 s) — the same value the wire wait uses. Previously the field was set
        # only for an explicit override, so a *default* acquire reached the node as
        # ``pull_deadline_s == 0`` → no node-side deadline, and the node's create
        # retry could keep going after the control plane had already timed the
        # command out. ``DEFAULT_ACQUIRE_TIMEOUT_S`` matches the node's own default
        # pull timeout (``ImageCacheConfig.default_pull_timeout_s``), so stamping it
        # is behaviour-preserving for the pull and only *adds* the retry-deadline.
        # (Audit M1, `86afca1`: a consumer passing ``acquire_timeout_s=1800`` must
        # still widen the node pull past its 600 s default — the explicit branch
        # of ``effective_timeout_s`` carries that through unchanged.)
        effective_timeout_s = (
            acquire_timeout_s
            if acquire_timeout_s is not None
            else DEFAULT_ACQUIRE_TIMEOUT_S
        )
        cmd.pull_deadline_s = float(effective_timeout_s)
        # P1 — stamp the effective ResourceSpec so the node can apply
        # cpu/memory cgroup limits. Field 6 has existed since P1.7.A.1
        # (DEFERRED until now); the node reads it in _exec_acquire_container.
        if resources is not None:
            cmd.resources.CopyFrom(conv.resource_spec_to_proto(resources))
        # P0b — stamp the container-shape RuntimeLimits so the node can
        # apply pids / shm / tmpfs / read-only at container creation.
        if runtime_limits is not None and not runtime_limits.is_empty():
            cmd.runtime_limits.CopyFrom(
                conv.runtime_limits_to_proto(runtime_limits),
            )
        # §5.1 — OCI runtime selector. Empty string = proto3 unset =
        # docker default runtime; the node reads it back as None.
        if container_runtime:
            cmd.container_runtime = container_runtime
        msg = self._control_msg(acquire_container=cmd)
        # Issue #12 — wire timeout for AcquireContainer must cover the
        # node-side ``ensure_image_present`` cold-pull path. The earlier
        # 60 s ceiling assumed "docker spawn + image-presence probe
        # finish in <2 s on a healthy node" — true for warm images, but
        # SWE-bench Pro tags routinely run 5-15 GB and don't fit in
        # 60 s on GCP↔docker.io links. ``DEFAULT_ACQUIRE_TIMEOUT_S``
        # (600 s) matches :py:attr:`ImageCacheConfig.default_pull_timeout_s`
        # so a legitimate cold-pull surfaces the node's "pull failed"
        # error rather than racing it with a "command timed out". A
        # wedged node still surfaces as ``XRLEnvError(…timed out…)``
        # at the deadline. Consumers with known-huger images override
        # per-call via ``acquire_timeout_s``. ``effective_timeout_s`` was
        # computed above (also stamped onto ``pull_deadline_s``).
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=effective_timeout_s,
        )
        return _RemoteRawContainerRecord(
            rollout_id=rollout_id,
            container_id=reply.acquire_container.container_id,
            container_name=reply.acquire_container.container_name,
            image=image,
        )

    async def acquire_compose_project(
        self,
        *,
        rollout_id: str,
        project_name: str,
        compose_yaml: str,
        images: list[str] | None = None,
        main_service: str = "main",
        up_timeout_s: float | None = None,
        backend: str = "docker",
    ) -> Any:
        """P1.7.C.2 — bring up a multi-service compose project on this node.

        The wire wait must cover the node's image ensure-present pulls **plus**
        ``docker compose up --wait`` (healthcheck settling), so it's sized off the
        ``up_timeout_s`` budget with headroom, floored at the acquire default."""
        del backend  # docker-only in phase 1 (node picks its docker manager)
        header = self._fresh_header(
            idempotency_key=f"{rollout_id}:acquire_compose_project:{uuid.uuid4().hex}",
        )
        cmd = pb.AcquireComposeProjectCommand(
            header=header,
            rollout_id=rollout_id,
            project_name=project_name,
            compose_yaml=compose_yaml,
            main_service=main_service or "main",
            up_timeout_s=float(up_timeout_s or 0.0),
        )
        if images:
            cmd.images.extend(images)
        msg = self._control_msg(acquire_compose_project=cmd)
        wire_timeout_s = max(
            float(up_timeout_s or 0.0), DEFAULT_ACQUIRE_TIMEOUT_S,
        ) + 120.0
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=wire_timeout_s,
        )
        r = reply.acquire_compose_project
        return _RemoteComposeProjectRecord(
            project_name=r.project_name,
            project_dir=r.project_dir,
            main_container_id=r.main_container_id,
            main_container_name=r.main_container_name,
            service_container_ids=dict(r.service_container_ids),
        )

    async def destroy_compose_project(
        self,
        *,
        rollout_id: str,
        project_name: str,
        force: bool = True,
        backend: str = "docker",
    ) -> None:
        """P1.7.C.2 — ``docker compose down`` the whole project on this node."""
        del backend
        header = self._fresh_header(
            idempotency_key=f"{rollout_id}:destroy_compose_project:{uuid.uuid4().hex}",
        )
        cmd = pb.DestroyComposeProjectCommand(
            header=header,
            rollout_id=rollout_id,
            project_name=project_name,
            force=force,
        )
        msg = self._control_msg(destroy_compose_project=cmd)
        # Same 300 s ceiling as destroy_container — a compose down of N containers
        # + network can be slower than a single rm under a busy daemon.
        await self._send_and_wait(
            msg, header.command_id, timeout_s=300.0,
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
        backend: str = "docker",
    ) -> dict[str, Any]:
        # Per-call UUID — exec is a workload action.
        header = self._fresh_header(
            idempotency_key=(
                f"{rollout_id}:container_exec:{container_id[:12]}:{uuid.uuid4().hex}"
            ),
        )
        proto_cmd = pb.ContainerExecCommand(
            header=header,
            rollout_id=rollout_id,
            container_id=container_id,
            cmd=cmd,
            timeout_s=timeout_s,
            cwd=cwd or "",
            env=env or {},
            user=user or "",
        )
        msg = self._control_msg(container_exec=proto_cmd)
        # Per-call ceiling = caller's exec timeout + 30s buffer for
        # the wire round-trip + node-side dispatch overhead.
        # Without the buffer a tight inner ``timeout_s`` would
        # race the wire-level wait and surface as a node-side
        # timeout one round-trip too early.
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=timeout_s + 30.0,
        )
        return {
            "exit_code": int(reply.exec.exit_code),
            "stdout": bytes(reply.exec.stdout),
            "stderr": bytes(reply.exec.stderr),
            "timed_out": bool(reply.exec.timed_out),
        }

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
        backend: str = "docker",
    ) -> None:
        header = self._fresh_header(
            idempotency_key=(
                f"{rollout_id}:apply_egress:{container_id[:12]}:{uuid.uuid4().hex}"
            ),
        )
        proto_cmd = pb.ApplyEgressCommand(
            header=header,
            rollout_id=rollout_id,
            container_id=container_id,
            allow=[
                pb.EgressAllowEntry(cidr=r.cidr, ports=list(r.ports or ()))
                for r in allowlist.rules
            ],
            dns_resolver=dns_resolver or "",
        )
        msg = self._control_msg(apply_egress=proto_cmd)
        await self._send_and_wait(msg, header.command_id, timeout_s=60.0)

    async def destroy_container(
        self,
        *,
        rollout_id: str,
        container_id: str,
        force: bool = True,
        backend: str = "docker",
    ) -> None:
        # Per-call UUID — destroy is workload (would no-op on the
        # second call, masking the harness's intent to retry).
        header = self._fresh_header(
            idempotency_key=(
                f"{rollout_id}:destroy_container:{container_id[:12]}:{uuid.uuid4().hex}"
            ),
        )
        proto_cmd = pb.DestroyContainerCommand(
            header=header,
            rollout_id=rollout_id,
            container_id=container_id,
            force=force,
        )
        msg = self._control_msg(destroy_container=proto_cmd)
        # Issue #18 fix #2: 300 s ceiling. The original 30 s was set on
        # the assumption that ``docker rm -f`` finishes in <1 s — true
        # for an idle daemon, but under heavy parallel teardowns
        # (SWE-bench Pro at ``--num-workers=64``) the docker daemon
        # serialised overlay-fs work and individual removes stretched
        # to 35-90 s. The coordinator's 30 s wire timeout would fire
        # while the node was still in ``container.remove``, popping
        # ``_sessions[rollout_id]`` and silently freeing capacity
        # while the disk usage stayed real. Combined with fix #4
        # (per-node ``destroy_concurrency=4``), 300 s is comfortably
        # above the worst-case capped teardown latency. Idempotent
        # on the node side regardless. A still-wedged daemon
        # surfaces as ``XRLEnvError(timed out)`` at 300 s and the
        # reconciler picks up node-side leftovers.
        await self._send_and_wait(
            msg, header.command_id, timeout_s=300.0,
        )
        # DestroyReply is empty; success is OK status on the wrapping
        # CommandReply (raised inside _send_and_wait on FAILED).
        return None

    # ── P1.7.A.2 — raw container archives ──────────────────────────────────

    async def container_put_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        target_dir: str,
        tarball: bytes,
        backend: str = "docker",
    ) -> None:
        # Per-call UUID idempotency key — put_archive is workload
        # (replay would extract the bytes a second time, masking
        # any agent-created residue between calls).
        header = self._fresh_header(
            idempotency_key=(
                f"{rollout_id}:container_put_archive:"
                f"{container_id[:12]}:{uuid.uuid4().hex}"
            ),
        )
        # 300s ceiling — same tier as ``destroy_container`` (issue #18): a busy-but-live
        # node servicing multi-GB cold pulls can't always answer inside 60s, and a
        # genuinely wedged node still surfaces a clean XRLEnvError at the deadline.
        if not self._chunked_put_archive_capable:
            # Old node (pre-WS1) — single unary command, capped at the 128 MiB gRPC
            # message limit. Kept for back-compat with a not-yet-redeployed node.
            msg = self._control_msg(
                container_put_archive=pb.ContainerPutArchiveCommand(
                    header=header,
                    rollout_id=rollout_id,
                    container_id=container_id,
                    target_dir=target_dir,
                    tarball=tarball,
                ),
            )
            await self._send_and_wait(msg, header.command_id, timeout_s=300.0)
            return
        # WS1 — slice the tarball into ~4 MiB ContainerPutArchiveChunk ControlMsg frames
        # all sharing ``header.command_id``. The FIRST carries the routing metadata; the
        # LAST sets ``done``. Each frame is well under the wire limit and small enough to
        # interleave with the heartbeat on the shared bidi stream. All frames register a
        # single Future (via ``extra_frames`` + the terminal ``msg``); the node reassembles
        # them and replies once. Removes the 128 MiB upload ceiling (e.g. a 340 MB SDK).
        frames: list[pb.ControlMsg] = []
        n = len(tarball)
        first = True
        for start in range(0, n, ARCHIVE_CHUNK_BYTES):
            chunk = pb.ContainerPutArchiveChunk(
                header=header,
                data=tarball[start:start + ARCHIVE_CHUNK_BYTES],
                done=start + ARCHIVE_CHUNK_BYTES >= n,
            )
            if first:
                chunk.rollout_id = rollout_id
                chunk.container_id = container_id
                chunk.target_dir = target_dir
                first = False
            frames.append(self._control_msg(container_put_archive_chunk=chunk))
        if first:
            # empty tarball — one terminator frame carrying the metadata.
            frames.append(self._control_msg(
                container_put_archive_chunk=pb.ContainerPutArchiveChunk(
                    header=header, rollout_id=rollout_id,
                    container_id=container_id, target_dir=target_dir, done=True,
                ),
            ))
        await self._send_and_wait(
            frames[-1], header.command_id, timeout_s=300.0,
            extra_frames=frames[:-1],
        )

    async def list_raw_container_ids(
        self, *, backend: str = "docker",
    ) -> list[str]:
        # Reconciler-driven; no per-rollout idempotency anchor —
        # use a UUID so each sweep gets a fresh request.
        header = self._fresh_header(
            idempotency_key=f"list_raw_containers:{uuid.uuid4().hex}",
        )
        msg = self._control_msg(
            list_raw_containers=pb.ListRawContainersCommand(
                header=header, backend=backend,
            ),
        )
        # 90s backstop ceiling. A label-filtered ``docker ps`` is a
        # node-local op, but a docker daemon swamped by concurrent
        # multi-GB cold pulls answers its API slowly (issue #18). The
        # GC reconciler — the primary caller — wraps this in a shorter
        # per-node ``wait_for`` and owns the effective timeout; this
        # ceiling is only the fallback for any caller without one.
        reply = await self._send_and_wait(
            msg, header.command_id, timeout_s=90.0,
        )
        # Audit P3 — stash the node's autonomous-reap reasons (disk guard)
        # so the reconciler can seal coordinator-only orphans with the
        # real cause. Kept as a side-channel to preserve this method's
        # ``list[str]`` return (callers/tests unchanged).
        self._last_reaped_reasons = dict(
            reply.list_raw_containers.reaped_reasons,
        )
        # P1.7.C.2 — same side-channel for the per-container correlation labels
        # (container_id -> (rollout_id, compose_project)), read by the raw-GC
        # reconciler to recognise/route compose-project mains. Empty on older
        # nodes (the field is absent → no entries).
        self._last_container_info = {
            c.container_id: (c.rollout_id, c.compose_project)
            for c in reply.list_raw_containers.containers
        }
        return list(reply.list_raw_containers.container_ids)

    async def list_managed_container_info(
        self, *, backend: str = "docker",
    ) -> list[tuple[str, str, str, str]]:
        """Audit H11 — EVERY xrlenv-managed container on this node WITH labels:
        ``(container_id, rollout_id, compose_project, session_kind)`` — including compose
        SIDECARS (``session_kind=compose``) the raw-only :meth:`list_raw_container_ids`
        omits. Sends ``ListRawContainersCommand(include_all_managed=True)``. readopt-on-connect
        uses it to quarantine a node with a sidecar-only compose survivor.

        CAPABILITY-CHECKED (audit H11): an OLDER agent silently ignores ``include_all_managed``
        and returns a raw-only ``containers`` list — which would look like a "clean" node while
        hiding sidecar-only survivors. The node ACKs support via ``all_managed_supported``; if it
        is not set, RAISE :class:`ManagedInventoryUnsupported` so readopt-on-connect fails CLOSED
        (the node stays unschedulable until it is upgraded) rather than trusting an incomplete
        inventory."""
        header = self._fresh_header(
            idempotency_key=f"list_managed_containers:{uuid.uuid4().hex}",
        )
        msg = self._control_msg(
            list_raw_containers=pb.ListRawContainersCommand(
                header=header, backend=backend, include_all_managed=True,
            ),
        )
        reply = await self._send_and_wait(msg, header.command_id, timeout_s=90.0)
        if not reply.list_raw_containers.all_managed_supported:
            raise ManagedInventoryUnsupported(
                "node agent did not ACK the broad managed-container inventory "
                "(include_all_managed) — an older agent; cannot rule out sidecar-only compose "
                "survivors. Upgrade the node agent to the control-plane version.",
            )
        return [
            (c.container_id, c.rollout_id, c.compose_project, c.session_kind)
            for c in reply.list_raw_containers.containers
        ]

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
        backend: str = "docker",
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming exec — async iterator yielding per-chunk
        dicts as the node produces them. Each yielded dict has
        the ``ContainerExecChunk`` shape; the terminator chunk
        has ``done=True``."""
        # Per-call UUID — workload, not infra. Streaming is not
        # cached (the dispatcher skips the idempotency cache for
        # streaming commands; replay would mean replaying every
        # chunk, which the cache isn't designed for).
        header = self._fresh_header(
            idempotency_key=(
                f"{rollout_id}:stream_container_exec:"
                f"{container_id[:12]}:{uuid.uuid4().hex}"
            ),
        )
        proto_cmd = pb.StreamContainerExecCommand(
            header=header,
            rollout_id=rollout_id,
            container_id=container_id,
            cmd=cmd,
            timeout_s=timeout_s,
            cwd=cwd or "",
            env=env or {},
            user=user or "",
        )
        msg = self._control_msg(stream_container_exec=proto_cmd)
        async for chunk in self._send_and_stream(
            msg, header.command_id,
            # Per-chunk (output-idle) wait ceiling. timeout_s is the WHOLE-exec
            # bound on the node side; per-chunk we just need a generous keepalive
            # so a chunk-less stretch on a quiet exec doesn't unblock prematurely.
            # Defaults to an hour (XRLENV_EXEC_CHUNK_TIMEOUT_S) so it defers to the
            # per-exec timeout_s — a silent test/compile phase in a benchmark
            # verifier is no longer aborted at a tight idle window + spuriously
            # retried. A dead node is caught by the heartbeat / disconnect path.
            chunk_timeout_s=_resolve_exec_chunk_timeout_s(),
            # Hard ceiling on the entire stream — timeout_s plus a
            # buffer for the wire round-trip + node-side dispatch
            # overhead. Mirrors the batched-exec ``timeout_s + 30``
            # pattern from container_exec.
            stream_timeout_s=timeout_s + 30.0,
        ):
            yield chunk

    async def _send_and_stream(
        self,
        msg: pb.ControlMsg,
        command_id: str,
        *,
        chunk_timeout_s: float,
        stream_timeout_s: float,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming command and yield chunks as they
        arrive. Closes the stream on terminator (chunk.done=True)
        OR on a non-chunk FAILED reply with the same command_id.

        ``chunk_timeout_s`` bounds the wait between chunks (not
        the whole stream — that's ``stream_timeout_s``). A quiet
        exec that goes more than ``chunk_timeout_s`` without
        producing output is a problem we surface as a clean
        XRLEnvError; an entire-stream timeout (consumer-side
        cap on the whole exec) is also surfaced via the same
        XRLEnvError shape.
        """
        if self._closed:
            raise XRLEnvError(f"node {self.node_id} is disconnected")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[pb.CommandReply] = asyncio.Queue()
        self._pending_streams[command_id] = queue
        await self._outbox.put(msg)
        stream_deadline = loop.time() + stream_timeout_s
        try:
            while True:
                remaining_stream = stream_deadline - loop.time()
                if remaining_stream <= 0:
                    raise XRLEnvError(
                        f"node {self.node_id}: stream {command_id} "
                        f"exceeded total stream timeout "
                        f"{stream_timeout_s:.1f}s",
                    )
                wait_s = min(chunk_timeout_s, remaining_stream)
                try:
                    reply = await asyncio.wait_for(
                        queue.get(), timeout=wait_s,
                    )
                except TimeoutError as exc:
                    raise XRLEnvError(
                        f"node {self.node_id}: stream {command_id} produced no "
                        f"output for {wait_s:.1f}s (exec-stream output-idle limit; "
                        f"the effective wait is min(XRLENV_EXEC_CHUNK_TIMEOUT_S="
                        f"{chunk_timeout_s:.0f}s, remaining exec budget)). The "
                        f"command emitted no stdout/stderr for that long — usually "
                        f"a silent/slow exec phase (raise the per-exec timeout or "
                        f"XRLENV_EXEC_CHUNK_TIMEOUT_S), occasionally a wedged node "
                        f"or an older node binary that doesn't recognise this "
                        f"command kind",
                    ) from exc
                if reply.status == pb.ReplyStatus.FAILED:
                    raise XRLEnvError(
                        f"remote stream {reply.error_kind}: "
                        f"{reply.error_message}",
                    )
                chunk = reply.container_exec_chunk
                yield {
                    "stdout": bytes(chunk.stdout),
                    "stderr": bytes(chunk.stderr),
                    "done": bool(chunk.done),
                    "exit_code": int(chunk.exit_code),
                    "timed_out": bool(chunk.timed_out),
                }
                if chunk.done:
                    return
        finally:
            self._pending_streams.pop(command_id, None)

    async def _send_and_collect_archive(
        self,
        msg: pb.ControlMsg,
        command_id: str,
        *,
        stream_timeout_s: float,
    ) -> bytes:
        """Send a ``container_get_archive`` command and reassemble the
        chunked reply into the full tarball.

        The node ships the tarball as a sequence of
        ``ContainerGetArchiveChunk`` replies (sharing ``command_id``)
        terminated by ``done=true`` — so no single NodeMsg can exceed
        the wire ceiling and sever the heartbeat stream the node-lost
        incidents traced to. Backward-compatible with an older node
        that still answers with a single ``ContainerGetArchiveReply``:
        that whole-tarball reply is returned as-is.

        ``stream_timeout_s`` bounds the whole collection. The node tars
        the path before emitting any chunk, so time-to-first-chunk can
        be the full archive-build time; a single overall deadline
        (rather than a short per-chunk keepalive) accommodates that. On
        timeout the node's degraded state is recorded for the scheduler
        placement gate, mirroring ``_send_and_wait``.
        """
        if self._closed:
            raise XRLEnvError(f"node {self.node_id} is disconnected")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[pb.CommandReply] = asyncio.Queue()
        self._pending_streams[command_id] = queue
        await self._outbox.put(msg)
        deadline = loop.time() + stream_timeout_s
        buf = bytearray()
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    self._last_command_timeout_monotonic = time.monotonic()
                    self._command_timeout_total += 1
                    raise NodeCommandTimeout(
                        f"node {self.node_id}: container_get_archive "
                        f"{command_id} timed out after "
                        f"{stream_timeout_s:.1f}s",
                    )
                try:
                    reply = await asyncio.wait_for(
                        queue.get(), timeout=remaining,
                    )
                except TimeoutError as exc:
                    self._last_command_timeout_monotonic = time.monotonic()
                    self._command_timeout_total += 1
                    raise NodeCommandTimeout(
                        f"node {self.node_id}: container_get_archive "
                        f"{command_id} timed out after "
                        f"{stream_timeout_s:.1f}s",
                    ) from exc
                if reply.status == pb.ReplyStatus.FAILED:
                    raise XRLEnvError(
                        f"node {self.node_id}: remote get_archive "
                        f"{reply.error_kind}: {reply.error_message}",
                    )
                if reply.WhichOneof("payload") == "container_get_archive":
                    # Backward-compat: an older node sent the whole
                    # tarball in one (unchunked) reply.
                    return bytes(reply.container_get_archive.tarball)
                chunk = reply.container_get_archive_chunk
                buf += chunk.data
                if chunk.done:
                    return bytes(buf)
        finally:
            self._pending_streams.pop(command_id, None)

    async def force_destroy_raw_container(
        self, *, container_id: str,
    ) -> None:
        header = self._fresh_header(
            idempotency_key=(
                f"force_destroy_container:{container_id[:12]}:"
                f"{uuid.uuid4().hex}"
            ),
        )
        msg = self._control_msg(
            force_destroy_container=pb.ForceDestroyContainerCommand(
                header=header, container_id=container_id,
            ),
        )
        # Issue #18 fix #2 (audit M2): 300s ceiling — kept in lock-step
        # with the regular ``destroy_container`` path. ``force_destroy``
        # is a ``docker rm -f`` against the same daemon and hits the
        # same 35-90s overlay-fs teardown latency under concurrent
        # load. The raw-GC reconciler drives this path for node-only
        # orphans after a control-plane restart — exactly when the
        # cluster is most likely mid-recovery and teardowns are slow —
        # so the old 30s ceiling would time out the cleanup it most
        # needs to complete.
        await self._send_and_wait(
            msg, header.command_id, timeout_s=300.0,
        )

    async def container_get_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
        backend: str = "docker",
    ) -> bytes:
        header = self._fresh_header(
            idempotency_key=(
                f"{rollout_id}:container_get_archive:"
                f"{container_id[:12]}:{uuid.uuid4().hex}"
            ),
        )
        msg = self._control_msg(
            container_get_archive=pb.ContainerGetArchiveCommand(
                header=header,
                rollout_id=rollout_id,
                container_id=container_id,
                source_path=source_path,
            ),
        )
        # node-lost root-cause fix: the node ships the tarball as a
        # sequence of ContainerGetArchiveChunk replies rather than one
        # giant ContainerGetArchiveReply, so a large verifier dir can
        # never produce a single oversized NodeMsg that severs the
        # heartbeat stream. We reassemble here. 300s ceiling kept in
        # lock-step with ``container_put_archive`` (cold-pull-saturation
        # rationale); the node tars before emitting any chunk, so this
        # bounds the whole tar-build + transfer.
        return await self._send_and_collect_archive(
            msg, header.command_id, stream_timeout_s=300.0,
        )

    async def fetch_trajectory(
        self,
        rollout_id: str,
        *,
        range_kind: FetchRangeKind = "whole",
        step_start: int = 0,
        step_end: int | None = None,
    ) -> Trajectory:
        # Sealed trajectories are immutable (spec 00 invariant 3); safe to
        # share the idempotency key across cache lookups for the same
        # range. Per-range key keeps SUMMARY + WHOLE in distinct slots.
        idem = (
            f"{rollout_id}:fetch:{range_kind}:{step_start}:"
            f"{step_end if step_end is not None else 'eof'}"
        )
        header = self._fresh_header(idempotency_key=idem)
        msg = self._control_msg(
            fetch_trajectory=pb.FetchTrajectoryCommand(
                header=header,
                rollout_id=rollout_id,
                range=_range_kind_to_proto(range_kind),
                step_start=int(step_start),
                step_end=int(step_end) if step_end is not None else 0,
                include_binary=False,  # phase-1
            )
        )
        reply = await self._send_and_wait(msg, header.command_id)
        body = reply.trajectory.body_json or "{}"
        return Trajectory.model_validate_json(body)

    # ── Reader-side hooks ────────────────────────────────────────────────────

    def deliver_reply(self, reply: pb.CommandReply) -> None:
        """Called by the servicer's reader loop on every ``CommandReply``.

        Routes single-reply commands to ``_pending`` (Future) and
        streaming commands to ``_pending_streams`` (Queue). The
        async iterator side decides when to remove the queue
        entry (on terminator OR FAILED reply); this method only
        posts.
        """
        # Streaming first — a streaming command_id stays in
        # ``_pending_streams`` until the iterator sees the
        # terminator. Multiple replies can arrive for the same
        # command_id, so we don't pop here.
        stream_q = self._pending_streams.get(reply.command_id)
        if stream_q is not None:
            stream_q.put_nowait(reply)
            return

        future = self._pending.pop(reply.command_id, None)
        if future is None:
            LOGGER.warning(
                "node=%s reply for unknown command_id=%s", self.node_id, reply.command_id
            )
            return
        if not future.done():
            future.set_result(reply)

    def touch(
        self,
        *,
        free_disk_bytes: int = 0,
        total_disk_bytes: int = 0,
        pinned_cpus_free: int = 0,
        pinned_cpus_total: int = 0,
        health: Any = None,
    ) -> None:
        """Called by the reader loop on every Heartbeat (NodeMsg-side).

        The NodeRegistry watchdog inspects ``last_heartbeat_at`` to decide
        when to mark the node dead. We use control-plane wall clock so a
        skewed node clock can't accidentally extend its lifetime. The
        ``on_heartbeat`` callback (set by the registry) mirrors the event
        into the state-store so out-of-process callers (``xrlenv nodes``,
        admin ``/nodes`` view) see a fresh ``last_seen_at`` rather than
        the frozen register-time stamp.

        Issue #14: ``free_disk_bytes`` / ``total_disk_bytes`` flow in from
        the heartbeat payload and feed the scheduler's disk-pressure gate
        + admin /capacity pressure indicator. ``0`` means "node didn't
        report" — kept as 0 so :py:meth:`disk_state` callers can detect
        the unknown case.
        """
        # Audit M1 fix: distinguish "sample failed on the node" (the
        # documented ``(0, 0)`` sentinel — keep last known) from
        # "disk really is full" (``free=0, total>0`` — the literal
        # 100 %-full state the gate must catch). The earlier
        # ``if free > 0`` gate kept a stale positive free reading
        # when the disk genuinely filled, hiding the failure.
        if total_disk_bytes > 0:
            self._last_free_disk_bytes = free_disk_bytes
            self._last_total_disk_bytes = total_disk_bytes
        # P6 — same sentinel discipline as disk: ``total > 0`` means the node
        # actually reported pinnable-CPU counts; ``(0, 0)`` (pre-field agent or
        # a node with no core ledger) keeps the last-known "unknown".
        if pinned_cpus_total > 0:
            self._last_pinned_cpus_free = pinned_cpus_free
            self._last_pinned_cpus_total = pinned_cpus_total
        # Stage-1: stash the per-node health stats (a ``NodeHealthStats``
        # proto) as a plain dict so the registry can mirror it to the
        # state store. Absent on a pre-Stage-1 node-agent → keep ``None``.
        if health is not None:
            self._last_health = {
                "window_s": int(health.window_s),
                "create_p50_ms": float(health.create_p50_ms),
                "create_p95_ms": float(health.create_p95_ms),
                "create_count": int(health.create_count),
                "docker_error_count": int(health.docker_error_count),
                "docker_timeout_count": int(health.docker_timeout_count),
                "create_inflight": int(health.create_inflight),
                "create_queued": int(health.create_queued),
            }
        self.mark_seen()

    @property
    def health_json(self) -> str | None:
        """Stage-1 — the last heartbeat's health stats as a JSON string
        for the state-store mirror, or ``None`` if none reported yet."""
        if self._last_health is None:
            return None
        return json.dumps(self._last_health)

    def mark_seen(self) -> None:
        """Refresh the liveness clock the :class:`NodeRegistry` watchdog reads.

        Issue #18: the reader loop calls this on *every* inbound
        ``NodeMsg`` — reply and ack, not only heartbeat. A node busy
        enough to starve its heartbeat task still streams command
        replies, and any inbound message proves the bidi stream (and
        thus the node) is alive. Keying liveness on heartbeats alone
        false-flagged a working node ``lost``: the registry + state
        store recorded ``lost`` while the scheduler kept routing
        acquires to it, because the two node-loss paths disagreed.
        """
        self._last_heartbeat_at = time.monotonic()
        if self._on_heartbeat is not None:
            try:
                self._on_heartbeat(self.node_id)
            except Exception:
                LOGGER.exception(
                    "on_heartbeat callback raised for node=%s", self.node_id,
                )

    def set_on_heartbeat(self, cb: Callable[[str], None] | None) -> None:
        """Install / clear the per-heartbeat callback. Wired by
        :class:`NodeRegistry.register` to ``state.update_node_seen``.
        """
        self._on_heartbeat = cb

    def request_terminate(self, reason: str) -> None:
        """Ask the servicer to END this node's control stream so the node
        observes the RPC completing and REDIALS (fresh NodeHello →
        re-register). Called from the heartbeat watchdog's loss path:
        marking a node lost only removes it from the in-memory registry,
        which does not break the live bidi stream, so without this the
        node can never recover from a transient/false loss. Idempotent —
        the second call is a no-op."""
        if self._terminate.is_set():
            return
        self._terminate_reason = reason
        self._terminate.set()

    @property
    def terminate_event(self) -> asyncio.Event:
        """The event the servicer main loop waits on to close the stream
        (set by :meth:`request_terminate`)."""
        return self._terminate

    @property
    def terminate_reason(self) -> str:
        return self._terminate_reason

    def send_keepalive(self) -> None:
        """Enqueue an empty-body ``ControlMsg`` so an otherwise-idle control
        plane still proves liveness to the node. Defense-in-depth for
        ``request_terminate``: a node that stops receiving keepalives — because
        its transport was deregistered, or this stream went half-open — redials
        on its own even if the active stream-abort never reached it. Empty body
        (no command) → the node treats it purely as a liveness beat: it does not
        dispatch it or advance replay coordinates. Consuming a control ``seq`` is
        harmless (seq resets per epoch; there is no command replay buffer).
        Non-blocking + no-op once closed/terminating."""
        if self._closed or self._terminate.is_set():
            return
        with suppress(asyncio.QueueFull):
            self._outbox.put_nowait(self._control_msg())

    @property
    def last_heartbeat_at(self) -> float:
        return self._last_heartbeat_at

    def close(self) -> None:
        """Disconnect: cancel any in-flight waiters (single-reply
        and streaming both)."""
        if self._closed:
            return
        self._closed = True
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(
                    XRLEnvError(
                        f"node {self.node_id} disconnected before reply arrived"
                    )
                )
        self._pending.clear()
        # Streaming iterators check ``_closed`` between chunks
        # and raise XRLEnvError on disconnect; pushing a synthetic
        # FAILED reply here is the cleanest signal that wakes the
        # await regardless of whether the consumer is mid-await
        # on ``queue.get()``.
        for cmd_id, queue in list(self._pending_streams.items()):
            queue.put_nowait(
                pb.CommandReply(
                    command_id=cmd_id,
                    status=pb.ReplyStatus.FAILED,
                    error_kind="ControlPlaneLost",
                    error_message=(
                        f"node {self.node_id} disconnected mid-stream"
                    ),
                ),
            )
        self._pending_streams.clear()

    # ── Internals ────────────────────────────────────────────────────────────

    def _fresh_header(self, *, idempotency_key: str) -> pb.CommandHeader:
        return pb.CommandHeader(
            command_id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
        )

    def _control_msg(self, **kwargs: Any) -> pb.ControlMsg:
        return pb.ControlMsg(
            stream_epoch=self._stream_epoch,
            control_instance_id=self._control_instance_id,
            seq=self._seq.next(),
            **kwargs,
        )

    async def _send_and_wait(
        self,
        msg: pb.ControlMsg,
        command_id: str,
        *,
        timeout_s: float | None = None,
        extra_frames: list[pb.ControlMsg] | None = None,
    ) -> pb.CommandReply:
        with get_tracer().start_as_current_span(
            "xrlenv.transport.rpc",
            attributes={
                "command_kind": msg.WhichOneof("body") or "",
                "node_id": self.node_id,
                "timeout_s": timeout_s if timeout_s is not None else -1.0,
            },
        ):
            return await self._send_and_wait_impl(
                msg, command_id, timeout_s=timeout_s, extra_frames=extra_frames,
            )

    async def _send_and_wait_impl(
        self,
        msg: pb.ControlMsg,
        command_id: str,
        *,
        timeout_s: float | None = None,
        extra_frames: list[pb.ControlMsg] | None = None,
    ) -> pb.CommandReply:
        """Send a command and await the matching CommandReply.

        ``timeout_s`` is opt-in (default ``None`` preserves the
        original unbounded behaviour of every existing caller).
        New callers should set a concrete bound so a hung node
        surfaces a clean ``XRLEnvError`` instead of a silent
        indefinite wait. Operator-reported hang 2026-05-06: an
        old node binary received a new proto field its bindings
        didn't know, the dispatcher silently no-op'd, and the
        consumer's ``await future`` blocked forever. The ceiling
        here is the safety net for that and any analogous
        future scenario; per-command bounds in the node-side
        handler are still the right primary defence.
        """
        if self._closed:
            raise XRLEnvError(f"node {self.node_id} is disconnected")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[pb.CommandReply] = loop.create_future()
        # Register the reply Future BEFORE enqueuing any frame so a node that fails
        # mid-stream (e.g. on an early put-archive chunk) still finds a pending entry for
        # its FAILED reply. ``extra_frames`` (a multi-frame command like chunked
        # put_archive) all share this one command_id + Future and are sent in order,
        # ahead of the terminal ``msg``; the node reassembles them and replies once.
        self._pending[command_id] = future
        for frame in extra_frames or ():
            await self._outbox.put(frame)
        await self._outbox.put(msg)
        try:
            if timeout_s is None:
                reply = await future
            else:
                reply = await asyncio.wait_for(future, timeout=timeout_s)
        except asyncio.CancelledError:
            # An outer cancellation (e.g. an admin-level ``asyncio.wait_for``
            # ceiling firing) must not leave a dangling pending entry — a
            # dropped reply would otherwise keep it until disconnect. Pop and
            # re-raise so cancellation semantics are preserved. Health
            # accounting stays on the explicit timeout branch below, which a
            # per-call ``timeout_s`` now exercises for every RPC.
            self._pending.pop(command_id, None)
            raise
        except TimeoutError as exc:
            self._pending.pop(command_id, None)
            # Issue #18 (Ask #2): record the timeout so the scheduler's
            # placement gate can exclude this node while it's degraded.
            # A reply-timeout means the node-agent didn't answer within
            # the ceiling — wedged event loop, stalled docker daemon,
            # or version mismatch — all reasons to stop routing new
            # work here until it recovers.
            self._last_command_timeout_monotonic = time.monotonic()
            self._command_timeout_total += 1
            # NodeCommandTimeout is a subclass of XRLEnvError, so existing
            # ``except XRLEnvError`` callers are unaffected; the raw-destroy
            # path catches the specific type to defer teardown cleanup
            # instead of false-failing the rollout (#2).
            raise NodeCommandTimeout(
                f"node {self.node_id}: command {command_id} timed "
                f"out after {timeout_s:.1f}s waiting for reply. The "
                f"node-agent may be wedged or running an older "
                f"binary that doesn't recognise this command kind "
                f"(redeploy + restart fixes the version-mismatch "
                f"variant).",
            ) from exc
        if reply.status == pb.ReplyStatus.FAILED:
            # Name the node so a consumer-visible failure points at the
            # culprit directly — without this the operator has to
            # cross-reference timing against the control-plane log to
            # learn *which* node a "remote command ..." error came
            # from. Mirrors the ``node {node_id}: ...`` prefix the
            # timeout branch above already carries.
            raise XRLEnvError(
                f"node {self.node_id}: remote command "
                f"{reply.error_kind}: {reply.error_message}"
            )
        return reply


# ──────────────────────────────────────────────────────────────────────────────
# NodeControlServicer
# ──────────────────────────────────────────────────────────────────────────────


class NodeControlServicer(pb_grpc.NodeControlServicer):
    """gRPC service the node connects out to (spec 21).

    Per connected node:
    - reads ``NodeHello`` (first ``NodeMsg``)
    - constructs a :class:`RemoteNodeTransport`
    - publishes it via ``on_connected`` so the runtime can wire it into the
      coordinator
    - drains incoming :class:`NodeMsg` (replies + acks) into the transport
    - yields outgoing :class:`ControlMsg` from the transport's outbox
    - publishes via ``on_disconnected`` when the stream closes
    """

    def __init__(
        self,
        *,
        on_connected: NodeConnectedCallback,
        on_disconnected: NodeDisconnectedCallback,
        control_instance_id: str | None = None,
    ) -> None:
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._control_instance_id = control_instance_id or str(uuid.uuid4())

    async def NodeControlStream(
        self,
        request_iterator: AsyncIterator[pb.NodeMsg],
        context: Any,
    ) -> AsyncIterator[pb.ControlMsg]:
        request_iter = aiter(request_iterator)
        try:
            first = await anext(request_iter)
        except StopAsyncIteration:
            LOGGER.warning("node stream closed before NodeHello")
            return
        if not first.HasField("hello"):
            LOGGER.warning("node stream's first message was not NodeHello")
            return
        hello = first.hello

        stream_epoch = first.stream_epoch
        control_seq = _MonotonicCounter()
        outbox: asyncio.Queue[pb.ControlMsg] = asyncio.Queue()

        # ControlHello — first ControlMsg of every new epoch consumes seq=0.
        ctrl_hello = pb.ControlMsg(
            stream_epoch=stream_epoch,
            control_instance_id=self._control_instance_id,
            seq=control_seq.next() - 1,  # consume seq=0 explicitly
            hello=pb.ControlHello(
                control_instance_id=self._control_instance_id,
                last_node_seq_acked_in_prior_epoch=0,
            ),
        )
        # bump the counter so subsequent messages start at seq=1
        # (the line above used .next() then -1; reset to 0 cleanly)
        control_seq = _MonotonicCounter()
        ctrl_hello.seq = 0
        yield ctrl_hello

        transport = RemoteNodeTransport(
            node_id=hello.node_id,
            backends=list(hello.backends),
            hardware=conv.hardware_info_from_proto(hello.hardware),
            outbox=outbox,
            stream_epoch=stream_epoch,
            control_instance_id=self._control_instance_id,
            control_seq=control_seq,
            # §5.3 — carry the node's advertised runtimes + daemon default.
            supported_runtimes=list(hello.supported_runtimes),
            default_runtime=hello.default_runtime,
            # P6 (§8.6) — carry the node's shared-parent-cpuset capability.
            isolation_capable=bool(hello.isolation_capable),
            # WS1 — carry the node's chunked-put_archive capability.
            chunked_put_archive_capable=bool(hello.chunked_put_archive_capable),
        )
        # §5.3/§9 — the daemon default runtime is security-relevant: if a
        # node advertises a non-runc default, the ``allowed_runtimes``
        # opt-in is silently bypassed for every acquire on it. WARN loudly
        # at connect so a mis-provisioned node is caught in one log line
        # rather than by a container escaping the policy.
        if transport.default_runtime() not in ("", "runc"):
            LOGGER.warning(
                "node %s advertises docker default-runtime=%r (expected "
                "'runc'): every acquire on this node silently gets that "
                "runtime, bypassing allowed_runtimes. Fix the node's "
                "daemon.json (remove default-runtime / set it to runc) "
                "and restart docker.",
                hello.node_id, transport.default_runtime(),
            )
        self._on_connected(transport)
        # Issue #18 (Ask #2) — log the node-agent build identity at
        # connect and WARN on skew against the control plane's own.
        # ``hello.agent_version`` is empty when the node-agent binary
        # predates the field — itself a strong "this node is stale,
        # redeploy it" signal, so an empty value gets its own WARN.
        node_agent_version = hello.agent_version or ""
        control_version = agent_identity()
        if not node_agent_version:
            LOGGER.warning(
                "node=%s connected but reported NO agent_version — its "
                "binary predates the version-exchange field (issue #18). "
                "This node is almost certainly stale; redeploy it "
                "(deploy/refresh.sh). control_plane=%s",
                transport.node_id, control_version,
            )
        elif node_agent_version != control_version:
            LOGGER.warning(
                "node=%s version skew: node agent_version=%s, "
                "control plane=%s. Redeploy the node-agent "
                "(deploy/refresh.sh) if this is unexpected — a node on "
                "an older build won't carry recent fixes.",
                transport.node_id, node_agent_version, control_version,
            )
        LOGGER.info(
            "node=%s connected (epoch=%s, instance=%s, agent_version=%s)",
            transport.node_id,
            stream_epoch,
            self._control_instance_id,
            node_agent_version or "<unknown>",
        )

        reader_done = asyncio.Event()
        reader_task = asyncio.create_task(
            self._reader_loop(request_iter, transport, reader_done),
            name=f"node-reader-{transport.node_id}",
        )

        # Track the per-iteration tasks at function scope so the
        # ``finally`` block can always cancel them on teardown. The
        # previous implementation only cancelled the per-iteration
        # tasks in the normal-return branches of the loop body;
        # any teardown path that bypassed those branches (a
        # ``GeneratorExit`` at the ``yield``, a ``CancelledError``
        # propagating out of ``asyncio.wait`` when the gRPC handler
        # is being shut down, or the reader_done branch racing the
        # outbox.get in done) leaked the still-pending
        # ``outbox.get()`` / ``reader_done.wait()`` task wrappers,
        # which asyncio later reaped with
        # ``Task was destroyed but it is pending!`` warnings.
        outbox_get: asyncio.Task[pb.ControlMsg] | None = None
        reader_wait: asyncio.Task[bool] | None = None
        # ``term_wait`` wakes an otherwise-idle pump when the watchdog asks
        # to close this stream (``transport.request_terminate``), so a node
        # marked lost during a control-plane stall is actively disconnected
        # and reconnects instead of lingering half-open forever.
        term_wait: asyncio.Task[bool] | None = None

        async def _drain(*waiters: asyncio.Task[Any] | None) -> None:
            for w in waiters:
                if w is not None and not w.done():
                    w.cancel()
                    with suppress(asyncio.CancelledError):
                        await w

        try:
            while not reader_done.is_set() and not transport.terminate_event.is_set():
                outbox_get = asyncio.create_task(outbox.get())
                reader_wait = asyncio.create_task(reader_done.wait())
                term_wait = asyncio.create_task(transport.terminate_event.wait())
                done, _pending = await asyncio.wait(
                    {outbox_get, reader_wait, term_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if outbox_get in done:
                    yield outbox_get.result()
                    outbox_get = None  # consumed; nothing to cancel
                    await _drain(reader_wait, term_wait)
                    reader_wait = None
                    term_wait = None
                else:
                    # reader_done (stream ended) or terminate (watchdog close).
                    await _drain(outbox_get)
                    outbox_get = None
                    await _drain(reader_wait, term_wait)
                    reader_wait = None
                    term_wait = None
                    if transport.terminate_event.is_set():
                        LOGGER.info(
                            "node=%s control stream closed by control plane "
                            "(reason=%s); the node will reconnect and re-register.",
                            transport.node_id,
                            transport.terminate_reason or "unspecified",
                        )
                    break
        except asyncio.CancelledError:
            # The gRPC server cancels in-flight bidi RPCs on graceful
            # shutdown (the ``xrlenv up`` signal handler). An idle
            # stream parked in ``asyncio.wait`` never drains on its
            # own, so cancellation is the expected teardown path. Let
            # the ``finally`` below run its cleanup, then swallow the
            # cancellation so it doesn't escape to cygrpc as a
            # spurious "Exception not handled by _handle_exceptions"
            # ERROR + traceback in the operator log.
            LOGGER.debug(
                "node=%s stream cancelled (server shutdown / disconnect)",
                transport.node_id,
            )
        finally:
            # Cancel any task still pending from the last iteration —
            # covers GeneratorExit at the yield, CancelledError out of
            # asyncio.wait, and any other early-exit path that
            # bypassed the in-loop cleanup.
            for leftover in (outbox_get, reader_wait, term_wait):
                if leftover is not None and not leftover.done():
                    leftover.cancel()
                    with suppress(asyncio.CancelledError):
                        await leftover
            reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await reader_task
            transport.close()
            self._on_disconnected(transport)
            LOGGER.info("node=%s disconnected", transport.node_id)

    async def _reader_loop(
        self,
        request_iter: AsyncIterator[pb.NodeMsg],
        transport: RemoteNodeTransport,
        done_event: asyncio.Event,
    ) -> None:
        try:
            async for node_msg in request_iter:
                kind = node_msg.WhichOneof("body")
                # Issue #18: any inbound message refreshes liveness —
                # ``touch`` does it for heartbeats, ``mark_seen`` for
                # everything else — so a node busy streaming replies
                # is not false-flagged ``lost`` by the watchdog.
                if kind == "reply":
                    transport.mark_seen()
                    transport.deliver_reply(node_msg.reply)
                elif kind == "heartbeat":
                    hb = node_msg.heartbeat
                    transport.touch(
                        free_disk_bytes=int(hb.free_disk_bytes),
                        total_disk_bytes=int(hb.total_disk_bytes),
                        # P6 (§8.6, R6) — live pinnable-CPU accounting.
                        pinned_cpus_free=int(hb.pinned_cpus_free),
                        pinned_cpus_total=int(hb.pinned_cpus_total),
                        # Stage-1: ``health`` is absent on a pre-Stage-1
                        # node-agent — pass None so ``touch`` keeps the
                        # last-known (or unknown) value.
                        health=hb.health if hb.HasField("health") else None,
                    )
                elif kind == "ack":
                    transport.mark_seen()  # TODO: track for flow control in Slice 3.5
                elif kind == "hello":
                    transport.mark_seen()
                    LOGGER.warning(
                        "node=%s sent NodeHello mid-stream — ignored",
                        transport.node_id,
                    )
        finally:
            done_event.set()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _MonotonicCounter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def _range_kind_to_proto(kind: FetchRangeKind) -> pb.FetchRangeKind.ValueType:
    """Project the FetchRangeKind Literal onto the wire enum."""
    if kind == "summary_only":
        return pb.FetchRangeKind.SUMMARY_ONLY
    if kind == "step_range":
        return pb.FetchRangeKind.STEP_RANGE
    return pb.FetchRangeKind.WHOLE


__all__ = ["NodeControlServicer", "RemoteNodeTransport"]
