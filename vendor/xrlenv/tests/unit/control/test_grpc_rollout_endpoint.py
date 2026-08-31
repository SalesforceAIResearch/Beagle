"""Round-trip tests for the consumer-facing gRPC service (spec 05).

Spins up a real :class:`grpc.aio.Server` on loopback with a fake
``RolloutService`` and drives it through :class:`Client.grpc`. Covers:

- Happy-path round-trip for every RPC on the
  :class:`ClientTransport` Protocol.
- Trajectory metadata + node_id propagation through the wire.
- Error mapping: each xrlenv exception class round-trips to the
  matching gRPC status code AND back to the right Python exception.
- Carrier exceptions (``RolloutTruncated``, ``RolloutCancelled``,
  ``RolloutFailed``) carry their partial trajectory across the wire.
- Auth: missing token → ``AuthDenied``; wrong-role token →
  ``AuthDenied``; correct ``consumer`` token → success.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import grpc
import pytest
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.api._pb2 import rollout_control_pb2_grpc as rpb_grpc
from xrlenv.client import Client
from xrlenv.client.transport import GrpcClientTransport
from xrlenv.control.auth_interceptor import BearerScopeInterceptor
from xrlenv.control.rollout_endpoint import RolloutControlServicer
from xrlenv.control.security import TokenStore
from xrlenv.control.service import RawExecChunk, StartRolloutRequest, StartRolloutResponse
from xrlenv.errors import (
    AuthDenied,
    CapacityExhausted,
    ControlPlaneLost,
    NodeLost,
    RolloutCancelled,
    RolloutFailed,
    RolloutTruncated,
    SessionReaped,
    TemplateUnknown,
    XRLEnvError,
)
from xrlenv.types import (
    CancelGroupReport,
    RolloutStatus,
    Step,
    StepResult,
    TerminateRawGroupReport,
    Trajectory,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fake RolloutService — programmable per-method response or exception
# ──────────────────────────────────────────────────────────────────────────────


class _FakeRolloutService:
    """Minimal in-process service. Each method either returns the
    pre-configured value or raises the pre-configured exception."""

    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.exceptions: dict[str, Exception] = {}
        self.calls: dict[str, list[tuple[Any, ...]]] = {
            k: [] for k in (
                "start_rollout", "step", "finish", "cancel",
                "cancel_group", "terminate_raw_group", "replay",
                "heartbeat", "heartbeat_many",
                "set_final_reward", "list_nodes",
            )
        }

    def _maybe_raise(self, method: str) -> None:
        exc = self.exceptions.get(method)
        if exc is not None:
            raise exc

    async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
        self.calls["start_rollout"].append((req,))
        self._maybe_raise("start_rollout")
        return self.responses["start_rollout"]

    async def step(self, rollout_id: str, action: Any) -> StepResult:
        self.calls["step"].append((rollout_id, action))
        self._maybe_raise("step")
        return self.responses["step"]

    async def finish(self, rollout_id: str) -> Trajectory:
        self.calls["finish"].append((rollout_id,))
        self._maybe_raise("finish")
        return self.responses["finish"]

    async def cancel(self, rollout_id: str, reason: str) -> Trajectory:
        self.calls["cancel"].append((rollout_id, reason))
        self._maybe_raise("cancel")
        return self.responses["cancel"]

    async def cancel_group(
        self, group_id: str, reason: str, *, owner_id: str | None = None,
    ) -> CancelGroupReport:
        self.calls["cancel_group"].append((group_id, reason))
        self._maybe_raise("cancel_group")
        return self.responses["cancel_group"]

    async def terminate_raw_group(
        self, group_id: str, reason: str, *, owner_id: str | None = None,
    ) -> TerminateRawGroupReport:
        self.calls["terminate_raw_group"].append((group_id, reason, owner_id))
        self._maybe_raise("terminate_raw_group")
        return self.responses["terminate_raw_group"]

    async def replay(self, rollout_id: str) -> Trajectory:
        self.calls["replay"].append((rollout_id,))
        self._maybe_raise("replay")
        return self.responses["replay"]

    async def heartbeat(self, rollout_id: str) -> None:
        self.calls["heartbeat"].append((rollout_id,))
        self._maybe_raise("heartbeat")

    async def heartbeat_many(self, rollout_ids: list[str]) -> None:
        self.calls["heartbeat_many"].append((tuple(rollout_ids),))
        self._maybe_raise("heartbeat_many")

    async def set_final_reward(self, rollout_id: str, final_reward: float) -> None:
        self.calls["set_final_reward"].append((rollout_id, final_reward))
        self._maybe_raise("set_final_reward")

    async def list_nodes(self) -> list[Any]:
        self.calls["list_nodes"].append(())
        self._maybe_raise("list_nodes")
        return self.responses.get("list_nodes", [])

    # Raw-container surface (spec 07 ApplyEgress round-trip). ``apply_egress``
    # records its kwargs; ``raw_session_owner`` answers the endpoint's guard
    # (None → single-tenant "default", which passes with no token store).
    def raw_session_owner(self, rollout_id: str) -> str | None:
        return None

    async def container_get_archive(
        self, *, rollout_id: str, container_id: str, source_path: str,
    ) -> bytes:
        self.calls.setdefault("container_get_archive", []).append(
            (rollout_id, container_id, source_path),
        )
        self._maybe_raise("container_get_archive")
        return self.responses.get("container_get_archive", b"")

    async def apply_egress(self, **kw: Any) -> None:
        self.calls.setdefault("apply_egress", []).append((kw,))
        self._maybe_raise("apply_egress")

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
    ) -> AsyncIterator[Any]:
        # Server-streaming: yield any pre-programmed chunks, then raise the
        # pre-programmed exception mid-stream (matching a real reap, which is
        # discovered whenever the consumer next touches the session — not
        # necessarily on the very first RPC).
        self.calls.setdefault("container_exec_stream", []).append(
            (rollout_id, container_id, tuple(cmd)),
        )
        for chunk in self.responses.get("container_exec_stream", []):
            yield chunk
        self._maybe_raise("container_exec_stream")


# ──────────────────────────────────────────────────────────────────────────────
# Server fixture — boots a real grpc.aio server on an ephemeral port
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _serve(
    fake: _FakeRolloutService,
    *,
    token_store: TokenStore | None = None,
) -> AsyncIterator[int]:
    interceptors: tuple[grpc.aio.ServerInterceptor, ...] = ()
    if token_store is not None:
        interceptors = (
            BearerScopeInterceptor(store=token_store, state=None),
        )
    server = grpc.aio.server(interceptors=interceptors)
    rpb_grpc.add_RolloutControlServicer_to_server(
        RolloutControlServicer(service=fake), server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield port
    finally:
        await server.stop(grace=0.5)


def _traj(
    rollout_id: str = "rid-1",
    *,
    status: RolloutStatus = RolloutStatus.FINISHED,
    final_reward: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    return Trajectory(
        rollout_id=rollout_id,
        template="hello-shell",
        steps=[Step(
            index=0, action={"cmd": "echo hi"}, obs={"stdout": "hi\n"},
            reward=0.5, done=False, truncated=False, info={}, ts=0.1,
        )],
        status=status,
        final_reward=final_reward,
        metadata=metadata or {"node_id": "aws-1"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 — QueueStatus RPC
# ──────────────────────────────────────────────────────────────────────────────


async def test_queue_status_rpc_reports_position_and_depth() -> None:
    """Stage-2: the QueueStatus RPC reports a request's admission-queue
    position + depth, straight from the AdmissionQueue."""

    class _FakeAdmission:
        def queue_status(
            self, request_id: str, owner_id: str | None = None,
        ) -> tuple[int, int, str]:
            if request_id == "r-1":
                return (3, 7, "queued")
            return (0, 7, "not_in_queue")

    servicer = RolloutControlServicer(
        service=_FakeRolloutService(), admission=_FakeAdmission(),
    )
    resp = await servicer.QueueStatus(
        rpb.QueueStatusRequest(request_id="r-1"), context=None,
    )
    assert (resp.position, resp.queue_depth, resp.state) == (3, 7, "queued")

    unknown = await servicer.QueueStatus(
        rpb.QueueStatusRequest(request_id="r-other"), context=None,
    )
    assert unknown.state == "not_in_queue"


async def test_queue_status_rpc_empty_when_no_admission_wired() -> None:
    """No AdmissionQueue (single-node setup) → QueueStatus reports an
    empty queue rather than failing."""
    servicer = RolloutControlServicer(service=_FakeRolloutService())
    resp = await servicer.QueueStatus(
        rpb.QueueStatusRequest(request_id="x"), context=None,
    )
    assert (resp.position, resp.queue_depth, resp.state) == (0, 0, "not_in_queue")


# ──────────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────────


async def test_start_rollout_round_trips() -> None:
    fake = _FakeRolloutService()
    fake.responses["start_rollout"] = StartRolloutResponse(
        rollout_id="rid-7",
        init_obs={"stdout": "hello"},
        reward_mode="env_step",
    )
    async with _serve(fake) as port:
        client = Client.grpc("127.0.0.1", port)
        try:
            session = await client.rollout(
                template="hello-shell", init={"max_steps": 3, "cwd": "/sandbox"},
            )
            assert session.rollout_id == "rid-7"
            # The fake recorded the inflated request.
            sent_req = fake.calls["start_rollout"][0][0]
            assert sent_req.template == "hello-shell"
            assert sent_req.init == {"max_steps": 3, "cwd": "/sandbox"}
        finally:
            await client.close()


async def test_step_round_trips_action_and_observation() -> None:
    fake = _FakeRolloutService()
    fake.responses["step"] = StepResult(
        obs={"stdout": "world\n"}, reward=1.0, done=True, info={"k": "v"},
    )
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            res = await transport.step("rid-1", {"cmd": "echo world"})
        finally:
            await transport.close()
    assert res.reward == 1.0
    assert res.obs == {"stdout": "world\n"}
    assert res.done is True
    assert fake.calls["step"][0] == ("rid-1", {"cmd": "echo world"})


async def test_apply_egress_round_trips() -> None:
    # T1 (audit): exercise the public ApplyEgress RPC end to end — client
    # serialization (GrpcClientTransport.apply_egress) + servicer decode over a
    # real gRPC channel. The allowlist (cidrs, ports, dns_resolver) must arrive
    # intact at the service.
    from xrlenv.backends.egress import EgressAllowlist, EgressRule

    fake = _FakeRolloutService()
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            await transport.apply_egress(
                rollout_id="rid-1",
                container_id="cid-1",
                allowlist=EgressAllowlist(rules=(
                    EgressRule(cidr="3.149.157.52/32", ports=(443,)),
                    EgressRule(cidr="18.225.81.238/32"),
                )),
                dns_resolver="10.0.0.2/32",
            )
        finally:
            await transport.close()
    kw = fake.calls["apply_egress"][0][0]
    assert kw["rollout_id"] == "rid-1"
    assert kw["container_id"] == "cid-1"
    assert [(r.cidr, r.ports) for r in kw["allowlist"].rules] == [
        ("3.149.157.52/32", (443,)), ("18.225.81.238/32", None),
    ]
    assert kw["dns_resolver"] == "10.0.0.2/32"


async def test_apply_egress_empty_round_trips() -> None:
    # Empty allowlist (block-all) survives the wire as empty, not an error.
    from xrlenv.backends.egress import EgressAllowlist

    fake = _FakeRolloutService()
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            await transport.apply_egress(
                rollout_id="rid-1", container_id="cid-1",
                allowlist=EgressAllowlist(),
            )
        finally:
            await transport.close()
    kw = fake.calls["apply_egress"][0][0]
    assert kw["allowlist"].rules == ()
    assert kw["dns_resolver"] is None


async def test_container_get_archive_stream_roundtrips_over_real_grpc() -> None:
    """WS1 client hop, end-to-end over a real gRPC channel: a multi-chunk
    tarball is streamed by ``ContainerGetArchiveStream`` and reassembled
    byte-exact by ``GrpcClientTransport.container_get_archive`` — the path
    that replaces the unary 128 MiB-capped ``ContainerGetArchive`` for large
    consumer fetches (harbor's download_dir of verifier logs / artifacts)."""
    import os

    fake = _FakeRolloutService()
    payload = os.urandom(12 * 1024 * 1024)  # 3x the 4 MiB chunk size
    fake.responses["container_get_archive"] = payload
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            got = await transport.container_get_archive(
                rollout_id="rid-1", container_id="cid-1", source_path="/logs",
            )
        finally:
            await transport.close()
    assert got == payload
    assert fake.calls["container_get_archive"][0] == ("rid-1", "cid-1", "/logs")


async def test_finish_returns_full_trajectory_with_node_id() -> None:
    fake = _FakeRolloutService()
    fake.responses["finish"] = _traj(
        rollout_id="rid-fin", metadata={"node_id": "gcp-vm-1", "foo": "bar"},
    )
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            traj = await transport.finish("rid-fin")
        finally:
            await transport.close()
    assert traj.rollout_id == "rid-fin"
    assert traj.metadata["node_id"] == "gcp-vm-1"
    assert traj.metadata["foo"] == "bar"
    assert traj.final_reward == 0.5
    assert len(traj.steps) == 1
    assert traj.steps[0].action == {"cmd": "echo hi"}


async def test_cancel_group_round_trip() -> None:
    fake = _FakeRolloutService()
    fake.responses["cancel_group"] = CancelGroupReport(
        group_id="g-1", cancelled=("a", "b"), already_terminal=("c",),
    )
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            report = await transport.cancel_group("g-1", "consumer_cancelled")
        finally:
            await transport.close()
    assert report.group_id == "g-1"
    assert report.cancelled == ("a", "b")
    assert report.already_terminal == ("c",)


async def test_terminate_raw_group_round_trip() -> None:
    fake = _FakeRolloutService()
    fake.responses["terminate_raw_group"] = TerminateRawGroupReport(
        group_id="run-1", terminated=("r1", "r2"), already_terminal=("r3",),
    )
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            report = await transport.terminate_raw_group("run-1", "run_aborted")
        finally:
            await transport.close()
    assert report.group_id == "run-1"
    assert report.terminated == ("r1", "r2")
    assert report.already_terminal == ("r3",)
    assert fake.calls["terminate_raw_group"][0][:2] == ("run-1", "run_aborted")


async def test_heartbeat_and_set_final_reward_round_trip() -> None:
    fake = _FakeRolloutService()
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            await transport.heartbeat("rid-9")
            await transport.set_final_reward("rid-9", 0.75)
        finally:
            await transport.close()
    # The server routes a single-id heartbeat through the batched path.
    assert fake.calls["heartbeat_many"] == [(("rid-9",),)]
    assert fake.calls["set_final_reward"] == [("rid-9", 0.75)]


async def test_consumer_final_reward_mode_round_trips() -> None:
    """``consumer_final`` works over gRPC: the server returns
    ``reward_mode=consumer_final`` from StartRollout, the consumer SDK
    runs its ``reward_fn`` locally on the sealed trajectory after
    Finish, and the resulting scalar is pushed back via SetFinalReward.

    The round-trip exercises:
      - reward_mode enum survives both directions of the proto enum,
      - SetFinalReward mutates the server's stored final_reward,
      - the client-rehydrated trajectory carries the new value.

    Regression for the audit's L1 — the proto used to claim
    consumer_final was unsupported on the gRPC path.
    """
    fake = _FakeRolloutService()
    fake.responses["start_rollout"] = StartRolloutResponse(
        rollout_id="rid-cf",
        init_obs={"stdout": "go"},
        reward_mode="consumer_final",
    )
    finished = _traj(
        rollout_id="rid-cf", final_reward=0.0,  # placeholder, client will overwrite
    )
    fake.responses["finish"] = finished
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            resp = await transport.start_rollout(
                StartRolloutRequest(template="hello-shell"),
            )
            assert resp.reward_mode == "consumer_final"
            traj = await transport.finish("rid-cf")
            # Consumer-side: compute the final reward and push it back.
            await transport.set_final_reward("rid-cf", 0.875)
        finally:
            await transport.close()
    assert traj.rollout_id == "rid-cf"
    assert fake.calls["set_final_reward"] == [("rid-cf", 0.875)]


# ──────────────────────────────────────────────────────────────────────────────
# Error mapping — server raises xrlenv exception → client sees the same class
# ──────────────────────────────────────────────────────────────────────────────


async def test_template_unknown_round_trips() -> None:
    fake = _FakeRolloutService()
    fake.exceptions["start_rollout"] = TemplateUnknown("no such template 'foo'")
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(TemplateUnknown, match="no such template"):
                await transport.start_rollout(StartRolloutRequest(template="foo"))
        finally:
            await transport.close()


async def test_capacity_exhausted_round_trips() -> None:
    fake = _FakeRolloutService()
    fake.exceptions["start_rollout"] = CapacityExhausted("queue full")
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(CapacityExhausted, match="queue full"):
                await transport.start_rollout(StartRolloutRequest(template="t"))
        finally:
            await transport.close()


async def test_node_lost_and_control_plane_lost_round_trip_distinctly() -> None:
    """``NodeLost`` and ``ControlPlaneLost`` share ONE gRPC status code
    (``UNAVAILABLE`` — see ``rollout_endpoint._EXC_TO_CODE``), so the only thing
    that keeps them from colliding client-side is the ``xrlenv-error-kind``
    trailing-metadata key surviving the real wire round trip (a fake
    ``AioRpcError`` can't prove this — ``_classify_unmarked_rpc_error`` also
    maps bare ``UNAVAILABLE`` to ``ControlPlaneLost`` as its no-metadata
    fallback, so a metadata-loss bug would silently misreport every genuine
    ``NodeLost`` as ``ControlPlaneLost`` instead of raising or failing loudly).

    Both are raised by ``_raise_for_missing_session`` (the discriminator this
    module's other CP/node-lost tests exercise in-process); this is the
    over-the-wire half — a real ``grpc.aio.Server`` + real channel, matching
    every raw-container RPC (``ContainerExec``, ``DestroyContainer``, …) that
    shares this same servicer error path.
    """
    node_lost_fake = _FakeRolloutService()
    node_lost_fake.exceptions["start_rollout"] = NodeLost(
        "raw-container session: rollout 'rid-9' is gone because its node was "
        "lost (node_lost: node aws-1 went away; raw session sealed). The "
        "platform tore this down, not you — acquire a fresh session to retry.",
    )
    async with _serve(node_lost_fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(NodeLost, match="node was lost") as excinfo:
                await transport.start_rollout(StartRolloutRequest(template="t"))
            assert not isinstance(excinfo.value, ControlPlaneLost)
        finally:
            await transport.close()

    cp_lost_fake = _FakeRolloutService()
    cp_lost_fake.exceptions["start_rollout"] = ControlPlaneLost(
        "raw-container session: rollout 'rid-9' is gone because the control "
        "plane lost track of it (raw-gc-reconciler: lost-mid-run (no "
        "in-memory session; row age 700s)). The platform tore this down, "
        "not you — acquire a fresh session to retry.",
    )
    async with _serve(cp_lost_fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(ControlPlaneLost, match="lost track") as excinfo:
                await transport.start_rollout(StartRolloutRequest(template="t"))
            assert not isinstance(excinfo.value, NodeLost)
        finally:
            await transport.close()


async def test_session_reaped_round_trips_with_reason() -> None:
    """A liveness-reaper teardown must reach the client as a typed
    ``SessionReaped``, not crash the rehydration path.

    ``SessionReaped.__init__`` requires ``reason`` with no default — unlike
    every other entry in ``_KIND_TO_EXC``, which the generic ``cls(msg)``
    fallback in ``_rehydrate_xrlenv_error`` can construct with just the
    message. Before the fix, this raised ``TypeError: SessionReaped.__init__()
    missing 1 required positional argument: 'reason'`` from *inside*
    ``_rehydrate_xrlenv_error`` — so every consumer of a raw-container RPC
    (``ContainerExec``/``DestroyContainer``/etc, which share this same
    error-mapping path) that hit a reaped session over real gRPC got a bare
    ``TypeError`` instead of ``SessionReaped``, defeating both the
    ``except XRLEnvError`` pattern used throughout the codebase (e.g. the
    docker-compat drop-in's ``_run_op``) and the whole point of this feature.
    """
    fake = _FakeRolloutService()
    fake.exceptions["start_rollout"] = SessionReaped(
        "raw-container session: rollout 'rid-9' was reaped by the control "
        "plane and no longer exists (quarantine horizon exceeded).",
        reason="quarantine horizon exceeded",
        reaped_at=1_700_000_000.0,
    )
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(SessionReaped, match="quarantine horizon exceeded") as excinfo:
                await transport.start_rollout(StartRolloutRequest(template="t"))
            exc = excinfo.value
            assert exc.reason == "quarantine horizon exceeded"
            assert exc.retryable is True
            # Known wire gap: only ``reason`` rides the xrlenv-error-reason
            # metadata key. ``reaped_at`` has no metadata key of its own, so it
            # is always lost across the wire even though the server-side
            # exception carried it — the client-side SessionReaped always has
            # reaped_at=None. If a consumer starts depending on reaped_at,
            # this assertion is the one to update alongside adding a wire
            # field for it.
            assert exc.reaped_at is None
        finally:
            await transport.close()


async def test_session_reaped_round_trips_over_container_exec_stream() -> None:
    """Same defect, the OTHER surface: ``ContainerExecStream`` is a
    server-streaming RPC and reaches ``_rehydrate_xrlenv_error`` through a
    different client-side call site (``GrpcClientTransport.
    container_exec_stream``'s ``except grpc.aio.AioRpcError`` handler,
    ``xrlenv/client/transport.py``) than the unary path exercised by
    ``test_session_reaped_round_trips_with_reason``.

    The fixup commit's own message claims ``ContainerExecStream`` was one of
    the broken RPCs ("Every RPC sharing the servicer's error path
    (ContainerExec, ContainerExecStream, Get/PutArchive, DestroyContainer,
    ApplyEgress) delivered a bare TypeError") but nothing in the test suite
    actually drove a reap through a streaming call over real gRPC — this
    closes that gap. A reap can surface mid-stream (the consumer is already
    reading output when the reaper tears the session down), so the fake
    yields one real chunk before raising, mirroring
    ``compat/docker_client.py``'s ``_on_error`` streaming-error path.
    """
    fake = _FakeRolloutService()
    fake.responses["container_exec_stream"] = [
        RawExecChunk(stdout=b"working...\n", stderr=b"", done=False, exit_code=0, timed_out=False),
    ]
    fake.exceptions["container_exec_stream"] = SessionReaped(
        "raw-container session: rollout 'rid-9' was reaped by the control "
        "plane and no longer exists (quarantine horizon exceeded).",
        reason="quarantine horizon exceeded",
        reaped_at=1_700_000_000.0,
    )
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            chunks = []
            with pytest.raises(SessionReaped, match="quarantine horizon exceeded") as excinfo:
                async for chunk in transport.container_exec_stream(
                    rollout_id="rid-9", container_id="cid-1", cmd=["echo", "hi"],
                ):
                    chunks.append(chunk)
            # The chunk emitted before the reap must have been delivered —
            # a reap mid-stream doesn't retroactively erase output already
            # read.
            assert len(chunks) == 1
            assert chunks[0].stdout == b"working...\n"
            exc = excinfo.value
            assert exc.reason == "quarantine horizon exceeded"
            assert exc.retryable is True
            assert exc.reaped_at is None  # same wire gap as the unary path
        finally:
            await transport.close()


async def test_rollout_truncated_carries_partial() -> None:
    fake = _FakeRolloutService()
    partial = _traj(rollout_id="rid-trunc", status=RolloutStatus.TRUNCATED)
    fake.exceptions["step"] = RolloutTruncated("hard deadline", partial=partial)
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(RolloutTruncated) as info:
                await transport.step("rid-trunc", {"cmd": "x"})
        finally:
            await transport.close()
    assert info.value.partial is not None
    assert info.value.partial.rollout_id == "rid-trunc"
    assert info.value.partial.status == RolloutStatus.TRUNCATED


async def test_rollout_failed_carries_reason_and_partial() -> None:
    fake = _FakeRolloutService()
    partial = _traj(rollout_id="rid-fail", status=RolloutStatus.FAILED)
    fake.exceptions["step"] = RolloutFailed(
        "init_cmd exit_code=1", reason="init_failed", partial=partial,
    )
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(RolloutFailed) as info:
                await transport.step("rid-fail", {"cmd": "x"})
        finally:
            await transport.close()
    assert info.value.reason == "init_failed"
    assert info.value.partial is not None
    assert info.value.partial.rollout_id == "rid-fail"


async def test_rollout_cancelled_carries_partial() -> None:
    fake = _FakeRolloutService()
    partial = _traj(rollout_id="rid-can", status=RolloutStatus.CANCELLED)
    fake.exceptions["finish"] = RolloutCancelled("group_cancelled", partial=partial)
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(RolloutCancelled) as info:
                await transport.finish("rid-can")
        finally:
            await transport.close()
    assert info.value.partial is not None
    assert info.value.partial.status == RolloutStatus.CANCELLED


# ──────────────────────────────────────────────────────────────────────────────
# Auth — bearer token + scope enforcement
# ──────────────────────────────────────────────────────────────────────────────


async def test_missing_token_returns_auth_denied() -> None:
    """Server has a non-empty TokenStore → unauth calls must reject."""
    fake = _FakeRolloutService()
    fake.responses["start_rollout"] = StartRolloutResponse(
        rollout_id="x", init_obs=None, reward_mode="env_step",
    )
    store = TokenStore()
    store.add("consumer", "the-correct-token")
    async with _serve(fake, token_store=store) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(AuthDenied):
                await transport.start_rollout(StartRolloutRequest(template="t"))
        finally:
            await transport.close()


async def test_wrong_role_token_returns_auth_denied() -> None:
    """A node-role token must not be allowed to drive consumer RPCs."""
    fake = _FakeRolloutService()
    fake.responses["start_rollout"] = StartRolloutResponse(
        rollout_id="x", init_obs=None, reward_mode="env_step",
    )
    store = TokenStore()
    store.add("node", "node-token")
    async with _serve(fake, token_store=store) as port:
        transport = GrpcClientTransport(
            host="127.0.0.1", port=port, token="node-token",
        )
        try:
            with pytest.raises(AuthDenied):
                await transport.start_rollout(StartRolloutRequest(template="t"))
        finally:
            await transport.close()


async def test_correct_consumer_token_succeeds() -> None:
    fake = _FakeRolloutService()
    fake.responses["start_rollout"] = StartRolloutResponse(
        rollout_id="x", init_obs=None, reward_mode="env_step",
    )
    store = TokenStore()
    store.add("consumer", "the-correct-token")
    async with _serve(fake, token_store=store) as port:
        transport = GrpcClientTransport(
            host="127.0.0.1", port=port, token="the-correct-token",
        )
        try:
            resp = await transport.start_rollout(
                StartRolloutRequest(template="t"),
            )
        finally:
            await transport.close()
    assert resp.rollout_id == "x"


# ──────────────────────────────────────────────────────────────────────────────
# Channel-layer failure mapping — ensure unrelated gRPC errors don't masquerade
# as something more specific.
# ──────────────────────────────────────────────────────────────────────────────


async def test_unreachable_endpoint_raises_xrlenv_error() -> None:
    """Connecting to a port nobody's listening on should surface as a
    transport-class XRLEnvError (we use ControlPlaneLost) rather than
    bubbling up a raw grpc exception."""
    transport = GrpcClientTransport(host="127.0.0.1", port=1, token=None)
    try:
        with pytest.raises(XRLEnvError):
            await transport.start_rollout(StartRolloutRequest(template="t"))
    finally:
        await transport.close()


# ──────────────────────────────────────────────────────────────────────────────
# D21 — ListNodes RPC + Client.list_nodes() / Client.wait_for_nodes()
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_nodes_round_trips_node_records() -> None:
    """Servicer reads ``list_nodes()`` from the service, converts each
    ``NodeRecord`` to the wire ``NodeInfo``, and the gRPC client reads
    them back as ``NodeRecord`` again. Pin the round-trip so a future
    proto change to ``NodeInfo`` (additional fields) doesn't silently
    drop data on the way back.
    """
    from xrlenv.control.state import NodeRecord

    fake = _FakeRolloutService()
    fake.responses["list_nodes"] = [
        NodeRecord(
            node_id="gcp-1", status="connected",
            connected_at=1700000000.0, last_seen_at=1700000050.0,
            stream_epoch="ep-1", instance_id="inst-1",
            backends=["docker"],
        ),
        NodeRecord(
            node_id="gcp-2", status="lost",
            connected_at=1700000010.0, last_seen_at=1700000045.0,
            stream_epoch="ep-2", instance_id="inst-2",
            backends=["docker", "cube"],
        ),
    ]

    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            nodes = await transport.list_nodes()
        finally:
            await transport.close()

    assert len(fake.calls["list_nodes"]) == 1
    assert len(nodes) == 2
    n0, n1 = nodes
    assert n0.node_id == "gcp-1"
    assert n0.status == "connected"
    assert n0.backends == ["docker"]
    assert n0.connected_at == 1700000000.0
    assert n0.last_seen_at == 1700000050.0
    assert n0.stream_epoch == "ep-1"
    assert n0.instance_id == "inst-1"
    assert n1.node_id == "gcp-2"
    assert n1.status == "lost"
    assert n1.backends == ["docker", "cube"]


@pytest.mark.asyncio
async def test_client_list_nodes_via_grpc() -> None:
    """The high-level ``Client.list_nodes()`` surface returns the same
    rows the transport returned — covers the SDK call path end-to-end.
    """
    from xrlenv.control.state import NodeRecord

    fake = _FakeRolloutService()
    fake.responses["list_nodes"] = [
        NodeRecord(
            node_id="A", status="connected",
            connected_at=1.0, last_seen_at=2.0,
            backends=["docker"],
        ),
    ]

    async with _serve(fake) as port:
        client = Client.grpc(host="127.0.0.1", port=port)
        try:
            nodes = await client.list_nodes()
        finally:
            await client.close()

    assert [n.node_id for n in nodes] == ["A"]
    assert nodes[0].status == "connected"


@pytest.mark.asyncio
async def test_client_wait_for_nodes_succeeds_when_min_attached() -> None:
    """``wait_for_nodes(min_nodes=2)`` returns immediately when the
    cluster already has ≥2 connected nodes."""
    from xrlenv.control.state import NodeRecord

    fake = _FakeRolloutService()
    fake.responses["list_nodes"] = [
        NodeRecord(
            node_id="A", status="connected",
            connected_at=1.0, last_seen_at=2.0, backends=["docker"],
        ),
        NodeRecord(
            node_id="B", status="connected",
            connected_at=1.5, last_seen_at=2.5, backends=["docker"],
        ),
    ]

    async with _serve(fake) as port:
        client = Client.grpc(host="127.0.0.1", port=port)
        try:
            nodes = await client.wait_for_nodes(min_nodes=2, timeout_s=5.0)
        finally:
            await client.close()

    assert len(nodes) == 2
    assert {n.node_id for n in nodes} == {"A", "B"}


@pytest.mark.asyncio
async def test_client_wait_for_nodes_times_out_when_short() -> None:
    """``wait_for_nodes(min_nodes=2)`` raises ``TimeoutError`` when only
    1 node is attached and the grace window expires. Error message
    surfaces the count + the configured min so the operator can
    diagnose."""
    from xrlenv.control.state import NodeRecord

    fake = _FakeRolloutService()
    fake.responses["list_nodes"] = [
        NodeRecord(
            node_id="solo", status="connected",
            connected_at=1.0, last_seen_at=2.0, backends=["docker"],
        ),
    ]

    async with _serve(fake) as port:
        client = Client.grpc(host="127.0.0.1", port=port)
        try:
            with pytest.raises(TimeoutError, match="only 1 of the required 2"):
                await client.wait_for_nodes(
                    min_nodes=2, timeout_s=0.5, poll_interval_s=0.1,
                )
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_client_wait_for_nodes_filters_by_backend() -> None:
    """The ``backend=`` filter drops nodes that don't advertise the
    requested backend; useful when the cluster has heterogeneous
    backends and the consumer only wants the docker subset."""
    from xrlenv.control.state import NodeRecord

    fake = _FakeRolloutService()
    fake.responses["list_nodes"] = [
        NodeRecord(
            node_id="docker-1", status="connected",
            connected_at=1.0, last_seen_at=2.0, backends=["docker"],
        ),
        NodeRecord(
            node_id="cube-1", status="connected",
            connected_at=1.0, last_seen_at=2.0, backends=["cube"],
        ),
    ]

    async with _serve(fake) as port:
        client = Client.grpc(host="127.0.0.1", port=port)
        try:
            nodes = await client.wait_for_nodes(
                min_nodes=1, backend="docker", timeout_s=2.0,
            )
        finally:
            await client.close()

    assert [n.node_id for n in nodes] == ["docker-1"]


@pytest.mark.asyncio
async def test_client_wait_for_nodes_skips_lost_status() -> None:
    """A node in ``status='lost'`` (heartbeat-watchdog tripped) is NOT
    counted toward ``min_nodes``. Pre-fix to D21, the smoke driver's
    probe-and-retry counted any rollout-accepting node; the cleaner
    surface is "connected only". Pin that contract."""
    from xrlenv.control.state import NodeRecord

    fake = _FakeRolloutService()
    fake.responses["list_nodes"] = [
        NodeRecord(
            node_id="A", status="connected",
            connected_at=1.0, last_seen_at=2.0, backends=["docker"],
        ),
        NodeRecord(
            node_id="B", status="lost",
            connected_at=1.0, last_seen_at=2.0, backends=["docker"],
        ),
    ]

    async with _serve(fake) as port:
        client = Client.grpc(host="127.0.0.1", port=port)
        try:
            with pytest.raises(TimeoutError, match="only 1 of the required 2"):
                await client.wait_for_nodes(
                    min_nodes=2, timeout_s=0.3, poll_interval_s=0.1,
                )
        finally:
            await client.close()


async def test_cancel_unknown_rollout_does_not_crash_servicer() -> None:
    """Cancel must be idempotent. ``service.cancel`` raises ``KeyError`` for
    a rollout that was never placed / already reaped / lives in the raw
    path (not the gym/step ``rollouts`` table). Pre-fix the bare KeyError
    escaped as an unhandled servicer crash (the prod "Unexpected [KeyError]
    raised by servicer method … /Cancel" spam). It must now surface as a
    clean NOT_FOUND, which the client classifies as ReplayUnavailable —
    never an opaque UNKNOWN servicer crash."""
    from xrlenv.errors import ReplayUnavailable

    fake = _FakeRolloutService()
    fake.exceptions["cancel"] = KeyError("unknown rollout_id r-gone")
    async with _serve(fake) as port:
        transport = GrpcClientTransport(host="127.0.0.1", port=port, token=None)
        try:
            with pytest.raises(ReplayUnavailable, match="not found or already"):
                await transport.cancel("r-gone", "consumer_cancelled")
        finally:
            await transport.close()
    # The endpoint still invoked the service (and absorbed the KeyError).
    assert fake.calls["cancel"] == [("r-gone", "consumer_cancelled")]


# ──────────────────────────────────────────────────────────────────────────────
# P6 — the control-plane ingress derives the CPU-isolation contract ONCE
#
# This closes the audit's "AcquireContainer still does not consume
# resources.cpu_isolation" gap: the servicer must resolve the effective mode
# from the explicit wire field (preferred) falling back to the legacy
# cpu_pinning bool, and pass the resulting CpuIsolation to acquire_container.
# ──────────────────────────────────────────────────────────────────────────────


class _AcquireSpyService:
    """Records the ``cpu_isolation`` (and other) kwargs the ingress passes to
    ``acquire_container``. No token store → owner resolves to 'default'."""

    def __init__(self) -> None:
        self.acquire_calls: list[dict[str, Any]] = []

    def raw_session_owner(self, rollout_id: str) -> str | None:
        return None

    async def acquire_container(self, **kwargs: Any) -> Any:
        self.acquire_calls.append(kwargs)
        from xrlenv.control.service import RawAcquireResult
        return RawAcquireResult(
            rollout_id="r-iso", container_id="c-iso",
            container_name="cname-iso", node_id="node-iso",
        )


async def _ingress_cpu_isolation(
    *, cpu_isolation_wire: str = "", cpu_pinning: bool = False,
) -> Any:
    """Drive one AcquireContainer through the real servicer and return the
    derived cpu_isolation the ingress forwarded to the service."""
    spy = _AcquireSpyService()
    servicer = RolloutControlServicer(service=spy)  # type: ignore[arg-type]
    req = rpb.AcquireContainerRequest(image="busybox:1")
    if cpu_isolation_wire:
        req.resources.cpu_isolation = cpu_isolation_wire
    if cpu_pinning:
        req.runtime_limits.cpu_pinning = True
    await servicer.AcquireContainer(req, context=None)  # type: ignore[arg-type]
    assert len(spy.acquire_calls) == 1
    return spy.acquire_calls[0]["cpu_isolation"]


async def test_ingress_derives_cpu_isolation_from_explicit_wire_field() -> None:
    """Explicit ``resources.cpu_isolation='required'`` reaches the service as
    ``CpuIsolation.REQUIRED`` — the field is no longer inert at the ingress."""
    from xrlenv.backends.base import CpuIsolation

    got = await _ingress_cpu_isolation(cpu_isolation_wire="required")
    assert got is CpuIsolation.REQUIRED


async def test_ingress_derives_best_effort_from_legacy_cpu_pinning() -> None:
    """Legacy alias: ``runtime_limits.cpu_pinning=True`` with no explicit
    field → ``BEST_EFFORT`` (scheduling-neutral, exactly the old pin)."""
    from xrlenv.backends.base import CpuIsolation

    got = await _ingress_cpu_isolation(cpu_pinning=True)
    assert got is CpuIsolation.BEST_EFFORT


async def test_ingress_explicit_required_wins_over_legacy_pinning() -> None:
    """Derive-once precedence: an explicit ``required`` wins over the legacy
    ``cpu_pinning=True`` best-effort alias."""
    from xrlenv.backends.base import CpuIsolation

    got = await _ingress_cpu_isolation(
        cpu_isolation_wire="required", cpu_pinning=True,
    )
    assert got is CpuIsolation.REQUIRED


async def test_ingress_defaults_off_when_neither_set() -> None:
    """Neither the explicit field nor the legacy alias set → ``OFF`` (safe
    default; the ingress never accidentally isolates)."""
    from xrlenv.backends.base import CpuIsolation

    got = await _ingress_cpu_isolation()
    assert got is CpuIsolation.OFF
