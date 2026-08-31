"""Tests for the spec-21 bidi wire (Slice 3).

End-to-end via real grpc.aio loopback: stand up a NodeControlServicer on a
local port, point a NodeGrpcLink at it backed by a fake NodeAgent, and
exercise the round trip. Keeps the wire format honest without requiring
Docker.
"""

from __future__ import annotations

import asyncio
import socket
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import grpc
import pytest
from xrlenv.api._pb2 import node_control_pb2_grpc as pb_grpc
from xrlenv.backends.base import (
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.control.grpc_endpoint import (
    NodeControlServicer,
    RemoteNodeTransport,
)
from xrlenv.node.grpc_link import NodeGrpcLink, _IdempotencyCache
from xrlenv.node.hw_probe import HardwareInfo

# ──────────────────────────────────────────────────────────────────────────────
# Fakes / fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _free_port() -> int:
    """Grab an OS-assigned free port. Race-prone but fine for test setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4,
        mem_bytes=16 * 1024**3,
        disk_bytes=200 * 1024**3,
        has_kvm=False,
        has_gpu=False,
        gpu_model=None,
        kernel_version="0.0.0",
        platform="linux",
    )


class FakeAgent:
    """In-memory NodeAgent stand-in: records calls, returns canned replies."""

    def __init__(self, *, node_id: str = "fake-node") -> None:
        self.node_id = node_id
        self.create_calls = 0
        self.destroy_calls = 0
        self.setup_calls = 0
        self.step_calls = 0
        self.teardown_calls = 0
        # A5 / D17 stage 2: most-recent per-call HTTP cap that arrived
        # via the corresponding ``EnvSetupCommand.request_timeout_s`` /
        # ``EnvStepCommand.request_timeout_s`` /
        # ``EnvTeardownCommand.request_timeout_s`` proto field. ``None``
        # means the field was unset (proto3 default 0.0 → ``_per_call_cap``
        # returns ``None``); a positive float means the control plane
        # opted into the per-call override.
        self.last_setup_request_timeout_s: float | None = None
        self.last_step_request_timeout_s: float | None = None
        self.last_teardown_request_timeout_s: float | None = None
        # A3 / D15: per-instance sandbox-ID set advertised by
        # ``list_sandbox_ids``. Tests preset this before invoking the
        # round-trip RPC.
        self._sandbox_ids: list[str] = []
        # A5 / D17 stage 1: most-recent stub_request_timeout_s
        # received via ``create_sandbox``; ``None`` until first call.
        self.last_stub_request_timeout_s: float | None = None
        # A1 / D18+D19: per-image query_image replies; tests pre-set.
        self._image_query_replies: dict[str, object] = {}
        # B7.6: full per-node image cache snapshot returned by
        # report_images(). Tests pre-set.
        self._image_report: object | None = None
        # Egress: record the most-recent apply_egress for the round-trip test.
        self.apply_egress_calls = 0
        self.last_apply_egress: dict[str, Any] | None = None
        # P6 step-2a: capability + pinned-CPU accounting the NodeGrpcLink reads
        # to build NodeHello / Heartbeat. Preset so a connect test can prove the
        # value rides the wire to the control transport.
        self.isolation_capable_flag = True
        self.pinned_cpu_capacity_value = (6, 8)

    def supported_backends(self) -> list[str]:
        return ["docker"]

    def isolation_capable(self) -> bool:
        return self.isolation_capable_flag

    def pinned_cpu_capacity(self) -> tuple[int, int]:
        return self.pinned_cpu_capacity_value

    def health_snapshot(self) -> None:
        # Stage-1: the real NodeAgent returns a NodeHealthSnapshot here;
        # the fake has no raw manager, so the heartbeat carries no health.
        return None

    def hardware(self) -> HardwareInfo:
        return _hw()

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
        self.create_calls += 1
        # A5 / D17 stage 1 (audit response): record the cap for tests
        # that assert the kwarg propagated through the wire.
        self.last_stub_request_timeout_s = stub_request_timeout_s
        return SandboxHandle(
            id=f"sb-{self.create_calls}",
            backend=backend,
            backend_ref=f"cid-{self.create_calls}",
            stub_endpoint="tcp://127.0.0.1:0",
        )

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        self.destroy_calls += 1

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: Any,
        dns_resolver: str | None = None,
        backend: str = "docker",
    ) -> None:
        self.apply_egress_calls += 1
        self.last_apply_egress = {
            "rollout_id": rollout_id,
            "container_id": container_id,
            "allowlist": allowlist,
            "dns_resolver": dns_resolver,
        }

    async def env_setup(
        self,
        sb: SandboxHandle,
        *,
        adapter_module: str,
        adapter_class: str,
        init_params: dict[str, Any],
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.setup_calls += 1
        self.last_setup_request_timeout_s = request_timeout_s
        return {"obs": {"hello": True}, "capabilities": {"supported_reward_modes": []}}

    async def env_step(
        self,
        sb: SandboxHandle,
        action: Any,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.step_calls += 1
        self.last_step_request_timeout_s = request_timeout_s
        return {
            "obs": {"echo": action},
            "reward": 0.1 * self.step_calls,
            "done": self.step_calls >= 2,
            "truncated": False,
            "info": {"steps": self.step_calls},
        }

    async def env_teardown(
        self,
        sb: SandboxHandle,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.teardown_calls += 1
        self.last_teardown_request_timeout_s = request_timeout_s
        return {"status": "ok"}

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=0.0, rss_bytes=0, disk_bytes=0, rx_bytes=0, tx_bytes=0)

    async def list_sandbox_ids(
        self, *, backend: str | None = None,
    ) -> list[str]:
        return list(self._sandbox_ids)

    async def query_image(self, image: str) -> object:
        # A1 / D18+D19: round-trip support. Tests pre-set
        # ``_image_query_replies`` keyed by image tag; default-absent.
        from xrlenv.node.image_cache import ImageQueryResult
        return self._image_query_replies.get(
            image, ImageQueryResult(present=False),
        )

    async def report_images(self, *, include_shared_size=False) -> object:
        # B7.6: round-trip support. Tests pre-set ``_image_report``;
        # default-empty.
        from xrlenv.node.image_cache import NodeImageReport
        return self._image_report or NodeImageReport()

    async def fetch_trajectory(
        self,
        rollout_id: str,
        *,
        range_kind: str = "whole",
        step_start: int = 0,
        step_end: int | None = None,
    ) -> Any:
        from xrlenv.types import RolloutStatus, Step, Trajectory

        full = Trajectory(
            rollout_id=rollout_id, template="t",
            steps=[
                Step(index=i, action={"a": i}, obs={"o": i}, reward=0.0,
                     done=(i == 2), truncated=False, info={}, ts=float(i))
                for i in range(3)
            ],
            status=RolloutStatus.FINISHED, final_reward=0.5, metadata={},
        )
        if range_kind == "summary_only":
            return full.model_copy(update={"steps": []})
        if range_kind == "step_range":
            end = step_end if step_end is not None and step_end > 0 else len(full.steps)
            return full.model_copy(update={"steps": full.steps[step_start:end]})
        return full


@pytest.fixture
def free_port() -> Iterator[int]:
    yield _free_port()


@pytest.fixture
async def grpc_endpoint(
    free_port: int,
) -> AsyncIterator[tuple[int, list[RemoteNodeTransport], asyncio.Event]]:
    """Spin up a NodeControlServicer on ``free_port`` and yield references
    to the connected-transports list + a "node connected" event.
    """
    connected: list[RemoteNodeTransport] = []
    node_event = asyncio.Event()

    def _on_connected(t: RemoteNodeTransport) -> None:
        connected.append(t)
        node_event.set()

    def _on_disconnected(t: RemoteNodeTransport) -> None:
        # Leave the appended transport in `connected` so tests can assert on it.
        pass

    server = grpc.aio.server()
    pb_grpc.add_NodeControlServicer_to_server(
        NodeControlServicer(
            on_connected=_on_connected,
            on_disconnected=_on_disconnected,
            control_instance_id=str(uuid.uuid4()),
        ),
        server,
    )
    server.add_insecure_port(f"127.0.0.1:{free_port}")
    await server.start()
    try:
        yield free_port, connected, node_event
    finally:
        await server.stop(grace=1.0)


@pytest.fixture
async def linked_pair(
    grpc_endpoint: tuple[int, list[RemoteNodeTransport], asyncio.Event],
) -> AsyncIterator[
    tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]]
]:
    """Connect a FakeAgent-backed link to the grpc_endpoint and hand back
    the agent + remote transport + link + link's task (so tests can await it
    on shutdown).
    """
    port, connected, node_event = grpc_endpoint
    agent = FakeAgent()
    link = NodeGrpcLink(agent, control_addr=f"127.0.0.1:{port}")  # type: ignore[arg-type]
    link_task = asyncio.create_task(link.run_forever(), name="test-link")

    try:
        await asyncio.wait_for(node_event.wait(), timeout=5.0)
        transport = connected[0]
        yield agent, transport, link, link_task
    finally:
        link.request_stop()
        link_task.cancel()
        with suppress(asyncio.CancelledError):
            await link_task


# ──────────────────────────────────────────────────────────────────────────────
# Idempotency cache
# ──────────────────────────────────────────────────────────────────────────────


def test_idempotency_cache_returns_cached_reply() -> None:
    from xrlenv.api._pb2 import node_control_pb2 as pb

    cache = _IdempotencyCache(max_size=4, ttl_s=60.0)
    reply = pb.CommandReply(command_id="c1", status=pb.ReplyStatus.OK)
    cache.put("key-1", reply)
    fetched = cache.get("key-1")
    assert fetched is not None
    assert fetched.command_id == "c1"


def test_idempotency_cache_lru_eviction() -> None:
    from xrlenv.api._pb2 import node_control_pb2 as pb

    cache = _IdempotencyCache(max_size=2, ttl_s=60.0)
    for k in ("a", "b", "c"):  # third put evicts oldest
        cache.put(k, pb.CommandReply(command_id=k, status=pb.ReplyStatus.OK))
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end gRPC round trip
# ──────────────────────────────────────────────────────────────────────────────


async def test_node_connects_and_publishes_hardware(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    agent, transport, _link, _task = linked_pair
    assert transport.node_id == agent.node_id
    assert transport.supported_backends() == ["docker"]
    hw = transport.hardware()
    assert hw.vcpus == 4
    assert hw.platform == "linux"


async def test_create_sandbox_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    agent, transport, _link, _task = linked_pair
    handle = await transport.create_sandbox(
        rollout_id="r-1",
        backend="docker",
        template=TemplateRef(name="t", image="im:1", digest="sha256:abc"),
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=128_000_000,
            mem_limit_bytes=256_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",
    )
    assert agent.create_calls == 1
    assert handle.id == "sb-1"
    assert handle.backend == "docker"


async def test_env_setup_step_teardown_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    agent, transport, _link, _task = linked_pair
    sb = await transport.create_sandbox(
        rollout_id="r-2",
        backend="docker",
        template=TemplateRef(name="t", image="im:1"),
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",
    )
    setup = await transport.env_setup(
        sb,
        adapter_module="xrlenv.templates.hello_shell.adapter",
        adapter_class="ShellEnvAdapter",
        init_params={"max_steps": 2},
    )
    assert setup["obs"] == {"hello": True}

    step1 = await transport.env_step(sb, {"cmd": "echo hi"})
    assert step1["obs"] == {"echo": {"cmd": "echo hi"}}
    assert step1["done"] is False

    step2 = await transport.env_step(sb, {"cmd": "echo bye"})
    assert step2["done"] is True

    teardown = await transport.env_teardown(sb)
    assert teardown["status"] == "ok"

    assert agent.setup_calls == 1
    assert agent.step_calls == 2
    assert agent.teardown_calls == 1


async def test_destroy_sandbox_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    agent, transport, _link, _task = linked_pair
    sb = await transport.create_sandbox(
        rollout_id="r-3",
        backend="docker",
        template=TemplateRef(name="t", image="im:1"),
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",
    )
    await transport.destroy_sandbox(sb)
    assert agent.destroy_calls == 1


async def test_apply_egress_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """The allowlist survives EgressAllowlist→pb→EgressAllowlist across the
    control→node wire (cidrs, ports, dns_resolver)."""
    from xrlenv.backends.egress import EgressAllowlist, EgressRule

    agent, transport, _link, _task = linked_pair
    al = EgressAllowlist(
        rules=(
            EgressRule(cidr="3.149.157.52/32"),
            EgressRule(cidr="18.225.81.238/32", ports=(443,)),
        ),
    )
    await transport.apply_egress(
        rollout_id="r-eg", container_id="cid-9",
        allowlist=al, dns_resolver="10.0.0.2/32",
    )
    assert agent.apply_egress_calls == 1
    got = agent.last_apply_egress
    assert got is not None
    assert got["rollout_id"] == "r-eg"
    assert got["container_id"] == "cid-9"
    assert got["dns_resolver"] == "10.0.0.2/32"
    assert got["allowlist"] == al  # reconstructed equal to what was sent


async def test_apply_egress_empty_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """An empty allowlist (block-all) round-trips as empty, not an error."""
    from xrlenv.backends.egress import EgressAllowlist

    agent, transport, _link, _task = linked_pair
    await transport.apply_egress(
        rollout_id="r-eg", container_id="cid-9", allowlist=EgressAllowlist(),
    )
    assert agent.last_apply_egress["allowlist"] == EgressAllowlist()  # type: ignore[index]


async def test_remote_failure_propagates_as_xrlenverror(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """If the FakeAgent raises, the reply carries FAILED + the transport
    raises XRLEnvError on the calling side.
    """
    from xrlenv.errors import XRLEnvError

    agent, transport, _link, _task = linked_pair

    async def boom(**_: Any) -> SandboxHandle:
        raise RuntimeError("create blew up on the node")

    agent.create_sandbox = boom  # type: ignore[method-assign]

    with pytest.raises(XRLEnvError, match="create blew up on the node"):
        await transport.create_sandbox(
            rollout_id="r-4",
            backend="docker",
            template=TemplateRef(name="t", image="im:1"),
            resources=ResourceSpec(
                cpu_request=0.25,
                cpu_limit=1.0,
                mem_request_bytes=64_000_000,
                mem_limit_bytes=128_000_000,
                disk_request_bytes=64_000_000,
            ),
            network_policy="open",
        )


async def test_stats_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Slice 3.5: StatsRequest carries a ResourceUsage snapshot back to the
    control plane.
    """
    _agent, transport, _link, _task = linked_pair
    sb = SandboxHandle(id="sb-x", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    usage = await transport.stats(sb)
    assert usage.cpu_seconds == 0.0
    assert usage.rss_bytes == 0


async def test_run_in_sandbox_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Slice 3.5: RunInSandboxCommand round-trip closes audit M1."""
    agent, transport, _link, _task = linked_pair
    sb = SandboxHandle(id="sb-x", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")

    async def fake_run(
        sb_arg: SandboxHandle,
        cmd: list[str],
        *,
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Any:
        from xrlenv.backends.base import ExecResult

        return ExecResult(
            exit_code=0,
            stdout=f"ran: {' '.join(cmd)}".encode(),
            stderr=b"",
            timed_out=False,
        )

    agent.run_in_sandbox = fake_run  # type: ignore[attr-defined]

    result = await transport.run_in_sandbox(
        sb, ["echo", "hi"], timeout_s=5.0, cwd="/sandbox", env={"K": "v"}
    )
    assert result.exit_code == 0
    assert result.stdout == b"ran: echo hi"
    assert result.timed_out is False


# ──────────────────────────────────────────────────────────────────────────────
# A3 / D15 — ListSandboxesCommand round trip (spec 09 GC layer 3)
# ──────────────────────────────────────────────────────────────────────────────


async def test_create_sandbox_carries_stub_request_timeout_s_over_wire(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Audit response (H2): the per-sandbox HTTP cap added to
    ``CreateSandboxCommand`` round-trips end-to-end so the node-side
    NodeAgent can stage it on the record before any stub-touching
    call. Pin both the populated case and the zero-sentinel
    (None → 0.0 on the wire → None on the node).
    """
    agent, transport, _link, _task = linked_pair

    # Populated case.
    await transport.create_sandbox(
        rollout_id="r-1",
        backend="docker",
        template=TemplateRef(name="t", image="im:1", digest="sha256:t"),
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",
        stub_request_timeout_s=42.5,
    )
    assert agent.last_stub_request_timeout_s == 42.5

    # Unset case (None → 0.0 sentinel → None on receive).
    await transport.create_sandbox(
        rollout_id="r-2",
        backend="docker",
        template=TemplateRef(name="t", image="im:1", digest="sha256:t"),
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",
    )
    assert agent.last_stub_request_timeout_s is None


async def test_list_sandbox_ids_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """End-to-end: control plane sends ListSandboxesCommand, node
    replies with the IDs from its in-memory sandbox table. Pins
    both the proto wire shape and the per-call UUID idempotency
    contract — back-to-back calls must each reach the node, not
    return a cached prior reply (otherwise the reconciler would
    miss live state changes).
    """
    agent, transport, _link, _task = linked_pair

    agent._sandbox_ids = ["sb-1", "sb-2", "sb-3"]
    ids = await transport.list_sandbox_ids()
    assert sorted(ids) == ["sb-1", "sb-2", "sb-3"]

    # Mutate node-side state and call again — the cache MUST NOT
    # absorb the second call (per-call UUID in idempotency_key).
    agent._sandbox_ids = ["sb-2"]
    ids2 = await transport.list_sandbox_ids()
    assert ids2 == ["sb-2"]


async def test_query_image_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """A1 / D18+D19 (P1.2) — wire round-trip for QueryImageCommand /
    QueryImageReply. Pin: presence + digest + last_used_at survive
    the proto encoding; per-call UUID idempotency means back-to-back
    calls observe live state, not cached prior replies; absent
    image returns present=False with empty digest + 0.0 last_used.
    """
    from xrlenv.node.image_cache import ImageQueryResult

    agent, transport, _link, _task = linked_pair

    agent._image_query_replies = {
        "registry/foo:1": ImageQueryResult(
            present=True,
            digest="sha256:" + "ab" * 32,
            last_used_at=42.5,
        ),
    }
    result_present = await transport.query_image("registry/foo:1")
    assert result_present.present is True
    assert result_present.digest == "sha256:" + "ab" * 32
    assert result_present.last_used_at == 42.5

    # Different image → reply absent. Per-call UUID means the cache
    # cannot return the prior call's hit.
    result_absent = await transport.query_image("never/seen:1")
    assert result_absent.present is False
    assert result_absent.digest is None
    assert result_absent.last_used_at == 0.0


async def test_list_sandbox_ids_empty_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Empty reply is the steady-state for a freshly-bootstrapped
    node — must round-trip as ``[]``, not ``None``."""
    agent, transport, _link, _task = linked_pair
    agent._sandbox_ids = []
    assert await transport.list_sandbox_ids() == []


# ──────────────────────────────────────────────────────────────────────────────
# Slice 7b — FetchTrajectoryCommand round trip (spec 17)
# ──────────────────────────────────────────────────────────────────────────────


async def test_fetch_trajectory_whole_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """End-to-end: control plane sends FetchTrajectoryCommand, node returns
    the full body, and the bytes round-trip through Trajectory.model_dump_json
    + model_validate_json without losing precision."""
    _agent, transport, _link, _task = linked_pair
    traj = await transport.fetch_trajectory("r-1")
    assert traj.rollout_id == "r-1"
    assert traj.status.value == "finished"
    assert len(traj.steps) == 3


async def test_fetch_trajectory_summary_only(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    _agent, transport, _link, _task = linked_pair
    traj = await transport.fetch_trajectory("r-2", range_kind="summary_only")
    assert traj.rollout_id == "r-2"
    assert traj.steps == []


async def test_fetch_trajectory_step_range_slices(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    _agent, transport, _link, _task = linked_pair
    traj = await transport.fetch_trajectory(
        "r-3", range_kind="step_range", step_start=1, step_end=3,
    )
    assert [s.index for s in traj.steps] == [1, 2]


async def test_fetch_trajectory_missing_returns_failed_reply(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Spec 17 ReplayUnavailable: a node that has no body for the rollout
    must reply with FAILED + error_kind=ReplayUnavailable, surfaced as a
    RemoteCommandError on the control side."""
    from xrlenv.errors import XRLEnvError

    agent, transport, _link, _task = linked_pair

    async def _missing(*_a: object, **_kw: object) -> Any:
        raise FileNotFoundError("no run dir for r-missing")

    agent.fetch_trajectory = _missing  # type: ignore[attr-defined]
    with pytest.raises(XRLEnvError, match="ReplayUnavailable"):
        await transport.fetch_trajectory("r-missing")


# ──────────────────────────────────────────────────────────────────────────────
# Converter round-trip (no gRPC)
# ──────────────────────────────────────────────────────────────────────────────


def test_converters_round_trip_template_ref() -> None:
    from xrlenv.api import converters as conv

    original = TemplateRef(name="hello-shell", image="xrlenv/hello-shell:0.1", digest="sha256:foo")
    back = conv.template_ref_from_proto(conv.template_ref_to_proto(original))
    assert back == original


def test_converters_round_trip_resource_spec() -> None:
    from xrlenv.api import converters as conv
    from xrlenv.backends.base import MountSpec

    original = ResourceSpec(
        cpu_request=0.5,
        cpu_limit=2.0,
        mem_request_bytes=128_000_000,
        mem_limit_bytes=512_000_000,
        disk_request_bytes=8_000_000_000,
        gpu_required=False,
        mounts=(MountSpec(host_path="/h", sandbox_path="/s", readonly=True),),
    )
    back = conv.resource_spec_from_proto(conv.resource_spec_to_proto(original))
    assert back == original


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-2a — isolation capability + pinned-CPU accounting on the wire
# ──────────────────────────────────────────────────────────────────────────────


def test_nodehello_isolation_capable_wire_round_trip() -> None:
    """NodeHello.isolation_capable (field 11) survives serialization; proto3
    default (a pre-2a agent) reads False."""
    from xrlenv.api._pb2 import node_control_pb2 as pb

    back = pb.NodeHello.FromString(
        pb.NodeHello(node_id="n", isolation_capable=True).SerializeToString(),
    )
    assert back.isolation_capable is True
    assert pb.NodeHello().isolation_capable is False


def test_heartbeat_pinned_cpus_wire_round_trip() -> None:
    """Heartbeat.pinned_cpus_free/total (fields 8/9) survive serialization;
    proto3 default (a pre-2a agent) reads 0/0 = 'unknown'."""
    from xrlenv.api._pb2 import node_control_pb2 as pb

    back = pb.Heartbeat.FromString(
        pb.Heartbeat(pinned_cpus_free=6, pinned_cpus_total=8).SerializeToString(),
    )
    assert (back.pinned_cpus_free, back.pinned_cpus_total) == (6, 8)
    assert (pb.Heartbeat().pinned_cpus_free, pb.Heartbeat().pinned_cpus_total) == (0, 0)


@pytest.mark.asyncio
async def test_nodehello_isolation_capable_reaches_control_transport(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """End-to-end: the node advertises isolation_capable on NodeHello and the
    control plane stores it on the node transport (node build → wire → control),
    proving the step-2a plumbing without any scheduling behavior."""
    _agent, transport, _link, _task = linked_pair
    assert transport.isolation_capable() is True


async def test_env_setup_step_teardown_per_call_cap_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """A5 / D17 stage 2 (P1.2.b): the per-call ``request_timeout_s``
    kwarg the coordinator passes to ``RemoteNodeTransport`` must
    arrive at the node-side dispatcher and reach NodeAgent's matching
    method. Pinned by exercising all three env_* RPCs with distinct
    cap values; the FakeAgent records the last cap each method saw,
    proving the proto field flowed through ``EnvSetupCommand`` /
    ``EnvStepCommand`` / ``EnvTeardownCommand`` and back into the
    Python-side kwarg.
    """
    agent, transport, _link, _task = linked_pair
    sb = await transport.create_sandbox(
        rollout_id="r-cap",
        backend="docker",
        template=TemplateRef(name="t", image="im:1"),
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",
    )
    await transport.env_setup(
        sb,
        adapter_module="m",
        adapter_class="C",
        init_params={"max_steps": 1},
        request_timeout_s=11.5,
    )
    await transport.env_step(sb, {"cmd": "noop"}, request_timeout_s=22.5)
    await transport.env_teardown(sb, request_timeout_s=33.5)

    assert agent.last_setup_request_timeout_s == 11.5
    assert agent.last_step_request_timeout_s == 22.5
    assert agent.last_teardown_request_timeout_s == 33.5


async def test_env_calls_without_per_call_cap_decode_to_none(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """When the control plane omits ``request_timeout_s``, the proto
    field is encoded as the proto3 default 0.0; the node-side decoder
    (``_per_call_cap``) maps that back to ``None`` so the per-sandbox
    stage-1 cap continues to apply (no per-call override). Ensures
    backward compatibility for old control planes that never set the
    field.
    """
    agent, transport, _link, _task = linked_pair
    sb = await transport.create_sandbox(
        rollout_id="r-default",
        backend="docker",
        template=TemplateRef(name="t", image="im:1"),
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",
    )
    await transport.env_setup(
        sb, adapter_module="m", adapter_class="C", init_params={"max_steps": 1},
    )
    await transport.env_step(sb, {"cmd": "noop"})
    await transport.env_teardown(sb)

    assert agent.last_setup_request_timeout_s is None
    assert agent.last_step_request_timeout_s is None
    assert agent.last_teardown_request_timeout_s is None


# ──────────────────────────────────────────────────────────────────────────────
# B7.6 (P1.2.c) — ReportImagesCommand wire round-trip
# ──────────────────────────────────────────────────────────────────────────────


async def test_report_images_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """B7.6: a full ``NodeImageReport`` round-trips through the
    ``ReportImagesCommand`` proto, the node-side dispatcher, and the
    control-side ``RemoteNodeTransport.report_images`` decoder. Pin
    every observable field — image list (with tier / size / refcount /
    last_used / pinned), free disk, operator pin set — so a future
    proto reshape that drops a field surfaces here.
    """
    from xrlenv.node.image_cache import (
        ImageStateRecord,
        NodeImageReport,
    )

    agent, transport, _link, _task = linked_pair
    agent._image_report = NodeImageReport(
        images=[
            ImageStateRecord(
                name="bench/task-a:1", tier="in_use",
                size_bytes=2 * 1024**3, in_use_count=1,
                last_used_at=123.5, pinned=False,
                digest=f"bench/task-a@sha256:{'a' * 64}",
            ),
            ImageStateRecord(
                # digest unset → "" on the wire → decoded back to None.
                name="bench-base/task-a:1", tier="cold",
                size_bytes=8 * 1024**3, in_use_count=0,
                last_used_at=None, pinned=False,
            ),
            ImageStateRecord(
                name="ops/sidecar:1", tier="pinned",
                size_bytes=1 * 1024**3, in_use_count=0,
                last_used_at=42.0, pinned=True,
            ),
        ],
        free_disk_bytes=20 * 1024**3,
        pinned=("ops/sidecar:1",),
    )

    report = await transport.report_images()

    by_name = {img.name: img for img in report.images}
    assert by_name["bench/task-a:1"].tier == "in_use"
    assert by_name["bench/task-a:1"].size_bytes == 2 * 1024**3
    assert by_name["bench/task-a:1"].in_use_count == 1
    assert by_name["bench/task-a:1"].last_used_at == 123.5
    assert by_name["bench-base/task-a:1"].tier == "cold"
    # last_used_at=None on the wire encodes as 0.0; decoder maps back to None.
    assert by_name["bench-base/task-a:1"].last_used_at is None
    assert by_name["ops/sidecar:1"].pinned is True
    # repo_digest plumbing: a set digest survives the round-trip; an unset one
    # encodes as "" on the wire and decodes back to None (calibrate's
    # digest-match fallback keys on this).
    assert by_name["bench/task-a:1"].digest == f"bench/task-a@sha256:{'a' * 64}"
    assert by_name["bench-base/task-a:1"].digest is None
    assert report.free_disk_bytes == 20 * 1024**3
    assert report.pinned == ("ops/sidecar:1",)


async def test_report_images_round_trip_empty(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Empty cache → empty report. Pins the no-images / zero-disk
    edge case so it doesn't silently raise on the decode path.
    """
    _agent, transport, _link, _task = linked_pair
    # _image_report stays None → fake returns NodeImageReport().

    report = await transport.report_images()

    assert report.images == []
    assert report.free_disk_bytes == 0
    assert report.pinned == ()


def test_converters_round_trip_hardware_info() -> None:
    from xrlenv.api import converters as conv

    original = _hw()
    back = conv.hardware_info_from_proto(conv.hardware_info_to_proto(original))
    assert back == original


# ──────────────────────────────────────────────────────────────────────────────
# BuildImageCommand: tarball wire round trip (sub-slice 1.b)
# ──────────────────────────────────────────────────────────────────────────────


async def test_build_image_tarball_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """A TarballSource with content_b64 round-trips end-to-end via
    real grpc.aio loopback. The node-side handler reconstructs a
    TarballSource from the wire bytes and dispatches to the
    GitSourceBuilder; we patch the builder so the test doesn't need
    a docker daemon.
    """
    import base64
    from unittest.mock import AsyncMock

    from xrlenv.control.build_plan import TarballSource
    from xrlenv.node.source_builder import GitSourceBuilder

    _agent, transport, link, _task = linked_pair

    # Inject a fake builder so the dispatch path doesn't need
    # docker. Capture the source the builder receives so we can
    # assert on the bytes round-trip.
    captured: list[Any] = []

    async def _fake_build(
        *, image_ref: str, source: Any, timeout_s: float,
        labels: dict[str, str], skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        captured.append((image_ref, source))
        return ("ok", None)

    fake_builder = AsyncMock(spec=GitSourceBuilder)
    fake_builder.build = _fake_build
    link._source_builder = fake_builder  # type: ignore[attr-defined]

    payload = b"<tarball-payload-content>"
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(payload).decode("ascii"),
    )
    status, error = await transport.build_image(
        image_ref="my/from-tar:1", source=source,
        timeout_s=60.0, labels={"team": "ops"},
    )
    assert status == "ok"
    assert error == ""

    # The bytes survived: wire encode (control plane) → bytes on
    # CommandReply payload → re-encode (node side) → schema decode
    # in the builder.
    assert len(captured) == 1
    received_ref, received_source = captured[0]
    assert received_ref == "my/from-tar:1"
    assert isinstance(received_source, TarballSource)
    assert received_source.content_b64 is not None
    assert base64.b64decode(received_source.content_b64) == payload


# ──────────────────────────────────────────────────────────────────────────────
# CancelBuildImageCommand wire round trip
# ──────────────────────────────────────────────────────────────────────────────


async def test_cancel_build_image_round_trip_with_no_builder(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Cancel against a node that has no GitSourceBuilder (no build
    has ever been dispatched on this link) returns ``ok`` — operator-
    idempotent at the wire level. Pins the wire shape end-to-end."""
    _agent, transport, _link, _task = linked_pair

    status, error = await transport.cancel_build_image(image_ref="never/built:1")
    assert status == "ok"
    assert error == ""


async def test_cancel_build_image_round_trip_with_builder(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """Cancel against a node WITH a GitSourceBuilder (some build was
    dispatched earlier on this link) round-trips through the builder's
    cancel method. Uses a fake docker client so the no-active-build
    path returns ``ok`` cleanly without contacting the daemon."""
    _agent, transport, link, _task = linked_pair

    from unittest.mock import MagicMock

    from xrlenv.node.source_builder import GitSourceBuilder

    fake_docker = MagicMock()
    fake_docker.containers.list = MagicMock(return_value=[])
    builder = GitSourceBuilder(docker_client=fake_docker)
    # Mutate the link so the cancel handler finds an existing builder
    # (parallel to what _exec_build_image lazy-creates).
    link._source_builder = builder  # type: ignore[attr-defined]

    status, error = await transport.cancel_build_image(image_ref="absent:1")
    assert status == "ok"
    assert error == ""
    # The builder's docker client was queried — cancel went all the
    # way through to the kill path.
    fake_docker.containers.list.assert_called_once()




# ── Issue #14 — heartbeat samples disk state from the image cache ────────────


async def test_sample_disk_state_returns_zero_when_no_image_cache() -> None:
    """A fake agent with no ``_image_cache`` attribute (legacy fixture)
    must return the documented ``(0, 0)`` "unknown" sentinel rather
    than crashing. The control-plane gate treats unknown as healthy."""
    from xrlenv.node.grpc_link import NodeGrpcLink

    agent = FakeAgent()
    # FakeAgent doesn't set _image_cache; the sampler must tolerate that.
    link = NodeGrpcLink(agent, control_addr="127.0.0.1:0")
    free, total = await link._sample_disk_state()
    assert (free, total) == (0, 0)


async def test_sample_disk_state_reads_from_backend_when_cache_wired() -> None:
    """When an ImageCacheManager is wired, the sampler reads
    free / total from its backend so the heartbeat carries fresh
    numbers for the placement gate + admin pressure indicator."""
    from xrlenv.node.grpc_link import NodeGrpcLink

    class _Backend:
        async def free_disk_bytes(self) -> int:
            return 42 * 1024**3

        async def total_disk_bytes(self) -> int:
            return 200 * 1024**3

    class _Cache:
        _backend = _Backend()

    agent = FakeAgent()
    agent._image_cache = _Cache()  # type: ignore[attr-defined]

    link = NodeGrpcLink(agent, control_addr="127.0.0.1:0")
    free, total = await link._sample_disk_state()
    assert free == 42 * 1024**3
    assert total == 200 * 1024**3


async def test_sample_disk_state_swallows_backend_exception() -> None:
    """A transient daemon hiccup (docker info raises mid-tick) must
    not block the heartbeat — the sampler returns ``(0, 0)`` and the
    next tick tries again. Heartbeats keep the node alive in the
    registry; blocking them on a flaky disk probe would cause
    spurious node-lost events."""
    from xrlenv.node.grpc_link import NodeGrpcLink

    class _Backend:
        async def free_disk_bytes(self) -> int:
            raise RuntimeError("daemon hiccup")

        async def total_disk_bytes(self) -> int:
            raise RuntimeError("daemon hiccup")

    class _Cache:
        _backend = _Backend()

    agent = FakeAgent()
    agent._image_cache = _Cache()  # type: ignore[attr-defined]

    link = NodeGrpcLink(agent, control_addr="127.0.0.1:0")
    assert (await link._sample_disk_state()) == (0, 0)


# ── Issue #18 — heartbeat decoupled from the inline disk probe ───────────────


async def test_disk_sample_loop_populates_cached_state() -> None:
    """The standalone disk-sample task refreshes the cached
    ``_disk_state`` that the heartbeat loop reads."""

    class _Backend:
        async def free_disk_bytes(self) -> int:
            return 7 * 1024**3

        async def total_disk_bytes(self) -> int:
            return 99 * 1024**3

    class _Cache:
        _backend = _Backend()

    agent = FakeAgent()
    agent._image_cache = _Cache()  # type: ignore[attr-defined]
    link = NodeGrpcLink(
        agent, control_addr="127.0.0.1:0", heartbeat_interval_s=0.01,
    )
    assert link._disk_state == (0, 0)

    task = asyncio.create_task(link._disk_sample_loop())
    try:
        for _ in range(200):
            if link._disk_state != (0, 0):
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert link._disk_state == (7 * 1024**3, 99 * 1024**3)


async def test_heartbeat_uses_cached_disk_state_not_inline_probe() -> None:
    """Issue #18: the heartbeat reads the cached ``_disk_state``; it
    must NOT probe the docker daemon inline. A slow ``docker info``
    (the failure mode that false-flagged li-5 `lost` under cold-pull
    load) therefore can't delay a beat past the watchdog grace."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import _MonotonicCounter

    agent = FakeAgent()
    agent._sandboxes = {}  # type: ignore[attr-defined]
    link = NodeGrpcLink(
        agent, control_addr="127.0.0.1:0", heartbeat_interval_s=0.01,
    )
    link._disk_state = (123, 456)

    async def _boom() -> tuple[int, int]:
        raise AssertionError("heartbeat must not probe disk inline")

    link._sample_disk_state = _boom  # type: ignore[method-assign]

    outbox: asyncio.Queue[pb.NodeMsg] = asyncio.Queue()
    task = asyncio.create_task(
        link._heartbeat_loop(outbox, "ep-1", _MonotonicCounter()),
    )
    try:
        msg = await asyncio.wait_for(outbox.get(), timeout=2.0)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert msg.HasField("heartbeat")
    assert msg.heartbeat.free_disk_bytes == 123
    assert msg.heartbeat.total_disk_bytes == 456


# ── §5.3 NodeHello docker-ready gate — don't advertise runtimes too early ────


class _RuntimeProbeAgent:
    """Minimal agent stub for the docker-ready gate: ``probe_docker_runtimes_ready``
    returns False for the first ``ready_after`` calls, then True (simulating a
    daemon that becomes enumerable a few probes after the agent starts)."""

    node_id = "probe-node"

    def __init__(self, *, ready_after: int) -> None:
        self._ready_after = ready_after
        self.probe_calls = 0

    def supported_backends(self) -> list[str]:
        return ["docker"]

    def probe_docker_runtimes_ready(self) -> bool:
        self.probe_calls += 1
        return self.probe_calls > self._ready_after


@pytest.mark.asyncio
async def test_await_docker_ready_waits_until_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate polls until docker answers, so the NodeHello enumerates the real
    runtime set instead of a startup-race fallback."""
    from xrlenv.node import grpc_link as gl

    monkeypatch.setattr(gl, "_DOCKER_READY_INTERVAL_S", 0.001)
    monkeypatch.setattr(gl, "_DOCKER_READY_TIMEOUT_S", 5.0)
    agent = _RuntimeProbeAgent(ready_after=3)  # first 3 probes False
    link = gl.NodeGrpcLink(agent, control_addr="x")  # type: ignore[arg-type]

    await link._await_docker_ready()

    assert agent.probe_calls == 4  # 3 not-ready + 1 ready


@pytest.mark.asyncio
async def test_await_docker_ready_gives_up_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If docker never answers, the gate returns after the timeout (proceeding
    with the conservative set) rather than hanging the connect forever."""
    from xrlenv.node import grpc_link as gl

    monkeypatch.setattr(gl, "_DOCKER_READY_INTERVAL_S", 0.001)
    monkeypatch.setattr(gl, "_DOCKER_READY_TIMEOUT_S", 0.02)
    agent = _RuntimeProbeAgent(ready_after=10**9)  # never ready
    link = gl.NodeGrpcLink(agent, control_addr="x")  # type: ignore[arg-type]

    await link._await_docker_ready()  # must return, not hang

    assert agent.probe_calls >= 1


@pytest.mark.asyncio
async def test_await_docker_ready_noop_for_agent_without_probe() -> None:
    """An agent/test-double predating the probe method advertises as-is — the
    gate is a no-op (backward compatible)."""
    from xrlenv.node import grpc_link as gl

    class _OldAgent:
        node_id = "old"

        def supported_backends(self) -> list[str]:
            return ["docker"]

    link = gl.NodeGrpcLink(_OldAgent(), control_addr="x")  # type: ignore[arg-type]
    await link._await_docker_ready()  # no attribute, no exception


# ── Issue #12 audit M1 — pull_deadline_s flows through dispatcher ────────────


async def test_exec_acquire_container_threads_pull_deadline_to_agent() -> None:
    """Issue #12 audit M1: the proto ``pull_deadline_s`` field must
    flow from the control plane's ``AcquireContainerCommand`` all the
    way to ``NodeAgent.acquire_container(acquire_timeout_s=...)`` so
    the node-side ``ImageCacheManager.ensure_present`` widens to
    match the wire wait. Otherwise the override is half-wired and the
    pull fails at 600 s while the wire patiently waits.
    """
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    captured: dict[str, Any] = {}

    class _StubAgent(FakeAgent):
        async def acquire_container(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)

            class _R:
                container_id = "ci"
                container_name = "cn"

            return _R()

    link = NodeGrpcLink(_StubAgent(), control_addr="127.0.0.1:0")
    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-1"),
        rollout_id="r-1",
        image="huge:1",
        pull_deadline_s=1800.0,
    )
    reply = await link._exec_acquire_container(cmd)
    assert reply.status == pb.ReplyStatus.OK
    assert captured["acquire_timeout_s"] == 1800.0


async def test_exec_acquire_container_passes_none_when_pull_deadline_unset() -> None:
    """Proto3 default ``0.0`` is the "use node default" sentinel — must
    map to ``acquire_timeout_s=None`` so the cache applies its config'd
    default instead of trying to honor a zero-second deadline."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    captured: dict[str, Any] = {}

    class _StubAgent(FakeAgent):
        async def acquire_container(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)

            class _R:
                container_id = "ci"
                container_name = "cn"

            return _R()

    link = NodeGrpcLink(_StubAgent(), control_addr="127.0.0.1:0")
    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-1"),
        rollout_id="r-1",
        image="busybox:1",
        # pull_deadline_s deliberately unset — proto3 default 0.0.
    )
    await link._exec_acquire_container(cmd)
    assert captured["acquire_timeout_s"] is None


async def test_exec_acquire_container_threads_resources_to_agent() -> None:
    """P1 — AcquireContainerCommand.resources flows to
    NodeAgent.acquire_container(resources=...) so the node applies
    cpu/memory cgroup limits."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    captured: dict[str, Any] = {}

    class _StubAgent(FakeAgent):
        async def acquire_container(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)

            class _R:
                container_id = "ci"
                container_name = "cn"

            return _R()

    link = NodeGrpcLink(_StubAgent(), control_addr="127.0.0.1:0")
    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-1"),
        rollout_id="r-1",
        image="busybox:1",
        resources=pb.ResourceSpec(
            cpu_request=4.0, cpu_limit=4.0,
            mem_request_bytes=8 * 1024**3, mem_limit_bytes=8 * 1024**3,
        ),
    )
    await link._exec_acquire_container(cmd)
    assert captured["resources"] is not None
    assert captured["resources"].cpu_limit == 4.0
    assert captured["resources"].mem_limit_bytes == 8 * 1024**3


async def test_exec_acquire_container_resources_unset_passes_none() -> None:
    """P1 — an unset resources field maps to ``None``; the manager then
    applies its node-default cap rather than trying to honor an
    all-zero ResourceSpec."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    captured: dict[str, Any] = {}

    class _StubAgent(FakeAgent):
        async def acquire_container(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)

            class _R:
                container_id = "ci"
                container_name = "cn"

            return _R()

    link = NodeGrpcLink(_StubAgent(), control_addr="127.0.0.1:0")
    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-1"),
        rollout_id="r-1",
        image="busybox:1",
        # resources deliberately unset.
    )
    await link._exec_acquire_container(cmd)
    assert captured["resources"] is None


async def test_exec_acquire_container_threads_runtime_limits_to_agent() -> None:
    """P0b — AcquireContainerCommand.runtime_limits flows to
    NodeAgent.acquire_container(runtime_limits=...) so the node applies
    the container-shape limits."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    captured: dict[str, Any] = {}

    class _StubAgent(FakeAgent):
        async def acquire_container(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)

            class _R:
                container_id = "ci"
                container_name = "cn"

            return _R()

    link = NodeGrpcLink(_StubAgent(), control_addr="127.0.0.1:0")
    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-1"),
        rollout_id="r-1",
        image="busybox:1",
        runtime_limits=pb.RuntimeLimits(
            pids_limit=2048, shm_size_bytes=33554432, readonly_rootfs=True,
            cpu_pinning=True,
        ),
    )
    await link._exec_acquire_container(cmd)
    rl = captured["runtime_limits"]
    assert rl is not None
    assert rl.pids_limit == 2048
    assert rl.shm_size_bytes == 33554432
    assert rl.readonly_rootfs is True
    assert rl.cpu_pinning is True


async def test_exec_acquire_container_runtime_limits_unset_passes_none() -> None:
    """P0b — an unset runtime_limits field maps to ``None``."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    captured: dict[str, Any] = {}

    class _StubAgent(FakeAgent):
        async def acquire_container(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)

            class _R:
                container_id = "ci"
                container_name = "cn"

            return _R()

    link = NodeGrpcLink(_StubAgent(), control_addr="127.0.0.1:0")
    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-1"),
        rollout_id="r-1",
        image="busybox:1",
    )
    await link._exec_acquire_container(cmd)
    assert captured["runtime_limits"] is None


# ──────────────────────────────────────────────────────────────────────────────
# RegisterScratchSourceCommand wire round trip (scratch build-on-demand, 2c-iii)
# ──────────────────────────────────────────────────────────────────────────────


async def test_register_scratch_source_git_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
    tmp_path: Path,
) -> None:
    """A GitSource + durable_to round-trips control → wire → node handler,
    which records it on the real (docker-free) GitSourceBuilder."""
    from xrlenv.control.build_plan import GitSource
    from xrlenv.node.source_builder import GitSourceBuilder

    _agent, transport, link, _task = linked_pair
    builder = GitSourceBuilder(cache_root=tmp_path / "cache")
    link._source_builder = builder  # type: ignore[attr-defined]

    src = GitSource(repo="https://x/y", ref="abc123", subdir="env", dockerfile="Dockerfile")
    ref = "cp:5012/scratch/deadbeef"
    await transport.register_scratch_source(ref, src, durable_to="reg:5000/team/env:v1")

    stored = builder._scratch_specs.get(ref)
    assert isinstance(stored, GitSource)
    assert (stored.repo, stored.ref, stored.subdir) == ("https://x/y", "abc123", "env")
    assert builder._scratch_durable.get(ref) == "reg:5000/team/env:v1"


async def test_register_scratch_source_tarball_round_trip(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
    tmp_path: Path,
) -> None:
    """A TarballSource's bytes survive the wire; no durable_to → not recorded."""
    import base64

    from xrlenv.control.build_plan import TarballSource
    from xrlenv.node.source_builder import GitSourceBuilder

    _agent, transport, link, _task = linked_pair
    builder = GitSourceBuilder(cache_root=tmp_path / "cache")
    link._source_builder = builder  # type: ignore[attr-defined]

    payload = b"<scratch-context-tarball>"
    src = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(payload).decode("ascii"),
    )
    ref = "cp:5012/scratch/cafef00d"
    await transport.register_scratch_source(ref, src)

    stored = builder._scratch_specs.get(ref)
    assert isinstance(stored, TarballSource)
    assert stored.content_b64 is not None
    assert base64.b64decode(stored.content_b64) == payload
    assert ref not in builder._scratch_durable  # no durable_to


async def test_register_scratch_source_failure_raises(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """A node-side registration failure surfaces as XRLEnvError to the caller."""
    from xrlenv.control.build_plan import GitSource
    from xrlenv.errors import XRLEnvError

    _agent, transport, link, _task = linked_pair

    class _RaisingBuilder:
        def register_scratch_source(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("disk full")

    link._source_builder = _RaisingBuilder()  # type: ignore[attr-defined]
    with pytest.raises(XRLEnvError, match="failed to register scratch source"):
        await transport.register_scratch_source(
            "cp:5012/scratch/x", GitSource(repo="https://x/y", ref="main"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# build+push node handler (c1fb9e9 — native distributed build+push)
# ──────────────────────────────────────────────────────────────────────────────


async def test_exec_build_image_push_true_returns_repo_digest(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """``push=True`` on BuildImageCommand routes to ``build_and_push`` and
    the resulting repo_digest propagates back through the reply."""
    from unittest.mock import AsyncMock

    from xrlenv.control.build_plan import GitSource
    from xrlenv.node.source_builder import BuildAndPushResult, GitSourceBuilder

    _agent, transport, link, _task = linked_pair

    fake_builder = AsyncMock(spec=GitSourceBuilder)
    fake_builder.build_and_push = AsyncMock(
        return_value=BuildAndPushResult(
            "ok", None, "registry.example.com/team/env@sha256:abc123",
        ),
    )
    link._source_builder = fake_builder  # type: ignore[attr-defined]

    source = GitSource(
        repo="https://github.com/example/repo",
        ref="main",
        subdir=".",
        dockerfile="Dockerfile",
    )
    status, error, repo_digest = await transport.build_and_push_image(
        image_ref="registry.example.com/team/env:v1",
        source=source,
        timeout_s=300.0,
        labels={"built_by": "xrlenv"},
    )

    assert status == "ok"
    assert error == ""
    assert repo_digest == "registry.example.com/team/env@sha256:abc123"
    # build_and_push was called with check_registry_first=True (build-once semantics).
    fake_builder.build_and_push.assert_awaited_once()
    call_kwargs = fake_builder.build_and_push.call_args
    assert call_kwargs.kwargs.get("check_registry_first") is True
    # The plain build() path must NOT have been touched.
    fake_builder.build.assert_not_called()


async def test_exec_build_image_push_true_failing_build_and_push(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """A ``build_and_push`` that returns status='failed' propagates
    status='failed' and repo_digest is empty / None in the reply."""
    from unittest.mock import AsyncMock

    from xrlenv.control.build_plan import GitSource
    from xrlenv.node.source_builder import BuildAndPushResult, GitSourceBuilder

    _agent, transport, link, _task = linked_pair

    fake_builder = AsyncMock(spec=GitSourceBuilder)
    fake_builder.build_and_push = AsyncMock(
        return_value=BuildAndPushResult("failed", "docker daemon unreachable", None),
    )
    link._source_builder = fake_builder  # type: ignore[attr-defined]

    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    status, error, repo_digest = await transport.build_and_push_image(
        image_ref="registry.example.com/team/env:v1",
        source=source, timeout_s=300.0,
    )

    assert status == "failed"
    # repo_digest is empty string or None (proto default → empty, transport
    # returns None for falsy empty string).
    assert not repo_digest  # None or ""


async def test_exec_build_image_push_false_calls_build_not_build_and_push(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """``push=False`` (the default ``build_image`` path) still calls the
    plain ``build`` method; ``build_and_push`` must never be touched."""
    from unittest.mock import AsyncMock

    from xrlenv.control.build_plan import GitSource
    from xrlenv.node.source_builder import GitSourceBuilder

    _agent, transport, link, _task = linked_pair

    async def _fake_build(
        *, image_ref: str, source: Any, timeout_s: float,
        labels: dict[str, str], skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        return ("ok", None)

    fake_builder = AsyncMock(spec=GitSourceBuilder)
    fake_builder.build = _fake_build
    link._source_builder = fake_builder  # type: ignore[attr-defined]

    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    status, error = await transport.build_image(
        image_ref="registry.example.com/team/env:v1",
        source=source, timeout_s=60.0, labels={},
    )

    assert status == "ok"
    assert error == ""
    # repo_digest must be absent from a build_image (push=False) call —
    # verified indirectly: build_and_push was never called.
    fake_builder.build_and_push.assert_not_called()


async def test_build_and_push_image_transport_sets_push_flag_and_returns_3tuple(
    linked_pair: tuple[FakeAgent, RemoteNodeTransport, NodeGrpcLink, asyncio.Task[None]],
) -> None:
    """``transport.build_and_push_image`` sets ``push=True`` on the wire
    command and returns a 3-tuple ``(status, error, repo_digest)``."""
    from unittest.mock import AsyncMock

    from xrlenv.control.build_plan import GitSource
    from xrlenv.node.source_builder import BuildAndPushResult, GitSourceBuilder

    _agent, transport, link, _task = linked_pair

    DIGEST = "reg.internal:5000/team/env@sha256:deadbeefcafe"
    fake_builder = AsyncMock(spec=GitSourceBuilder)
    fake_builder.build_and_push = AsyncMock(
        return_value=BuildAndPushResult("ok", None, DIGEST),
    )
    link._source_builder = fake_builder  # type: ignore[attr-defined]

    result = await transport.build_and_push_image(
        image_ref="reg.internal:5000/team/env:v2",
        source=GitSource(
            repo="https://github.com/example/repo", ref="main",
            subdir=".", dockerfile="Dockerfile",
        ),
        timeout_s=120.0,
    )

    # Must be a 3-tuple: (status, error, repo_digest)
    assert len(result) == 3
    status, error, repo_digest = result
    assert status == "ok"
    assert error == ""
    assert repo_digest == DIGEST
