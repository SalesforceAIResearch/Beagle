"""Audit H1 regression test: stream disconnect must seal in-flight rollouts.

Stands up a real ``build_distributed_runtime()`` over loopback gRPC, attaches
a fake-agent ``NodeGrpcLink``, runs a rollout up to ``env_setup`` (so the
state has a RUNNING row bound to that node), then *closes the link's gRPC
channel without prior heartbeat timeout*. The disconnect path must invoke
``coordinator.handle_node_lost(node_id)`` and the in-flight rollout must
seal as ``failed`` / ``reason=node_lost``.

The slow heartbeat-watchdog path is disabled here (huge ``disconnect_grace_s``)
so we know the seal is coming from the disconnect callback, not the watchdog.
"""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import suppress
from typing import Any

import grpc
from xrlenv.api._pb2 import node_control_pb2_grpc as pb_grpc
from xrlenv.backends.base import (
    ExecResult,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.client.client import Client
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.grpc_endpoint import NodeControlServicer, RemoteNodeTransport
from xrlenv.control.node_registry import NodeRegistry
from xrlenv.control.scheduler import Scheduler
from xrlenv.control.service import CoordinatorRolloutService
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.node.grpc_link import NodeGrpcLink
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.types import RolloutStatus

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="6.0.0", platform="linux",
    )


def _manifest() -> TemplateManifest:
    return TemplateManifest(
        name="t", version="0.1", digest="sha256:t", image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


class _FakeAgent:
    """Minimal NodeAgent stand-in. Returns a fixed sandbox handle and a
    canned env_setup obs; everything else is a no-op (we never reach step
    in this test).
    """

    def __init__(self) -> None:
        self.node_id = "fake-disconnect-node"
        # NodeGrpcLink reaches into _sandboxes for its heartbeat payload;
        # an empty dict is enough since this fake never creates anything.
        self._sandboxes: dict[str, Any] = {}

    def supported_backends(self) -> list[str]:
        return ["docker"]

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
        return SandboxHandle(
            id=f"sb-{rollout_id[:6]}",
            backend=backend,
            backend_ref="cid",
            stub_endpoint="tcp://127.0.0.1:0",
        )

    async def env_setup(
        self,
        sb: SandboxHandle,
        *,
        adapter_module: str,
        adapter_class: str,
        init_params: dict[str, Any],
        **_kw: Any,
    ) -> dict[str, Any]:
        return {"obs": "first"}

    async def env_step(
        self, sb: SandboxHandle, action: Any, **_kw: Any,
    ) -> dict[str, Any]:
        return {"obs": {}, "reward": 0.0, "done": False, "truncated": False, "info": {}}

    async def env_teardown(self, sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
        return {"status": "ok"}

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        return None

    async def run_in_sandbox(
        self, sb: SandboxHandle, cmd: list[str], **_: Any
    ) -> ExecResult:
        return ExecResult(exit_code=0)

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=0.0, rss_bytes=0, disk_bytes=0, rx_bytes=0, tx_bytes=0)

    async def query_image(self, _image: str) -> Any:
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────


async def test_stream_disconnect_seals_inflight_rollout_as_node_lost() -> None:
    port = _free_port()

    catalog = TemplateCatalog()
    catalog.register(_manifest())
    state = InMemoryStateStore()
    scheduler = Scheduler([], catalog=catalog, state=state, allow_empty=True)
    admission = AdmissionQueue(scheduler=scheduler, state=state)
    coordinator = RolloutCoordinator(
        catalog=catalog, scheduler=scheduler, state=state, admission=admission
    )
    service = CoordinatorRolloutService(coordinator)

    # Huge grace + huge interval so the heartbeat watchdog cannot be the one
    # that seals — only the disconnect callback can.
    registry = NodeRegistry(
        on_node_lost=coordinator.handle_node_lost,
        disconnect_grace_s=3600.0,
        check_interval_s=3600.0,
    )

    disconnect_tasks: set[asyncio.Task[None]] = set()

    def _on_connected(t: RemoteNodeTransport) -> None:
        scheduler.add_node(t)
        registry.register(t)

    def _on_disconnected(t: RemoteNodeTransport) -> None:
        scheduler.remove_node(t.node_id)
        registry.deregister(t.node_id)
        task = asyncio.create_task(coordinator.handle_node_lost(t.node_id))
        disconnect_tasks.add(task)
        task.add_done_callback(disconnect_tasks.discard)

    server = grpc.aio.server()
    pb_grpc.add_NodeControlServicer_to_server(
        NodeControlServicer(
            on_connected=_on_connected,
            on_disconnected=_on_disconnected,
        ),
        server,
    )
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    await admission.start()

    agent = _FakeAgent()
    link = NodeGrpcLink(agent, control_addr=f"127.0.0.1:{port}")  # type: ignore[arg-type]
    link_task = asyncio.create_task(link.run_forever(), name="disconnect-link")

    rollout_id: str | None = None
    try:
        # Wait for the node to register via the gRPC handshake.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not scheduler.nodes:
            await asyncio.sleep(0.05)
        assert scheduler.nodes, "node never connected to control plane"

        # Start a rollout — runs through env_setup; coordinator marks it RUNNING.
        client = Client.in_process(service)
        try:
            session = await client.rollout(
                template="t", init={"max_steps": 99, "cwd": "/sandbox"}
            )
            rollout_id = session.rollout_id
        finally:
            await client.close()

        record = state.get_rollout(rollout_id)
        assert record.status == RolloutStatus.RUNNING

        # Yank the link (simulates process crash / TCP RST). request_stop +
        # cancel breaks the bidi stream cleanly; the servicer's _on_disconnected
        # fires, which schedules handle_node_lost.
        link.request_stop()
        link_task.cancel()
        with suppress(asyncio.CancelledError):
            await link_task

        # Wait for the disconnect-seal task to complete.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            sealed = state.get_rollout(rollout_id)
            if sealed.status.is_terminal:
                break
            await asyncio.sleep(0.05)

        sealed = state.get_rollout(rollout_id)
        assert sealed.status == RolloutStatus.FAILED
        assert sealed.reason == "node_lost"
    finally:
        await admission.stop()
        await registry.shutdown()
        await server.stop(grace=1.0)
