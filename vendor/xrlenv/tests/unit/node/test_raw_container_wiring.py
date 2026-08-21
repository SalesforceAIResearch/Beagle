"""P1.7.A.1 — NodeAgent + grpc_link dispatcher wiring for the
raw-container session.

These tests verify the wiring between layers:

- ``NodeAgent.acquire_container`` / ``container_exec`` /
  ``destroy_container`` delegate to ``RawContainerManager``.
- The agent's __init__ surfaces a docker backend's ``docker_client``
  to the manager.
- ``NodeAgent._require_raw_manager`` raises ``XRLEnvError`` when no
  docker backend is registered.
- The grpc_link dispatcher methods correctly serialize / deserialize
  the spec-21 proto messages.

The algorithmic behaviour of the manager itself (label merging,
ownership rejection, exec timeout, idempotent destroy) is covered
in ``test_raw_container.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.errors import XRLEnvError
from xrlenv.node.agent import NodeAgent, NodeAgentConfig
from xrlenv.node.grpc_link import NodeGrpcLink

# Reuse the same fakes from test_raw_container.py — paste-trim is
# intentional, keeps each test file self-contained.
from tests.unit.node.test_raw_container import _ExecResult, _FakeDockerClient


class _FakeDockerBackend:
    """Minimal stand-in for ``DockerBackend`` — exposes only the
    ``docker_client`` property NodeAgent looks at via
    ``getattr(backend, "docker_client", None)`` to build raw
    managers. Doesn't pretend to be a SandboxBackend; tests that
    hit ``create_sandbox`` / etc. use the existing ``FakeBackend``
    in test_node_agent.py."""

    def __init__(self, *, images_present: set[str]) -> None:
        self._client = _FakeDockerClient(images_present=images_present)

    @property
    def docker_client(self) -> Any:
        return self._client


def _make_agent_with_docker(images: set[str]) -> tuple[NodeAgent, _FakeDockerBackend]:
    backend = _FakeDockerBackend(images_present=images)
    cfg = NodeAgentConfig(
        node_id="test-node",
        backends={"docker": backend},  # type: ignore[arg-type]
    )
    return NodeAgent(cfg), backend


# ──────────────────────────────────────────────────────────────────────────────
# NodeAgent ↔ RawContainerManager wiring
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_agent_acquire_delegates_to_manager() -> None:
    agent, backend = _make_agent_with_docker({"busybox:1"})
    record = await agent.acquire_container(
        rollout_id="r1", backend="docker", image="busybox:1",
        command=["sleep", "infinity"],
    )
    assert record.rollout_id == "r1"
    assert record.image == "busybox:1"
    container = backend.docker_client.containers.get(record.container_id)
    assert container.command == ["sleep", "infinity"]


@pytest.mark.asyncio
async def test_node_agent_container_exec_delegates_to_manager() -> None:
    agent, backend = _make_agent_with_docker({"busybox:1"})
    record = await agent.acquire_container(
        rollout_id="r1", backend="docker", image="busybox:1",
    )
    container = backend.docker_client.containers.get(record.container_id)
    container.exec_results[("echo", "hi")] = _ExecResult(
        exit_code=0, output=(b"hi\n", b""),
    )
    result = await agent.container_exec(
        rollout_id="r1", container_id=record.container_id,
        cmd=["echo", "hi"],
    )
    assert result["exit_code"] == 0
    assert result["stdout"] == b"hi\n"


@pytest.mark.asyncio
async def test_node_agent_destroy_container_delegates_to_manager() -> None:
    agent, backend = _make_agent_with_docker({"busybox:1"})
    record = await agent.acquire_container(
        rollout_id="r1", backend="docker", image="busybox:1",
    )
    await agent.destroy_container(
        rollout_id="r1", container_id=record.container_id,
    )
    # Container removed (fake registry tracks .removed).
    container = backend.docker_client.containers._registry[record.container_id]
    assert container.removed is True


@pytest.mark.asyncio
async def test_node_agent_raises_when_no_docker_backend_registered() -> None:
    """Agent without a docker-typed backend has no raw manager;
    acquire surfaces a clean XRLEnvError rather than crashing."""

    class _NoDockerBackend:
        # Deliberately no ``docker_client`` attribute.
        async def create(self, *a: Any, **kw: Any) -> None: ...
        async def destroy(self, *a: Any, **kw: Any) -> None: ...

    cfg = NodeAgentConfig(
        node_id="test-node",
        backends={"docker": _NoDockerBackend()},  # type: ignore[arg-type]
    )
    agent = NodeAgent(cfg)
    with pytest.raises(XRLEnvError, match="no raw-container manager"):
        await agent.acquire_container(
            rollout_id="r1", backend="docker", image="busybox:1",
        )


# ──────────────────────────────────────────────────────────────────────────────
# grpc_link dispatcher ↔ NodeAgent ↔ proto messages
# ──────────────────────────────────────────────────────────────────────────────


def _make_link(agent: NodeAgent) -> NodeGrpcLink:
    return NodeGrpcLink(agent, control_addr="127.0.0.1:0")


@pytest.mark.asyncio
async def test_dispatcher_acquire_container_returns_correct_reply() -> None:
    agent, _ = _make_agent_with_docker({"busybox:1"})
    link = _make_link(agent)

    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-A"),
        rollout_id="r1",
        image="busybox:1",
        command=["sleep", "infinity"],
        labels={"my-label": "abc"},
    )
    reply = await link._exec_acquire_container(cmd)

    assert reply.command_id == "cmd-A"
    assert reply.status == pb.ReplyStatus.OK
    assert reply.acquire_container.container_id.startswith("fake-container-id-")
    assert reply.acquire_container.container_name


@pytest.mark.asyncio
async def test_dispatcher_container_exec_returns_exec_reply() -> None:
    agent, backend = _make_agent_with_docker({"busybox:1"})
    record = await agent.acquire_container(
        rollout_id="r1", backend="docker", image="busybox:1",
    )
    container = backend.docker_client.containers.get(record.container_id)
    container.exec_results[("echo", "hi")] = _ExecResult(
        exit_code=0, output=(b"hi\n", b""),
    )

    link = _make_link(agent)
    cmd = pb.ContainerExecCommand(
        header=pb.CommandHeader(command_id="cmd-E"),
        rollout_id="r1",
        container_id=record.container_id,
        cmd=["echo", "hi"],
        timeout_s=5.0,
    )
    reply = await link._exec_container_exec(cmd)

    assert reply.command_id == "cmd-E"
    assert reply.status == pb.ReplyStatus.OK
    assert reply.exec.exit_code == 0
    assert reply.exec.stdout == b"hi\n"
    assert reply.exec.timed_out is False


@pytest.mark.asyncio
async def test_dispatcher_acquire_carries_environment_field_through() -> None:
    """Audit Raw-Policy-M1: ``environment`` is wired end-to-end on
    the raw-container path. Earlier the dispatcher dropped the
    field on the floor; the proto + dispatcher + manager now plumb
    it through to ``docker.containers.run(environment=...)``."""
    agent, backend = _make_agent_with_docker({"busybox:1"})
    link = _make_link(agent)

    cmd = pb.AcquireContainerCommand(
        header=pb.CommandHeader(command_id="cmd-env"),
        rollout_id="r1",
        image="busybox:1",
        environment={"HF_TOKEN": "abc", "GIT_AUTHOR": "smoke"},
    )
    reply = await link._exec_acquire_container(cmd)

    assert reply.status == pb.ReplyStatus.OK
    container = backend.docker_client.containers.get(
        reply.acquire_container.container_id,
    )
    assert container.environment == {"HF_TOKEN": "abc", "GIT_AUTHOR": "smoke"}


@pytest.mark.asyncio
async def test_dispatcher_destroy_container_returns_destroy_reply() -> None:
    agent, _ = _make_agent_with_docker({"busybox:1"})
    record = await agent.acquire_container(
        rollout_id="r1", backend="docker", image="busybox:1",
    )

    link = _make_link(agent)
    cmd = pb.DestroyContainerCommand(
        header=pb.CommandHeader(command_id="cmd-D"),
        rollout_id="r1",
        container_id=record.container_id,
        force=True,
    )
    reply = await link._exec_destroy_container(cmd)

    assert reply.command_id == "cmd-D"
    assert reply.status == pb.ReplyStatus.OK
    # The DestroyReply is empty by design — success is the OK status.
