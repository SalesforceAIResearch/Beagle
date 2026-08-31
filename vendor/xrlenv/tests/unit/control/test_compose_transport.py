"""RemoteNodeTransport compose-project wire methods (multi-service step 3a-2).

Verifies the CP→node transport builds the right ``AcquireComposeProjectCommand`` /
``DestroyComposeProjectCommand`` and unpacks the reply into a duck-typed record the
coordinator consumes identically to the in-process (NodeAgent) transport. The
node round-trip is mocked at ``_send_and_wait``.
"""
from __future__ import annotations

import asyncio
from typing import Any

from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.control.grpc_endpoint import (
    RemoteNodeTransport,
    _RemoteComposeProjectRecord,
)
from xrlenv.node.hw_probe import HardwareInfo


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _make_transport() -> RemoteNodeTransport:
    from xrlenv.control.grpc_endpoint import _MonotonicCounter

    return RemoteNodeTransport(
        node_id="test-node",
        backends=["docker"],
        hardware=_hw(),
        outbox=asyncio.Queue(),
        stream_epoch="test-epoch",
        control_instance_id="ctrl-1",
        control_seq=_MonotonicCounter(),
    )


async def test_acquire_compose_project_builds_command_and_unpacks_reply() -> None:
    transport = _make_transport()
    sent: list[Any] = []

    async def fake_send(msg, command_id, *, timeout_s=None):
        sent.append((msg, timeout_s))
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            acquire_compose_project=pb.AcquireComposeProjectReply(
                main_container_id="cidmainfull",
                main_container_name="proj-main",
                project_name="proj",
                project_dir="/tmp/proj",
                service_container_ids={"main": "cidmainfull", "pg": "cidpgfull"},
            ),
        )

    transport._send_and_wait = fake_send  # type: ignore[method-assign]

    rec = await transport.acquire_compose_project(
        rollout_id="r1",
        project_name="proj",
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app@sha256:abc", "postgres:14"],
        main_service="main",
        up_timeout_s=300.0,
    )

    # command was built with the compose fields
    msg, timeout_s = sent[0]
    cmd = msg.acquire_compose_project
    assert cmd.rollout_id == "r1"
    assert cmd.project_name == "proj"
    assert cmd.main_service == "main"
    assert cmd.up_timeout_s == 300.0
    assert list(cmd.images) == ["ns/app@sha256:abc", "postgres:14"]
    # wire timeout covers ensure-present pulls + up --wait (> the up budget)
    assert timeout_s > 300.0
    # reply unpacked into the duck-typed record (same shape as ComposeProjectRecord)
    assert isinstance(rec, _RemoteComposeProjectRecord)
    assert rec.main_container_id == "cidmainfull"
    assert rec.main_container_name == "proj-main"
    assert rec.project_name == "proj"
    assert rec.project_dir == "/tmp/proj"
    assert rec.service_container_ids == {"main": "cidmainfull", "pg": "cidpgfull"}
    assert set(rec.member_container_ids) == {"cidmainfull", "cidpgfull"}


async def test_acquire_compose_project_defaults_main_and_timeout() -> None:
    transport = _make_transport()

    async def fake_send(msg, command_id, *, timeout_s=None):
        # capture then reply minimally
        fake_send.msg = msg  # type: ignore[attr-defined]
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            acquire_compose_project=pb.AcquireComposeProjectReply(
                main_container_id="x", project_name="p",
            ),
        )

    transport._send_and_wait = fake_send  # type: ignore[method-assign]
    await transport.acquire_compose_project(
        rollout_id="r", project_name="p", compose_yaml="services:\n  main: {}\n",
    )
    cmd = fake_send.msg.acquire_compose_project  # type: ignore[attr-defined]
    assert cmd.main_service == "main"  # default
    assert cmd.up_timeout_s == 0.0  # 0 = node default
    assert list(cmd.images) == []


async def test_destroy_compose_project_builds_command() -> None:
    transport = _make_transport()
    sent: list[Any] = []

    async def fake_send(msg, command_id, *, timeout_s=None):
        sent.append(msg)
        return pb.CommandReply(
            command_id=command_id, status=pb.ReplyStatus.OK,
            destroy=pb.DestroyReply(),
        )

    transport._send_and_wait = fake_send  # type: ignore[method-assign]
    await transport.destroy_compose_project(
        rollout_id="r1", project_name="proj",
    )
    cmd = sent[0].destroy_compose_project
    assert cmd.rollout_id == "r1"
    assert cmd.project_name == "proj"
    assert cmd.force is True


async def test_list_raw_container_ids_stashes_compose_info() -> None:
    # 3c-2: the transport stashes per-container correlation labels
    # (container_id -> (rollout_id, compose_project)) for the raw-GC reconciler.
    transport = _make_transport()

    async def fake_send(msg, command_id, *, timeout_s=None):
        reply = pb.CommandReply(
            command_id=command_id, status=pb.ReplyStatus.OK,
            list_raw_containers=pb.ListRawContainersReply(
                container_ids=["cid_main", "cid_plain"],
            ),
        )
        reply.list_raw_containers.containers.add(
            container_id="cid_main", rollout_id="r1", compose_project="proj",
        )
        reply.list_raw_containers.containers.add(
            container_id="cid_plain", rollout_id="r2", compose_project="",
        )
        return reply

    transport._send_and_wait = fake_send  # type: ignore[method-assign]
    ids = await transport.list_raw_container_ids()
    assert ids == ["cid_main", "cid_plain"]
    assert transport._last_container_info == {
        "cid_main": ("r1", "proj"),
        "cid_plain": ("r2", ""),
    }


def test_sdk_client_exposes_compose_acquire() -> None:
    # Step 4: the consumer SDK now exposes the compose surface (this replaces the
    # transitional "no compose method yet" guard). ``Client.acquire_compose_project``
    # returns a ClusterComposeSession; teardown is on the session (``.destroy()`` →
    # transport.destroy_compose_project), not a top-level Client method — mirroring
    # how ``acquire_container`` returns a session that owns its own destroy.
    import inspect

    from xrlenv.client.client import Client
    from xrlenv.client.container_session import ClusterComposeSession

    assert hasattr(Client, "acquire_compose_project")
    sig = inspect.signature(Client.acquire_compose_project)
    # the whole-stack footprint arrives as scalars (ResourceSpec-free surface)
    assert {"compose_yaml", "images", "footprint_cpu", "footprint_mem_bytes"} <= set(
        sig.parameters,
    )
    # the session (not the Client) owns teardown
    assert hasattr(ClusterComposeSession, "destroy")
    assert not hasattr(Client, "destroy_compose_project")
