"""Coverage for the silent-hang fix on
``RemoteNodeTransport._send_and_wait``.

Background: ``_send_and_wait`` previously did an unbounded
``await future`` so a node that never replied (wedged, stale
binary that didn't recognise the command field, etc.) hung the
consumer indefinitely. Operator-reported 2026-05-06 against a
stale node binary + the new ``AcquireContainerCommand`` field.

Fix: opt-in ``timeout_s`` parameter. ``None`` (default) preserves
the original unbounded behaviour for every existing call-site;
new raw-container methods pass concrete bounds (60s acquire,
``timeout_s + 30s`` exec, 30s destroy).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.control.grpc_endpoint import RemoteNodeTransport
from xrlenv.errors import XRLEnvError
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


@pytest.mark.asyncio
async def test_send_and_wait_with_timeout_raises_xrlenv_error_on_hang() -> None:
    """A future that never resolves surfaces a clean XRLEnvError
    when ``timeout_s`` is set."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    with pytest.raises(XRLEnvError, match=r"timed out after 0\.1s"):
        await transport._send_and_wait(
            msg, command_id="cmd-stuck", timeout_s=0.1,
        )


@pytest.mark.asyncio
async def test_send_and_wait_timeout_raises_node_command_timeout_subclass() -> None:
    """#2 — a command reply timeout raises the dedicated
    ``NodeCommandTimeout`` (a subclass of ``XRLEnvError``, so legacy
    ``except XRLEnvError`` callers are unaffected) so the raw-destroy path
    can distinguish a slow teardown from a hard failure."""
    from xrlenv.errors import NodeCommandTimeout

    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    with pytest.raises(NodeCommandTimeout):
        await transport._send_and_wait(
            msg, command_id="cmd-stuck", timeout_s=0.1,
        )


@pytest.mark.asyncio
async def test_send_and_wait_timeout_drops_pending_entry() -> None:
    """After a timeout the pending-future map is cleaned up so a
    late reply doesn't get posted to a stale future. Defends
    against a pathological replay where a stuck node finally
    replies after the consumer has moved on."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    with pytest.raises(XRLEnvError):
        await transport._send_and_wait(
            msg, command_id="cmd-stuck", timeout_s=0.05,
        )

    assert "cmd-stuck" not in transport._pending


@pytest.mark.asyncio
async def test_send_and_wait_failed_reply_names_the_node() -> None:
    """Issue #18 (Ask #1): a FAILED ``CommandReply`` surfaces as an
    ``XRLEnvError`` that names the node, so a consumer-visible failure
    points at the culprit without cross-referencing the control log."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    async def _send_then_fail() -> pb.CommandReply:
        return await transport._send_and_wait(msg, command_id="cmd-fail")

    task = asyncio.create_task(_send_then_fail())
    await asyncio.sleep(0.05)
    transport._pending["cmd-fail"].set_result(
        pb.CommandReply(
            command_id="cmd-fail",
            status=pb.ReplyStatus.FAILED,
            error_kind="ReadTimeout",
            error_message="docker daemon slow",
        ),
    )
    with pytest.raises(XRLEnvError) as excinfo:
        await task
    text = str(excinfo.value)
    assert "node test-node" in text
    assert "ReadTimeout" in text and "docker daemon slow" in text


@pytest.mark.asyncio
async def test_send_and_wait_without_timeout_preserves_legacy_unbounded() -> None:
    """The default ``timeout_s=None`` preserves the original
    unbounded behaviour. Verified by completing the future after
    a delay that would have tripped any reasonable default
    timeout — ``None`` lets the await proceed."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    async def _send_with_late_reply() -> pb.CommandReply:
        return await transport._send_and_wait(
            msg, command_id="cmd-late",
        )

    task = asyncio.create_task(_send_with_late_reply())
    # Yield once so the task's ``await self._outbox.put`` runs and
    # the future gets registered in self._pending.
    await asyncio.sleep(0.05)
    # Resolve the future "late" — well after any sane default
    # timeout would have fired.
    transport._pending["cmd-late"].set_result(
        pb.CommandReply(
            command_id="cmd-late", status=pb.ReplyStatus.OK,
        ),
    )
    reply = await task
    assert reply.command_id == "cmd-late"
    assert reply.status == pb.ReplyStatus.OK


# ── Issue #14 — heartbeat carries disk state ─────────────────────────────────


def test_disk_state_unknown_until_first_heartbeat() -> None:
    transport = _make_transport()
    assert transport.disk_state() == (0, 0)


def test_touch_records_disk_state_from_heartbeat() -> None:
    transport = _make_transport()
    transport.touch(
        free_disk_bytes=42 * 1024**3,
        total_disk_bytes=200 * 1024**3,
    )
    assert transport.disk_state() == (42 * 1024**3, 200 * 1024**3)


def test_touch_with_full_zero_disk_state_keeps_last_known() -> None:
    # ``(0, 0)`` is the "sample failed on the node" sentinel — a
    # single transient daemon hiccup must NOT erase a previously-good
    # reading, otherwise the gate's signal blanks on every flake.
    transport = _make_transport()
    transport.touch(
        free_disk_bytes=42 * 1024**3,
        total_disk_bytes=200 * 1024**3,
    )
    transport.touch(free_disk_bytes=0, total_disk_bytes=0)
    assert transport.disk_state() == (42 * 1024**3, 200 * 1024**3)


def test_touch_records_real_disk_full_after_healthy_sample() -> None:
    """Audit M1 regression: a node that first reports a healthy
    sample and then reports ``free=0, total>0`` (disk genuinely
    full) must update the transport's free-byte cache to ``0``.
    The earlier ``if free > 0`` gate kept the stale positive
    reading, leaving the scheduler / admin pill seeing the node
    as healthy while it sat at 100 % full — the exact failure
    mode issue #14 set out to catch.
    """
    transport = _make_transport()
    transport.touch(
        free_disk_bytes=42 * 1024**3,
        total_disk_bytes=200 * 1024**3,
    )
    transport.touch(free_disk_bytes=0, total_disk_bytes=200 * 1024**3)
    assert transport.disk_state() == (0, 200 * 1024**3)


# ── Issue #12 — AcquireContainer wire timeout default + override ─────────────


@pytest.mark.asyncio
async def test_acquire_container_uses_600s_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #12 regression: the wire timeout for AcquireContainer
    must default to 600 s (matching the node-side image-cache pull
    timeout) so cold-pull cases no longer race the wire at the
    pre-fix 60 s ceiling. Verified by capturing the ``timeout_s``
    argument the transport hands to ``_send_and_wait``.
    """
    transport = _make_transport()
    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            acquire_container=pb.AcquireContainerReply(
                container_id="ci", container_name="cn",
            ),
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.acquire_container(
        rollout_id="r-1", backend="docker", image="busybox:1",
    )
    assert captured["timeout_s"] == 600.0


@pytest.mark.asyncio
async def test_acquire_container_honours_per_call_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #12: a consumer with a known-huge image must be able
    to widen the wire deadline via ``acquire_timeout_s=...``.
    """
    transport = _make_transport()
    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            acquire_container=pb.AcquireContainerReply(
                container_id="ci", container_name="cn",
            ),
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.acquire_container(
        rollout_id="r-1", backend="docker", image="huge:1",
        acquire_timeout_s=1800.0,
    )
    assert captured["timeout_s"] == 1800.0


async def _capture_acquire_command(
    transport: Any, monkeypatch: pytest.MonkeyPatch, **acquire_kwargs: Any,
) -> Any:
    """Run acquire_container with a stubbed _send_and_wait, returning the
    AcquireContainerCommand the transport put on the wire."""
    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["msg"] = msg
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            acquire_container=pb.AcquireContainerReply(
                container_id="ci", container_name="cn",
            ),
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.acquire_container(
        rollout_id="r-1", backend="docker", image="busybox:1", **acquire_kwargs,
    )
    return captured["msg"].acquire_container


@pytest.mark.asyncio
async def test_acquire_stamps_default_pull_deadline_for_node_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit P2: a DEFAULT acquire (no explicit ``acquire_timeout_s``) must still
    stamp ``pull_deadline_s`` = the 600 s effective wire budget, so the node bounds
    its create retry-with-backoff by the same deadline the CP waits on — not
    ``0.0`` (which the node maps to 'no deadline')."""
    cmd = await _capture_acquire_command(_make_transport(), monkeypatch)
    assert cmd.pull_deadline_s == 600.0


@pytest.mark.asyncio
async def test_acquire_stamps_explicit_pull_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``acquire_timeout_s`` still flows through to ``pull_deadline_s``
    (Issue #12 audit M1 — widen the node pull for known-huge images)."""
    cmd = await _capture_acquire_command(
        _make_transport(), monkeypatch, acquire_timeout_s=1800.0,
    )
    assert cmd.pull_deadline_s == 1800.0


# ── Issue #18 (Ask #2) — command-timeout node-health tracking ────────────────


def test_seconds_since_last_command_timeout_none_until_first_timeout() -> None:
    """A fresh transport has never timed out — the gate probe must
    return ``None`` so the scheduler classifies it as healthy."""
    transport = _make_transport()
    assert transport.seconds_since_last_command_timeout() is None


@pytest.mark.asyncio
async def test_send_and_wait_timeout_records_command_timeout() -> None:
    """When ``_send_and_wait`` hits its ceiling, the transport must
    record the event so the scheduler's node-health gate can exclude
    the node. Before the timeout the probe is ``None``; after, it's
    a small non-negative elapsed value and the cumulative counter
    has incremented."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    assert transport.seconds_since_last_command_timeout() is None
    assert transport._command_timeout_total == 0

    with pytest.raises(XRLEnvError):
        await transport._send_and_wait(
            msg, command_id="cmd-stuck", timeout_s=0.05,
        )

    elapsed = transport.seconds_since_last_command_timeout()
    assert elapsed is not None
    assert 0.0 <= elapsed < 5.0
    assert transport._command_timeout_total == 1


@pytest.mark.asyncio
async def test_command_timeout_total_accumulates_across_timeouts() -> None:
    """The cumulative counter (admin observability) increments once
    per timeout — used by dashboards to surface a chronically-
    degraded node, distinct from the time-decayed gate signal."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    for i in range(3):
        with pytest.raises(XRLEnvError):
            await transport._send_and_wait(
                msg, command_id=f"cmd-{i}", timeout_s=0.02,
            )

    assert transport._command_timeout_total == 3


@pytest.mark.asyncio
async def test_successful_reply_does_not_record_timeout() -> None:
    """A command that gets a reply within the ceiling must NOT mark
    the node degraded — only genuine reply-timeouts feed the gate."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    async def _send_then_reply() -> pb.CommandReply:
        return await transport._send_and_wait(
            msg, command_id="cmd-ok", timeout_s=5.0,
        )

    task = asyncio.create_task(_send_then_reply())
    await asyncio.sleep(0.05)
    transport._pending["cmd-ok"].set_result(
        pb.CommandReply(command_id="cmd-ok", status=pb.ReplyStatus.OK),
    )
    await task

    assert transport.seconds_since_last_command_timeout() is None
    assert transport._command_timeout_total == 0


# ── Issue #18 — container archive wire ceilings (60s → 300s) ─────────────────


@pytest.mark.asyncio
async def test_container_put_archive_uses_300s_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``container_put_archive`` must pass ``timeout_s=300.0`` to
    ``_send_and_wait``. The old 60s ceiling crashed graders when a
    pull-saturated node-agent couldn't service the archive RPC in
    time (issue #18)."""
    transport = _make_transport()
    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(command_id=command_id, status=pb.ReplyStatus.OK)

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.container_put_archive(
        rollout_id="r-1", container_id="c" * 12,
        target_dir="/workspace", tarball=b"<tar>",
    )
    assert captured["timeout_s"] == 300.0


@pytest.mark.asyncio
async def test_container_get_archive_reassembles_chunked_reply() -> None:
    """``container_get_archive`` now collects a *chunked* reply (the
    node-lost oversized-reply fix: a single whole-tarball CommandReply
    could exceed the wire ceiling and sever the heartbeat stream)
    rather than a single ``_send_and_wait`` reply. Feeding a chunk +
    terminator through the reader hook reproduces the tarball. The 300s
    stream ceiling itself is asserted in
    test_get_archive_chunking.py::test_control_get_archive_uses_300s_stream_ceiling.
    """
    transport = _make_transport()
    task = asyncio.create_task(
        transport.container_get_archive(
            rollout_id="r-1", container_id="c" * 12,
            source_path="/workspace/log",
        ),
    )
    await asyncio.sleep(0.05)  # let it register the stream + enqueue cmd
    sent = transport._outbox.get_nowait()
    cid = sent.container_get_archive.header.command_id

    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid, status=pb.ReplyStatus.OK,
            container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                data=b"<tar>", done=False,
            ),
        ),
    )
    transport.deliver_reply(
        pb.CommandReply(
            command_id=cid, status=pb.ReplyStatus.OK,
            container_get_archive_chunk=pb.ContainerGetArchiveChunk(
                data=b"", done=True,
            ),
        ),
    )

    out = await asyncio.wait_for(task, timeout=2.0)
    assert out == b"<tar>"


# ── Audit M2 — cancel_build_image transport timeout + cancellation cleanup ───


@pytest.mark.asyncio
async def test_cancel_build_image_passes_timeout_to_send_and_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M2: ``cancel_build_image`` must hand ``_send_and_wait`` a
    concrete ``timeout_s`` (like ensure_present / build_image) so a
    no-reply node takes the clean timeout branch rather than relying on
    an outer ``asyncio.wait_for`` cancelling the coroutine."""
    transport = _make_transport()
    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            cancel_build_image=pb.CancelBuildImageReply(status="ok", error=""),
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    status, error = await transport.cancel_build_image(
        image_ref="bench/x:1", timeout_s=30.0,
    )
    assert captured["timeout_s"] == 30.0
    assert (status, error) == ("ok", "")


@pytest.mark.asyncio
async def test_cancel_build_image_timeout_records_health_and_drops_pending(
) -> None:
    """Audit M2: a cancel whose node never replies takes the timeout
    branch — popping the pending entry AND flagging the command-timeout
    health marker — instead of leaking pending state via outer
    cancellation. Mirrors the ensure_present / build_image accounting."""
    transport = _make_transport()
    assert transport.seconds_since_last_command_timeout() is None
    assert transport._command_timeout_total == 0

    with pytest.raises(XRLEnvError):
        await transport.cancel_build_image(image_ref="bench/x:1", timeout_s=0.05)

    assert transport._command_timeout_total == 1
    assert transport.seconds_since_last_command_timeout() is not None
    assert not transport._pending  # pending table fully drained


@pytest.mark.asyncio
async def test_send_and_wait_cancellation_drops_pending_entry() -> None:
    """Audit M2 (general guard): an outer cancellation — e.g. the admin's
    ``asyncio.wait_for`` ceiling firing on an unbounded call — must pop
    the pending future so a later dropped reply can't leave stale state
    lingering until disconnect."""
    transport = _make_transport()
    msg = pb.ControlMsg(stream_epoch="test-epoch", seq=1)

    task = asyncio.create_task(
        transport._send_and_wait(msg, command_id="cmd-cancel"),
    )
    await asyncio.sleep(0.05)  # let it register the pending future
    assert "cmd-cancel" in transport._pending

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "cmd-cancel" not in transport._pending


# ── B7.6 — report_images carries its own transport timeout ──────────────────


@pytest.mark.asyncio
async def test_report_images_passes_default_timeout_to_send_and_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``report_images`` must hand ``_send_and_wait`` a concrete timeout
    (default from XRLENV_REPORT_IMAGES_TIMEOUT_S) so a slow ``docker
    system df`` on a large catalog surfaces a clean XRLEnvError instead of
    being cancelled by the admin's outer wait_for and leaking pending
    state (which then logs 'reply for unknown command_id')."""
    from xrlenv.control.grpc_endpoint import DEFAULT_REPORT_IMAGES_TIMEOUT_S

    transport = _make_transport()
    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            report_images=pb.ReportImagesReply(free_disk_bytes=0),
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.report_images()
    assert captured["timeout_s"] == DEFAULT_REPORT_IMAGES_TIMEOUT_S


@pytest.mark.asyncio
async def test_report_images_honours_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller can widen the report_images deadline for a very large
    per-node catalog."""
    transport = _make_transport()
    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(
            command_id=command_id,
            status=pb.ReplyStatus.OK,
            report_images=pb.ReportImagesReply(free_disk_bytes=0),
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.report_images(timeout_s=120.0)
    assert captured["timeout_s"] == 120.0
