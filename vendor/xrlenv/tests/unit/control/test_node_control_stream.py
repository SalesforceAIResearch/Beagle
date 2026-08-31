"""Regression test for the NodeControlStream per-iteration task leak.

The bidi server-side generator in :class:`NodeControlServicer.NodeControlStream`
creates two short-lived tasks per outer-loop iteration (one wrapping
``outbox.get()``, one wrapping ``reader_done.wait()``) and uses
``asyncio.wait(..., return_when=FIRST_COMPLETED)`` to race them. Before
the fix, any teardown path that bypassed the in-loop cleanup
branches — ``GeneratorExit`` at the ``yield``, ``CancelledError``
out of ``asyncio.wait``, the reader_done branch racing with a
finished outbox.get — left those per-iteration tasks pending. asyncio
later reaped them with ``Task was destroyed but it is pending!``
warnings (operator-reported during a 2-minute production session
2026-05-19).

This test drives :py:meth:`NodeControlServicer.NodeControlStream`
directly via :py:meth:`agen.aclose` while the generator is parked
in ``asyncio.wait`` and asserts no leftover task wrapping
``Queue.get`` or ``Event.wait`` survives the teardown.
"""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator
from contextlib import suppress

import pytest
from xrlenv.api._pb2 import node_control_pb2 as pb
from xrlenv.control.grpc_endpoint import NodeControlServicer
from xrlenv.node.hw_probe import HardwareInfo


def _hello(agent_version: str | None = None) -> pb.NodeMsg:
    hw = HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )
    from xrlenv.api import converters as conv
    from xrlenv.buildinfo import agent_identity

    # Default to the control plane's own identity so non-skew tests
    # (e.g. the per-iteration task-leak regression) don't trip the
    # issue-#18 version-skew WARN.
    version = agent_identity() if agent_version is None else agent_version
    return pb.NodeMsg(
        stream_epoch="test-epoch",
        seq=0,
        hello=pb.NodeHello(
            node_id="test-node",
            backends=["docker"],
            hardware=conv.hardware_info_to_proto(hw),
            agent_version=version,
        ),
    )


class _BlockingRequestIter:
    """Async iterator that yields NodeHello then blocks forever on the
    next ``__anext__`` call. Matches what a healthy real node looks
    like to the server (Hello arrives, then long idle waiting for
    replies / heartbeats), so the generator parks in
    ``asyncio.wait({outbox_get, reader_wait})``.
    """

    def __init__(self, agent_version: str | None = None) -> None:
        self._yielded_hello = False
        self._wait_forever: asyncio.Future[None] | None = None
        self._agent_version = agent_version

    def __aiter__(self) -> AsyncIterator[pb.NodeMsg]:
        return self

    async def __anext__(self) -> pb.NodeMsg:
        if not self._yielded_hello:
            self._yielded_hello = True
            return _hello(self._agent_version)
        # Block indefinitely; the reader_loop sits inside this future
        # until the generator is aclose'd (which surfaces as
        # CancelledError on the reader_task).
        self._wait_forever = asyncio.get_running_loop().create_future()
        await self._wait_forever
        raise StopAsyncIteration  # unreachable, satisfies type-checker


async def _drain_until_parked(
    agen: AsyncIterator[pb.ControlMsg],
) -> None:
    """Pull until the generator is parked inside the
    ``asyncio.wait`` call — i.e. it's emitted the ControlHello + the
    in-process state has both per-iteration tasks live.

    The ControlHello is the only message the generator yields
    unprompted (the rest depend on outbox traffic the test doesn't
    produce). One ``anext`` therefore returns ControlHello; the
    second ``anext`` enters the loop body and parks in
    ``asyncio.wait``.
    """
    first = await asyncio.wait_for(anext(agen), timeout=1.0)
    assert first.HasField("hello"), "expected ControlHello as first msg"
    # Kick off the second pull so the generator enters the
    # outbox_get / reader_wait state. Don't await it — schedule and
    # yield so it starts running.
    pull_task = asyncio.create_task(anext(agen))
    # Several event-loop ticks for the generator to advance into
    # ``asyncio.wait``. Empirically 2 ticks is enough on cpython,
    # take 10 for safety.
    for _ in range(10):
        await asyncio.sleep(0)
    return pull_task


def _pending_queue_or_event_waiters() -> list[asyncio.Task[object]]:
    """All currently-running tasks whose coroutine is parked inside
    ``asyncio.Queue.get`` or ``asyncio.Event.wait``. These are the
    exact shapes the pre-fix leak produced — ``outbox.get()`` is the
    first, ``reader_done.wait()`` is the second.
    """
    out: list[asyncio.Task[object]] = []
    for t in asyncio.all_tasks():
        if t.done():
            continue
        coro = t.get_coro()
        qual = getattr(coro, "__qualname__", "") or ""
        if qual.endswith(("Queue.get", "Event.wait")):
            out.append(t)
    return out


@pytest.mark.asyncio
async def test_node_control_stream_absorbs_cancellation_on_shutdown() -> None:
    """gRPC cancels idle bidi RPCs on graceful shutdown (the
    ``xrlenv up`` signal handler). The servicer must run its teardown
    and then *absorb* that CancelledError rather than re-raising it —
    a re-raise escapes to cygrpc as a spurious 'Exception not handled
    by _handle_exceptions' ERROR + traceback in the operator log
    (operator-reported 2026-05-20).
    """
    disconnected: list[str] = []
    servicer = NodeControlServicer(
        on_connected=lambda _t: None,
        on_disconnected=lambda t: disconnected.append(t.node_id),
        control_instance_id="test-ctrl",
    )
    request_iter = _BlockingRequestIter()
    agen = servicer.NodeControlStream(request_iter, context=None)

    pull_task = await _drain_until_parked(agen)

    # Cancel the in-flight anext the way a cancelled gRPC RPC task
    # does — this throws CancelledError into the parked generator.
    pull_task.cancel()
    with suppress(StopAsyncIteration, asyncio.CancelledError):
        await pull_task

    # The teardown (`finally`) still ran to completion ...
    assert disconnected == ["test-node"]
    # ... and the cancellation did not re-propagate out of the
    # generator as a bare CancelledError.
    assert not pull_task.cancelled()


@pytest.mark.asyncio
async def test_node_control_stream_does_not_leak_per_iteration_tasks() -> None:
    """The generator's teardown via ``aclose`` while parked in
    ``asyncio.wait`` must cancel both per-iteration tasks. Before the
    fix this leaked the ``outbox.get()`` task wrapper (operator-
    reported as ``Task was destroyed but it is pending!``).
    """
    servicer = NodeControlServicer(
        on_connected=lambda _t: None,
        on_disconnected=lambda _t: None,
        control_instance_id="test-ctrl",
    )

    request_iter = _BlockingRequestIter()
    agen = servicer.NodeControlStream(request_iter, context=None)

    pull_task = await _drain_until_parked(agen)

    # Snapshot the leaked-shape candidates before teardown so the
    # post-aclose assertion has a baseline to subtract from.
    before = set(_pending_queue_or_event_waiters())
    assert before, (
        "expected the generator to be parked with at least one "
        "Queue.get / Event.wait task; fixture didn't trigger the loop"
    )

    # Tear the generator down the same way a real gRPC handler does
    # on stream close (client disconnect, server shutdown). Cancel
    # the in-flight ``anext`` first so the generator is idle, then
    # aclose() injects GeneratorExit at the next await point.
    pull_task.cancel()
    with suppress(StopAsyncIteration, asyncio.CancelledError):
        await pull_task
    # ``aclose`` on an already-cancelled generator returns
    # immediately; in this test we want the finally block to actually
    # run, which the gen's finalizer (driven by gc) handles. Force
    # the cycle by dropping the reference + collecting.
    del agen
    gc.collect()

    # Give the cancellation cycle a chance to settle + force a gc
    # pass so any orphaned task would surface here rather than at an
    # arbitrary later point.
    for _ in range(10):
        await asyncio.sleep(0)
    gc.collect()

    after = set(_pending_queue_or_event_waiters())
    leaked = after & before
    assert not leaked, (
        f"NodeControlStream leaked {len(leaked)} per-iteration "
        f"task(s) after aclose: {[repr(t) for t in leaked]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 (Ask #2) — node-agent version-skew detection at connect
# ──────────────────────────────────────────────────────────────────────────────


async def _drive_connect_then_close(
    agent_version: str | None,
) -> None:
    """Drive NodeControlStream far enough that the connect-time
    logging (issue-#18 skew check) runs, then tear the generator
    down. ``agent_version=None`` means the NodeHello carries no
    agent_version field at all (a pre-#18 node-agent)."""
    servicer = NodeControlServicer(
        on_connected=lambda _t: None,
        on_disconnected=lambda _t: None,
        control_instance_id="test-ctrl",
    )
    request_iter = _BlockingRequestIter(agent_version=agent_version)
    agen = servicer.NodeControlStream(request_iter, context=None)
    pull_task = await _drain_until_parked(agen)
    pull_task.cancel()
    with suppress(StopAsyncIteration, asyncio.CancelledError):
        await pull_task
    with suppress(Exception):
        await agen.aclose()


@pytest.mark.asyncio
async def test_connect_warns_on_version_skew(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A node reporting an agent_version different from the control
    plane's own triggers a 'version skew' WARN naming both sides."""
    with caplog.at_level("WARNING", logger="xrlenv.control.grpc_endpoint"):
        await _drive_connect_then_close(agent_version="0.0.1+stale0000beef")

    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("version skew" in r.message for r in warns), (
        f"expected a version-skew WARN; got {[r.message for r in warns]}"
    )
    assert any("0.0.1+stale0000beef" in r.message for r in warns)


@pytest.mark.asyncio
async def test_connect_warns_on_missing_agent_version(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A node that reports an empty agent_version (its binary predates
    the field) gets its own 'redeploy' WARN — this is exactly the
    stale-node signal issue #18 wanted surfaced."""
    with caplog.at_level("WARNING", logger="xrlenv.control.grpc_endpoint"):
        await _drive_connect_then_close(agent_version="")

    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("NO agent_version" in r.message for r in warns), (
        f"expected a missing-version WARN; got {[r.message for r in warns]}"
    )
    assert any("redeploy" in r.message.lower() for r in warns)


@pytest.mark.asyncio
async def test_connect_no_skew_warn_when_versions_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A node reporting the same identity as the control plane must
    NOT trigger a skew WARN."""
    with caplog.at_level("WARNING", logger="xrlenv.control.grpc_endpoint"):
        await _drive_connect_then_close(agent_version=None)  # matches control

    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any(
        "version skew" in r.message or "NO agent_version" in r.message
        for r in warns
    ), f"unexpected skew WARN for a matching version: {[r.message for r in warns]}"
