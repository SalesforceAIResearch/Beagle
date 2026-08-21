"""P1.7.A.1 — Tests for ``ClusterContainerSession`` SDK class.

The session wraps the 3 raw-container RPCs into an ergonomic
async context manager. Tests use a fake ``ClientTransport`` so
they don't need a live control plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.client.client import Client
from xrlenv.client.container_session import ClusterContainerSession
from xrlenv.control.service import RawAcquireResult, RawExecResult
from xrlenv.errors import XRLEnvError

# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeTransport:
    """Minimal ClientTransport stand-in. Records calls so tests can
    assert on routing + arguments."""

    next_acquire: RawAcquireResult = field(
        default_factory=lambda: RawAcquireResult(
            rollout_id="r-1", container_id="c-1",
            container_name="cname-1", node_id="node-A",
        ),
    )
    exec_calls: list[dict] = field(default_factory=list)
    destroy_calls: list[dict] = field(default_factory=list)
    next_exec: RawExecResult = field(
        default_factory=lambda: RawExecResult(
            exit_code=0, stdout=b"hi\n", stderr=b"",
            timed_out=False,
        ),
    )
    raise_on_exec: Exception | None = None
    raise_on_destroy: Exception | None = None
    closed: bool = False

    async def acquire_container(self, **kwargs: Any) -> RawAcquireResult:
        return self.next_acquire

    async def container_exec(self, **kwargs: Any) -> RawExecResult:
        if self.raise_on_exec:
            raise self.raise_on_exec
        self.exec_calls.append(kwargs)
        return self.next_exec

    async def destroy_container(self, **kwargs: Any) -> None:
        if self.raise_on_destroy:
            raise self.raise_on_destroy
        self.destroy_calls.append(kwargs)

    async def container_put_archive(self, **kwargs: Any) -> None:
        if not hasattr(self, "put_archive_calls"):
            self.put_archive_calls = []
        self.put_archive_calls.append(kwargs)

    async def container_get_archive(self, **kwargs: Any) -> bytes:
        if not hasattr(self, "get_archive_calls"):
            self.get_archive_calls = []
        self.get_archive_calls.append(kwargs)
        return getattr(self, "get_archive_return", b"<tar>")

    def container_exec_stream(self, **kwargs: Any) -> Any:
        if not hasattr(self, "exec_stream_calls"):
            self.exec_stream_calls = []
        self.exec_stream_calls.append(kwargs)
        from xrlenv.control.service import RawExecChunk
        chunks = getattr(self, "exec_stream_chunks", [
            RawExecChunk(
                stdout=b"hi\n", stderr=b"", done=False,
                exit_code=0, timed_out=False,
            ),
            RawExecChunk(
                stdout=b"", stderr=b"", done=True,
                exit_code=0, timed_out=False,
            ),
        ])

        async def _gen() -> Any:
            for c in chunks:
                yield c
        return _gen()

    async def close(self) -> None:
        self.closed = True

    # The other ClientTransport methods aren't exercised here — Client
    # only calls the raw-container three for the session API.


# ──────────────────────────────────────────────────────────────────────────────
# Session lifecycle
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_carries_acquire_result_attrs() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    assert session.rollout_id == "r-1"
    assert session.container_id == "c-1"
    assert session.container_name == "cname-1"
    assert session.node_id == "node-A"
    assert session.destroyed is False


@pytest.mark.asyncio
async def test_session_exec_routes_through_transport() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    result = await session.exec(["echo", "hi"], timeout_s=5.0)

    assert isinstance(result, RawExecResult)
    assert result.stdout == b"hi\n"
    assert len(transport.exec_calls) == 1
    call = transport.exec_calls[0]
    assert call["rollout_id"] == "r-1"
    assert call["container_id"] == "c-1"
    assert call["cmd"] == ["echo", "hi"]
    assert call["timeout_s"] == 5.0


@pytest.mark.asyncio
async def test_session_exec_passes_optional_kwargs() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    await session.exec(
        ["echo", "hi"], cwd="/tmp",
        env={"FOO": "bar"}, user="root",
    )

    call = transport.exec_calls[0]
    assert call["cwd"] == "/tmp"
    assert call["env"] == {"FOO": "bar"}
    assert call["user"] == "root"


@pytest.mark.asyncio
async def test_session_exec_raises_after_destroy() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)
    await session.destroy()

    with pytest.raises(XRLEnvError, match="already destroyed"):
        await session.exec(["echo", "hi"])


@pytest.mark.asyncio
async def test_destroy_routes_through_transport_and_marks_destroyed() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    await session.destroy()

    assert session.destroyed is True
    assert len(transport.destroy_calls) == 1
    assert transport.destroy_calls[0]["rollout_id"] == "r-1"
    assert transport.destroy_calls[0]["container_id"] == "c-1"


@pytest.mark.asyncio
async def test_destroy_is_idempotent() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    await session.destroy()
    await session.destroy()  # no-op
    await session.destroy()  # no-op

    assert len(transport.destroy_calls) == 1


@pytest.mark.asyncio
async def test_destroy_marks_destroyed_even_when_wire_call_fails() -> None:
    """If the wire-level destroy raises, the session is still
    marked destroyed locally — operator's intent is "this session
    is over." Stale node-side container cleanup falls to GC."""
    transport = _FakeTransport(
        raise_on_destroy=RuntimeError("network blip"),
    )
    session = ClusterContainerSession(transport, transport.next_acquire)

    with pytest.raises(RuntimeError, match="network blip"):
        await session.destroy()

    assert session.destroyed is True


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.A.2 — archive ergonomics
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_put_archive_routes_through_transport() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    await session.put_archive("/tmp", b"<tar bytes>")

    assert len(transport.put_archive_calls) == 1  # type: ignore[attr-defined]
    call = transport.put_archive_calls[0]  # type: ignore[attr-defined]
    assert call["rollout_id"] == "r-1"
    assert call["container_id"] == "c-1"
    assert call["target_dir"] == "/tmp"
    assert call["tarball"] == b"<tar bytes>"


@pytest.mark.asyncio
async def test_session_get_archive_returns_bytes() -> None:
    transport = _FakeTransport()
    transport.get_archive_return = b"<received tar>"  # type: ignore[attr-defined]
    session = ClusterContainerSession(transport, transport.next_acquire)

    tarball = await session.get_archive("/logs/artifacts")

    assert tarball == b"<received tar>"
    assert len(transport.get_archive_calls) == 1  # type: ignore[attr-defined]
    assert transport.get_archive_calls[0]["source_path"] == "/logs/artifacts"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_session_exec_stream_yields_chunks() -> None:
    """SDK exec_stream returns the transport's iterator;
    consumer iterates with ``async for`` until terminator."""
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    chunks = [
        c async for c in session.exec_stream(
            ["bash", "-c", "echo hi"],
            timeout_s=600.0,
        )
    ]

    assert len(chunks) == 2
    assert chunks[0].stdout == b"hi\n"
    assert chunks[1].done is True


@pytest.mark.asyncio
async def test_session_exec_stream_raises_after_destroy() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)
    await session.destroy()

    with pytest.raises(XRLEnvError, match="cannot exec_stream"):
        # Triggering the iterator construction is enough — the
        # check fires synchronously on entry to ``exec_stream``.
        async for _ in session.exec_stream(["echo", "hi"]):
            pass


@pytest.mark.asyncio
async def test_session_archives_raise_after_destroy() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)
    await session.destroy()

    with pytest.raises(XRLEnvError, match="cannot put_archive"):
        await session.put_archive("/tmp", b"<tar>")
    with pytest.raises(XRLEnvError, match="cannot get_archive"):
        await session.get_archive("/logs")


# ──────────────────────────────────────────────────────────────────────────────
# Async context manager
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_context_manager_destroys_on_exit() -> None:
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    async with session:
        await session.exec(["echo", "hi"])

    assert session.destroyed is True
    assert len(transport.destroy_calls) == 1


@pytest.mark.asyncio
async def test_async_context_manager_destroys_on_exception() -> None:
    """Harness raises mid-evaluation — destroy still fires."""
    transport = _FakeTransport()
    session = ClusterContainerSession(transport, transport.next_acquire)

    with pytest.raises(ValueError, match="harness blew up"):
        async with session:
            await session.exec(["echo", "hi"])
            raise ValueError("harness blew up")

    # Original exception propagates, destroy still ran.
    assert session.destroyed is True
    assert len(transport.destroy_calls) == 1


@pytest.mark.asyncio
async def test_context_exit_swallows_destroy_failure_after_harness_exc() -> None:
    """If the harness raised AND the destroy fails, the harness's
    exception is the one that propagates (caller sees the real
    failure, not the cleanup noise). The destroy failure is logged
    via the LOGGER.warning in __aexit__."""
    transport = _FakeTransport(
        raise_on_destroy=RuntimeError("destroy blip"),
    )
    session = ClusterContainerSession(transport, transport.next_acquire)

    with pytest.raises(ValueError, match="harness blew up"):
        async with session:
            raise ValueError("harness blew up")


# ──────────────────────────────────────────────────────────────────────────────
# Client.acquire_container — the entry point
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_client_acquire_container_returns_session() -> None:
    transport = _FakeTransport()
    client = Client(transport)  # type: ignore[arg-type]

    session = await client.acquire_container(image="busybox:1")

    assert isinstance(session, ClusterContainerSession)
    assert session.rollout_id == "r-1"


@pytest.mark.asyncio
async def test_client_acquire_container_pipeline_full_flow() -> None:
    """End-to-end SDK happy path through the fake transport:
    acquire → exec → destroy via the async context manager."""
    transport = _FakeTransport()
    client = Client(transport)  # type: ignore[arg-type]

    async with await client.acquire_container(
        image="busybox:1",
        command=["sleep", "infinity"],
    ) as session:
        result = await session.exec(["echo", "hi"])
        assert result.exit_code == 0
        assert result.stdout == b"hi\n"

    assert session.destroyed
    assert len(transport.destroy_calls) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Raw-session keepalive (SDK side): batched heartbeats so the control plane can
# reap a dead consumer's sessions without false-reaping a live one.
# ──────────────────────────────────────────────────────────────────────────────


async def test_raw_keepalive_first_beat_is_prompt_not_after_interval() -> None:
    # Audit M1: a consumer hard-killed shortly after acquire must still opt into
    # the liveness reaper — so the FIRST beat is prompt, not delayed a full
    # interval. With a long interval, registering still produces a beat almost
    # immediately (pre-fix this slept the whole interval before the first beat,
    # leaving the session in deadline-only fallback).
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    beats: list[list[str]] = []

    class _T:
        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            beats.append(list(rollout_ids))

    ka = _RawSessionKeepalive(_T(), interval_s=100.0)  # type: ignore[arg-type]
    ka.register("a")
    await asyncio.sleep(0.05)  # << interval — would see no beat before the fix
    assert beats and beats[0] == ["a"]  # opted in without waiting the interval

    # A session registered later also opts in promptly: register() wakes the
    # loop so the next beat fires now, not at the next periodic tick.
    beats.clear()
    ka.register("b")
    await asyncio.sleep(0.05)
    assert beats and set(beats[0]) == {"a", "b"}

    await ka.close()


async def test_raw_keepalive_register_during_heartbeat_beats_promptly() -> None:
    # Audit M1 (follow-up): a session registered *while a heartbeat RPC is in
    # flight* must still opt in promptly. The lost-wake race was: the loop
    # cleared _wake AFTER the in-flight beat, dropping the registration's
    # signal, then waited a full interval before beating the new id. With a
    # long interval, the new id must still be beaten almost immediately.
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    beats: list[list[str]] = []
    first_beat_in_flight = asyncio.Event()
    release_first_beat = asyncio.Event()

    class _T:
        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            snap = list(rollout_ids)
            beats.append(snap)
            if snap == ["a"]:
                # First beat: signal we're mid-RPC, then block so the test can
                # register "b" while this heartbeat is still in flight.
                first_beat_in_flight.set()
                await release_first_beat.wait()

    ka = _RawSessionKeepalive(_T(), interval_s=100.0)  # type: ignore[arg-type]
    ka.register("a")
    await first_beat_in_flight.wait()  # first beat ([a]) is in flight
    ka.register("b")                   # registers DURING the in-flight beat
    release_first_beat.set()           # let the first beat complete
    await asyncio.sleep(0.05)          # << interval — pre-fix "b" waited 100s
    assert any(set(batch) == {"a", "b"} for batch in beats), beats
    await ka.close()


async def test_raw_keepalive_beats_then_unregisters_then_closes() -> None:
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    beats: list[list[str]] = []

    class _T:
        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            beats.append(list(rollout_ids))

    ka = _RawSessionKeepalive(_T(), interval_s=0.01)  # type: ignore[arg-type]
    ka.register("a")
    ka.register("b")
    await asyncio.sleep(0.05)
    assert beats, "keepalive should have beaten the live sessions"
    assert set(beats[-1]) == {"a", "b"}  # one batched RPC carries both ids

    # Unregister one → it stops being beaten.
    ka.unregister("a")
    beats.clear()
    await asyncio.sleep(0.05)
    assert beats and all(batch == ["b"] for batch in beats)

    # Close → loop cancelled → no more beats.
    await ka.close()
    beats.clear()
    await asyncio.sleep(0.05)
    assert beats == []


async def test_client_acquire_registers_destroy_deregisters_keepalive() -> None:
    from xrlenv.client.client import Client
    from xrlenv.control.service import RawAcquireResult

    class _T:
        def __init__(self) -> None:
            self.destroyed: list[str] = []

        async def acquire_container(self, **_kw: object) -> RawAcquireResult:
            return RawAcquireResult(
                rollout_id="r1", container_id="c1",
                container_name="n1", node_id="node-A",
            )

        async def destroy_container(self, **kw: object) -> None:
            self.destroyed.append(kw["rollout_id"])  # type: ignore[arg-type]

        async def heartbeat_many(self, rollout_ids: list[str]) -> None: ...
        async def close(self) -> None: ...

    transport = _T()
    client = Client(transport)  # type: ignore[arg-type]
    session = await client.acquire_container(image="busybox:1")
    # acquire registered the session with the keepalive.
    assert "r1" in client._keepalive._ids
    # destroy deregisters it (and tears down the container).
    await session.destroy()
    assert "r1" not in client._keepalive._ids
    assert transport.destroyed == ["r1"]
    await client.close()
