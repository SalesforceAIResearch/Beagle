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


# ──────────────────────────────────────────────────────────────────────────────
# Keepalive robustness (design note Part 2). None of this would have saved a
# session in the 2026-08-19 incident — a frozen process cannot beat with or
# without a deadline — but a wedged or silently-failing keepalive is its own way
# to lose work, and the silence is why that incident was undiagnosable from the
# consumer side.
# ──────────────────────────────────────────────────────────────────────────────


async def test_keepalive_survives_a_hung_beat_and_retries() -> None:
    """A beat that never returns must not silence the loop forever.

    The keepalive is single-threaded and swallows *errors*, but a hang is not an
    error — it just never returns. Pre-fix, one wedged RPC stopped every
    subsequent beat for that process, and every quiet session it owned died at
    the quarantine horizon.
    """
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    beats: list[list[str]] = []

    class _T:
        def __init__(self) -> None:
            self.calls = 0

        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(3600)      # wedged forever
            beats.append(list(rollout_ids))

    transport = _T()
    # A real deadline lives in the transport; here the transport IS the fake, so
    # emulate the wire timeout the production path now passes.
    orig = transport.heartbeat_many

    async def _with_deadline(ids: list[str]) -> None:
        await asyncio.wait_for(orig(ids), timeout=0.05)

    transport.heartbeat_many = _with_deadline  # type: ignore[assignment]

    ka = _RawSessionKeepalive(transport, interval_s=0.05)  # type: ignore[arg-type]
    ka.register("a")
    await asyncio.sleep(0.5)
    await ka.close()

    assert transport.calls >= 2, "loop stalled on the hung beat"
    assert beats, "no beat ever succeeded after the hang"


async def test_keepalive_flags_liveness_at_risk_after_repeated_failures() -> None:
    """Reported, never raised (decision 4).

    The SDK must not kill healthy work on its own suspicion — that is the exact
    failure this contract exists to remove — so a failing keepalive surfaces a
    flag the caller may act on, and nothing more.
    """
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    class _T:
        def __init__(self) -> None:
            self.fail = True

        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            if self.fail:
                raise RuntimeError("control plane unreachable")

    transport = _T()
    ka = _RawSessionKeepalive(transport, interval_s=0.02)  # type: ignore[arg-type]
    assert ka.liveness_at_risk is False        # nothing has failed yet

    ka.register("a")
    await asyncio.sleep(0.3)
    assert ka.liveness_at_risk is True         # sustained failure is visible

    transport.fail = False                     # control plane comes back
    await asyncio.sleep(0.2)
    assert ka.liveness_at_risk is False        # and the flag clears

    await ka.close()


def test_heartbeat_rpc_deadline_is_under_the_beat_cadence() -> None:
    """A deadline at or above the cadence would let beats pile up instead of
    failing fast, which is the stall it exists to prevent."""
    from xrlenv.api.constants import HEARTBEAT_RPC_TIMEOUT_S
    from xrlenv.client.client import _RAW_HEARTBEAT_INTERVAL_S

    assert 0 < HEARTBEAT_RPC_TIMEOUT_S < _RAW_HEARTBEAT_INTERVAL_S


def test_server_tolerates_the_ping_cadence_clients_actually_send() -> None:
    """Client and server keepalive settings must be consistent.

    A client-only change is worse than none: gRPC servers allow one ping per
    5 minutes without data by default and answer GOAWAY ``too_many_pings``
    beyond that, so the connections the pings exist to protect get torn down and
    the client's keepalive is throttled.
    """
    from xrlenv.api.constants import (
        GRPC_CHANNEL_OPTIONS,
        GRPC_KEEPALIVE_TIME_MS,
        GRPC_SERVER_MIN_PING_INTERVAL_MS,
        GRPC_SERVER_OPTIONS,
    )

    assert GRPC_SERVER_MIN_PING_INTERVAL_MS <= GRPC_KEEPALIVE_TIME_MS
    client = dict(GRPC_CHANNEL_OPTIONS)
    server = dict(GRPC_SERVER_OPTIONS)
    assert client["grpc.keepalive_time_ms"] == GRPC_KEEPALIVE_TIME_MS
    assert client["grpc.keepalive_permit_without_calls"] == 1
    assert (server["grpc.http2.min_recv_ping_interval_without_data_ms"]
            == GRPC_SERVER_MIN_PING_INTERVAL_MS)
    # The server must NOT inherit the client's ping timer — that would make it
    # ping its own clients, a different decision than permitting theirs.
    assert "grpc.keepalive_time_ms" not in server


async def test_at_risk_clears_once_there_is_nothing_left_to_beat() -> None:
    """Nothing to beat means nothing at risk.

    The reset only ran on a SUCCESSFUL beat, and beats are only attempted while
    sessions exist — so a client that drained its last session mid-failure-streak
    reported itself at risk forever, about sessions it no longer had.
    """
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    class _T:
        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            raise RuntimeError("unreachable")

    ka = _RawSessionKeepalive(_T(), interval_s=0.02)  # type: ignore[arg-type]
    ka.register("a")
    await asyncio.sleep(0.2)
    assert ka.liveness_at_risk is True

    ka.unregister("a")                 # last session gone
    await asyncio.sleep(0.15)
    assert ka.liveness_at_risk is False

    await ka.close()
    assert ka.liveness_at_risk is False


async def test_beat_budget_stays_under_any_configured_cadence() -> None:
    """The wire deadline is fixed but the cadence is operator-configurable.

    At XRLENV_RAW_HEARTBEAT_INTERVAL_S=5 a fixed 10s deadline inverts the
    relationship: a wedged beat stretches the effective cadence to
    deadline+interval, eating the margin the 120s TTL is sized against. The
    per-beat budget must therefore derive from the interval.
    """
    from xrlenv.client.client import _RawSessionKeepalive

    class _T:
        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            return None

    for interval in (5.0, 12.0, 30.0, 120.0):
        ka = _RawSessionKeepalive(_T(), interval_s=interval)  # type: ignore[arg-type]
        assert 0 < ka._beat_budget_s < interval, (
            f"budget {ka._beat_budget_s}s is not under a {interval}s cadence"
        )


def test_beat_budget_stays_under_even_a_sub_second_interval() -> None:
    """The per-beat budget must stay under the cadence at EVERY setting.

    A fixed deadline against a floor-less, operator-configurable cadence
    (``XRLENV_RAW_HEARTBEAT_INTERVAL_S``) inverts the invariant and lets a wedged
    beat stretch the effective interval to budget+interval. The first fix derived
    the budget from the interval but clamped it with a 0.5s floor — itself a fixed
    constant against the same configurable interval — so the identical inversion
    reappeared below ~1s (at interval=0.1 the budget was 0.5s, 5x the cadence).
    The floor is gone; this pins that it stays gone.
    """
    from xrlenv.client.client import _RawSessionKeepalive

    class _T:
        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            return None

    for interval in (0.05, 0.1, 0.2, 0.5, 0.9):
        ka = _RawSessionKeepalive(_T(), interval_s=interval)  # type: ignore[arg-type]
        assert 0 < ka._beat_budget_s < interval, (
            f"budget {ka._beat_budget_s}s is not under a {interval}s cadence "
            "-- the 0.5s floor dominates below ~1s and inverts the same "
            "deadline-vs-cadence relationship the derived budget exists to fix"
        )


async def test_close_terminates_promptly_with_a_beat_in_flight() -> None:
    """``close()`` must cancel a wedged beat and return -- not hang, and not
    leak the gRPC-call-equivalent task the transport was awaiting on.

    Regression guard for the ``asyncio.wait_for`` + ``close()`` interaction
    introduced by e04b579: the beat is now wrapped in ``wait_for``, and
    ``close()`` cancels the outer ``_run`` task while a beat may be
    in-flight. This passes today -- recorded as a positive lock-in, not a
    defect -- but the interaction is exactly the kind that silently breaks
    under a refactor (e.g. swapping ``wait_for`` for a raw
    ``asyncio.timeout()`` block, or moving the transport call off-loop).
    """
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    class _HangingTransport:
        def __init__(self) -> None:
            self.calls = 0
            self.cancelled = 0

        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            self.calls += 1
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

    transport = _HangingTransport()
    ka = _RawSessionKeepalive(transport, interval_s=0.05)  # type: ignore[arg-type]
    ka.register("a")
    await asyncio.sleep(0.02)  # let the beat actually start (in-flight, not yet timed out)
    assert transport.calls == 1
    assert transport.cancelled == 0  # confirm we're testing the in-flight case

    close_task = asyncio.create_task(ka.close())
    await asyncio.wait_for(close_task, timeout=1.0)  # close() must not hang

    assert transport.cancelled == 1, (
        "close() returned without the in-flight beat's CancelledError "
        "actually completing -- a leaked task, not a clean cancel"
    )
    remaining = [
        t for t in asyncio.all_tasks() if t is not asyncio.current_task()
    ]
    assert remaining == [], f"tasks leaked past close(): {remaining}"


async def test_liveness_at_risk_is_exposed_on_the_session_handle() -> None:
    """The signal has to live where the caller already looks.

    Decision 4 promised it "surfaced on the session handle". It shipped only on
    Client, and the documented usage pattern is
    ``async with await client.acquire_container(...) as session:`` — so a harness
    holding a session had nothing to read, and no shipped integration read it.
    """
    from xrlenv.client.container_session import ClusterContainerSession

    class _R:
        rollout_id = "r1"
        container_id = "c1"
        container_name = "n1"
        node_id = "node-A"
        queue_wait_s = 0.0

    at_risk = False
    session = ClusterContainerSession(
        object(), _R(), liveness_probe=lambda: at_risk,  # type: ignore[arg-type]
    )
    assert session.liveness_at_risk is False
    at_risk = True
    assert session.liveness_at_risk is True

    # A session built without a probe must not explode — the attribute is
    # advisory, and the in-process paths do not wire one.
    assert ClusterContainerSession(object(), _R()).liveness_at_risk is False  # type: ignore[arg-type]


def test_heartbeat_interval_contract_across_the_whole_input_space() -> None:
    """One test for the whole input space, because patching cases lost three times.

    NaN was fixed, then +inf, then 1e300 — each a different literal reaching the
    identical failure: the loop beats once at registration and never again, so the
    consumer looks healthy while the control plane hears nothing and reaps its
    idle sessions. The contract has three tiers, and this pins all of them:

    * a usable cadence is taken as given (0 stays a documented "disable");
    * a cadence at or above the control plane's DEFAULT liveness TTL is ACCEPTED
      but warned about — the server's TTL may have been raised to match, and we
      cannot see it from here, so refusing would break a legitimate setup;
    * anything that is not a cadence at all — unparseable, negative, non-finite,
      or absurd on its face — falls back to the default loudly.
    """
    import logging
    import os
    from unittest.mock import patch

    from xrlenv.client.client import _read_heartbeat_interval_s

    def read(value: str) -> float:
        with patch.dict(os.environ, {"XRLENV_RAW_HEARTBEAT_INTERVAL_S": value}):
            return _read_heartbeat_interval_s()

    # taken as given
    assert read("7") == 7.0
    assert read("30") == 30.0
    assert read("0") == 0.0          # documented disable, not an error

    # accepted, but must not pass silently -- exercise both edges of the band,
    # not just its low end: 120 is the TTL-hint floor, 86399.999999 is the
    # largest value the ceiling still lets through (86400 itself is rejected,
    # covered in the junk loop below).
    for plausible_but_wrong in ("120", "300", "86399.999999"):
        with patch.object(logging.getLogger("xrlenv.client.client"), "warning") as warn:
            assert read(plausible_but_wrong) == float(plausible_but_wrong)
            assert warn.called, f"{plausible_but_wrong}s cannot keep a session alive, unwarned"

    # not a cadence at all -> default
    for junk in ("nan", "inf", "+inf", "Infinity", "1e400", "1e300", "86400",
                 "-5", "-inf", "not-a-number", ""):
        assert read(junk) == 30.0, f"{junk!r} was accepted as an interval"


def test_unusable_heartbeat_interval_rejects_infinity() -> None:
    import os
    from unittest.mock import patch

    from xrlenv.client.client import _read_heartbeat_interval_s

    for bad in ("inf", "+inf", "Infinity", "1e400"):
        with patch.dict(os.environ, {"XRLENV_RAW_HEARTBEAT_INTERVAL_S": bad}):
            assert _read_heartbeat_interval_s() == 30.0, (
                f"{bad!r} was not rejected -- it parses to a positive infinite "
                "float that the NaN/negative guard does not catch"
            )


async def test_infinite_interval_beats_once_then_never_again() -> None:
    """Runtime consequence of the ``inf`` gap above, isolated from env parsing.

    Even if a caller supplies ``interval_s=float("inf")`` directly to
    ``_RawSessionKeepalive`` (bypassing env parsing entirely), the beat loop
    still wedges after the first beat: ``asyncio.wait_for(wake.wait(),
    timeout=inf)`` never times out, so the periodic re-beat that keeps the
    control plane's liveness reaper convinced this consumer is alive never
    fires. This is the exact "looks healthy, control plane hears nothing"
    failure mode the interval-validation fix exists to prevent -- it just
    isn't reachable through env-var validation for this specific value.
    """
    import asyncio

    from xrlenv.client.client import _RawSessionKeepalive

    calls = 0

    class _T:
        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            nonlocal calls
            calls += 1

    ka = _RawSessionKeepalive(_T(), interval_s=float("inf"))  # type: ignore[arg-type]
    ka.register("a")
    await asyncio.sleep(0.3)  # far longer than any sane operator cadence
    assert calls == 1, (
        f"expected exactly one beat (the registration opt-in) with "
        f"interval_s=inf, got {calls} -- if this is >1 the wedge was fixed"
    )
    await ka.close()


async def test_liveness_at_risk_reflects_real_keepalive_state_via_acquire_container() -> None:
    """Drive the actual ``Client.acquire_container`` production path, not the
    session class in isolation.

    Every prior fix to this contract shipped correct at the layer it edited
    and inert on the path production actually takes (wired into the keepalive,
    then into ``Client.__init__``, then only reachable via a constructor no
    real caller used). The regression this contract needs pinned is: does the
    SESSION returned by ``acquire_container`` -- what a harness actually holds
    under ``async with await client.acquire_container(...) as session:`` --
    report ``liveness_at_risk`` truthfully when the real keepalive is failing
    and recovers when it stops failing, end to end through the acquire call
    and not via a hand-constructed ``liveness_probe`` lambda.
    """
    import asyncio

    class _T:
        def __init__(self) -> None:
            self.fail = True

        async def acquire_container(self, **kwargs: Any) -> RawAcquireResult:
            return RawAcquireResult(
                rollout_id="r-1", container_id="c-1",
                container_name="cname-1", node_id="node-A",
            )

        async def heartbeat_many(self, rollout_ids: list[str]) -> None:
            if self.fail:
                raise RuntimeError("control plane unreachable")

        async def destroy_container(self, **kwargs: Any) -> None:
            pass

    transport = _T()
    client = Client(transport)  # type: ignore[arg-type]
    # Fast cadence so the test doesn't wait on the real 30s default.
    client._keepalive._interval_s = 0.02
    client._keepalive._beat_budget_s = 0.01

    session = await client.acquire_container(image="busybox:1")
    assert session.liveness_at_risk is False  # nothing has failed yet

    await asyncio.sleep(0.3)
    assert session.liveness_at_risk is True, (
        "session.liveness_at_risk did not go True on the real "
        "acquire_container() path despite sustained heartbeat failure -- "
        "the flag is wired to something that never changes"
    )

    transport.fail = False  # control plane comes back
    await asyncio.sleep(0.2)
    assert session.liveness_at_risk is False, (
        "session.liveness_at_risk did not clear on the real "
        "acquire_container() path after the keepalive recovered"
    )

    await session.destroy()
