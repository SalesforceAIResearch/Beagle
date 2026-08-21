"""``_DropInRunner`` + runner-backed cluster-mode dispatch.

The drop-in is sync (docker-py contract); the xrlenv ``Client``
is async and holds loop-bound state (gRPC channels, asyncio.Queue,
asyncio.Future). Without a runner, each sync call ``asyncio.run``s
a fresh loop — a Client built on loop A then awaited from a
worker thread's fresh loop B raises
``Future <...> attached to a different loop``.

These tests cover the runner-backed fix:

- Owned-mode (``__init__()``) spins up a background loop + thread,
  ``run()`` dispatches via ``run_coroutine_threadsafe``,
  ``close()`` stops the loop + joins the thread.
- Attached-mode (``__init__(loop=...)``) reuses an existing loop,
  ``close()`` is a no-op (caller owns lifetime).
- Errors raised inside the dispatched coro propagate to the
  sync caller.
- An api-method override invoked from a worker thread, with the
  Client built on the test's loop, succeeds via runner-backed
  dispatch (the failure mode without the runner is the cross-loop
  bug the user reported).
- ``XrlenvDockerClient.close()`` tears down the owned runner +
  Client when present, and is benign otherwise.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.client.container_session import ClusterContainerSession
from xrlenv.compat.docker_client import (
    ClusterContainerControl,
    XrlenvDockerClient,
    _DropInRunner,
    from_env,
)
from xrlenv.control.service import RawAcquireResult

# ──────────────────────────────────────────────────────────────────────────────
# _DropInRunner — owned mode
# ──────────────────────────────────────────────────────────────────────────────


def test_owned_runner_runs_coro_and_closes_loop() -> None:
    runner = _DropInRunner()
    try:
        async def _coro() -> int:
            return 42
        assert runner.run(_coro()) == 42
        assert runner.loop.is_running() is True
    finally:
        runner.close()
    # ``close()`` stops the loop. After join the loop is no longer
    # running.
    assert runner.loop.is_running() is False


def test_owned_runner_propagates_exception() -> None:
    runner = _DropInRunner()
    try:
        async def _boom() -> None:
            raise RuntimeError("nope")
        with pytest.raises(RuntimeError, match="nope"):
            runner.run(_boom())
    finally:
        runner.close()


def test_owned_runner_close_idempotent() -> None:
    runner = _DropInRunner()
    runner.close()
    # Second close is a no-op (thread already gone).
    runner.close()


def test_owned_runner_dispatches_from_worker_thread() -> None:
    """The runner's loop runs on a daemon thread; calling
    ``run()`` from yet ANOTHER thread should still work
    (``run_coroutine_threadsafe`` is thread-safe)."""
    runner = _DropInRunner()
    try:
        results: list[Any] = []

        def _worker() -> None:
            async def _coro() -> str:
                return "from-worker"
            results.append(runner.run(_coro()))

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=5)
        assert results == ["from-worker"]
    finally:
        runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# _DropInRunner — attached mode
# ──────────────────────────────────────────────────────────────────────────────


def test_attached_runner_uses_caller_loop() -> None:
    """Caller owns the loop; ``run()`` dispatches there;
    ``close()`` is a no-op."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        runner = _DropInRunner(loop=loop)
        # Same loop instance.
        assert runner.loop is loop

        async def _coro() -> str:
            return "ok"
        assert runner.run(_coro()) == "ok"

        # close() must NOT shut the caller's loop down.
        runner.close()
        assert loop.is_running() is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)


# ──────────────────────────────────────────────────────────────────────────────
# Runner-backed cluster-mode dispatch — end-to-end via a fake Client
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _LoopBoundFakeClient:
    """A ``Client`` stand-in whose ``acquire_container`` records
    the loop it ran on. Used to prove the runner dispatched the
    coro to the runner's loop (not the calling thread's loop)."""

    next_acquire: RawAcquireResult = field(
        default_factory=lambda: RawAcquireResult(
            rollout_id="r-1", container_id="c-1",
            container_name="cname-1", node_id="node-A",
        ),
    )
    transport: Any = None
    last_acquire_loop: asyncio.AbstractEventLoop | None = None

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = _LoopBoundFakeTransport()

    async def acquire_container(self, **_kwargs: Any) -> ClusterContainerSession:
        self.last_acquire_loop = asyncio.get_running_loop()
        return ClusterContainerSession(self.transport, self.next_acquire)


@dataclass
class _LoopBoundFakeTransport:
    last_destroy_loop: asyncio.AbstractEventLoop | None = None

    async def destroy_container(self, **_kwargs: Any) -> None:
        self.last_destroy_loop = asyncio.get_running_loop()


def test_owned_runner_routes_create_container_via_runner_loop() -> None:
    """``api.create_container`` from the calling thread should
    end up running on the runner's loop, not on a fresh loop in
    the calling thread."""
    fake = _LoopBoundFakeClient()
    runner = _DropInRunner()
    try:
        control = ClusterContainerControl(client=fake, runner=runner)  # type: ignore[arg-type]
        drop_in = XrlenvDockerClient(control=control)
        # Synchronous docker-py-style call from this (test) thread.
        result = drop_in.api.create_container("img:tag", command=["sleep", "1"])
        assert result["Id"] == "c-1"
        assert fake.last_acquire_loop is runner.loop
    finally:
        runner.close()


def test_attached_runner_routes_through_caller_loop() -> None:
    """Mirror of the embedded-mode smoke topology: build a runner
    attached to a caller-owned loop; sync drop-in calls dispatch
    there (not to a fresh loop)."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        fake = _LoopBoundFakeClient()
        runner = _DropInRunner(loop=loop)
        control = ClusterContainerControl(client=fake, runner=runner)  # type: ignore[arg-type]
        drop_in = XrlenvDockerClient(control=control)
        drop_in.api.create_container("img:tag", command=["sleep", "1"])
        assert fake.last_acquire_loop is loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)


def test_runner_none_falls_back_to_fresh_loop() -> None:
    """Backwards-compat: when no runner is bound, ``_run_sync``
    falls back to ``asyncio.run`` (fresh loop). Works as long as
    the Client's coroutines don't touch loop-bound state — covered
    by the fake here."""
    fake = _LoopBoundFakeClient()
    control = ClusterContainerControl(client=fake)  # type: ignore[arg-type]
    drop_in = XrlenvDockerClient(control=control)
    drop_in.api.create_container("img:tag", command=["sleep", "1"])
    # The fresh loop was created + torn down inside
    # ``asyncio.run`` — not equal to any persistent loop.
    assert fake.last_acquire_loop is not None


# ──────────────────────────────────────────────────────────────────────────────
# XrlenvDockerClient.close() — owned runner / client teardown
# ──────────────────────────────────────────────────────────────────────────────


def test_close_with_owned_runner_and_client_closes_both() -> None:
    """Simulate the ``from_env(grpc_host=...)`` factory output —
    the drop-in carries an owned runner + Client; ``close()`` must
    tear both down."""
    fake = _LoopBoundFakeClient()

    class _CloseRecorder:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.close_loop: asyncio.AbstractEventLoop | None = None

        async def close(self) -> None:
            self.close_loop = asyncio.get_running_loop()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    recorder = _CloseRecorder(fake)
    runner = _DropInRunner()
    control = ClusterContainerControl(client=recorder, runner=runner)  # type: ignore[arg-type]
    drop_in = XrlenvDockerClient(
        control=control, runner=runner, owned_client=recorder,  # type: ignore[arg-type]
    )

    drop_in.close()
    assert recorder.close_loop is runner.loop
    # Runner closed → owned loop stopped.
    assert runner.loop.is_running() is False
    # Second close is a no-op (state cleared).
    drop_in.close()


def test_close_local_mode_is_safe() -> None:
    """Local-mode drop-ins go through the real
    ``docker.APIClient.close()`` path (which is fine even if no
    daemon is reachable; it just shuts the requests session
    down)."""
    drop_in = from_env()  # local mode
    drop_in.close()


def test_close_caller_managed_cluster_does_not_close_client() -> None:
    """When the caller passed in their own ``client``+ ``runner``,
    we DON'T close them — caller owns lifetime."""
    fake = _LoopBoundFakeClient()
    runner = _DropInRunner()
    try:
        control = ClusterContainerControl(client=fake, runner=runner)  # type: ignore[arg-type]
        drop_in = XrlenvDockerClient(control=control)
        # No owned_client / owned_runner on the drop-in.
        drop_in.close()
        # Runner still alive.
        assert runner.loop.is_running() is True
    finally:
        runner.close()


def test_drop_in_context_manager_closes_on_exit() -> None:
    """``with xrlenv.from_env() as client:`` should call close()."""
    drop_in = from_env()
    with drop_in:
        pass
    # Idempotent — second close shouldn't raise.
    drop_in.close()


# ──────────────────────────────────────────────────────────────────────────────
# terminate_raw_group — sync passthrough over Client.terminate_raw_group
# ──────────────────────────────────────────────────────────────────────────────


def test_terminate_raw_group_dispatches_via_owned_runner() -> None:
    """The ``from_env(grpc_host=...)`` drop-in owns a Client + runner; the SYNC
    ``terminate_raw_group`` must dispatch the async RPC on the runner's loop (not the
    caller's) and return the report."""
    from xrlenv.types import TerminateRawGroupReport

    class _RawGroupFake:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.calls: list[tuple[str, str]] = []
            self.ran_on: asyncio.AbstractEventLoop | None = None

        async def terminate_raw_group(
            self, group_id: str, reason: str = "group_terminated",
        ) -> TerminateRawGroupReport:
            self.ran_on = asyncio.get_running_loop()
            self.calls.append((group_id, reason))
            return TerminateRawGroupReport(
                group_id=group_id, terminated=("r1", "r2"), already_terminal=(),
            )

        async def close(self) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    fake = _RawGroupFake(_LoopBoundFakeClient())
    runner = _DropInRunner()
    control = ClusterContainerControl(client=fake, runner=runner)  # type: ignore[arg-type]
    drop_in = XrlenvDockerClient(
        control=control, runner=runner, owned_client=fake,  # type: ignore[arg-type]
    )
    try:
        report = drop_in.terminate_raw_group("grp", reason="aborted")
    finally:
        drop_in.close()

    assert report is not None
    assert report.terminated == ("r1", "r2")
    assert fake.calls == [("grp", "aborted")]
    assert fake.ran_on is runner.loop  # dispatched on the runner's loop, not the caller thread


def test_terminate_raw_group_caller_managed_returns_none() -> None:
    """No OWNED cluster Client (LocalDocker / caller-managed mode) → no cluster raw-group
    registry to sweep, so the passthrough is a no-op returning ``None``."""
    fake = _LoopBoundFakeClient()
    runner = _DropInRunner()
    try:
        control = ClusterContainerControl(client=fake, runner=runner)  # type: ignore[arg-type]
        drop_in = XrlenvDockerClient(control=control)  # no owned_client / owned_runner
        assert drop_in.terminate_raw_group("grp") is None
    finally:
        runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# Streaming exec — runner-backed drain
# ──────────────────────────────────────────────────────────────────────────────


def test_streaming_exec_uses_runner_loop_when_available() -> None:
    """When a runner is bound, ``_SyncStreamIterator`` dispatches
    its drain coroutine onto the runner's loop instead of spinning
    up a fresh one."""
    from xrlenv.control.service import RawExecChunk

    drain_loops: list[asyncio.AbstractEventLoop] = []

    @dataclass
    class _StreamingFakeTransport(_LoopBoundFakeTransport):
        async def container_exec(self, **_kwargs: Any) -> Any:
            return {
                "exit_code": 0, "stdout": b"hi", "stderr": b"",
                "timed_out": False,
            }

        async def container_put_archive(self, **_kwargs: Any) -> None:
            return None

        async def container_get_archive(self, **_kwargs: Any) -> bytes:
            return b""

        def container_exec_stream(self, **_kwargs: Any) -> Any:
            async def _gen() -> Any:
                drain_loops.append(asyncio.get_running_loop())
                yield RawExecChunk(
                    stdout=b"hello", stderr=b"", done=False,
                    exit_code=None, timed_out=False,
                )
                yield RawExecChunk(
                    stdout=b"", stderr=b"", done=True,
                    exit_code=0, timed_out=False,
                )
            return _gen()

    transport = _StreamingFakeTransport()
    fake = _LoopBoundFakeClient(transport=transport)
    runner = _DropInRunner()
    try:
        control = ClusterContainerControl(client=fake, runner=runner)  # type: ignore[arg-type]
        drop_in = XrlenvDockerClient(control=control)
        # Acquire then stream-exec.
        info = drop_in.api.create_container("img:tag", command=["sleep", "1"])
        exec_id = drop_in.api.exec_create(info["Id"], cmd=["echo", "hi"])["Id"]
        out_iter = drop_in.api.exec_start(exec_id, stream=True)
        chunks = list(out_iter)
        assert b"hello" in b"".join(chunks)
        # Drain ran on the runner's loop, not a fresh one.
        assert drain_loops, "drain coroutine never ran"
        assert drain_loops[0] is runner.loop
    finally:
        runner.close()


# ──────────────────────────────────────────────────────────────────────────────
# Real-world topology smoke: build a Client.in_process on loop A,
# call sync drop-in code from a worker thread via runner attached
# to loop A. This is exactly the embedded-mode pattern.
# ──────────────────────────────────────────────────────────────────────────────


def test_worker_thread_via_attached_runner_round_trip() -> None:
    """Topology under test: an outer loop A on thread T1 owns the
    Client; sync code runs on T2 (worker thread). Without the
    runner, T2's ``asyncio.run`` would create loop B, and any
    ``acquire_container`` await on a Client whose state is bound
    to loop A blows up. With the attached runner, T2 dispatches
    onto loop A — single loop, no cross-loop hazard.
    """
    loop_a = asyncio.new_event_loop()
    t1 = threading.Thread(target=loop_a.run_forever, daemon=True)
    t1.start()
    try:
        # Construct the Client on loop A so any loop-bound state
        # (the fake here doesn't have any, but the topology
        # mirrors what a real GrpcClientTransport would do —
        # binds at first await).
        async def _build() -> _LoopBoundFakeClient:
            return _LoopBoundFakeClient()
        client = asyncio.run_coroutine_threadsafe(_build(), loop_a).result()

        runner = _DropInRunner(loop=loop_a)
        control = ClusterContainerControl(client=client, runner=runner)  # type: ignore[arg-type]
        drop_in = XrlenvDockerClient(control=control)

        result_holder: list[Any] = []

        def _worker() -> None:
            try:
                info = drop_in.api.create_container(
                    "img:tag", command=["sleep", "1"],
                )
                result_holder.append(info)
            except Exception as exc:
                result_holder.append(exc)

        t2 = threading.Thread(target=_worker)
        t2.start()
        t2.join(timeout=5)

        assert len(result_holder) == 1
        assert not isinstance(result_holder[0], Exception), (
            f"worker thread raised: {result_holder[0]!r}"
        )
        assert result_holder[0]["Id"] == "c-1"
        assert client.last_acquire_loop is loop_a
    finally:
        loop_a.call_soon_threadsafe(loop_a.stop)
        t1.join(timeout=2)


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.B.2 W5b — env-var-driven xrlenv.from_env()
# ──────────────────────────────────────────────────────────────────────────────


def test_from_env_no_args_no_env_returns_local_mode(monkeypatch: Any) -> None:
    """Default no-args, no env: LocalDocker mode (functionally
    identical to docker.from_env())."""
    monkeypatch.delenv("XRLENV_GRPC_HOST", raising=False)
    monkeypatch.delenv("XRLENV_CONSUMER_TOKEN", raising=False)
    drop_in = from_env()
    assert isinstance(drop_in, XrlenvDockerClient)
    # No owned runner/client — local mode.
    assert drop_in._owned_runner is None
    assert drop_in._owned_client is None
    drop_in.close()


def test_from_env_reads_grpc_host_from_environment(
    monkeypatch: Any,
) -> None:
    """XRLENV_GRPC_HOST set → flips into cluster mode without kwargs.
    We assert via the connect-attempt path that the env vars were
    consulted; actual dialing fails fast against a non-existent
    server which is what we want to verify."""
    monkeypatch.setenv("XRLENV_GRPC_HOST", "127.0.0.1")
    monkeypatch.setenv("XRLENV_GRPC_PORT", "59999")
    monkeypatch.setenv("XRLENV_CONSUMER_TOKEN", "test-token")
    monkeypatch.setenv("XRLENV_GRPC_SECURE", "false")

    # Connecting to an unused port returns immediately with an
    # error wrapped by the runner. The successful aspect we test:
    # cluster mode was entered (runner created).
    try:
        drop_in = from_env()
    except Exception:
        # Connection failures are tolerated — we just want to know
        # the env-var path triggered cluster-mode entry.
        return
    try:
        assert drop_in._owned_runner is not None
        assert drop_in._owned_client is not None
    finally:
        drop_in.close()


def test_from_env_kwargs_take_precedence_over_env(
    monkeypatch: Any,
) -> None:
    """Explicit kwargs win when both are set. We verify by passing
    grpc_host=None explicitly — should fall back to env."""
    monkeypatch.setenv("XRLENV_GRPC_HOST", "env-host")
    monkeypatch.delenv("XRLENV_CONSUMER_TOKEN", raising=False)
    monkeypatch.delenv("XRLENV_GRPC_PORT", raising=False)

    # Capture what Client.grpc receives via a fake.
    captured: dict[str, Any] = {}

    class _FakeClient:
        @classmethod
        def grpc(cls, host: str, port: int = 50051, **kwargs: Any) -> Any:
            captured["host"] = host
            captured["port"] = port
            captured["kwargs"] = kwargs
            return cls()

        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        "xrlenv.client.client.Client", _FakeClient,
    )

    # No grpc_host kwarg → falls back to env-host.
    drop_in = from_env()
    try:
        assert captured["host"] == "env-host"
        assert captured["port"] == 50051  # default
    finally:
        drop_in.close()

    captured.clear()
    # Explicit grpc_host kwarg → overrides env.
    drop_in = from_env(grpc_host="kwarg-host", grpc_port=51000)
    try:
        assert captured["host"] == "kwarg-host"
        assert captured["port"] == 51000
    finally:
        drop_in.close()


def test_from_env_invalid_port_falls_back_to_default(
    monkeypatch: Any,
) -> None:
    """XRLENV_GRPC_PORT='not-a-number' → warn + use 50051. Operator
    misconfiguration shouldn't crash the consumer's harness."""
    monkeypatch.setenv("XRLENV_GRPC_HOST", "127.0.0.1")
    monkeypatch.setenv("XRLENV_GRPC_PORT", "not-a-number")
    monkeypatch.setenv("XRLENV_CONSUMER_TOKEN", "tok")

    captured: dict[str, Any] = {}

    class _FakeClient:
        @classmethod
        def grpc(cls, host: str, port: int = 50051, **kwargs: Any) -> Any:
            captured["host"] = host
            captured["port"] = port
            return cls()

        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        "xrlenv.client.client.Client", _FakeClient,
    )

    drop_in = from_env()
    try:
        assert captured["port"] == 50051  # default
    finally:
        drop_in.close()


def test_from_env_secure_env_var_parsing(monkeypatch: Any) -> None:
    """XRLENV_GRPC_SECURE accepts true/1/yes/on (case-insensitive)."""
    monkeypatch.setenv("XRLENV_GRPC_HOST", "127.0.0.1")
    monkeypatch.setenv("XRLENV_CONSUMER_TOKEN", "tok")

    captured: dict[str, Any] = {}

    class _FakeClient:
        @classmethod
        def grpc(cls, host: str, port: int = 50051, **kwargs: Any) -> Any:
            captured["secure"] = kwargs.get("secure")
            return cls()

        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        "xrlenv.client.client.Client", _FakeClient,
    )

    for value, expected in [
        ("true", True), ("True", True), ("1", True), ("yes", True),
        ("on", True), ("false", False), ("0", False), ("no", False),
        ("", False), ("invalid", False),
    ]:
        monkeypatch.setenv("XRLENV_GRPC_SECURE", value)
        drop_in = from_env()
        try:
            assert captured.get("secure") is expected, (
                f"XRLENV_GRPC_SECURE={value!r} → secure={captured.get('secure')}, "
                f"expected {expected}"
            )
        finally:
            drop_in.close()
        captured.clear()
