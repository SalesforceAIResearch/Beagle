"""Ctrl-C → cluster teardown: the ``stop_run_on_sigint`` scope + ``XrlenvDockerRuntime.stop_run``."""

from __future__ import annotations

import signal
import threading
import time
from types import SimpleNamespace

import pytest

from beagle.rollout import interrupt as interrupt_mod
from beagle.rollout.interrupt import stop_group_on_sigint, stop_run_on_sigint


class _FakeRuntime:
    def __init__(self) -> None:
        self.stopped: list[str] = []

    def stop_run(self, run_id: str) -> SimpleNamespace:
        self.stopped.append(run_id)
        return SimpleNamespace(terminated=("a", "b"), already_terminal=())


def test_sigint_stops_run_then_aborts() -> None:
    # A first Ctrl-C tears the run's containers down, then raises KeyboardInterrupt to abort.
    rt = _FakeRuntime()
    prev = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        with stop_run_on_sigint(rt, "run-1"):
            handler = signal.getsignal(signal.SIGINT)
            assert handler is not prev  # our handler is installed inside the scope
            handler(signal.SIGINT, None)  # simulate Ctrl-C, deterministically
    assert rt.stopped == ["run-1"]
    assert signal.getsignal(signal.SIGINT) is prev  # restored on exit


def test_noop_without_run_id_leaves_handler_untouched() -> None:
    rt = _FakeRuntime()
    prev = signal.getsignal(signal.SIGINT)
    with stop_run_on_sigint(rt, None):
        assert signal.getsignal(signal.SIGINT) is prev  # no handler installed
    assert rt.stopped == []


def test_noop_when_runtime_has_no_stop_run() -> None:
    prev = signal.getsignal(signal.SIGINT)
    with stop_run_on_sigint(object(), "run-1"):  # object() has no stop_run → transparent no-op
        assert signal.getsignal(signal.SIGINT) is prev


def test_slow_teardown_does_not_wedge_the_first_ctrl_c(monkeypatch) -> None:
    # The reported bug: a slow ``terminate_raw_group`` (cluster at its admission limit) ran inline in
    # the signal handler and blocked the FIRST Ctrl-C indefinitely — nothing cancelled, and the
    # deferred second Ctrl-C couldn't get through. The teardown now runs on a bounded daemon thread,
    # so the first Ctrl-C aborts within the timeout even while the cluster call is still going.
    monkeypatch.setattr(interrupt_mod, "_TEARDOWN_JOIN_TIMEOUT_S", 0.2)
    started = threading.Event()
    release = threading.Event()

    class _SlowRuntime:
        def stop_run(self, run_id: str) -> SimpleNamespace:
            started.set()
            release.wait(5)  # simulate a cluster call blocked at the admission limit
            return SimpleNamespace(terminated=(), already_terminal=())

    t0 = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            with stop_run_on_sigint(_SlowRuntime(), "run-slow"):
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)  # simulate Ctrl-C
        elapsed = time.monotonic() - t0
        assert started.is_set()   # the teardown WAS kicked off (containers being torn down)
        assert elapsed < 2.0      # but the handler did NOT block on the full 5s cluster call
    finally:
        release.set()             # let the daemon teardown thread finish


def test_teardown_failure_is_swallowed_but_still_aborts() -> None:
    # A stop_run that raises must NOT mask the interrupt — the run still aborts.
    class _Boom:
        def stop_run(self, run_id: str) -> None:
            raise RuntimeError("cluster unreachable")

    with pytest.raises(KeyboardInterrupt):
        with stop_run_on_sigint(_Boom(), "run-1"):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)


def test_stop_group_on_sigint_terminates_group(monkeypatch) -> None:
    # The evolve path has no runtime object — on Ctrl-C it terminates the group via a fresh
    # from_env() client. Assert the client's terminate_raw_group is called with the group id.
    import xrlenv

    calls: list[tuple[str, str]] = []

    class _FakeClient:
        def terminate_raw_group(self, group_id: str, reason: str = "group_terminated"):
            calls.append((group_id, reason))
            return SimpleNamespace(terminated=("a", "b"), already_terminal=())

        def close(self) -> None:
            pass

    monkeypatch.setattr(xrlenv, "from_env", lambda **_kw: _FakeClient())

    with pytest.raises(KeyboardInterrupt):
        with stop_group_on_sigint("grp-1"):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert calls == [("grp-1", "group_terminated")]


def test_already_gone_treats_session_not_found_as_benign() -> None:
    # A per-task destroy that races the Ctrl-C group teardown hits a cluster "session … not found"
    # — the container is already gone (what destroy wanted), so it's benign (no "may leak" traceback).
    from beagle.rollout.runtime.xrlenv_runtime import _already_gone

    assert _already_gone(RuntimeError(
        "raw-container session: rollout 'abc' not found. Acquire first."))
    assert not _already_gone(RuntimeError("connection refused"))


def test_stop_group_on_sigint_noop_without_group() -> None:
    prev = signal.getsignal(signal.SIGINT)
    with stop_group_on_sigint(None):  # nothing to scope by → no handler installed
        assert signal.getsignal(signal.SIGINT) is prev


def test_xrlenv_runtime_stop_run_delegates_to_terminate_raw_group(monkeypatch) -> None:
    import xrlenv

    calls: list[tuple[str, str]] = []

    class _FakeClient:
        def terminate_raw_group(self, group_id: str, reason: str = "group_terminated"):
            calls.append((group_id, reason))
            return SimpleNamespace(terminated=("x",), already_terminal=())

    monkeypatch.setattr(xrlenv, "from_env", lambda **_kw: _FakeClient())
    from beagle.rollout.runtime.xrlenv_runtime import XrlenvDockerRuntime

    rt = XrlenvDockerRuntime(run_id="run-9")
    report = rt.stop_run("run-9")
    assert calls == [("run-9", "group_terminated")]
    assert report.terminated == ("x",)
    # empty run_id → no-op (nothing to scope by), and no extra client call
    assert rt.stop_run("") is None
    assert calls == [("run-9", "group_terminated")]
