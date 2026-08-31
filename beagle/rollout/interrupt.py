"""Ctrl-C → clean cluster teardown for a run.

When a user aborts a run with Ctrl-C, the process would normally just die and leave this run's
containers running on the cluster until xrlenv's raw-liveness reaper collects them (~120 s),
holding capacity the whole time. These scopes install a SIGINT handler that instead **actively**
tears this run's containers down (a node-confirmed destroy frees capacity immediately), then
aborts. A *second* Ctrl-C force-quits, so a hung teardown never traps the user.

Two entry points, same teardown-by-``group_id`` mechanism (containers are tagged
``xrlenv.group_id``, and ``terminate_raw_group(group_id)`` tears that cohort down):

* :func:`stop_run_on_sigint` — ``beagle evaluate``: the runtime stamps the label and exposes
  ``runtime.stop_run(run_id)``.
* :func:`stop_group_on_sigint` — the evolve eval subprocess: there's no runtime object (harbor
  owns containers), so it tags via ``rollout_metadata(group_id=…)`` and terminates through a
  fresh ``xrlenv.from_env()`` client.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

#: How long the first Ctrl-C waits for the cluster teardown to CONFIRM before it aborts anyway.
#: ``teardown()`` is a live gRPC to the cluster (``terminate_raw_group``) that can BLOCK when the
#: cluster is at its admission limit — running it inline in the signal handler would wedge the first
#: Ctrl-C indefinitely (the deferred second Ctrl-C can't even be delivered while the handler runs).
#: So we run it on a daemon thread and wait at most this long; if it's still going we abort and let
#: it finish in the background (a second Ctrl-C force-quits regardless). Small enough to feel
#: responsive, large enough for a healthy terminate to confirm.
_TEARDOWN_JOIN_TIMEOUT_S = 5.0


@contextlib.contextmanager
def _sigint_teardown(teardown: Callable[[], Any], label: str) -> Iterator[None]:
    """Install a SIGINT handler that runs ``teardown()`` once on the first Ctrl-C (then raises
    ``KeyboardInterrupt`` to abort) and hard-exits (``os._exit(130)``) on a second. ``teardown``
    is best-effort — its exceptions are swallowed so a failed teardown never masks the interrupt.

    The teardown runs on a **daemon thread bounded by** ``_TEARDOWN_JOIN_TIMEOUT_S`` so a slow/hung
    cluster call can't wedge the first Ctrl-C (the reported bug: the handler blocked inside
    ``terminate_raw_group`` while the cluster was at its admission limit, so nothing cancelled and a
    second Ctrl-C couldn't get through). If it doesn't confirm in time we abort anyway and it keeps
    tearing down in the background; the cluster reaper is the backstop.

    A no-op off the main thread (``signal.signal`` only installs there); the previous handler is
    always restored on exit."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    fired = threading.Event()

    def _run_teardown() -> None:
        try:
            report = teardown()
            n = len(report.terminated) if report is not None else 0
            print(f"[beagle] requested teardown of {n} container(s) on the cluster.",
                  file=sys.stderr, flush=True)
        except BaseException as exc:  # noqa: BLE001 — best-effort; never mask the interrupt
            print(f"[beagle] teardown failed ({exc!r}); the cluster will reap them (~120 s).",
                  file=sys.stderr, flush=True)

    def _handler(_signum: int, _frame: Any) -> None:
        if fired.is_set():
            print("\n[beagle] second Ctrl-C — exiting now.", file=sys.stderr, flush=True)
            os._exit(130)
        fired.set()
        print(f"\n[beagle] interrupted — stopping {label} on the cluster "
              f"(Ctrl-C again to force-quit)…", file=sys.stderr, flush=True)
        worker = threading.Thread(target=_run_teardown, name="beagle-teardown", daemon=True)
        worker.start()
        worker.join(_TEARDOWN_JOIN_TIMEOUT_S)
        if worker.is_alive():
            print(f"[beagle] teardown still running after {_TEARDOWN_JOIN_TIMEOUT_S:.0f}s — aborting "
                  f"now; it continues in the background (Ctrl-C again to force-quit).",
                  file=sys.stderr, flush=True)
        raise KeyboardInterrupt

    prev = signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


@contextlib.contextmanager
def stop_run_on_sigint(runtime: Any, run_id: str | None) -> Iterator[None]:
    """First Ctrl-C tears ``run_id``'s cluster containers down via ``runtime.stop_run(run_id)``
    then aborts; a second hard-exits. A transparent no-op when the runtime has no ``stop_run``
    (local/harbor path) or there's no ``run_id``."""
    stop = getattr(runtime, "stop_run", None)
    if stop is None or not run_id:
        yield
        return
    with _sigint_teardown(lambda: stop(run_id), f"run {run_id!r}"):
        yield


@contextlib.contextmanager
def stop_group_on_sigint(group_id: str | None) -> Iterator[None]:
    """First Ctrl-C tears down every cluster container tagged ``xrlenv.group_id == group_id``
    (via a fresh ``xrlenv.from_env()`` client → ``terminate_raw_group``) then aborts; a second
    hard-exits. For the evolve eval path, which has no runtime object to call ``stop_run`` on —
    pair it with ``xrlenv.rollout_metadata(group_id=group_id)`` so the containers carry the tag.
    A no-op when there's no ``group_id``."""
    if not group_id:
        yield
        return

    def _teardown() -> Any:
        import xrlenv  # noqa: PLC0415 — cluster-only; imported at teardown, not scope entry

        client = xrlenv.from_env()  # reads XRLENV_GRPC_* / token from env
        try:
            return client.terminate_raw_group(group_id)
        finally:
            with contextlib.suppress(Exception):
                client.close()

    with _sigint_teardown(_teardown, f"group {group_id!r}"):
        yield


__all__ = ["stop_run_on_sigint", "stop_group_on_sigint"]
