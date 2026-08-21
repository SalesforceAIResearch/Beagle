"""Unit tests for node-saturation create recovery — approach A + C.

Covers ``RawContainerManager._create_with_retry`` and its helpers:

* **A (recovery):** a transient busy-daemon create fault (5xx / timeout — e.g.
  ``sysbox-fs pre-register … DeadlineExceeded`` under a create burst) is retried
  with bounded exponential backoff instead of failing the acquire.
* **AIMD accounting:** exactly ONE health error is recorded per acquire that hit
  any health fault — not one per attempt (that would collapse the node's
  admission limit) and not only on give-up (a retry-then-success node WAS
  saturated).
* **Retryable set:** ``_is_retryable_create_error`` retries 5xx / timeout but NOT
  a clean dead-daemon ``ConnectionError`` (which still feeds AIMD) nor 4xx.
* **No duplicate leak:** an ambiguous timeout that actually created a container is
  reaped by the unique ``xrlenv.rollout_id`` label before the retry recreates.
* **Bounded wait:** the total wall-clock cap short-circuits the series.
* **C (prevention) + observability:** the sysbox-specific create gate serializes
  non-runc creates and is folded into ``health_snapshot``.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import docker.errors
import pytest
import requests.exceptions
from xrlenv.errors import XRLEnvError
from xrlenv.node import raw_container as rc
from xrlenv.node.raw_container import (
    RawContainerManager,
    _is_retryable_create_error,
)

# ──────────────────────────────────────────────────────────────────────────────
# APIError factory (mirrors the name-reclaim test's — status via .response)
# ──────────────────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.url = "http+docker://localhost/v1.54/containers/create"
        self.reason = "Server Error" if status_code >= 500 else "Error"


def _api_error(status: int, message: str) -> docker.errors.APIError:
    return docker.errors.APIError(message, response=_Resp(status))


# ──────────────────────────────────────────────────────────────────────────────
# _is_retryable_create_error — the narrower retry predicate
# ──────────────────────────────────────────────────────────────────────────────


def test_5xx_is_retryable() -> None:
    assert _is_retryable_create_error(_api_error(500, "boom")) is True


def test_timeout_is_retryable() -> None:
    assert _is_retryable_create_error(
        requests.exceptions.ReadTimeout("timed out"),
    ) is True


def test_clean_connection_error_is_not_retryable() -> None:
    """A down-daemon ConnectionError feeds AIMD (via _is_node_health_error) but
    is NOT retried in place — a 31s backoff can't revive a down daemon."""
    exc = requests.exceptions.ConnectionError("connection refused")
    assert _is_retryable_create_error(exc) is False
    assert rc._is_node_health_error(exc) is True  # still an AIMD signal


def test_409_and_404_are_not_retryable() -> None:
    assert _is_retryable_create_error(_api_error(409, "in use")) is False
    assert _is_retryable_create_error(_api_error(404, "no image")) is False


def test_statusless_non_timeout_is_not_retryable() -> None:
    assert _is_retryable_create_error(
        docker.errors.DockerException("broken pipe"),
    ) is False


# ──────────────────────────────────────────────────────────────────────────────
# Scripted docker fake — run() consumes a per-call action list
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeContainer:
    id: str
    name: str
    labels: dict[str, str]
    removed: bool = False
    raise_on_remove: BaseException | None = None

    def remove(self, *, force: bool = False) -> None:
        if self.raise_on_remove is not None:
            raise self.raise_on_remove
        self.removed = True


class _FakeImages:
    def get(self, image: str) -> Any:
        return object()


class _ScriptedContainers:
    """``run`` consumes one action per call from ``script`` (default 'ok' when
    exhausted). Actions: 'ok', '500', '404', 'conn', '500_after_create' (register
    an orphan wearing the acquire's labels, THEN raise 500 — the ambiguous
    timeout that actually created a container)."""

    def __init__(self, script: list[str], *, list_raises: bool = False) -> None:
        self.script = list(script)
        self._by_id: dict[str, _FakeContainer] = {}
        self._name_to_id: dict[str, str] = {}
        self._next = 0
        self.run_calls = 0
        self.list_raises = list_raises  # reap's list() raises DockerException

    def _register(self, c: _FakeContainer) -> None:
        self._by_id[c.id] = c
        self._name_to_id[c.name] = c.id

    def _make(self, name: str | None, labels: dict[str, str]) -> _FakeContainer:
        self._next += 1
        cid = f"c-{self._next:04d}"
        c = _FakeContainer(id=cid, name=name or cid, labels=dict(labels))
        self._register(c)
        return c

    def run(self, *, name: str | None = None, labels: dict[str, str],
            **_: Any) -> _FakeContainer:
        self.run_calls += 1
        action = self.script.pop(0) if self.script else "ok"
        if action == "500":
            raise _api_error(500, "pre-register with sysbox-fs: DeadlineExceeded")
        if action == "404":
            raise _api_error(404, "image not found")
        if action == "conn":
            raise requests.exceptions.ConnectionError("connection refused")
        if action == "500_after_create":
            # Docker DID create the container; the client just never saw the
            # reply. The orphan wears the acquire's rollout_id label.
            self._make(name, labels)
            raise _api_error(500, "pre-register with sysbox-fs: DeadlineExceeded")
        if action == "500_after_create_stuck":
            # Same, but the orphan refuses to be removed (reap fails → the retry
            # loop must fail closed rather than stack a duplicate).
            orphan = self._make(name, labels)
            orphan.raise_on_remove = _api_error(500, "remove failed")
            raise _api_error(500, "pre-register with sysbox-fs: DeadlineExceeded")
        return self._make(name, labels)

    def get(self, key: str) -> _FakeContainer:
        c = self._by_id.get(key) or self._by_id.get(self._name_to_id.get(key, ""))
        if c is None or c.removed:
            raise docker.errors.NotFound(f"no such container: {key}")
        return c

    def list(self, *, all: bool = False,
             filters: dict[str, str] | None = None, **_: Any,
             ) -> list[_FakeContainer]:
        if self.list_raises:
            raise docker.errors.APIError("500 Server Error: cannot list")
        live = [c for c in self._by_id.values() if not c.removed]
        if filters and "label" in filters:
            k, _sep, v = str(filters["label"]).partition("=")
            return [c for c in live if c.labels.get(k) == v]
        return [c for c in live if c.labels.get("xrlenv.session_kind") == "raw"]


class _FakeClient:
    def __init__(self, containers: Any,
                 runtimes: set[str] | None = None) -> None:
        self.images = _FakeImages()
        self.containers = containers
        self.api = type("_API", (), {"timeout": 600.0})()
        self._runtimes = runtimes or {"runc"}

    def info(self) -> dict[str, Any]:
        return {
            "Runtimes": {r: {} for r in self._runtimes},
            "DefaultRuntime": "runc",
        }


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the backoff so retry tests don't actually wait ~31s (jitter only
    shrinks the already-tiny wait, so no determinism patch is needed).
    Individual tests that exercise the caps override these."""
    monkeypatch.setattr(rc, "_HEALTH_RETRY_BASE_S", 0.001)
    monkeypatch.setattr(rc, "_HEALTH_RETRY_CAP_S", 0.001)


# ──────────────────────────────────────────────────────────────────────────────
# A — retry recovery + AIMD accounting
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transient_500_then_success_recovers() -> None:
    containers = _ScriptedContainers(["500", "500", "ok"])
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    rec = await mgr.acquire(rollout_id="r-1", image="busybox:1")

    assert rec.rollout_id == "r-1"
    assert containers.run_calls == 3          # two transient 500s + success
    snap = mgr._health.snapshot()
    assert snap.docker_error_count == 1       # ONE AIMD signal for the acquire
    assert snap.create_count == 1             # only the successful create timed


@pytest.mark.asyncio
async def test_persistent_500_gives_up_after_max_with_one_aimd_error() -> None:
    containers = _ScriptedContainers(["500"] * 20)  # always saturated
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    with pytest.raises(XRLEnvError):
        await mgr.acquire(rollout_id="r-2", image="busybox:1")

    # _HEALTH_RETRY_MAX(5) retries → 6 attempts total.
    assert containers.run_calls == rc._HEALTH_RETRY_MAX + 1
    # Exactly one health error despite six failed attempts — a burst must not
    # multiplicatively collapse the node's AIMD admission limit.
    assert mgr._health.snapshot().docker_error_count == 1


@pytest.mark.asyncio
async def test_retry_then_success_still_records_one_health() -> None:
    """A create that retries and succeeds still leaves one AIMD signal — the
    node WAS saturated, so future admits should throttle."""
    containers = _ScriptedContainers(["500", "ok"])
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    await mgr.acquire(rollout_id="r-3", image="busybox:1")

    assert mgr._health.snapshot().docker_error_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# Retryable set — 404 terminal, clean ConnectionError not retried (but AIMD)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_404_is_not_retried() -> None:
    containers = _ScriptedContainers(["404"])
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    with pytest.raises(XRLEnvError):
        await mgr.acquire(rollout_id="r-4", image="busybox:1")

    assert containers.run_calls == 1                       # no retry
    assert mgr._health.snapshot().docker_error_count == 0  # 4xx is not health


@pytest.mark.asyncio
async def test_clean_connection_error_not_retried_but_feeds_aimd() -> None:
    containers = _ScriptedContainers(["conn"])
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    with pytest.raises(XRLEnvError):
        await mgr.acquire(rollout_id="r-5", image="busybox:1")

    assert containers.run_calls == 1                       # dead daemon → no retry
    assert mgr._health.snapshot().docker_error_count == 1  # but throttle admits


# ──────────────────────────────────────────────────────────────────────────────
# No duplicate-container leak on ambiguous timeout
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ambiguous_timeout_orphan_is_reaped_before_retry() -> None:
    """A create that 500s AFTER actually spawning the container leaves an orphan
    wearing the acquire's rollout_id label. The retry reaps it before recreating,
    so exactly one live container carries that label on success."""
    containers = _ScriptedContainers(["500_after_create", "ok"])
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    rec = await mgr.acquire(rollout_id="r-6", image="busybox:1")

    live = containers.list(filters={"label": "xrlenv.rollout_id=r-6"})
    assert len(live) == 1                     # the orphan was reaped
    assert live[0].id == rec.container_id     # only the successful one remains
    assert containers.run_calls == 2


@pytest.mark.asyncio
async def test_unremovable_orphan_fails_closed_no_second_create() -> None:
    """If the reap finds an orphan it CANNOT remove, the loop fails closed — it
    must not create a second container on top of a possibly-live one."""
    containers = _ScriptedContainers(["500_after_create_stuck", "ok"])
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    with pytest.raises(XRLEnvError):
        await mgr.acquire(rollout_id="r-8", image="busybox:1")

    assert containers.run_calls == 1          # the second create never ran
    orphans = containers.list(filters={"label": "xrlenv.rollout_id=r-8"})
    assert len(orphans) == 1 and not orphans[0].removed  # orphan left for raw-GC


@pytest.mark.asyncio
async def test_reap_list_failure_fails_closed() -> None:
    """If the pre-retry orphan list itself fails, cleanliness can't be confirmed —
    fail closed rather than risk recreating over an unknown orphan."""
    containers = _ScriptedContainers(["500", "ok"], list_raises=True)
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    with pytest.raises(XRLEnvError):
        await mgr.acquire(rollout_id="r-9", image="busybox:1")

    assert containers.run_calls == 1          # never retried past the failed list


# ──────────────────────────────────────────────────────────────────────────────
# Bounded wait — retries never exceed the caller's acquire wire deadline
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_respects_caller_wire_deadline() -> None:
    """A fail-fast caller (tiny acquire wire budget = ensure_image_deadline_s)
    must fail fast on the node too — one create attempt, no retry past the
    control-plane's _send_and_wait timeout."""
    containers = _ScriptedContainers(["500"] * 20)
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    with pytest.raises(XRLEnvError):
        # 0.0 budget → the anchored retry deadline is already reached after the
        # first attempt, so no backoff sleep is allowed.
        await mgr.acquire(
            rollout_id="r-10", image="busybox:1", ensure_image_deadline_s=0.0,
        )

    assert containers.run_calls == 1
    assert mgr._health.snapshot().docker_error_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# Bounded total wait — the wall-clock cap short-circuits the series
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_total_wall_clock_cap_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a tiny total cap and a full-size first backoff, the loop gives up
    before sleeping past the cap — fewer than _HEALTH_RETRY_MAX attempts."""
    monkeypatch.setattr(rc, "_HEALTH_RETRY_BASE_S", 1.0)   # first backoff ≈ 1s
    monkeypatch.setattr(rc, "_HEALTH_RETRY_CAP_S", 30.0)
    monkeypatch.setattr(rc, "_HEALTH_RETRY_TOTAL_CAP_S", 0.5)  # < first backoff
    containers = _ScriptedContainers(["500"] * 20)
    mgr = RawContainerManager(docker_client=_FakeClient(containers))

    with pytest.raises(XRLEnvError):
        await mgr.acquire(rollout_id="r-7", image="busybox:1")

    assert containers.run_calls == 1                       # capped after attempt 0
    assert mgr._health.snapshot().docker_error_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# C — sysbox create gate: routing, serialization, snapshot folding
# ──────────────────────────────────────────────────────────────────────────────


def test_create_gate_routes_sysbox_to_its_own_semaphore() -> None:
    mgr = RawContainerManager(
        docker_client=_FakeClient(_ScriptedContainers([])),
        sysbox_create_concurrency=1,
    )
    assert mgr._create_gate(sysbox=True) is mgr._sysbox_create_semaphore
    assert mgr._create_gate(sysbox=False) is mgr._create_semaphore


def test_create_gate_sysbox_falls_back_when_disabled() -> None:
    mgr = RawContainerManager(
        docker_client=_FakeClient(_ScriptedContainers([])),
        sysbox_create_concurrency=0,   # sysbox-specific gate disabled
    )
    assert mgr._sysbox_create_semaphore is None
    assert mgr._create_gate(sysbox=True) is mgr._create_semaphore


def test_destroy_gate_routes_sysbox_to_its_own_semaphore() -> None:
    # Symmetric with the create gate: a sysbox teardown must serialise on the
    # tighter sysbox destroy semaphore (concurrent FUSE unmounts wedge sysbox-fs
    # and leak the container), NOT the looser general destroy cap.
    mgr = RawContainerManager(
        docker_client=_FakeClient(_ScriptedContainers([])),
        sysbox_destroy_concurrency=1,
    )
    assert mgr._destroy_gate(sysbox=True) is mgr._sysbox_destroy_semaphore
    assert mgr._destroy_gate(sysbox=False) is mgr._destroy_semaphore


def test_destroy_gate_sysbox_falls_back_when_disabled() -> None:
    mgr = RawContainerManager(
        docker_client=_FakeClient(_ScriptedContainers([])),
        sysbox_destroy_concurrency=0,   # sysbox-specific gate disabled
    )
    assert mgr._sysbox_destroy_semaphore is None
    assert mgr._destroy_gate(sysbox=True) is mgr._destroy_semaphore


def test_health_snapshot_folds_the_sysbox_gate() -> None:
    """A sysbox create in-flight (+ one queued) shows in the node health
    snapshot; without folding the sysbox gate the node would read idle."""
    mgr = RawContainerManager(
        docker_client=_FakeClient(_ScriptedContainers([])),
        sysbox_create_concurrency=1,
    )
    sem = mgr._sysbox_create_semaphore
    assert sem is not None

    async def _drive() -> tuple[int, int]:
        await sem.acquire()                       # one sysbox create in-flight
        waiter = asyncio.ensure_future(sem.acquire())  # a second queues behind it
        await asyncio.sleep(0)
        snap = mgr.health_snapshot()
        waiter.cancel()
        sem.release()
        return snap.create_inflight, snap.create_queued

    inflight, queued = asyncio.run(_drive())
    assert inflight == 1                           # sysbox gate counted
    assert queued == 1


@dataclass
class _BlockingContainers:
    """``run`` blocks on ``release`` and records peak concurrency — proves the
    sysbox gate serializes creates (only one run() in flight at a time)."""

    release: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    inflight: int = 0
    max_inflight: int = 0
    run_calls: int = 0
    _next: int = 0

    def run(self, *, name: str | None = None, labels: dict[str, str],
            **_: Any) -> _FakeContainer:
        with self.lock:
            self.run_calls += 1
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self._next += 1
            cid = f"c-{self._next:04d}"
        self.release.wait(timeout=5)
        with self.lock:
            self.inflight -= 1
        return _FakeContainer(id=cid, name=name or cid, labels=dict(labels))

    def list(self, **_: Any) -> list[_FakeContainer]:
        return []


@pytest.mark.asyncio
async def test_sysbox_gate_serializes_concurrent_creates() -> None:
    containers = _BlockingContainers()
    mgr = RawContainerManager(
        docker_client=_FakeClient(containers, runtimes={"runc", "sysbox-runc"}),
        sysbox_create_concurrency=1,
    )
    t1 = asyncio.create_task(
        mgr.acquire(rollout_id="a", image="x", container_runtime="sysbox-runc"),
    )
    t2 = asyncio.create_task(
        mgr.acquire(rollout_id="b", image="x", container_runtime="sysbox-runc"),
    )
    await asyncio.sleep(0.1)  # let both reach the gate
    assert containers.max_inflight == 1   # sysbox semaphore(1) serializes creates
    containers.release.set()
    await asyncio.gather(t1, t2)
    assert containers.max_inflight == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
