"""Issue #18 follow-up B — unit tests for ClusterContainerControl admission
telemetry and queue_timeout_s label hoisting.

Covers:
4. Drop-in telemetry: _record_admission / _log_admission_summary directly
   (a) fast-path session does not increment _acquire_queued
   (b) queued session increments counter, accumulates wait, emits one-shot WARN
   (c) _log_admission_summary silent when _acquire_total==0; INFO when none
       queued; WARN summary (with suggested worker count) when queued
   (d) _live_peak tracks max concurrent _sessions count

5. Label hoisting:
   (a) labels={"xrlenv.queue_timeout_s": "7200"} → acquire_container called
       with queue_timeout_s=7200.0, label stripped
   (b) explicit queue_timeout_s kwarg wins over label
   (c) non-numeric label value → WARN + falls back to None (server default)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.client.container_session import ClusterContainerSession
from xrlenv.compat.docker_client import ClusterContainerControl, from_env
from xrlenv.control.service import RawAcquireResult

# ──────────────────────────────────────────────────────────────────────────────
# Fake session object (minimal — just needs container_id + queue_wait_s)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_atexit_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """ClusterContainerControl.__init__ registers an atexit summary
    hook. Each control built in these tests would otherwise leave a
    live handler that fires (and logs a summary) at pytest process
    exit. Neuter atexit.register for the duration of this file —
    the summary method is exercised directly, not via atexit."""
    monkeypatch.setattr(
        "xrlenv.compat.docker_client.atexit.register",
        lambda *a, **k: None,
    )


@dataclass
class _FakeSession:
    """Minimal stand-in for ClusterContainerSession — only the fields
    _record_admission / _live_peak tracking actually read."""
    container_id: str
    queue_wait_s: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Fake Client for the compat-layer tests
# (mirrors _FakeClient in test_compat_docker_cluster_mode.py)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeTransport:
    destroy_calls: list[dict] = field(default_factory=list)
    exec_calls: list[dict] = field(default_factory=list)
    put_archive_calls: list[dict] = field(default_factory=list)
    get_archive_calls: list[dict] = field(default_factory=list)
    next_exec_result: dict[str, Any] = field(default_factory=lambda: {
        "exit_code": 0, "stdout": b"hi\n", "stderr": b"", "timed_out": False,
    })
    next_get_archive: bytes = b"<tar bytes>"
    next_exec_stream_chunks: list[Any] = field(default_factory=list)

    async def container_exec(self, **kwargs: Any) -> Any:
        self.exec_calls.append(kwargs)
        return dict(self.next_exec_result)

    async def container_put_archive(self, **kwargs: Any) -> None:
        self.put_archive_calls.append(kwargs)

    async def container_get_archive(self, **kwargs: Any) -> bytes:
        self.get_archive_calls.append(kwargs)
        return self.next_get_archive

    def container_exec_stream(self, **kwargs: Any) -> Any:
        chunks = list(self.next_exec_stream_chunks)

        async def _gen() -> Any:
            for c in chunks:
                yield c
        return _gen()

    async def destroy_container(self, **kwargs: Any) -> None:
        self.destroy_calls.append(kwargs)


@dataclass
class _FakeClient:
    next_acquire: RawAcquireResult = field(
        default_factory=lambda: RawAcquireResult(
            rollout_id="r-1", container_id="c-1",
            container_name="cname-1", node_id="node-A",
        ),
    )
    acquire_calls: list[dict] = field(default_factory=list)
    transport: Any = None

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = _FakeTransport()

    async def acquire_container(self, **kwargs: Any) -> ClusterContainerSession:
        self.acquire_calls.append(kwargs)
        return ClusterContainerSession(self.transport, self.next_acquire)


# ──────────────────────────────────────────────────────────────────────────────
# 4a. Fast-path session does not increment _acquire_queued
# ──────────────────────────────────────────────────────────────────────────────


def test_record_admission_fast_path_does_not_increment_queued() -> None:
    """A session with queue_wait_s=0.0 increments _acquire_total but
    NOT _acquire_queued."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]

    session = _FakeSession(container_id="c-1", queue_wait_s=0.0)
    ctrl._sessions["c-1"] = (session, "busybox:1")  # type: ignore[assignment]
    ctrl._record_admission(session)  # type: ignore[arg-type]

    assert ctrl._acquire_total == 1
    assert ctrl._acquire_queued == 0
    assert ctrl._queue_wait_sum_s == pytest.approx(0.0)
    assert ctrl._queue_wait_max_s == pytest.approx(0.0)
    assert ctrl._first_queue_warning_emitted is False


# ──────────────────────────────────────────────────────────────────────────────
# 4b. Queued session increments counter, accumulates wait, one-shot WARN
# ──────────────────────────────────────────────────────────────────────────────


def test_record_admission_queued_session_increments_counters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A session with queue_wait_s >= 1.0 increments _acquire_queued,
    accumulates _queue_wait_sum_s and _queue_wait_max_s."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]

    session = _FakeSession(container_id="c-1", queue_wait_s=5.0)
    ctrl._sessions["c-1"] = (session, "busybox:1")  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger="xrlenv.compat.docker_client"):
        ctrl._record_admission(session)  # type: ignore[arg-type]

    assert ctrl._acquire_total == 1
    assert ctrl._acquire_queued == 1
    assert ctrl._queue_wait_sum_s == pytest.approx(5.0)
    assert ctrl._queue_wait_max_s == pytest.approx(5.0)


def test_record_admission_emits_one_shot_warn_on_first_queued_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first queued acquire emits a WARNING. Subsequent queued acquires
    do NOT emit additional warnings (one-shot guard)."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]

    s1 = _FakeSession(container_id="c-1", queue_wait_s=5.0)
    s2 = _FakeSession(container_id="c-2", queue_wait_s=3.0)
    ctrl._sessions["c-1"] = (s1, "busybox:1")  # type: ignore[assignment]
    ctrl._sessions["c-2"] = (s2, "busybox:1")  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger="xrlenv.compat.docker_client"):
        ctrl._record_admission(s1)  # type: ignore[arg-type]
        ctrl._record_admission(s2)  # type: ignore[arg-type]

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, (
        f"expected exactly one queue WARN, got {[r.message for r in warnings]}"
    )
    assert "QUEUEING" in warnings[0].message


def test_record_admission_accumulates_multiple_queued_sessions() -> None:
    """Multiple queued sessions accumulate wait_sum and track max."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]

    for i, wait in enumerate([2.0, 7.0, 1.5]):
        s = _FakeSession(container_id=f"c-{i}", queue_wait_s=wait)
        ctrl._sessions[f"c-{i}"] = (s, "busybox:1")  # type: ignore[assignment]
        ctrl._record_admission(s)  # type: ignore[arg-type]

    assert ctrl._acquire_total == 3
    assert ctrl._acquire_queued == 3
    assert ctrl._queue_wait_sum_s == pytest.approx(10.5)
    assert ctrl._queue_wait_max_s == pytest.approx(7.0)


# ──────────────────────────────────────────────────────────────────────────────
# 4c. _log_admission_summary behaviour
# ──────────────────────────────────────────────────────────────────────────────


def test_log_admission_summary_silent_when_nothing_acquired(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When _acquire_total==0, _log_admission_summary emits nothing."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]

    with caplog.at_level(logging.DEBUG, logger="xrlenv.compat.docker_client"):
        ctrl._log_admission_summary()

    assert caplog.records == []


def test_log_admission_summary_info_when_none_queued(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When acquires happened but none queued, logs at INFO level and
    mentions 'none queued'."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]
    ctrl._acquire_total = 5

    with caplog.at_level(logging.INFO, logger="xrlenv.compat.docker_client"):
        ctrl._log_admission_summary()

    infos = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(infos) == 1
    assert "none queued" in infos[0].message
    assert "5" in infos[0].message


def test_log_admission_summary_warns_with_suggested_worker_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When some acquires queued, logs a WARN summary that includes the
    suggested worker count (= _live_peak)."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]
    ctrl._acquire_total = 10
    ctrl._acquire_queued = 4
    ctrl._queue_wait_sum_s = 20.0
    ctrl._queue_wait_max_s = 8.0
    ctrl._live_peak = 6

    with caplog.at_level(logging.WARNING, logger="xrlenv.compat.docker_client"):
        ctrl._log_admission_summary()

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    msg = warnings[0].message
    # Must mention the peak count as the suggested worker count.
    assert "6" in msg
    assert "mean" in msg.lower() or "mean" in warnings[0].getMessage().lower()


# ──────────────────────────────────────────────────────────────────────────────
# 4d. _live_peak tracks max concurrent _sessions count
# ──────────────────────────────────────────────────────────────────────────────


def test_live_peak_tracks_max_concurrent_sessions() -> None:
    """_live_peak reflects the maximum len(_sessions) at the time each
    _record_admission call runs (i.e. peak concurrency observed)."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]

    # Add three sessions without removing any — peak should reach 3.
    for i in range(3):
        s = _FakeSession(container_id=f"c-{i}", queue_wait_s=0.0)
        ctrl._sessions[f"c-{i}"] = (s, "busybox:1")  # type: ignore[assignment]
        ctrl._record_admission(s)  # type: ignore[arg-type]

    assert ctrl._live_peak == 3

    # Add a fourth.
    s4 = _FakeSession(container_id="c-3", queue_wait_s=0.0)
    ctrl._sessions["c-3"] = (s4, "busybox:1")  # type: ignore[assignment]
    ctrl._record_admission(s4)  # type: ignore[arg-type]

    assert ctrl._live_peak == 4

    # Drop two sessions and add one more — peak remains 4 (not lowered).
    del ctrl._sessions["c-0"]
    del ctrl._sessions["c-1"]
    s5 = _FakeSession(container_id="c-4", queue_wait_s=0.0)
    ctrl._sessions["c-4"] = (s5, "busybox:1")  # type: ignore[assignment]
    ctrl._record_admission(s5)  # type: ignore[arg-type]

    assert ctrl._live_peak == 4  # still the historical max


# ──────────────────────────────────────────────────────────────────────────────
# 5a. Label hoisting: xrlenv.queue_timeout_s label → acquire kwarg
# ──────────────────────────────────────────────────────────────────────────────


def test_create_container_hoists_queue_timeout_s_label_to_kwarg() -> None:
    """labels={"xrlenv.queue_timeout_s": "7200"} is parsed and forwarded
    as acquire_container(queue_timeout_s=7200.0); the label is stripped
    from what's sent to acquire_container."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        labels={"xrlenv.queue_timeout_s": "7200", "other": "x"},
    )

    call = fake.acquire_calls[0]
    assert call["queue_timeout_s"] == pytest.approx(7200.0)
    # Label must be stripped from the forwarded labels dict.
    assert call["labels"] == {"other": "x"}


def test_create_container_queue_timeout_s_only_label_clears_labels() -> None:
    """If xrlenv.queue_timeout_s is the only label, popping it leaves
    labels=None on the wire (don't send an empty labels dict)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        labels={"xrlenv.queue_timeout_s": "3600"},
    )

    call = fake.acquire_calls[0]
    assert call["queue_timeout_s"] == pytest.approx(3600.0)
    assert call["labels"] is None


# ──────────────────────────────────────────────────────────────────────────────
# 5b. Explicit kwarg wins over label
# ──────────────────────────────────────────────────────────────────────────────


def test_create_container_explicit_kwarg_wins_over_label() -> None:
    """When both queue_timeout_s kwarg and the label are present, the
    explicit kwarg takes precedence."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        labels={"xrlenv.queue_timeout_s": "9999"},
        queue_timeout_s=1800.0,
    )

    call = fake.acquire_calls[0]
    assert call["queue_timeout_s"] == pytest.approx(1800.0)


# ──────────────────────────────────────────────────────────────────────────────
# 5c. Non-numeric label value → WARN + fallback to None
# ──────────────────────────────────────────────────────────────────────────────


def test_create_container_non_numeric_label_warns_and_uses_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-numeric xrlenv.queue_timeout_s label value emits a WARN and
    falls back to None (server default), rather than crashing."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger="xrlenv.compat.docker_client"):
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            labels={"xrlenv.queue_timeout_s": "not-a-number"},
        )

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("non-numeric" in r.message for r in warnings), (
        f"expected non-numeric WARN, got: {[r.message for r in warnings]}"
    )
    call = fake.acquire_calls[0]
    # Falls back to None — transport will apply the server default (3600.0).
    assert call["queue_timeout_s"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Issue #12 / #18 — xrlenv.acquire_timeout_s label → acquire kwarg
# ──────────────────────────────────────────────────────────────────────────────


def test_create_container_hoists_acquire_timeout_s_label_to_kwarg() -> None:
    """labels={"xrlenv.acquire_timeout_s": "1800"} is parsed and
    forwarded as acquire_container(acquire_timeout_s=1800.0); the label
    is stripped from what's sent to acquire_container."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        labels={"xrlenv.acquire_timeout_s": "1800", "other": "x"},
    )

    call = fake.acquire_calls[0]
    assert call["acquire_timeout_s"] == pytest.approx(1800.0)
    assert call["labels"] == {"other": "x"}


def test_create_container_acquire_timeout_s_explicit_kwarg_wins() -> None:
    """An explicit acquire_timeout_s kwarg beats the label."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        labels={"xrlenv.acquire_timeout_s": "9999"},
        acquire_timeout_s=1234.0,
    )

    assert fake.acquire_calls[0]["acquire_timeout_s"] == pytest.approx(1234.0)


def test_create_container_no_acquire_timeout_s_forwards_none() -> None:
    """With neither kwarg nor label, acquire_timeout_s reaches
    acquire_container as None — the server applies its 600s default."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container("busybox:1", command=["sleep", "infinity"])

    assert fake.acquire_calls[0]["acquire_timeout_s"] is None
