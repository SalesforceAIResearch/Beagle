"""Unit tests for the high-concurrency guard wired through the oracle sweep.

The load-bearing, silently-regression-prone pieces are:

  * ``_build_job_config`` must set harbor's ``RetryConfig.max_retries`` (a prior
    ``max_attempts=`` kwarg was dropped by pydantic → retries were always 0), and
  * the retry gate must fire on INFRA-transient errors ONLY. Retrying a
    task-content failure (AgentTimeoutError, a verifier error) would re-roll a
    genuinely-failed task into a fluke pass and poison the eval — the whole reason
    the guard rides on ``CapacityExhausted`` (raised BEFORE the task runs) instead.

These tests pin both, plus the adapter's fail-fast queue-timeout constant.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.terminalworld.run_oracle_sweep import (
    _INFRA_RETRY_EXCEPTIONS,
    _build_job_config,
)


def _mirror_harbor_gate(rc, exception_type: str) -> bool:
    """Reproduce ``harbor.trial.queue.TrialQueue._should_retry_exception``.

    exclude takes precedence over include; include=None means "retry all".
    Kept in lock-step with the vendored harbor logic so this test guards our
    *configuration*, not harbor's implementation.
    """
    if rc.exclude_exceptions and exception_type in rc.exclude_exceptions:
        return False
    return not (
        rc.include_exceptions and exception_type not in rc.include_exceptions
    )


def _job_config(retries: int, tmp_path: Path):
    return _build_job_config(
        task_ids=["tw_000001"],
        shard_root=tmp_path / "shard",
        jobs_dir=tmp_path / "jobs",
        job_id="test-job",
        n_concurrent_trials=32,
        override_cpus=None,
        override_memory_mb=None,
        cpus_multiplier=1.0,
        memory_multiplier=1.0,
        cpu_pinning=False,
        timeout_multiplier=1.0,
        retries=retries,
    )


# ── retry count is actually applied (regression: max_attempts= was a no-op) ───

def test_retries_map_to_max_retries(tmp_path: Path) -> None:
    cfg = _job_config(6, tmp_path)
    assert cfg.retry.max_retries == 6


def test_zero_retries_honoured(tmp_path: Path) -> None:
    cfg = _job_config(0, tmp_path)
    assert cfg.retry.max_retries == 0


# ── the retry gate: infra retries, task content never does (no eval pollution) ─

def test_infra_errors_are_retried(tmp_path: Path) -> None:
    rc = _job_config(6, tmp_path).retry
    assert rc.include_exceptions == set(_INFRA_RETRY_EXCEPTIONS)
    for infra in _INFRA_RETRY_EXCEPTIONS:
        assert _mirror_harbor_gate(rc, infra) is True, infra


@pytest.mark.parametrize(
    "task_error",
    [
        "AgentTimeoutError",       # the task's own solve out-ran its budget
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "VerifierOutputParseError",
        "RuntimeError",            # any non-infra error → final, not re-rolled
    ],
)
def test_task_content_errors_never_retried(task_error: str, tmp_path: Path) -> None:
    rc = _job_config(6, tmp_path).retry
    assert _mirror_harbor_gate(rc, task_error) is False


def test_capacity_exhausted_is_in_the_infra_set() -> None:
    # The load-bearing member: the sysbox cap surfaces this, and it is what turns
    # a queued-out acquire into a retry rather than a trial failure.
    assert "CapacityExhausted" in _INFRA_RETRY_EXCEPTIONS


def test_infra_set_excludes_task_content_errors() -> None:
    # Guard rail: never let a task-outcome error leak into the retry set.
    forbidden = {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
    assert _INFRA_RETRY_EXCEPTIONS.isdisjoint(forbidden)


# ── adapter fail-fast queue-timeout constant ──────────────────────────────────

def test_acquire_queue_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S", raising=False)
    mod = importlib.reload(
        importlib.import_module("xrlenv_plugins.harbor.environment")
    )
    # Below harbor's default 360 s setup window, with headroom for post-acquire
    # pull/start/agent-upload so an at-cap acquire fails fast (CapacityExhausted)
    # rather than being cancelled into a non-retriable timeout.
    assert 0 < mod._ACQUIRE_QUEUE_TIMEOUT_S < 360.0


def test_acquire_queue_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S", "300")
    mod = importlib.reload(
        importlib.import_module("xrlenv_plugins.harbor.environment")
    )
    assert mod._ACQUIRE_QUEUE_TIMEOUT_S == 300.0
    # restore module state for any later importer
    monkeypatch.delenv("XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S", raising=False)
    importlib.reload(mod)


def test_resolve_tasks_empty_selector_raises(tmp_path: object) -> None:
    # audit M5/Low: a present-but-empty --tasks ("" or ",") must FAIL, not fall through.
    import pytest as _pytest
    import xrlenv_plugins.benchmarks.terminalworld.run_oracle_sweep as _s
    for empty in ("", ",", " , "):
        with _pytest.raises(SystemExit, match="selected no tasks"):
            _s._resolve_tasks(tmp_path, empty)  # type: ignore[arg-type]
