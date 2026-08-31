"""Unit tests for the retry wiring in the tb2.1 oracle-sweep driver.

Load-bearing, silently-regression-prone pieces (mirrors the TerminalWorld
runner's guard):

  * ``_build_job_config`` must set harbor's ``RetryConfig.max_retries`` — a prior
    ``max_attempts=`` kwarg was dropped by pydantic, so ``--retries`` was a no-op
    (always 0), and
  * the retry gate must fire on INFRA-transient errors ONLY. Retrying a
    task-content failure would re-roll a broken oracle into a fluke pass — exactly
    the corpus defect this sweep exists to catch — so it must never retry those.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.terminal_bench_2_1.run_oracle_sweep import (
    _INFRA_RETRY_EXCEPTIONS,
    _build_job_config,
)


def _mirror_harbor_gate(rc, exception_type: str) -> bool:
    """Reproduce ``harbor.trial.queue.TrialQueue._should_retry_exception``:
    exclude wins over include; include=None means "retry all"."""
    if rc.exclude_exceptions and exception_type in rc.exclude_exceptions:
        return False
    return not (
        rc.include_exceptions and exception_type not in rc.include_exceptions
    )


def _retry_cfg(retries: int, tmp_path: Path):
    return _build_job_config(
        task_ids=["build-cython-ext"],
        dataset_root=tmp_path / "terminal-bench-2-1",
        jobs_dir=tmp_path / "jobs",
        job_id="test-job",
        n_concurrent_trials=32,
        retries=retries,
    ).retry


def test_retries_map_to_max_retries(tmp_path: Path) -> None:
    assert _retry_cfg(6, tmp_path).max_retries == 6


def test_zero_retries_honoured(tmp_path: Path) -> None:
    assert _retry_cfg(0, tmp_path).max_retries == 0


def test_infra_errors_are_retried(tmp_path: Path) -> None:
    rc = _retry_cfg(6, tmp_path)
    assert rc.include_exceptions == set(_INFRA_RETRY_EXCEPTIONS)
    for infra in _INFRA_RETRY_EXCEPTIONS:
        assert _mirror_harbor_gate(rc, infra) is True, infra


@pytest.mark.parametrize(
    "task_error",
    [
        "AgentTimeoutError",   # a genuinely-slow / broken oracle — must stay FAIL
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "VerifierOutputParseError",
        "RuntimeError",
    ],
)
def test_task_content_errors_never_retried(task_error: str, tmp_path: Path) -> None:
    assert _mirror_harbor_gate(_retry_cfg(6, tmp_path), task_error) is False


def test_infra_set_excludes_task_content_errors() -> None:
    forbidden = {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
    assert _INFRA_RETRY_EXCEPTIONS.isdisjoint(forbidden)


def test_resolve_tasks_empty_selector_raises(tmp_path: object) -> None:
    # audit M5/Low: a present-but-empty --tasks ("" or ",") must FAIL, not fall through.
    import pytest as _pytest
    import xrlenv_plugins.benchmarks.terminal_bench_2_1.run_oracle_sweep as _s
    for empty in ("", ",", " , "):
        with _pytest.raises(SystemExit, match="selected no tasks"):
            _s._resolve_tasks(tmp_path, empty)  # type: ignore[arg-type]
