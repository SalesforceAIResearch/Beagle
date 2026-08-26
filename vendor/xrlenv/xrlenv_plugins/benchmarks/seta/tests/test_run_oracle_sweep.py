"""Unit tests for the pure logic in the seta oracle-sweep driver.

The cluster run itself is I/O (harbor + a control plane), but the deterministic,
regression-prone pieces are: blacklist reading, task selection (default / --tasks
/ --all mutex), shard discovery (blacklist-excluded), and the pass/fail verdict.
Offline: a synthetic shard in tmp, no harbor, no cluster.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest
from xrlenv_plugins.benchmarks.seta import run_oracle_sweep as sweep
from xrlenv_plugins.benchmarks.seta.run_oracle_sweep import _INFRA_RETRY_EXCEPTIONS

# ── blacklist ─────────────────────────────────────────────────────────────────


def test_blacklist_ids_reads_committed_file() -> None:
    assert {"25", "305", "387", "683", "999"} <= sweep._blacklist_ids()


# ── task selection ────────────────────────────────────────────────────────────


def _args(**kw: object) -> types.SimpleNamespace:
    base = {"all": False, "tasks": None}
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_default_task_list_is_the_smoke_subset() -> None:
    assert sweep._resolve_task_list(_args()) == list(sweep.SMOKE_TASKS)


def test_all_and_tasks_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        sweep._resolve_task_list(_args(all=True, tasks="0,1"))


def test_explicit_tasks_are_returned_verbatim() -> None:
    # --tasks passes ids through (even a blacklisted one — it only warns).
    assert sweep._resolve_task_list(_args(tasks="0, 1 ,25")) == ["0", "1", "25"]


# ── shard discovery ───────────────────────────────────────────────────────────


def _make_task(shard: Path, task_id: str) -> None:
    (shard / task_id / "solution").mkdir(parents=True)
    (shard / task_id / "solution" / "solve.sh").write_text("#!/bin/bash\n")


def test_discover_all_tasks_excludes_blacklist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))
    shard = tmp_path / sweep.SETA_CACHE_SHARD
    for tid in ("0", "1", "2", "25"):  # 25 is on the committed blacklist
        _make_task(shard, tid)
    assert sweep._discover_all_tasks() == ["0", "1", "2"]  # numeric order, 25 dropped


def test_discover_all_tasks_hard_fails_without_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))
    with pytest.raises(SystemExit, match="requires the seta shard"):
        sweep._discover_all_tasks()


# ── pass/fail verdict ─────────────────────────────────────────────────────────


def _trial(*, exc: str | None = None, rewards: dict[str, float] | None = None):
    exception_info = (
        types.SimpleNamespace(exception_type=exc) if exc is not None else None
    )
    verifier_result = (
        types.SimpleNamespace(rewards=rewards) if rewards is not None else None
    )
    return types.SimpleNamespace(
        exception_info=exception_info, verifier_result=verifier_result,
    )


def test_trial_passes_on_all_positive_rewards() -> None:
    ok, reason = sweep._trial_passes(_trial(rewards={"a": 1.0, "b": 0.5}))
    assert ok and reason is None


def test_trial_fails_on_exception() -> None:
    ok, reason = sweep._trial_passes(_trial(exc="NodeLost"))
    assert not ok and reason is not None and "NodeLost" in reason


def test_trial_fails_on_missing_rewards() -> None:
    ok, reason = sweep._trial_passes(_trial())
    assert not ok and reason == "no verifier rewards recorded"


def test_trial_fails_on_nonpositive_reward() -> None:
    ok, reason = sweep._trial_passes(_trial(rewards={"a": 1.0, "b": 0.0}))
    assert not ok and reason is not None and "non-positive" in reason


def test_resolve_task_list_empty_selector_raises() -> None:
    # audit M5/Low: a present-but-empty --tasks ("" or ",") must FAIL, not fall through.
    for empty in ("", ",", " , "):
        with pytest.raises(SystemExit, match="selected no tasks"):
            sweep._resolve_task_list(_args(tasks=empty))


# ── the retry gate names real exceptions ──────────────────────────────────────


def test_infra_retry_exception_names_all_resolve() -> None:
    """Every name in the retry gate must be a real ``xrlenv.errors`` class.

    The gate matches on ``type(exc).__name__``, so a typo'd string is not an
    error — it is a silent no-op: the exception simply never matches and the
    trial is scored as a content failure instead of retried. That is exactly how
    ``SessionReaped`` sat inert after being declared ``retryable = True``, and
    this plug-in has no other test asserting the gate's contents, so a future
    typo here would reach production unnoticed.
    """
    import xrlenv.errors as errors

    assert _INFRA_RETRY_EXCEPTIONS, "the retry gate is empty"
    for name in _INFRA_RETRY_EXCEPTIONS:
        cls = getattr(errors, name, None)
        assert cls is not None, f"{name!r} names no exception in xrlenv.errors"
        assert cls.__name__ == name
        assert issubclass(cls, errors.XRLEnvError)

    # A platform teardown is infra, not a content result — the reason this gate
    # was touched at all.
    assert "SessionReaped" in _INFRA_RETRY_EXCEPTIONS
