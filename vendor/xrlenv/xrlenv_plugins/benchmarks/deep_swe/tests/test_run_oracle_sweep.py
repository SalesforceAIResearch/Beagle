"""Unit tests for the DeepSWE oracle sweep's pure logic (no cluster).

The gate MUST key on the ``reward`` field only — DeepSWE ``reward.json`` also carries
f2p/p2p totals + fractions + ``partial`` that are legitimately 0, so an "all values
> 0" gate (the terminalworld shape) would false-FAIL a passing task.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from xrlenv_plugins.benchmarks.deep_swe import run_oracle_sweep as s
from xrlenv_plugins.benchmarks.deep_swe.run_oracle_sweep import _INFRA_RETRY_EXCEPTIONS


def _tr(rewards: dict | None, *, exc: str | None = None, name: str = "t") -> SimpleNamespace:
    exception_info = SimpleNamespace(exception_type=exc) if exc else None
    verifier_result = SimpleNamespace(rewards=rewards)
    return SimpleNamespace(
        task_name=name, verifier_result=verifier_result, exception_info=exception_info,
    )


def test_pass_when_reward_positive_even_if_other_metrics_zero() -> None:
    # the crux: reward==1 passes even though p2p_total / partial are 0
    ok, reason = s._trial_passes(_tr({"reward": 1.0, "p2p_total": 0, "partial": 0}))
    assert ok
    assert reason is None


def test_fail_when_reward_zero() -> None:
    ok, reason = s._trial_passes(_tr({"reward": 0.0, "f2p_total": 3, "f2p": 0.0}))
    assert not ok
    assert reason == "reward=0.0"


def test_fail_on_exception() -> None:
    ok, reason = s._trial_passes(_tr({"reward": 1.0}, exc="AgentTimeoutError"))
    assert not ok
    assert "AgentTimeoutError" in reason


def test_fail_when_no_reward_key() -> None:
    ok, reason = s._trial_passes(_tr({"f2p_total": 3}))
    assert not ok
    assert "reward" in reason


def test_fail_when_no_rewards_recorded() -> None:
    assert s._trial_passes(_tr(None))[0] is False
    assert s._trial_passes(_tr({}))[0] is False


def test_reward_value_and_side_metrics() -> None:
    tr = _tr({"reward": 1, "f2p": 0.5, "p2p": 1.0, "partial": 0})
    assert s._reward_value(tr) == 1.0
    assert s._side_metrics(tr) == {"f2p": 0.5, "p2p": 1.0, "partial": 0}
    # non-numeric reward -> None (defensive)
    assert s._reward_value(_tr({"reward": "oops"})) is None


def test_task_key_is_dir_basename_not_namespaced_task_name() -> None:
    # Regression (2026-07-31 ci false-fail): the content-retry loop keys `best` on
    # _task_key — the requested id == shard dir name (basename of config.task.path) —
    # NOT trial_result.task_name. deep-swe task.toml names are "datacurve/<id>", so
    # keying `best` on task_name never matched the requested bare ids: every task read
    # as non-passing (re-ran all retries) and the tally showed 0/N even at reward=1.
    tr = SimpleNamespace(
        config=SimpleNamespace(
            task=SimpleNamespace(path="/cache/deep-swe/tengo-callable-instance-isolation"),
        ),
        task_name="datacurve/tengo-callable-instance-isolation",
    )
    assert s._task_key(tr) == "tengo-callable-instance-isolation"


def test_resolve_tasks(tmp_path: Path) -> None:
    shard = tmp_path / "deep-swe"
    for name in ("a", "b"):
        (shard / name).mkdir(parents=True)
        (shard / name / "task.toml").write_text("[environment]\n")

    assert s._resolve_tasks(shard, None) == ["a", "b"]  # all, sorted
    assert s._resolve_tasks(shard, "b") == ["b"]  # subset
    with pytest.raises(SystemExit, match="unknown task"):
        s._resolve_tasks(shard, "nope")


def test_resolve_tasks_empty_selector_raises(tmp_path: Path) -> None:
    # audit M5: `--tasks ","` parses to [] — must FAIL, not fall through to a 0/0 pass.
    shard = tmp_path / "deep-swe"
    (shard / "a").mkdir(parents=True)
    (shard / "a" / "task.toml").write_text("[environment]\n")
    # includes the literal "" (present-but-empty): it must NOT fall through to "all" (M5).
    for empty in ("", ",", " , ", ",,"):
        with pytest.raises(SystemExit, match="selected no tasks"):
            s._resolve_tasks(shard, empty)
    assert s._resolve_tasks(shard, None) == ["a"]   # None (absent) still -> all


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
