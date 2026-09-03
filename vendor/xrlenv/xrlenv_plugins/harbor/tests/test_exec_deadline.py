"""The exec deadline must never be tighter than the budget harbor will enforce.

harbor enforces a phase budget one level up (``asyncio.wait_for(verifier.verify(),
timeout=verifier.timeout_sec)`` in ``trial.py``) and its ``Verifier`` calls
``exec`` with **no** ``timeout_sec``; harbor's own DockerEnvironment passes that
``None`` through untouched. A flat default in this plug-in therefore became the
binding constraint and killed long verifiers early — a truncated log and reward 0,
indistinguishable from broken task content.

The deadline is a transport backstop, not policy: harbor cancels the exec when
the real budget expires. It only has to be no smaller than what harbor will
enforce — which means honouring the trial's timeout MULTIPLIERS, not just the
task's declared base.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from xrlenv_plugins.harbor.environment import (
    _DEFAULT_EXEC_TIMEOUT_S,
    XrlenvHarborEnvironmentCluster,
)


class _Paths:
    """Stand-in for harbor's TrialPaths — only ``config_path`` is read."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path


def _env(task_dir: Path, trial_cfg: dict | None = None) -> XrlenvHarborEnvironmentCluster:
    """An instance carrying only what the resolver reads (harbor's __init__
    needs a live cluster)."""
    env = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)
    env.environment_dir = task_dir / "environment"
    # deliberately NOT setting _phase_budget_s_cache: __init__ is skipped here,
    # so these tests only pass via the CLASS-level default. See
    # test_budget_readable_on_an_instance_that_never_ran_init.
    cfg = task_dir / "trial-config.json"
    if trial_cfg is not None:
        cfg.write_text(json.dumps(trial_cfg))
    env.trial_paths = _Paths(cfg)
    return env


def _task(tmp_path: Path, *, agent: int | None, verifier: int | None,
          name: str = "some-task") -> Path:
    d = tmp_path / name
    (d / "environment").mkdir(parents=True)
    body = "[environment]\ncpus = 1\n"
    if agent is not None:
        body += f"\n[agent]\ntimeout_sec = {agent}\n"
    if verifier is not None:
        body += f"\n[verifier]\ntimeout_sec = {verifier}\n"
    (d / "task.toml").write_text(body)
    return d


# ── the reported regression ───────────────────────────────────────────────────


def test_deadline_follows_the_declared_verifier_budget(tmp_path: Path) -> None:
    """verifier.timeout_sec=3000 must not be capped at the 1800 s floor."""
    env = _env(_task(tmp_path, agent=3000, verifier=3000), {})
    assert env._default_exec_timeout_s() == 3000
    assert env._default_exec_timeout_s() > _DEFAULT_EXEC_TIMEOUT_S


def test_deadline_uses_the_LARGEST_declared_phase(tmp_path: Path) -> None:
    """exec() cannot tell which phase it serves, so it must cover the longest."""
    env = _env(_task(tmp_path, agent=3000, verifier=6000), {})
    assert env._default_exec_timeout_s() == 6000


# ── the multiplier: the reason we read the trial config at all ────────────────


@pytest.mark.parametrize("multiplier", [1.0, 2.0, 3.0, 10.0])
def test_deadline_honours_any_timeout_multiplier(
    tmp_path: Path, multiplier: float,
) -> None:
    """Harbor's budget is base x multiplier. A FIXED margin would under-size the
    deadline once the multiplier exceeded it and truncate the verifier again —
    the exact bug this method exists to prevent. Reading the real value is
    correct for every multiplier, not just the ones we guessed."""
    env = _env(
        _task(tmp_path, agent=3000, verifier=3000, name=f"m{multiplier}"),
        {"timeout_multiplier": multiplier},
    )
    assert env._default_exec_timeout_s() == max(_DEFAULT_EXEC_TIMEOUT_S, 3000 * multiplier)


def test_phase_multiplier_overrides_the_job_default(tmp_path: Path) -> None:
    """harbor: `phase_multiplier if not None else timeout_multiplier`."""
    env = _env(
        _task(tmp_path, agent=3000, verifier=3000),
        {"timeout_multiplier": 2.0, "verifier_timeout_multiplier": 5.0},
    )
    assert env._default_exec_timeout_s() == 3000 * 5.0


def test_override_timeout_sec_replaces_the_task_base(tmp_path: Path) -> None:
    env = _env(
        _task(tmp_path, agent=3000, verifier=3000),
        {"verifier": {"override_timeout_sec": 9000}},
    )
    assert env._default_exec_timeout_s() == 9000


def test_max_timeout_sec_caps_the_base(tmp_path: Path) -> None:
    """harbor: `min(base_sec, max_sec or inf) * multiplier`. The agent phase is
    kept small here so the capped verifier is the largest budget — otherwise the
    agent's 3000x2 would legitimately win the max()."""
    env = _env(
        _task(tmp_path, agent=60, verifier=6000),
        {"verifier": {"max_timeout_sec": 2000}, "timeout_multiplier": 2.0},
    )
    assert env._default_exec_timeout_s() == 2000 * 2.0   # capped, then scaled


# ── floors and degradation ────────────────────────────────────────────────────


def test_floor_applies_to_short_and_undeclared_budgets(tmp_path: Path) -> None:
    """Bookkeeping execs (chmod, mkdir) carry no phase budget; a task declaring
    a tiny one must not shrink the transport deadline below the floor."""
    undeclared = _env(_task(tmp_path, agent=None, verifier=None, name="undeclared"), {})
    tiny = _env(_task(tmp_path, agent=10, verifier=10, name="tiny"), {})
    assert undeclared._default_exec_timeout_s() == _DEFAULT_EXEC_TIMEOUT_S
    assert tiny._default_exec_timeout_s() == _DEFAULT_EXEC_TIMEOUT_S


def test_missing_trial_config_degrades_to_multiplier_one(tmp_path: Path) -> None:
    """The trial config.json is written before the environment starts, but never
    assume it: absent -> multiplier 1.0, not a crash."""
    env = _env(_task(tmp_path, agent=3000, verifier=3000), trial_cfg=None)
    assert env._default_exec_timeout_s() == 3000


@pytest.mark.parametrize("body", ["", "not toml [[[", "[verifier]\ntimeout_sec = 'x'\n"])
def test_unreadable_or_malformed_task_toml_falls_back(tmp_path: Path, body: str) -> None:
    """Never raise from a deadline lookup — a bad task.toml degrades to the
    floor rather than failing the exec."""
    d = tmp_path / "t"
    (d / "environment").mkdir(parents=True)
    (d / "task.toml").write_text(body)
    assert _env(d, {})._default_exec_timeout_s() == _DEFAULT_EXEC_TIMEOUT_S


def test_missing_task_toml_falls_back(tmp_path: Path) -> None:
    d = tmp_path / "t"
    (d / "environment").mkdir(parents=True)
    assert _env(d, {})._default_exec_timeout_s() == _DEFAULT_EXEC_TIMEOUT_S


def test_budget_readable_on_an_instance_that_never_ran_init(tmp_path: Path) -> None:
    """REGRESSION. The memo was first declared only in ``__init__``, which
    AttributeError'd on every instance built with ``__new__`` — the shape a large
    body of unit tests uses (tests/unit/plugins/harbor/test_harbor_cluster.py,
    39 of them) — because the deadline lookup runs on the exec path they cover.
    A class-level default is what makes the read total."""
    cls = XrlenvHarborEnvironmentCluster
    assert cls._phase_budget_s_cache is None
    bare = cls.__new__(cls)
    bare.environment_dir = tmp_path / "nonexistent"
    bare.trial_paths = _Paths(tmp_path / "no-config.json")
    assert bare._default_exec_timeout_s() == _DEFAULT_EXEC_TIMEOUT_S


def test_memoising_does_not_leak_across_instances(tmp_path: Path) -> None:
    """Writing the memo must bind to the INSTANCE, never to the class default —
    otherwise the first trial's budget would silently become every trial's."""
    a = _env(_task(tmp_path, agent=3000, verifier=3000, name="a"), {})
    assert a._default_exec_timeout_s() == 3000
    assert XrlenvHarborEnvironmentCluster._phase_budget_s_cache is None
    b = _env(_task(tmp_path, agent=6000, verifier=6000, name="b"), {})
    assert b._default_exec_timeout_s() == 6000


def test_budget_is_memoised(tmp_path: Path) -> None:
    """exec() runs many times per trial; the files are read once."""
    d = _task(tmp_path, agent=3000, verifier=3000)
    env = _env(d, {})
    first = env._default_exec_timeout_s()
    (d / "task.toml").unlink()               # cache must survive the file going away
    assert env._default_exec_timeout_s() == first
