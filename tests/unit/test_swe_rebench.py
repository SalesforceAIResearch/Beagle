"""SWE-rebench onboarding — harbor-family benchmark with a declared agent-phase workspace.

Registration is the zero-code HarborBenchmark path; what needs covering is the one thing it adds:
the ``task_env`` seam (repo-dir resolver + env preamble) reaching the harbor shim, and the shim's
two-tier check — raise when the workspace has no repo, record when the task env didn't activate.

The corpus's own verifier resolves both for itself, so a green oracle sweep can't catch a broken
agent workspace; these tests are where that contract is pinned. See
``notes/swe-rebench-onboarding.md``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import beagle as bgl
from beagle.benchmarks.base import BenchmarkSpec
from beagle.benchmarks.grader import InBandGrader
from beagle.benchmarks.harness import HarborBenchmark, HarborHarness
from beagle.benchmarks.source import HarborCache
from beagle.types import Task, TaskContext, TaskResult

# A real swe-rebench task.toml (trimmed): note there is NO [environment] workdir and no PATH help —
# the two facts the benchmark's task_env supplies instead.
_TASK_TOML = """\
[task]
name = "swe-rebench/ASPP__pelita-863"

[verifier]
timeout_sec = 3000

[agent]
timeout_sec = 3000

[environment]
build_timeout_sec = 1800.0
cpus = 1
memory = '8G'
docker_image = "swerebench/sweb.eval.x86_64.aspp_1776_pelita-863:latest"
"""


class _Env:
    """Fake harbor environment: records every exec and replays queued stdout."""

    def __init__(self, *outs: str) -> None:
        self.outs = list(outs)
        self.calls: list[str] = []

    async def exec(self, command: str, *a, **kw):
        self.calls.append(command)
        return SimpleNamespace(stdout=self.outs.pop(0) if self.outs else "", stderr="")


def _shim(monkeypatch, task_env=None):
    """The harbor shim, behind a FAKE `harbor` — so the wiring is covered in a plain dev
    environment too. harbor is an optional extra (`beagle[terminal-bench]`), and gating these on
    it left the seam's regression path unrun wherever it isn't installed. Same trick the pier
    tests use; the seam's DECISIONS are framework-free and tested directly above."""
    import importlib
    import sys
    import types

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("harbor"); _mod("harbor.agents"); _mod("harbor.agents.installed")
    _mod("harbor.agents.installed.base").BaseInstalledAgent = type(  # type: ignore[attr-defined]
        "BaseInstalledAgent", (), {"__init__": lambda self, *a, **k: None})
    _mod("harbor.environments")
    _mod("harbor.environments.base").BaseEnvironment = type("BaseEnvironment", (), {})  # type: ignore[attr-defined]
    _mod("harbor.models"); _mod("harbor.models.agent")
    _mod("harbor.models.agent.context").AgentContext = type("AgentContext", (), {})  # type: ignore[attr-defined]
    monkeypatch.delitem(sys.modules, "beagle.benchmarks.harness._harbor_agent", raising=False)
    mod = importlib.import_module("beagle.benchmarks.harness._harbor_agent")

    obj = mod.BeagleInstalledAgent.__new__(mod.BeagleInstalledAgent)
    obj._identity = {"agent": "mini-swe"}
    obj._task_env = dict(task_env or {})
    return obj


def test_registered_with_harbor_defaults() -> None:
    b = bgl.benchmarks.get("swe-rebench")
    assert b.name == "swe-rebench" and isinstance(b, HarborBenchmark)
    assert b.cache_name == "swe-rebench"          # the shard xrlenv materialized
    assert isinstance(b.harness(), HarborHarness) and isinstance(b.grader(), InBandGrader)


def test_task_env_declares_both_container_facts() -> None:
    b = bgl.benchmarks.get("swe-rebench")
    assert isinstance(b, HarborBenchmark)
    env = b.task_env()
    # The resolver must prefer /testbed (what upstream's own setup assumes for all 860) and still
    # carry the verifier's find-fallback; the preamble must try the `testbed` conda env.
    assert "/testbed/.git" in env["repo_path_cmd"] and "find /" in env["repo_path_cmd"]
    assert "envs/$n/bin" in env["shell_preamble"] and "testbed" in env["shell_preamble"]


def test_task_env_reaches_the_harness_and_defaults_empty_elsewhere() -> None:
    h = bgl.benchmarks.get("swe-rebench").harness()
    assert isinstance(h, HarborHarness)
    assert set(h.task_env) == {"repo_path_cmd", "shell_preamble"}

    class _Plain(HarborBenchmark):     # a benchmark that declares nothing (terminal-bench shape)
        name = "t-plain"

    plain = _Plain().harness()
    assert isinstance(plain, HarborHarness) and plain.task_env == {}   # unchanged behaviour


def test_source_reads_a_swe_rebench_task_dir(tmp_path) -> None:
    td = tmp_path / "swe-rebench" / "ASPP__pelita-863"
    td.mkdir(parents=True)
    (td / "task.toml").write_text(_TASK_TOML)
    (td / "instruction.md").write_text("Better Bot repr")
    src = HarborCache("swe-rebench", cache_name="swe-rebench", cache_root=tmp_path)

    (t, c), = list(src.tasks(BenchmarkSpec(name="swe-rebench")))
    assert t.task_id == "ASPP__pelita-863" and t.benchmark == "swe-rebench"
    assert c.image == "swerebench/sweb.eval.x86_64.aspp_1776_pelita-863:latest"
    # No [environment] workdir in the corpus → the cache-derived repo_path is empty. THIS is why
    # the harbor path needs task_env: nothing on disk says where the agent should work.
    assert c.repo_path == ""
    assert t.extras["harbor_task_dir"] == str(td)


def test_workspace_probe_and_check_commands() -> None:
    """What the shim runs in the container. Harbor-free: these are the seam's real decisions, and
    gating them behind the optional extra left them unverified in a plain dev environment."""
    from beagle.benchmarks.harness._common import workspace_check_command, workspace_probe_command

    # a benchmark that declares nothing keeps the framework's own `pwd`
    assert workspace_probe_command({}) == "pwd"
    assert workspace_probe_command({"repo_path_cmd": "resolve-it"}) == "resolve-it"

    cmd = workspace_check_command("/testbed", "activate-it")
    assert cmd.startswith("activate-it")          # runs UNDER the agent's own preamble
    assert "cd /testbed" in cmd
    # `git rev-parse`, not `[ -d .git ]`: a worktree's .git is a FILE
    assert "rev-parse --is-inside-work-tree" in cmd and "[ -d .git ]" not in cmd
    assert "command -v python" in cmd


def test_workspace_check_interpretation() -> None:
    """The two tiers: a missing repo refuses to start, a base interpreter is only recorded."""
    from beagle.benchmarks.harness._common import (
        WorkspaceSetupError,
        interpret_workspace_check,
    )

    ok = interpret_workspace_check("/testbed", "BEAGLE_REPO_OK\n/opt/conda/envs/testbed/bin/python")
    assert ok == {"workspace": "/testbed", "task_python": "/opt/conda/envs/testbed/bin/python",
                  "task_env_active": True}
    # base conda (the ~58% of images with no env on PATH) — recorded, NOT fatal, until its
    # false-positive rate is measured
    degraded = interpret_workspace_check("/testbed", "BEAGLE_REPO_OK\n/opt/conda/bin/python")
    assert degraded["task_env_active"] is False
    assert degraded["task_python"] == "/opt/conda/bin/python"
    # a venv counts too
    assert interpret_workspace_check("/w", "BEAGLE_REPO_OK\n/w/.venv/bin/python")["task_env_active"]
    # no marker → the agent has no repo, which would score 0 for a reason that isn't the agent
    with pytest.raises(WorkspaceSetupError, match="no git worktree"):
        interpret_workspace_check("/", "/usr/bin/python3")


def test_shim_without_task_env_keeps_the_bare_pwd_probe(monkeypatch) -> None:
    # Regression guard for terminal-bench / deep-swe: one exec (`pwd`), no check, no metadata.
    shim, env = _shim(monkeypatch), _Env("/app\n")
    repo, setup = asyncio.run(shim._resolve_workspace(env))
    assert (repo, setup) == ("/app", {})
    assert env.calls == ["pwd"]


def test_shim_resolves_workspace_and_records_active_env(monkeypatch) -> None:
    shim = _shim(monkeypatch, {"repo_path_cmd": "resolve-it", "shell_preamble": "activate-it"})
    env = _Env("/testbed", "BEAGLE_REPO_OK\n/opt/conda/envs/testbed/bin/python\n")
    repo, setup = asyncio.run(shim._resolve_workspace(env))

    assert repo == "/testbed"
    assert setup == {"workspace": "/testbed",
                     "task_python": "/opt/conda/envs/testbed/bin/python",
                     "task_env_active": True}
    assert env.calls[0] == "resolve-it"                      # benchmark's resolver, not `pwd`
    # The check runs UNDER the preamble and inside the resolved dir — what's verified is what the
    # agent gets, not a different shell.
    assert "activate-it" in env.calls[1] and "cd /testbed" in env.calls[1]


def test_mini_swe_prefixes_the_shell_preamble() -> None:
    # mini-swe drove every exec with a bare `cd <repo>`, dropping the benchmark's preamble that
    # opencode/monet already honour — so its `python` was the base interpreter on swe-rebench.
    from beagle.agents.core.spec import AgentSpec, ModelSpec
    from beagle.agents.mini_swe import MiniSweAgent
    from beagle.types import Task, TaskContext

    calls: list[str] = []

    class _Rt:
        def exec(self, handle, argv, **kw):
            calls.append(argv[-1])
            return SimpleNamespace(stdout="", stderr="", returncode=0, ok=True)

    agent = MiniSweAgent(AgentSpec(name="mini-swe", model=ModelSpec(name="gpt-5")))
    ctx = TaskContext(image=None, repo_path="/testbed", agent_timeout_s=1800,
                      shell_preamble="export PATH=/opt/conda/envs/testbed/bin:$PATH")
    agent.run_in(object(), Task(task_id="t", problem_statement="fix"), ctx, runtime=_Rt())

    # Every exec that works IN the repo (base commit, the `mini` run, commit, diff) must carry it;
    # the trajectory read-back (`cat /logs/agent/...`) is repo-independent and correctly doesn't.
    in_repo = [c for c in calls if "cd /testbed" in c]
    assert len(in_repo) >= 4, [c[:60] for c in calls]
    assert all(c.startswith("export PATH=/opt/conda/envs/testbed/bin:$PATH\n") for c in in_repo), (
        [c[:60] for c in in_repo])


# --- timeout policy: the task's budget outranks every house constant -------------------------

def _trial(tmp_path, *, task_toml_agent_sec=None, **cfg):
    """A trial dir the way harbor lays one out: <trial>/config.json + the task dir it names."""
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    body = "[task]\nname = 't'\n"
    if task_toml_agent_sec is not None:
        body += f"\n[agent]\ntimeout_sec = {task_toml_agent_sec}\n"
    (task_dir / "task.toml").write_text(body)
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True, exist_ok=True)
    payload = {"task": {"path": str(task_dir)}, "agent": {}}
    payload.update(cfg)
    (trial / "config.json").write_text(json.dumps(payload))
    return trial


def test_budget_mirrors_harbors_formula(tmp_path) -> None:
    from beagle.benchmarks.harness._common import effective_agent_budget_s

    # base = task.toml
    t = _trial(tmp_path / "a", task_toml_agent_sec=3000)
    assert effective_agent_budget_s(t / "config.json") == 3000.0
    # override beats task.toml
    t = _trial(tmp_path / "b", task_toml_agent_sec=3000, agent={"override_timeout_sec": 900})
    assert effective_agent_budget_s(t / "config.json") == 900.0
    # max_timeout_sec CLAMPS DOWN only — it never extends a smaller task budget
    t = _trial(tmp_path / "c", task_toml_agent_sec=6000, agent={"max_timeout_sec": 3600})
    assert effective_agent_budget_s(t / "config.json") == 3600.0
    t = _trial(tmp_path / "d", task_toml_agent_sec=900, agent={"max_timeout_sec": 3600})
    assert effective_agent_budget_s(t / "config.json") == 900.0
    # multipliers: the phase-specific one wins, an ABSENT one means 1.0 (config.json may be
    # dumped with exclude_defaults=True, so absent must not read as "unknown")
    t = _trial(tmp_path / "e", task_toml_agent_sec=1000, timeout_multiplier=1.5)
    assert effective_agent_budget_s(t / "config.json") == 1500.0
    t = _trial(tmp_path / "f", task_toml_agent_sec=1000, timeout_multiplier=1.5,
               agent_timeout_multiplier=2.0)
    assert effective_agent_budget_s(t / "config.json") == 2000.0
    # a task that declares NO agent budget → None (the framework imposes no deadline either)
    t = _trial(tmp_path / "g")
    assert effective_agent_budget_s(t / "config.json") is None
    # unreadable → None, never an exception: an undiscoverable budget falls back, it can't fail a trial
    assert effective_agent_budget_s(tmp_path / "nope" / "config.json") is None


def test_reserve_is_flat_not_proportional_to_the_budget() -> None:
    from beagle.benchmarks.harness._common import graceful_agent_timeout_s

    # The reserve pays for a FIXED post-run sequence (commit, diff, read trajectory), so it must not
    # scale with the budget: a proportional margin would have cost a 6000 s task ten minutes to buy
    # seconds of bookkeeping.
    assert graceful_agent_timeout_s(3000.0, None) == 2940.0
    assert graceful_agent_timeout_s(6000.0, None) == 5940.0     # same 60 s, not 10x more
    assert graceful_agent_timeout_s(900.0, None) == 840.0
    # capped at half the budget — headroom can't cost more time than it protects
    assert graceful_agent_timeout_s(60.0, None) == 30.0
    # no budget → the configured value stands, whatever it is
    assert graceful_agent_timeout_s(None, 600) == 600
    assert graceful_agent_timeout_s(None, None) is None


def test_shim_gives_the_agent_the_tasks_clock(monkeypatch, tmp_path) -> None:
    shim = _shim(monkeypatch)
    shim._logs_dir = _trial(tmp_path, task_toml_agent_sec=3000) / "agent"
    assert shim._task_budget_s() == 2940.0        # the task's 3000 s, less the capture reserve

    # a task that declares nothing → None, and the run config's agent.timeout becomes the bound
    shim._logs_dir = _trial(tmp_path / "none") / "agent"
    assert shim._task_budget_s() is None


def test_no_house_wall_clock_anywhere() -> None:
    # Four adapters each hardcoded 1800 s, so a house number silently outranked a benchmark's own
    # budget in four places and truncated every task worth more. There is now NO rollout default:
    # an undeclared budget raises instead (see resolve_agent_timeout).
    from beagle.agents.core import base as agent_base

    assert not hasattr(agent_base, "DEFAULT_AGENT_TIMEOUT_S")
    for mod in ("beagle/agents/mini_swe/__init__.py", "beagle/agents/monet/__init__.py",
                "beagle/agents/opencode/__init__.py", "beagle/agents/monet/_helpers.py",
                "beagle/agents/opencode/_helpers.py"):
        src = Path(mod).read_text()
        assert "1800" not in src, f"{mod} still carries a hardcoded wall clock"


def test_undeclared_budget_raises_instead_of_guessing() -> None:
    from beagle.agents.core.base import AgentBudgetUndeclared, resolve_agent_timeout
    from beagle.types import TaskContext

    declared = TaskContext(image=None, agent_timeout_s=2940.0)
    assert resolve_agent_timeout({}, declared) == 2940.0
    # The DECLARED budget wins over the config's fallback, in BOTH directions. The generated
    # configs carry agent.timeout explicitly, so treating it as a ceiling would truncate every
    # harbor task back to that number — the bug this precedence exists to prevent.
    assert resolve_agent_timeout({"timeout": 600}, declared) == 2940.0
    assert resolve_agent_timeout({"timeout": 9999}, declared) == 2940.0
    # the config value IS the bound when the benchmark declares none (docker-drop-in path)
    assert resolve_agent_timeout({"timeout": 900}, TaskContext(image=None)) == 900.0
    # nothing stated anywhere → loud, with the benchmark named
    with pytest.raises(AgentBudgetUndeclared, match="swe-rebench"):
        resolve_agent_timeout({}, TaskContext(image=None, benchmark_name="swe-rebench"))


# --- audit regressions ------------------------------------------------------------------------

def test_task_context_is_frozen_so_budget_must_be_replaced(tmp_path) -> None:
    # TaskContext is frozen: assigning agent_timeout_s raises FrozenInstanceError, which would have
    # failed EVERY pier trial that got as far as declaring its budget.
    import dataclasses

    from beagle.benchmarks.harness._common import declare_task_budget

    ctx = TaskContext(image=None, repo_path="/testbed")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.agent_timeout_s = 100.0                      # type: ignore[misc]

    trial = _trial(tmp_path, task_toml_agent_sec=3000)
    out = declare_task_budget(ctx, trial / "config.json")
    assert out is not ctx and out.agent_timeout_s == 2940.0
    assert out.repo_path == "/testbed" and ctx.agent_timeout_s is None   # original untouched


def test_pier_shim_declares_the_budget_without_mutating(monkeypatch, tmp_path) -> None:
    """The exact pier line the frozen-dataclass bug lived on, behind a fake `pier`."""
    import importlib
    import sys
    import types

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("pier"); _mod("pier.agents"); _mod("pier.agents.installed")
    _mod("pier.agents.installed.base").BaseInstalledAgent = type(  # type: ignore[attr-defined]
        "BaseInstalledAgent", (), {"__init__": lambda self, *a, **k: None})
    _mod("pier.environments")
    _mod("pier.environments.base").BaseEnvironment = type("BaseEnvironment", (), {})  # type: ignore[attr-defined]
    _mod("pier.models"); _mod("pier.models.agent")
    _mod("pier.models.agent.context").AgentContext = type("AgentContext", (), {})  # type: ignore[attr-defined]
    monkeypatch.delitem(sys.modules, "beagle.benchmarks.harness._pier_agent", raising=False)
    m = importlib.import_module("beagle.benchmarks.harness._pier_agent")

    shim = m.BeaglePierAgent.__new__(m.BeaglePierAgent)
    shim._logs_dir = _trial(tmp_path, task_toml_agent_sec=3000) / "agent"
    shim._task_ctx = TaskContext(image=None, repo_path="/testbed")
    shim._declare_budget()                               # must not raise FrozenInstanceError
    assert shim._task_ctx.agent_timeout_s == 2940.0

    # pier's own pre-agent work (the egress seal) comes off the same budget
    shim._task_ctx = TaskContext(image=None, repo_path="/testbed")
    shim._declare_budget(spent_s=40.0)
    assert shim._task_ctx.agent_timeout_s == 2900.0

    # no task budget → left as None, so the run config's agent.timeout is the bound
    shim._logs_dir = _trial(tmp_path / "none") / "agent"
    shim._task_ctx = TaskContext(image=None)
    shim._declare_budget()
    assert shim._task_ctx.agent_timeout_s is None


def test_install_time_is_deducted_from_the_declared_budget(monkeypatch) -> None:
    # The harbor agent phase covers install + run_in, but only the capture reserve sat outside
    # run_in — so a slow clone/build plus a full-clock run overran harbor's deadline and was
    # cancelled mid-capture. run() must hand run_in only what install left.
    from beagle.agents.core import base as agent_base
    from beagle.agents.core.base import Agent, Runnable, Topology
    from beagle.agents.core.spec import AgentSpec

    clock = iter([0, 0, 120, 120])   # acquire@0, install ends@120 (t0,t1,t2,t3 in run())
    monkeypatch.setattr(agent_base, "_utcnow",
                        lambda: __import__("datetime").datetime.fromtimestamp(
                            next(clock), __import__("datetime").timezone.utc))
    seen: list[float | None] = []

    class _A(Agent, Runnable):
        topology = Topology.IN_CONTAINER

        def install(self, handle, task_ctx, *, runtime): pass

        def run_in(self, handle, task, task_ctx, *, runtime):
            seen.append(task_ctx.agent_timeout_s)
            return TaskResult(task_id=task.task_id)

    class _Rt:
        def acquire(self, **kw): return "h"
        def destroy(self, h): pass

    _A(AgentSpec(name="a")).run(Task(task_id="t"),
                                TaskContext(image=None, agent_timeout_s=2940.0), runtime=_Rt())
    assert seen == [2820.0]          # 2940 declared - 120 spent on acquire+install


def test_swe_rebench_resolver_accepts_a_git_worktree() -> None:
    # A worktree's .git is a FILE, so the resolver must not filter on directories.
    env_snippet = bgl.benchmarks.get("swe-rebench").task_env()["repo_path_cmd"]
    assert "-type d" not in env_snippet and "[ -e /testbed/.git ]" in env_snippet


def test_phase_time_before_the_agent_is_deducted(tmp_path) -> None:
    """The framework's deadline starts when it calls the SHIM, not when Runnable.run() begins: the
    workspace probe + check (and pier's egress seal) are already spending it. Each layer deducts
    only its own segment, so nothing is double-counted and nothing is missed.

    Harbor-FREE on purpose — this is the deduction contract both shims share, and gating it behind
    the optional harbor extra would leave it uncovered wherever harbor isn't installed.
    """
    from beagle.benchmarks.harness._common import declare_task_budget

    trial = _trial(tmp_path, task_toml_agent_sec=3000)
    ctx = declare_task_budget(TaskContext(image=None), trial / "config.json")
    assert ctx.agent_timeout_s == 2940.0                        # nothing spent yet
    assert declare_task_budget(TaskContext(image=None), trial / "config.json",
                               spent_s=30.0).agent_timeout_s == 2910.0
    # a phase that hangs before the agent starts drives it to the floor — fail fast, rather than
    # be cancelled by the framework mid-capture
    assert declare_task_budget(TaskContext(image=None), trial / "config.json",
                               spent_s=9999.0).agent_timeout_s == 1.0
    # still None when the task declares nothing, whatever was spent
    assert declare_task_budget(TaskContext(image=None), _trial(tmp_path / "n") / "config.json",
                               spent_s=30.0).agent_timeout_s is None


def test_harbor_shim_deducts_its_own_resolution_time(monkeypatch, tmp_path) -> None:
    # Same contract through the harbor shim's accessor.
    shim = _shim(monkeypatch)
    shim._logs_dir = _trial(tmp_path, task_toml_agent_sec=3000) / "agent"
    assert shim._task_budget_s() == 2940.0
    assert shim._task_budget_s(spent_s=45.0) == 2895.0
    assert shim._task_budget_s(spent_s=9999.0) == 1.0
