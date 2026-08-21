"""Unit tests for run_all_xrlenv batch driver (no harness/cluster needed).

Locks in the contract that run_all is the SINGLE parallel entry point for BOTH the
oracle and a real agent: the agent/model are parameterized (default oracle) and flow
into the worker command, while all the scaffolding (--force, fleet footprint, cpu
pinning, mem cap) is preserved regardless of agent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_all_xrlenv as R


def _worker_cmd(agent="oracle", model="none", **kw):
    t = R._Task(
        "repo-x", "mid-y",
        data_root=Path("/d"), run_ws=Path("/w"), log_dir=Path("/l"),
        fleet_cpu=kw.get("fleet_cpu"), fleet_mem_gb=kw.get("fleet_mem_gb"),
        mem_per_cpu_gb=2.0, copy_testbed=False, passthru=[],
        agent=agent, model=model,
    )
    return t._worker_cmd(Path("/ws"))


# --- agent/model parameterization ---
def test_default_agent_is_oracle():
    a = R._parse_args(["--run-name", "x"])
    assert (a.agent, a.model) == ("oracle", "none")


def test_real_agent_and_model_parsed():
    a = R._parse_args(["--run-name", "x", "--agent", "claude-code", "--model", "sonnet-4-6"])
    assert (a.agent, a.model) == ("claude-code", "sonnet-4-6")


def test_real_agent_without_model_is_rejected():
    with pytest.raises(SystemExit):
        R._parse_args(["--run-name", "x", "--agent", "claude-code"])  # model defaults to 'none'


def test_oracle_still_allows_model_none():
    a = R._parse_args(["--run-name", "x", "--agent", "oracle"])  # 'none' is fine for oracle
    assert a.model == "none"


# --- the worker command reflects the agent, scaffolding preserved ---
def test_worker_cmd_uses_oracle_by_default():
    cmd = _worker_cmd()
    i = cmd.index("--agent")
    assert cmd[i:i + 4] == ["--agent", "oracle", "--model", "none"]


def test_worker_cmd_uses_real_agent():
    cmd = _worker_cmd(agent="claude-code", model="sonnet-4-6")
    i = cmd.index("--agent")
    assert cmd[i:i + 4] == ["--agent", "claude-code", "--model", "sonnet-4-6"]


def test_worker_cmd_preserves_scaffolding_for_real_agent():
    """A real-agent worker still gets --force + the fleet footprint + mem cap — i.e.
    all the batch scaffolding, not just the oracle path."""
    cmd = _worker_cmd(agent="claude-code", model="sonnet-4-6", fleet_cpu=66, fleet_mem_gb=132)
    assert "--force" in cmd
    assert "--repo-name" in cmd and "repo-x" in cmd
    assert "--mem-per-cpu-gb" in cmd
    assert "--fleet-footprint-cpu" in cmd and "66" in cmd
    assert "--fleet-footprint-mem-gb" in cmd and "132" in cmd
