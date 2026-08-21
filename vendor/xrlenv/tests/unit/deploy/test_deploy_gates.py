"""Fail-closed behavior of the shared deploy gates (slurm_scripts/_deploy_gates.sh).

Runs the REAL bash gate functions with STUBBED squeue/scontrol/ssh/xrlenv/sleep
on PATH, so the audit's deploy failure paths (empty host expansion,
scheduler/SSH/probe errors, force behavior, unregistered agents) are exercised
without a live cluster or Slurm. Each gate must return nonzero on an unsafe
condition (aborting the `set -e` deploy) unless XRLENV_DEPLOY_FORCE=1.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

GATES = Path(__file__).resolve().parents[3] / "slurm_scripts" / "_deploy_gates.sh"
_SLURM = Path(__file__).resolve().parents[3] / "slurm_scripts"


def _stub(bindir: Path, name: str, body: str) -> None:
    p = bindir / name
    p.write_text("#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_gate(
    tmp_path: Path, stubs: dict[str, str], call: str, *, force: bool = False
) -> subprocess.CompletedProcess[str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    # `sleep` -> no-op so the poll loops run instantly.
    for name, body in {"sleep": "exit 0", **stubs}.items():
        _stub(bindir, name, body)
    script = (
        "set -euo pipefail\n"
        f'export PATH="{bindir}:$PATH"\n'
        "SSH_OPTS=(-o BatchMode=yes)\n"
        + ("export XRLENV_DEPLOY_FORCE=1\n" if force else "")
        + f'source "{GATES}"\n'
        + call
        + "\n"
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )


# ── deploy_wait_node_running ──────────────────────────────────────────────────


def test_node_running_ok(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"squeue": "echo RUNNING"},
                  "deploy_wait_node_running J 3 0")
    assert r.returncode == 0


def test_node_never_running_is_fatal(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"squeue": "echo PENDING"},
                  "deploy_wait_node_running J 3 0")
    assert r.returncode == 1
    assert "did not reach RUNNING" in r.stderr


def test_node_never_running_force_overrides(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"squeue": "echo PENDING"},
                  "deploy_wait_node_running J 3 0", force=True)
    assert r.returncode == 0


# ── deploy_wait_control_gone ──────────────────────────────────────────────────


def test_control_gone_and_released(tmp_path: Path) -> None:
    # squeue empty output => job gone; ssh exit 0 => process released.
    r = _run_gate(tmp_path, {"squeue": "exit 0", "ssh": "exit 0"},
                  "deploy_wait_control_gone J cp 3 0")
    assert r.returncode == 0


def test_control_still_queued_is_fatal(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"squeue": "echo RUNNING", "ssh": "exit 0"},
                  "deploy_wait_control_gone J cp 3 0")
    assert r.returncode == 1
    assert "not confirmed gone" in r.stderr


def test_control_squeue_error_not_read_as_gone(tmp_path: Path) -> None:
    # A FAILING scheduler query (exit 1, no output) must NOT be treated as
    # "gone" — the set -e-safe `if q=$(...)` capture keeps polling, then aborts.
    r = _run_gate(tmp_path, {"squeue": "exit 1", "ssh": "exit 0"},
                  "deploy_wait_control_gone J cp 3 0")
    assert r.returncode == 1
    assert "not confirmed gone" in r.stderr


def test_control_process_still_present_is_fatal(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"squeue": "exit 0", "ssh": "exit 3"},
                  "deploy_wait_control_gone J cp 3 0")
    assert r.returncode == 1
    assert "still holds the DB" in r.stderr


def test_control_ssh_error_is_fatal(tmp_path: Path) -> None:
    # An SSH failure while checking the old process must fail closed, not warn.
    r = _run_gate(tmp_path, {"squeue": "exit 0", "ssh": "exit 255"},
                  "deploy_wait_control_gone J cp 3 0")
    assert r.returncode == 1
    assert "could not verify the old CP is gone" in r.stderr


def test_control_ssh_error_force_overrides(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"squeue": "exit 0", "ssh": "exit 255"},
                  "deploy_wait_control_gone J cp 3 0", force=True)
    assert r.returncode == 0
    assert "proceeding despite unverified" in r.stdout


# ── deploy_verify_fleet ───────────────────────────────────────────────────────

_HOSTS = 'echo -e "h1\\nh2"'


def test_verify_fleet_all_good(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {
        "squeue": 'echo "nodelist"',            # -o %N nodelist (nonempty)
        "scontrol": _HOSTS,                     # expands to h1 h2
        "ssh": 'echo "cp:50051"',               # node.env CP matches
        "xrlenv": 'echo -e "aws-h1 connected\\naws-h2 connected"',
    }, 'deploy_verify_fleet J "cp:50051" /tmp/s.db xrlenv 2 0')
    assert r.returncode == 0
    assert "OK: all" in r.stdout


def test_verify_fleet_empty_host_expansion_is_fatal(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {
        "squeue": 'echo "nodelist"',
        "scontrol": "exit 0",                   # empty expansion
        "ssh": 'echo cp', "xrlenv": "echo",
    }, 'deploy_verify_fleet J "cp:50051" /tmp/s.db xrlenv 2 0')
    assert r.returncode == 1
    assert "produced no hosts" in r.stderr


def test_verify_fleet_unresolved_nodelist_is_fatal(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {
        "squeue": "exit 0",                     # empty nodelist
        "scontrol": _HOSTS, "ssh": 'echo cp', "xrlenv": "echo",
    }, 'deploy_verify_fleet J "cp:50051" /tmp/s.db xrlenv 2 0')
    assert r.returncode == 1
    assert "could not resolve" in r.stderr


def test_verify_fleet_config_mismatch_is_fatal(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {
        "squeue": 'echo nodelist', "scontrol": _HOSTS,
        "ssh": 'echo "WRONG:50051"',            # node.env CP != expected
        "xrlenv": 'echo -e "aws-h1 connected\\naws-h2 connected"',
    }, 'deploy_verify_fleet J "cp:50051" /tmp/s.db xrlenv 2 0')
    assert r.returncode == 1
    assert "stale/missing agent config" in r.stderr


def test_verify_fleet_unregistered_is_fatal(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {
        "squeue": 'echo nodelist', "scontrol": _HOSTS,
        "ssh": 'echo "cp:50051"',               # config OK
        "xrlenv": "echo",                       # nothing 'connected'
    }, 'deploy_verify_fleet J "cp:50051" /tmp/s.db xrlenv 2 0')
    assert r.returncode == 1
    assert "never registered" in r.stderr


def test_verify_fleet_unregistered_force_overrides(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {
        "squeue": 'echo nodelist', "scontrol": _HOSTS,
        "ssh": 'echo "cp:50051"', "xrlenv": "echo",
    }, 'deploy_verify_fleet J "cp:50051" /tmp/s.db xrlenv 2 0', force=True)
    assert r.returncode == 0
    assert "not registered (XRLENV_DEPLOY_FORCE=1)" in r.stdout


def test_verify_fleet_empty_hosts_force_skips_without_false_success(tmp_path: Path) -> None:
    # Under FORCE the empty-host abort is bypassed, but the gate must NOT then
    # loop zero times and print the misleading "all nodes connected" success.
    # It should WARN that it skipped and return 0 (proof of nothing).
    r = _run_gate(tmp_path, {
        "squeue": 'echo "nodelist"',
        "scontrol": "exit 0",                   # empty host expansion
        "ssh": 'echo cp', "xrlenv": "echo",
    }, 'deploy_verify_fleet J "cp:50051" /tmp/s.db xrlenv 2 0', force=True)
    assert r.returncode == 0
    assert "no hosts to verify" in r.stdout
    assert "OK: all" not in r.stdout, (
        "empty host set under FORCE must not report the fleet as connected"
    )


# ── full-script wiring (not just the gate functions in isolation) ─────────────
#
# The tests above drive the gate FUNCTIONS. These drive the REAL
# deploy_{prod,dev}.sh end-to-end under PATH stubs to prove the gates are
# actually WIRED into the scripts: a failing node-running gate must abort the
# whole `set -e` deploy BEFORE the control-plane restart. All scheduler/ssh
# commands are stubbed, and the node gate fails before any ssh to a real host,
# so nothing touches a live cluster.


@pytest.mark.parametrize("script", ["deploy_prod.sh", "deploy_dev.sh"])
def test_full_deploy_script_aborts_before_cp_restart_on_node_gate_failure(
    tmp_path: Path, script: str,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # squeue NEVER reports RUNNING → deploy_wait_node_running times out → abort.
    # sleep is a no-op so the gate's 60-iteration wait runs instantly.
    for name, body in {
        "sbatch": 'echo "Submitted batch job 1"',
        "scancel": "exit 0",
        "sleep": "exit 0",
        "squeue": "echo PENDING",
        "scontrol": "exit 0",
        "ssh": "exit 0",
        "xrlenv": "exit 0",
    }.items():
        _stub(bindir, name, body)

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env.pop("XRLENV_DEPLOY_FORCE", None)  # FORCE would defeat the fail-closed gate

    r = subprocess.run(
        ["bash", str(_SLURM / script)],
        capture_output=True, text=True, env=env,
    )
    combined = r.stdout + r.stderr
    assert r.returncode != 0, (
        f"{script}: an unsatisfied node-running gate must abort the deploy"
    )
    assert "did not reach RUNNING" in combined, (
        f"{script}: expected the node-running gate's abort message"
    )
    # The CP restart (step 2) must NOT have started — proof the gate gates the
    # real script in-line, not just as a standalone function.
    assert "restarting control" not in combined, (
        f"{script}: proceeded to the control-plane restart despite the node gate "
        "failing — the gate is not actually wired into the script's control flow"
    )
