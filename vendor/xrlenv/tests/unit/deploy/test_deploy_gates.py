"""Fail-closed behavior of the shared deploy gates (slurm_scripts/lib/_deploy_gates.sh).

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

GATES = Path(__file__).resolve().parents[3] / "slurm_scripts" / "lib" / "_deploy_gates.sh"
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


# ── deploy_check_topology ─────────────────────────────────────────────────────
#
# The pre-deploy topology gate: clusters.yaml is authoritative, so the checkout's
# .env + generated scripts must match it (proved by the generator's --check). The
# generator command is passed as args so tests inject a stub instead of running
# the real generator.


def test_check_topology_in_sync_ok(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"gen": "exit 0"}, "deploy_check_topology dev gen")
    assert r.returncode == 0
    assert "in sync with clusters.yaml" in r.stdout


def test_check_topology_drift_is_fatal(tmp_path: Path) -> None:
    # generator --check exits nonzero → .env/scripts drifted → fail closed.
    r = _run_gate(tmp_path, {"gen": "exit 1"}, "deploy_check_topology dev gen")
    assert r.returncode == 1
    assert "drifted from clusters.yaml" in r.stderr


def test_check_topology_drift_force_overrides(tmp_path: Path) -> None:
    r = _run_gate(tmp_path, {"gen": "exit 1"},
                  "deploy_check_topology dev gen", force=True)
    assert r.returncode == 0
    assert "XRLENV_DEPLOY_FORCE=1" in r.stderr


def test_check_topology_skip_flag_bypasses_even_drift(tmp_path: Path) -> None:
    r = _run_gate(
        tmp_path, {"gen": "exit 1"},
        "export XRLENV_DEPLOY_SKIP_ENV_CHECK=1\ndeploy_check_topology dev gen",
    )
    assert r.returncode == 0
    assert "skipping" in r.stdout


def test_check_topology_skip_wins_over_force(tmp_path: Path) -> None:
    # When BOTH XRLENV_DEPLOY_SKIP_ENV_CHECK=1 AND XRLENV_DEPLOY_FORCE=1 are
    # set with a drifted generator (exit 1), SKIP must win: the function
    # returns before consulting FORCE, so the output must be the "skipping"
    # message, NOT the force-warn. This rules out an implementation that
    # checked FORCE before SKIP and accidentally applied force semantics.
    r = _run_gate(
        tmp_path, {"gen": "exit 1"},
        "export XRLENV_DEPLOY_SKIP_ENV_CHECK=1\ndeploy_check_topology dev gen",
        force=True,
    )
    assert r.returncode == 0
    assert "skipping" in r.stdout
    assert "XRLENV_DEPLOY_FORCE" not in r.stderr, (
        "SKIP=1 must short-circuit before FORCE is consulted — "
        "force-warn output must NOT appear when SKIP is set"
    )


def test_check_topology_passes_cluster_flags_to_generator(tmp_path: Path) -> None:
    # The gate must invoke the generator with --cluster/--env-cluster <c> --check.
    r = _run_gate(tmp_path, {"gen": 'echo "$@"; exit 0'},
                  "deploy_check_topology prod gen")
    assert r.returncode == 0
    assert "--cluster prod --env-cluster prod --check" in r.stdout


# ── deploy_check_registry_storage ─────────────────────────────────────────────
#
# Registry-launching clusters (deploy_registry: true) must set the registry
# blob-store paths in .env — they have NO default (the /fsx assumption isn't
# portable). The gate fails closed on any missing key BEFORE jobs restart.


def _write_env(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "envfile"
    p.write_text(body)
    return p


def test_check_registry_storage_all_present_ok(tmp_path: Path) -> None:
    env = _write_env(
        tmp_path,
        "XRLENV_MIRROR_REGISTRY_STORAGE=/x/m\n"
        "XRLENV_PRIVATE_REGISTRY_STORAGE=/x/p\n"
        "XRLENV_SCRATCH_REGISTRY_STORAGE=/x/s\n",
    )
    r = _run_gate(
        tmp_path, {},
        f"deploy_check_registry_storage {env} XRLENV_MIRROR_REGISTRY_STORAGE "
        "XRLENV_PRIVATE_REGISTRY_STORAGE XRLENV_SCRATCH_REGISTRY_STORAGE",
    )
    assert r.returncode == 0


def test_check_registry_storage_missing_is_fatal(tmp_path: Path) -> None:
    env = _write_env(
        tmp_path,
        "XRLENV_MIRROR_REGISTRY_STORAGE=/x/m\n"
        "XRLENV_PRIVATE_REGISTRY_STORAGE=/x/p\n",  # scratch missing
    )
    r = _run_gate(
        tmp_path, {},
        f"deploy_check_registry_storage {env} XRLENV_MIRROR_REGISTRY_STORAGE "
        "XRLENV_PRIVATE_REGISTRY_STORAGE XRLENV_SCRATCH_REGISTRY_STORAGE",
    )
    assert r.returncode == 1
    assert "XRLENV_SCRATCH_REGISTRY_STORAGE is not set" in r.stderr


def test_check_registry_storage_empty_value_counts_as_missing(tmp_path: Path) -> None:
    env = _write_env(tmp_path, "XRLENV_PRIVATE_REGISTRY_STORAGE=\n")  # empty value
    r = _run_gate(tmp_path, {},
                  f"deploy_check_registry_storage {env} XRLENV_PRIVATE_REGISTRY_STORAGE")
    assert r.returncode == 1


def test_check_registry_storage_alias_satisfies_spec(tmp_path: Path) -> None:
    # The mirror accepts the deprecated XRLENV_REGISTRY_STORAGE alias via KEY|ALIAS.
    env = _write_env(tmp_path, "XRLENV_REGISTRY_STORAGE=/x/legacy\n")
    r = _run_gate(
        tmp_path, {},
        f'deploy_check_registry_storage {env} '
        '"XRLENV_MIRROR_REGISTRY_STORAGE|XRLENV_REGISTRY_STORAGE"',
    )
    assert r.returncode == 0


def test_check_registry_storage_force_overrides(tmp_path: Path) -> None:
    env = _write_env(tmp_path, "# nothing\n")
    r = _run_gate(tmp_path, {},
                  f"deploy_check_registry_storage {env} XRLENV_PRIVATE_REGISTRY_STORAGE",
                  force=True)
    assert r.returncode == 0
    assert "XRLENV_DEPLOY_FORCE=1" in r.stderr


def test_check_registry_storage_skip_flag_bypasses(tmp_path: Path) -> None:
    # Same skip as the topology gate — the full deploy-script node-gate test
    # relies on it to get past both .env pre-flight checks.
    env = _write_env(tmp_path, "# nothing\n")
    r = _run_gate(
        tmp_path, {},
        "export XRLENV_DEPLOY_SKIP_ENV_CHECK=1\n"
        f"deploy_check_registry_storage {env} XRLENV_PRIVATE_REGISTRY_STORAGE",
    )
    assert r.returncode == 0
    assert "skipping" in r.stdout


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
    # Skip the .env↔clusters.yaml topology gate — it runs the real generator
    # against the real checkout's .env, which this test doesn't control; this
    # test targets the NODE gate specifically. (deploy_check_topology has its own
    # unit tests below.) Skipping it is independent of FORCE, so the node gate
    # still fails closed.
    env["XRLENV_DEPLOY_SKIP_ENV_CHECK"] = "1"

    r = subprocess.run(
        ["bash", str(_SLURM / "generated" / script)],
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


# ── Deploy-time config == runtime config ──────────────────────────────────────
#
# The node + control jobs run ``set -a; source ./.env; set +a`` before starting.
# The deploy script's pre-flights must read that SAME file, or they judge a
# different cluster than the one they are about to bounce.
#
# Regression (2026-08-26): the deploy scripts never loaded ``.env``, while
# ``xrlenv_plugins/sysbox/pin.env`` — which they source — reads
# ``$XRLENV_SYSBOX_VENDOR_DIR`` from the ENVIRONMENT. Every sysbox-pool deploy
# aborted with "XRLENV_SYSBOX_VENDOR_DIR is not set" on a checkout whose .env
# set it correctly. The topology + registry gates were unaffected only because
# they grep the file rather than reading the environment, which is what made the
# failure look like an operator misconfiguration rather than a script bug.


@pytest.mark.parametrize("script", ["deploy_prod.sh", "deploy_dev.sh", "deploy_cn.sh"])
def test_deploy_script_loads_dotenv(script: str) -> None:
    body = (_SLURM / "generated" / script).read_text()
    assert 'source "${REPO_ROOT}/.env"' in body, (
        f"{script}: does not load the checkout's .env, so deploy-time pre-flights "
        "read different config than the jobs they launch"
    )
    assert "set -a" in body, (
        f"{script}: .env must be sourced under `set -a` so the values are EXPORTED "
        "— pin.env and the ssh/sbatch calls read them from the environment"
    )


@pytest.mark.parametrize("script", ["deploy_prod.sh", "deploy_dev.sh", "deploy_cn.sh"])
def test_deploy_script_loads_dotenv_before_sourcing_pin_env(script: str) -> None:
    """Ordering is the whole point: pin.env reads the environment."""
    body = (_SLURM / "generated" / script).read_text()
    pin = 'source "${REPO_ROOT}/xrlenv_plugins/sysbox/pin.env"'
    if pin not in body:
        pytest.skip(f"{script} has no sysbox pin gate")
    assert body.index('source "${REPO_ROOT}/.env"') < body.index(pin), (
        f"{script}: sources pin.env BEFORE loading .env — pin.env reads "
        "$XRLENV_SYSBOX_VENDOR_DIR from the environment and would abort the deploy"
    )


@pytest.mark.parametrize("script", ["deploy_prod.sh", "deploy_dev.sh", "deploy_cn.sh"])
def test_deploy_script_tolerates_a_missing_dotenv(script: str) -> None:
    """A checkout with no .env must reach the topology gate, which reports it
    properly — not die on an unguarded ``source``."""
    body = (_SLURM / "generated" / script).read_text()
    assert '[ -f "${REPO_ROOT}/.env" ]' in body, (
        f"{script}: the .env source is unguarded; a checkout without one would "
        "fail with a raw `source: No such file` instead of the topology gate's "
        "explanation"
    )
