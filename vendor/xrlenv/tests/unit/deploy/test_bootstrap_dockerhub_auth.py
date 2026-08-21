"""Unit tests for the Docker Hub auth bootstrap step.

``deploy/bootstrap-common.sh::install_dockerhub_auth_credentials`` is
the **only** place in the cluster that decides whether per-node
``docker pull`` calls (initiated by ``xrlenv-node``) are
authenticated against Docker Hub. End users submitting jobs to the
control plane never touch Docker Hub directly; their cold acquires
and large ``xrlenv build apply`` sweeps hit the rate-limit (or not)
based on what the operator wired up here.

The function under test:
- Writes ``${INSTALL_ROOT}/.docker/config.json`` with base64-encoded
  ``$DOCKERHUB_USER:$DOCKERHUB_TOKEN`` when both env vars are set.
- Skips silently when either is unset.
- Preserves an existing ``config.json`` on the unset path (don't
  clobber a prior `docker login` if the operator did one manually).

Tests source the bash script in a subprocess with ``chown`` /
``chmod`` stubbed to no-ops (the test environment doesn't have the
runtime user) and ``INSTALL_ROOT`` redirected at a tmp dir.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BOOTSTRAP_COMMON = _REPO_ROOT / "deploy" / "bootstrap-common.sh"
_REFRESH = _REPO_ROOT / "deploy" / "refresh.sh"


def _drive_function(
    fn_name: str,
    *,
    install_root: Path,
    runtime_user: str = "xrlenv",
    env_extras: dict[str, str] | None = None,
) -> tuple[str, str, int]:
    """Source bootstrap-common.sh in a subprocess, override the
    side-effecting external commands (``chown`` / ``chmod``) with
    no-op shell functions, redirect ``INSTALL_ROOT`` /
    ``RUNTIME_USER`` at the supplied tmp dir, then invoke ``fn_name``.

    Returns ``(stdout, stderr, returncode)``.
    """
    script = f"""
        set -euo pipefail
        # Stub side-effecting commands the test env can't actually
        # perform (we're not root and ``xrlenv`` may not exist as a
        # system user). Shell-function lookup precedes PATH so these
        # shadow the real binaries.
        chown() {{ :; }}
        chmod() {{ :; }}
        # XRLENV_CONTROL_PLANE / XRLENV_NODE_ID are checked only by
        # ``validate_required_env_for_bootstrap``; sourcing alone
        # doesn't run that gate, but harmless to set defaults.
        export XRLENV_CONTROL_PLANE="${{XRLENV_CONTROL_PLANE:-test:0}}"
        export XRLENV_NODE_ID="${{XRLENV_NODE_ID:-test-node}}"
        source "{_BOOTSTRAP_COMMON}"
        INSTALL_ROOT="{install_root}"
        RUNTIME_USER="{runtime_user}"
        {fn_name}
    """
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("DOCKERHUB_")
    }
    if env_extras:
        env.update(env_extras)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result.stdout, result.stderr, result.returncode


def test_install_writes_correct_json_when_creds_set(tmp_path: Path) -> None:
    """Both env vars set → config.json is written with the
    base64-encoded ``USER:PAT`` at the canonical Docker Hub registry
    URL ``https://index.docker.io/v1/`` (the legacy URL form docker
    daemon recognizes)."""
    _stdout, stderr, rc = _drive_function(
        "install_dockerhub_auth_credentials",
        install_root=tmp_path,
        env_extras={
            "DOCKERHUB_USER": "alice",
            "DOCKERHUB_TOKEN": "dckr_pat_xyz",
        },
    )
    assert rc == 0, f"function exited {rc}; stderr={stderr}"

    config_path = tmp_path / ".docker" / "config.json"
    assert config_path.is_file(), (
        f"config.json not written; stderr={stderr}, files in "
        f"{tmp_path}: {list(tmp_path.iterdir())}"
    )
    config = json.loads(config_path.read_text())
    assert "auths" in config
    assert "https://index.docker.io/v1/" in config["auths"]
    auth_b64 = config["auths"]["https://index.docker.io/v1/"]["auth"]
    decoded = base64.b64decode(auth_b64).decode("utf-8")
    assert decoded == "alice:dckr_pat_xyz"


def test_install_skips_when_user_missing(tmp_path: Path) -> None:
    """Token set, user unset → no file written, no error."""
    _, stderr, rc = _drive_function(
        "install_dockerhub_auth_credentials",
        install_root=tmp_path,
        env_extras={"DOCKERHUB_TOKEN": "dckr_pat_xyz"},
    )
    assert rc == 0, f"function exited {rc}; stderr={stderr}"
    assert not (tmp_path / ".docker").exists()


def test_install_skips_when_token_missing(tmp_path: Path) -> None:
    """User set, token unset → no file written, no error."""
    _, stderr, rc = _drive_function(
        "install_dockerhub_auth_credentials",
        install_root=tmp_path,
        env_extras={"DOCKERHUB_USER": "alice"},
    )
    assert rc == 0, f"function exited {rc}; stderr={stderr}"
    assert not (tmp_path / ".docker").exists()


def test_install_preserves_existing_config_when_creds_absent(
    tmp_path: Path,
) -> None:
    """No new creds + existing config.json → don't clobber. Lets the
    operator's prior ``sudo -u xrlenv docker login`` survive a
    bootstrap re-run that doesn't carry creds."""
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    existing = docker_dir / "config.json"
    existing.write_text('{"auths": {"prior": {"auth": "preexisting"}}}')

    _, stderr, rc = _drive_function(
        "install_dockerhub_auth_credentials",
        install_root=tmp_path,
    )
    assert rc == 0, f"function exited {rc}; stderr={stderr}"

    # Preserved verbatim.
    assert existing.is_file()
    config = json.loads(existing.read_text())
    assert config["auths"]["prior"]["auth"] == "preexisting"


def test_install_overwrites_existing_config_when_new_creds_supplied(
    tmp_path: Path,
) -> None:
    """Existing config.json + new creds → overwrite with the new
    creds. Operator workflow: re-run bootstrap with rotated PAT,
    expect the new value to land."""
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    existing = docker_dir / "config.json"
    existing.write_text('{"auths": {"old": {"auth": "stale"}}}')

    _drive_function(
        "install_dockerhub_auth_credentials",
        install_root=tmp_path,
        env_extras={
            "DOCKERHUB_USER": "alice",
            "DOCKERHUB_TOKEN": "dckr_pat_NEW",
        },
    )
    config = json.loads(existing.read_text())
    # Old entry must be gone; new entry must be present + correct.
    assert "old" not in config["auths"]
    auth_b64 = config["auths"]["https://index.docker.io/v1/"]["auth"]
    assert base64.b64decode(auth_b64).decode() == "alice:dckr_pat_NEW"


def test_warn_fires_when_no_creds_and_no_file(tmp_path: Path) -> None:
    """Loud end-of-bootstrap warning when both env vars missing AND
    no surviving config.json exists. The warning must mention the
    env vars + the recovery path so the operator knows what to do."""
    _, stderr, rc = _drive_function(
        "warn_if_no_dockerhub_auth",
        install_root=tmp_path,
    )
    assert rc == 0
    assert "WARNING: no Docker Hub auth" in stderr
    assert "DOCKERHUB_USER" in stderr
    assert "DOCKERHUB_TOKEN" in stderr
    assert "docker login" in stderr


def test_warn_silent_when_creds_set(tmp_path: Path) -> None:
    """No warning when creds were supplied — the install step
    already wrote the config."""
    _, stderr, rc = _drive_function(
        "warn_if_no_dockerhub_auth",
        install_root=tmp_path,
        env_extras={
            "DOCKERHUB_USER": "alice",
            "DOCKERHUB_TOKEN": "dckr_pat_xyz",
        },
    )
    assert rc == 0
    assert "WARNING: no Docker Hub auth" not in stderr


def test_warn_silent_when_existing_config_present(tmp_path: Path) -> None:
    """No warning when the runtime user already has a config.json
    on disk (operator did ``sudo -u xrlenv docker login`` previously)."""
    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    (docker_dir / "config.json").write_text("{}")

    _, stderr, rc = _drive_function(
        "warn_if_no_dockerhub_auth",
        install_root=tmp_path,
    )
    assert rc == 0
    assert "WARNING: no Docker Hub auth" not in stderr


@pytest.mark.skipif(
    not _BOOTSTRAP_COMMON.is_file(),
    reason="deploy/bootstrap-common.sh missing",
)
def test_bootstrap_common_is_source_safe() -> None:
    """Sanity: sourcing the script alone must not raise. The whole
    bootstrap flow's idempotency relies on this. Catch regressions
    where someone moves a top-level invocation outside a function."""
    script = f"""
        set -euo pipefail
        chown() {{ :; }}
        chmod() {{ :; }}
        source "{_BOOTSTRAP_COMMON}"
    """
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"sourcing bootstrap-common.sh raised: stderr={result.stderr}"
    )


@pytest.mark.skipif(
    not _BOOTSTRAP_COMMON.is_file(),
    reason="deploy/bootstrap-common.sh missing",
)
def test_bootstrap_xrlenv_installs_auth_before_systemd_unit() -> None:
    """Audit M1 (2026-05-12): ``install_dockerhub_auth_credentials``
    must run BEFORE ``install_systemd_unit`` inside
    ``bootstrap_xrlenv``. The latter calls ``systemctl restart
    xrlenv-node``; docker-py's ``APIClient.__init__`` reads
    ``~/.docker/config.json`` once at construction and caches the
    auth dict. If we write the auth file after the daemon has
    started, the running process keeps an empty-auth cache and
    every pull stays unauthenticated until the next restart.

    Reads the bash source directly and asserts the function-call
    order in ``bootstrap_xrlenv`` so the legacy wrapper path can't
    drift from this invariant.
    """
    src = _BOOTSTRAP_COMMON.read_text(encoding="utf-8")

    # Locate the ``bootstrap_xrlenv()`` body — between the opening
    # ``bootstrap_xrlenv() {`` and the matching closing ``}`` at
    # column 0. Bash doesn't give us a precise AST, but the function
    # is small and follows the project's consistent formatting.
    import re

    match = re.search(
        r"^bootstrap_xrlenv\(\)\s*\{\n(.*?)^\}",
        src, re.MULTILINE | re.DOTALL,
    )
    assert match is not None, (
        "couldn't locate bootstrap_xrlenv() in bootstrap-common.sh"
    )
    body = match.group(1)

    auth_pos = body.find("install_dockerhub_auth_credentials")
    systemd_pos = body.find("install_systemd_unit")
    assert auth_pos != -1, (
        "install_dockerhub_auth_credentials missing from "
        "bootstrap_xrlenv body"
    )
    assert systemd_pos != -1, (
        "install_systemd_unit missing from bootstrap_xrlenv body"
    )
    assert auth_pos < systemd_pos, (
        "M1 regression: install_dockerhub_auth_credentials must "
        "run BEFORE install_systemd_unit so the auth file lands "
        "before docker-py's APIClient first construction caches "
        f"an empty auth dict. Current order: auth at offset "
        f"{auth_pos}, systemd_unit at offset {systemd_pos} "
        f"(auth must be lower-offset = earlier)."
    )


@pytest.mark.skipif(
    not _REFRESH.is_file(),
    reason="deploy/refresh.sh missing",
)
def test_refresh_calls_dockerhub_auth_before_systemd_unit_restart() -> None:
    """``deploy/refresh.sh`` is the fast path operators use to push
    a new xrlenv release to an existing node (``git pull`` +
    reinstall + restart the daemon). It must honor the same
    "auth before unit-restart" ordering as ``bootstrap_xrlenv``,
    otherwise a PAT-rotation via refresh leaves the running daemon
    on the old auth until the NEXT restart — the very docker-py
    APIClient-caches-auth-at-init bug the bootstrap fix closes.
    """
    src = _REFRESH.read_text(encoding="utf-8")

    # Refresh.sh is a top-level script (not function-wrapped). The
    # function NAMES appear in comments as well as call sites, so
    # match only line-start (no leading ``#``) call invocations to
    # skip doc-comment references.
    import re

    def _call_pos(name: str) -> int:
        m = re.search(rf"^{name}\b", src, re.MULTILINE)
        return m.start() if m else -1

    auth_pos = _call_pos("install_dockerhub_auth_credentials")
    systemd_pos = _call_pos("install_systemd_unit")
    assert auth_pos != -1, (
        "install_dockerhub_auth_credentials call missing from "
        "refresh.sh; operator can't rotate Docker Hub PAT via the "
        "refresh path"
    )
    assert systemd_pos != -1, (
        "install_systemd_unit call missing from refresh.sh — refresh "
        "is supposed to re-write node.env + restart the daemon"
    )
    assert auth_pos < systemd_pos, (
        "refresh.sh regression: install_dockerhub_auth_credentials "
        "must run BEFORE install_systemd_unit (which restarts the "
        "daemon). PAT rotation would otherwise not take effect "
        "until the next daemon restart. Current order: auth at "
        f"offset {auth_pos}, systemd_unit at offset {systemd_pos}."
    )


@pytest.mark.skipif(
    not _REFRESH.is_file(),
    reason="deploy/refresh.sh missing",
)
def test_refresh_calls_warn_if_no_dockerhub_auth() -> None:
    """Refresh should advise the operator at the end of a successful
    run when no Docker Hub auth is wired (same UX as bootstrap).
    Self-short-circuits when a config.json exists, so operators
    who set it once aren't spammed every refresh."""
    src = _REFRESH.read_text(encoding="utf-8")
    assert "warn_if_no_dockerhub_auth" in src, (
        "refresh.sh should invoke warn_if_no_dockerhub_auth so the "
        "operator gets the same loud advisory as the bootstrap path "
        "when no Docker Hub auth is configured on this node."
    )
