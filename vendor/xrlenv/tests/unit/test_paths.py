"""Tests for :mod:`xrlenv.paths` — the ``$XRLENV_HOME`` operator-state root.

Two layers are covered:

1. The resolver itself (:func:`xrlenv.paths.xrlenv_home` + the derived
   helpers), exercised in-process with ``monkeypatch`` since it reads
   ``os.environ`` fresh on every call.
2. The *wiring* — the module-level path constants in the control plane / CLI
   (``DEFAULT_STATE_DB``, ``DEFAULT_SECRETS_ROOT``, …) are evaluated at import
   time, so a fresh interpreter is spawned with ``XRLENV_HOME`` set (or a
   ``.env`` in CWD) to prove the whole tree relocates. This is the property
   that makes a side-by-side dev cluster on a shared FSx home safe: a dev
   checkout's ``.env`` names its own ``XRLENV_HOME`` and its ``state.db`` /
   token store never collide with prod's.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from xrlenv import paths

# ──────────────────────────────────────────────────────────────────────────────
# Resolver (call-time, in-process)
# ──────────────────────────────────────────────────────────────────────────────


def test_xrlenv_home_defaults_to_dot_xrlenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XRLENV_HOME", raising=False)
    assert paths.xrlenv_home() == Path.home() / ".xrlenv"


def test_xrlenv_home_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_HOME", "/srv/clusters/dev")
    assert paths.xrlenv_home() == Path("/srv/clusters/dev")


def test_xrlenv_home_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_HOME", "~/cluster-dev")
    assert paths.xrlenv_home() == Path.home() / "cluster-dev"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_env_falls_back_to_default(
    blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare ``XRLENV_HOME=`` line in ``.env`` must not relocate state to the
    filesystem root — it's treated as unset."""
    monkeypatch.setenv("XRLENV_HOME", blank)
    assert paths.xrlenv_home() == Path.home() / ".xrlenv"


def test_derived_helpers_hang_off_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_HOME", "/srv/clusters/dev")
    monkeypatch.delenv("XRLENV_STATE_DB_PATH", raising=False)  # state.db not relocated here
    home = Path("/srv/clusters/dev")
    assert paths.state_db_path() == home / "state.db"
    assert paths.runs_root() == home / "runs"
    assert paths.secrets_root() == home / "secrets"
    assert paths.admin_cache_root() == home / "admin-cache" / "trajectories"
    assert paths.build_context_cache_root() == home / "build-context-cache"


def test_state_db_path_honors_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # XRLENV_STATE_DB_PATH relocates ONLY state.db (CP-box-local disk) — the rest of the tree
    # still hangs off XRLENV_HOME. This is the Lustre-latency / WAL fix.
    monkeypatch.setenv("XRLENV_HOME", "/srv/clusters/dev")
    monkeypatch.setenv("XRLENV_STATE_DB_PATH", "/opt/xrlenv/state.db")
    assert paths.state_db_path() == Path("/opt/xrlenv/state.db")
    # secrets/runs are unaffected — only state.db moved.
    assert paths.secrets_root() == Path("/srv/clusters/dev") / "secrets"
    assert paths.runs_root() == Path("/srv/clusters/dev") / "runs"


def test_state_db_path_override_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_STATE_DB_PATH", "~/local/state.db")
    assert paths.state_db_path() == Path.home() / "local" / "state.db"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_state_db_path_blank_override_falls_back(
    blank: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bare XRLENV_STATE_DB_PATH= line is treated as unset (don't relocate state.db to "/").
    monkeypatch.setenv("XRLENV_HOME", "/srv/clusters/dev")
    monkeypatch.setenv("XRLENV_STATE_DB_PATH", blank)
    assert paths.state_db_path() == Path("/srv/clusters/dev") / "state.db"


def test_resolver_is_dynamic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read fresh each call — a process that sets the var late still sees it."""
    monkeypatch.delenv("XRLENV_HOME", raising=False)
    assert paths.xrlenv_home() == Path.home() / ".xrlenv"
    monkeypatch.setenv("XRLENV_HOME", "/srv/late")
    assert paths.xrlenv_home() == Path("/srv/late")


# ──────────────────────────────────────────────────────────────────────────────
# Wiring (import-time constants, fresh interpreter)
# ──────────────────────────────────────────────────────────────────────────────


_PROBE = """
    from xrlenv import paths
    from xrlenv.cli.commands import DEFAULT_XRLENV_HOME, DEFAULT_STATE_DB, DEFAULT_RUNS_ROOT
    from xrlenv.control.security import DEFAULT_SECRETS_ROOT
    from xrlenv.control.trajectory_cache import DEFAULT_CACHE_ROOT
    from xrlenv.control.distributed_runtime import DEFAULT_RUNS_ROOT as DIST_RUNS
    print("HOME", paths.xrlenv_home())
    print("STATE_DB", DEFAULT_STATE_DB)
    print("RUNS", DEFAULT_RUNS_ROOT)
    print("DIST_RUNS", DIST_RUNS)
    print("SECRETS", DEFAULT_SECRETS_ROOT)
    print("CACHE", DEFAULT_CACHE_ROOT)
"""


def _probe(env: dict[str, str], cwd: str) -> dict[str, str]:
    """Import the path constants in a fresh interpreter and return them.

    ``cwd`` controls where the ``.env`` upward-walk starts; ``env`` is the full
    environment (no inheritance) so the repo-root ``.env`` / shell vars can't
    leak in and make the assertion non-deterministic.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE)],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=env,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stderr}"
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, val = line.partition(" ")
        out[key] = val
    return out


def _base_env(tmp_path: Path) -> dict[str, str]:
    # Minimal, hermetic env. HOME points at a scratch dir so the *default*
    # (~/.xrlenv) path is also under tmp and never the real operator home.
    home = tmp_path / "fake-home"
    home.mkdir()
    return {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "XRLENV_DOTENV": "off",  # default: ignore any ambient .env
    }


def test_constants_default_to_home_when_unset(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    got = _probe(env, cwd=str(tmp_path))
    expected_home = Path(env["HOME"]) / ".xrlenv"
    assert got["HOME"] == str(expected_home)
    assert got["STATE_DB"] == str(expected_home / "state.db")
    assert got["SECRETS"] == str(expected_home / "secrets")
    assert got["CACHE"] == str(expected_home / "admin-cache" / "trajectories")


def test_constants_relocate_with_xrlenv_home_env(tmp_path: Path) -> None:
    cluster = tmp_path / "dev-cluster"
    env = {**_base_env(tmp_path), "XRLENV_HOME": str(cluster)}
    got = _probe(env, cwd=str(tmp_path))
    assert got["HOME"] == str(cluster)
    assert got["STATE_DB"] == str(cluster / "state.db")
    assert got["RUNS"] == str(cluster / "runs")
    assert got["DIST_RUNS"] == str(cluster / "runs")
    assert got["SECRETS"] == str(cluster / "secrets")
    assert got["CACHE"] == str(cluster / "admin-cache" / "trajectories")


def test_constants_relocate_from_dotenv_in_cwd(tmp_path: Path) -> None:
    """The operator's ask: ``XRLENV_HOME`` is picked up from the checkout's
    ``.env`` (auto-loaded at ``import xrlenv``), no shell export needed."""
    checkout = tmp_path / "xrlenv-dev"
    checkout.mkdir()
    cluster = tmp_path / "dev-state"
    (checkout / ".env").write_text(f"XRLENV_HOME={cluster}\n", encoding="utf-8")
    # Note: XRLENV_DOTENV is NOT off here — we *want* the .env auto-load.
    env = {"HOME": str(tmp_path / "h"), "PATH": "/usr/bin:/bin"}
    Path(env["HOME"]).mkdir()
    got = _probe(env, cwd=str(checkout))
    assert got["HOME"] == str(cluster)
    assert got["STATE_DB"] == str(cluster / "state.db")
    assert got["SECRETS"] == str(cluster / "secrets")


def test_shell_env_wins_over_dotenv(tmp_path: Path) -> None:
    """Precedence: an exported ``XRLENV_HOME`` overrides the ``.env`` value
    (``.env`` is the fallback layer, per the auto-loader contract)."""
    checkout = tmp_path / "xrlenv-dev"
    checkout.mkdir()
    (checkout / ".env").write_text(
        f"XRLENV_HOME={tmp_path / 'from-dotenv'}\n", encoding="utf-8"
    )
    shell_home = tmp_path / "from-shell"
    env = {
        "HOME": str(tmp_path / "h"),
        "PATH": "/usr/bin:/bin",
        "XRLENV_HOME": str(shell_home),
    }
    Path(env["HOME"]).mkdir()
    got = _probe(env, cwd=str(checkout))
    assert got["HOME"] == str(shell_home)
    assert got["STATE_DB"] == str(shell_home / "state.db")
