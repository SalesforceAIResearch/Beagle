"""Unit tests for the project .env / .env_private loader."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env_loader


def test_loads_env_and_env_private(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# comment\nexport XRLENV_GRPC_HOST=internal-ip\nXRLENV_GRPC_PORT=50051\n"
    )
    (tmp_path / ".env_private").write_text('EVOCLAW_DATA_ROOT="/data/EvoClaw-data"\n')
    for k in ("XRLENV_GRPC_HOST", "XRLENV_GRPC_PORT", "EVOCLAW_DATA_ROOT"):
        monkeypatch.delenv(k, raising=False)

    root = env_loader.load_project_dotenv(tmp_path)
    assert root == tmp_path.resolve()
    import os

    assert os.environ["XRLENV_GRPC_HOST"] == "internal-ip"
    assert os.environ["XRLENV_GRPC_PORT"] == "50051"
    assert os.environ["EVOCLAW_DATA_ROOT"] == "/data/EvoClaw-data"  # quotes stripped


def test_env_private_overrides_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("EVOCLAW_DATA_ROOT=/from-env\n")
    (tmp_path / ".env_private").write_text("EVOCLAW_DATA_ROOT=/from-private\n")
    monkeypatch.delenv("EVOCLAW_DATA_ROOT", raising=False)
    env_loader.load_project_dotenv(tmp_path)
    import os

    assert os.environ["EVOCLAW_DATA_ROOT"] == "/from-private"  # .env_private wins


def test_existing_shell_var_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("XRLENV_GRPC_HOST=from-dotenv\n")
    monkeypatch.setenv("XRLENV_GRPC_HOST", "from-shell")
    env_loader.load_project_dotenv(tmp_path)
    import os

    assert os.environ["XRLENV_GRPC_HOST"] == "from-shell"  # setdefault: shell wins


def test_walks_up_to_project_root(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("EVOCLAW_DATA_ROOT=/d\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.delenv("EVOCLAW_DATA_ROOT", raising=False)
    root = env_loader.load_project_dotenv(sub)
    assert root == tmp_path.resolve()


def test_returns_none_when_absent(tmp_path):
    assert env_loader.load_project_dotenv(tmp_path) is None


def test_warns_when_shell_shadows_file(tmp_path, monkeypatch, capsys):
    # the exact footgun: shell has the committed .env placeholder, .env_private
    # has the real value -> shell wins (kept) but we warn loudly with the fix.
    (tmp_path / ".env").write_text("EVOCLAW_DATA_ROOT=/path/to/EvoClaw-data\n")
    (tmp_path / ".env_private").write_text("EVOCLAW_DATA_ROOT=/real/EvoClaw-data\n")
    monkeypatch.setenv("EVOCLAW_DATA_ROOT", "/path/to/EvoClaw-data")
    env_loader.load_project_dotenv(tmp_path)
    import os

    assert os.environ["EVOCLAW_DATA_ROOT"] == "/path/to/EvoClaw-data"  # shell still wins
    err = capsys.readouterr().err
    assert "EVOCLAW_DATA_ROOT" in err and "unset EVOCLAW_DATA_ROOT" in err
    assert "/real/EvoClaw-data" in err  # names the file value it's shadowing


def test_no_warning_when_shell_matches(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env_private").write_text("EVOCLAW_DATA_ROOT=/same\n")
    monkeypatch.setenv("EVOCLAW_DATA_ROOT", "/same")
    env_loader.load_project_dotenv(tmp_path)
    assert "WARNING" not in capsys.readouterr().err  # identical value: no noise
