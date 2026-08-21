"""beagle's own .env loader — bucket-1 facts/secrets only, host-env-wins, run-knob nudge."""

from __future__ import annotations

import os

from beagle.dotenv import find_dotenv, load_project_dotenv, parse_dotenv


def test_parse_handles_comments_export_and_quotes() -> None:
    parsed = parse_dotenv(
        "# a comment\n\nexport XRLENV_GRPC_HOST=ip-10-0-0-1\n"
        'GH_TOKEN="ghp_xxx"\nEMPTY=\nBAD LINE NO EQUALS\n'
    )
    assert parsed == {"XRLENV_GRPC_HOST": "ip-10-0-0-1", "GH_TOKEN": "ghp_xxx", "EMPTY": ""}


def test_load_host_env_wins_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XEV_KEEP", "from-host")     # already in the environment
    monkeypatch.delenv("XEV_NEW", raising=False)    # register for teardown cleanup
    env = tmp_path / ".env"
    env.write_text("XEV_KEEP=from-dotenv\nXEV_NEW=set-by-dotenv\n")

    load_project_dotenv(env, verbose=False)

    assert os.environ["XEV_KEEP"] == "from-host"    # host wins — .env does not clobber
    assert os.environ["XEV_NEW"] == "set-by-dotenv"  # but fills what's unset


def test_load_reports_kept_when_host_already_has_vars(tmp_path, monkeypatch, capsys) -> None:
    # The confusing case: the user `source`d .env first, so every var is already in the host env.
    # The loader adds 0 NEW ones but must NOT read as ".env ignored" — it says how many were kept.
    monkeypatch.setenv("XEV_A", "host")
    monkeypatch.setenv("XEV_B", "host")
    env = tmp_path / ".env"
    env.write_text("XEV_A=x\nXEV_B=y\n")

    load_project_dotenv(env, verbose=True)

    out = capsys.readouterr().out
    assert "loaded 0 env var(s)" in out and "2 already set in host env, kept" in out


def test_load_override_true_clobbers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XEV_KEEP", "from-host")
    env = tmp_path / ".env"
    env.write_text("XEV_KEEP=from-dotenv\n")
    load_project_dotenv(env, override=True, verbose=False)
    assert os.environ["XEV_KEEP"] == "from-dotenv"


def test_load_missing_file_is_noop(tmp_path) -> None:
    assert load_project_dotenv(tmp_path / "nope.env", verbose=False) is None


def test_find_dotenv_walks_up_to_repo_root(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").write_text("X=1\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert find_dotenv(sub) == tmp_path / ".env"


def test_find_dotenv_does_not_cross_above_git(tmp_path) -> None:
    (tmp_path / ".env").write_text("X=1\n")          # .env ABOVE the repo root
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    assert find_dotenv(repo / "x") is None           # not picked up — stops at .git


def test_warns_when_run_knobs_are_in_dotenv(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ATELIER_GATE_ENABLED", raising=False)
    monkeypatch.delenv("XRLENV_GRPC_HOST", raising=False)
    env = tmp_path / ".env"
    env.write_text("XRLENV_GRPC_HOST=h\nATELIER_GATE_ENABLED=1\n")

    load_project_dotenv(env, verbose=True)

    out = capsys.readouterr().out
    assert "ATELIER_GATE_ENABLED" in out and "belong in config" in out
    assert os.environ["ATELIER_GATE_ENABLED"] == "1"   # still loaded (nothing breaks)


def test_cli_loads_dotenv_before_dispatch(monkeypatch) -> None:
    from beagle import cli

    seen: dict = {}
    monkeypatch.setattr("beagle.dotenv.load_project_dotenv",
                        lambda path=None, **k: seen.setdefault("path", path))
    monkeypatch.setattr("beagle.cli.evaluate._cmd_evaluate", lambda args: 0)

    assert cli.main(["evaluate", "--config", "cfg.yaml", "--env-file", "/tmp/x/.env"]) == 0
    assert seen["path"] == "/tmp/x/.env"
