"""Tests for ``xrlenv.client.dotenv`` (operator-side .env helpers).

Coverage:

- ``parse_dotenv`` accepts the conservative subset (bare, double-
  quoted, single-quoted, ``export`` prefix, comments, blank lines).
- ``parse_dotenv`` silently skips malformed lines (matches the
  forgiving behavior of ``set -a; source .env``).
- ``parse_dotenv`` rejects non-POSIX env-var keys (hyphens etc.) so
  the dict round-trips through ``os.environ`` cleanly.
- ``upload_dotenv`` builds the right tarball + calls put_archive,
  with the pre-mkdir guard for Docker's "target must exist" gotcha.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
from xrlenv.client.dotenv import parse_dotenv, upload_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# parse_dotenv — shapes accepted, shapes skipped
# ──────────────────────────────────────────────────────────────────────────────


def test_parse_dotenv_bare_value(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=bar\n", encoding="utf-8")
    assert parse_dotenv(p) == {"FOO": "bar"}


def test_parse_dotenv_double_quoted(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text('GREETING="hello world"\n', encoding="utf-8")
    assert parse_dotenv(p) == {"GREETING": "hello world"}


def test_parse_dotenv_single_quoted(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("KEY='one two'\n", encoding="utf-8")
    assert parse_dotenv(p) == {"KEY": "one two"}


def test_parse_dotenv_export_prefix(tmp_path: Path) -> None:
    """``export KEY=value`` is a common shell-friendly shape; the
    leading ``export `` is stripped, the rest parsed as normal."""
    p = tmp_path / ".env"
    p.write_text("export OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    assert parse_dotenv(p) == {"OPENAI_API_KEY": "sk-test"}


def test_parse_dotenv_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text(
        "# operator's API keys\n"
        "\n"
        "ANTHROPIC_API_KEY=sk-ant-1\n"
        "  # indented comment\n"
        "OPENAI_API_KEY=sk-oai-1\n",
        encoding="utf-8",
    )
    assert parse_dotenv(p) == {
        "ANTHROPIC_API_KEY": "sk-ant-1",
        "OPENAI_API_KEY": "sk-oai-1",
    }


def test_parse_dotenv_skips_malformed_lines_silently(tmp_path: Path) -> None:
    """Match the forgiving behavior of ``set -a; source .env``.
    Operators wanting strictness can validate the returned dict."""
    p = tmp_path / ".env"
    p.write_text(
        "NOEQUAL\n"
        "GOOD=value\n"
        "DATABASE-URL=postgres://x\n"   # hyphen in key — skipped
        "=value-with-no-key\n"           # missing key — skipped
        "ALSO_GOOD=another\n",
        encoding="utf-8",
    )
    assert parse_dotenv(p) == {"GOOD": "value", "ALSO_GOOD": "another"}


def test_parse_dotenv_does_not_expand_variables(tmp_path: Path) -> None:
    """No ``$OTHER`` expansion: the dict the operator gets is exactly
    what lands in ``acquire_container(environment=...)`` — no
    surprise interpolation."""
    p = tmp_path / ".env"
    p.write_text(
        "BASE=foo\n"
        "DERIVED=$BASE\n",
        encoding="utf-8",
    )
    parsed = parse_dotenv(p)
    assert parsed["BASE"] == "foo"
    assert parsed["DERIVED"] == "$BASE"  # not expanded.


def test_parse_dotenv_missing_file_raises(tmp_path: Path) -> None:
    """Operators must not accidentally proceed with no secrets."""
    with pytest.raises(FileNotFoundError):
        parse_dotenv(tmp_path / "nonexistent.env")


def test_parse_dotenv_handles_empty_value(tmp_path: Path) -> None:
    """``KEY=`` is a legitimate shape for "set to empty string."""
    p = tmp_path / ".env"
    p.write_text("EMPTY=\nFILLED=x\n", encoding="utf-8")
    assert parse_dotenv(p) == {"EMPTY": "", "FILLED": "x"}


# ──────────────────────────────────────────────────────────────────────────────
# upload_dotenv — calls session.exec(mkdir) + session.put_archive(tar)
# ──────────────────────────────────────────────────────────────────────────────


class _FakeSession:
    """Minimal stand-in capturing the exec + put_archive calls
    upload_dotenv makes. Lets us assert the mkdir pre-step fires and
    the tarball lands the file under the right arcname."""

    def __init__(self) -> None:
        self.execs: list[tuple[list[str], dict[str, Any]]] = []
        self.put_archives: list[tuple[str, bytes]] = []

    async def exec(self, cmd: list[str], **kwargs: Any) -> Any:
        self.execs.append((cmd, kwargs))

        class _R:
            exit_code = 0
            stdout = b""
            stderr = b""
        return _R()

    async def put_archive(
        self, *, target_dir: str, tarball: bytes,
    ) -> None:
        self.put_archives.append((target_dir, tarball))


def _tarball_paths(tarball: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r") as tf:
        return tf.getnames()


@pytest.mark.asyncio
async def test_upload_dotenv_runs_mkdir_then_put_archive(
    tmp_path: Path,
) -> None:
    src = tmp_path / ".env"
    src.write_text("FOO=bar\n", encoding="utf-8")
    sess = _FakeSession()
    landed_at = await upload_dotenv(
        sess,  # type: ignore[arg-type]
        source=src,
    )
    assert landed_at == "/workspace/.env"
    # mkdir -p runs FIRST so put_archive can extract into the dir.
    assert sess.execs == [
        (["mkdir", "-p", "/workspace"], {"timeout_s": 30.0}),
    ]
    # Single tarball pushed to /workspace, containing one .env file.
    assert len(sess.put_archives) == 1
    target, tarball = sess.put_archives[0]
    assert target == "/workspace"
    assert _tarball_paths(tarball) == [".env"]


@pytest.mark.asyncio
async def test_upload_dotenv_respects_target_dir_and_arcname(
    tmp_path: Path,
) -> None:
    src = tmp_path / "secrets.env"
    src.write_text("X=y\n", encoding="utf-8")
    sess = _FakeSession()
    landed_at = await upload_dotenv(
        sess,  # type: ignore[arg-type]
        source=src,
        target_dir="/app",
        arcname="agent.env",
    )
    assert landed_at == "/app/agent.env"
    assert sess.execs[0][0] == ["mkdir", "-p", "/app"]
    assert _tarball_paths(sess.put_archives[0][1]) == ["agent.env"]


@pytest.mark.asyncio
async def test_upload_dotenv_skips_mkdir_when_disabled(
    tmp_path: Path,
) -> None:
    """For operators who know the target dir already exists (saves
    one round trip in tight loops)."""
    src = tmp_path / ".env"
    src.write_text("X=1\n", encoding="utf-8")
    sess = _FakeSession()
    await upload_dotenv(
        sess,  # type: ignore[arg-type]
        source=src,
        mkdir=False,
    )
    assert sess.execs == []  # no mkdir.
    assert len(sess.put_archives) == 1


@pytest.mark.asyncio
async def test_upload_dotenv_refuses_missing_source(tmp_path: Path) -> None:
    sess = _FakeSession()
    with pytest.raises(FileNotFoundError, match="is not a file"):
        await upload_dotenv(
            sess,  # type: ignore[arg-type]
            source=tmp_path / "missing.env",
        )
    assert sess.execs == []
    assert sess.put_archives == []


# ──────────────────────────────────────────────────────────────────────────────
# load_dotenv — operator-side "set once, every script picks it up"
# ──────────────────────────────────────────────────────────────────────────────


import os  # noqa: E402

import xrlenv._dotenv_autoload as _dotenv_module  # noqa: E402
from xrlenv._dotenv_autoload import (  # noqa: E402
    _find_dotenv_upward,
    _maybe_auto_load_dotenv,
    load_dotenv,
)


@pytest.fixture
def _reset_auto_loaded() -> Any:
    """Each test starts with the auto-load flag cleared so
    ``load_dotenv()`` doesn't no-op on a previous test's run."""
    saved = _dotenv_module._AUTO_LOADED
    _dotenv_module._AUTO_LOADED = False
    yield
    _dotenv_module._AUTO_LOADED = saved


def test_load_dotenv_applies_keys_to_environ(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_auto_loaded: Any,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "XRLENV_GRPC_HOST=127.0.0.1\n"
        "XRLENV_CONSUMER_TOKEN=tok-abc\n",
        encoding="utf-8",
    )
    # Wipe pre-existing env so this test is deterministic.
    monkeypatch.delenv("XRLENV_GRPC_HOST", raising=False)
    monkeypatch.delenv("XRLENV_CONSUMER_TOKEN", raising=False)

    applied = load_dotenv(path=env_path)
    assert applied == {
        "XRLENV_GRPC_HOST": "127.0.0.1",
        "XRLENV_CONSUMER_TOKEN": "tok-abc",
    }
    assert os.environ["XRLENV_GRPC_HOST"] == "127.0.0.1"
    assert os.environ["XRLENV_CONSUMER_TOKEN"] == "tok-abc"


def test_load_dotenv_does_not_override_existing_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_auto_loaded: Any,
) -> None:
    """Operator's shell-exported value must always win over the
    file — the file is a fallback, not a forced override."""
    env_path = tmp_path / ".env"
    env_path.write_text("XRLENV_GRPC_HOST=from-file\n", encoding="utf-8")
    monkeypatch.setenv("XRLENV_GRPC_HOST", "from-shell")

    applied = load_dotenv(path=env_path)
    # The file's value was NOT applied; shell wins.
    assert "XRLENV_GRPC_HOST" not in applied
    assert os.environ["XRLENV_GRPC_HOST"] == "from-shell"


def test_load_dotenv_override_true_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_auto_loaded: Any,
) -> None:
    """``override=True`` lets the file beat the shell. Useful for
    tests / forced reload; consumer code rarely wants this."""
    env_path = tmp_path / ".env"
    env_path.write_text("KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("KEY", "from-shell")

    applied = load_dotenv(path=env_path, override=True)
    assert applied == {"KEY": "from-file"}
    assert os.environ["KEY"] == "from-file"


def test_load_dotenv_missing_path_returns_empty(
    tmp_path: Path,
    _reset_auto_loaded: Any,
) -> None:
    """An explicit ``path=`` that doesn't exist is a silent no-op
    — doesn't raise, returns empty. Auto-load on a fresh checkout
    without a .env shouldn't blow up."""
    assert load_dotenv(path=tmp_path / "nonexistent.env") == {}


def test_load_dotenv_searches_upward_when_no_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_auto_loaded: Any,
) -> None:
    """Walks from CWD upward until it finds a ``.env``. Mirrors how
    python-dotenv's ``find_dotenv`` behaves so operators with the
    file at the repo root can ``cd`` into a sub-directory and have
    it still picked up."""
    repo = tmp_path / "repo"
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    (repo / ".env").write_text("FOUND_VIA_UPWARD=1\n", encoding="utf-8")

    monkeypatch.chdir(sub)
    monkeypatch.delenv("FOUND_VIA_UPWARD", raising=False)

    applied = load_dotenv()
    assert applied == {"FOUND_VIA_UPWARD": "1"}


def test_find_dotenv_upward_returns_none_when_absent(
    tmp_path: Path,
) -> None:
    assert _find_dotenv_upward(tmp_path) is None


def test_maybe_auto_load_dotenv_respects_off_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_auto_loaded: Any,
) -> None:
    """Setting ``XRLENV_DOTENV=off`` (or false/0/no/disabled) skips
    the auto-load even if a ``.env`` exists in CWD. Useful for tests
    that want strict env isolation."""
    (tmp_path / ".env").write_text(
        "SHOULDNT_BE_LOADED=1\n", encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHOULDNT_BE_LOADED", raising=False)

    for off_value in ("off", "false", "0", "no", "disabled", "OFF"):
        _dotenv_module._AUTO_LOADED = False
        monkeypatch.setenv("XRLENV_DOTENV", off_value)
        _maybe_auto_load_dotenv()
        assert "SHOULDNT_BE_LOADED" not in os.environ, (
            f"XRLENV_DOTENV={off_value!r} should have skipped the load"
        )


def test_maybe_auto_load_dotenv_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _reset_auto_loaded: Any,
) -> None:
    """First call walks the filesystem, subsequent calls no-op via
    the module-level flag. Means ``import xrlenv`` is cheap across
    re-imports (test suites, plugin reloads)."""
    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("X", raising=False)
    monkeypatch.delenv("XRLENV_DOTENV", raising=False)

    _maybe_auto_load_dotenv()
    assert os.environ.get("X") == "1"

    # Simulate someone removing the env var and calling again —
    # idempotency means we don't re-walk + re-apply.
    monkeypatch.delenv("X", raising=False)
    _maybe_auto_load_dotenv()
    assert "X" not in os.environ, (
        "second call should have been a no-op via _AUTO_LOADED flag"
    )
