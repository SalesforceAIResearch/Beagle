"""Shared pytest fixtures for the xrlenv test suite."""

from __future__ import annotations

import os

# Test-isolation: prevent ``xrlenv``'s import-time ``.env`` auto-load
# from polluting test runs with the developer's local config. Without
# this, a ``.env`` at the repo root (typically holds the developer's
# operator/consumer/node tokens, harbor cache path, etc.) would leak
# into ``TokenStore.load()`` and any other os.environ-fallback path —
# tests would pass on a clean checkout and fail after the developer
# created a ``.env`` for their own dev workflow. Tests that
# specifically want to exercise the auto-load flip this back via
# ``monkeypatch.delenv("XRLENV_DOTENV")``.
os.environ.setdefault("XRLENV_DOTENV", "off")

# Same-shape isolation for shell-exported tokens: a developer running
# ``xrlenv up`` in their interactive shell often has
# ``XRLENV_OPERATOR_TOKEN`` etc. exported via the deploy scripts.
# ``TokenStore.load()`` reads those first; tests that construct a
# TokenStore against a tmp_path expect a clean slate. Strip them at
# session start so the suite is reproducible regardless of the
# developer's shell state.
_LEAKABLE_TOKEN_ENVS = (
    "XRLENV_NODE_TOKEN",
    "XRLENV_CONSUMER_TOKEN",
    "XRLENV_OPERATOR_TOKEN",
    "XRLENV_VIEWER_TOKEN",
)
for _token_env in _LEAKABLE_TOKEN_ENVS:
    os.environ.pop(_token_env, None)

from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_xrlenv_token_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Strip any ``XRLENV_*_TOKEN`` env vars from every test's
    starting state. Tests that need a token set use
    ``monkeypatch.setenv`` which restores cleanly. Without this
    autouse fixture, cross-module pollution leaks the operator's
    shell-exported tokens into later test modules (saw in 2026-05-11
    debugging: tokens issued via ``cmd_tokens_issue`` write env-var
    sentinels somewhere that survives test boundaries)."""
    for env in _LEAKABLE_TOKEN_ENVS:
        monkeypatch.delenv(env, raising=False)
    yield


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def package_root(repo_root: Path) -> Path:
    return repo_root / "xrlenv"


@pytest.fixture
def hello_shell_manifest_path(package_root: Path) -> Path:
    return package_root / "templates" / "hello_shell" / "manifest.yaml"


@pytest.fixture
def tmp_template_dir(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
