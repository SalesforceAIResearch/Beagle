"""Issue #18 — ``xrlenv-node`` CLI: XRLENV_PULL_CONCURRENCY override."""

from __future__ import annotations

import pytest
from xrlenv.node.cli import (
    _image_cache_config_from_env,
)


def test_pull_concurrency_unset_uses_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var → ``None`` → ImageCacheManager applies the library
    default (pull_concurrency=2)."""
    monkeypatch.delenv("XRLENV_PULL_CONCURRENCY", raising=False)
    assert _image_cache_config_from_env() is None


def test_pull_concurrency_blank_uses_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XRLENV_PULL_CONCURRENCY", "   ")
    assert _image_cache_config_from_env() is None


def test_pull_concurrency_env_override_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid value builds an ImageCacheConfig carrying it."""
    monkeypatch.setenv("XRLENV_PULL_CONCURRENCY", "6")
    cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.pull_concurrency == 6  # type: ignore[attr-defined]


def test_pull_concurrency_non_integer_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-integer value warns and falls back to the default — a
    typo must not silently no-op without a trace."""
    monkeypatch.setenv("XRLENV_PULL_CONCURRENCY", "lots")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        assert _image_cache_config_from_env() is None
    assert any("XRLENV_PULL_CONCURRENCY" in r.message for r in caplog.records)


def test_pull_concurrency_non_positive_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("XRLENV_PULL_CONCURRENCY", "0")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        assert _image_cache_config_from_env() is None
    assert any("XRLENV_PULL_CONCURRENCY" in r.message for r in caplog.records)
