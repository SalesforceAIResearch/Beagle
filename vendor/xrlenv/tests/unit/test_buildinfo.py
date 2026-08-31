"""Issue #18 (Ask #2) — xrlenv.buildinfo build-identity sourcing."""

from __future__ import annotations

import pytest
from xrlenv import buildinfo
from xrlenv._version import __version__


def test_build_sha_prefers_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """``XRLENV_BUILD_SHA`` (set by the deploy scripts at install
    time) is the authoritative source — it reflects the *installed*
    binary, so it wins over any git fallback."""
    buildinfo.build_sha.cache_clear()
    monkeypatch.setenv("XRLENV_BUILD_SHA", "deadbeef1234")
    assert buildinfo.build_sha() == "deadbeef1234"
    buildinfo.build_sha.cache_clear()


def test_build_sha_strips_env_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buildinfo.build_sha.cache_clear()
    monkeypatch.setenv("XRLENV_BUILD_SHA", "  abc123  \n")
    assert buildinfo.build_sha() == "abc123"
    buildinfo.build_sha.cache_clear()


def test_build_sha_falls_back_when_env_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no env var, ``build_sha`` falls back to a git lookup
    against the package source tree (this repo is a checkout, so it
    resolves) or ``"unknown"`` — never raises, never empty."""
    buildinfo.build_sha.cache_clear()
    monkeypatch.delenv("XRLENV_BUILD_SHA", raising=False)
    sha = buildinfo.build_sha()
    assert isinstance(sha, str) and sha
    buildinfo.build_sha.cache_clear()


def test_agent_identity_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """``agent_identity()`` is ``"{version}+{sha}"`` — the string the
    node reports in NodeHello and the control plane compares."""
    buildinfo.build_sha.cache_clear()
    monkeypatch.setenv("XRLENV_BUILD_SHA", "cafe9999")
    assert buildinfo.agent_identity() == f"{__version__}+cafe9999"
    buildinfo.build_sha.cache_clear()
