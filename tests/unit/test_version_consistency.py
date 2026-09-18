"""The project version is declared in three places; they must agree.

Tracking changes by version only works if "bump the version" is one reliable action. It isn't
by default: the number lives in pyproject.toml, in the CHANGELOG's newest released heading, and
in a README badge. Any of the three can be updated alone, and nothing else notices -- a stale
badge or a changelog heading that doesn't match the shipped package is the kind of thing found
by a reader, not by CI.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
#: Docs write the version as ``v0.0.2``; ``pyproject.toml`` cannot -- PEP 440 has no ``v``
#: prefix, and packaging tools would reject it. So the prefix is optional here and stripped
#: before comparing, rather than each place being expected to match character for character.
_SEMVER = r"v?\d+\.\d+\.\d+"


def _norm(v: str) -> str:
    return v.lstrip("vV")


def project_version() -> str:
    raw = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert not raw.lower().startswith("v"), "pyproject version must be PEP 440 (no 'v' prefix)"
    return _norm(raw)


def changelog_version() -> str:
    """The newest RELEASED heading — `## [x.y.z]`. `## [Unreleased]` is skipped by the pattern,
    so work-in-progress can sit above the release without tripping this."""
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        m = re.match(rf"##\s*\[({_SEMVER})\]", line.strip())
        if m:
            return _norm(m.group(1))
    pytest.fail("CHANGELOG.md has no released `## [x.y.z]` heading")


def readme_badge_version() -> str:
    m = re.search(rf"badge/status-({_SEMVER})-", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert m, "README status badge no longer carries a version"
    return _norm(m.group(1))


def test_changelog_matches_the_packaged_version() -> None:
    """A released changelog heading that doesn't match the package it describes makes the
    changelog untrustworthy — the one thing it cannot afford to be."""
    assert changelog_version() == project_version()


def test_readme_badge_matches_the_packaged_version() -> None:
    """The badge is the version most people see first, and the easiest to forget."""
    assert readme_badge_version() == project_version()


def test_docs_use_the_v_prefix() -> None:
    """Docs display `vX.Y.Z`; the package metadata must not. Both are checked so the convention
    is enforced rather than left to habit."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"##\s*\[v\d+\.\d+\.\d+\]", text), "changelog release heading needs a 'v' prefix"
    assert "badge/status-v" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_links_the_changelog() -> None:
    """A changelog nobody can find is only half-written."""
    assert "(CHANGELOG.md)" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_changelog_keeps_an_unreleased_section() -> None:
    """So the next change has somewhere to go that isn't a released heading."""
    assert "## [Unreleased]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
