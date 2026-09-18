"""The in-container git bootstrap, and its retired-suite fallback.

Task images are minimal and usually lack git, so this script runs for most trials. It is shell
executed as root inside someone else's image, so the properties worth pinning are the ones a
reader can't check by eye: that the fallback only fires after the normal path fails, that it
leaves the container's apt config alone, and that it derives its downgrade target instead of
hardcoding one.

Verified against real images on a cluster node during development: the two bullseye
terminal-bench images (qemu-alpine-ssh, qemu-startup) install git 2.30.2 through the fallback
with /etc/apt/sources.list unchanged, while the bookworm and Ubuntu images take the normal path
and never reach it.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from beagle.benchmarks.harness._common import _GIT_BOOTSTRAP


def test_bootstrap_is_valid_shell() -> None:
    """It is executed by `sh -c` inside an arbitrary image; a syntax error would fail every
    trial that needs git, before the agent ever runs."""
    sh = shutil.which("bash") or shutil.which("sh")
    assert sh, "no shell available to syntax-check with"
    # check=False: a non-zero exit IS the assertion here, not an error to raise on.
    proc = subprocess.run([sh, "-n"], input=_GIT_BOOTSTRAP, text=True,
                          capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_fallback_runs_only_after_the_normal_path_fails() -> None:
    """The normal apt line ends in `&& break`, so the fallback is unreachable unless it failed.
    A bookworm/Ubuntu image must never pay for a bullseye-specific workaround."""
    normal = "apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates && break"
    assert normal in _GIT_BOOTSTRAP
    assert _GIT_BOOTSTRAP.index(normal) < _GIT_BOOTSTRAP.index("--allow-downgrades")


def test_fallback_does_not_mutate_the_container_apt_config() -> None:
    """Redirect apt at temp files rather than rewriting /etc/apt: an agent that shells out to
    apt later must still see its image's real sources. Writing sources.list would silently
    strip the security suite for the rest of the trial."""
    assert "Dir::Etc::SourceList=" in _GIT_BOOTSTRAP
    assert "Dir::State::Lists=" in _GIT_BOOTSTRAP
    assert not re.search(r">\s*/etc/apt/sources\.list", _GIT_BOOTSTRAP)
    assert "rm -f /etc/apt/sources.list.d" not in _GIT_BOOTSTRAP


def test_downgrade_target_is_derived_not_hardcoded() -> None:
    """The affected images ship perl-base deb11u4 while a bare bullseye-slim ships deb11u5, so
    any pinned constant is wrong somewhere. Read it from the container instead."""
    assert "apt-cache" in _GIT_BOOTSTRAP and "madison perl-base" in _GIT_BOOTSTRAP
    assert not re.search(r"perl-base=5\.\d", _GIT_BOOTSTRAP)   # no baked version


def _code_lines() -> str:
    """The script minus comments — what actually executes. The comments SHOULD name bullseye
    (that's the incident being explained); the logic should not."""
    return "\n".join(ln for ln in _GIT_BOOTSTRAP.splitlines()
                     if not ln.lstrip().startswith("#"))


def test_codename_is_read_from_the_container() -> None:
    """`bullseye` must not be hardcoded in the LOGIC — the same fallback should do the right
    thing on any Debian-family image whose suite gets retired next."""
    assert "VERSION_CODENAME" in _GIT_BOOTSTRAP
    assert "bullseye" not in _code_lines()


def test_fallback_is_documented_as_droppable() -> None:
    """It exists for a live Debian archive inconsistency. When that clears, it should go --
    so the removal condition is recorded rather than left for someone to reverse-engineer."""
    assert "archive.debian.org" in _GIT_BOOTSTRAP
    assert "DROP THIS" in _GIT_BOOTSTRAP


@pytest.mark.parametrize("mgr", ["apk", "microdnf", "dnf", "yum", "zypper", "pacman"])
def test_non_apt_managers_are_untouched(mgr: str) -> None:
    """The fallback is apt-specific; every other package manager keeps its single-line path."""
    assert mgr in _GIT_BOOTSTRAP


def test_bootstrap_is_skipped_entirely_when_git_exists() -> None:
    """Most of the corpus already has git; the whole block must stay behind that guard."""
    assert _GIT_BOOTSTRAP.index("if ! command -v git") < _GIT_BOOTSTRAP.index("apt-get")
