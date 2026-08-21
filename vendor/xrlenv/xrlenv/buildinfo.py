"""Build identity for the running xrlenv process (issue #18, Ask #2).

A node-agent and the control plane should run the same xrlenv build.
When they drift — typically a node-agent left running on an old binary
after the control plane was updated — the symptoms are opaque: a
behaviour fix that landed weeks ago "doesn't work" because the node
never picked it up. The SWE-bench-Pro run that motivated this hit
exactly that: nodes still on the pre-``DOCKER_CLIENT_HTTP_TIMEOUT_S``
binary surfaced 60 s ``ReadTimeout`` errors the control-plane operator
had no way to attribute to a stale node.

This module produces a single identity string the node reports in
``NodeHello`` and the control plane logs + compares against its own.

``__version__`` alone is insufficient — the project version
(``xrlenv/_version.py``) is static across many commits — so the
identity also carries a build SHA. The SHA is sourced, in order:

1. ``XRLENV_BUILD_SHA`` environment variable. The deploy scripts
   (``deploy/bootstrap-common.sh`` → ``install_systemd_unit``) capture
   ``git rev-parse HEAD`` at install time and write it into the
   node-agent's systemd ``EnvironmentFile`` (``/etc/xrlenv/node.env``).
   This is the authoritative source on a deployed node: it reflects
   the *installed* binary, not whatever the source checkout happens to
   be at now.
2. ``git rev-parse`` against the package's own source tree — covers
   dev / editable installs and the control plane run straight from a
   checkout.
3. ``"unknown"`` when neither is available.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

from xrlenv._version import __version__

__all__ = ["__version__", "agent_identity", "build_sha"]


@lru_cache(maxsize=1)
def build_sha() -> str:
    """Best-effort short build SHA for the running process.

    Cached — the answer can't change within a process lifetime.
    """
    env = os.environ.get("XRLENV_BUILD_SHA")
    if env and env.strip():
        return env.strip()
    # Dev / editable fallback: ask git about the package's source tree.
    # Bounded + fully swallowed so a missing git binary, a non-repo
    # install, or a slow filesystem can never wedge process startup.
    try:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def agent_identity() -> str:
    """Identity string reported in ``NodeHello`` / logged at connect.

    Shape: ``"{version}+{sha}"`` — e.g. ``"0.0.1+a1b2c3d4e5f6"`` or
    ``"0.0.1+unknown"``. Stable for the process lifetime.
    """
    return f"{__version__}+{build_sha()}"
