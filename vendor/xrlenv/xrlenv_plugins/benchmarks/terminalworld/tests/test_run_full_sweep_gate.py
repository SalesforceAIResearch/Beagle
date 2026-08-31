"""Wrapper-level test for terminalworld run_full_sweep.sh catalog-completeness gate
(audit H4/B1). `run_full_sweep.sh --list-green` must reject a partial cache (present !=
200) BEFORE the early return, so a partially-populated shard can't silently shrink the
"requested" set and let a subset sweep pass the integration gate.

Drives the real bash wrapper against a fabricated shard. terminalworld discovers tasks by
`tw_*` dir name. We assert the present-count gate (the green-count gate needs the pinned
EXCLUDE ids and is covered by the lhtb/deep_swe tests that share the mechanism).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRAPPER = _REPO_ROOT / "xrlenv_plugins" / "benchmarks" / "terminalworld" / "run_full_sweep.sh"


@pytest.fixture(autouse=True)
def _require_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


def _make_shard(root: Path, n: int) -> Path:
    """Fabricate <root>/terminalworld-verified/tw_NNNNNN dirs for n tasks."""
    shard = root / "terminalworld-verified"
    for i in range(n):
        (shard / f"tw_{i:06d}").mkdir(parents=True, exist_ok=True)
    return root


def _list_green(root: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root),
        "XRLENV_BENCHMARK_CACHE": str(root),
    }
    return subprocess.run(
        ["bash", str(_WRAPPER), "--list-green", "--skip-build-cache"],
        env=env, capture_output=True, text=True,
    )


def test_partial_cache_fails(tmp_path: Path) -> None:
    res = _list_green(_make_shard(tmp_path, 50))
    assert res.returncode != 0
    assert "expected 200 present tasks, got 50" in res.stderr
