"""Wrapper-level test for terminal_bench_2_1 run_full_sweep.sh catalog-completeness gate
(audit H4/B1). `run_full_sweep.sh --list-green` must reject a partial cache (present != 89)
BEFORE the early return, so a partially-populated shard can't silently shrink the
"requested" set and let a subset sweep pass the integration gate.

Drives the real bash wrapper against a fabricated shard. tb2.1 discovers a task by the
presence of solution/solve.sh, so each fabricated task carries one. We assert the
present-count gate (the green-count gate needs the pinned EXCLUDE ids and is covered by
the lhtb/deep_swe tests that share the mechanism).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRAPPER = (_REPO_ROOT / "xrlenv_plugins" / "benchmarks"
            / "terminal_bench_2_1" / "run_full_sweep.sh")


@pytest.fixture(autouse=True)
def _require_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


def _make_shard(root: Path, n: int) -> Path:
    """Fabricate <root>/terminal-bench-2-1/task-NNNN/solution/solve.sh for n tasks."""
    shard = root / "terminal-bench-2-1"
    for i in range(n):
        d = shard / f"task-{i:04d}" / "solution"
        d.mkdir(parents=True, exist_ok=True)
        (d / "solve.sh").write_text("#!/bin/sh\n", encoding="utf-8")
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
    assert "expected 89 present tasks, got 50" in res.stderr
