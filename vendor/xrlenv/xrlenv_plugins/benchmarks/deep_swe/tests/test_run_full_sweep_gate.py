"""Wrapper-level test for deep_swe run_full_sweep.sh catalog-completeness gate (audit
H4/B1). `run_full_sweep.sh --list-green` must reject a partial cache (present != 113)
BEFORE the early return, so a partially-populated shard can't silently shrink the
"requested" set and let a subset sweep pass the integration gate. deep_swe's EXCLUDE is
empty, so green == present == 113.

Drives the real bash wrapper against a fabricated shard (per-task dirs each with a
task.toml — the discovery contract the wrapper's `find` uses); no Docker / CP / build.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRAPPER = _REPO_ROOT / "xrlenv_plugins" / "benchmarks" / "deep_swe" / "run_full_sweep.sh"


@pytest.fixture(autouse=True)
def _require_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


def _make_shard(root: Path, n: int) -> Path:
    """Fabricate <root>/deep-swe/task-NNNN/task.toml for n tasks; return the cache ROOT."""
    shard = root / "deep-swe"
    for i in range(n):
        d = shard / f"task-{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "task.toml").write_text("[environment]\n", encoding="utf-8")
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


def test_complete_catalog_lists_113(tmp_path: Path) -> None:
    res = _list_green(_make_shard(tmp_path, 113))
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    assert len(ids) == 113


def test_partial_cache_fails(tmp_path: Path) -> None:
    # a partial populate (100, not 113) must exit non-zero, NOT emit a smaller green set.
    res = _list_green(_make_shard(tmp_path, 100))
    assert res.returncode != 0
    assert "expected 113 present tasks, got 100" in res.stderr


def test_oversized_cache_fails(tmp_path: Path) -> None:
    res = _list_green(_make_shard(tmp_path, 120))
    assert res.returncode != 0
    assert "expected 113 present tasks, got 120" in res.stderr
