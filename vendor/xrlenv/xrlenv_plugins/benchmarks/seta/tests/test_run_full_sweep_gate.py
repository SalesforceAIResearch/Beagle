"""Wrapper-level test for seta's run_full_sweep.sh --list-green floor check.

seta's corpus is DYNAMIC BY DESIGN (green set = all present - black_list.txt, no fixed
catalog size), so — unlike deep_swe/lhtb — there is no fixed-count completeness gate.
The only invariant is a nonzero floor: an EMPTY green set is never legitimate and must
FAIL rather than hand a --list-green consumer a "0 tasks -> nothing to run -> green"
no-op. This drives the real bash wrapper against a fabricated shard (no Docker / CP).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SETA_DIR = _REPO_ROOT / "xrlenv_plugins" / "benchmarks" / "seta"
_WRAPPER = _SETA_DIR / "run_full_sweep.sh"
_BLACKLIST_FILE = _SETA_DIR / "black_list.txt"


@pytest.fixture(autouse=True)
def _require_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


def _blacklisted_ids() -> list[str]:
    """The task ids seta's black_list.txt excludes (first token of each non-# line)."""
    ids: list[str] = []
    for line in _BLACKLIST_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split()[0])
    return ids


def _make_shard(root: Path, task_names: list[str]) -> Path:
    shard = root / "seta-env"
    for name in task_names:
        # Real seta tasks carry a solution/ dir (harbor's OracleAgent applies it, and
        # _locate_task_dir hard-fails without one). --list-green lists only tasks that
        # have one — list vs execution parity (audit M5) — so the fixture must too.
        (shard / name / "solution").mkdir(parents=True, exist_ok=True)
        (shard / name / "solution" / "solve.sh").write_text("#!/bin/sh\n")
    return root


def _run_list_green(cache_root: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cache_root),
        "XRLENV_BENCHMARK_CACHE": str(cache_root),
    }
    return subprocess.run(
        ["bash", str(_WRAPPER), "--list-green", "--skip-build-cache"],
        env=env, capture_output=True, text=True,
    )


def test_nonempty_green_set_lists_and_exits_zero(tmp_path: Path) -> None:
    # A present set with at least one non-blacklisted task lists it and exits 0.
    root = _make_shard(tmp_path, ["task-alpha", "task-beta"])
    res = _run_list_green(root)
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    assert set(ids) == {"task-alpha", "task-beta"}


def test_empty_green_set_fails(tmp_path: Path) -> None:
    # Every present dir is a blacklisted id -> green set is empty -> must exit non-zero
    # (an empty listing is never a valid "green" answer).
    black = _blacklisted_ids()
    assert black, "expected seta black_list.txt to list at least one id"
    root = _make_shard(tmp_path, black)
    res = _run_list_green(root)
    assert res.returncode != 0
    assert "green set is EMPTY" in res.stderr


def test_empty_shard_fails(tmp_path: Path) -> None:
    # A present shard that EXISTS but has zero task dirs -> empty green set -> fail via
    # the floor check (not the earlier "shard not found" guard). Create the shard dir
    # explicitly so the wrapper reaches the --list-green floor.
    root = _make_shard(tmp_path, [])
    (root / "seta-env").mkdir(parents=True, exist_ok=True)
    res = _run_list_green(root)
    assert res.returncode != 0
    assert "green set is EMPTY" in res.stderr


def test_task_without_solution_dir_is_not_listed(tmp_path: Path) -> None:
    # list vs execution parity (audit M5): a stray/partial present dir lacking solution/
    # would be sampled then crash run_oracle_sweep's OracleAgent at setup, so --list-green
    # must NOT list it — only runnable (solution-bearing) tasks.
    shard = tmp_path / "seta-env"
    (shard / "has-solution" / "solution").mkdir(parents=True)
    (shard / "has-solution" / "solution" / "solve.sh").write_text("#!/bin/sh\n")
    (shard / "no-solution").mkdir(parents=True)          # present but not runnable
    res = _run_list_green(tmp_path)
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    assert ids == ["has-solution"]
