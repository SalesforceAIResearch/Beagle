"""Wrapper-level tests for the LHTB run_full_sweep.sh catalog-completeness gate.

The B1 gate: `run_full_sweep.sh --list-green` must validate the pinned catalog size
(46 present / 43 default-mode run set / 37 upstream-mode run set — from STATUS.md)
BEFORE the --list-green early return, so a partially-populated cache can't silently
shrink the "requested" set and let a subset sweep pass the integration gate.

These drive the real bash wrapper with a *fabricated* cache shard (a temp dir of
per-task subdirs, each with a task.toml — exactly the discovery contract the wrapper's
`find` uses), so no Docker / control-plane / build_cache is needed. We always pass
--skip-build-cache (so the registry-dependent build block is skipped and the gate is
reached) and --list-green (so the wrapper exits right after the gate).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRAPPER = _REPO_ROOT / "xrlenv_plugins" / "benchmarks" / "lhtb" / "run_full_sweep.sh"

# The 3 BLACKLIST ids the wrapper drops in EVERY mode (kept in sync with the wrapper /
# STATUS.md). Including these by-name in the fabricated catalog makes the default-mode
# run set land at present-3; any other 43 fillers are GREEN/TBD/REBUILD-agnostic (all
# RUN in §2 mode), so 46 present - 3 blacklist == 43 as the contract pins.
_BLACKLIST = ["super-mario", "sudoku-recovery", "apex-openroad-ibex-signoff"]


def _make_shard(root: Path, task_names: list[str]) -> Path:
    """Fabricate <root>/lhtb/<task>/task.toml for each name; return the cache ROOT."""
    shard = root / "lhtb"
    for name in task_names:
        d = shard / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "task.toml").write_text("[environment]\n", encoding="utf-8")
    return root


def _catalog(n: int) -> list[str]:
    """A catalog of n task ids: the 3 BLACKLIST names + (n-3) unique fillers."""
    fillers = [f"filler-task-{i:03d}" for i in range(n - len(_BLACKLIST))]
    return _BLACKLIST + fillers


def _run_list_green(cache_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Invoke the wrapper's --list-green path against a fabricated cache."""
    env = {
        "PATH": __import__("os").environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cache_root),  # keep any HOME-derived defaults inside the sandbox
        "XRLENV_BENCHMARK_CACHE": str(cache_root),
        # §2 (default) mode needs a private registry set, but only in the build block —
        # which --skip-build-cache skips. Set it anyway so nothing upstream of the gate
        # bails for an unrelated reason.
        "XRLENV_PRIVATE_REGISTRY_HOST": "registry.invalid:5000",
    }
    return subprocess.run(
        ["bash", str(_WRAPPER), "--list-green", "--skip-build-cache", *extra],
        env=env, capture_output=True, text=True,
    )


@pytest.fixture(autouse=True)
def _require_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


# ── the happy path: a complete 46-task catalog lists the 43-task default run set ──


def test_complete_catalog_lists_default_run_set(tmp_path: Path) -> None:
    root = _make_shard(tmp_path, _catalog(46))
    res = _run_list_green(root)
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    # default §2 mode drops only the 3 BLACKLIST -> 43 tasks, none of them blacklisted.
    assert len(ids) == 43
    assert not (set(ids) & set(_BLACKLIST))


def test_complete_catalog_upstream_run_set(tmp_path: Path) -> None:
    # In --use-upstream-image mode the 6 REBUILD tasks also drop. Include the 6 REBUILD
    # ids by name so the count lands at the pinned 37 (46 - 3 blacklist - 6 rebuild).
    rebuild = [
        "chess-mate", "duckdb-optimizer-closure", "robotics-slam-benchmark-repair",
        "unknown-config-semantics", "climate-netcdf-extreme-event-audit",
        "materials-phase-diagram-audit",
    ]
    fillers = [f"filler-task-{i:03d}" for i in range(46 - len(_BLACKLIST) - len(rebuild))]
    root = _make_shard(tmp_path, _BLACKLIST + rebuild + fillers)
    res = _run_list_green(root, "--use-upstream-image")
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    assert len(ids) == 37


# ── the B1 gate: a truncated / oversized catalog must FAIL, not shrink the set ──


def test_truncated_present_count_fails(tmp_path: Path) -> None:
    # A partial populate (40 present, not 46) must exit non-zero — NOT emit a smaller
    # green set that the integration gate would then accept as "all green".
    root = _make_shard(tmp_path, _catalog(40))
    res = _run_list_green(root)
    assert res.returncode != 0
    assert "expected 46 present tasks, got 40" in res.stderr


def test_oversized_present_count_fails(tmp_path: Path) -> None:
    root = _make_shard(tmp_path, _catalog(47))
    res = _run_list_green(root)
    assert res.returncode != 0
    assert "expected 46 present tasks, got 47" in res.stderr


def test_run_set_drift_fails(tmp_path: Path) -> None:
    # Present count is right (46) but the run set is wrong: an EXTRA blacklist-shaped
    # exclusion would shrink RUN below 43. Simulate by omitting a blacklist name (so
    # only 2 of 3 drop -> RUN=44) — a run-set drift the gate must catch even though
    # present==46.
    names = ["super-mario", "sudoku-recovery"] + [f"filler-{i:03d}" for i in range(44)]
    assert len(names) == 46
    root = _make_shard(tmp_path, names)
    res = _run_list_green(root)
    assert res.returncode != 0
    assert "run set" in res.stderr and "got 44" in res.stderr


def test_skip_ultra_long_relaxes_run_set_gate(tmp_path: Path) -> None:
    # --skip-ultra-long legitimately drops the ULTRA_LONG tasks, so the run-set count
    # gate is intentionally NOT enforced in that mode; only the present==46 gate is.
    # Include both ULTRA_LONG ids so they actually drop (RUN=41 != 43), and confirm the
    # wrapper still exits 0 (present is complete; the smaller set is by-design).
    ultra = ["unknown-config-semantics", "nbody-accel-iterative"]
    fillers = [f"filler-{i:03d}" for i in range(46 - len(_BLACKLIST) - len(ultra))]
    root = _make_shard(tmp_path, _BLACKLIST + ultra + fillers)
    res = _run_list_green(root, "--skip-ultra-long")
    assert res.returncode == 0, res.stderr
    ids = [ln for ln in res.stdout.splitlines() if ln and not ln.startswith("==")]
    assert len(ids) == 41   # 43 default - 2 ultra-long, allowed only because of the flag
