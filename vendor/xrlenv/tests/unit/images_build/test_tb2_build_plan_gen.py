"""Unit tests for the terminal-bench-2 example build-plan generator (audit M17).

The phase-0 example generator must: (a) reject the RETIRED ``XRLENV_HARBOR_CACHE`` var / path
rather than silently ignoring it, and (b) FAIL LOUD on ``--all`` when its cache is absent/empty
rather than silently emitting the 8-task smoke plan as if it were the full corpus.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv_plugins.images_build.terminal_bench_2 import build_plan_gen as g


def test_discover_all_fails_loud_on_absent_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XRLENV_HARBOR_CACHE", raising=False)
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="cache not found"):
        g._discover_all_tasks()


def test_discover_all_fails_loud_on_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XRLENV_HARBOR_CACHE", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()   # exists but has no task with solution/solve.sh
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(cache))
    with pytest.raises(SystemExit, match="no tasks"):
        g._discover_all_tasks()


def test_discover_all_fails_loud_on_shared_multibenchmark_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # audit M17 (corpus scoping): pointed at the SHARED cache (thousands of tasks across
    # benchmarks), --all must fail loud rather than emit unrelated ids as tb2 images.
    monkeypatch.delenv("XRLENV_HARBOR_CACHE", raising=False)
    cache = tmp_path / "shared"
    for i in range(g._TB2_MAX_PLAUSIBLE_TASKS + 5):
        solve = cache / f"task-{i:04d}" / "solution" / "solve.sh"
        solve.parent.mkdir(parents=True)
        solve.write_text("#!/bin/sh\n")
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(cache))
    with pytest.raises(SystemExit, match="SHARED"):
        g._discover_all_tasks()


def test_discover_all_returns_tasks_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XRLENV_HARBOR_CACHE", raising=False)
    cache = tmp_path / "cache"
    for task in ("fix-git", "build-pov-ray"):
        solve = cache / task / "solution" / "solve.sh"
        solve.parent.mkdir(parents=True)
        solve.write_text("#!/bin/sh\n")
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(cache))
    assert g._discover_all_tasks() == ["build-pov-ray", "fix-git"]


def test_cache_root_rejects_retired_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRLENV_HARBOR_CACHE", "/whatever")
    with pytest.raises(SystemExit, match="retired"):
        g._harbor_cache_root()
