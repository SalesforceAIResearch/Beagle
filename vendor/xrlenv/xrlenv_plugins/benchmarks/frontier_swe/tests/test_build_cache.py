"""Unit tests for the FrontierSWE cache builder's pure/filesystem logic.

Network-free: the ``populate`` git clone is not exercised here (covered by running
``--stage populate`` live); these cover ``normalize_task_toml_text``, the idempotent
``_copy_tasks``, the ``patches/`` overlay, and the oracle-gateable count.
"""
from __future__ import annotations

from pathlib import Path

from xrlenv_plugins.benchmarks.frontier_swe.build_cache import (
    _apply_patch,
    _copy_tasks,
    _count_gateable,
    _count_tasks,
    apply_all_patches,
    is_populated,
    normalize_task_toml_text,
)

# ── normalize_task_toml_text ──────────────────────────────────────────────────


def test_normalize_strips_deprecated_when_canonical_present() -> None:
    text = (
        "[environment]\n"
        'docker_image = "ghcr.io/x/img:v4"\n'
        'memory = "8G"\n'
        "memory_mb = 8192\n"
        'storage = "20G"\n'
        "storage_mb = 20480\n"
    )
    out, changed = normalize_task_toml_text(text)
    assert changed
    assert "memory =" not in out
    assert "storage =" not in out
    assert "memory_mb = 8192" in out
    assert "storage_mb = 20480" in out
    assert 'docker_image = "ghcr.io/x/img:v4"' in out  # untouched


def test_normalize_noop_when_only_canonical() -> None:
    # This is the FrontierSWE case: tasks use _mb only, so the normalizer is a no-op.
    text = (
        "[environment]\n"
        'docker_image = "ghcr.io/proximal-labs/frontier-swe/ffmpeg-swscale-rewrite:v4"\n'
        "cpus = 8\n"
        "memory_mb = 65536\n"
        "storage_mb = 20480\n"
    )
    out, changed = normalize_task_toml_text(text)
    assert not changed
    assert out == text


def test_normalize_noop_when_only_deprecated() -> None:
    # No conflict (harbor migrates a lone ``memory`` into ``memory_mb``), so we leave
    # it — dropping it would be lossy.
    text = '[environment]\nmemory = "8G"\n'
    out, changed = normalize_task_toml_text(text)
    assert not changed
    assert out == text


# ── _copy_tasks (idempotent) + normalize-on-copy + gateable count ─────────────


def _make_src_task(
    root: Path, name: str, task_toml: str, *, with_solve: bool = True
) -> Path:
    d = root / name
    (d / "environment").mkdir(parents=True)
    (d / "task.toml").write_text(task_toml)
    (d / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (d / "tests").mkdir()
    (d / "tests" / "test.sh").write_text("#!/bin/sh\n")
    if with_solve:
        (d / "solution").mkdir()
        (d / "solution" / "solve.sh").write_text("#!/bin/sh\n")
    return d


def test_copy_tasks_copies_normalizes_and_is_idempotent(tmp_path: Path) -> None:
    tasks_root = tmp_path / "repo" / "tasks"
    tasks_root.mkdir(parents=True)
    _make_src_task(
        tasks_root, "t1", '[environment]\nmemory = "8G"\nmemory_mb = 8192\n',
    )
    _make_src_task(tasks_root, "t2", "[environment]\nmemory_mb = 4096\n")
    # a solution-withheld task (no solve.sh) — still copied, just not gateable
    _make_src_task(
        tasks_root, "withheld", "[environment]\nmemory_mb = 4096\n", with_solve=False,
    )
    # a stray dir without task.toml must be ignored
    (tasks_root / "not-a-task").mkdir()

    shard = tmp_path / "cache" / "frontier-swe"
    moved, normalized = _copy_tasks(tasks_root, shard)
    assert moved == 3
    assert normalized == 1  # only t1 had the conflict
    assert _count_tasks(shard) == 3
    assert _count_gateable(shard) == 2  # t1, t2 ship solve.sh; withheld does not
    assert "memory =" not in (shard / "t1" / "task.toml").read_text()

    # second run: everything already present -> copies nothing
    moved2, normalized2 = _copy_tasks(tasks_root, shard)
    assert (moved2, normalized2) == (0, 0)
    assert _count_tasks(shard) == 3


# ── patches overlay ───────────────────────────────────────────────────────────


def test_apply_patch_overrides_and_adds_files(tmp_path: Path) -> None:
    task = tmp_path / "task"
    (task / "solution").mkdir(parents=True)
    (task / "solution" / "solve.sh").write_text("original\n")

    patch = tmp_path / "patch"
    (patch / "solution").mkdir(parents=True)
    (patch / "solution" / "solve.sh").write_text("patched\n")  # override
    (patch / "extra.txt").write_text("added\n")  # new file
    (patch / "README.md").write_text("ignored\n")  # skipped

    overridden = _apply_patch(task, patch)
    assert set(overridden) == {"solution/solve.sh", "extra.txt"}
    assert (task / "solution" / "solve.sh").read_text() == "patched\n"
    assert (task / "extra.txt").read_text() == "added\n"
    assert not (task / "README.md").exists()


def test_apply_all_patches_skips_absent_task(tmp_path: Path, monkeypatch) -> None:
    import xrlenv_plugins.benchmarks.frontier_swe.build_cache as bc

    shard = tmp_path / "shard"
    (shard / "present").mkdir(parents=True)
    (shard / "present" / "task.toml").write_text("[environment]\n")

    patches = tmp_path / "patches"
    (patches / "present").mkdir(parents=True)
    (patches / "present" / "note.txt").write_text("x\n")
    (patches / "absent").mkdir(parents=True)
    (patches / "absent" / "note.txt").write_text("y\n")

    monkeypatch.setattr(bc, "PATCHES_DIR", patches)
    assert apply_all_patches(shard) == 1
    assert (shard / "present" / "note.txt").exists()
    assert not (shard / "absent").exists()


def test_dependent_type_checker_patch_is_present_and_selfcontained() -> None:
    """Guard the curated dependent-type-checker overlay (frontier-swe): the oracle
    must read its reference from the bundled solution/reference_impl sibling, NOT
    from /tests (absent during the solve phase). Regression guard for the 2026-08-06
    fix — see patches/README.md."""
    import xrlenv_plugins.benchmarks.frontier_swe.build_cache as bc

    pdir = bc.PATCHES_DIR / "dependent-type-checker" / "solution"
    solve = pdir / "solve.sh"
    assert solve.is_file(), "patched solve.sh overlay missing"
    assert (pdir / "reference_impl" / "Cargo.toml").is_file()
    assert (pdir / "reference_impl" / "src" / "main.rs").is_file()

    # Inspect the ACTIVE code only (strip comment lines — the header comment
    # intentionally mentions /tests to explain the old bug it fixes).
    code = "\n".join(
        ln for ln in solve.read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "reference_impl" in code  # reads the bundled sibling under solution/
    assert "/tests" not in code and "../tests" not in code  # never the verify-only dir
    assert "cargo build" in code


def test_is_populated_and_counts(tmp_path: Path) -> None:
    shard = tmp_path / "frontier-swe"
    assert not is_populated(shard)
    (shard / "t1").mkdir(parents=True)
    (shard / "t1" / "task.toml").write_text("[environment]\n")
    assert is_populated(shard)
    assert _count_tasks(shard) == 1
    assert _count_gateable(shard) == 0  # no solve.sh yet
    (shard / "t1" / "solution").mkdir()
    (shard / "t1" / "solution" / "solve.sh").write_text("#!/bin/sh\n")
    assert _count_gateable(shard) == 1
