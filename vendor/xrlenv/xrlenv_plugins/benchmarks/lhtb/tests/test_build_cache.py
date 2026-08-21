"""Unit tests for the LHTB cache builder's pure/filesystem logic (network-free)."""
from __future__ import annotations

from pathlib import Path

from xrlenv_plugins.benchmarks.lhtb.build_cache import (
    PRIVATE_REGISTRY_PLACEHOLDER,
    REBUILD_TASKS,
    _all_should_repin,
    _apply_patch,
    _copy_tasks,
    _count_tasks,
    apply_all_patches,
    fix_nproc_scaling_oracles,
    is_populated,
    normalize_task_toml_text,
    repin_docker_image_text,
    repin_to_private_registry,
)


def test_all_should_repin_default() -> None:
    """`--stage all` repins to the host-agnostic placeholder by default (no --registry
    needed — the host resolves from .env at run time)."""
    assert _all_should_repin(use_upstream=False) is True


def test_all_should_repin_use_upstream_skips() -> None:
    """`--use-upstream-image` keeps the docker.io refs (no repin) for the out-of-box gate."""
    assert _all_should_repin(use_upstream=True) is False


def test_verifier_timeout_raised(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import _verifier_timeout_raised

    p = tmp_path / "task.toml"
    p.write_text("[verifier]\ntimeout_sec = 2400\n")
    assert _verifier_timeout_raised(p, 2400.0)
    p.write_text("[verifier]\ntimeout_sec = 600\n")
    assert not _verifier_timeout_raised(p, 2400.0)
    assert not _verifier_timeout_raised(tmp_path / "absent.toml", 2400.0)


def test_fix_status_classifies_repinned_vs_upstream(tmp_path: Path) -> None:
    """The full-state report classifies a REBUILD task by its docker_image: a
    private-registry ref (host:port/…) is 'repinned', a docker.io ref is 'upstream'."""
    from xrlenv_plugins.benchmarks.lhtb.build_cache import _fix_status_lines

    shard = tmp_path / "lhtb"
    (shard / "chess-mate").mkdir(parents=True)
    (shard / "chess-mate" / "task.toml").write_text(
        '[environment]\ndocker_image = "node-host:5011/lhtb/chess-mate:main"\n',
    )
    (shard / "duckdb-optimizer-closure").mkdir(parents=True)
    (shard / "duckdb-optimizer-closure" / "task.toml").write_text(
        '[environment]\ndocker_image = "zli12321/lhtb-duckdb:20260709"\n',
    )
    body = "\n".join(_fix_status_lines(shard))
    assert "REBUILD repinned : 1" in body and "chess-mate" in body
    assert "REBUILD upstream : 1" in body and "duckdb-optimizer-closure" in body


def test_normalize_strips_deprecated_when_canonical_present() -> None:
    text = (
        "[environment]\n"
        'docker_image = "zli12321/lhtb-2048:x"\n'
        'memory = "4G"\nmemory_mb = 4096\n'
        'storage = "10G"\nstorage_mb = 10240\n'
    )
    out, changed = normalize_task_toml_text(text)
    assert changed
    assert "memory =" not in out and "storage =" not in out
    assert "memory_mb = 4096" in out and "storage_mb = 10240" in out
    assert 'docker_image = "zli12321/lhtb-2048:x"' in out


def test_normalize_noop_when_only_canonical() -> None:
    text = "[environment]\nmemory_mb = 4096\nstorage_mb = 10240\n"
    out, changed = normalize_task_toml_text(text)
    assert not changed and out == text


def test_normalize_noop_when_only_deprecated() -> None:
    text = '[environment]\nmemory = "4G"\n'
    out, changed = normalize_task_toml_text(text)
    assert not changed and out == text


def _mk(root: Path, name: str, toml: str) -> None:
    d = root / name
    (d / "environment").mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    (d / "environment" / "Dockerfile").write_text("FROM scratch\n")


def test_copy_tasks_idempotent_and_normalizes(tmp_path: Path) -> None:
    tasks = tmp_path / "repo" / "tasks"
    tasks.mkdir(parents=True)
    _mk(tasks, "2048", '[environment]\nmemory = "4G"\nmemory_mb = 4096\n')
    _mk(tasks, "sokoban", "[environment]\nmemory_mb = 4096\n")
    (tasks / "README.md").write_text("x")  # not a task dir
    shard = tmp_path / "cache" / "lhtb"
    moved, normalized = _copy_tasks(tasks, shard)
    assert (moved, normalized) == (2, 1)
    assert _count_tasks(shard) == 2
    assert "memory =" not in (shard / "2048" / "task.toml").read_text()
    # second run: nothing new
    assert _copy_tasks(tasks, shard) == (0, 0)


def test_apply_patch_overrides_and_adds(tmp_path: Path) -> None:
    task = tmp_path / "t"
    (task / "solution").mkdir(parents=True)
    (task / "solution" / "solve.sh").write_text("orig\n")
    patch = tmp_path / "p"
    (patch / "solution").mkdir(parents=True)
    (patch / "solution" / "solve.sh").write_text("patched\n")
    (patch / "extra.txt").write_text("added\n")
    (patch / "README.md").write_text("skip\n")
    assert set(_apply_patch(task, patch)) == {"solution/solve.sh", "extra.txt"}
    assert (task / "solution" / "solve.sh").read_text() == "patched\n"
    assert not (task / "README.md").exists()


def test_apply_all_patches_skips_absent(tmp_path: Path, monkeypatch) -> None:
    import xrlenv_plugins.benchmarks.lhtb.build_cache as bc

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


def test_is_populated(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    assert not is_populated(shard)
    (shard / "2048").mkdir(parents=True)
    (shard / "2048" / "task.toml").write_text("[environment]\n")
    assert is_populated(shard)


def test_recover_files_based_oracles(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import recover_files_based_oracles

    shard = tmp_path / "lhtb"
    # a files-based task (reference in solution/files/, no solve.sh) -> recovered
    fb = shard / "audit-task" / "solution" / "files" / "src"
    fb.mkdir(parents=True)
    (fb / "audit.py").write_text("# reference impl\n")
    # a task that already has solve.sh -> left alone
    sc = shard / "self-contained" / "solution"
    sc.mkdir(parents=True)
    (sc / "solve.sh").write_text("#!/bin/bash\necho hi\n")
    (sc / "files").mkdir()
    # a game task (no files/) -> not touched
    (shard / "game" / "solution").mkdir(parents=True)

    recovered = recover_files_based_oracles(shard)
    assert recovered == ["audit-task"]
    solve = shard / "audit-task" / "solution" / "solve.sh"
    assert solve.exists()
    body = solve.read_text()
    assert 'cp -a "$SCRIPT_DIR/files/." "$APP_DIR/"' in body
    # idempotent + doesn't clobber an existing solve.sh
    assert recover_files_based_oracles(shard) == []
    assert (shard / "self-contained" / "solution" / "solve.sh").read_text() == "#!/bin/bash\necho hi\n"


def _mk_duckdb(shard: Path, *, env_section: str = "[environment.env]\n") -> Path:
    """A minimal duckdb-optimizer-closure task dir with the os.cpu_count()-based
    build parallelism in both the harness and the verifier rebuild."""
    d = shard / "duckdb-optimizer-closure"
    (d / "environment" / "harness").mkdir(parents=True)
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir(parents=True)
    (d / "task.toml").write_text(
        "[environment]\ncpus = 4\nmemory_mb = 8192\n\n" + env_section + "[solution.env]\n",
    )
    harness_body = (
        "import os, subprocess\n"
        "nproc = os.cpu_count() or 4\n"
        'subprocess.run(["ninja", f"-j{nproc}"])\n'
    )
    (d / "environment" / "harness" / "duckdb_harness.py").write_text(harness_body)
    (d / "tests" / "test_outputs.py").write_text(harness_body)
    (d / "solution" / "solve.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd /app\nbench run\n",
    )
    return d


def test_fix_nproc_scaling_oracles(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    d = _mk_duckdb(shard)
    assert fix_nproc_scaling_oracles(shard) == ["duckdb-optimizer-closure"]

    # both the build harness AND the verifier rebuild now ask for *usable* CPUs
    for rel in ("environment/harness/duckdb_harness.py", "tests/test_outputs.py"):
        body = (d / rel).read_text()
        assert "os.cpu_count()" not in body
        assert "len(os.sched_getaffinity(0)) or 4" in body
    # the task is marked for xrlenv cpuset pinning, under [environment.env]
    toml = (d / "task.toml").read_text()
    assert 'XRLENV_CPU_PINNING = "1"' in toml
    env_idx = toml.index("[environment.env]")
    assert toml.index("XRLENV_CPU_PINNING", env_idx) < toml.index("[solution.env]")

    # ROOT fix, not a solve.sh workaround: solve.sh is left untouched (the harness fix
    # is baked into the image on rebuild).
    solve = (d / "solution" / "solve.sh").read_text()
    assert "sched_getaffinity" not in solve
    assert "sed -i" not in solve

    # idempotent — a second pass changes nothing
    assert fix_nproc_scaling_oracles(shard) == []


def test_fix_nproc_scaling_oracles_adds_env_section_when_missing(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    d = _mk_duckdb(shard, env_section="")  # no [environment.env] at all
    assert fix_nproc_scaling_oracles(shard) == ["duckdb-optimizer-closure"]
    toml = (d / "task.toml").read_text()
    assert "[environment.env]" in toml and 'XRLENV_CPU_PINNING = "1"' in toml


def test_fix_nproc_scaling_oracles_skips_absent(tmp_path: Path) -> None:
    shard = tmp_path / "lhtb"
    (shard / "some-other-task").mkdir(parents=True)
    (shard / "some-other-task" / "task.toml").write_text("[environment]\n")
    assert fix_nproc_scaling_oracles(shard) == []


def _mk_audit_image(shard: Path, name: str, dockerfile: str) -> Path:
    d = shard / name
    (d / "environment").mkdir(parents=True)
    (d / "task.toml").write_text("[environment]\nallow_internet = true\n")
    (d / "environment" / "Dockerfile").write_text(dockerfile)
    return d


def test_bake_patch_binary_adds_run_after_from(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import bake_patch_binary

    shard = tmp_path / "lhtb"
    # a curated patch-less image (in _PATCHLESS_IMAGE_TASKS) -> baked
    d = _mk_audit_image(
        shard, "materials-phase-diagram-audit",
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app/\n",
    )
    # a task NOT in the curated list -> untouched, even with a Dockerfile
    _mk_audit_image(shard, "some-other-audit", "FROM python:3.11-slim\n")

    baked = bake_patch_binary(shard)
    assert baked == ["materials-phase-diagram-audit"]

    df = (d / "environment" / "Dockerfile").read_text()
    # the RUN lands right after FROM (early layer), installs patch, before COPY
    assert "apt-get install -y -qq --no-install-recommends patch" in df
    assert df.index("FROM python") < df.index("install -y -qq") < df.index("COPY . /app/")
    # not touched: a non-curated task
    assert "install" not in (shard / "some-other-audit" / "environment" / "Dockerfile").read_text()

    # idempotent
    assert bake_patch_binary(shard) == []


def test_bake_patch_binary_skips_absent(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import bake_patch_binary

    shard = tmp_path / "lhtb"  # none of the curated tasks present
    (shard / "unrelated").mkdir(parents=True)
    assert bake_patch_binary(shard) == []


def test_regenerate_sokoban_reference(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import regenerate_sokoban_reference

    shard = tmp_path / "lhtb"
    gen = shard / "sokoban" / "solution" / "gen"
    gen.mkdir(parents=True)
    # a stand-in generator that writes ../reference_moves.log (mirrors the real one's
    # contract: run from gen/, write the sibling solution/reference_moves.log)
    (gen / "generate_reference.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).resolve().parent.parent.joinpath('reference_moves.log')"
        ".write_text('UDLR' * 10)\n",
    )
    assert regenerate_sokoban_reference(shard) is True
    ref = shard / "sokoban" / "solution" / "reference_moves.log"
    assert ref.is_file() and ref.stat().st_size > 0
    # idempotent — a second call skips (log already present)
    assert regenerate_sokoban_reference(shard) is False


def test_regenerate_sokoban_reference_absent(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import regenerate_sokoban_reference

    shard = tmp_path / "lhtb"
    (shard / "other").mkdir(parents=True)
    assert regenerate_sokoban_reference(shard) is False


def test_raise_slow_verifier_timeouts(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import raise_slow_verifier_timeouts

    shard = tmp_path / "lhtb"
    d = shard / "vector-db-iterative-build"
    d.mkdir(parents=True)
    (d / "task.toml").write_text(
        "[verifier]\ntimeout_sec = 600.0\n\n[agent]\ntimeout_sec = 14400.0\n",
    )
    assert raise_slow_verifier_timeouts(shard) == ["vector-db-iterative-build"]
    toml = (d / "task.toml").read_text()
    assert "timeout_sec = 2400.0" in toml            # verifier raised
    assert "timeout_sec = 14400.0" in toml           # agent untouched
    # idempotent (already >= target)
    assert raise_slow_verifier_timeouts(shard) == []


def test_regenerate_2048_reference(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import regenerate_2048_reference

    shard = tmp_path / "lhtb"
    gen = shard / "2048" / "solution" / "gen"
    gen.mkdir(parents=True)
    # stand-in generator that honours --max-moves and writes ../reference_moves.log
    (gen / "generate_reference.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--max-moves', type=int)\n"
        "a = ap.parse_args()\n"
        "Path(__file__).resolve().parent.parent.joinpath('reference_moves.log')"
        ".write_text('\\n'.join('UDLR'[i % 4] for i in range(a.max_moves)))\n"
    )
    assert regenerate_2048_reference(shard) is True
    ref = shard / "2048" / "solution" / "reference_moves.log"
    assert ref.is_file() and ref.stat().st_size > 0
    # the --max-moves cap (250) was actually threaded through to the generator
    assert len(ref.read_text().split()) == 250
    # idempotent — second call skips
    assert regenerate_2048_reference(shard) is False


def test_regenerate_snake_maze_reference(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import regenerate_snake_maze_reference

    shard = tmp_path / "lhtb"
    sol = shard / "snake_maze_campaign" / "solution"
    sol.mkdir(parents=True)
    # stand-in solver: honours --out (and ignores the search knobs)
    (sol / "search_solver.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "for f in ('--beam','--paths-per-state','--path-attempts','--max-foods','--seed'):\n"
        "    ap.add_argument(f)\n"
        "ap.add_argument('--out', type=Path)\n"
        "a = ap.parse_args()\n"
        "a.out.write_text('U\\nD\\nL\\nR\\n')\n"
    )
    assert regenerate_snake_maze_reference(shard) is True
    ref = sol / "reference_moves.log"
    assert ref.is_file() and ref.stat().st_size > 0
    # idempotent — second call skips
    assert regenerate_snake_maze_reference(shard) is False


def test_regenerate_snake_maze_reference_absent(tmp_path: Path) -> None:
    from xrlenv_plugins.benchmarks.lhtb.build_cache import regenerate_snake_maze_reference

    shard = tmp_path / "lhtb"
    (shard / "snake_maze_campaign" / "solution").mkdir(parents=True)
    assert regenerate_snake_maze_reference(shard) is False  # no search_solver.py


# ── --stage repin (point REBUILD tasks at the private registry) ───────────────

_REG = "node-host:5011"


def test_repin_docker_image_text_rewrites_value() -> None:
    text = '[environment]\ndocker_image = "zli12321/lhtb-duckdb:20260615"\nmemory_mb = 8192\n'
    out, changed = repin_docker_image_text(text, f"{_REG}/lhtb/duckdb-optimizer-closure:main")
    assert changed
    assert f'docker_image = "{_REG}/lhtb/duckdb-optimizer-closure:main"' in out
    assert "memory_mb = 8192" in out  # other lines preserved


def test_repin_docker_image_text_noop_when_already_target() -> None:
    ref = f"{_REG}/lhtb/duckdb-optimizer-closure:main"
    text = f'[environment]\ndocker_image = "{ref}"\n'
    out, changed = repin_docker_image_text(text, ref)
    assert not changed and out == text


def _mk_pinned(shard: Path, name: str, docker_image: str) -> None:
    d = shard / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text(
        f'[environment]\ndocker_image = "{docker_image}"\nmemory_mb = 8192\n',
    )


def test_repin_only_rebuild_tasks_idempotent(tmp_path: Path) -> None:
    import tomllib

    shard = tmp_path / "lhtb"
    # a REBUILD task + a green task; only the REBUILD one is repinned.
    _mk_pinned(shard, "duckdb-optimizer-closure", "zli12321/lhtb-duckdb-optimizer-closure:20260615")
    _mk_pinned(shard, "2048", "zli12321/lhtb-2048:20260615")
    repinned = repin_to_private_registry(shard)
    assert repinned == ["duckdb-optimizer-closure"]

    got = tomllib.loads((shard / "duckdb-optimizer-closure" / "task.toml").read_text())
    # host-agnostic placeholder (host resolved from .env at run time) — NOT a baked IP
    expected = f"{PRIVATE_REGISTRY_PLACEHOLDER}/lhtb/duckdb-optimizer-closure:main"
    assert got["environment"]["docker_image"] == expected
    assert "${XRLENV_PRIVATE_REGISTRY_HOST}" in expected
    # green task left on docker.io
    green = tomllib.loads((shard / "2048" / "task.toml").read_text())
    assert green["environment"]["docker_image"] == "zli12321/lhtb-2048:20260615"

    # second run is a no-op (already at the placeholder).
    assert repin_to_private_registry(shard) == []


def test_repin_uses_rebuild_tasks_set() -> None:
    # the repin scope is exactly the shared REBUILD_TASKS.
    assert "chess-mate" in REBUILD_TASKS
    assert "duckdb-optimizer-closure" in REBUILD_TASKS
    assert "2048" not in REBUILD_TASKS
