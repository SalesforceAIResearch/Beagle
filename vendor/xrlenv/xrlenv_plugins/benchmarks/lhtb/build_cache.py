#!/usr/bin/env python3
"""Build a shared harbor task cache for LHTB (Long-Horizon Terminal-Bench).

Self-contained pipeline, mirroring ``deep_swe/build_cache.py`` / ``terminal_bench_2_1``
in shape (``--stage`` driven, idempotent). LHTB is a **pure harbor-format task corpus**
(`zli12321/LHTB`, 46 tasks) committed in-git under ``tasks/<name>/`` — each task unpacks
to ``task.toml`` + ``instruction.md`` + ``environment/Dockerfile`` + ``tests/`` +
``solution/solve.sh``, the same contract tb2.1 / deep-swe use. Tasks pin a **prebuilt
public Docker Hub image** (``[environment] docker_image = zli12321/lhtb-<task>:<date>``),
so this is the tb2.1/deep-swe *prebuilt-image* shape — nothing is built on our side.

Stages
------
    populate  Shallow-clone ``zli12321/LHTB`` (with ``git lfs pull`` for the large
              task assets — ``*.zip``/``*.mp4``/``*.gif``), copy each ``tasks/<name>/``
              into ``<dest>/lhtb/<name>/``, and normalize its ``task.toml``. Needs
              network + ``git`` (+ ``git-lfs`` for the APEX tasks' assets).
    patch     Overlay curated ``patches/<name>/`` full-file fixes + the programmatic
              task-level fixes (files-based oracle recovery, nproc right-sizing, the
              ``patch``-binary bake, game-reference regen, verifier-timeout bumps). The
              image-defect fixes write into the **build context** (Dockerfile / harness)
              so a rebuild bakes them — persistent for the oracle AND a real agent.
    repin     Re-point ONLY the :data:`REBUILD_TASKS`' task.toml ``docker_image`` at
              ``<registry>/lhtb/<task>:main`` (needs ``--registry``), so the sweep
              resolves the fixed image after they're built+pushed. Standalone stage;
              ``all`` does this by default too (this stage is the "just repin, don't
              re-patch" shortcut).
    all       populate (if missing) + patch + **repin the REBUILD tasks** — the default.
              Repin is on by default (a docker.io-ref cache silently ships the 6 broken
              rebuild images), so ``all`` REFUSES to guess: pass ``--registry`` (repin at
              your private registry) OR ``--use-upstream-image`` (keep the broken
              docker.io refs — the out-of-box-gate path, which excludes those 6 anyway).

Why git, not HF: the HF mirror ``IntelligenceLab/Long-Horizon-Terminal-Bench``
intentionally withholds ``tests/`` and ``solution/`` — the git repo is the only
complete source (we need both for the verifier + the oracle sweep).

Images: most tasks pull a prebuilt docker.io image (``build_plan_gen.py --all`` warm
plan). The :data:`REBUILD_TASKS` are built by us instead — ``build_plan_gen.py --all``
→ ``scripts/build_and_push_images.py`` → this script's ``--stage repin``. This script
populates the *task-dir* cache and (in ``patch``) fixes the build context of the
rebuild tasks.

How xrlenv consumers pick it up
-------------------------------
xrlenv's harbor onboarding resolves a task by local *path* (``TaskConfig(path=…)`` →
harbor ``LocalTaskId``), searching under ``$XRLENV_BENCHMARK_CACHE``. A local task is used
as-is, so the normalized ``task.toml`` this script writes is authoritative. Point every
consumer's ``XRLENV_BENCHMARK_CACHE`` at the shared root this script writes; tasks land at
``<root>/lhtb/<name>/`` beside the tb2.1 / deep-swe / terminalworld shards.

Env overrides (populate):
    LHTB_REPO_URL   git remote        (default https://github.com/zli12321/LHTB)
    LHTB_REPO_REF   git ref/branch    (default main)
    LHTB_SHARD      cache shard name  (default lhtb)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

# ── Dataset identity ──────────────────────────────────────────────────────────
REPO_URL = os.environ.get("LHTB_REPO_URL", "https://github.com/zli12321/LHTB")
REPO_REF = os.environ.get("LHTB_REPO_REF", "main")
# Shard subdir name == image namespace convention (one name, double duty).
SHARD = os.environ.get("LHTB_SHARD", "lhtb")

# The tasks whose image we BUILD ourselves (``type: local``) instead of pulling the
# prebuilt docker.io one — because it's published nowhere (chess-mate's sidecar) or the
# fix must live in the image, not the task dir (the baked-defect rebuilds, fixed at the
# root by ``fix_nproc_scaling_oracles`` / ``bake_patch_binary`` / a stale-daemon
# rebuild). ``build_plan_gen --all`` materializes these + ``--stage repin`` points
# their task.toml at the pushed ref. Single source of truth, shared with build_plan_gen.
REBUILD_TASKS = frozenset({
    "chess-mate",                          # multi-service: main + game sidecar (unpublished)
    "duckdb-optimizer-closure",            # baked ``-j{os.cpu_count()}`` harness
    "climate-netcdf-extreme-event-audit",  # zhongzhi660 image lacks GNU ``patch``
    "materials-phase-diagram-audit",       # zhongzhi660 image lacks GNU ``patch``
    "robotics-slam-benchmark-repair",      # zhongzhi660 image lacks GNU ``patch``
    "unknown-config-semantics",            # stale baked daemon (missing ``nonce``)
})

# Curated per-task fixes live beside this script. Each ``patches/<name>/<rel>`` file is
# overlaid (full-file replacement) onto the extracted task dir AFTER extraction +
# normalization, so the fixes survive re-populate. See patches/README.md.
PATCHES_DIR = Path(__file__).resolve().parent / "patches"

_TASK_ANCHOR = "task.toml"


# ── task.toml normalization (pure) ────────────────────────────────────────────


def normalize_task_toml_text(text: str) -> tuple[str, bool]:
    """Strip the harbor-deprecated ``memory`` / ``storage`` string keys from a
    task.toml's ``[environment]`` section when the canonical ``memory_mb`` /
    ``storage_mb`` are also present. Returns ``(new_text, changed)``.

    harbor's ``EnvironmentConfig`` carries BOTH the deprecated string field and the
    canonical ``_mb`` integer; if a task sets them inconsistently the model rejects the
    conflict and the task won't load. The least-lossy fix is to drop the deprecated
    duplicate and keep the canonical ``_mb`` field. Surgical (line-level). Pure (no I/O)
    so the logic is unit-testable. A no-op when the conflict isn't present (LHTB uses
    ``_mb`` only — kept defensively).
    """
    env = tomllib.loads(text).get("environment", {})
    strip = {
        legacy
        for legacy, canonical in (("memory", "memory_mb"), ("storage", "storage_mb"))
        if legacy in env and canonical in env
    }
    if not strip:
        return text, False

    out: list[str] = []
    section: str | None = None
    for line in text.splitlines(keepends=True):
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            section = header.group(1).strip()
        if section == "environment":
            key = re.match(r"\s*([A-Za-z_]+)\s*=", line)
            if key and key.group(1) in strip:
                continue  # drop the deprecated duplicate
        out.append(line)
    return "".join(out), True


def _normalize_task_toml(path: Path) -> bool:
    new_text, changed = normalize_task_toml_text(path.read_text(encoding="utf-8"))
    if changed:
        path.write_text(new_text, encoding="utf-8")
    return changed


# ── Stage 1: populate ─────────────────────────────────────────────────────────


def _copy_tasks(tasks_root: Path, shard_dir: Path) -> tuple[int, int]:
    """Copy every ``<tasks_root>/<name>/`` that carries a ``task.toml`` into
    ``<shard_dir>/<name>/`` (idempotent — skip a task already present), normalizing
    each ``task.toml``. Returns ``(moved, normalized)``."""
    if not tasks_root.is_dir():
        raise SystemExit(
            f"ERROR: expected a 'tasks/' directory at {tasks_root} — is this the LHTB "
            f"repo layout? (tasks/<name>/task.toml)",
        )
    shard_dir.mkdir(parents=True, exist_ok=True)
    task_dirs = sorted(
        p for p in tasks_root.iterdir()
        if p.is_dir() and (p / _TASK_ANCHOR).is_file()
    )
    if not task_dirs:
        raise SystemExit(f"ERROR: no tasks with {_TASK_ANCHOR} under {tasks_root}")
    moved = normalized = 0
    for i, src in enumerate(task_dirs, 1):
        dest = shard_dir / src.name
        if (dest / _TASK_ANCHOR).is_file():
            continue  # already present — idempotent
        shutil.copytree(src, dest)
        moved += 1
        if _normalize_task_toml(dest / _TASK_ANCHOR):
            normalized += 1
        if i % 10 == 0 or i == len(task_dirs):
            print(f"   [{i}/{len(task_dirs)}] {src.name}", file=sys.stderr)
    return moved, normalized


def populate_git(shard_dir: Path) -> tuple[int, int]:
    """Shallow-clone the LHTB repo (+ ``git lfs pull``) and copy its ``tasks/`` into
    the shard. LFS covers the large ``*.zip``/``*.mp4``/``*.gif`` task assets (APEX
    tasks); ``git-lfs`` is required for those to be real content, not pointer files."""
    with tempfile.TemporaryDirectory(prefix="lhtb-clone-") as tmp:
        print(f">> git clone --depth 1 --branch {REPO_REF} {REPO_URL}", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, tmp],
            check=True,
        )
        # Ensure LFS assets are materialized (the smudge filter usually does this on
        # clone, but pull explicitly so a no-smudge config doesn't leave pointers).
        lfs = subprocess.run(
            ["git", "-C", tmp, "lfs", "pull"], capture_output=True, text=True,
        )
        if lfs.returncode != 0:
            print(
                f">> WARNING: `git lfs pull` failed ({lfs.stderr.strip()[:200]}); "
                f"LFS assets (APEX .zip/.mp4) may be pointer files. Install git-lfs.",
                file=sys.stderr,
            )
        return _copy_tasks(Path(tmp) / "tasks", shard_dir)


# ── Stage 2: patch (curated full-file overlays) ───────────────────────────────


def _apply_patch(task_dir: Path, patch_dir: Path) -> list[str]:
    """Overlay every file under ``patch_dir`` onto ``task_dir`` (full-file
    replacement), preserving relative layout + exec bits. Returns the relative paths
    overridden. A patch file with no counterpart is still copied (lets a patch ADD a
    missing file)."""
    overridden: list[str] = []
    for f in sorted(patch_dir.rglob("*")):
        if not f.is_file() or f.name == "README.md":
            continue
        rel = f.relative_to(patch_dir)
        target = task_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        overridden.append(rel.as_posix())
    return overridden


def apply_all_patches(shard_dir: Path) -> int:
    """Apply every ``patches/<name>/`` overlay to its present task in the shard.
    Idempotent; a task absent from the shard is skipped with a SKIP note. Returns the
    number of tasks patched (0 while ``patches/`` is empty)."""
    if not PATCHES_DIR.is_dir():
        return 0
    patched = 0
    for patch_dir in sorted(PATCHES_DIR.iterdir()):
        if not patch_dir.is_dir():
            continue  # skip README.md etc.
        name = patch_dir.name
        dest = shard_dir / name
        if not (dest / _TASK_ANCHOR).is_file():
            print(
                f"   [patch] {name}: SKIP — not present in shard (populate first)",
                file=sys.stderr,
            )
            continue
        files = _apply_patch(dest, patch_dir)
        if files:
            patched += 1
            print(f"   [patch] {name}: overrode {files}", file=sys.stderr)
    return patched


# ── Stage 2b: recover files-based oracles ─────────────────────────────────────
# Several LHTB *-audit / *-regression tasks ship a reference IMPLEMENTATION under
# ``solution/files/`` (mirroring the app layout, WORKDIR ``/app``) but NO
# ``solution/solve.sh``. harbor's OracleAgent requires a solve.sh (it uploads the
# whole solution/ dir and runs solve.sh), so without one it errors "Solution script
# not found". The reference IS shipped, so the oracle just needs to install it — we
# synthesize a generic files-install solve.sh. Faithful (installs the shipped
# reference, no re-implementation) and verified: epa-swmm → reward 1.0.

_FILES_INSTALL_SOLVE = """#!/bin/bash
# xrlenv onboarding (build_cache.py): install the shipped reference implementation.
# This LHTB task ships its reference under solution/files/ (mirroring the app layout)
# but no solve.sh; the oracle just copies it into the app dir. Faithful — the
# reference is the benchmark's own, unmodified.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${APP_DIR:-/app}"
cp -a "$SCRIPT_DIR/files/." "$APP_DIR/"
echo "installed reference from solution/files/ -> $APP_DIR"
"""


def recover_files_based_oracles(shard_dir: Path) -> list[str]:
    """Write a generic files-install ``solve.sh`` for each task that ships a
    reference under ``solution/files/`` but has no ``solution/solve.sh``. Idempotent
    (skips a task that already has a solve.sh). Returns the recovered task names."""
    recovered: list[str] = []
    for d in sorted(shard_dir.glob("*/")):
        sol = d / "solution"
        if (sol / "files").is_dir() and not (sol / "solve.sh").exists():
            (sol / "solve.sh").write_text(_FILES_INSTALL_SOLVE)
            (sol / "solve.sh").chmod(0o755)
            recovered.append(d.name)
    return recovered


# ── Stage 2c: right-size nproc-scaling build oracles ──────────────────────────
# A few LHTB oracles build a big C++ project from source and size the compiler
# job count from ``os.cpu_count()`` (e.g. ``ninja -j{os.cpu_count()}``). On a large
# host (our 192-core HyperPod nodes) that fans out to ~192 parallel compilers inside
# the task's declared memory cap (duckdb-optimizer-closure: cpus=4 / 8 GiB) → OOM →
# the build fails → no binary → every graded query fails → reward 0. This is the
# big-node ``nproc`` trap (cf. tb2.1 install-windows), but harder: ``os.cpu_count()``
# ignores CPU affinity by CPython design, so xrlenv's cpuset PIN alone doesn't lower
# ``-j`` — the harness must ask for the *usable* CPUs instead. **Root-cause** fix
# (persistent for the oracle AND a real agent, not a solve.sh-only workaround):
#   1. Rewrite ``os.cpu_count()`` -> ``len(os.sched_getaffinity(0))`` (the process's
#      *usable* CPUs — respects the cpuset) wherever the harness/verifier size ``-j``
#      off it. The **verifier** (``tests/*.py``) is uploaded at runtime. The build
#      **harness** (``environment/harness/*.py``) is COPYed INTO the image by the
#      Dockerfile — so patching the task-dir copy only bites once the image is
#      **rebuilt** (``build_plan_gen --all`` → ``build_and_push_images.py``); the
#      rebuilt image bakes the fixed harness, so ``bench run`` fans out correctly for
#      a real agent too. ``duckdb-optimizer-closure`` is therefore a REBUILD task
#      (see :data:`REBUILD_TASKS`), not an out-of-box docker.io pass.
#   2. Mark the task for xrlenv cpuset pinning (``[environment.env] XRLENV_CPU_PINNING``)
#      so the affinity mask is sized to ``ceil(cpus)`` — then (1) reads that budget.
# Curated allowlist (like tb2.1's surgical per-task marking): only tasks confirmed to
# hit this trap. Add a name here when the oracle sweep surfaces another.
_NPROC_SCALING_ORACLES = ("duckdb-optimizer-closure",)
_CPU_COUNT_SRC = "os.cpu_count()"
_CPU_COUNT_DST = "len(os.sched_getaffinity(0))"
_PIN_MARKER = 'XRLENV_CPU_PINNING = "1"'


def _rewrite_cpu_count(py: Path) -> bool:
    """Swap ``os.cpu_count()`` -> ``len(os.sched_getaffinity(0))`` in a harness .py
    (both size ``-j`` off it). Idempotent; returns True if the file changed. ``os`` is
    already imported in these harnesses and ``sched_getaffinity`` is Linux-only —
    fine, harbor tasks run in Linux containers."""
    text = py.read_text(encoding="utf-8")
    if _CPU_COUNT_SRC not in text:
        return False
    py.write_text(text.replace(_CPU_COUNT_SRC, _CPU_COUNT_DST), encoding="utf-8")
    return True


def _mark_cpu_pinning(task_toml: Path) -> bool:
    """Ensure ``XRLENV_CPU_PINNING = "1"`` lives under ``[environment.env]`` in a
    task.toml (xrlenv's per-task cpuset-pin opt-in). Idempotent; returns True if
    added. Inserts right after the ``[environment.env]`` header (LHTB tasks always
    carry the — often empty — section)."""
    text = task_toml.read_text(encoding="utf-8")
    if _PIN_MARKER in text:
        return False
    out: list[str] = []
    inserted = False
    for line in text.splitlines(keepends=True):
        out.append(line)
        if not inserted and re.match(r"\s*\[environment\.env\]\s*$", line):
            out.append(_PIN_MARKER + "\n")
            inserted = True
    if not inserted:  # no [environment.env] section — append one
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append(f"\n[environment.env]\n{_PIN_MARKER}\n")
    task_toml.write_text("".join(out), encoding="utf-8")
    return True


def fix_nproc_scaling_oracles(shard_dir: Path) -> list[str]:
    """For each curated nproc-scaling oracle present in the shard, right-size its build
    parallelism at the **root**: rewrite ``os.cpu_count()`` ->
    ``len(os.sched_getaffinity(0))`` in the build harness (``environment/harness/*.py``,
    baked into the image on rebuild) and the uploaded verifier (``tests/*.py``), and
    mark the task for cpuset pinning. No solve.sh workaround — the fix ships in the
    (rebuilt) image, so it holds for a real agent too. Idempotent. Returns the task
    names actually changed."""
    fixed: list[str] = []
    for name in _NPROC_SCALING_ORACLES:
        d = shard_dir / name
        if not (d / _TASK_ANCHOR).is_file():
            continue
        changed = False
        for py in sorted((d / "environment" / "harness").glob("*.py")):
            changed |= _rewrite_cpu_count(py)
        for py in sorted((d / "tests").glob("*.py")):
            changed |= _rewrite_cpu_count(py)
        changed |= _mark_cpu_pinning(d / _TASK_ANCHOR)
        if changed:
            fixed.append(name)
    return fixed


# ── Stage 2d: bake the `patch` binary the audit oracles rely on into the image ─
# The *-audit / reconstruction oracles apply their reference fix with GNU patch
# (`patch --dry-run -p0 < fix_audit.patch` then `patch -p0 < …`). Three prebuilt
# ``zhongzhi660/lhtb-*`` images ship WITHOUT the ``patch`` binary, so the invocation
# fails "command not found", the reference fix is never applied, and the grade is on
# UNPATCHED code → near-zero reward (materials 0.058, robotics 0.021, climate 0.067).
# Proof it's the tool and not a partial reference: the identical pattern on an image
# that *has* patch (microscopy) scores 1.0.
#
# **Root-cause** fix: add a ``RUN apt-get install -y patch`` to the task's own
# ``environment/Dockerfile`` so the REBUILT image ships ``patch`` — for the oracle AND
# a real agent, not just the oracle's solve.sh. These are therefore REBUILD tasks
# (:data:`REBUILD_TASKS`), materialized by ``build_plan_gen --all`` →
# ``build_and_push_images.py``. Curated list (the 3 images confirmed to lack it); the
# other patch-using audits already ship it, so they need no change and stay on
# docker.io.
_PATCHLESS_IMAGE_TASKS = (
    "climate-netcdf-extreme-event-audit",
    "materials-phase-diagram-audit",
    "robotics-slam-benchmark-repair",
)
_PATCH_BAKE_MARKER = "# xrlenv: bake GNU patch (this image ships without it)"
_PATCH_BAKE_BLOCK = (
    f"{_PATCH_BAKE_MARKER}\n"
    "RUN (apt-get update -qq && apt-get install -y -qq --no-install-recommends patch) "
    "|| apk add --no-cache patch\n"
)


def _bake_dockerfile_run(dockerfile: Path, marker: str, block: str) -> bool:
    """Insert a ``RUN`` ``block`` into ``dockerfile`` right after its first ``FROM``
    (so it layers early and lands in the build stage). Idempotent (keyed off
    ``marker``). Returns True if inserted; False if absent or already baked."""
    if not dockerfile.is_file():
        return False
    text = dockerfile.read_text(encoding="utf-8")
    if marker in text:
        return False
    lines = text.splitlines(keepends=True)
    insert_at = len(lines)  # fall back to end if there's somehow no FROM
    for i, line in enumerate(lines):
        if line.lstrip().upper().startswith("FROM "):
            insert_at = i + 1
            break
    if insert_at < len(lines) and not lines[insert_at - 1].endswith("\n"):
        lines[insert_at - 1] += "\n"
    lines.insert(insert_at, block)
    dockerfile.write_text("".join(lines), encoding="utf-8")
    return True


def bake_patch_binary(shard_dir: Path) -> list[str]:
    """Add a ``patch``-install ``RUN`` to the ``environment/Dockerfile`` of each
    curated ``patch``-less image, so the rebuilt image ships GNU ``patch``. Idempotent.
    Returns the task names changed."""
    baked: list[str] = []
    for name in _PATCHLESS_IMAGE_TASKS:
        d = shard_dir / name
        if not (d / _TASK_ANCHOR).is_file():
            continue
        if _bake_dockerfile_run(
            d / "environment" / "Dockerfile", _PATCH_BAKE_MARKER, _PATCH_BAKE_BLOCK,
        ):
            baked.append(name)
    return baked


# ── Stage 2e: regenerate the game references in-cache (FALLBACK) ───────────────
# NB these are **fallbacks**. A stronger reference is committed under
# ``patches/<task>/solution/reference_moves.log`` and copied by ``apply_all_patches``
# (which runs FIRST in main()); each regen below skips when a non-empty log already
# exists, so the committed patch wins. The regen only fires when no patch is present
# (e.g. a build that excludes patches) — a cheap, low-band safety net. See patches/README.md.
#
# The game tasks' oracle installs a pre-generated ``reference_moves.log`` the public
# repo does NOT ship (confirmed: ``tasks/<game>/solution/`` carries only ``gen/`` +
# ``solve.sh`` + LFS-pointer video stubs). Two are cheap enough to regenerate here:
#
#   * **sokoban** — its ``gen/reference_solutions.json`` ships the per-level move
#     strings as DATA; ``generate_reference.py`` just replays them (pure-Python,
#     seconds). Solves levels 1..92/155 → oracle ``reward≈0.59``.
#   * **2048** — a bounded run of the shipped expectimax generator. NB 2048's reward
#     is ``min(raw_band / 11, 1.0)`` where band 11 = the (impossible) 65536 tile, so
#     the classic 2048 *win* is only 0.55 and 1.0 is unreachable BY DESIGN. The oracle
#     just needs a reference that passes (``band >= PASS_BAND``, PASS_BAND defaults to
#     1) with a positive reward; a ``--max-moves 250`` run reaches band 4 (tile 512) in
#     ~1 min → ``reward≈0.36``. Going higher (band 6 = 2048 tile ≈ 0.55) needs ~1000+
#     moves of increasingly slow late-game search — too slow to run here, and pointless
#     since the reward is capped well under the 0.95 success bar regardless.
#
# Both are faithful (the benchmark's own unmodified generator) and written in-cache but
# NOT committed (they carry the "never in training corpora" canary). super-mario (a
# torch+net docker gen image) and snake_maze_campaign (a heavy bounded beam search)
# stay offline manual steps — see STATUS.md.
_GAME_REF = "reference_moves.log"
# 2048's move cap: band 4 (tile 512, reward≈0.36) in ~1 min; the sweet spot between
# "passes with a non-trivial reward" and "fast enough to run in --stage all".
_G2048_MAX_MOVES = 250


def _regen_game_reference(
    shard_dir: Path, task: str, extra_args: list[str], timeout: int,
) -> bool:
    """Run ``<task>/solution/gen/generate_reference.py`` (with ``extra_args``) to write
    ``solution/reference_moves.log``. Idempotent (skips a non-empty log). Best-effort —
    returns True iff it wrote the log; False if absent/skipped/failed."""
    gen_dir = shard_dir / task / "solution" / "gen"
    ref = shard_dir / task / "solution" / _GAME_REF
    if not (gen_dir / "generate_reference.py").is_file():
        return False
    if ref.is_file() and ref.stat().st_size > 0:
        return False  # already present — idempotent
    try:
        subprocess.run(
            [sys.executable, "generate_reference.py", *extra_args],
            cwd=str(gen_dir), check=True, capture_output=True, timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:  # pragma: no cover
        print(f"   [{task}] reference regen skipped: {exc}", file=sys.stderr)
        return False
    return ref.is_file() and ref.stat().st_size > 0


def regenerate_sokoban_reference(shard_dir: Path) -> bool:
    """Replay sokoban's shipped move data → ``reference_moves.log`` (seconds).
    Idempotent; returns True iff it wrote the log."""
    return _regen_game_reference(shard_dir, "sokoban", [], timeout=300)


def regenerate_2048_reference(shard_dir: Path) -> bool:
    """Bounded run of 2048's shipped expectimax generator → ``reference_moves.log``
    (band ~4, reward≈0.36, ~1 min). Idempotent; returns True iff it wrote the log."""
    return _regen_game_reference(
        shard_dir, "2048", ["--max-moves", str(_G2048_MAX_MOVES)], timeout=600,
    )


# snake_maze_campaign ships NO ``gen/`` — its reference is produced by
# ``solution/search_solver.py`` (a bounded beam search over the shipped snake_engine).
# The reward is ``min(1, foods·(foods+1)/2 / 1830)`` (60 foods = 1.0) and PASS needs
# band ≥ 3 ⇒ ≥ 15 foods. A modest bounded search (``--max-foods 25`` at beam 30)
# reaches 25 foods → band 5, ``reward≈0.18``, PASS — in ~75 s, no docker. A near-1.0
# reference (60 foods) needs the heavy default search (beam 250, hours), so we stop at
# a solid PASS. Deterministic (``--seed 1``).
_SNAKE_SOLVER_ARGS = [
    "--beam", "30", "--paths-per-state", "4", "--path-attempts", "4",
    "--max-foods", "25", "--seed", "1",
]


def regenerate_snake_maze_reference(shard_dir: Path) -> bool:
    """Run snake_maze_campaign's shipped ``search_solver.py`` (bounded) → ``solution/
    reference_moves.log`` (25 foods, band 5, reward≈0.18, ~75 s). Idempotent; returns
    True iff it wrote the log."""
    sol = shard_dir / "snake_maze_campaign" / "solution"
    ref = sol / _GAME_REF
    if not (sol / "search_solver.py").is_file():
        return False
    if ref.is_file() and ref.stat().st_size > 0:
        return False  # already present — idempotent
    try:
        subprocess.run(
            [sys.executable, "search_solver.py", *_SNAKE_SOLVER_ARGS,
             "--out", str(ref)],
            cwd=str(sol), check=True, capture_output=True, timeout=600,
        )
    except (subprocess.SubprocessError, OSError) as exc:  # pragma: no cover
        print(f"   [snake_maze] reference regen skipped: {exc}", file=sys.stderr)
        return False
    return ref.is_file() and ref.stat().st_size > 0


# ── Stage 2f: raise verifier timeout for a genuinely-slow-to-grade oracle ──────
# ``vector-db-iterative-build`` declares ``[verifier] timeout_sec = 600`` but its
# grader rebuilds + re-benchmarks the HNSW index from scratch, which on our nodes runs
# ~20 min — well past 600 s (x the sweep's 1.5 multiplier = 900 s), so the oracle died
# ``VerifierTimeoutError`` at 0 reward. It is NOT a content defect: at 4x it completes
# and the reference scores ``reward=0.86``. Raise the task's own verifier budget so it
# passes the default gate without a global multiplier. Faithful — a budget change, not
# a grading change. Curated map ``task -> verifier timeout_sec`` (idempotent: only
# raises, never lowers).
_VERIFIER_TIMEOUT_OVERRIDES = {"vector-db-iterative-build": 2400.0}


def _raise_verifier_timeout(task_toml: Path, want: float) -> bool:
    """Raise ``[verifier] timeout_sec`` to ``want`` if it's currently lower. Surgical
    (line-level, only inside the ``[verifier]`` section). Idempotent; returns True if
    changed."""
    text = task_toml.read_text(encoding="utf-8")
    out: list[str] = []
    section: str | None = None
    changed = False
    for line in text.splitlines(keepends=True):
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            section = header.group(1).strip()
        m = re.match(r"(\s*timeout_sec\s*=\s*)([0-9.]+)(.*)$", line)
        if section == "verifier" and m and float(m.group(2)) < want:
            out.append(f"{m.group(1)}{want}{m.group(3)}\n")
            changed = True
            continue
        out.append(line)
    if changed:
        task_toml.write_text("".join(out), encoding="utf-8")
    return changed


def raise_slow_verifier_timeouts(shard_dir: Path) -> list[str]:
    """Raise the verifier timeout for curated slow-to-grade oracles. Idempotent.
    Returns the task names changed."""
    changed: list[str] = []
    for name, want in _VERIFIER_TIMEOUT_OVERRIDES.items():
        toml = shard_dir / name / _TASK_ANCHOR
        if toml.is_file() and _raise_verifier_timeout(toml, want):
            changed.append(name)
    return changed


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_tasks(shard_dir: Path) -> int:
    return sum(1 for _ in shard_dir.glob(f"*/{_TASK_ANCHOR}"))


def is_populated(shard_dir: Path) -> bool:
    return shard_dir.is_dir() and _count_tasks(shard_dir) > 0


# ── CLI ───────────────────────────────────────────────────────────────────────


# ── repin — point the REBUILD tasks at the private registry (the `all` default when
#    given --registry; also the standalone `--stage repin`) ─────────────────────────


def repin_docker_image_text(text: str, new_ref: str) -> tuple[str, bool]:
    """Rewrite the ``[environment] docker_image`` value in a task.toml to ``new_ref``.
    Returns ``(new_text, changed)`` — a no-op when the value already equals ``new_ref``
    or there's no ``docker_image`` line. Line-level + pure so it is unit-testable and
    preserves surrounding formatting."""
    out: list[str] = []
    section: str | None = None
    changed = False
    for line in text.splitlines(keepends=True):
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            section = header.group(1).strip()
        if section == "environment" and not line.lstrip().startswith("#"):
            m = re.match(r"(\s*docker_image\s*=\s*).*", line, re.DOTALL)
            if m:
                newline = f'{m.group(1)}"{new_ref}"\n'
                changed |= newline != line
                out.append(newline)
                continue
        out.append(line)
    return "".join(out), changed


# Host-agnostic private-registry prefix written into a REBUILD task's docker_image. The
# harbor plugin (``_resolve_image_ref``) and ``build_plan_gen`` expand it from ``.env``
# (``os.path.expandvars``) at acquire / plan-gen — so the cache NEVER bakes a registry
# host and survives a control-plane/registry IP change (the drift that stranded these
# tasks on a dead registry). This is the seta pattern (resolve at run time). GUIDELINE §5.3.1.
PRIVATE_REGISTRY_PLACEHOLDER = "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}"


def repin_to_private_registry(
    shard_dir: Path, *,
    rebuild_tasks: frozenset[str] | set[str] = REBUILD_TASKS,
    namespace: str = SHARD, tag: str = "main",
) -> list[str]:
    """Repoint each **REBUILD** task's ``[environment] docker_image`` at the
    **host-agnostic** private-registry ref
    ``${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}/<namespace>/<task>:<tag>``
    — the host is resolved from ``.env`` at run time, never baked (so the cache survives a
    CP/registry IP change; GUIDELINE §5.3.1). No ``--registry`` needed. **Green tasks are
    untouched** (they keep their docker.io ref). **Idempotent**: a ref already at the
    placeholder is left alone; a fresh ``--stage populate`` restores the docker.io refs.
    Returns the repinned task names."""
    safe_tag = tag.replace("/", "-").replace(":", "-")
    repinned: list[str] = []
    for task in sorted(rebuild_tasks):
        toml_path = shard_dir / task / _TASK_ANCHOR
        if not toml_path.is_file():
            continue
        text = toml_path.read_text(encoding="utf-8")
        current = tomllib.loads(text).get("environment", {}).get("docker_image")
        new_ref = f"{PRIVATE_REGISTRY_PLACEHOLDER}/{namespace}/{task}:{safe_tag}"
        if current == new_ref:
            continue  # already repinned (idempotent)
        new_text, changed = repin_docker_image_text(text, new_ref)
        if changed:
            toml_path.write_text(new_text, encoding="utf-8")
            repinned.append(task)
    return repinned


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_cache",
        description=(
            "Materialize the LHTB task corpus into a shared harbor cache. Stages "
            "(each idempotent): populate -> patch -> repin; `all` (default) runs all "
            "three. `all` repins the REBUILD tasks by default and REFUSES to guess: "
            "pass --registry to repin them at your private registry, or "
            "--use-upstream-image to keep the (broken) docker.io refs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=("all", "populate", "patch", "repin"),
        default="all",
        help="all (default): populate (if missing) + patch + repin the REBUILD tasks "
        "(requires --registry OR --use-upstream-image — no silent default). populate: "
        "git clone + LFS pull + normalize (needs network). patch: curated overlays + "
        "task-level fixes only (no repin). repin: only re-point the REBUILD tasks' "
        "docker_image at the private registry (needs --registry).",
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help="Shared cache ROOT (the shard lands under "
        f"<dest>/{SHARD}/). Defaults to $XRLENV_BENCHMARK_CACHE. Point every xrlenv "
        "consumer's XRLENV_BENCHMARK_CACHE at this path.",
    )
    p.add_argument(
        "--registry",
        default=os.environ.get("XRLENV_PRIVATE_REGISTRY_HOST"),
        help="Private registry host[:port] to repin the REBUILD tasks at (used by "
        "`--stage all` and `--stage repin`), e.g. node-host:5011. Defaults to "
        "$XRLENV_PRIVATE_REGISTRY_HOST (a bare host gets :5011 appended).",
    )
    p.add_argument(
        "--use-upstream-image",
        action="store_true",
        help="`--stage all` only: skip the default repin and keep the REBUILD tasks on "
        "their upstream docker.io refs (which are broken until §2 rebuild+repin, but "
        "excluded from the out-of-box gate anyway). The explicit opt-out that lets `all` "
        "run without a private registry. Wins over --registry if both are given.",
    )
    return p


def _all_should_repin(use_upstream: bool) -> bool:
    """``--stage all`` repins the REBUILD tasks to the **host-agnostic** private-registry
    placeholder by default (the host resolves from ``.env`` at run time — no ``--registry``
    needed, no baked host; GUIDELINE §5.3.1). ``--use-upstream-image`` keeps the (broken)
    docker.io refs for the out-of-box gate."""
    return not use_upstream


def _resolve_shard(dest: str | None) -> Path:
    # Hard-reject the retired cache env var / path FIRST (renamed 2026-07-31): a caller
    # still pointing at xrlenv_harbor_cache would silently build against the wrong/stale
    # cache. Lazy import to match plugin style (plugin -> xrlenv core is allowed).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env(dest)
    if not dest:
        raise SystemExit(
            "error: no destination — pass --dest or set XRLENV_BENCHMARK_CACHE.",
        )
    return Path(dest).expanduser() / SHARD


def _verifier_timeout_raised(task_toml: Path, want: float) -> bool:
    if not task_toml.is_file():
        return False
    v = tomllib.loads(task_toml.read_text(encoding="utf-8")).get(
        "verifier", {},
    ).get("timeout_sec")
    return isinstance(v, (int, float)) and float(v) >= want


def _fix_status_lines(shard_dir: Path) -> list[str]:
    """Read-only report of the CURRENT applied-state of every task-level fix — checking
    the same markers the fixes write — so a re-run over an already-fixed cache shows the
    full ✓ picture, not an empty delta. Independent of what this run changed."""

    def _contains(p: Path, needle: str) -> bool:
        return p.is_file() and needle in p.read_text(encoding="utf-8")

    files_oracles = [
        d.name for d in sorted(shard_dir.glob("*/"))
        if _contains(d / "solution" / "solve.sh", 'cp -a "$SCRIPT_DIR/files/."')
    ]
    games = [
        d.name for d in sorted(shard_dir.glob("*/"))
        if (g := d / "solution" / _GAME_REF).is_file() and g.stat().st_size > 0
    ]
    verifier = [
        t for t, want in sorted(_VERIFIER_TIMEOUT_OVERRIDES.items())
        if _verifier_timeout_raised(shard_dir / t / _TASK_ANCHOR, want)
    ]
    nproc = [
        t for t in _NPROC_SCALING_ORACLES
        if any(_contains(p, _CPU_COUNT_DST)
               for p in (shard_dir / t / "environment" / "harness").glob("*.py"))
    ]
    baked = [
        t for t in _PATCHLESS_IMAGE_TASKS
        if _contains(shard_dir / t / "environment" / "Dockerfile", _PATCH_BAKE_MARKER)
    ]
    repinned: list[str] = []
    upstream: list[str] = []
    for t in sorted(REBUILD_TASKS):
        p = shard_dir / t / _TASK_ANCHOR
        if not p.is_file():
            continue
        img = str(
            tomllib.loads(p.read_text(encoding="utf-8"))
            .get("environment", {}).get("docker_image", ""),
        )
        head = img.split("/", 1)[0]
        # a private-registry ref carries a host with '.'/':' in the first segment;
        # a docker.io ref (``zli12321/lhtb-…``) does not — same rule as image_refs.
        (repinned if ("." in head or ":" in head) else upstream).append(t)

    lines = [
        "task-level fixes present in the cache (full state — ✓ = applied):",
        f"  files-based oracles : {len(files_oracles)} ✓  (synthesized solve.sh)",
        f"  game references     : {len(games)} ✓  {games}",
        f"  verifier timeout    : {len(verifier)} ✓  {verifier}",
        f"  🔁 duckdb nproc     : {len(nproc)} ✓  {nproc}",
        f"  🔁 patch-bake       : {len(baked)} ✓  {baked}  (audits shipped w/o GNU patch)",
    ]
    if repinned:
        lines.append(f"  🔁 REBUILD repinned : {len(repinned)} → private registry  {repinned}")
    if upstream:
        lines.append(f"  🔁 REBUILD upstream : {len(upstream)} (docker.io refs)  {upstream}")
    if repinned and baked:
        lines.append(
            "  Remark: the patch-bake audits are ALSO under 'REBUILD repinned' — the SAME tasks,"
            " two different fixes (bake puts GNU patch in the image; repin points the task"
            " at the rebuilt image). Expected, not a duplicate.",
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_dir = _resolve_shard(args.dest)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)

    # `--stage all` repins the REBUILD tasks by default — resolve the choice UP FRONT so
    # a missing one fails loud *before* the expensive git clone, not after it.
    do_repin = False
    if args.stage == "all":
        do_repin = _all_should_repin(args.use_upstream_image)

    if args.stage == "repin":
        if not is_populated(shard_dir):
            raise SystemExit(
                f"cannot repin: {shard_dir} is not populated. Run `--stage all` first.",
            )
        repinned = repin_to_private_registry(shard_dir)
        print(
            f"repinned {len(repinned)} REBUILD task.toml(s) to the host-agnostic "
            f"placeholder {PRIVATE_REGISTRY_PLACEHOLDER}/{SHARD}/<task>:main (host "
            f"resolved from .env at run time): {repinned}\n"
            f"NB build+push first: build_plan_gen --all | build_and_push_images.py "
            f"--registry <host:port> (builds the {len(REBUILD_TASKS)} type: local "
            f"entries, skips type: registry). A fresh `--stage populate` restores the "
            f"docker.io refs.",
            file=sys.stderr,
        )
        return 0

    moved = normalized = 0
    if args.stage in ("all", "populate"):
        if args.stage == "populate" or not is_populated(shard_dir):
            moved, normalized = populate_git(shard_dir)
        else:
            print(
                f">> {shard_dir} already populated "
                f"({_count_tasks(shard_dir)} tasks) — skipping clone.",
                file=sys.stderr,
            )
    if args.stage == "populate":
        print(
            f"\npopulated {moved} task(s) ({normalized} task.toml normalized) "
            f"-> {shard_dir}",
            file=sys.stderr,
        )
        return 0

    # patch (also reached by --stage all)
    if not is_populated(shard_dir):
        raise SystemExit(
            f"cannot patch: {shard_dir} is not populated. Run "
            f"`--stage populate` first (or `--stage all`).",
        )
    patched = apply_all_patches(shard_dir)
    recovered = recover_files_based_oracles(shard_dir)
    nproc_fixed = fix_nproc_scaling_oracles(shard_dir)
    patch_baked = bake_patch_binary(shard_dir)
    sokoban_regen = regenerate_sokoban_reference(shard_dir)
    g2048_regen = regenerate_2048_reference(shard_dir)
    snake_regen = regenerate_snake_maze_reference(shard_dir)
    verifier_bumped = raise_slow_verifier_timeouts(shard_dir)
    # `all` repins the REBUILD tasks by default (the choice was validated up front).
    # `--stage patch` alone shares the fix code above but never repins.
    repinned_now: list[str] = []
    if args.stage == "all" and do_repin:
        repinned_now = repin_to_private_registry(shard_dir)

    total = _count_tasks(shard_dir)

    # The DELTA (what changed this run) is a one-line footnote; the ✓ block below is the
    # FULL applied state, so a re-run over an already-fixed cache never reads as "empty".
    changed: list[str] = []
    if moved:
        changed.append(f"onboarded {moved}")
    if patched:
        changed.append(f"{patched} curated patch(es)")
    if recovered:
        changed.append(f"recovered {len(recovered)}")
    if nproc_fixed:
        changed.append("duckdb nproc")
    if patch_baked:
        changed.append(f"patch-baked {patch_baked}")
    for label, did in (("sokoban", sokoban_regen), ("2048", g2048_regen),
                       ("snake_maze", snake_regen)):
        if did:
            changed.append(f"regen {label}")
    if verifier_bumped:
        changed.append(f"verifier {verifier_bumped}")
    if repinned_now:
        changed.append(f"repinned {len(repinned_now)}")

    print(
        f"\nOK: {total} task(s) in {shard_dir}"
        + (f" ({moved} onboarded, {normalized} normalized)" if moved else ""),
        file=sys.stderr,
    )
    for _line in _fix_status_lines(shard_dir):
        print(_line, file=sys.stderr)
    print(
        "changed this run: "
        + (", ".join(changed) if changed
           else "nothing — idempotent re-run over the existing cache"),
        file=sys.stderr,
    )
    if repinned_now:
        print(
            "   ⚠ the repinned images DO NOT EXIST YET — build+push before ANY run:\n"
            "     build_plan_gen.py --all  &&  build_and_push_images.py "
            "--registry <host:port>   (host resolved from .env at run time; the "
            "repinned refs are host-agnostic)",
            file=sys.stderr,
        )

    print(
        f"Point consumers at:  export XRLENV_BENCHMARK_CACHE={shard_dir.parent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
