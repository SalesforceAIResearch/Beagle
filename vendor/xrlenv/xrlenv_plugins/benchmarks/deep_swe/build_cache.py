#!/usr/bin/env python3
"""Build a shared harbor/pier task cache for the DeepSWE benchmark.

Self-contained pipeline, mirroring ``terminalworld/build_cache.py`` in shape
(``--stage`` driven, idempotent), retargeted at DeepSWE — a **pure harbor/pier-format
task corpus** (``datacurve-ai/deep-swe``) whose 113 tasks live directly in-git under
``tasks/<id>/`` (mirror on the HF hub ``datacurve/deep-swe``). Each task unpacks to
``task.toml`` + ``instruction.md`` + ``environment/Dockerfile`` +
``tests/{test.sh,test.patch,grader.py,config.json,Dockerfile}`` + ``solution/`` +
``pre_artifacts.sh`` — the exact contract pier's ``TaskConfig(path=…)`` consumes.

Stages
------
    populate  Materialize the 113 task dirs into ``<dest>/deep-swe/<id>/`` and
              normalize each ``task.toml``. Two sources (``--source``):
                * ``git`` (default): shallow-clone ``datacurve-ai/deep-swe`` and
                  copy ``tasks/<id>/`` into the shard. Needs network + ``git``.
                * ``hf``: snapshot ``datacurve/deep-swe`` from the HF hub. Needs
                  network + ``huggingface_hub``.
    patch     Overlay curated ``patches/<id>/`` full-file fixes onto the extracted
              tasks (start empty — DeepSWE grades behaviorally against baked tests,
              so unpinned-dep drift risk is lower than tb2.1's live-pip oracles; the
              hook stays for when the oracle sweep surfaces broken content).
    all       populate (only if missing) then patch. The default.

Image cache (separate from this task-dir cache)
-----------------------------------------------
DeepSWE ships **prebuilt per-task images on public ECR**
(``[environment] docker_image = public.ecr.aws/d3j8x8q7/swe-bench-202605:<ext_id>-v1.1``)
— the tb2.1 shape. This script populates only the *task-dir* cache; the images are
warmed separately via ``build_plan_gen.py`` (a ``type: registry`` plan) +
``xrlenv build apply``. Do NOT re-push them into the private registry.

How xrlenv consumers pick it up
-------------------------------
pier resolves a task by local *path* (``TaskConfig(path=…)``), used as-is, so the
normalized ``task.toml`` this script writes is authoritative. Point every consumer's
``XRLENV_BENCHMARK_CACHE`` at the shared root this script writes; tasks land at
``<root>/deep-swe/<id>/`` beside the tb2.1 / terminalworld shards.

Env overrides (populate):
    DEEPSWE_REPO_URL   git remote        (default https://github.com/datacurve-ai/deep-swe)
    DEEPSWE_REPO_REF   git ref/branch    (default main)
    DEEPSWE_HF_REPO    HF dataset id      (default datacurve/deep-swe)
    DEEPSWE_SHARD      cache shard name  (default deep-swe)
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
REPO_URL = os.environ.get("DEEPSWE_REPO_URL", "https://github.com/datacurve-ai/deep-swe")
REPO_REF = os.environ.get("DEEPSWE_REPO_REF", "main")
HF_REPO = os.environ.get("DEEPSWE_HF_REPO", "datacurve/deep-swe")
# Shard subdir name == image namespace (one name, double duty — same convention as
# the terminalworld shard). The name every consumer's shard-scan sees.
SHARD = os.environ.get("DEEPSWE_SHARD", "deep-swe")

# Curated per-task fixes live beside this script. Each ``patches/<id>/<rel>`` file
# is overlaid (full-file replacement) onto the extracted task dir AFTER
# extraction+normalization, so the fixes survive re-populate. See patches/README.md.
PATCHES_DIR = Path(__file__).resolve().parent / "patches"

# A task dir is valid iff it carries a task.toml (the pier-format anchor).
_TASK_ANCHOR = "task.toml"


# ── task.toml normalization (pure) ────────────────────────────────────────────


def normalize_task_toml_text(text: str) -> tuple[str, bool]:
    """Strip the harbor/pier-deprecated ``memory`` / ``storage`` string keys from a
    task.toml's ``[environment]`` (and ``[verifier.environment]``) section when the
    canonical ``memory_mb`` / ``storage_mb`` are also present. Returns
    ``(new_text, changed)``.

    pier's ``EnvironmentConfig`` (like harbor's) carries BOTH the deprecated string
    field and the canonical ``_mb`` integer; if a task sets them inconsistently the
    model rejects the conflict and the task won't load. The least-lossy fix is to
    drop the deprecated duplicate and keep the canonical ``_mb`` field. Surgical
    (line-level) so the rest of the file is byte-preserved. Pure (no I/O) so the
    logic is unit-testable in isolation. A no-op when the conflict isn't present
    (DeepSWE mostly uses ``_mb`` only — kept defensively).
    """
    parsed = tomllib.loads(text)

    def _conflicts(section: dict) -> set[str]:
        return {
            legacy
            for legacy, canonical in (("memory", "memory_mb"), ("storage", "storage_mb"))
            if legacy in section and canonical in section
        }

    env = parsed.get("environment", {})
    ver_env = (parsed.get("verifier", {}) or {}).get("environment", {})
    strip_env = _conflicts(env if isinstance(env, dict) else {})
    strip_ver = _conflicts(ver_env if isinstance(ver_env, dict) else {})
    if not strip_env and not strip_ver:
        return text, False

    out: list[str] = []
    section: str | None = None
    for line in text.splitlines(keepends=True):
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            section = header.group(1).strip()
        strip = (
            strip_env if section == "environment"
            else strip_ver if section == "verifier.environment"
            else set()
        )
        if strip:
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
    """Copy every ``<tasks_root>/<id>/`` that carries a ``task.toml`` into
    ``<shard_dir>/<id>/`` (idempotent — skip a task already present), normalizing
    each ``task.toml``. Returns ``(moved, normalized)``."""
    if not tasks_root.is_dir():
        raise SystemExit(
            f"ERROR: expected a 'tasks/' directory at {tasks_root} — is this the "
            f"deep-swe repo layout? (tasks/<id>/task.toml)",
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
        if i % 25 == 0 or i == len(task_dirs):
            print(f"   [{i}/{len(task_dirs)}] {src.name}", file=sys.stderr)
    return moved, normalized


def populate_git(shard_dir: Path) -> tuple[int, int]:
    """Shallow-clone the deep-swe repo and copy its ``tasks/`` into the shard."""
    with tempfile.TemporaryDirectory(prefix="deepswe-clone-") as tmp:
        print(
            f">> git clone --depth 1 --branch {REPO_REF} {REPO_URL}",
            file=sys.stderr,
        )
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, tmp],
            check=True,
        )
        return _copy_tasks(Path(tmp) / "tasks", shard_dir)


def populate_hf(shard_dir: Path) -> tuple[int, int]:
    """Snapshot the HF-hub mirror ``datacurve/deep-swe`` and copy its ``tasks/``."""
    from huggingface_hub import snapshot_download

    print(f">> huggingface snapshot_download({HF_REPO}, repo_type=dataset)", file=sys.stderr)
    local = snapshot_download(repo_id=HF_REPO, repo_type="dataset")
    return _copy_tasks(Path(local) / "tasks", shard_dir)


def populate(shard_dir: Path, source: str) -> tuple[int, int]:
    if source == "hf":
        return populate_hf(shard_dir)
    return populate_git(shard_dir)


# ── Stage 2: patch (curated full-file overlays) ───────────────────────────────


def _apply_patch(task_dir: Path, patch_dir: Path) -> list[str]:
    """Overlay every file under ``patch_dir`` onto ``task_dir`` (full-file
    replacement), preserving relative layout + exec bits. Returns the relative
    paths overridden. A patch file with no counterpart is still copied (lets a
    patch ADD a missing file)."""
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
    """Apply every ``patches/<id>/`` overlay to its present task in the shard.
    Idempotent; a task absent from the shard is skipped with a SKIP note. Returns
    the number of tasks patched (0 while ``patches/`` is empty)."""
    if not PATCHES_DIR.is_dir():
        return 0
    patched = 0
    for patch_dir in sorted(PATCHES_DIR.iterdir()):
        if not patch_dir.is_dir():
            continue  # skip README.md etc.
        tid = patch_dir.name
        dest = shard_dir / tid
        if not (dest / _TASK_ANCHOR).is_file():
            print(
                f"   [patch] {tid}: SKIP — not present in shard (populate first)",
                file=sys.stderr,
            )
            continue
        files = _apply_patch(dest, patch_dir)
        if files:
            patched += 1
            print(f"   [patch] {tid}: overrode {files}", file=sys.stderr)
    return patched


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_tasks(shard_dir: Path) -> int:
    return sum(1 for _ in shard_dir.glob(f"*/{_TASK_ANCHOR}"))


def is_populated(shard_dir: Path) -> bool:
    return shard_dir.is_dir() and _count_tasks(shard_dir) > 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_cache",
        description=(
            "Materialize the DeepSWE task corpus into a shared harbor/pier cache. "
            "Stages (each idempotent): populate -> patch; `all` (default) runs both."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=("all", "populate", "patch"),
        default="all",
        help="all (default): populate (if missing) + patch. populate: "
        "download+normalize only (needs network). patch: curated overlays only.",
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help="Shared cache ROOT (the shard lands under "
        f"<dest>/{SHARD}/). Defaults to $XRLENV_BENCHMARK_CACHE. Point every xrlenv "
        "consumer's XRLENV_BENCHMARK_CACHE at this path.",
    )
    p.add_argument(
        "--source",
        choices=("git", "hf"),
        default="git",
        help="How to populate. git (default): shallow-clone the deep-swe repo. "
        "hf: snapshot the datacurve/deep-swe HF mirror.",
    )
    return p


def _resolve_shard(dest: str | None) -> Path:
    # Hard-reject the retired cache env var/path first (renamed 2026-07-31:
    # XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, xrlenv_harbor_cache -> ...
    # xrlenv_benchmark_cache). Reusing the old var/path reads stale/absent data and
    # yields unreliable results, so fail loud before resolving a shard from it. Lazy
    # import to match the plugin style (plugin -> xrlenv core).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env(dest)
    if not dest:
        raise SystemExit(
            "error: no destination — pass --dest or set XRLENV_BENCHMARK_CACHE.",
        )
    return Path(dest).expanduser() / SHARD


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_dir = _resolve_shard(args.dest)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)

    moved = normalized = 0
    if args.stage in ("all", "populate"):
        if args.stage == "populate" or not is_populated(shard_dir):
            moved, normalized = populate(shard_dir, args.source)
        else:
            print(
                f">> {shard_dir} already populated "
                f"({_count_tasks(shard_dir)} tasks) — skipping download.",
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
    total = _count_tasks(shard_dir)
    print(
        f"\nOK: {total} task(s) in {shard_dir}"
        + (f" ({moved} onboarded, {normalized} normalized)" if moved else "")
        + f"; applied {patched} curated patch(es).",
        file=sys.stderr,
    )
    print(
        f"Point consumers at:  export XRLENV_BENCHMARK_CACHE={shard_dir.parent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
