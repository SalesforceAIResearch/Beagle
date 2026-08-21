#!/usr/bin/env python3
"""Build a patched, shared harbor task cache for terminal-bench-2-1.

Self-contained pipeline — two stages:

    1. POPULATE  materialize a faithful copy of the upstream tasks into
                 ``<dest>/terminal-bench-2-1/`` (registry pull, or copy from
                 an existing harbor export).
    2. PATCH     apply xrlenv's curated pins on top — one-line ``solve.sh``
                 dependency pins (``PATCHES``) *and* per-task
                 ``[environment.env]`` cpuset-pinning markers (``ENV_PATCHES``)
                 for oracles whose parallelism scales to ``nproc``.

Run both in one shot (``--stage all``, the default: populate only if the
dataset is missing, then patch), or run either stage alone (populate on a
network box, patch anywhere). Both stages are idempotent.

Why this exists
---------------
terminal-bench-2-1 task *oracle solutions* are not hermetic: several
``solution/solve.sh`` scripts run ``pip install -e .`` (or a bare
``pip install <pkg>``) that resolves transitive dependencies live from
PyPI *at solve time*. When upstream publishes a breaking release, a
previously-green oracle silently starts scoring ``0`` — the world drifted
under a frozen task.

First casualty: ``build-cython-ext``. ``planarity 1.0.0`` (published
2026-06-29 22:44 UTC) changed ``networkx_graph()`` so graph nodes no
longer carry the ``pos`` attribute pyknotid 0.5.3's test suite expects.
The oracle's ``pip install -e .`` resolves ``planarity`` unpinned, pulls
``1.0.0``, and the verifier's ``test_reconstructed_space_curve`` dies with
``KeyError: 'pos'`` (reward ``0``) even though the Cython build succeeds.
Pinning ``planarity==0.6`` (the last release before the break) makes the
verifier pass again (``11 passed``).

Second dimension: large-host ``nproc`` blow-ups. harbor honors a task's
``cpus``/``memory`` as a CFS quota + hard memory cap but never sets cpuset,
so ``nproc`` / CPU affinity inside the container reports the *host* core
count. On a big node (e.g. the 192-core dev cluster) oracles that scale to
``nproc`` — ``make -j$(nproc)``, ninja's auto ``-j``, OpenBLAS/OMP thread
pools — fan out to ~host-count workers and OOM under their declared memory
cap (proven: ``install-windows-3.11``'s QEMU build → ``cc: fatal error:
Killed signal terminated program cc1``). ``ENV_PATCHES`` marks each such
task with ``[environment.env] XRLENV_CPU_PINNING = "1"``; the harbor plugin
reads it and sizes the affinity mask to the declared ``cpus`` so ``nproc``
matches the task budget again — with the quota + memory cap still enforced.

The fix belongs in the benchmark content, **not** in xrlenv core — keep
benchmark code faithful, xrlenv only manages containers/images. This
script carries an explicit, auditable set of one-line pins on top of an
otherwise-faithful copy of the upstream tasks.

How xrlenv consumers pick it up
-------------------------------
xrlenv's harbor onboarding resolves a task via ``TaskConfig(path=...)`` →
harbor ``LocalTaskId`` → ``_locate_task_dir()``, which searches under
``$XRLENV_BENCHMARK_CACHE`` (or ``~/.cache/harbor/tasks``) in a flat
(``<root>/<task>/``) or sharded (``<root>/<shard>/<task>/``) layout. A
``LocalTaskId`` is used *as-is* — harbor never re-downloads or overwrites
it — so the pinned ``solve.sh`` we write is authoritative and sticky.
Point every consumer's ``XRLENV_BENCHMARK_CACHE`` at the shared root this
script writes. The dataset lands as a shard subdir
(``<root>/terminal-bench-2-1/<task>/``) so the shard-scan finds every task
unchanged and future datasets sit beside it.

Note: these pins only reach a consumer that resolves the task by local
*path* (the xrlenv onboarding shape). A job that resolves by registry
*source* (``source: terminal-bench/terminal-bench-2-1`` + a ``sha256``
ref) reads harbor's own ``packages/<hash>/`` cache and will NOT see them —
such a job must switch to the path-based source to get the fix.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# The upstream dataset this script curates. ``ORG_NAME`` is the harbor
# registry identifier; ``DATASET_DIR`` is the shard subdir name harbor's
# export mode produces (``name.split('/')[-1]``) and the name every
# consumer's ``_locate_task_dir`` shard-scan will see.
ORG_NAME = "terminal-bench/terminal-bench-2-1"
DATASET_DIR = "terminal-bench-2-1"

DEFAULT_SEED_DIR = Path("~/.cache/harbor/tasks/terminal-bench-2-1").expanduser()


@dataclass(frozen=True)
class SolvePatch:
    """A single, idempotent one-line pin applied to a task's
    ``solution/solve.sh``.

    ``insert_line`` is inserted immediately before the first line matching
    ``anchor``. If ``sentinel`` already matches anywhere in the file the
    task is considered already-patched and left untouched (idempotent). If
    ``anchor`` is not found the patch fails loudly rather than silently
    no-op'ing — a missing anchor means upstream changed the solve script's
    shape and the pin needs re-checking.
    """

    task: str
    reason: str
    anchor: re.Pattern[str]
    insert_line: str
    sentinel: re.Pattern[str]


# ── The declarative pin table ────────────────────────────────────────────────
# Add a row here whenever an oracle-sweep surfaces another unpinned-dep
# drift victim. Keep each pin to the minimal one-liner that restores the
# last-known-good resolution, and cite the trigger in ``reason``.
PATCHES: tuple[SolvePatch, ...] = (
    SolvePatch(
        task="build-cython-ext",
        reason=(
            "planarity 1.0.0 (2026-06-29) drops the networkx-graph 'pos' "
            "attr pyknotid 0.5.3's test_reconstructed_space_curve needs; "
            "pin to last-good 0.6 (verified: 11 passed)."
        ),
        anchor=re.compile(r"^\s*pip install -e \.\s*$"),
        insert_line="pip install 'planarity==0.6'",
        sentinel=re.compile(r"planarity==0\.6"),
    ),
    SolvePatch(
        task="mcmc-sampling-stan",
        reason=(
            "StanHeaders 2.32.10 pulls RcppParallel transitively, which switched "
            "to a cmake build in 6.x; the base image has no cmake, so the rstan "
            "install fails ('RcppParallel requires cmake (>= 3.5); cmake was not "
            "found') and the oracle scores 0 (was 87/88). Install cmake before the "
            "R build."
        ),
        anchor=re.compile(r'^echo "=== DEBUG: Installing R packages'),
        insert_line=(
            "sudo apt-get update -qq && sudo apt-get install -y cmake && "
            "rm -rf /var/lib/apt/lists/*"
        ),
        sentinel=re.compile(r"install -y cmake"),
    ),
)


@dataclass(frozen=True)
class TaskEnvPatch:
    """A per-task ``[environment.env]`` marker inserted into ``task.toml``.

    Used to opt a specific oracle into xrlenv cpuset pinning (nproc-sizing)
    without touching xrlenv core or turning pinning on globally. The harbor
    plugin reads ``XRLENV_CPU_PINNING`` from the task's env at acquire time
    and, when truthy, sizes the container's CPU affinity mask to the declared
    ``cpus`` (so ``nproc`` inside == the task budget) while still applying the
    CFS quota + hard memory cap. ``sentinel`` makes the insert idempotent; a
    missing ``[environment.env]`` header fails loudly (upstream task.toml
    shape changed).
    """

    task: str
    reason: str
    env_key: str
    env_value: str
    sentinel: re.Pattern[str]


# The task.toml table the marker lives under (present, usually empty, in every
# terminal-bench-2-1 task.toml — see the populate seed).
_ENV_TABLE_HEADER = re.compile(r"^\[environment\.env\]\s*$")

# ── The cpuset-pinning marker table ───────────────────────────────────────────
# These oracles size their build/compute parallelism to ``nproc``
# (``make -j$(nproc)``, ninja's auto ``-j``, OpenBLAS/OMP thread pools). On a
# large host, harbor applies only a CFS quota + a hard memory cap and never
# sets cpuset, so ``nproc`` inside the container reports the *host* core count,
# not the task's declared ``cpus``. The oracle then fans out to ~host-count
# workers and blows past its memory limit — proven for install-windows-3.11
# (QEMU build: ``cc: fatal error: Killed signal terminated program cc1``). The
# marker opts each into cpuset pinning so ``nproc`` == the declared ``cpus``.
_CPUSET_SENTINEL = re.compile(r"XRLENV_CPU_PINNING\s*=")
ENV_PATCHES: tuple[TaskEnvPatch, ...] = tuple(
    TaskEnvPatch(
        task=task,
        reason=reason,
        env_key="XRLENV_CPU_PINNING",
        env_value="1",
        sentinel=_CPUSET_SENTINEL,
    )
    for task, reason in (
        (
            "install-windows-3.11",
            "make -j$(nproc) QEMU build OOMs (cc1 SIGKILL) at nproc=host "
            "under the declared 4 GiB — confirmed failure.",
        ),
        (
            "caffe-cifar-10",
            "OpenBLAS training threads scale to nproc=host; thread "
            "oversubscription drags it toward its 60m timeout.",
        ),
        (
            "build-pov-ray",
            "make -j$(nproc) build; OOM-risk at nproc=host under 2 GiB.",
        ),
        (
            "rstan-to-pystan",
            "make -j$(nproc) build; OOM-risk at nproc=host under 8 GiB.",
        ),
        (
            "sqlite-with-gcov",
            "make -j$(nproc) build; OOM-risk at nproc=host under 2 GiB.",
        ),
    )
)


# ── Stage 2: patch (pure logic + filesystem application) ──────────────────────


def apply_solve_patch(text: str, patch: SolvePatch) -> tuple[str, str]:
    """Apply ``patch`` to solve-script contents ``text``.

    Returns ``(new_text, status)`` where ``status`` is one of
    ``"patched"``, ``"already"`` (sentinel present — unchanged) or raises
    ``ValueError`` if the anchor line is absent. Pure function — no I/O —
    so the patch logic is unit-testable in isolation.
    """
    if patch.sentinel.search(text):
        return text, "already"

    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if patch.anchor.match(line):
            # Preserve the file's newline convention by mirroring the
            # anchor line's line-ending on the inserted line.
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines.insert(i, patch.insert_line + newline)
            return "".join(lines), "patched"

    raise ValueError(
        f"anchor {patch.anchor.pattern!r} not found in solve.sh for task "
        f"{patch.task!r} — upstream solve script shape changed; re-check "
        f"the pin.",
    )


def apply_task_env_patch(text: str, patch: TaskEnvPatch) -> tuple[str, str]:
    """Insert ``KEY = "VALUE"`` under the ``[environment.env]`` table of a
    task.toml given as ``text``.

    Returns ``(new_text, status)`` where ``status`` is ``"patched"`` or
    ``"already"`` (sentinel present — unchanged), or raises ``ValueError`` if
    the ``[environment.env]`` header is absent (upstream task.toml shape
    changed). Pure function — no I/O — so the logic is unit-testable in
    isolation, mirroring ``apply_solve_patch``.
    """
    if patch.sentinel.search(text):
        return text, "already"

    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if _ENV_TABLE_HEADER.match(line):
            # Insert as the first entry *inside* the table, mirroring the
            # header line's newline convention.
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            entry = f'{patch.env_key} = "{patch.env_value}"'
            lines.insert(i + 1, entry + newline)
            return "".join(lines), "patched"

    raise ValueError(
        f"[environment.env] table not found in task.toml for task "
        f"{patch.task!r} — upstream task.toml shape changed; re-check the "
        f"cpuset-pinning marker.",
    )


def patch_cache(dataset_root: Path) -> list[tuple[str, str]]:
    """Apply every ``PATCHES`` (solve.sh dependency pins) and ``ENV_PATCHES``
    (task.toml cpuset-pinning markers) row to ``dataset_root``. Returns a list
    of ``(label, status)`` for reporting. Idempotent."""
    results: list[tuple[str, str]] = []
    for patch in PATCHES:
        solve = dataset_root / patch.task / "solution" / "solve.sh"
        if not solve.is_file():
            raise SystemExit(
                f"patch target missing: {solve} — is the cache populated? "
                f"(task {patch.task!r} not found under {dataset_root}). Run "
                f"the populate stage first.",
            )
        text = solve.read_text()
        new_text, status = apply_solve_patch(text, patch)
        if status == "patched":
            solve.write_text(new_text)
        # Post-condition: sentinel must now be present either way.
        assert patch.sentinel.search(solve.read_text()), (
            f"post-patch verification failed for {patch.task}: sentinel "
            f"{patch.sentinel.pattern!r} not present"
        )
        results.append((patch.task, status))
    for env_patch in ENV_PATCHES:
        task_toml = dataset_root / env_patch.task / "task.toml"
        if not task_toml.is_file():
            raise SystemExit(
                f"env-patch target missing: {task_toml} — is the cache "
                f"populated? (task {env_patch.task!r} not found under "
                f"{dataset_root}). Run the populate stage first.",
            )
        text = task_toml.read_text()
        new_text, status = apply_task_env_patch(text, env_patch)
        if status == "patched":
            task_toml.write_text(new_text)
        assert env_patch.sentinel.search(task_toml.read_text()), (
            f"post-patch verification failed for {env_patch.task}: sentinel "
            f"{env_patch.sentinel.pattern!r} not present in task.toml"
        )
        results.append((f"{env_patch.task} (cpuset)", status))
    return results


# ── Stage 1: populate ─────────────────────────────────────────────────────────


def _count_tasks(dataset_root: Path) -> int:
    return len(list(dataset_root.glob("*/solution/solve.sh")))


def is_populated(dataset_root: Path) -> bool:
    """True if ``dataset_root`` already holds at least one task with a
    ``solution/solve.sh`` — the "cache present" signal for --stage all."""
    return _count_tasks(dataset_root) > 0


def populate_from_seed(seed_dir: Path, dataset_root: Path, overwrite: bool) -> int:
    """Copy a populated harbor export from ``seed_dir`` into
    ``dataset_root``. Returns the number of task dirs materialized."""
    if not seed_dir.is_dir():
        raise SystemExit(
            f"--seed-dir {seed_dir} does not exist or is not a directory. "
            f"Populate it (a harbor export of {ORG_NAME}) or use "
            f"--source registry.",
        )
    if is_populated(dataset_root) and not overwrite:
        return _count_tasks(dataset_root)
    print(f">> populate(seed): copying {seed_dir} -> {dataset_root}", file=sys.stderr)
    shutil.copytree(seed_dir, dataset_root, dirs_exist_ok=True)
    return _count_tasks(dataset_root)


def populate_from_registry(dataset_root: Path, overwrite: bool) -> int:
    """Pull the frozen upstream dataset via harbor's own downloader into
    ``dataset_root``. Returns the number of task dirs materialized.

    Delegates to ``harbor.cli.download._download_dataset`` rather than
    re-implementing the registry/storage plumbing — the same faithful path
    the ``harbor download`` CLI uses, minus the CLI's slow startup. harbor
    appends the ``terminal-bench-2-1`` wrapper dir itself, so we pass the
    parent as ``output_dir``.
    """
    import asyncio

    from harbor.cli.download import _download_dataset  # type: ignore[import-untyped]

    if is_populated(dataset_root) and not overwrite:
        return _count_tasks(dataset_root)

    parent = dataset_root.parent
    print(
        f">> populate(registry): harbor export of {ORG_NAME} -> {parent}/ "
        f"(slow; needs network + harbor registry reachability)",
        file=sys.stderr,
    )
    asyncio.run(
        _download_dataset(
            name=ORG_NAME,
            version=None,  # -> @latest
            overwrite=overwrite,
            output_dir=parent,
            registry_url=None,
            registry_path=None,
            export=True,  # -> <parent>/terminal-bench-2-1/<task>/
        ),
    )
    return _count_tasks(dataset_root)


def populate(
    dataset_root: Path, source: str, seed_dir: Path, overwrite: bool,
) -> int:
    if source == "seed":
        return populate_from_seed(seed_dir, dataset_root, overwrite)
    return populate_from_registry(dataset_root, overwrite)


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_cache",
        description=(
            "Materialize terminal-bench-2-1 into a shared harbor cache and "
            "apply xrlenv's curated dependency pins. Stages: populate, patch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=("all", "populate", "patch"),
        default="all",
        help="all (default): populate only if missing, then patch. "
        "populate: materialize the raw dataset only. patch: apply pins to an "
        "already-populated cache only.",
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help="Shared harbor cache ROOT to write into (dataset lands under "
        "<dest>/terminal-bench-2-1/). Defaults to $XRLENV_BENCHMARK_CACHE. "
        "Point every xrlenv consumer's XRLENV_BENCHMARK_CACHE at this path.",
    )
    p.add_argument(
        "--source",
        choices=("registry", "seed"),
        default="registry",
        help="How to populate. registry (default): pull the frozen upstream "
        "dataset via harbor (needs network; canonical). seed: copy from an "
        "existing harbor export at --seed-dir (offline; fast).",
    )
    p.add_argument(
        "--seed-dir",
        type=Path,
        default=DEFAULT_SEED_DIR,
        help=f"Source for --source seed (default {DEFAULT_SEED_DIR}).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-populate even if the dataset dir already looks populated "
        "(default: reuse existing, only re-assert patches).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Hard-reject the retired cache env var/path (renamed 2026-07-31) BEFORE any
    # cache use: a caller still on XRLENV_HARBOR_CACHE / .../xrlenv_harbor_cache
    # would silently populate+patch the wrong (stale/absent) cache. Lazy import to
    # match plugin style (plugin -> xrlenv is allowed; no intra-plugin import).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env(args.dest)
    if not args.dest:
        print(
            "error: no destination — pass --dest or set XRLENV_BENCHMARK_CACHE.",
            file=sys.stderr,
        )
        return 2

    dest_root = Path(args.dest).expanduser()
    dataset_root = dest_root / DATASET_DIR
    dest_root.mkdir(parents=True, exist_ok=True)

    n: int | None = None
    if args.stage in ("all", "populate"):
        n = populate(dataset_root, args.source, args.seed_dir, args.overwrite)
    if args.stage == "populate":
        print(f"\npopulated {n} tasks -> {dataset_root}", file=sys.stderr)
        return 0

    # patch (also reached by --stage all)
    if not is_populated(dataset_root):
        raise SystemExit(
            f"cannot patch: {dataset_root} is not populated. Run "
            f"`--stage populate` first (or `--stage all`).",
        )
    results = patch_cache(dataset_root)

    if n is None:
        n = _count_tasks(dataset_root)
    print(f"\n{n} tasks in {dataset_root}", file=sys.stderr)
    print("patches:", file=sys.stderr)
    for task, status in results:
        mark = "+ patched" if status == "patched" else "= already pinned"
        print(f"  {mark:18s} {task}", file=sys.stderr)
    print(
        f"\nDone. Point consumers at:  export "
        f"XRLENV_BENCHMARK_CACHE={dest_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
