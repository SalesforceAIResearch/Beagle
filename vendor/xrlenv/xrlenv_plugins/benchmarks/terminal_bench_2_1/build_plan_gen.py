"""Build-plan generator for terminal-bench-2-1.

Like the other per-benchmark generators under ``xrlenv_plugins/benchmarks/``,
terminal-bench-2-1 tasks ship a **prebuilt registry image** (Docker Hub,
``alexgshaw/<task>:<tag>``) — the
cluster pulls each task's ``docker_image`` on first acquire — so the plan uses
``context_source: type: registry`` throughout and is about *eager warmup*
(``xrlenv build apply`` lowers each entry to an ``EnsurePresentCommand`` and FFD
bin-packs them across nodes), not building images. This is unlike terminalworld
(``type: local`` Dockerfile builds); it follows the newer terminalworld *layout*
convention only — the generator lives **here**, co-located with the rest of the
terminal-bench-2-1 tooling (``build_cache.py`` / ``run_oracle_sweep.py``).

**Per-task image refs, read from ``task.toml`` — not a single hard-coded tag.**
An earlier tb2.0 generator synthesized every ref as
``alexgshaw/<task>:<tag>``. terminal-bench-2-1 can't: while most tasks are
on ``:20251031``, a handful were rebuilt at newer tags (``:20260403`` /
``:20260430`` as of this writing). So each entry's ``image_ref`` is read from the
task's authoritative ``[environment] docker_image`` in the populated
``terminal-bench-2-1`` shard — the same ref the cluster resolves on acquire, so
the image the plan warms is exactly the image the eval uses. This also keeps the
plug-in faithful (no reinventing the tag the benchmark already declares); a task
with no ``docker_image`` fails loud rather than being silently synthesized.

Sizes are probed from Docker Hub's v2 manifest API at generation time
(``size_hint_source: registry-probe``); ``--no-probe`` falls back to a
conservative heuristic. The ``cluster-reported`` upgrade path via
``xrlenv build calibrate`` refines them post-build.

Usage::

    # whole populated shard -> committed 89-full plan (registry-probe sizes):
    XRLENV_BENCHMARK_CACHE=/path/to/cache \\
    .venv/bin/python -m xrlenv_plugins.benchmarks.terminal_bench_2_1.build_plan_gen \\
        --all --output xrlenv_plugins/benchmarks/terminal_bench_2_1/build_plan_89_full.yaml

    # a subset, to stdout:
    .venv/bin/python -m xrlenv_plugins.benchmarks.terminal_bench_2_1.build_plan_gen \\
        --tasks fix-git,build-pov-ray,overfull-hbox --output -

The committed ``build_plan_89_full.yaml`` next to this generator is the result of
running with ``--all`` against the populated shard; operators can use it directly,
regenerate it, or refine its sizes with ``xrlenv build calibrate``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from xrlenv_plugins.benchmarks._dockerhub_probe import (
    announce_auth_status,
    print_probe_summary,
    probe_image_size,
)

# Shard subdir name == the dataset dir build_cache.py writes (and the name every
# consumer's shard-scan sees). Tasks live at ``<cache>/terminal-bench-2-1/<id>/``.
SHARD = "terminal-bench-2-1"

# Conservative default when the Docker Hub probe fails (network down,
# rate-limited, repo private) or ``--no-probe`` is passed. The calibrate flow
# replaces these with ``cluster-reported`` sizes after the first cluster build.
DEFAULT_SIZE_HINT_BYTES = 1_500_000_000  # 1.5 GiB


def _harbor_cache_root() -> Path:
    explicit = os.environ.get("XRLENV_BENCHMARK_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path("~/.cache/harbor/tasks").expanduser()


def _shard_dir() -> Path:
    return _harbor_cache_root() / SHARD


def _discover_all_tasks(shard_dir: Path) -> list[str]:
    """Every task in the shard that ships a ``solution/solve.sh`` (the same
    predicate the oracle sweep uses to enumerate tasks)."""
    if not shard_dir.is_dir():
        return []
    return sorted(
        p.name for p in shard_dir.iterdir()
        if p.is_dir() and (p / "solution" / "solve.sh").is_file()
    )


def _task_image_ref(shard_dir: Path, task_id: str) -> str:
    """The authoritative ``[environment] docker_image`` from a task's
    ``task.toml``. Fails loud if the task or the field is missing — a task with
    no declared image can't be warmed, and a synthesized fallback would drift
    from the ref the cluster actually resolves on acquire."""
    toml_path = shard_dir / task_id / "task.toml"
    if not toml_path.is_file():
        raise SystemExit(
            f"ERROR: {task_id}: no {toml_path} — is the shard populated? Run "
            f"build_cache.py --stage populate first.",
        )
    env = tomllib.loads(toml_path.read_text()).get("environment", {})
    image_ref = env.get("docker_image")
    if not image_ref:
        raise SystemExit(
            f"ERROR: {task_id}: task.toml has no [environment] docker_image — "
            f"terminal-bench-2-1 tasks are expected to declare a prebuilt image.",
        )
    return str(image_ref)


def _split_repo_tag(image_ref: str) -> tuple[str, str]:
    """Split ``namespace/name:tag`` into ``(repo, tag)`` for the Docker Hub
    probe. Uses the last ``:`` so a registry-host:port prefix (none in the
    upstream alexgshaw refs, but harmless) doesn't confuse the tag split.
    Defaults the tag to ``latest`` if the ref carries none."""
    repo, sep, tag = image_ref.rpartition(":")
    if not sep or "/" in tag:  # no tag (or the ':' was a host:port, not a tag)
        return image_ref, "latest"
    return repo, tag


def generate_plan(
    tasks: list[str],
    *,
    shard_dir: Path | None = None,
    preferred_home_count: int = 1,
    pinned: bool = False,
    probe_sizes: bool = True,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
) -> dict[str, Any]:
    """Build a YAML-shaped ``BuildPlan`` dict.

    Returned shape is the per-image-ref schema (``entries: [...]``) documented in
    :mod:`xrlenv.control.build_plan`. Every entry uses
    ``context_source: { type: registry }`` and carries the task's authoritative
    ``docker_image`` ref (read from its ``task.toml``), so mixed upstream tags are
    handled without a hard-coded default. Raises ``SystemExit`` if a requested
    task is missing or has no ``docker_image``.
    """
    shard = shard_dir if shard_dir is not None else _shard_dir()
    entries: list[dict[str, Any]] = []
    if probe_sizes:
        announce_auth_status()
    for task_id in tasks:
        image_ref = _task_image_ref(shard, task_id)
        repo, tag = _split_repo_tag(image_ref)
        size_bytes: int | None = None
        size_source = "heuristic"
        if probe_sizes:
            size_bytes = probe_image_size(repo, tag)
            if size_bytes is not None:
                size_source = "registry-probe"
        if size_bytes is None:
            size_bytes = DEFAULT_SIZE_HINT_BYTES
        entries.append({
            "image_ref": image_ref,
            "context_source": {"type": "registry"},
            "placement": {
                "preferred_home_count": preferred_home_count,
                "size_hint_bytes": size_bytes,
                "size_hint_source": size_source,
            },
            "pinned": pinned,
            "priority": 0,
        })
    if probe_sizes:
        print_probe_summary(DEFAULT_SIZE_HINT_BYTES)
    return {
        "version": 1,
        "name": f"terminal-bench-2-1-{len(tasks)}-task",
        "replication": 1,
        "budget": {
            "reserved_runtime_gb": reserved_runtime_gb,
            "buffer_gb": buffer_gb,
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env from cwd / ancestors so operators who set DOCKERHUB_USER +
    # DOCKERHUB_TOKEN there get authenticated probes without needing to
    # shell-export. Stdlib-only helper; the generator never imports ``xrlenv`` at
    # module load so the package-level import hook doesn't fire.
    from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    _maybe_auto_load_dotenv()

    p = argparse.ArgumentParser(
        prog="terminal-bench-2-1-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for terminal-bench-2-1. Every task has a "
            "prebuilt registry image (read per-task from task.toml's "
            "[environment] docker_image); the plan uses type: registry entries "
            "throughout, for eager warmup via `xrlenv build apply`."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="Every task in the populated shard.")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated explicit task ids.")
    p.add_argument("--preferred-home-count", type=int, default=1,
                   help="Per-image preferred_home_count (default 1).")
    p.add_argument("--pinned", action="store_true",
                   help="Pin entries so eviction skips them.")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip Docker Hub size probe; use the conservative "
                        "default hint (1.5 GiB) for every entry.")
    p.add_argument("--output", default="-",
                   help="Output path (default '-' for stdout).")
    args = p.parse_args(argv)

    # Hard-reject the retired cache env var/path (renamed 2026-07-31) BEFORE the
    # shard is resolved from $XRLENV_BENCHMARK_CACHE: a caller still on
    # XRLENV_HARBOR_CACHE / .../xrlenv_harbor_cache would warm images off the wrong
    # (stale/absent) cache. No --dest flag here, so pass none. Lazy import to match
    # plugin style (plugin -> xrlenv is allowed).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env()

    shard = _shard_dir()
    if args.all:
        tasks = _discover_all_tasks(shard)
        if not tasks:
            raise SystemExit(
                f"no tasks with solution/solve.sh under {shard} — populate the "
                f"shard first (build_cache.py --stage all).",
            )
    else:
        tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]

    plan = generate_plan(
        tasks,
        shard_dir=shard,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        probe_sizes=not args.no_probe,
    )

    import yaml
    text = yaml.safe_dump(plan, sort_keys=False, default_flow_style=False)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
        print(f"wrote {len(plan['entries'])} entries -> {args.output}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
