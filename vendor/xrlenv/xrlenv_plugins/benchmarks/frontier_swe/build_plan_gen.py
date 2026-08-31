"""Build-plan generator for FrontierSWE (image warm plan).

Like ``deep_swe/build_plan_gen.py``, FrontierSWE tasks ship a **prebuilt registry
image** — per task, ``[environment] docker_image =
ghcr.io/proximal-labs/frontier-swe/<id>:<tag>`` — so the cluster pulls each task's
image (nothing needs *building* on our side). The plan therefore uses
``context_source: {type: registry}`` throughout and is about **eager warmup**
(``xrlenv build apply`` lowers each entry to an ``EnsurePresentCommand`` → node-side
``docker pull`` → FFD bin-packed across nodes), not building images.

**Per-task refs are read from ``task.toml``** (the authoritative ``docker_image``,
the same ref the cluster resolves on acquire) — never synthesized — so a task with
no ``docker_image`` fails loud rather than warming the wrong image.

**Direct GHCR (public, anonymous-pullable) is the fetch path.** ``xrlenv build
apply --fill-missing`` node-side ``docker pull ghcr.io/...`` needs no new infra
(the deep_swe public-registry decision). Do NOT push these into the private
registry.

**Sizes: probe OFF by default.** The shared size probe
(``benchmarks/_dockerhub_probe``) targets **Docker Hub**, not GHCR, so it can't
size these refs. We default to a conservative hint and refine to true on-disk
``cluster-reported`` sizes via ``xrlenv build calibrate`` after the first warm.
``--probe`` opts into the (best-effort, likely-miss) Docker Hub probe anyway.

**Committed plan = the green subset, not ``--all``.** FrontierSWE's committed
``frontier_swe_build_plan.yaml`` warms only the **oracle-gateable CPU green set**
(the tasks the sweep actually pulls). The other corpus tasks are either GPU (their
multi-GB CUDA images are useless on a CPU cluster) or solution-withheld (never
oracle-run), so warming them wastes fleet disk. ``--all`` still emits the full
17-task corpus if you need it; ``--tasks`` selects any subset (the wrapper's
``--list-green`` prints the green set to feed it).

Usage::

    # green subset -> committed plan (piped from the wrapper's green set):
    XRLENV_BENCHMARK_CACHE=/path/to/cache \\
    .venv/bin/python -m xrlenv_plugins.benchmarks.frontier_swe.build_plan_gen \\
        --tasks "$(bash .../frontier_swe/run_full_sweep.sh --list-green | paste -sd,)" \\
        --output xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.yaml

    # whole populated shard (all 17), to stdout:
    .venv/bin/python -m xrlenv_plugins.benchmarks.frontier_swe.build_plan_gen \\
        --all --output -
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

# Shard subdir name == the dataset dir build_cache.py writes. Tasks live at
# ``<cache>/frontier-swe/<id>/``.
SHARD = "frontier-swe"

# Conservative default when probing is off/failed. FrontierSWE images span slim
# language toolchains up to full CUDA/ML stacks; 4 GiB is a middle-of-the-road
# placeholder — the calibrate flow replaces these with cluster-reported sizes
# after the first warm.
DEFAULT_SIZE_HINT_BYTES = 4_000_000_000  # 4 GiB


def _harbor_cache_root() -> Path:
    # Hard-reject the retired cache env var/path before reading it (renamed
    # 2026-07-31: XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, xrlenv_harbor_cache
    # -> xrlenv_benchmark_cache). The old var/path reads stale/absent data, so a plan
    # generated against it would warm the wrong images — fail loud instead. Lazy
    # import to match the plugin style (plugin -> xrlenv core).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env()
    explicit = os.environ.get("XRLENV_BENCHMARK_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path("~/.cache/harbor/tasks").expanduser()


def _shard_dir() -> Path:
    return _harbor_cache_root() / SHARD


def _discover_all_tasks(shard_dir: Path) -> list[str]:
    """Every task in the shard that ships a ``task.toml`` (the harbor-format
    anchor)."""
    if not shard_dir.is_dir():
        return []
    return sorted(
        p.name for p in shard_dir.iterdir()
        if p.is_dir() and (p / "task.toml").is_file()
    )


def _task_image_ref(shard_dir: Path, task_id: str) -> str:
    """The authoritative ``[environment] docker_image`` from a task's ``task.toml``.
    Fails loud if the task or the field is missing — a task with no declared image
    can't be warmed, and a synthesized fallback would drift from the ref the cluster
    resolves on acquire."""
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
            f"FrontierSWE tasks are expected to declare a prebuilt GHCR image.",
        )
    return str(image_ref)


def _split_repo_tag(image_ref: str) -> tuple[str, str]:
    """Split ``host/namespace/name:tag`` into ``(repo, tag)`` on the LAST ``:`` so a
    registry ``host:port`` prefix doesn't confuse the tag split. Defaults the tag to
    ``latest`` if the ref carries none."""
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
    probe_sizes: bool = False,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
) -> dict[str, Any]:
    """Build a YAML-shaped ``BuildPlan`` dict (``entries: [...]``), every entry
    ``context_source: {type: registry}`` carrying the task's authoritative GHCR
    ``docker_image``. Raises ``SystemExit`` if a requested task is missing or has no
    ``docker_image``."""
    shard = shard_dir if shard_dir is not None else _shard_dir()
    entries: list[dict[str, Any]] = []
    probe = None
    print_summary = None
    if probe_sizes:
        # Best-effort Docker Hub probe (will miss for ghcr.io refs; falls to the
        # default hint). Imported lazily so the generator stays import-light.
        from xrlenv_plugins.benchmarks._dockerhub_probe import (
            announce_auth_status,
            print_probe_summary,
            probe_image_size,
        )

        announce_auth_status()
        probe = probe_image_size
        print_summary = print_probe_summary
    for task_id in tasks:
        image_ref = _task_image_ref(shard, task_id)
        size_bytes: int | None = None
        size_source = "heuristic"
        if probe is not None:
            repo, tag = _split_repo_tag(image_ref)
            size_bytes = probe(repo, tag)
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
    if print_summary is not None:
        print_summary(DEFAULT_SIZE_HINT_BYTES)
    return {
        "version": 1,
        "name": f"frontier-swe-{len(tasks)}-task",
        "replication": 1,
        "budget": {
            "reserved_runtime_gb": reserved_runtime_gb,
            "buffer_gb": buffer_gb,
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="frontier-swe-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for FrontierSWE. Every task has a prebuilt "
            "GHCR image (read per-task from task.toml's [environment] docker_image); "
            "the plan uses type: registry entries throughout, for eager warmup via "
            "`xrlenv build apply`."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="Every task in the populated shard (full 17-task corpus).")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated explicit task ids (e.g. the green set).")
    p.add_argument("--preferred-home-count", type=int, default=1,
                   help="Per-image preferred_home_count (default 1).")
    p.add_argument("--pinned", action="store_true",
                   help="Pin entries so eviction skips them.")
    p.add_argument("--probe", action="store_true",
                   help="Opt into the (best-effort, Docker-Hub-only) size probe. "
                        "Off by default — FrontierSWE images are on public GHCR, "
                        "which the probe can't size; use `xrlenv build calibrate` "
                        "after the first warm instead.")
    p.add_argument("--output", default="-",
                   help="Output path (default '-' for stdout).")
    args = p.parse_args(argv)

    shard = _shard_dir()
    if args.all:
        tasks = _discover_all_tasks(shard)
        if not tasks:
            raise SystemExit(
                f"no tasks with task.toml under {shard} — populate the shard "
                f"first (build_cache.py --stage all).",
            )
    else:
        tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]

    plan = generate_plan(
        tasks,
        shard_dir=shard,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        probe_sizes=args.probe,
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
