"""Build-plan generator for terminal-bench-2.

Emits a per-image-ref ``build-plan.yaml`` with one ``BuildEntry`` per
task in the harbor cache. All 89 phase-0 tasks ship a prebuilt
``alexgshaw/<task>:20251031`` image on Docker Hub; the generated
plan uses ``context_source: type: registry`` for each task.
``xrlenv build apply --plan <yaml>`` lowers these entries to
``EnsurePresentCommand``s and FFD bin-packs them across nodes for
eager warmup (registry-source dispatch landed in B.1.next).

Sizes are probed from Docker Hub's v2 manifest API at generation
time (``size_hint_source: registry-probe``). The
``cluster-reported`` upgrade path via ``xrlenv build calibrate`` is
planned for B.1.next.b; until then sizes stay registry-probe.

Usage::

    .venv/bin/python -m xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen \\
        --tasks fix-git,build-pov-ray,overfull-hbox \\
        --output build_plan.yaml

    .venv/bin/python -m xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen \\
        --all --output build_plan.yaml

The committed ``build_plan.yaml`` next to this generator is the
result of running with ``--all``; operators can use it directly or
regenerate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from xrlenv_plugins.images_build._dockerhub_probe import (
    announce_auth_status,
    print_probe_summary,
    probe_image_size,
)

# Phase-0 acceptance task list. Mirrors the hard-coded set in
# ``examples/benchmarks-onboarding/terminal-bench-2/smoke.py::SMOKE_TASKS``;
# the list is committed in two places so an operator can drive the
# generator without standing up the smoke driver.
PHASE_0_TASKS: tuple[str, ...] = (
    "fix-git",
    "build-pov-ray",
    "overfull-hbox",
    "cobol-modernization",
    "prove-plus-comm",
    "constraints-scheduling",
    "nginx-request-logging",
    "dna-insert",
)

# Default Docker Hub namespace + tag for the upstream-published
# task images. Override via ``--namespace`` / ``--tag`` if a fork
# pushes to a different registry.
DEFAULT_NAMESPACE = "alexgshaw"
DEFAULT_TAG = "20251031"

# Conservative default when the Docker Hub probe fails (network
# down, rate-limited, repo private). The B.1.next.b calibrate flow
# will replace these with ``cluster-reported`` sizes after the
# first cluster build.
DEFAULT_SIZE_HINT_BYTES = 1_500_000_000  # 1.5 GiB


def _harbor_cache_root() -> Path:
    # Fail loud on the RETIRED XRLENV_HARBOR_CACHE var / .../xrlenv_harbor_cache path (audit
    # M17) — otherwise a user who still exports only the legacy var has it silently ignored
    # here, falls through to the default, and emits the 8-task smoke plan as if it were --all.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env()
    explicit = os.environ.get("XRLENV_BENCHMARK_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    # Phase-0 example default (where populate-harbor-cache.sh clones the tb2 catalog); the
    # golden-path benchmarks default to the shared FSx root instead.
    return Path("~/.cache/harbor/tasks").expanduser()


# terminal-bench-2 is ~89 tasks; cap discovery well above that but far below a shared
# multi-benchmark cache (~1700+). Exceeding it means the cache root is almost certainly a
# SHARED cache holding other benchmarks' shards, not a terminal-bench-2-only clone (audit M17).
_TB2_MAX_PLAUSIBLE_TASKS = 300


def _discover_all_tasks() -> list[str]:
    """Walk the harbor cache and return every task that has a
    ``solution/solve.sh`` (the same predicate the onboarding smoke
    uses). ``--all`` FAILS LOUD on an absent/empty cache (audit M17) —
    silently emitting the 8-task smoke plan as if it were the full
    corpus is a footgun; use ``--smoke`` for that subset. It ALSO fails
    loud if it discovers implausibly many tasks (corpus-scoping, audit
    M17): the terminal-bench-2 example expects a tb2-only cache, and a
    huge count means it's pointed at the SHARED multi-benchmark cache."""
    cache_root = _harbor_cache_root()
    if not cache_root.is_dir():
        raise SystemExit(
            f"--all: harbor cache not found at {cache_root}. Populate it "
            f"(scripts/populate-harbor-cache.sh) or set XRLENV_BENCHMARK_CACHE. "
            f"Refusing to silently emit the 8-task smoke plan — use --smoke for that."
        )
    seen: set[str] = set()
    for top in sorted(cache_root.iterdir()):
        if not top.is_dir():
            continue
        if (top / "solution" / "solve.sh").is_file():
            seen.add(top.name)
            continue
        for inner in sorted(top.iterdir()):
            if inner.is_dir() and (inner / "solution" / "solve.sh").is_file():
                seen.add(inner.name)
    if not seen:
        raise SystemExit(
            f"--all: no tasks with solution/solve.sh under {cache_root}. Populate the "
            f"cache or use --smoke for the 8-task subset."
        )
    if len(seen) > _TB2_MAX_PLAUSIBLE_TASKS:
        raise SystemExit(
            f"--all: discovered {len(seen)} tasks under {cache_root} — far more than the "
            f"~89-task terminal-bench-2 corpus. This is almost certainly the SHARED "
            f"multi-benchmark cache, not a terminal-bench-2-only clone. Point "
            f"XRLENV_BENCHMARK_CACHE at a tb2-only cache (scripts/populate-harbor-cache.sh) "
            f"or pass an explicit --tasks list."
        )
    return sorted(seen)


def generate_plan(
    tasks: list[str],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    tag: str = DEFAULT_TAG,
    preferred_home_count: int = 1,
    pinned: bool = False,
    probe_sizes: bool = True,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
) -> dict[str, Any]:
    """Build a YAML-shaped ``BuildPlan`` dict.

    Returned shape is the per-image-ref schema (``entries: [...]``)
    documented in :mod:`xrlenv.control.build_plan`. Every entry
    uses ``context_source: { type: registry }`` since the upstream
    catalog publishes prebuilt images.
    """
    entries: list[dict[str, Any]] = []
    if probe_sizes:
        announce_auth_status()
    for task_id in tasks:
        repo = f"{namespace}/{task_id}"
        image_ref = f"{repo}:{tag}"
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
    suffix = (
        "smoke-8" if list(tasks) == list(PHASE_0_TASKS)
        else f"{len(tasks)}-task"
    )
    return {
        "version": 1,
        "name": f"terminal-bench-2-{suffix}",
        "replication": 1,
        "budget": {
            "reserved_runtime_gb": reserved_runtime_gb,
            "buffer_gb": buffer_gb,
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env from cwd / ancestors so operators who set
    # DOCKERHUB_USER + DOCKERHUB_TOKEN there get authenticated probes
    # without needing to shell-export. The generator never imports
    # ``xrlenv``, so the package-level import hook doesn't fire
    # (caught in 2026-05-11 audit). Stdlib-only helper module;
    # no docker/grpc/prometheus fan-out.
    from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    _maybe_auto_load_dotenv()

    p = argparse.ArgumentParser(
        prog="terminal-bench-2-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for terminal-bench-2. All "
            "tasks have prebuilt registry images; the plan uses "
            "type: registry entries throughout."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="Every task in the harbor cache.")
    sel.add_argument("--smoke", action="store_true",
                     help="The 8 phase-0 acceptance tasks.")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated explicit task ids.")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                   help=f"Docker Hub namespace (default {DEFAULT_NAMESPACE}).")
    p.add_argument("--tag", default=DEFAULT_TAG,
                   help=f"Image tag (default {DEFAULT_TAG}).")
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

    # audit Low: reject the retired cache var/path in EVERY mode, not just --all (which reaches
    # it via _harbor_cache_root). --smoke / --tasks don't resolve the cache, so guard here too.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env()

    if args.all:
        tasks = _discover_all_tasks()
    elif args.smoke:
        tasks = list(PHASE_0_TASKS)
    elif args.tasks:
        tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    else:
        # Argparse mutex group guarantees this is unreachable.
        return 2

    plan = generate_plan(
        tasks,
        namespace=args.namespace,
        tag=args.tag,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
