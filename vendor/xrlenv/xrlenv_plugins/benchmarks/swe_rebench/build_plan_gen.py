"""Build-plan generator for SWE-rebench (image warm plan).

SWE-rebench ships a **prebuilt Docker Hub image per task**
(``swerebench/sweb.eval.x86_64.<slug>:latest``), which ``build_cache.py --stage
repin`` writes into each task's ``[environment] docker_image``. So the default
plan is ``context_source: {type: registry}`` throughout and is about **eager
warmup** (``xrlenv build apply`` lowers each entry to an ``EnsurePresentCommand``
-> node-side ``docker pull`` -> FFD bin-packed across nodes), not building.

**Per-task refs are read from ``task.toml``** (the authoritative
``docker_image``, the same ref the cluster resolves on acquire) — never
synthesized — so a task with no ``docker_image`` fails loud with a pointer at
the repin stage rather than warming the wrong image.

**Nothing is ever built.** Every task pulls its upstream prebuilt, so the plan
is 100 % ``type: registry`` and the committed YAML carries no registry host at
all (GUIDELINE §5.3.1).

**Sizes: probe ON by default, and RESUMABLE.** The shared size probe
(``benchmarks/_dockerhub_probe``) targets Docker Hub, which is exactly where
these images live, so it returns real compressed sizes for the FFD bin-packer.
Set ``DOCKERHUB_USER`` / ``DOCKERHUB_TOKEN`` (see ``.env``) before an ``--all``
run.

Even authenticated, Docker Hub rate-limits an 860-image sweep partway through
(``HTTP 429 {"detail": "Rate limit exceeded"}`` at roughly 600 probes on a
personal-tier account), and every 429 silently falls back to the conservative
heuristic — which over-reserves disk in FFD and can reject an otherwise-fine
plan at apply time. So ``--reuse-sizes <plan.yaml>`` seeds already-measured
``registry-probe`` sizes from a previous plan and probes **only** the entries
still on the heuristic. Re-run it after the rate window resets (~6 h) until the
generator reports 0 fallbacks; each pass is idempotent for the refs it already
knows. ``--no-probe`` skips probing entirely; ``xrlenv build calibrate`` refines
to true on-disk ``cluster-reported`` sizes after the first warm.

Usage::

    # full corpus -> the committed plan:
    XRLENV_BENCHMARK_CACHE=/path/to/cache \\
    .venv/bin/python -m xrlenv_plugins.benchmarks.swe_rebench.build_plan_gen \\
        --all --output xrlenv_plugins/benchmarks/swe_rebench/swe_rebench_build_plan.yaml

    # a subset (e.g. the wrapper's green set), to stdout:
    .venv/bin/python -m xrlenv_plugins.benchmarks.swe_rebench.build_plan_gen \\
        --tasks "$(bash .../swe_rebench/run_full_sweep.sh --list-green | paste -sd,)" \\
        --output -
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Shard subdir name — must match build_cache.py's SHARD. Tasks live at
# ``<cache>/swe-rebench/<task_id>/``.
SHARD = "swe-rebench"

# Conservative default when probing is off / a probe failed. SWE-rebench images
# are SWE-bench-shaped (base + conda env + instance layers); a 60-image sample
# averaged ~2.3 GB compressed, so 3 GiB is a safe middle-of-distribution
# placeholder. `xrlenv build calibrate` replaces these with cluster-reported
# on-disk sizes after the first warm.
DEFAULT_SIZE_HINT_BYTES = 3_000_000_000  # 3 GiB


def _harbor_cache_root() -> Path:
    """The cache ROOT, on exactly the same contract ``build_cache.py`` uses.

    ``benchmark_cache_root`` hard-rejects the retired var/path (renamed 2026-07-31:
    XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, xrlenv_harbor_cache ->
    xrlenv_benchmark_cache) and raises when nothing is set. Deferring to it rather than
    re-implementing the lookup is the point: this generator previously fell back to
    ``~/.cache/harbor/tasks`` when the var was unset, which silently resolved to a
    directory that does not exist here, discovered 0 tasks, and emitted an EMPTY plan —
    the exact "plan generated against the wrong root warms the wrong images" failure the
    guard exists to prevent, just reached by omission instead of by a stale value.
    One variable, one contract, fail loud. Lazy import to match the plugin style
    (plugin -> xrlenv core).
    """
    from xrlenv_plugins.benchmarks._benchmark_cache import benchmark_cache_root

    return Path(benchmark_cache_root()).expanduser()


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
            f"build_cache.py --stage all first.",
        )
    env = tomllib.loads(toml_path.read_text()).get("environment", {})
    image_ref = env.get("docker_image")
    if not image_ref:
        raise SystemExit(
            f"ERROR: {task_id}: task.toml has no [environment] docker_image. "
            f"SWE-rebench tasks carry their prebuilt image in tests/config.json, "
            f"not task.toml — run `build_cache.py --stage repin` to write it.",
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


def known_sizes_from_plan(path: Path) -> dict[str, int]:
    """``{image_ref: size_hint_bytes}`` for every entry of an existing plan whose
    size was actually **measured** (``size_hint_source: registry-probe``).

    Heuristic entries are deliberately excluded — reusing one would freeze the
    fallback in place and make the resumable-probe loop never converge. A
    missing / unreadable / malformed plan yields ``{}`` (probe everything).
    """
    if not path.is_file():
        return {}
    import yaml

    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(doc, dict):
        return {}
    known: dict[str, int] = {}
    for entry in doc.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        placement = entry.get("placement") or {}
        ref, size = entry.get("image_ref"), placement.get("size_hint_bytes")
        if (
            placement.get("size_hint_source") == "registry-probe"
            and isinstance(ref, str)
            and isinstance(size, int)
            and size > 0
        ):
            known[ref] = size
    return known


def generate_plan(
    tasks: list[str],
    *,
    shard_dir: Path | None = None,
    preferred_home_count: int = 1,
    pinned: bool = False,
    probe_sizes: bool = True,
    probe_workers: int = 8,
    known_sizes: dict[str, int] | None = None,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
) -> dict[str, Any]:
    """Build a YAML-shaped ``BuildPlan`` dict (``entries: [...]``).

    Every entry carries the task's authoritative ``docker_image`` as its
    ``image_ref`` with ``context_source: {type: registry}`` — SWE-rebench images
    are all upstream prebuilts, so nothing is built. Raises ``SystemExit`` if a
    requested task is missing or has no ``docker_image``.
    """
    shard = shard_dir if shard_dir is not None else _shard_dir()

    refs: list[tuple[str, str]] = [
        (task_id, _task_image_ref(shard, task_id)) for task_id in tasks
    ]

    # Seed already-measured sizes (--reuse-sizes), then probe only what is still
    # unknown. Docker Hub 429s partway through an 860-image sweep even
    # authenticated, so this is what lets successive runs converge on a fully
    # measured plan instead of re-burning the budget on refs we already know.
    sizes: dict[str, int] = dict(known_sizes or {})
    if probe_sizes:
        from xrlenv_plugins.benchmarks._dockerhub_probe import (
            announce_auth_status,
            print_probe_summary,
            probe_image_size,
        )

        announce_auth_status()
        to_probe = [ref for _, ref in refs if ref not in sizes]
        if sizes:
            print(
                f"reusing {len(sizes)} measured size(s); probing the remaining "
                f"{len(to_probe)}.",
                file=sys.stderr,
            )

        def _probe(ref: str) -> tuple[str, int | None]:
            repo, tag = _split_repo_tag(ref)
            return ref, probe_image_size(repo, tag)

        if to_probe:
            with ThreadPoolExecutor(max_workers=max(1, probe_workers)) as pool:
                for ref, size in pool.map(_probe, to_probe):
                    if size is not None:
                        sizes[ref] = size
            print_probe_summary(DEFAULT_SIZE_HINT_BYTES)

    entries: list[dict[str, Any]] = []
    for task_id, image_ref in refs:
        size_bytes = sizes.get(image_ref)
        size_source = "registry-probe" if size_bytes is not None else "heuristic"
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
            "labels": {
                "xrlenv.benchmark": SHARD,
                "xrlenv.task_id": task_id,
            },
        })

    n_heuristic = sum(
        1 for e in entries if e["placement"]["size_hint_source"] == "heuristic"
    )
    if n_heuristic:
        print(
            f"NOTE: {n_heuristic}/{len(entries)} entr(ies) still "
            f"carry the {DEFAULT_SIZE_HINT_BYTES} B heuristic size (Docker Hub rate "
            f"limit). Re-run with `--reuse-sizes <this plan>` after the rate window "
            f"resets to fill them in; each pass keeps what it already measured.",
            file=sys.stderr,
        )

    return {
        "version": 1,
        "name": f"swe-rebench-{len(tasks)}-task",
        "replication": 1,
        "budget": {
            "reserved_runtime_gb": reserved_runtime_gb,
            "buffer_gb": buffer_gb,
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="swe-rebench-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for SWE-rebench. Every task has a prebuilt "
            "Docker Hub image (read per-task from task.toml's [environment] "
            "docker_image, written by build_cache.py --stage repin); the plan uses "
            "type: registry entries throughout, for eager warmup via "
            "`xrlenv build apply`. Nothing is ever built."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="Every task in the populated shard (the full 860-task "
                          "corpus).")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated explicit task ids (e.g. the green set).")
    p.add_argument("--preferred-home-count", type=int, default=1,
                   help="Per-image preferred_home_count (default 1).")
    p.add_argument("--pinned", action="store_true",
                   help="Pin entries so eviction skips them.")
    p.add_argument("--no-probe", dest="probe", action="store_false",
                   help="Skip the Docker Hub size probe and use the conservative "
                        f"{DEFAULT_SIZE_HINT_BYTES} B heuristic for every entry. "
                        "Probing is ON by default (these images ARE on Docker Hub); "
                        "set DOCKERHUB_USER/DOCKERHUB_TOKEN for an --all run so the "
                        "anonymous rate limit doesn't force silent fallbacks.")
    p.add_argument("--probe-workers", type=int, default=8,
                   help="Parallel Docker Hub probes (default 8).")
    p.add_argument("--reuse-sizes", default=None, metavar="PLAN",
                   help="Seed measured (size_hint_source: registry-probe) sizes from "
                        "an existing plan and probe ONLY the entries still on the "
                        "heuristic. Docker Hub 429s partway through an 860-image "
                        "sweep even authenticated, so re-running with this flag after "
                        "the rate window resets (~6 h) is how a fully-measured plan is "
                        "reached. Usually the plan you are (re)writing.")
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
        if not tasks:
            raise SystemExit(f"--tasks {args.tasks!r} selected no tasks")

    plan = generate_plan(
        tasks,
        shard_dir=shard,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        probe_sizes=args.probe,
        probe_workers=args.probe_workers,
        known_sizes=(
            known_sizes_from_plan(Path(args.reuse_sizes)) if args.reuse_sizes else None
        ),
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
