"""Build-plan generator for seta-env (camel-ai/seta-env).

Unlike terminal-bench-2 and swebench-verified, seta-env tasks do
**not** publish prebuilt registry images — each task ships an
``environment/Dockerfile`` that must be built. The generated plan
uses ``context_source: type: git`` entries pointing at upstream's
GitHub repo. The B.1.next coordinator dispatches registry-source
entries today; ``type: git`` (and ``type: tarball``) entries are
rejected at apply time with a clear ``ManifestInvalid`` and ship
in B.1.next.b alongside the node-side git+tarball builder.

Image refs are tagged ``seta-env/<task_id>:<git_ref>`` so a plan
rebuild on a new commit produces fresh refs (idempotent re-pulls /
re-builds key on the resulting digest). The private registry host
prefixes every ref at push time (``<host>:5011/seta-env/<id>:<ref>``),
so the bare ``seta-env`` namespace can't collide with anything on a
public registry — no extra prefix needed.

Known-unbuildable tasks (upstream Dockerfile bugs — a ``COPY`` of a
file that isn't committed, a ``RUN python3`` with no python installed,
a ``wget`` of a dead URL) are listed in ``black_list.txt`` next to the
output and excluded automatically (``--no-blacklist`` to keep them).

Sizes are heuristic by default (``size_hint_source: heuristic``)
since git-source builds can't be probed before they run. The
``cluster-reported`` upgrade path via ``xrlenv build calibrate`` is
planned for B.1.next.b; until then sizes stay heuristic and the
bin packer adds a safety margin.

Usage::

    .venv/bin/python -m xrlenv_plugins.benchmarks.seta.build_plan_gen \\
        --tasks 0,1,42 --output build_plan.yaml

    .venv/bin/python -m xrlenv_plugins.benchmarks.seta.build_plan_gen \\
        --range 0-99 --output build_plan.yaml

    .venv/bin/python -m xrlenv_plugins.benchmarks.seta.build_plan_gen \\
        --range 0-1375 --output build_plan_1376_full.yaml

    # Pull the canonical task list from the upstream repo
    # (network-only; does NOT need a local clone of seta-env):
    .venv/bin/python -m xrlenv_plugins.benchmarks.seta.build_plan_gen \\
        --remote --output build_plan.yaml

The committed ``build_plan.yaml`` is a 16-entry phase-1 starter set;
operators driving the full Harbor-Dataset (1000+ tasks) should
regenerate with ``--remote``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# BASE_IMAGE_FIX_TASKS (FROM-rewrite) + DROPPED_COMMAND_TASKS (ENTRYPOINT-restore)
# live in build_cache.py; imported here so the plan generator and the cache never
# drift on which tasks build type: local (from the patched cache Dockerfile).
from xrlenv_plugins.benchmarks.seta.build_cache import (
    BASE_IMAGE_FIX_TASKS,
    DROPPED_COMMAND_TASKS,
)

# Phase-1 starter list. 16 contiguous task ids; chosen as a small
# representative slice rather than as a difficulty-stratified set
# (seta-env's Harbor-Dataset isn't published with difficulty bands
# like swebench-verified is, so a contiguous prefix is the
# conservative default).
STARTER_TASKS: tuple[str, ...] = tuple(str(i) for i in range(16))

# Upstream repo. Used both for the ``context_source.repo`` field
# and for the network-only ``--remote`` task discovery.
DEFAULT_REPO = "https://github.com/camel-ai/seta-env"
DEFAULT_REF = "main"
DEFAULT_SUBDIR_TEMPLATE = "Harbor-Dataset/{task_id}/environment"

# Conservative default; seta-env images vary widely (a node-only
# task is ~200 MB compressed, a CUDA-bearing one is ~3 GiB). 1.5 GiB
# is a middle-of-distribution heuristic. The B.1.next.b calibrate
# flow will replace these with ``cluster-reported`` values after
# the first cluster build.
DEFAULT_SIZE_HINT_BYTES = 1_500_000_000  # 1.5 GiB

# Per-task image-ref repository namespace. Kept bare ("seta-env", not
# "xrlenv-seta-env") because the private registry host already prefixes every
# pushed ref (``<host>:5011/seta-env/<id>:<ref>``), which is what disambiguates
# from public-registry images — an extra "xrlenv-" prefix is redundant.
IMAGE_NAMESPACE = "seta-env"

# Default blacklist filename, looked up next to the plan's --output.
DEFAULT_BLACKLIST_NAME = "black_list.txt"

# Base-image-restore tasks (:data:`BASE_IMAGE_FIX_TASKS`) build ``type: local`` from
# the CACHE Dockerfile (which ``build_cache.py --stage all`` rewrote to the t-bench
# base) — a ``type: git`` build would use upstream's broken ``FROM ubuntu:24.04``.
# ``shared_fs`` names the cluster-shared FS that makes the local path resolve on
# every build node.
DEFAULT_SHARED_FS = "hyperpod"


def _discover_remote_tasks(
    repo: str = DEFAULT_REPO,
    ref: str = DEFAULT_REF,
    *,
    timeout_s: float = 30.0,
) -> list[str]:
    """List every Harbor-Dataset task id via GitHub's **git Trees** API.

    The Contents API caps a directory listing at 1000 entries and does NOT
    paginate it — that silently dropped ~376 of the 1376 Harbor-Dataset tasks.
    The Trees API returns the whole subtree in one call (with a ``truncated``
    flag for the multi-thousand case), so this gets them all. Falls back to the
    16-task starter set on any network/parse error."""
    slug = repo.rstrip("/").removeprefix("https://github.com/")

    def _gh(url: str) -> Any:
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise ValueError(f"GitHub returned {resp.status}")
            return json.loads(resp.read())

    try:
        root = _gh(f"https://api.github.com/repos/{slug}/git/trees/{ref}")
        hd = next(
            e for e in root.get("tree", [])
            if e.get("path") == "Harbor-Dataset" and e.get("type") == "tree"
        )
        tree = _gh(f"https://api.github.com/repos/{slug}/git/trees/{hd['sha']}")
        ids = [e["path"] for e in tree.get("tree", []) if e.get("type") == "tree"]
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError, KeyError, StopIteration):
        return list(STARTER_TASKS)
    if not ids:
        return list(STARTER_TASKS)
    # GitHub returns mixed string/numeric task ids in arbitrary order;
    # numeric-aware sort matches the canonical layout.
    def _key(s: str) -> tuple[int, str]:
        return (int(s) if s.isdigit() else 1 << 30, s)
    return sorted(ids, key=_key)


def _load_blacklist(path: Path) -> set[str]:
    """Read task ids to exclude. Each non-blank, non-``#`` line contributes its
    first whitespace token as a task id (the rest is a human comment). A missing
    file yields an empty set."""
    ids: set[str] = set()
    if not path.is_file():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.add(stripped.split()[0])
    return ids


def _parse_range(spec: str) -> list[str]:
    """Parse a comma-separated mix of ``N`` and ``N-M`` ranges into
    a flat list of string ids."""
    out: list[str] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            out.extend(str(i) for i in range(lo, hi + 1))
        else:
            out.append(chunk)
    return out


def _image_ref(task_id: str, ref: str) -> str:
    # Replace any tag-illegal characters in ref (slashes, etc.).
    safe_ref = ref.replace("/", "-").replace(":", "-")
    return f"{IMAGE_NAMESPACE}/{task_id}:{safe_ref}"


def generate_plan(
    task_ids: list[str],
    *,
    repo: str = DEFAULT_REPO,
    ref: str = DEFAULT_REF,
    preferred_home_count: int = 1,
    pinned: bool = False,
    size_hint_bytes: int = DEFAULT_SIZE_HINT_BYTES,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
    cache_root: str | None = None,
    shared_fs: str = DEFAULT_SHARED_FS,
) -> dict[str, Any]:
    """Build a YAML-shaped ``BuildPlan`` dict (per-image-ref schema), one entry per
    task id. Most entries are ``context_source: type: git`` (seta-env publishes
    Dockerfiles, not prebuilt images). The exception is the tasks ``build_cache.py
    --stage all`` patches in the cache — base-image-restore (:data:`BASE_IMAGE_FIX_TASKS`,
    FROM rewritten to the t-bench base) and dropped-command-restore
    (:data:`DROPPED_COMMAND_TASKS`, ENTRYPOINT baked): those build ``type: local`` from
    ``<cache_root>/seta-env/<id>/environment`` because a git build would use upstream's
    unpatched Dockerfile. Such a task requested without ``cache_root`` falls back to
    ``type: git`` (unpatched → will still FAIL) with a warning."""
    # Tasks whose cache Dockerfile was patched by build_cache (`--stage all`) — base
    # restored (FROM rewritten) and/or dropped command restored (ENTRYPOINT baked) —
    # must build type: local from that patched cache, not type: git (upstream's).
    local_build = BASE_IMAGE_FIX_TASKS | frozenset(DROPPED_COMMAND_TASKS)
    entries: list[dict[str, Any]] = []
    unrestored: list[str] = []
    for tid in task_ids:
        if tid in local_build and cache_root:
            env_dir = Path(cache_root).expanduser() / IMAGE_NAMESPACE / tid / "environment"
            context_source: dict[str, Any] = {
                "type": "local",
                "path": str(env_dir),
                "dockerfile": "Dockerfile",
                "shared_fs": shared_fs,
            }
        else:
            if tid in local_build:
                unrestored.append(tid)
            context_source = {
                "type": "git",
                "repo": repo,
                "ref": ref,
                "subdir": DEFAULT_SUBDIR_TEMPLATE.format(task_id=tid),
                "dockerfile": "Dockerfile",
            }
        entries.append({
            "image_ref": _image_ref(tid, ref),
            "context_source": context_source,
            "placement": {
                "preferred_home_count": preferred_home_count,
                "size_hint_bytes": size_hint_bytes,
                "size_hint_source": "heuristic",
            },
            "pinned": pinned,
            "priority": 0,
            "labels": {
                "xrlenv.benchmark": "seta-env",
                "xrlenv.task_id": tid,
            },
        })
    if unrestored:
        print(
            f"WARNING: {len(unrestored)} cache-patched task(s) emitted as type: git "
            f"(unpatched — will still FAIL): {', '.join(unrestored)}. Pass "
            f"--cache-root (a populated, `build_cache --stage all` cache) so they "
            f"build type: local from the patched Dockerfile.",
            file=sys.stderr,
        )
    suffix = (
        "starter-16" if list(task_ids) == list(STARTER_TASKS)
        else f"{len(task_ids)}-task"
    )
    return {
        "version": 1,
        "name": f"seta-env-{suffix}",
        "replication": 1,
        "budget": {
            "reserved_runtime_gb": reserved_runtime_gb,
            "buffer_gb": buffer_gb,
        },
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="seta-env-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for camel-ai/seta-env. "
            "Tasks ship Dockerfiles only (no prebuilt registry "
            "images), so every entry uses context_source: type: git."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--remote", action="store_true",
                     help="Discover all Harbor-Dataset tasks via the "
                          "GitHub Contents API.")
    sel.add_argument("--starter", action="store_true",
                     help="The 16 phase-1 starter tasks (ids 0-15).")
    sel.add_argument("--range", dest="range_spec", default=None,
                     help="Comma-separated mix of N and N-M ranges, e.g. "
                          "'0-7,42,100-105'.")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated explicit task ids.")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"Upstream git repo (default {DEFAULT_REPO}).")
    p.add_argument("--ref", default=DEFAULT_REF,
                   help=f"Git ref / branch / tag (default {DEFAULT_REF}).")
    p.add_argument("--preferred-home-count", type=int, default=1,
                   help="Per-image preferred_home_count (default 1).")
    p.add_argument("--pinned", action="store_true",
                   help="Pin entries so eviction skips them.")
    p.add_argument("--size-hint-bytes", type=int,
                   default=DEFAULT_SIZE_HINT_BYTES,
                   help="Heuristic per-image size hint "
                        f"(default {DEFAULT_SIZE_HINT_BYTES}).")
    p.add_argument("--cache-root", default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
                   help="Cache ROOT (the shard is <cache-root>/seta-env/). Required "
                        "for the base-image-restore tasks (BASE_IMAGE_FIX_TASKS): they "
                        "build type: local from <cache-root>/seta-env/<id>/environment "
                        "(the Dockerfile build_cache --stage all rewrote to the t-bench "
                        "base). Defaults to $XRLENV_BENCHMARK_CACHE.")
    p.add_argument("--shared-fs", default=DEFAULT_SHARED_FS,
                   help=f"Cluster-shared-FS topology label stamped on every type: "
                        f"local entry (default {DEFAULT_SHARED_FS}).")
    p.add_argument("--output", default="-",
                   help="Output path (default '-' for stdout).")
    p.add_argument("--blacklist", default=None,
                   help="Path to a blacklist of task ids to exclude (upstream-"
                        "unbuildable). Default: black_list.txt next to --output.")
    p.add_argument("--no-blacklist", action="store_true",
                   help="Include every task, even ids listed in the blacklist.")
    args = p.parse_args(argv)

    # Fail loud if the caller still points at the retired XRLENV_HARBOR_CACHE var /
    # xrlenv_harbor_cache path (renamed 2026-07-31). This generator resolves the git
    # repo, not the benchmark cache, but the guard runs at every seta entrypoint so a
    # stale env var never lets ANY seta command run against the wrong world. No --dest
    # here, so the check is env-var-only. Lazy import matches the plugin style.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env()

    if args.remote:
        task_ids = _discover_remote_tasks(repo=args.repo, ref=args.ref)
    elif args.starter:
        task_ids = list(STARTER_TASKS)
    elif args.range_spec:
        task_ids = _parse_range(args.range_spec)
    elif args.tasks:
        task_ids = [s.strip() for s in args.tasks.split(",") if s.strip()]
    else:
        return 2

    # Drop known-unbuildable tasks (upstream Dockerfile bugs). Default blacklist
    # lives next to the output; --no-blacklist keeps everything.
    if not args.no_blacklist:
        bl_path = (
            Path(args.blacklist) if args.blacklist
            else (Path(args.output).parent / DEFAULT_BLACKLIST_NAME
                  if args.output != "-" else Path(DEFAULT_BLACKLIST_NAME))
        )
        blacklisted = _load_blacklist(bl_path)
        before = len(task_ids)
        task_ids = [t for t in task_ids if t not in blacklisted]
        if before != len(task_ids):
            print(f"blacklist {bl_path}: excluded {before - len(task_ids)} "
                  f"task(s) ({before} → {len(task_ids)})", file=sys.stderr)

    plan = generate_plan(
        task_ids,
        repo=args.repo,
        ref=args.ref,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        size_hint_bytes=args.size_hint_bytes,
        cache_root=args.cache_root,
        shared_fs=args.shared_fs,
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
