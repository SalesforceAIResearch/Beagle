"""Build-plan generator for terminalworld-verified.

Like seta-env, TerminalWorld tasks do **not** publish prebuilt registry
images — each task ships an ``environment/Dockerfile`` that must be built.
Unlike seta-env (whose Dockerfiles live in a GitHub repo, so it uses
``context_source: type: git``), TerminalWorld tasks are materialized into the
**local** harbor cache by ``benchmarks/terminalworld/build_cache.py``, so the
plan uses ``context_source: type: local`` — each entry points at a task's
``environment/`` dir in the cache shard on shared FSx, built in place (no clone,
no tarball, no copy). ``type: local`` requires ``shared_fs`` (default
``hyperpod``): the assertion that every build node mounts the same FSx path.

Image refs are ``terminalworld-verified/<task_id>:main`` — the shard name
doubles as the image namespace, and the private-registry host prefixes every
ref at push time (``<host>:5011/terminalworld-verified/<id>:main``). That ref is
exactly what ``benchmarks/terminalworld/run_oracle_sweep.py`` constructs as the
``xrlenv_image_template`` per-run kwarg, so the image the build pushes is the
image the eval acquires.

Sizes are heuristic (``size_hint_source: heuristic``) — a local Dockerfile build
can't be probed before it runs, exactly as seta-env's git-source plan.

Usage::

    # generate a plan for the whole populated shard, then build+push:
    .venv/bin/python -m xrlenv_plugins.benchmarks.terminalworld.build_plan_gen \\
        --all --output /tmp/tw_build_plan.yaml
    .venv/bin/python deploy/registry/build_and_push_images.py \\
        --plan /tmp/tw_build_plan.yaml --registry <host>:5011

    # a subset:
    .venv/bin/python -m xrlenv_plugins.benchmarks.terminalworld.build_plan_gen \\
        --tasks tw_245733,tw_247958 --output -

No committed ``build_plan.yaml`` sits next to this generator (unlike the
registry/git-source benchmarks): ``type: local`` entries embed absolute cache
paths that are specific to the operator's ``XRLENV_BENCHMARK_CACHE``, so the plan
is regenerated on demand rather than checked in stale.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Shard subdir == image namespace. Matches build_cache.py's SHARD.
SHARD = "terminalworld-verified"

# Shared-fs topology asserted by every ``type: local`` entry (the build context
# dirs live in the harbor cache on shared FSx, so each build node sees the same
# path). Rides into the plan as ``shared_fs``.
DEFAULT_SHARED_FS = "hyperpod"

# Conservative per-image heuristic. TerminalWorld images are ubuntu-based +
# apt/docker installs — a few hundred MB to ~1.5 GiB. Used only for size-aware
# sharding across build nodes; the calibrate flow can refine post-build.
DEFAULT_SIZE_HINT_BYTES = 1_500_000_000  # 1.5 GiB


def _harbor_cache_root() -> Path:
    explicit = os.environ.get("XRLENV_BENCHMARK_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path("~/.cache/harbor/tasks").expanduser()


def _shard_dir() -> Path:
    return _harbor_cache_root() / SHARD


def _discover_all_tasks() -> list[str]:
    """Every task in the shard that ships an ``environment/Dockerfile`` (the
    buildable predicate)."""
    shard = _shard_dir()
    if not shard.is_dir():
        return []
    return sorted(
        p.name for p in shard.iterdir()
        if p.is_dir() and (p / "environment" / "Dockerfile").is_file()
    )


def _entry(
    *,
    image_ref: str,
    path: Path,
    dockerfile: str,
    shared_fs: str,
    preferred_home_count: int,
    pinned: bool,
    size_hint_bytes: int,
    namespace: str,
    task_id: str,
    service: str | None = None,
) -> dict[str, Any]:
    """One ``type: local`` build entry pointing at ``path`` (a task's
    ``environment/`` dir, or a per-service build sub-context)."""
    labels = {"xrlenv.benchmark": namespace, "xrlenv.task_id": task_id}
    if service is not None:
        labels["xrlenv.compose_service"] = service
    return {
        "image_ref": image_ref,
        "context_source": {
            "type": "local",
            "path": str(path.resolve()),
            "dockerfile": dockerfile,
            "shared_fs": shared_fs,
        },
        "placement": {
            "preferred_home_count": preferred_home_count,
            "size_hint_bytes": size_hint_bytes,
            "size_hint_source": "heuristic",
        },
        "pinned": pinned,
        "priority": 0,
        "labels": labels,
    }


def _service_dockerfile(build_stanza: Any) -> str:
    """The ``dockerfile`` a compose ``build:`` stanza names (default
    ``Dockerfile``). Accepts the string and mapping forms."""
    if isinstance(build_stanza, dict):
        return str(build_stanza.get("dockerfile", "Dockerfile"))
    return "Dockerfile"


def _compose_service_entries(
    task_id: str,
    env_dir: Path,
    *,
    namespace: str,
    tag: str,
    shared_fs: str,
    preferred_home_count: int,
    pinned: bool,
    size_hint_bytes: int,
) -> list[dict[str, Any]]:
    """Extra build entries for a multi-service task's **sub-directory** build
    contexts (e.g. tw_188260's ``solr-node`` / ``ambari-server``). Services that
    build from ``.`` reuse the task's canonical ``<id>`` image (already emitted);
    ``image:``-only sidecars (``postgres:14``) are pulled, not built. Empty for a
    task with no compose file or no sub-context services — the common case."""
    from xrlenv_plugins.harbor import compose as hc

    compose_path = env_dir / "docker-compose.yaml"
    if not compose_path.is_file():
        compose_path = env_dir / "docker-compose.yml"
    if not compose_path.is_file():
        return []
    doc = hc.load_compose(compose_path.read_text(errors="replace"))
    subdir = hc.subdir_build_services(doc)
    if not subdir:
        return []
    refs = hc.default_image_refs(task_id, doc, namespace=namespace, tag=tag)
    services = hc.service_map(doc)
    entries: list[dict[str, Any]] = []
    for service, ctx in sorted(subdir.items()):
        if not hc.is_safe_relative_context(ctx):
            raise SystemExit(
                f"ERROR: {task_id}: compose service {service!r} build context "
                f"{ctx!r} escapes the task environment dir — refusing to emit a "
                f"type: local build context outside {env_dir}.",
            )
        ctx_dir = env_dir / ctx
        dockerfile = _service_dockerfile(services.get(service, {}).get("build"))
        if not (ctx_dir / dockerfile).is_file():
            raise SystemExit(
                f"ERROR: {task_id}: compose service {service!r} builds from "
                f"{ctx_dir}/{dockerfile}, which is missing — is the shard "
                f"populated? Run build_cache.py --stage all first.",
            )
        entries.append(_entry(
            image_ref=refs[service],
            path=ctx_dir,
            dockerfile=dockerfile,
            shared_fs=shared_fs,
            preferred_home_count=preferred_home_count,
            pinned=pinned,
            size_hint_bytes=size_hint_bytes,
            namespace=namespace,
            task_id=task_id,
            service=service,
        ))
    return entries


def generate_plan(
    tasks: list[str],
    *,
    namespace: str = SHARD,
    tag: str = "main",
    shared_fs: str = DEFAULT_SHARED_FS,
    preferred_home_count: int = 1,
    pinned: bool = False,
    size_hint_bytes: int = DEFAULT_SIZE_HINT_BYTES,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
) -> dict[str, Any]:
    """Build a YAML-shaped per-image-ref ``BuildPlan`` dict with one
    ``type: local`` entry per task, pointing at its ``environment/`` dir in the
    cache shard, **plus** one entry per sub-directory build context for
    multi-service compose tasks. Raises ``SystemExit`` if a requested task has no
    Dockerfile (or a declared service build context is missing)."""
    shard = _shard_dir()
    safe_tag = tag.replace("/", "-").replace(":", "-")
    entries: list[dict[str, Any]] = []
    for task_id in tasks:
        env_dir = shard / task_id / "environment"
        if not (env_dir / "Dockerfile").is_file():
            raise SystemExit(
                f"ERROR: {task_id}: no {env_dir}/Dockerfile — is the shard "
                f"populated? Run build_cache.py --stage all first.",
            )
        entries.append(_entry(
            image_ref=f"{namespace}/{task_id}:{safe_tag}",
            path=env_dir,
            dockerfile="Dockerfile",
            shared_fs=shared_fs,
            preferred_home_count=preferred_home_count,
            pinned=pinned,
            size_hint_bytes=size_hint_bytes,
            namespace=namespace,
            task_id=task_id,
        ))
        entries.extend(_compose_service_entries(
            task_id, env_dir,
            namespace=namespace, tag=tag, shared_fs=shared_fs,
            preferred_home_count=preferred_home_count, pinned=pinned,
            size_hint_bytes=size_hint_bytes,
        ))
    if not entries:
        raise SystemExit(
            f"ERROR: no buildable tasks under {shard} (populate the shard first).",
        )
    return {
        "version": 1,
        "name": f"{namespace}-{len(entries)}-task",
        "replication": 1,
        "budget": {"reserved_runtime_gb": reserved_runtime_gb, "buffer_gb": buffer_gb},
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env from cwd / ancestors so an operator's registry/host vars are
    # picked up without shell-export, mirroring the other generators. Stdlib-only
    # helper; the generator never imports ``xrlenv`` at module load.
    from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    _maybe_auto_load_dotenv()

    p = argparse.ArgumentParser(
        prog="terminalworld-build-plan-gen",
        description=(
            "Generate a build-plan.yaml for terminalworld-verified. Tasks ship "
            "a local Dockerfile in the harbor cache; the plan uses type: local "
            "entries throughout. Feed the output to deploy/registry/build_and_push_images.py."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="Every buildable task in the shard.")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated explicit task ids.")
    p.add_argument("--namespace", default=SHARD,
                   help=f"Image namespace (default {SHARD}).")
    p.add_argument("--tag", default="main", help="Image tag (default 'main').")
    p.add_argument("--shared-fs", default=DEFAULT_SHARED_FS,
                   help=f"shared_fs topology asserted per entry (default "
                        f"{DEFAULT_SHARED_FS}).")
    p.add_argument("--preferred-home-count", type=int, default=1,
                   help="Per-image preferred_home_count (default 1).")
    p.add_argument("--pinned", action="store_true",
                   help="Pin entries so eviction skips them.")
    p.add_argument("--size-hint-bytes", type=int, default=DEFAULT_SIZE_HINT_BYTES,
                   help=f"Heuristic per-image size hint (default "
                        f"{DEFAULT_SIZE_HINT_BYTES}).")
    p.add_argument("--output", default="-",
                   help="Output path (default '-' for stdout).")
    args = p.parse_args(argv)

    # Fail loud before any cache read (_harbor_cache_root reads
    # $XRLENV_BENCHMARK_CACHE): a plan generated against the retired
    # var/path would embed stale/absent build-context paths. No --dest flag
    # here, so the guard checks the retired env var + the resolved env path.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env()

    if args.all:
        tasks = _discover_all_tasks()
        if not tasks:
            raise SystemExit(
                f"no buildable tasks under {_shard_dir()} — populate the shard "
                f"first (build_cache.py --stage all).",
            )
    else:
        tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]

    plan = generate_plan(
        tasks,
        namespace=args.namespace,
        tag=args.tag,
        shared_fs=args.shared_fs,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        size_hint_bytes=args.size_hint_bytes,
    )

    import yaml
    text = yaml.safe_dump(plan, sort_keys=False, default_flow_style=False)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
        print(f"wrote {len(plan['entries'])} entries -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
