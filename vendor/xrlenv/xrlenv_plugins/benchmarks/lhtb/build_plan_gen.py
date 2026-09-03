"""Build-plan generator for LHTB — **the single source of truth** for where each
task's image comes from.

One plan, one entry per task (plus one per compose sidecar), typed by the task's own
authoritative ``[environment] docker_image`` (read from ``task.toml``, never
synthesized — a task with no ``docker_image`` fails loud):

* **most** LHTB tasks ship a prebuilt public Docker Hub image (``docker_image =
  zli12321/lhtb-<task>:<date>``, a few under ``zhongzhi660/``) → ``context_source:
  {type: registry}``. Nothing is built for these: they pull on acquire through the
  ``:5010`` docker.io pull-through mirror (also how ``xrlenv build apply`` warms
  them), pass straight through the control-plane digest resolver — no registry-
  resolver changes needed. Sizes are probed from Docker Hub's v2 manifest API
  (``--no-probe`` falls back to a conservative hint; ``xrlenv build calibrate``
  refines to true on-disk sizes after the first warm).
* the **6 REBUILD tasks** whose image we build ourselves — chess-mate's ``game``
  sidecar (published nowhere) and the baked-defect images (duckdb's
  ``-j{os.cpu_count()}``, the ``patch``-less audits, unknown-config's stale daemon).
  After ``build_cache --stage all --registry <host>`` repins their ``docker_image``
  to ``<host>/lhtb/<task>:main``, they become ``context_source: {type: local}`` build
  entries (+ one per compose sidecar) → built and pushed to the ``:5011`` private
  registry by ``deploy/registry/build_and_push_images.py``.

So the plan is a genuine **mixture** on the full path (6 local, ~40 registry) and
all-registry out-of-box (cache not repinned). The one plan feeds both consumers:
``build_and_push_images.py`` builds the ``local`` entries and skips ``registry``;
``xrlenv build apply`` warms them all.

Usage::

    XRLENV_BENCHMARK_CACHE=/path/to/cache \\
    .venv/bin/python -m xrlenv_plugins.benchmarks.lhtb.build_plan_gen \\
        --all --output xrlenv_plugins/benchmarks/lhtb/lhtb_build_plan.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

# Shard subdir name == the dataset dir build_cache.py writes. Tasks live at
# ``<cache>/lhtb/<name>/``.
SHARD = "lhtb"

# Conservative default when the Docker Hub probe fails / ``--no-probe``. LHTB images
# are ~200-400 MB compressed; calibrate refines to cluster-reported sizes post-warm.
DEFAULT_SIZE_HINT_BYTES = 800_000_000  # 0.8 GiB

# ── type: local build entries (the images we build ourselves) ─────────────────
# Most LHTB tasks are prebuilt on docker.io (type: registry). The 6 REBUILD tasks are
# BUILT from the task's own Dockerfile and pushed to OUR private registry, exactly
# like every terminalworld image:
#   * ``chess-mate`` — a multi-service compose task whose ``game`` referee sidecar
#     (built from ``Dockerfile.game``) is published NOWHERE. The main image is built
#     + pushed alongside so the whole task resolves under one ``<registry>/lhtb/
#     <task_id>`` namespace — and both build-time (here) and run-time derive the
#     sidecar namespace from the repinned main ref (``compose.registry_namespace_and_tag``),
#     so chess-mate runs in the ordinary sweep without a sweep-injected image template kwarg.
#   * the baked-defect images — their fix lives in the image, not the task dir, so a
#     rebuild (+ repin) is the real fix a docker.io pull can't give.
# generate_plan() emits these as type: local entries (main + any sidecars) inline in
# the one unified plan; build_and_push_images.py builds them. See README §2.

# Shared-fs topology asserted by every ``type: local`` entry — the build-context
# dirs live in the harbor cache on shared FSx, so each build node sees the same
# path. Rides into the plan as ``shared_fs``.
DEFAULT_SHARED_FS = "hyperpod"

# Conservative per-image heuristic for the local-build path (chess-mate's images
# are ``python:3.11-slim`` + stockfish, a few hundred MB). Used only for size-aware
# sharding across build hosts; ``xrlenv build calibrate`` refines post-build.
DEFAULT_LOCAL_SIZE_HINT_BYTES = 500_000_000  # 0.5 GiB


def _harbor_cache_root() -> Path:
    """The cache ROOT, on exactly the same contract this kit's ``build_cache.py`` uses.

    ``benchmark_cache_root`` hard-rejects the retired var/path (renamed 2026-07-31:
    XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, xrlenv_harbor_cache ->
    xrlenv_benchmark_cache) and raises when nothing is set. Deferring to it rather than
    re-implementing the lookup is the point: this generator previously fell back to a
    home-directory cache when the var was unset, which silently resolved to a directory
    that need not exist, discovered 0 tasks, and emitted an EMPTY plan — a warm plan that
    warms nothing, with no error. One variable, one contract, fail loud. Lazy import to
    match the plugin style (plugin -> xrlenv core).
    """
    from xrlenv_plugins.benchmarks._benchmark_cache import benchmark_cache_root

    return Path(benchmark_cache_root()).expanduser()


def _shard_dir() -> Path:
    return _harbor_cache_root() / SHARD


def _discover_all_tasks(shard_dir: Path) -> list[str]:
    """Every task in the shard that ships a ``task.toml`` (the harbor-format anchor,
    same predicate the oracle sweep enumerates on)."""
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
            f"LHTB tasks are expected to declare a prebuilt Docker Hub image.",
        )
    # A repinned REBUILD task carries a HOST-AGNOSTIC placeholder
    # "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}/lhtb/<task>:main";
    # the plan needs a concrete ref for build/push/warm, resolved from .env at plan-gen
    # time (no-op for a literal docker.io ref). GUIDELINE §5.3.1.
    resolved = os.path.expandvars(str(image_ref))
    if "${" in resolved:
        raise SystemExit(
            f"ERROR: {task_id}: unresolved registry placeholder in docker_image "
            f"{resolved!r} — export XRLENV_PRIVATE_REGISTRY_HOST / "
            "XRLENV_PRIVATE_REGISTRY_PORT (source .env) before generating the plan.",
        )
    return resolved


def _split_repo_tag(image_ref: str) -> tuple[str, str]:
    """Split ``namespace/name:tag`` into ``(repo, tag)`` on the LAST ``:``. Defaults
    the tag to ``latest`` if the ref carries none."""
    repo, sep, tag = image_ref.rpartition(":")
    if not sep or "/" in tag:  # no tag (or the ':' was a host:port, not a tag)
        return image_ref, "latest"
    return repo, tag


def _registry_entry(
    image_ref: str,
    *,
    size_bytes: int,
    size_source: str,
    preferred_home_count: int,
    pinned: bool,
) -> dict[str, Any]:
    """One ``type: registry`` entry: a prebuilt docker.io image, pulled (and cached
    in the ``:5010`` pull-through mirror) on warm / acquire — nothing built by us."""
    return {
        "image_ref": image_ref,
        "context_source": {"type": "registry"},
        "placement": {
            "preferred_home_count": preferred_home_count,
            "size_hint_bytes": size_bytes,
            "size_hint_source": size_source,
        },
        "pinned": pinned,
        "priority": 0,
    }


def generate_plan(
    tasks: list[str],
    *,
    shard_dir: Path | None = None,
    shared_fs: str = DEFAULT_SHARED_FS,
    preferred_home_count: int = 1,
    pinned: bool = False,
    probe_sizes: bool = True,
    reserved_runtime_gb: int = 30,
    buffer_gb: int = 10,
    rebuild_tasks: frozenset[str] | None = None,
) -> dict[str, Any]:
    """THE single source of truth for where each LHTB task's image comes from.

    One entry per task (plus one per compose sidecar), **typed by the task's own
    authoritative** ``[environment] docker_image`` (read from task.toml, never
    synthesized):

    * a REBUILD task whose ``docker_image`` is a **private-registry ref** — i.e. it
      was repinned by ``build_cache --stage all --registry <host>`` to
      ``<host>/lhtb/<task>:main`` — is **ours to build**: emit
      ``context_source: {type: local}`` at the shard build context (+ one per compose
      sidecar, its namespace derived from that same repinned ref via
      :func:`compose.registry_namespace_and_tag`, identical to the runtime rewrite).
      ``build_and_push_images.py`` builds + pushes it to the ``:5011`` private registry.
    * every other task — its ``docker_image`` is a prebuilt **docker.io** ref — is
      ``context_source: {type: registry}``: nothing to build; it is served through the
      ``:5010`` pull-through mirror (warmed by ``xrlenv build apply`` / lazy-pull).

    So on the full path the plan is a genuine **mixture** (the 6 rebuilds local, the
    rest registry); out-of-box (cache not repinned) every ``docker_image`` is a
    docker.io ref, so every entry is ``type: registry``. The one plan feeds both
    consumers: ``build_and_push_images.py`` builds the ``local`` entries and skips the
    ``registry`` ones; ``xrlenv build apply`` warms them all. Raises ``SystemExit`` if
    a task is missing, has no ``docker_image``, or (a repinned rebuild) has no
    Dockerfile in the shard."""
    from xrlenv_plugins.harbor import compose as hc

    shard = shard_dir if shard_dir is not None else _shard_dir()
    if rebuild_tasks is None:
        from xrlenv_plugins.benchmarks.lhtb.build_cache import REBUILD_TASKS
        rebuild_tasks = REBUILD_TASKS

    probe = None
    print_summary = None
    if probe_sizes:
        from xrlenv_plugins.benchmarks._dockerhub_probe import (
            announce_auth_status,
            print_probe_summary,
            probe_image_size,
        )

        announce_auth_status()
        probe = probe_image_size
        print_summary = print_probe_summary

    entries: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        image_ref = _task_image_ref(shard, task_id)
        namespace, tag = hc.registry_namespace_and_tag(image_ref)
        # We build a task iff it's a REBUILD task AND its docker_image is a private-
        # registry ref (namespace is None for docker.io-relative refs). Otherwise it's
        # a prebuilt docker.io image → type: registry.
        if task_id in rebuild_tasks and namespace is not None:
            env_dir = shard / task_id / "environment"
            if not (env_dir / "Dockerfile").is_file():
                raise SystemExit(
                    f"ERROR: {task_id}: no {env_dir}/Dockerfile — a REBUILD task "
                    f"repinned to {image_ref} must be buildable from the shard. Is "
                    f"the shard populated? Run build_cache.py --stage all first.",
                )
            entries.append(_local_entry(
                image_ref=image_ref, path=env_dir, dockerfile="Dockerfile",
                shared_fs=shared_fs, benchmark=SHARD, task_id=task_id,
                preferred_home_count=preferred_home_count, pinned=pinned,
            ))
            entries.extend(_sidecar_entries(
                task_id, env_dir, benchmark=SHARD, ref_namespace=namespace, tag=tag,
                shared_fs=shared_fs, preferred_home_count=preferred_home_count,
                pinned=pinned, size_hint_bytes=DEFAULT_LOCAL_SIZE_HINT_BYTES,
            ))
            continue
        # type: registry — prebuilt docker.io ref, warmed through the :5010 mirror.
        size_bytes: int | None = None
        size_source = "heuristic"
        if probe is not None:
            repo, ptag = _split_repo_tag(image_ref)
            size_bytes = probe(repo, ptag)
            if size_bytes is not None:
                size_source = "registry-probe"
        if size_bytes is None:
            size_bytes = DEFAULT_SIZE_HINT_BYTES
        entries.append(_registry_entry(
            image_ref, size_bytes=size_bytes, size_source=size_source,
            preferred_home_count=preferred_home_count, pinned=pinned,
        ))
    if print_summary is not None:
        print_summary(DEFAULT_SIZE_HINT_BYTES)

    return {
        "version": 1,
        "name": f"lhtb-{len(tasks)}-task",
        "replication": 1,
        "budget": {
            "reserved_runtime_gb": reserved_runtime_gb,
            "buffer_gb": buffer_gb,
        },
        "entries": entries,
    }


def _compose_path(env_dir: Path) -> Path | None:
    """The task's ``docker-compose.yaml`` (or ``.yml``) in ``env_dir``, or None."""
    for name in ("docker-compose.yaml", "docker-compose.yml"):
        path = env_dir / name
        if path.is_file():
            return path
    return None


def _local_entry(
    *,
    image_ref: str,
    path: Path,
    dockerfile: str,
    shared_fs: str,
    benchmark: str,
    task_id: str,
    service: str | None = None,
    preferred_home_count: int = 1,
    pinned: bool = False,
    size_hint_bytes: int = DEFAULT_LOCAL_SIZE_HINT_BYTES,
) -> dict[str, Any]:
    """One ``type: local`` build entry pointing at ``path`` (a task's
    ``environment/`` dir, or a per-service build sub-context). Mirrors the
    terminalworld generator's entry shape so ``deploy/registry/build_and_push_images.py``
    consumes both identically. ``benchmark`` is only the ``xrlenv.benchmark`` label
    (always ``lhtb``); the image ref is passed in whole (already the repinned
    private-registry ref), never composed from ``benchmark`` here."""
    labels = {"xrlenv.benchmark": benchmark, "xrlenv.task_id": task_id}
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


def _sidecar_entries(
    task_id: str,
    env_dir: Path,
    *,
    benchmark: str,
    ref_namespace: str,
    tag: str,
    shared_fs: str,
    preferred_home_count: int,
    pinned: bool,
    size_hint_bytes: int,
) -> list[dict[str, Any]]:
    """Build entries for a multi-service task's sidecar build contexts. A sidecar
    is a service whose ``build:`` is a **sub-dir context OR a custom ``dockerfile:``**
    (chess-mate's ``game`` builds ``.`` with ``Dockerfile.game``) — the single
    source of truth is :func:`compose.subdir_build_services`, shared with the
    runtime rewrite so the pushed ref == the acquired ref. ``ref_namespace`` is the
    registry namespace **derived from the task's repinned main ref** (identical to
    the runtime :func:`compose.registry_namespace_and_tag`), so the pushed sidecar ref
    matches what the sweep resolves. Services that build ``.`` with the default
    Dockerfile reuse the task's canonical ``<id>`` image (emitted separately);
    ``image:``-only sidecars are pulled, not built."""
    from xrlenv_plugins.harbor import compose as hc

    compose_path = _compose_path(env_dir)
    if compose_path is None:
        return []
    doc = hc.load_compose(compose_path.read_text(errors="replace"))
    subdir = hc.subdir_build_services(doc)
    if not subdir:
        return []
    refs = hc.default_image_refs(task_id, doc, namespace=ref_namespace, tag=tag)
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
                f"populated? Run build_cache.py --stage all --use-upstream-image first.",
            )
        entries.append(_local_entry(
            image_ref=refs[service],
            path=ctx_dir,
            dockerfile=dockerfile,
            shared_fs=shared_fs,
            benchmark=benchmark,
            task_id=task_id,
            service=service,
            preferred_home_count=preferred_home_count,
            pinned=pinned,
            size_hint_bytes=size_hint_bytes,
        ))
    return entries


def main(argv: list[str] | None = None) -> int:
    from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
    _maybe_auto_load_dotenv()

    p = argparse.ArgumentParser(
        prog="lhtb-build-plan-gen",
        description=(
            "Generate THE build plan for LHTB — one plan, the single source of truth "
            "for where each task's image comes from. Each task is typed by its own "
            "task.toml [environment] docker_image: a REBUILD task repinned to the "
            "private registry (via build_cache --stage all --registry) becomes a "
            "type: local build entry (+ any compose sidecars); every prebuilt "
            "docker.io task becomes a type: registry entry. build_and_push_images.py "
            "builds the local entries and skips the registry ones; xrlenv build apply "
            "warms them all."
        ),
    )
    sel = p.add_mutually_exclusive_group(required=False)
    sel.add_argument("--all", action="store_true",
                     help="Every task in the populated shard.")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated explicit task ids.")
    p.add_argument("--shared-fs", default=DEFAULT_SHARED_FS,
                   help=f"shared_fs topology asserted per type: local entry (default "
                        f"{DEFAULT_SHARED_FS!r}).")
    p.add_argument("--preferred-home-count", type=int, default=1,
                   help="Per-image preferred_home_count (default 1).")
    p.add_argument("--pinned", action="store_true",
                   help="Pin entries so eviction skips them.")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip the Docker Hub size probe for type: registry entries; "
                        "use the conservative default hint. (Probe is ON by default — "
                        "the docker.io images can be sized; local builds never probe.)")
    p.add_argument("--output", default="-",
                   help="Output path (default '-' for stdout).")
    args = p.parse_args(argv)

    # Hard-reject the retired cache env var / path (renamed 2026-07-31) before we read
    # the cache root (via _shard_dir -> _harbor_cache_root -> $XRLENV_BENCHMARK_CACHE),
    # so a plan built against a stale/absent cache fails loud. No --dest here — the root
    # comes only from the env var, so guard the env alone. Lazy import (plugin -> core).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env()

    shard = _shard_dir()
    if args.all:
        tasks = _discover_all_tasks(shard)
        if not tasks:
            raise SystemExit(
                f"no tasks under {shard} — populate the shard first "
                f"(build_cache.py --stage all).",
            )
    elif args.tasks:
        tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    else:
        raise SystemExit("select tasks: --all or --tasks <csv>.")

    plan = generate_plan(
        tasks,
        shard_dir=shard,
        shared_fs=args.shared_fs,
        preferred_home_count=args.preferred_home_count,
        pinned=args.pinned,
        probe_sizes=not args.no_probe,
    )
    n_local = sum(1 for e in plan["entries"] if e["context_source"]["type"] == "local")
    n_registry = len(plan["entries"]) - n_local
    print(
        f"plan: {len(tasks)} tasks -> {len(plan['entries'])} entries "
        f"({n_local} type: local [build + push to :5011], "
        f"{n_registry} type: registry [docker.io, warm via :5010 mirror])",
        file=sys.stderr,
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
