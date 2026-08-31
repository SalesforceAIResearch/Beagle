#!/usr/bin/env python3
"""Bulk-build a build-plan's images from their Dockerfiles and push them to the
xrlenv PRIVATE registry (run-registry-private.sh, default ``<host>:5011``).

This is the build-once-push glue: instead of every worker rebuilding a
benchmark's Dockerfile (the per-node-build model — slow, no digest pinning), one
build host builds the image once and pushes it here; every worker then pulls a
digest-pinnable ref over the internal network.

It is the SIBLING of ``deploy/registry/warm_images.py``: warm_images fills the *proxy*
(:5010) by streaming docker.io blobs; this fills the *private* registry (:5011) by
building our own Dockerfiles and pushing them.

Distribute across CPU instances (1000 seta-env Dockerfiles on one node is slow)
============================================================================
Builds are embarrassingly parallel: each build host talks to the one registry
over HTTP (``docker push``), and the registry — the single writer — persists to
shared FSx. So shard the plan and run one shard per host:

  * Single host, everything:
      python deploy/registry/build_and_push_images.py --plan build_plan.yaml \\
          --registry <registry-host>:5011

  * One shard of N (run N copies, one per CPU instance):
      python deploy/registry/build_and_push_images.py --plan build_plan.yaml \\
          --registry <registry-host>:5011 --shard-index 3 --num-shards 16

  * Under Slurm (one shard per node): shard index/count are auto-read from
      $SLURM_PROCID / $SLURM_NTASKS (or $SLURM_ARRAY_TASK_ID /
      $SLURM_ARRAY_TASK_COUNT for a job array) if you drive it from your own
      sbatch/srun. For the native, drift-free fleet path prefer ``xrlenv build
      push`` — the control plane shards this same plan across the connected node
      agents, with no Slurm nodelist to maintain. This script remains the
      single-host / build-host fallback (and the ``local``-source path).

Sharding is a size-aware greedy partition over ``placement.size_hint_bytes`` so
each shard gets roughly equal build *bytes*, not just equal image count.

Shared-disk aware (HyperPod / FSx)
==================================
Cluster nodes share one home filesystem, so ``~/.xrlenv/build-context-cache`` is
the *same physical directory* on every box. The git checkout for a repo is
therefore done **once for the whole campaign**, not once per shard: the first
shard to need ``(repo, ref)`` clones it under a cross-node ``flock`` and writes a
completion marker; every other shard (on any node) sees the marker and reuses the
snapshot read-only. seta-env's 1000 tasks are all subdirs of one repo at one ref,
so the whole fan-out shares a single clone. (If ``flock`` isn't available — FSx
not mounted ``-o flock`` — it degrades to a per-node checkout, still correct, just
not shared. ``--refresh-context`` forces a fresh clone for a new commit.)

Idempotent / resumable / overlap-safe
======================================
Before building, it HEADs the registry manifest for the target ref and SKIPS the
image if it's already present (``--force`` rebuilds). So re-runs are cheap, an
interrupted run resumes, and overlapping shards never double-push. Each image is
built once, pushed, the local tag is pruned (shared base layers stay for build
cache), and the result (status + pushed digest) is recorded to a per-shard JSON
report.

Prerequisites on each build host:
  * Docker daemon with the private registry in ``insecure-registries`` (and,
    recommended, the :5010 mirror in ``registry-mirrors`` so FROM base-image
    pulls are LAN-fast):  sudo PRIVATE_REGISTRY=<host>:5011 \\
        bash deploy/registry/configure_docker_registry.sh --restart
  * git + docker on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import fcntl
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xrlenv import paths

# Reuse the canonical build-plan schema + loaders so this tool can never drift
# from `xrlenv build apply`'s notion of a plan — no reinventing the schema.
from xrlenv.control.build_plan import (
    BuildEntry,
    GitSource,
    LocalSource,
    RegistrySource,
    TarballSource,
    load_build_plan,
    resolve_tarball_sources,
)

# Registry v2 manifest media types — sent as Accept on the existence probe so the
# registry returns the manifest (not a 404) for both Docker- and OCI-format
# images.
_MANIFEST_ACCEPT = ", ".join((
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
))

DEFAULT_TARBALL_MAX_BYTES = 200 * 1024 * 1024  # 200 MB; matches build-apply default ballpark


# ── Pure helpers (no Docker; unit-tested) ─────────────────────────────────────


def target_ref(registry: str, image_ref: str) -> str:
    """Rewrite a plan's portable ``image_ref`` to its address in ``registry``.

    ``registry`` is a bare ``host:port`` (no scheme). The plan stays portable —
    e.g. ``xrlenv-seta-env/0:main`` — and we prepend the registry host so the
    pushed/pulled ref is ``<registry-host>:5011/xrlenv-seta-env/0:main``. Idempotent:
    a ref already prefixed with this registry is returned unchanged (so a plan
    that already bakes in the host still works)."""
    if not registry:
        return image_ref
    if image_ref.startswith(registry + "/"):
        return image_ref
    return f"{registry}/{image_ref}"


def split_ref(ref: str) -> tuple[str, str, str]:
    """Split a fully-qualified ``host[:port]/repo[:tag|@digest]`` into
    ``(host, repo, reference)``. ``reference`` is the tag, the ``sha256:...``
    digest, or ``latest`` when neither is present.

    The leading segment is always treated as the registry host here because this
    tool only ever calls it on a ref it just prefixed with ``--registry``."""
    host, _, remainder = ref.partition("/")
    if not remainder:
        # No registry host present (shouldn't happen post-target_ref); treat the
        # whole thing as the repo on an implicit host.
        host, remainder = "", ref
    if "@" in remainder:
        repo, _, digest = remainder.partition("@")
        return host, repo, digest
    last_segment = remainder.rsplit("/", 1)[-1]
    if ":" in last_segment:
        repo, _, tag = remainder.rpartition(":")
        return host, repo, tag
    return host, remainder, "latest"


def partition_entries(
    entries: list[BuildEntry], num_shards: int,
) -> list[list[BuildEntry]]:
    """Size-aware greedy partition of ``entries`` into ``num_shards`` buckets.

    Each entry is assigned to the currently-lightest bucket by cumulative
    ``placement.size_hint_bytes`` (longest-processing-time-first: sort by size
    desc, tie-break by original index for determinism). Balances build *bytes*,
    which matters because benchmark images are heavy-tailed (a node-only seta-env
    task is ~200 MB; a CUDA one is ~3 GB)."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    buckets: list[list[BuildEntry]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    ordered = sorted(
        enumerate(entries),
        key=lambda iv: (-int(iv[1].placement.size_hint_bytes), iv[0]),
    )
    for _idx, entry in ordered:
        target = min(range(num_shards), key=lambda b: (loads[b], b))
        buckets[target].append(entry)
        loads[target] += int(entry.placement.size_hint_bytes)
    return buckets


def select_shard(
    entries: list[BuildEntry], shard_index: int, num_shards: int,
) -> list[BuildEntry]:
    """Return the subset of ``entries`` this shard is responsible for."""
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index {shard_index} out of range for num_shards {num_shards}",
        )
    return partition_entries(entries, num_shards)[shard_index]


def split_buildable(
    entries: list[BuildEntry],
) -> tuple[list[BuildEntry], list[BuildEntry]]:
    """Split ``entries`` into ``(buildable, registry_only)``.

    A ``type: registry`` (``RegistrySource``) entry is a prebuilt image served at
    runtime via the ``:5010`` pull-through mirror; this tool fills the ``:5011``
    PRIVATE registry from Dockerfiles, so it has nothing to build for those. Callers
    skip the second list unless ``--mirror-registry`` opts into copying them in."""
    buildable = [e for e in entries if not isinstance(e.context_source, RegistrySource)]
    registry_only = [e for e in entries if isinstance(e.context_source, RegistrySource)]
    return buildable, registry_only


def manifest_url(scheme: str, host: str, repo: str, reference: str) -> str:
    return f"{scheme}://{host}/v2/{repo}/manifests/{reference}"


# ── Registry probe (HTTP; injectable for tests) ───────────────────────────────


def registry_has_manifest(
    tref: str, *, scheme: str = "http", timeout_s: float = 15.0,
    opener: Any = urllib.request,
) -> bool | None:
    """Does ``tref`` already exist in its registry? ``True`` present, ``False``
    absent (404), ``None`` unknown (registry unreachable / unexpected error —
    caller should fall through to build, letting the push surface the real
    error). ``opener`` is injectable so tests don't hit the network."""
    host, repo, reference = split_ref(tref)
    if not host:
        return None
    req = urllib.request.Request(
        manifest_url(scheme, host, repo, reference), method="HEAD",
    )
    req.add_header("Accept", _MANIFEST_ACCEPT)
    try:
        with opener.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        # 401/403 etc. — registry reachable but the probe was rejected. Unknown.
        return None
    except (urllib.error.URLError, OSError, ValueError):
        return None


# ── Build/push one entry (shells out to docker/git) ───────────────────────────


@dataclass
class _CloneCache:
    """Clone-once cache of git checkouts, keyed by (repo, ref).

    ``root`` is the SHARED home filesystem (``~/.xrlenv/build-context-cache``,
    the same physical FSx dir on every cluster node), so a repo is cloned exactly
    once for the whole fan-out: the first shard to need a key clones it under a
    cross-node ``flock``; siblings on any node see the ``.complete`` marker and
    reuse the snapshot read-only (no re-fetch — every shard builds the same
    commit). ``refresh=True`` forces a fresh clone (new commit of a moving ref).
    """

    root: Path
    refresh: bool = False
    _paths: dict[tuple[str, str], Path] = field(default_factory=dict)
    _locks: dict[tuple[str, str], asyncio.Lock] = field(default_factory=dict)
    _master: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ensure(self, repo: str, ref: str, *, timeout_s: float) -> Path:
        key = (repo, ref)
        async with self._master:
            lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:  # intra-process: don't enter the blocking clone twice
            cached = self._paths.get(key)
            if cached is not None and cached.is_dir():
                return cached
            # flock + git are blocking; run off the event loop.
            checkout = await asyncio.to_thread(
                self._clone_once_blocking, repo, ref, timeout_s,
            )
            self._paths[key] = checkout
            return checkout

    def _key_dir(self, repo: str, ref: str) -> Path:
        return self.root / f"{_safe(repo)}__{_safe(ref)}"

    def _clone_once_blocking(self, repo: str, ref: str, timeout_s: float) -> Path:
        key_dir = self._key_dir(repo, ref)
        checkout = key_dir / "checkout"
        marker = key_dir / ".complete"
        # Fast path (no lock): a sibling shard already cloned it to shared FSx.
        if marker.is_file() and checkout.is_dir() and not self.refresh:
            return checkout
        key_dir.mkdir(parents=True, exist_ok=True)
        lock_file = None
        locked = False
        try:
            try:
                lock_file = open(key_dir / ".lock", "w")  # noqa: SIM115 - held for the flock lifetime
                fcntl.flock(lock_file, fcntl.LOCK_EX)  # cross-node on Lustre (-o flock)
                locked = True
            except OSError:
                locked = False  # flock unsupported → degrade to a per-node checkout
            # Re-check under the lock: another node may have finished while we waited.
            if marker.is_file() and checkout.is_dir() and not self.refresh:
                return checkout
            # Without a cross-node lock, isolate this node's clone so concurrent
            # shards on other nodes can't corrupt a shared dir mid-clone.
            dest = checkout if locked else (
                key_dir / f"checkout-{_safe(socket.gethostname())}-{os.getpid()}"
            )
            self._git_clone(repo, ref, dest, timeout_s)
            if locked:
                marker.write_text(f"{repo}\n{ref}\n")
            return dest
        finally:
            if lock_file is not None:
                lock_file.close()  # releases the flock

    @staticmethod
    def _git_clone(repo: str, ref: str, dest: Path, timeout_s: float) -> None:
        """Clone ``repo@ref`` into ``dest`` (atomic: clone to a tmp sibling, then
        rename). ``--depth=1 --branch`` covers tags/branches; a commit sha falls
        back to a full clone + checkout (matches source_builder pin-handling)."""
        tmp = dest.parent / (dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        rc, out = _run_sync(
            ["git", "clone", "--depth=1", "--branch", ref, repo, str(tmp)], timeout_s,
        )
        if rc != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            rc, out = _run_sync(["git", "clone", repo, str(tmp)], timeout_s)
            if rc == 0:
                rc, out = _run_sync(["git", "-C", str(tmp), "checkout", ref], 120.0)
        if rc != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise _BuildError(f"git clone {repo}@{ref} failed: {out[-800:]}")
        os.replace(tmp, dest)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)[:80]


def _run_sync(argv: list[str], timeout_s: float) -> tuple[int, str]:
    """Blocking subprocess (used inside the clone thread). Returns (rc, output)."""
    try:
        p = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout_s:.0f}s"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class _BuildError(RuntimeError):
    pass


async def _run(
    argv: list[str], *, cwd: str | None = None, timeout_s: float,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run argv, capturing combined stdout+stderr. Returns (rc, text). rc=124 on
    timeout. Never raises for a non-zero exit — the caller decides."""
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **(env or {})},
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"timed out after {timeout_s:.0f}s"
    return proc.returncode or 0, (out or b"").decode("utf-8", errors="replace")


def _label_args(labels: dict[str, str]) -> list[str]:
    out: list[str] = []
    for k, v in labels.items():
        out += ["--label", f"{k}={v}"]
    return out


async def build_one(
    entry: BuildEntry, *, registry: str, scheme: str, clone_cache: _CloneCache,
    build_timeout_s: float, force: bool, prune: bool, probe_timeout_s: float,
) -> dict[str, Any]:
    """Materialize + push one entry. Returns a result record (never raises)."""
    tref = target_ref(registry, entry.image_ref)
    started = time.monotonic()
    rec: dict[str, Any] = {
        "image_ref": entry.image_ref, "target_ref": tref, "status": "failed",
        "digest": None, "seconds": 0.0, "error": None,
    }
    try:
        if not force:
            present = await asyncio.to_thread(
                registry_has_manifest, tref, scheme=scheme,
                timeout_s=probe_timeout_s,
            )
            if present is True:
                rec.update(status="skipped", seconds=round(time.monotonic() - started, 1))
                return rec

        src = entry.context_source
        pulled_ref: str | None = None
        if isinstance(src, RegistrySource):
            rc, out = await _run(["docker", "pull", entry.image_ref], timeout_s=build_timeout_s)
            if rc != 0:
                raise _BuildError(f"docker pull {entry.image_ref} failed: {out[-800:]}")
            pulled_ref = entry.image_ref
            rc, out = await _run(["docker", "tag", entry.image_ref, tref], timeout_s=60.0)
            if rc != 0:
                raise _BuildError(f"docker tag failed: {out[-400:]}")
            built_kind = "pulled"
        elif isinstance(src, GitSource):
            clone = await clone_cache.ensure(
                src.repo, src.ref, timeout_s=max(build_timeout_s * 0.4, 120.0),
            )
            ctx = clone / src.subdir
            if not ctx.is_dir():
                raise _BuildError(f"subdir {src.subdir!r} not found in {src.repo}@{src.ref}")
            dockerfile = ctx / src.dockerfile
            if not dockerfile.is_file():
                raise _BuildError(f"{src.dockerfile!r} not found under {src.subdir!r}")
            rc, out = await _run(
                ["docker", "build", "-t", tref, "-f", str(dockerfile),
                 *_label_args(entry.labels), str(ctx)],
                timeout_s=build_timeout_s,
            )
            if rc != 0:
                raise _BuildError(f"docker build failed: {out[-1200:]}")
            built_kind = "built"
        elif isinstance(src, TarballSource):
            if src.content_b64 is None:
                raise _BuildError("tarball content not loaded (resolve_tarball_sources must run)")
            built_kind = await _build_tarball(
                tref, src, entry.labels, build_timeout_s=build_timeout_s,
            )
        elif isinstance(src, LocalSource):
            # The build context is already a directory on this build host (shared
            # FSx, per src.shared_fs) — build it where it sits. No clone, no
            # extract, no copy: the least-lossy source. Each Slurm shard reads
            # the same path off the shared filesystem.
            ctx = Path(src.path)
            if not ctx.is_dir():
                raise _BuildError(
                    f"local source path {src.path!r} is not a directory on this "
                    f"build host ({socket.gethostname()}) — local sources need "
                    f"the {src.shared_fs!r} shared filesystem mounted here. Is "
                    f"FSx mounted on this node?",
                )
            dockerfile = ctx / src.dockerfile
            if not dockerfile.is_file():
                raise _BuildError(
                    f"{src.dockerfile!r} not found under local source {src.path!r}",
                )
            rc, out = await _run(
                ["docker", "build", "-t", tref, "-f", str(dockerfile),
                 *_label_args(entry.labels), str(ctx)],
                timeout_s=build_timeout_s,
            )
            if rc != 0:
                raise _BuildError(f"docker build (local) failed: {out[-1200:]}")
            built_kind = "built"
        else:  # pragma: no cover - schema is a closed union
            raise _BuildError(f"unsupported context_source {type(src).__name__}")

        rc, out = await _run(["docker", "push", tref], timeout_s=build_timeout_s)
        if rc != 0:
            raise _BuildError(f"docker push {tref} failed: {out[-1200:]}")

        rc, out = await _run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", tref],
            timeout_s=60.0,
        )
        digest = None
        if rc == 0 and "@sha256:" in out:
            digest = out.strip().split("@", 1)[1]
        rec["digest"] = digest

        if prune:
            await _run(["docker", "rmi", tref], timeout_s=120.0)
            if pulled_ref and pulled_ref != tref:
                await _run(["docker", "rmi", pulled_ref], timeout_s=120.0)

        rec.update(status=built_kind, seconds=round(time.monotonic() - started, 1))
        return rec
    except _BuildError as exc:
        rec.update(error=str(exc), seconds=round(time.monotonic() - started, 1))
        return rec
    except Exception as exc:  # last-resort guard: one entry must not kill the shard
        rec.update(error=f"{type(exc).__name__}: {exc}", seconds=round(time.monotonic() - started, 1))
        return rec


async def _build_tarball(
    tref: str, src: TarballSource, labels: dict[str, str], *, build_timeout_s: float,
) -> str:
    assert src.content_b64 is not None
    content = base64.b64decode(src.content_b64)
    extract_dir = Path(tempfile.mkdtemp(prefix="xrlenv-bp-tarball-"))
    try:
        with tarfile.open(fileobj=io.BytesIO(content)) as tf:
            for m in tf.getmembers():
                target = (extract_dir / m.name).resolve()
                if not str(target).startswith(str(extract_dir.resolve())):
                    raise _BuildError(f"tarball entry {m.name!r} escapes build context")
            tf.extractall(extract_dir, filter="data")
        dockerfile = extract_dir / src.dockerfile
        if not dockerfile.is_file():
            raise _BuildError(f"{src.dockerfile!r} not found at tarball root")
        rc, out = await _run(
            ["docker", "build", "-t", tref, "-f", str(dockerfile),
             *_label_args(labels), str(extract_dir)],
            timeout_s=build_timeout_s,
        )
        if rc != 0:
            raise _BuildError(f"docker build (tarball) failed: {out[-1200:]}")
        return "built"
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


# ── Orchestration ─────────────────────────────────────────────────────────────


def _resolve_shard(args: argparse.Namespace) -> tuple[int, int]:
    """Pick (shard_index, num_shards): explicit flags win, else Slurm env, else
    (0, 1) = build everything on this one host."""
    def _envint(*names: str) -> int | None:
        for n in names:
            v = os.environ.get(n)
            if v and v.strip().isdigit():
                return int(v)
        return None

    num = args.num_shards
    if num is None:
        num = _envint("SLURM_NTASKS", "SLURM_ARRAY_TASK_COUNT") or 1
    idx = args.shard_index
    if idx is None:
        idx = _envint("SLURM_PROCID", "SLURM_ARRAY_TASK_ID") or 0
    return idx, num


def _should_prune(
    *, builds_since: int, prune_every: int, free_bytes: int, min_free_bytes: int,
) -> bool:
    """Whether to reclaim docker build cache now. Triggers on either threshold;
    each threshold is disabled when 0. Pure — unit-tested."""
    if prune_every > 0 and builds_since >= prune_every:
        return True
    return min_free_bytes > 0 and free_bytes < min_free_bytes


class _Pruner:
    """Periodically reclaims Docker build cache + dangling images so a long shard
    (hundreds of builds) does not fill the data-root. After each build it prunes
    when the build count since the last prune hits ``prune_every`` OR free space on
    the data-root drops below ``min_free_gb``. Serialized by a lock; ``docker
    builder prune -f`` only removes UNUSED cache (and ``image prune`` only dangling
    images), so it is safe to run while other builds are in flight."""

    def __init__(self, *, prune_every: int, min_free_gb: float, label: str) -> None:
        self._every = prune_every
        self._min_free_bytes = int(min_free_gb * 1_000_000_000)
        self._label = label
        self._lock = asyncio.Lock()
        self._since = 0
        self._data_root: str | None = None

    async def _docker_root(self) -> str:
        if self._data_root is None:
            rc, out = await _run(
                ["docker", "info", "--format", "{{.DockerRootDir}}"], timeout_s=30.0,
            )
            self._data_root = out.strip() if rc == 0 and out.strip() else "/var/lib/docker"
        return self._data_root

    @staticmethod
    def _free_bytes(root: str) -> int:
        try:
            return shutil.disk_usage(root).free
        except OSError:
            return 1 << 62  # unknown → don't trip the disk threshold

    async def after_build(self) -> None:
        if self._every <= 0 and self._min_free_bytes <= 0:
            return
        async with self._lock:
            self._since += 1
            root = await self._docker_root()
            free = self._free_bytes(root)
            if not _should_prune(
                builds_since=self._since, prune_every=self._every,
                free_bytes=free, min_free_bytes=self._min_free_bytes,
            ):
                return
            self._since = 0
            print(f"{self._label} pruning docker build cache "
                  f"(free {free / 1e9:.1f} GB on {root}) …", flush=True)
            await _run(["docker", "builder", "prune", "-f"], timeout_s=900.0)
            await _run(["docker", "image", "prune", "-f"], timeout_s=900.0)
            after = self._free_bytes(root)
            print(f"{self._label} prune done (free {free / 1e9:.1f} → "
                  f"{after / 1e9:.1f} GB)", flush=True)


async def _run_shard(
    shard_entries: list[BuildEntry], *, args: argparse.Namespace,
    shard_index: int, num_shards: int,
) -> list[dict[str, Any]]:
    # Default to the SHARED home cache (same physical FSx dir on every node), so
    # the repo is cloned once for the whole campaign, not once per shard.
    cache_root = (
        Path(args.clone_cache) if args.clone_cache
        else paths.build_context_cache_root()
    )
    clone_cache = _CloneCache(root=cache_root, refresh=args.refresh_context)
    pruner = _Pruner(
        prune_every=args.prune_every, min_free_gb=args.prune_min_free_gb,
        label=f"[shard {shard_index}/{num_shards}]",
    )
    sem = asyncio.Semaphore(args.concurrency)
    total = len(shard_entries)
    done = 0
    results: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def worker(entry: BuildEntry) -> None:
        nonlocal done
        async with sem:
            rec = await build_one(
                entry, registry=args.registry, scheme=args.registry_scheme,
                clone_cache=clone_cache, build_timeout_s=args.build_timeout,
                force=args.force, prune=not args.no_prune,
                probe_timeout_s=args.probe_timeout,
            )
        async with lock:
            done += 1
            results.append(rec)
            status = rec["status"]
            mark = {"built": "✓", "pulled": "✓", "skipped": "·"}.get(status, "✗")
            extra = f"  {rec['error']}" if rec.get("error") else f"  {rec['seconds']}s"
            print(
                f"[shard {shard_index}/{num_shards}] {done}/{total} {mark} "
                f"{status:<7} {rec['target_ref']}{extra}",
                flush=True,
            )
        # Outside the sem + record lock: reclaim build cache when count/disk
        # thresholds are hit, so a long shard does not fill the data-root.
        await pruner.after_build()

    await asyncio.gather(*(worker(e) for e in shard_entries))
    return results


def _write_report(
    path: Path, *, plan_name: str | None, registry: str, shard_index: int,
    num_shards: int, results: list[dict[str, Any]],
) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    report = {
        "plan_name": plan_name, "registry": registry,
        "shard_index": shard_index, "num_shards": num_shards,
        "counts": counts, "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="build_and_push_images",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--plan", required=True, help="Path to a build-plan.yaml (per-image-ref shape).")
    p.add_argument("--registry", required=True,
                   help="Private registry host:port to push to, e.g. <registry-host>:5011.")
    p.add_argument("--registry-scheme", default="http", choices=("http", "https"),
                   help="Scheme for the manifest existence probe (default http; "
                        "the private registry runs plain HTTP on the trusted VPC).")
    p.add_argument("--shard-index", type=int, default=None,
                   help="This shard's index (0-based). Default: $SLURM_PROCID / "
                        "$SLURM_ARRAY_TASK_ID, else 0.")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Total shard count. Default: $SLURM_NTASKS / "
                        "$SLURM_ARRAY_TASK_COUNT, else 1.")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Concurrent docker builds within this shard (default 4).")
    p.add_argument("--force", action="store_true",
                   help="Rebuild + repush even if the ref already exists in the registry.")
    p.add_argument("--no-prune", action="store_true",
                   help="Keep the local image tag after push (default: remove it; "
                        "shared base layers stay for build-cache reuse).")
    p.add_argument("--prune-every", type=int, default=25,
                   help="Run `docker builder prune` after every N builds in this "
                        "shard, so the data-root doesn't fill on a long run "
                        "(default 25; 0 disables).")
    p.add_argument("--prune-min-free-gb", type=float, default=30.0,
                   help="Also prune when free space on the docker data-root drops "
                        "below this many GB (default 30; 0 disables). This is the "
                        "real ENOSPC guard for heavy-tailed image sets.")
    p.add_argument("--build-timeout", type=float, default=3600.0,
                   help="Per-image build/pull/push timeout in seconds (default 3600).")
    p.add_argument("--probe-timeout", type=float, default=15.0,
                   help="Registry manifest-probe timeout in seconds (default 15).")
    p.add_argument("--tarball-max-bytes", type=int, default=DEFAULT_TARBALL_MAX_BYTES,
                   help="Max tarball-source build-context size to load (default 200MB).")
    p.add_argument("--clone-cache", default=None,
                   help="Dir for git checkouts (default ~/.xrlenv/build-context-cache, "
                        "the shared FSx cache — clone-once across all shards).")
    p.add_argument("--refresh-context", action="store_true",
                   help="Force a fresh git clone even if a cached checkout exists "
                        "(use when a moving ref like 'main' advanced to a new commit).")
    p.add_argument("--report", default=None,
                   help="JSON report path. Default: <plan-dir>/build-push-report"
                        ".shard<idx>of<n>.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Print this shard's assignment and exit without building.")
    p.add_argument("--mirror-registry", action="store_true",
                   help="Also process type: registry entries — pull each from its "
                        "origin (docker.io / ECR) and push it into the private "
                        "registry. OFF by default: prebuilt images are served at "
                        "runtime via the :5010 pull-through mirror (warmed by `xrlenv "
                        "build apply`), so this tool — which fills the :5011 PRIVATE "
                        "registry from Dockerfiles — has nothing to build for them.")
    args = p.parse_args(argv)

    plan_path = Path(args.plan)
    plan = load_build_plan(plan_path)
    if not plan.is_per_image_ref():
        print("ERROR: this tool only handles per-image-ref plans (top-level "
              "'entries:'), not the legacy 'benchmarks:' shape.", file=sys.stderr)
        return 2
    # Load any tarball-source bytes up front (reuses the build-apply helper).
    plan = resolve_tarball_sources(
        plan, max_bytes=args.tarball_max_bytes, base_dir=plan_path.parent,
    )

    shard_index, num_shards = _resolve_shard(args)
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        print(f"ERROR: invalid shard {shard_index}/{num_shards}.", file=sys.stderr)
        return 2

    all_entries = list(plan.entries)
    # A unified benchmark plan (e.g. LHTB's) mixes type: local (built here) with
    # type: registry (prebuilt docker.io/ECR, served at runtime via the :5010
    # pull-through mirror). This tool fills the :5011 PRIVATE registry from
    # Dockerfiles, so it has nothing to build for registry entries — skip them unless
    # --mirror-registry asks to copy them into the private registry.
    if not args.mirror_registry:
        buildable, registry_only = split_buildable(all_entries)
        if registry_only:
            n_skipped = len(registry_only)
            print(
                f"skipping {n_skipped} type: registry "
                f"entr{'y' if n_skipped == 1 else 'ies'} (prebuilt, served via the "
                f":5010 pull-through mirror; pass --mirror-registry to copy them into "
                f"{args.registry}).",
                flush=True,
            )
        all_entries = buildable
    shard_entries = select_shard(all_entries, shard_index, num_shards)
    shard_bytes = sum(int(e.placement.size_hint_bytes) for e in shard_entries)
    print(
        f"plan {plan.name or '(unnamed)'}: {len(all_entries)} entries → "
        f"shard {shard_index}/{num_shards} owns {len(shard_entries)} "
        f"(~{shard_bytes / 1e9:.1f} GB hinted) → registry {args.registry}",
        flush=True,
    )
    if args.dry_run:
        for e in shard_entries:
            print(f"  would build {target_ref(args.registry, e.image_ref)}  "
                  f"[{type(e.context_source).__name__}]")
        return 0
    if not shard_entries:
        print("nothing assigned to this shard; done.")
        return 0

    results = asyncio.run(_run_shard(
        shard_entries, args=args, shard_index=shard_index, num_shards=num_shards,
    ))

    report_path = Path(args.report) if args.report else (
        plan_path.parent / f"build-push-report.shard{shard_index}of{num_shards}.json"
    )
    _write_report(
        report_path, plan_name=plan.name, registry=args.registry,
        shard_index=shard_index, num_shards=num_shards, results=results,
    )
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    failed = counts.get("failed", 0)
    print(
        f"\nshard {shard_index}/{num_shards} done: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + f"  → report {report_path}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
