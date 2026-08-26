"""Per-node source-build pipeline.

Handles ``BuildImageCommand`` dispatch from the control plane:

- ``GitSource`` entries: clone the repo at the specified ref into
  a build-context cache, run ``docker build``, tag the result
  with ``image_ref``.
- ``TarballSource`` entries: the operator-shipped bytes are written
  to a tempdir, untarred (gzip auto-detected), and ``docker build``
  runs against that tree. Built images carry the
  ``xrlenv.image.rebuild-cost=local-build-cheap`` label since
  recovery is just "re-ship the bytes" without a network round-trip.

Both source types share the same active-builds registry +
``cancel(image_ref)`` surface — operators can interrupt either.

Build-context cache layout::

    ~/.xrlenv/build-context-cache/
      <sha256(repo)[:12]>/
        <safe(ref)>/                # one dir per (repo, ref)

Cache rules:

- Persistent across builds so re-builds of the same ref are
  ``git fetch + checkout`` rather than full clone.
- Total cache size capped (default 5 GB); LRU eviction by
  whole-context when over the cap.
- Per-context > 5 GB falls back to ephemeral mode: cloned into a
  tempdir, deleted after the build.

Auto-labeling:

- ``xrlenv.image.rebuild-cost=local-build-expensive`` for git
  sources (clone + docker build is the highest-cost recovery).
- The label is reserved — operator-supplied labels merge on top
  but cannot override this one.

Concurrency: a per-instance asyncio.Semaphore serializes builds
within one node. Default 1 because ``docker build`` is CPU + IO
heavy and concurrent builds compete for the same buildkit cache.
Operators can tune via ``GitSourceBuilder(concurrent_builds=N)``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrlenv import paths
from xrlenv.control.build_plan import GitSource, TarballSource
from xrlenv.observability.tracing import get_tracer

LOGGER = logging.getLogger(__name__)

# Reserved Docker label that controls eviction-tier sorting in
# ``ImageCacheManager``. The source builder always sets this; any
# operator-supplied value with the same key is overridden.
REBUILD_COST_LABEL = "xrlenv.image.rebuild-cost"
REBUILD_COST_GIT = "local-build-expensive"
REBUILD_COST_TARBALL = "local-build-cheap"

# Reserved Docker label set on every git build so the cancel path
# can find any docker build container (the temporary container
# Docker spawns to run RUN steps in the Dockerfile) by image_ref.
# ``cancel(image_ref)`` queries ``docker ps --filter
# label=xrlenv.cancel-key=<image_ref>`` and force-kills the
# matches. Reserved like ``xrlenv.image.rebuild-cost`` —
# operator labels merge on top but cannot override.
CANCEL_KEY_LABEL = "xrlenv.cancel-key"

# Default cache-cap defaults. Operators can override via the
# ``GitSourceBuilder`` constructor; the values match the design
# notes in ``notes/source-build-dispatch.md`` (F2 lock).
DEFAULT_CACHE_TOTAL_CAP_BYTES = 5 * 1024**3   # 5 GB total cache cap
DEFAULT_PER_CONTEXT_MAX_BYTES = 5 * 1024**3   # >5 GB → ephemeral mode

# git clone with --depth=1 covers the common case (operators pin
# refs). If the operator pins a deep history ref the shallow clone
# still gets the tip; we don't try to fetch full history.
_GIT_CLONE_ARGS = ("git", "clone", "--depth=1")


def _safe_ref(ref: str) -> str:
    """Sanitize a git ref for use as a directory name. Refs can
    contain ``/`` (e.g. ``refs/heads/main``); replace with ``__``.
    Strips any other shell-unsafe characters."""
    return re.sub(r"[^A-Za-z0-9._-]", "__", ref) or "_unknown"


def _repo_hash(repo: str) -> str:
    """First 12 chars of sha256(repo URL). Stable across runs;
    different URLs (e.g. ``https://`` vs ``git@``) produce different
    cache keys, which is intended."""
    return hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12]


def _dir_size_bytes(path: Path) -> int:
    """Recursively sum file sizes under ``path``. Best-effort:
    skips files that disappear mid-walk (race with builds)."""
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _resolve_cache_root(explicit: Path | None) -> Path:
    """Pick the build-context cache root, in priority order:

    1. ``explicit`` constructor arg (test fixtures, custom deploys).
    2. ``$XRLENV_BUILD_CONTEXT_CACHE`` env var (operator-set on
       node systemd units when the default isn't writable).
    3. ``~/.xrlenv/build-context-cache/`` (matches the rest of the
       node's config layout — runs root, secrets, etc).
    4. ``/tmp/xrlenv-build-context-cache-<uid>/`` fallback when the
       home-based path isn't writable, e.g. on bootstrap-managed
       nodes that mount ``~/.xrlenv`` read-only. Logged so
       operators see the fallback firing.

    Always returns a path with the directory ``mkdir -p``'d. Raises
    ``OSError`` if even the /tmp fallback can't be created (extreme
    cases like full disk; not seen in practice).
    """
    if explicit is not None:
        explicit.mkdir(parents=True, exist_ok=True)
        return explicit
    env_path = os.environ.get("XRLENV_BUILD_CONTEXT_CACHE")
    if env_path:
        path = Path(env_path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    home_default = paths.build_context_cache_root()
    try:
        home_default.mkdir(parents=True, exist_ok=True)
        return home_default
    except OSError as exc:
        # Common on bootstrap-managed nodes that mount ``~/.xrlenv``
        # read-only. Fall back to a uid-scoped /tmp path so the
        # cache works without operator intervention. Operators
        # wanting a persistent cache across reboots should set
        # ``XRLENV_BUILD_CONTEXT_CACHE`` in their node systemd unit.
        uid = os.getuid() if hasattr(os, "getuid") else 0
        fallback = Path(f"/tmp/xrlenv-build-context-cache-{uid}")
        fallback.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            "build-context cache default %s is not writable (%s); "
            "falling back to %s. Set $XRLENV_BUILD_CONTEXT_CACHE on "
            "this node to a persistent path if you want the cache to "
            "survive reboots.",
            home_default, exc, fallback,
        )
        return fallback


@dataclass
class _CacheEntry:
    repo: str
    ref: str
    path: Path
    last_used_at: float


@dataclass
class BuildAndPushResult:
    """Outcome of :meth:`GitSourceBuilder.build_and_push` — the scratch
    build-on-demand flow (``notes/scratch-registry-build-on-demand.md``).

    ``repo_digest`` is the ``<repo>@sha256:...`` the registry assigned the
    pushed manifest; the control plane pins the run to it (invariant 4). It is
    ``None`` only when the push succeeded but the daemon reported no
    ``RepoDigests`` (should not happen against a conforming registry)."""

    status: str
    error: str | None = None
    repo_digest: str | None = None


def _strip_tag(image_ref: str) -> str:
    """Drop a trailing ``:tag`` (a ``:`` after the last ``/``), leaving the
    registry-qualified repository. Leaves a bare ``host:port/repo`` (no tag)
    untouched and never strips a registry-host port."""
    slash = image_ref.rfind("/")
    tail = image_ref[slash + 1:]
    if ":" in tail:
        return image_ref[: slash + 1] + tail.split(":", 1)[0]
    return image_ref


class GitSourceBuilder:
    """Clone-and-build pipeline for ``GitSource`` entries.

    One instance per node. Maintains a persistent build-context
    cache under ``~/.xrlenv/build-context-cache/`` (or
    ``cache_root``) and serializes builds via a semaphore.
    """

    def __init__(
        self,
        *,
        cache_root: Path | None = None,
        cache_total_cap_bytes: int = DEFAULT_CACHE_TOTAL_CAP_BYTES,
        per_context_max_bytes: int = DEFAULT_PER_CONTEXT_MAX_BYTES,
        concurrent_builds: int = 1,
        docker_client: Any | None = None,
        source_registry_root: Path | None = None,
    ) -> None:
        self._cache_root = _resolve_cache_root(cache_root)
        self._cache_total_cap_bytes = cache_total_cap_bytes
        self._per_context_max_bytes = per_context_max_bytes
        self._build_sem = asyncio.Semaphore(concurrent_builds)
        # Lazy-construct the docker client so test fixtures can
        # inject a fake without import-time daemon contact.
        self._docker_client = docker_client
        # In-memory LRU index. Populated from disk on first use.
        self._cache_lock = asyncio.Lock()
        self._cache_index: dict[tuple[str, str], _CacheEntry] | None = None
        # Active-build registry — one entry per image_ref currently
        # being built (registered at _build_git start, popped in
        # finally). Used by ``cancel(image_ref)`` to look up the
        # owning task and signal it.
        self._active_builds: dict[str, asyncio.Task[Any]] = {}
        # Sub-slice 2 — persistent per-image_ref source-spec registry.
        # When a build succeeds, we save the source spec
        # (GitSource / TarballSource bytes-and-all) to disk so a
        # later ``ensure_present`` after eviction can rebuild
        # without re-shipping from the operator. Lazy-loaded on
        # first ``lookup_producer`` to keep cold-start fast.
        self._source_registry_root = (
            source_registry_root or (self._cache_root.parent / "source-registry")
        )
        self._source_specs: dict[str, GitSource | TarballSource] | None = None
        # Scratch build-on-demand singleflight: coalesce concurrent
        # build_and_push calls for the same image_ref so a burst of rollouts
        # for one task builds + pushes exactly once on this node.
        self._inflight: dict[str, asyncio.Future[BuildAndPushResult]] = {}
        # Scratch build-on-demand registrations: content-addressed
        # scratch ref → build source. Shipped by the control plane for
        # scratch_build templates (like lazy benchmark-builder
        # registrations); in-memory, re-shipped on reconnect. When
        # ``lookup_producer`` finds a ref here it returns a producer that
        # builds AND pushes to the scratch registry embedded in the ref.
        self._scratch_specs: dict[str, GitSource | TarballSource] = {}
        # Scratch ref → durable destination ref. When set, the scratch
        # producer copies the built image to the user-owned durable registry
        # (digest-preserved) after build_and_push, so it survives scratch GC.
        self._scratch_durable: dict[str, str] = {}

    def register_scratch_source(
        self,
        image_ref: str,
        source: GitSource | TarballSource,
        *,
        durable_to: str | None = None,
    ) -> None:
        """Register a content-addressed scratch ref → build source.

        The control plane calls this for ``scratch_build`` templates so a
        later ``ensure_present(scratch_ref)`` on this node builds the context
        and pushes it to the scratch registry embedded in ``image_ref``
        (via :meth:`build_and_push`) instead of a plain local-tag build.

        ``durable_to`` (optional) is a user-owned registry ref; when set, the
        built image is also copied there (digest-preserved) so it outlives the
        scratch GC. A copy failure is non-fatal — the image is still usable
        from scratch — and is logged, not raised.
        """
        self._scratch_specs[image_ref] = source
        if durable_to:
            self._scratch_durable[image_ref] = durable_to
        else:
            self._scratch_durable.pop(image_ref, None)

    def _get_docker_client(self) -> Any:
        """Lazily construct (and memoize) the docker-py client so test
        fixtures can inject a fake without import-time daemon contact."""
        if self._docker_client is None:
            import docker
            self._docker_client = docker.from_env()
        return self._docker_client

    async def build_and_push(
        self,
        *,
        image_ref: str,
        source: GitSource | TarballSource,
        timeout_s: float,
        labels: dict[str, str] | None = None,
        check_registry_first: bool = True,
    ) -> BuildAndPushResult:
        """Build ``source``, push the result to the registry embedded in
        ``image_ref``, and resolve the pushed manifest digest.

        The scratch build-on-demand flow: ``image_ref`` is the content-
        addressed ``<scratch-host>/scratch/<input_digest>`` ref, so ``docker
        push`` targets that registry. Two properties:

        - **Singleflight** — concurrent calls for the same ``image_ref`` on
          this node are coalesced into one build+push.
        - **Build once for the fleet** — when ``check_registry_first`` and the
          ref already exists in the registry (another node built it), the
          local build is skipped and the existing digest is resolved by a
          cheap registry HEAD.

        Returns a :class:`BuildAndPushResult`; never raises for an ordinary
        build/push failure (surfaced as ``status="failed"``).
        """
        existing = self._inflight.get(image_ref)
        if existing is not None:
            return await existing
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[BuildAndPushResult] = loop.create_future()
        self._inflight[image_ref] = fut
        try:
            result = await self._build_and_push_impl(
                image_ref=image_ref, source=source, timeout_s=timeout_s,
                labels=labels, check_registry_first=check_registry_first,
            )
            if not fut.done():
                fut.set_result(result)
            return result
        except BaseException as exc:
            # Propagate to any coalesced waiter, then re-raise.
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            self._inflight.pop(image_ref, None)

    async def _build_and_push_impl(
        self,
        *,
        image_ref: str,
        source: GitSource | TarballSource,
        timeout_s: float,
        labels: dict[str, str] | None,
        check_registry_first: bool,
    ) -> BuildAndPushResult:
        start = time.monotonic()
        if check_registry_first:
            digest = await self._registry_digest_if_present(image_ref)
            if digest is not None:
                LOGGER.info(
                    "scratch build_and_push: %s already in registry "
                    "(digest %s) — skipping build", image_ref, digest,
                )
                return BuildAndPushResult("ok", None, digest)
        status, error = await self.build(
            image_ref=image_ref, source=source,
            timeout_s=timeout_s, labels=labels,
        )
        if status != "ok":
            return BuildAndPushResult("failed", error, None)
        remaining = max(timeout_s - (time.monotonic() - start), 60.0)
        try:
            digest = await self._push_and_resolve_digest(
                image_ref, timeout_s=remaining,
            )
        except _BuildError as exc:
            return BuildAndPushResult("failed", f"push failed: {exc}", None)
        return BuildAndPushResult("ok", None, digest)

    async def _registry_digest_if_present(self, image_ref: str) -> str | None:
        """Cheap registry HEAD: return ``<repo>@sha256:...`` if ``image_ref``
        already exists in its registry, else ``None``. Any error (absent,
        registry unreachable) maps to ``None`` so the caller builds."""
        client = self._get_docker_client()
        loop = asyncio.get_running_loop()

        def _probe() -> str | None:
            try:
                data = client.images.get_registry_data(image_ref)
            except Exception:
                return None
            digest = getattr(data, "id", None)
            if not digest:
                return None
            return f"{_strip_tag(image_ref)}@{digest}"

        return await loop.run_in_executor(None, _probe)

    async def _push_and_resolve_digest(
        self, image_ref: str, *, timeout_s: float,
    ) -> str | None:
        """Push ``image_ref`` to its registry and return the assigned
        ``<repo>@sha256:...``. Raises :class:`_BuildError` on push failure
        (docker-py reports push errors in the JSON stream rather than by
        raising, so we scan the stream)."""
        client = self._get_docker_client()
        loop = asyncio.get_running_loop()

        def _push() -> str | None:
            for chunk in client.images.push(image_ref, stream=True, decode=True):
                if isinstance(chunk, dict) and chunk.get("error"):
                    raise _BuildError(str(chunk["error"]))
            img = client.images.get(image_ref)
            repo_digests: list[str] = [
                str(d) for d in (img.attrs.get("RepoDigests") or [])
            ]
            want_repo = _strip_tag(image_ref)
            for rd in repo_digests:
                if rd.split("@", 1)[0] == want_repo:
                    return rd
            return repo_digests[0] if repo_digests else None

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _push), timeout=timeout_s,
            )
        except TimeoutError as exc:
            raise _BuildError(
                f"docker push exceeded {timeout_s:.0f}s deadline",
            ) from exc

    async def _copy_to_durable(
        self, src_ref: str, dst_ref: str, *, timeout_s: float,
    ) -> None:
        """Copy the freshly built+pushed scratch image ``src_ref`` to the
        user-owned ``dst_ref`` so it outlives the scratch GC. Prefers
        ``crane``/``skopeo`` (registry-to-registry, digest-preserving, no
        daemon); falls back to docker retag-and-push of the already-local
        image. Raises :class:`_BuildError` on failure."""
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: self._copy_blocking(src_ref, dst_ref),
            ),
            timeout=timeout_s,
        )

    def _copy_blocking(self, src_ref: str, dst_ref: str) -> None:
        crane = shutil.which("crane")
        if crane is not None:
            _run_blocking([crane, "copy", "--insecure", src_ref, dst_ref])
            return
        skopeo = shutil.which("skopeo")
        if skopeo is not None:
            _run_blocking([
                skopeo, "copy",
                "--src-tls-verify=false", "--dest-tls-verify=false",
                f"docker://{src_ref}", f"docker://{dst_ref}",
            ])
            return
        # Docker fallback: the image was just built locally as ``src_ref``;
        # retag to the durable ref and push. Content-identical (same layers),
        # though not manifest-digest-preserving the way crane/skopeo is.
        client = self._get_docker_client()
        repo = _strip_tag(dst_ref)
        tag = dst_ref[len(repo) + 1:] if len(dst_ref) > len(repo) else "latest"
        client.images.get(src_ref).tag(repo, tag=tag)
        for chunk in client.images.push(f"{repo}:{tag}", stream=True, decode=True):
            if isinstance(chunk, dict) and chunk.get("error"):
                raise _BuildError(f"durable push error: {chunk['error']}")

    async def build(
        self,
        *,
        image_ref: str,
        source: GitSource | TarballSource,
        timeout_s: float,
        labels: dict[str, str] | None = None,
        skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        with get_tracer().start_as_current_span(
            "xrlenv.node.source_build",
            attributes={
                "image_ref": image_ref,
                "source_type": type(source).__name__,
                "skip_if_present": skip_if_present,
                "timeout_s": timeout_s,
            },
        ):
            return await self._build_impl(
                image_ref=image_ref,
                source=source,
                timeout_s=timeout_s,
                labels=labels,
                skip_if_present=skip_if_present,
            )

    async def _build_impl(
        self,
        *,
        image_ref: str,
        source: GitSource | TarballSource,
        timeout_s: float,
        labels: dict[str, str] | None,
        skip_if_present: bool,
    ) -> tuple[str, str | None]:
        """Materialize ``image_ref`` from ``source`` on this node.

        ``skip_if_present=True``: if ``image_ref`` is already tagged
        locally, return ``("ok", None)`` without cloning, untarring,
        or invoking ``docker build``. Used by operator-driven
        ``xrlenv build apply --skip-if-present`` for warm-cluster
        re-applies (post-calibrate, partial-failure recovery). The
        source-spec persistence step still fires so a later
        ``acquire_container``-after-eviction can rebuild.

        Returns ``("ok", None)`` on success;
        ``("failed", "<message>")`` on any error.
        """
        if skip_if_present and self._image_present_locally(image_ref):
            LOGGER.info(
                "source-build: skipping %s — image already tagged "
                "locally (skip_if_present=True)", image_ref,
            )
            # Persist the source spec on the skip path too, so a
            # later acquire-after-eviction has the recipe. Idempotent
            # (the prior build already wrote the same spec, but a
            # missing one — e.g. operator wiped source-registry —
            # gets restored).
            self._persist_source_spec(image_ref, source)
            return ("ok", None)
        if isinstance(source, TarballSource):
            if source.content_b64 is None:
                return ("failed", (
                    "tarball source has no content_b64; the CLI's "
                    "resolve_tarball_sources helper must run before "
                    "dispatch (or the wire transport must populate it)"
                ))
            import base64
            try:
                content = base64.b64decode(source.content_b64)
            except Exception as exc:
                return ("failed", f"tarball content_b64 decode failed: {exc}")
            async with self._build_sem:
                status, error = await self._build_tarball(
                    image_ref=image_ref, content=content,
                    dockerfile=source.dockerfile or "Dockerfile",
                    timeout_s=timeout_s,
                    labels=dict(labels) if labels else {},
                )
            if status == "ok":
                self._persist_source_spec(image_ref, source)
            return (status, error)
        if not isinstance(source, GitSource):
            return ("failed", (
                f"unsupported source type {type(source).__name__}"
            ))
        async with self._build_sem:
            status, error = await self._build_git(
                image_ref=image_ref, source=source,
                timeout_s=timeout_s,
                labels=dict(labels) if labels else {},
            )
        if status == "ok":
            self._persist_source_spec(image_ref, source)
        return (status, error)

    def _image_present_locally(self, image_ref: str) -> bool:
        """Best-effort check via docker-py: is ``image_ref`` already
        tagged on the local daemon? Used by the ``skip_if_present``
        short-circuit. Treats any error (daemon unreachable, image
        absent) as "not present" so callers fall through to the
        normal build path."""
        client = self._docker_client
        if client is None:
            try:
                import docker
                client = docker.from_env()
                self._docker_client = client
            except Exception:
                return False
        try:
            client.images.get(image_ref)
            return True
        except Exception:
            return False

    async def _build_git(
        self, *, image_ref: str, source: GitSource,
        timeout_s: float, labels: dict[str, str],
    ) -> tuple[str, str | None]:
        start = time.monotonic()
        # Reserve the rebuild-cost + cancel-key labels; operator labels
        # merge on top but cannot override either.
        merged_labels = {
            **labels,
            REBUILD_COST_LABEL: REBUILD_COST_GIT,
            CANCEL_KEY_LABEL: image_ref,
        }
        # Register THIS task so ``cancel(image_ref)`` can find it and
        # raise CancelledError at our next await. Held until the build
        # returns (success or failure); popped in the outer finally so
        # an inflight cancel never targets a stale task ref.
        current = asyncio.current_task()
        if current is not None:
            self._active_builds[image_ref] = current
        try:
            # 1. Resolve / refresh the build context.
            try:
                context_dir, ephemeral = await self._resolve_context(
                    source, timeout_s=max(timeout_s * 0.4, 60.0),
                )
            except _BuildError as exc:
                return ("failed", f"context resolve failed: {exc}")
            except asyncio.CancelledError:
                # Operator-cancelled before ``docker build`` even
                # started. No build container to kill on this path.
                return ("failed", "cancelled by operator")
            try:
                # 2. Run docker build against ``context_dir / source.subdir``.
                build_root = context_dir / source.subdir
                if not build_root.is_dir():
                    return ("failed", (
                        f"git context {source.repo}@{source.ref}: subdir "
                        f"{source.subdir!r} not found after clone"
                    ))
                elapsed = time.monotonic() - start
                remaining = max(timeout_s - elapsed, 60.0)
                try:
                    await self._docker_build(
                        image_ref=image_ref, build_root=build_root,
                        dockerfile=source.dockerfile,
                        labels=merged_labels, timeout_s=remaining,
                    )
                except _BuildError as exc:
                    return ("failed", f"docker build failed: {exc}")
                except asyncio.CancelledError:
                    # Operator-cancelled mid-build. The cancel handler
                    # already best-effort killed any container labeled
                    # ``xrlenv.cancel-key=<image_ref>``; the ``docker
                    # build`` thread observes that as the build failing
                    # and returns. We swallow CancelledError so the
                    # outer command_id wait completes normally with a
                    # ``failed: cancelled`` reply rather than surfacing
                    # an uncaught exception.
                    return ("failed", "cancelled by operator")
                return ("ok", None)
            finally:
                if ephemeral:
                    # Clean up oversized ephemeral checkouts so
                    # subsequent builds don't fill the disk.
                    shutil.rmtree(context_dir, ignore_errors=True)
        finally:
            self._active_builds.pop(image_ref, None)

    async def _build_tarball(
        self, *, image_ref: str, content: bytes,
        dockerfile: str, timeout_s: float,
        labels: dict[str, str],
    ) -> tuple[str, str | None]:
        """Materialize ``image_ref`` from a docker-context tarball.

        Same active-builds + cancel surface as ``_build_git``: the
        task self-registers, gets the cancel-key + rebuild-cost
        labels, and converts ``CancelledError`` to a normal
        ``("failed", "cancelled by operator")`` return.

        Tarball bytes are written to a private tempdir and untarred
        with ``tarfile.open`` (gzip auto-detected via the magic
        bytes). Compared to git, there's no persistent build-context
        cache — the bytes are the build context, and the operator
        re-ships them on every apply (cheap path: rebuild cost is
        ``local-build-cheap`` since recovery doesn't need network).
        """
        import io
        import tarfile

        merged_labels = {
            **labels,
            REBUILD_COST_LABEL: REBUILD_COST_TARBALL,
            CANCEL_KEY_LABEL: image_ref,
        }
        current = asyncio.current_task()
        if current is not None:
            self._active_builds[image_ref] = current
        try:
            extract_dir = Path(tempfile.mkdtemp(
                prefix="xrlenv-build-tarball-",
            ))
            try:
                # Untar bytes synchronously — the bottleneck is fs
                # writes, not CPU; offloading to a thread doesn't
                # buy much for typical <100 MB contexts. Wrap in
                # try/except so a malformed tar surfaces a clean
                # error instead of an unstructured TarError.
                try:
                    with tarfile.open(
                        fileobj=io.BytesIO(content),
                    ) as tf:
                        # Best-effort path traversal guard: refuse
                        # any member whose normalized path escapes
                        # the extract dir. Not bullet-proof against
                        # all symlink shenanigans; the daemon-level
                        # ProtectSystem=strict + per-systemd-unit
                        # tmpfs are the deeper guard.
                        for m in tf.getmembers():
                            target = (extract_dir / m.name).resolve()
                            if not str(target).startswith(
                                str(extract_dir.resolve()),
                            ):
                                return ("failed", (
                                    f"tarball entry {m.name!r} would "
                                    f"escape build context dir "
                                    f"{extract_dir}"
                                ))
                        # ``filter="data"`` is the Python 3.14
                        # forward-compat default — strips
                        # owner/group/setuid bits and refuses
                        # device/special files. We've already done
                        # the path-traversal check above so the
                        # filter is belt + suspenders.
                        tf.extractall(extract_dir, filter="data")
                except tarfile.TarError as exc:
                    return ("failed", f"tarball extraction failed: {exc}")
                except asyncio.CancelledError:
                    return ("failed", "cancelled by operator")

                if not (extract_dir / dockerfile).is_file():
                    return ("failed", (
                        f"tarball context for {image_ref}: "
                        f"{dockerfile!r} not found at the tarball "
                        f"root after extraction"
                    ))

                try:
                    await self._docker_build(
                        image_ref=image_ref, build_root=extract_dir,
                        dockerfile=dockerfile,
                        labels=merged_labels, timeout_s=timeout_s,
                    )
                except _BuildError as exc:
                    return ("failed", f"docker build failed: {exc}")
                except asyncio.CancelledError:
                    return ("failed", "cancelled by operator")
                return ("ok", None)
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)
        finally:
            self._active_builds.pop(image_ref, None)

    async def _resolve_context(
        self, source: GitSource, *, timeout_s: float,
    ) -> tuple[Path, bool]:
        """Return ``(context_dir, ephemeral)`` — a directory holding
        a checkout of ``source.repo`` at ``source.ref``. ``ephemeral``
        is True when the checkout is larger than the per-context cap
        and the caller should ``shutil.rmtree`` after the build."""
        cache_key = (source.repo, source.ref)
        async with self._cache_lock:
            await self._ensure_cache_index_loaded()
            assert self._cache_index is not None
            cached = self._cache_index.get(cache_key)

        if cached is not None and cached.path.is_dir():
            # Cache hit. Refresh the working tree to the requested
            # ref in case the operator pinned a moving ref like
            # ``main`` and upstream advanced.
            try:
                await _run_subprocess(
                    ["git", "fetch", "--depth=1", "origin", source.ref],
                    cwd=cached.path, timeout_s=timeout_s,
                )
                await _run_subprocess(
                    ["git", "checkout", "FETCH_HEAD"],
                    cwd=cached.path, timeout_s=30.0,
                )
            except _BuildError:
                # Refresh failed (network blip, ref renamed). Fall
                # through to clean clone rather than serve stale.
                shutil.rmtree(cached.path, ignore_errors=True)
                cached = None
            else:
                cached.last_used_at = time.time()
                return (cached.path, False)

        # Cache miss (or refresh-failed cache hit). Clone fresh.
        repo_dir = self._cache_root / _repo_hash(source.repo)
        repo_dir.mkdir(parents=True, exist_ok=True)
        ref_dir = repo_dir / _safe_ref(source.ref)
        if ref_dir.exists():
            shutil.rmtree(ref_dir, ignore_errors=True)
        clone_target: Path
        # Clone into a tempdir first so we can size-check before
        # promoting to the persistent cache.
        with tempfile.TemporaryDirectory(prefix="xrlenv-build-clone-") as tmp:
            tmp_target = Path(tmp) / "checkout"
            try:
                await _run_subprocess(
                    [*_GIT_CLONE_ARGS, "--branch", source.ref,
                     source.repo, str(tmp_target)],
                    cwd=None, timeout_s=timeout_s,
                )
            except _BuildError as exc:
                raise _BuildError(
                    f"git clone {source.repo}@{source.ref} failed: {exc}"
                ) from exc
            size = _dir_size_bytes(tmp_target)
            if size > self._per_context_max_bytes:
                # Oversize: promote to a fresh ephemeral location
                # outside the persistent cache. Caller will rmtree
                # after the build.
                ephemeral_target = Path(tempfile.mkdtemp(
                    prefix="xrlenv-build-ephemeral-",
                ))
                shutil.rmtree(ephemeral_target, ignore_errors=True)
                shutil.copytree(tmp_target, ephemeral_target)
                LOGGER.info(
                    "build context %s@%s is %.1f MB > %.1f MB cap; "
                    "using ephemeral mode",
                    source.repo, source.ref,
                    size / 1024**2,
                    self._per_context_max_bytes / 1024**2,
                )
                return (ephemeral_target, True)
            # Promote to persistent cache.
            shutil.copytree(tmp_target, ref_dir)
            clone_target = ref_dir

        async with self._cache_lock:
            assert self._cache_index is not None
            self._cache_index[cache_key] = _CacheEntry(
                repo=source.repo, ref=source.ref,
                path=clone_target, last_used_at=time.time(),
            )
            await self._evict_to_cap_locked()
        return (clone_target, False)

    async def _ensure_cache_index_loaded(self) -> None:
        if self._cache_index is not None:
            return
        index: dict[tuple[str, str], _CacheEntry] = {}
        # Cache layout: <root>/<repo_hash>/<ref>/. We don't carry
        # the original repo URL on disk (only its hash), so existing
        # entries get an empty repo string and are LRU-eligible
        # purely by mtime; the next build with a known repo URL
        # overwrites the index entry with the right key.
        if self._cache_root.is_dir():
            for repo_dir in self._cache_root.iterdir():
                if not repo_dir.is_dir():
                    continue
                for ref_dir in repo_dir.iterdir():
                    if not ref_dir.is_dir():
                        continue
                    try:
                        last_used = ref_dir.stat().st_mtime
                    except OSError:
                        continue
                    key = (f"<unknown:{repo_dir.name}>", ref_dir.name)
                    index[key] = _CacheEntry(
                        repo=key[0], ref=key[1], path=ref_dir,
                        last_used_at=last_used,
                    )
        self._cache_index = index

    async def _evict_to_cap_locked(self) -> None:
        assert self._cache_index is not None
        # Total size of the persistent cache. Cheap because we only
        # walk top-level dirs once per build.
        total = sum(
            _dir_size_bytes(e.path) for e in self._cache_index.values()
            if e.path.is_dir()
        )
        if total <= self._cache_total_cap_bytes:
            return
        ordered = sorted(
            self._cache_index.values(),
            key=lambda e: e.last_used_at,
        )
        for entry in ordered:
            if total <= self._cache_total_cap_bytes:
                break
            size = _dir_size_bytes(entry.path)
            shutil.rmtree(entry.path, ignore_errors=True)
            self._cache_index.pop(
                (entry.repo, entry.ref), None,
            )
            total -= size
            LOGGER.info(
                "evicted build context %s@%s (%.1f MB) to honor "
                "%.1f MB cache cap",
                entry.repo, entry.ref,
                size / 1024**2,
                self._cache_total_cap_bytes / 1024**2,
            )

    # ── Source-spec registry (sub-slice 2 — build-on-acquire) ─────────────

    def _registry_dir_for(self, image_ref: str) -> Path:
        """Stable per-image_ref registry path. The image_ref hash
        keeps the path filesystem-safe (image_refs commonly contain
        ``/`` and ``:``)."""
        h = hashlib.sha256(image_ref.encode("utf-8")).hexdigest()[:32]
        return self._source_registry_root / h

    def _load_source_specs(self) -> dict[str, GitSource | TarballSource]:
        """Walk the on-disk registry and reconstruct in-memory
        per-image_ref source specs.

        Best-effort: partially-corrupt entries (missing spec.json,
        missing tarball content.bin, malformed JSON) are skipped
        with a warning rather than crashing the daemon.
        """
        result: dict[str, GitSource | TarballSource] = {}
        if not self._source_registry_root.is_dir():
            return result
        import base64

        for entry_dir in self._source_registry_root.iterdir():
            if not entry_dir.is_dir():
                continue
            spec_path = entry_dir / "spec.json"
            if not spec_path.is_file():
                continue
            try:
                spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
                image_ref = spec_dict.get("image_ref")
                kind = spec_dict.get("type")
                if not isinstance(image_ref, str):
                    continue
                if kind == "git":
                    result[image_ref] = GitSource(
                        repo=spec_dict["repo"], ref=spec_dict["ref"],
                        subdir=spec_dict.get("subdir", "."),
                        dockerfile=spec_dict.get("dockerfile", "Dockerfile"),
                    )
                elif kind == "tarball":
                    content_path = entry_dir / "content.bin"
                    if not content_path.is_file():
                        continue
                    content_b64 = base64.b64encode(
                        content_path.read_bytes(),
                    ).decode("ascii")
                    result[image_ref] = TarballSource(
                        path=spec_dict.get("path", "<persisted>"),
                        dockerfile=spec_dict.get("dockerfile", "Dockerfile"),
                        content_b64=content_b64,
                    )
            except Exception as exc:
                LOGGER.warning(
                    "source-registry: failed to load %s: %s",
                    entry_dir, exc,
                )
                continue
        return result

    def _persist_source_spec(
        self, image_ref: str, source: GitSource | TarballSource,
    ) -> None:
        """Write the per-image_ref source spec to disk so a later
        ``lookup_producer`` after restart finds it. Tarball bytes
        live alongside the spec.json in ``content.bin`` to keep
        the JSON small + human-readable.
        """
        import base64

        entry_dir = self._registry_dir_for(image_ref)
        entry_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(source, GitSource):
            spec_dict = {
                "type": "git",
                "image_ref": image_ref,
                "repo": source.repo,
                "ref": source.ref,
                "subdir": source.subdir,
                "dockerfile": source.dockerfile,
            }
        else:
            assert isinstance(source, TarballSource)
            assert source.content_b64 is not None
            (entry_dir / "content.bin").write_bytes(
                base64.b64decode(source.content_b64),
            )
            spec_dict = {
                "type": "tarball",
                "image_ref": image_ref,
                "path": source.path,
                "dockerfile": source.dockerfile,
            }
        (entry_dir / "spec.json").write_text(
            json.dumps(spec_dict, indent=2), encoding="utf-8",
        )
        # Update the in-memory mirror if it's been loaded.
        if self._source_specs is not None:
            self._source_specs[image_ref] = source

    def lookup_producer(
        self, image_ref: str,
    ) -> Callable[[str, float], Awaitable[None]] | None:
        """Image-cache builder hook for sub-slice 2 build-on-acquire.

        Returns an async producer that re-runs the registered build
        if a source spec is on disk for ``image_ref``. The image
        cache calls this on cache miss before falling through to
        ``backend.pull_image`` — so an evicted git/tarball-built
        image rebuilds automatically on next ``acquire_container``,
        without operator intervention.

        Returns ``None`` if no source spec is registered for the
        ref (the cache then attempts a registry pull, matching the
        old behavior for unregistered refs).
        """
        # Scratch build-on-demand takes priority: build AND push to the
        # content-addressed scratch registry embedded in the ref, so the
        # image is materialized once for the fleet and pulled by peers
        # (notes/scratch-registry-build-on-demand.md).
        scratch_source = self._scratch_specs.get(image_ref)
        if scratch_source is not None:
            durable_to = self._scratch_durable.get(image_ref)

            async def _produce_scratch(_ref: str, timeout_s: float) -> None:
                result = await self.build_and_push(
                    image_ref=_ref, source=scratch_source,
                    timeout_s=timeout_s, labels={},
                )
                if result.status != "ok":
                    raise RuntimeError(
                        f"scratch build-and-push failed for {_ref}: "
                        f"{result.error or 'unknown'}",
                    )
                if durable_to:
                    # Best-effort: the image is already usable from scratch, so
                    # a durable-copy failure must not fail the rollout.
                    try:
                        await self._copy_to_durable(
                            _ref, durable_to, timeout_s=max(timeout_s, 120.0),
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "scratch: durable copy of %s -> %s failed "
                            "(image still available in scratch): %s",
                            _ref, durable_to, exc,
                        )

            return _produce_scratch

        if self._source_specs is None:
            self._source_specs = self._load_source_specs()
        spec = self._source_specs.get(image_ref)
        if spec is None:
            return None

        async def _produce(_ref: str, timeout_s: float) -> None:
            status, error = await self.build(
                image_ref=_ref, source=spec,
                timeout_s=timeout_s, labels={},
            )
            if status != "ok":
                raise RuntimeError(
                    f"source-registry rebuild failed for {_ref}: "
                    f"{error or 'unknown'}",
                )

        return _produce

    async def cancel(self, image_ref: str) -> tuple[str, str | None]:
        """Cancel an in-flight git build.

        Two layers of cancellation, both best-effort:

        1. ``asyncio.Task.cancel()`` on the registered build task —
           raises ``CancelledError`` at the next await inside
           ``_build_git`` (clone, checkout, the docker build worker
           thread completing). The ``CancelledError`` handler returns
           ``("failed", "cancelled by operator")`` so the outer
           ``BuildImageCommand`` command_id completes with a normal
           reply.

        2. ``docker kill`` on any container labeled ``xrlenv.cancel-key
           =<image_ref>`` — the temporary build containers Docker
           spawns to run RUN steps in the Dockerfile. Killing them
           causes ``docker build`` to fail-fast inside the worker
           thread; without this, the SDK would block on the build
           reaching its own end-of-step boundary before observing the
           cancel.

        Returns ``("ok", None)`` even when there's nothing to cancel
        (no in-flight build) — the operator-facing semantics treat
        cancel as idempotent. Returns ``("failed", msg)`` only on
        internal errors (e.g. docker daemon unreachable). Run order:
        kill containers FIRST (so the worker thread's ``docker build``
        SDK call sees the failure inside its own deadline), THEN
        cancel the asyncio task (so the cleanup path runs after the
        thread returns).
        """
        had_active = image_ref in self._active_builds
        kill_count = 0
        kill_error: str | None = None
        try:
            kill_count = await self._kill_build_containers(image_ref)
        except Exception as exc:
            kill_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "cancel(%s): docker kill failed: %s",
                image_ref, kill_error,
            )
        task = self._active_builds.get(image_ref)
        if task is not None and not task.done():
            task.cancel()
        if kill_error and kill_count == 0 and not had_active:
            # Docker failed AND we couldn't find anything to cancel —
            # surface the docker error so the operator sees that
            # cluster state is degraded (vs. silently reporting "ok").
            return ("failed", kill_error)
        if not had_active and kill_count == 0:
            # Race with normal completion, OR cancel arrived after
            # the assignment finished, OR cancel for an image_ref
            # that was never building on this node. All of these are
            # operator-visible no-ops.
            return ("ok", None)
        return ("ok", None)

    async def _kill_build_containers(self, image_ref: str) -> int:
        """Find docker containers carrying ``xrlenv.cancel-key=<image_ref>``
        and force-kill them. Returns the count killed (0 when there are
        no matches). Best-effort: docker daemon errors propagate so the
        caller can surface them; per-container kill errors are logged
        and ignored so one stuck container doesn't block the others.

        Runs the docker-py SDK call in a thread to avoid blocking the
        event loop on a slow daemon.
        """
        client = self._docker_client
        if client is None:
            import docker
            client = docker.from_env()
            self._docker_client = client
        loop = asyncio.get_running_loop()

        def _list_and_kill() -> int:
            containers = client.containers.list(
                filters={"label": f"{CANCEL_KEY_LABEL}={image_ref}"},
            )
            killed = 0
            for c in containers:
                try:
                    c.kill()
                    killed += 1
                except Exception as exc:
                    LOGGER.warning(
                        "cancel(%s): kill of container %s failed: %s",
                        image_ref, getattr(c, "id", "?"), exc,
                    )
            return killed

        return await loop.run_in_executor(None, _list_and_kill)

    async def _docker_build(
        self, *, image_ref: str, build_root: Path,
        dockerfile: str, labels: dict[str, str],
        timeout_s: float,
    ) -> None:
        """Run ``docker build`` via the docker-py SDK. Errors are
        re-raised as ``_BuildError`` so the caller can surface them
        to the operator without leaking docker-py exception types."""
        client = self._docker_client
        if client is None:
            import docker
            client = docker.from_env()
            self._docker_client = client
        # docker-py's build API is synchronous; offload to a thread
        # so we don't block the event loop. Timeout is enforced by
        # docker-py via the ``timeout`` kwarg.
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _docker_build_blocking(
                        client=client, image_ref=image_ref,
                        build_root=build_root, dockerfile=dockerfile,
                        labels=labels,
                    ),
                ),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            raise _BuildError(
                f"docker build exceeded {timeout_s:.0f}s deadline"
            ) from exc


def _docker_build_blocking(
    *, client: Any, image_ref: str, build_root: Path,
    dockerfile: str, labels: dict[str, str],
) -> None:
    """Synchronous body of the docker build. Runs in a thread pool
    via ``asyncio.run_in_executor``."""
    try:
        client.images.build(
            path=str(build_root),
            dockerfile=dockerfile,
            tag=image_ref,
            labels=labels,
            forcerm=True,
            rm=True,
        )
    except Exception as exc:
        # docker-py raises BuildError / APIError; keep the original
        # message for operator visibility but wrap so the caller
        # doesn't depend on docker-py exception types.
        raise _BuildError(f"{type(exc).__name__}: {exc}") from exc


class _BuildError(RuntimeError):
    """Internal exception used to signal any build failure between
    the resolve / build phases. Translated to ``("failed", msg)`` at
    the public boundary so callers don't pattern-match exceptions."""


def _run_blocking(cmd: list[str]) -> None:
    """Run ``cmd`` to completion, raising :class:`_BuildError` on non-zero
    exit (used by the crane/skopeo durable-copy path, which runs in a thread)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        out = f"{proc.stdout}{proc.stderr}".strip()
        raise _BuildError(
            f"command {cmd[0]!r} exited {proc.returncode}: {out[:500]}",
        )


async def _run_subprocess(
    cmd: list[str], *, cwd: Path | None, timeout_s: float,
) -> None:
    """Run ``cmd`` and raise ``_BuildError`` on non-zero exit or
    timeout. Captures stdout+stderr for the error message."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s,
        )
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise _BuildError(
            f"command {cmd[0]!r} exceeded {timeout_s:.0f}s deadline"
        ) from exc
    if proc.returncode != 0:
        body = (stdout or b"").decode("utf-8", errors="replace").strip()
        raise _BuildError(
            f"command {cmd[0]!r} exited {proc.returncode}: {body[:500]}"
        )


__all__ = [
    "CANCEL_KEY_LABEL",
    "DEFAULT_CACHE_TOTAL_CAP_BYTES",
    "DEFAULT_PER_CONTEXT_MAX_BYTES",
    "REBUILD_COST_GIT",
    "REBUILD_COST_LABEL",
    "REBUILD_COST_TARBALL",
    "BuildAndPushResult",
    "GitSourceBuilder",
]
