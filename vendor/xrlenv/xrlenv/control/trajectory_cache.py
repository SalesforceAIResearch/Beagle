"""Control-plane trajectory cache (spec 17 §"Control-plane cache").

Phase-0 surface: on-disk LRU + TTL at
``~/.xrlenv/admin-cache/trajectories/<rollout_id>.json``. The body is
``Trajectory.model_dump_json()``; cache hits skip the bidi gRPC fetch
entirely. Sealed trajectories are immutable (spec 00 invariant 3) so
the cache never invalidates anything older than the TTL or below the
LRU watermark.

The cache is **transport-agnostic** — callers hand in a
``fetch_fn`` callable that knows how to obtain a Trajectory from
whichever NodeTransport the rollout's owning node exposes. The admin
server resolves ``RolloutRecord.node_id`` against the live
``NodeRegistry`` and bridges the two; tests stub the fetch_fn for
unit-level isolation.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from xrlenv import paths
from xrlenv.types import Trajectory

LOGGER = logging.getLogger(__name__)

# ``$XRLENV_HOME/admin-cache/trajectories`` (default under ``~/.xrlenv``); see
# :mod:`xrlenv.paths`.
DEFAULT_CACHE_ROOT = paths.admin_cache_root()


class TrajectoryCacheConfig(BaseModel):
    """Tunables for :class:`TrajectoryCache`. Defaults match spec 17."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cache_root: Path = DEFAULT_CACHE_ROOT
    max_bytes: int = 5 * 1024**3
    """LRU watermark; the cache evicts the oldest files when total cached
    bytes would exceed this after a write. Default 5 GB per spec 17."""
    ttl_s: float = 60 * 60.0
    """Files older than ``ttl_s`` since their mtime are pruned on the next
    sweep + treated as cache miss on read. Default 1 h per spec 17."""


FetchFn = Callable[[str], Awaitable[Trajectory]]
"""``async def fetch(rollout_id) -> Trajectory`` — the cache calls this on
miss. Implementations bridge to ``NodeTransport.fetch_trajectory`` for
whichever node owns the rollout."""


class TrajectoryCache:
    """On-disk LRU + TTL cache for sealed trajectories.

    Concurrent reads of the same rollout coalesce onto a single fetch via
    an in-flight map (asyncio.Future); duplicate cache writes are a no-op
    because the on-disk file's atomic-rename ensures the last writer wins
    consistently.
    """

    def __init__(self, config: TrajectoryCacheConfig | None = None) -> None:
        self._cfg = config or TrajectoryCacheConfig()
        self._cfg.cache_root.mkdir(parents=True, exist_ok=True)
        # Per-rollout in-flight Future so concurrent get() calls for the
        # same id share one fetch.
        import asyncio
        self._inflight: dict[str, asyncio.Future[Trajectory]] = {}
        self._asyncio = asyncio

    @property
    def config(self) -> TrajectoryCacheConfig:
        return self._cfg

    async def get(self, rollout_id: str, fetch_fn: FetchFn) -> Trajectory:
        """Return the cached trajectory; on miss/stale, invoke ``fetch_fn``
        and write the result back through the LRU.
        """
        cached = self._read_disk(rollout_id)
        if cached is not None:
            return cached

        future = self._inflight.get(rollout_id)
        if future is not None:
            return await future

        future = self._asyncio.get_running_loop().create_future()
        self._inflight[rollout_id] = future
        try:
            trajectory = await fetch_fn(rollout_id)
        except BaseException as exc:
            future.set_exception(exc)
            # Eagerly retrieve the exception so asyncio doesn't log a
            # "Future exception was never retrieved" warning when no
            # concurrent caller picked the future up. Concurrent
            # awaiters that *did* pick it up still see the exception via
            # their await.
            future.exception()
            self._inflight.pop(rollout_id, None)
            raise
        self._inflight.pop(rollout_id, None)

        self._write_disk(rollout_id, trajectory)
        if not future.done():
            future.set_result(trajectory)
        self._evict_if_over_budget()
        return trajectory

    def invalidate(self, rollout_id: str) -> None:
        """Drop ``rollout_id``'s cached body. Sealed trajectories don't
        usually need this — call only when the operator changes a sink."""
        path = self._path_for(rollout_id)
        if path.exists():
            path.unlink()

    def sweep_expired(self) -> int:
        """Delete cached files older than ``ttl_s``; return the count.

        Operator entry point + admin-server background hook (slice 7b
        wires the periodic call in a follow-up). Independent from
        :py:meth:`_evict_if_over_budget` — TTL handles "too old," LRU
        handles "too big."
        """
        cutoff = time.time() - self._cfg.ttl_s
        pruned = 0
        for path in self._cfg.cache_root.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    pruned += 1
            except FileNotFoundError:
                continue
        return pruned

    # ── Internals ────────────────────────────────────────────────────────────

    def _path_for(self, rollout_id: str) -> Path:
        return self._cfg.cache_root / f"{rollout_id}.json"

    def _read_disk(self, rollout_id: str) -> Trajectory | None:
        path = self._path_for(rollout_id)
        if not path.exists():
            return None
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        if time.time() - stat.st_mtime > self._cfg.ttl_s:
            with suppress(FileNotFoundError):
                path.unlink()
            return None
        try:
            body = path.read_text(encoding="utf-8")
            traj = Trajectory.model_validate_json(body)
        except Exception:
            LOGGER.exception(
                "trajectory_cache: corrupt cache entry %s; dropping", path,
            )
            path.unlink(missing_ok=True)
            return None
        # Touch mtime so LRU eviction picks the genuinely-coldest file
        # rather than the one we just read.
        with suppress(OSError):
            path.touch()
        return traj

    def _write_disk(self, rollout_id: str, trajectory: Trajectory) -> None:
        path = self._path_for(rollout_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(trajectory.model_dump_json(), encoding="utf-8")
        tmp.replace(path)

    def _evict_if_over_budget(self) -> None:
        files = []
        total = 0
        for path in self._cfg.cache_root.glob("*.json"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
        if total <= self._cfg.max_bytes:
            return
        # Evict oldest-first until we're back under budget.
        files.sort(key=lambda triple: triple[0])
        for _mtime, size, path in files:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            total -= size
            if total <= self._cfg.max_bytes:
                break


def cache_size_bytes(cache_root: Path) -> int:
    """Convenience: total bytes in the cache dir (for the admin /health view)."""
    if not cache_root.exists():
        return 0
    return sum(p.stat().st_size for p in cache_root.glob("*.json"))


def _safely_dump(value: Any) -> str:
    """Round-trip ``value`` through json so non-serialisable nested types
    surface as a clean error rather than a binary tarpit."""
    return json.dumps(value, default=str)


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "FetchFn",
    "TrajectoryCache",
    "TrajectoryCacheConfig",
    "cache_size_bytes",
]
