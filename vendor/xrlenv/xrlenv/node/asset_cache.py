"""Per-node asset cache (spec 06 Pattern B + spec 15 §"Assets").

Companion to :class:`xrlenv.node.image_cache.ImageCacheManager`. Tracks
per-node :class:`AssetSpec` blobs with the same priority tiers (in_use,
pinned, recently_used, cold), the same LRU eviction logic, and the
same ``ensure_present`` / ``acquire`` / ``release`` lifecycle.

What's different from images:

- The fetcher is an :class:`AssetFetcher` (HTTP / S3 / GCS / HF), not
  ``docker pull``.
- The eviction unit is a file or extracted directory at
  ``extract_to``, not a Docker layer set.
- The mount-time hook is the runtime asking ``host_path_for(asset_id)``
  so the backend can substitute concrete bind-mount paths into the
  template's ``MountSpec``s.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from xrlenv.control.assets import (
    AssetFetchError,
    AssetRecord,
    AssetSpec,
    asset_default_root,
    evict_asset,
    extract_asset,
    fetcher_for,
    verify_existing,
)
from xrlenv.node.image_cache import ImageTier  # reuse the literal alias

LOGGER = logging.getLogger(__name__)


class AssetCacheConfig(BaseModel):
    """Tunables (mirrors :class:`ImageCacheConfig` defaults)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cache_root: Path = Path("/var/cache/xrlenv/assets")
    """Default ``extract_to`` parent when an :class:`AssetSpec` doesn't
    declare its own. Each asset gets a subdir named after its id."""
    evict_threshold_bytes: int = 5 * 1024**3
    evict_target_bytes: int = 10 * 1024**3
    recent_window_s: float = 30 * 60.0


class AssetReportEntry(BaseModel):
    """Per-asset row in :class:`NodeAssetReport`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tier: ImageTier
    size_bytes: int
    in_use_count: int
    last_used_at: float | None
    pinned: bool
    path: Path


class NodeAssetReport(BaseModel):
    """Snapshot of one node's asset cache state."""

    model_config = ConfigDict(extra="forbid")

    assets: list[AssetReportEntry] = Field(default_factory=list)
    pinned: tuple[str, ...] = ()


class AssetCacheManager:
    """LRU + pin + refcount asset cache for one node."""

    def __init__(
        self,
        *,
        config: AssetCacheConfig | None = None,
        pins: set[str] | None = None,
    ) -> None:
        self._cfg = config or AssetCacheConfig()
        self._cfg.cache_root.mkdir(parents=True, exist_ok=True)
        self._pins: set[str] = set(pins or set())
        self._records: dict[str, AssetRecord] = {}
        self._in_use: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._fetch_locks: dict[str, asyncio.Lock] = {}

    # ── State queries ──────────────────────────────────────────────────────

    @property
    def config(self) -> AssetCacheConfig:
        return self._cfg

    @property
    def pins(self) -> frozenset[str]:
        return frozenset(self._pins)

    def is_pinned(self, asset_id: str) -> bool:
        return asset_id in self._pins

    def in_use_count(self, asset_id: str) -> int:
        return self._in_use.get(asset_id, 0)

    def host_path_for(self, asset_id: str) -> Path | None:
        """Return the on-disk path the runtime should bind-mount, or
        ``None`` when the asset isn't cached. Used by the node-agent's
        mount-resolution step at create-sandbox time.
        """
        record = self._records.get(asset_id)
        return record.path if record is not None else None

    def tier(self, asset_id: str, *, now: float | None = None) -> ImageTier:
        """Spec-15 tier classification (collapsed to the four phase-0
        buckets, same as :class:`ImageCacheManager.tier`)."""
        if self._in_use.get(asset_id, 0) > 0:
            return "in_use"
        if asset_id in self._pins:
            return "pinned"
        ts = self._last_used.get(asset_id)
        cur = now if now is not None else time.monotonic()
        if ts is not None and cur - ts <= self._cfg.recent_window_s:
            return "recently_used"
        return "cold"

    # ── Refcount hooks ─────────────────────────────────────────────────────

    def acquire(self, asset_id: str) -> None:
        self._in_use[asset_id] = self._in_use.get(asset_id, 0) + 1
        self._last_used[asset_id] = time.monotonic()

    def release(self, asset_id: str) -> None:
        cur = self._in_use.get(asset_id, 0)
        if cur <= 1:
            self._in_use.pop(asset_id, None)
        else:
            self._in_use[asset_id] = cur - 1
        self._last_used[asset_id] = time.monotonic()

    def pin(self, asset_id: str) -> None:
        self._pins.add(asset_id)

    def unpin(self, asset_id: str) -> None:
        self._pins.discard(asset_id)

    # ── Fetch + evict ──────────────────────────────────────────────────────

    async def ensure_present(self, spec: AssetSpec) -> Path:
        """Download (and extract) ``spec`` if not already cached.

        Returns the on-disk path the runtime should bind-mount. Concurrent
        callers for the same asset coalesce onto a single in-flight
        fetch via :py:meth:`_fetch_lock_for`.
        """
        existing = self._records.get(spec.id)
        if existing is not None and existing.path.exists():
            self._last_used[spec.id] = time.monotonic()
            return existing.path

        async with self._fetch_lock_for(spec.id):
            existing = self._records.get(spec.id)
            if existing is not None and existing.path.exists():
                self._last_used[spec.id] = time.monotonic()
                return existing.path
            await self._evict_if_needed(incoming_bytes=spec.size_bytes)
            path = await asyncio.to_thread(self._fetch_blocking, spec)
            self._records[spec.id] = AssetRecord(
                id=spec.id,
                path=path,
                size_bytes=spec.size_bytes,
                sha256=spec.sha256,
                last_used_at=time.monotonic(),
            )
            self._last_used[spec.id] = time.monotonic()
            return path

    def _fetch_blocking(self, spec: AssetSpec) -> Path:
        """Synchronous fetch + extract. Runs in a thread so the event
        loop stays responsive on multi-GB downloads."""
        target_dir = (
            Path(spec.extract_to)
            if spec.extract_to is not None
            else asset_default_root(spec.id)
        )
        if spec.extract == "none":
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / Path(spec.source).name
            if verify_existing(spec, target_path):
                return target_path
        else:
            target_path = target_dir
            # When extracted, presence + a non-empty marker is enough;
            # full re-checksum on disk is too expensive for multi-GB
            # extracted directories. The fetcher already verified the
            # digest pre-extract; if the dir is gone we re-fetch.
            if target_path.exists() and any(target_path.iterdir()):
                return target_path

        fetcher = fetcher_for(spec.source)
        if fetcher is None:
            raise AssetFetchError(
                "no_fetcher",
                f"asset {spec.id!r}: no AssetFetcher claims source "
                f"{spec.source!r} (registered: "
                f"{[type(f).__name__ for f in __import__('xrlenv.control.assets', fromlist=['installed_fetchers']).installed_fetchers()]})",
            )
        archive_path = (
            target_dir / Path(spec.source).name
            if spec.extract == "none"
            else target_dir.parent / f"{spec.id}.archive"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        fetcher.fetch(spec, archive_path)
        if spec.extract == "none":
            return archive_path
        return extract_asset(spec, archive_path, target_dir)

    async def _evict_if_needed(self, *, incoming_bytes: int) -> None:
        """LRU sweep until eviction frees enough room for ``incoming_bytes``.

        Mirrors :class:`ImageCacheManager._evict_if_needed`; uses the
        in-memory size_bytes sum instead of polling free disk because
        asset extracted dirs are non-trivial to size on demand.
        """
        async with self._lock:
            current = sum(r.size_bytes for r in self._records.values())
            if current + incoming_bytes <= self._cfg.evict_target_bytes:
                return
            for record in self._cold_lru_order():
                if current + incoming_bytes <= self._cfg.evict_target_bytes:
                    break
                LOGGER.info(
                    "asset_cache: evicting cold asset %s (size=%dB)",
                    record.id, record.size_bytes,
                )
                evict_asset(record)
                current -= record.size_bytes
                self._records.pop(record.id, None)

    def _cold_lru_order(self) -> list[AssetRecord]:
        candidates = [
            r for r in self._records.values()
            if self._in_use.get(r.id, 0) == 0 and r.id not in self._pins
        ]

        def _key(r: AssetRecord) -> float:
            ts = self._last_used.get(r.id)
            return ts if ts is not None else 0.0

        return sorted(candidates, key=_key)

    # ── Reporting ──────────────────────────────────────────────────────────

    def report(self) -> NodeAssetReport:
        now = time.monotonic()
        rows = [
            AssetReportEntry(
                id=r.id,
                tier=self.tier(r.id, now=now),
                size_bytes=r.size_bytes,
                in_use_count=self._in_use.get(r.id, 0),
                last_used_at=self._last_used.get(r.id),
                pinned=r.id in self._pins,
                path=r.path,
            )
            for r in self._records.values()
        ]
        return NodeAssetReport(assets=rows, pinned=tuple(sorted(self._pins)))

    # ── Internal helpers ───────────────────────────────────────────────────

    def _fetch_lock_for(self, asset_id: str) -> asyncio.Lock:
        if asset_id not in self._fetch_locks:
            self._fetch_locks[asset_id] = asyncio.Lock()
        return self._fetch_locks[asset_id]


__all__ = [
    "AssetCacheConfig",
    "AssetCacheManager",
    "AssetReportEntry",
    "NodeAssetReport",
]
