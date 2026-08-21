"""Pattern B — assets block (spec 06 §"Pattern B" + spec 15 asset cache).

Templates like OSWorld need a multi-GB qcow2 disk image fetched from
Hugging Face. Hard-coding the download into the template's adapter
puts cluster-wide caching outside the platform's view; spec 06's
``assets:`` block makes the asset a first-class platform concern with
the same priority tiers + LRU + warmup machinery that images get.

This module ships:

- :class:`AssetSpec` — manifest serialization (sha256, size, extract,
  mode, source URL).
- :class:`AssetFetcher` Protocol — pluggable per-scheme downloaders
  (``https://``, ``s3://``, ``gs://``, ``hf://``).
- :class:`HttpAssetFetcher` — phase-0 default, supports resume.
- :class:`AssetRecord` — what the cache manager tracks (mirrors
  :class:`xrlenv.backends.base.ImageRecord`).

Phase-1 extensions deferred: stargz / overlaybd lazy mounts, cluster
mirror integration, signed publisher list.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict

LOGGER = logging.getLogger(__name__)


AssetMode = Literal["shared-readonly", "per-sandbox", "bake-into-image"]
AssetExtract = Literal["none", "zip", "tar", "tar.gz"]


class AssetSpec(BaseModel):
    """One ``assets:`` block entry on a template manifest (spec 06).

    All fields are required at register time except ``options`` so the
    operator can't accidentally pull a multi-GB blob without a
    checksum (the integrity floor for spec 19's supply-chain rule).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    """Stable identifier — used as the cache key + the mount-time
    placeholder template engines resolve via ``{asset:<id>}``."""
    source: str
    """Full URI: ``https://...``, ``s3://...``, ``gs://...``,
    ``hf://...``. The scheme picks the :class:`AssetFetcher`."""
    sha256: str
    """Required; the cache refuses to mount an asset whose digest
    diverges from the manifest's value (spec 19)."""
    size_bytes: int
    """Approximate; used for cache budgeting + the operator's
    ``xrlenv assets`` view. Doesn't have to match exactly — ``+/-10%``
    is fine."""
    extract: AssetExtract = "none"
    """Post-download extraction. ``none`` keeps the file as-is;
    ``zip`` / ``tar`` / ``tar.gz`` extract into ``extract_to``."""
    extract_to: str | None = None
    """Filesystem path the extracted artifact lands at. Required when
    ``extract != "none"``; for ``extract=none`` the asset lives at
    ``<extract_to>/<basename>``. Defaults to
    ``/var/cache/xrlenv/assets/<id>/`` when omitted."""
    mode: AssetMode = "shared-readonly"
    """How the runtime presents the asset to sandboxes (spec 06
    §"Asset modes")."""


class AssetRecord(BaseModel):
    """Cache-manager-side snapshot of one asset on disk."""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path
    size_bytes: int
    sha256: str | None = None
    last_used_at: float | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Fetcher Protocol + HTTP impl
# ──────────────────────────────────────────────────────────────────────────────


class AssetFetchError(RuntimeError):
    """Raised by an :class:`AssetFetcher` on download / checksum failure.

    Carries a stable ``reason`` string so the cache manager can log a
    structured ``asset_fetch_failed:<reason>`` audit row instead of a
    generic exception trace.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class AssetFetcher(Protocol):
    """One implementation per URI scheme.

    The cache manager picks the fetcher by ``urlparse(source).scheme``;
    plug-ins (``s3://``, ``gs://``, ``hf://``) register via
    :func:`register_fetcher`.
    """

    def supports(self, source: str) -> bool: ...

    def fetch(self, spec: AssetSpec, dst: Path) -> Path:
        """Download ``spec`` to ``dst`` (file path) atomically. Returns
        the final on-disk path. Must verify the sha256 before returning;
        partial / corrupted downloads raise :class:`AssetFetchError`."""
        ...


class HttpAssetFetcher:
    """Phase-0 ``https://`` (and ``http://`` for tests) fetcher.

    Spec 19 §"Image and asset supply chain": *HTTP fetchers refuse
    plaintext.* Phase-0 enforces this by default; tests pass
    ``allow_insecure=True`` to opt into ``http://`` for fixtures.
    """

    def __init__(self, *, allow_insecure: bool = False, chunk_bytes: int = 1 << 20) -> None:
        self._allow_insecure = allow_insecure
        self._chunk_bytes = chunk_bytes

    def supports(self, source: str) -> bool:
        scheme = urlparse(source).scheme
        if scheme == "https":
            return True
        if scheme == "http":
            return self._allow_insecure
        return False

    def fetch(self, spec: AssetSpec, dst: Path) -> Path:
        scheme = urlparse(spec.source).scheme
        if scheme == "http" and not self._allow_insecure:
            raise AssetFetchError(
                "plaintext_refused",
                f"refusing to fetch http:// asset {spec.id!r} — spec 19 "
                "requires https:// for asset supply chain integrity",
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        sha = hashlib.sha256()
        try:
            req = Request(spec.source, headers={"User-Agent": "xrlenv/0"})
            with urlopen(req) as response, tmp.open("wb") as out:
                while True:
                    chunk = response.read(self._chunk_bytes)
                    if not chunk:
                        break
                    sha.update(chunk)
                    out.write(chunk)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise AssetFetchError(
                "download_failed",
                f"asset {spec.id!r}: download from {spec.source!r} failed: {exc}",
            ) from exc

        digest = sha.hexdigest()
        if digest.lower() != spec.sha256.lower().removeprefix("sha256:"):
            # Quarantine the bad file alongside the destination so the
            # operator can inspect it (spec 19 §"Asset integrity").
            quarantine = dst.with_suffix(dst.suffix + f".bad-{digest[:8]}")
            tmp.rename(quarantine)
            raise AssetFetchError(
                "checksum_mismatch",
                f"asset {spec.id!r}: sha256 mismatch (expected "
                f"{spec.sha256!r}, got {digest!r}); quarantined at "
                f"{quarantine}",
            )
        tmp.rename(dst)
        return dst


# ──────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ──────────────────────────────────────────────────────────────────────────────


def extract_asset(spec: AssetSpec, archive_path: Path, extract_to: Path) -> Path:
    """Apply ``spec.extract`` to ``archive_path``. Returns the
    extracted directory (or the original file when ``extract=none``).
    """
    if spec.extract == "none":
        return archive_path
    extract_to.mkdir(parents=True, exist_ok=True)
    if spec.extract == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)
    elif spec.extract in ("tar", "tar.gz"):
        mode: Literal["r:gz", "r"] = "r:gz" if spec.extract == "tar.gz" else "r"
        with tarfile.open(archive_path, mode=mode) as tf:
            tf.extractall(extract_to, filter="data")
    else:
        raise AssetFetchError(
            "unsupported_extract",
            f"asset {spec.id!r}: extract={spec.extract!r} not supported",
        )
    # Drop the archive once we have the extracted form on disk.
    archive_path.unlink(missing_ok=True)
    return extract_to


def asset_default_root(asset_id: str) -> Path:
    """Default ``extract_to`` per spec 06 (``/var/cache/xrlenv/assets/<id>/``)."""
    return Path("/var/cache/xrlenv/assets") / asset_id


# ──────────────────────────────────────────────────────────────────────────────
# Fetcher registry
# ──────────────────────────────────────────────────────────────────────────────


_fetcher_registry: list[AssetFetcher] = []


def register_fetcher(fetcher: AssetFetcher) -> None:
    """Append ``fetcher`` to the global registry. The cache manager
    dispatches in registration order — first :py:meth:`supports` hit
    wins. Phase-1 plug-ins (``s3``, ``gs``, ``hf``) register on import.
    """
    _fetcher_registry.append(fetcher)


def fetcher_for(source: str) -> AssetFetcher | None:
    """Pick the first registered fetcher claiming ``source``. Returns
    ``None`` when no fetcher matches; the cache manager turns that into
    a clean ``asset_fetch_failed:no_fetcher`` audit row.
    """
    for fetcher in _fetcher_registry:
        if fetcher.supports(source):
            return fetcher
    return None


def reset_fetchers() -> None:
    """Test-only escape hatch: drop everything from the registry."""
    _fetcher_registry.clear()


def installed_fetchers() -> Iterable[AssetFetcher]:
    return tuple(_fetcher_registry)


# Default registration: HTTPS fetcher, secure-only by default.
register_fetcher(HttpAssetFetcher())


def _cleanup_quarantine(path: Path) -> None:
    """Test helper: drop a quarantined file silently."""
    from contextlib import suppress

    with suppress(FileNotFoundError):
        path.unlink()


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_existing(spec: AssetSpec, path: Path) -> bool:
    """Return True if ``path`` exists and matches ``spec.sha256``."""
    if not path.exists():
        return False
    expected = spec.sha256.lower().removeprefix("sha256:")
    return _file_sha256(path) == expected


def evict_asset(record: AssetRecord) -> None:
    """Remove an asset from disk (file or directory)."""
    if record.path.is_dir():
        shutil.rmtree(record.path, ignore_errors=True)
    elif record.path.exists():
        record.path.unlink(missing_ok=True)


__all__ = [
    "AssetExtract",
    "AssetFetchError",
    "AssetFetcher",
    "AssetMode",
    "AssetRecord",
    "AssetSpec",
    "HttpAssetFetcher",
    "asset_default_root",
    "evict_asset",
    "extract_asset",
    "fetcher_for",
    "installed_fetchers",
    "register_fetcher",
    "reset_fetchers",
    "verify_existing",
]
