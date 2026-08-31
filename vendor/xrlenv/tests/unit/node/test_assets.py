"""Tests for the Slice 9 Pattern B asset block + cache (spec 06 + spec 15)."""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import socket
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from xrlenv.control.assets import (
    AssetFetchError,
    AssetSpec,
    HttpAssetFetcher,
    asset_default_root,
    extract_asset,
    fetcher_for,
    register_fetcher,
    reset_fetchers,
    verify_existing,
)
from xrlenv.node.asset_cache import (
    AssetCacheConfig,
    AssetCacheManager,
)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def http_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Spin a tiny http.server in a daemon thread serving ``tmp_path``."""
    docroot = tmp_path / "www"
    docroot.mkdir()

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(docroot), **kwargs)

        def log_message(self, *_args: Any) -> None:
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    httpd = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", docroot
    finally:
        httpd.shutdown()
        thread.join(timeout=5.0)


# ──────────────────────────────────────────────────────────────────────────────
# AssetSpec model
# ──────────────────────────────────────────────────────────────────────────────


def test_asset_spec_requires_sha256_size() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AssetSpec.model_validate({"id": "x", "source": "https://x", "extract": "none"})


def test_asset_spec_extract_defaults_to_none() -> None:
    spec = AssetSpec(
        id="x", source="https://x/y.bin",
        sha256="00" * 32, size_bytes=10,
    )
    assert spec.extract == "none"
    assert spec.mode == "shared-readonly"


# ──────────────────────────────────────────────────────────────────────────────
# HttpAssetFetcher
# ──────────────────────────────────────────────────────────────────────────────


def test_http_fetcher_refuses_plaintext_by_default(tmp_path: Path) -> None:
    f = HttpAssetFetcher()
    spec = AssetSpec(
        id="x", source="http://x/y.bin",
        sha256="00" * 32, size_bytes=10,
    )
    with pytest.raises(AssetFetchError) as exc_info:
        f.fetch(spec, tmp_path / "nope")
    assert exc_info.value.reason == "plaintext_refused"


def test_http_fetcher_downloads_and_verifies_sha256(
    tmp_path: Path, http_server: tuple[str, Path],
) -> None:
    base_url, docroot = http_server
    payload = b"abcdefghij" * 100
    (docroot / "blob.bin").write_bytes(payload)
    spec = AssetSpec(
        id="blob",
        source=f"{base_url}/blob.bin",
        sha256=_sha(payload),
        size_bytes=len(payload),
    )
    dst = tmp_path / "blob.bin"
    f = HttpAssetFetcher(allow_insecure=True)
    out = f.fetch(spec, dst)
    assert out == dst
    assert dst.read_bytes() == payload


def test_http_fetcher_quarantines_on_checksum_mismatch(
    tmp_path: Path, http_server: tuple[str, Path],
) -> None:
    base_url, docroot = http_server
    (docroot / "wrong.bin").write_bytes(b"actual content")
    spec = AssetSpec(
        id="wrong", source=f"{base_url}/wrong.bin",
        sha256="ff" * 32, size_bytes=14,
    )
    dst = tmp_path / "wrong.bin"
    f = HttpAssetFetcher(allow_insecure=True)
    with pytest.raises(AssetFetchError) as exc_info:
        f.fetch(spec, dst)
    assert exc_info.value.reason == "checksum_mismatch"
    # Quarantined file present, original destination absent.
    assert not dst.exists()
    bads = list(tmp_path.glob("wrong.bin.bad-*"))
    assert len(bads) == 1
    assert bads[0].read_bytes() == b"actual content"


# ──────────────────────────────────────────────────────────────────────────────
# extract_asset
# ──────────────────────────────────────────────────────────────────────────────


def test_extract_asset_zip(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner.txt", "hello")
    spec = AssetSpec(
        id="z", source="https://x/y.zip",
        sha256="00" * 32, size_bytes=1,
        extract="zip", extract_to=str(tmp_path / "extracted"),
    )
    out = extract_asset(spec, archive, tmp_path / "extracted")
    assert out == tmp_path / "extracted"
    assert (out / "inner.txt").read_text() == "hello"
    # Archive is gone.
    assert not archive.exists()


def test_extract_asset_none_returns_archive_path(tmp_path: Path) -> None:
    archive = tmp_path / "blob.bin"
    archive.write_bytes(b"xyz")
    spec = AssetSpec(
        id="b", source="https://x/y.bin",
        sha256="00" * 32, size_bytes=3, extract="none",
    )
    out = extract_asset(spec, archive, tmp_path / "ignored")
    assert out == archive
    assert archive.exists()


# ──────────────────────────────────────────────────────────────────────────────
# fetcher_for / registry
# ──────────────────────────────────────────────────────────────────────────────


def test_fetcher_for_picks_https_in_default_registry() -> None:
    f = fetcher_for("https://example.com/x.bin")
    assert isinstance(f, HttpAssetFetcher)


def test_fetcher_for_returns_none_for_unknown_scheme() -> None:
    f = fetcher_for("s3://bucket/key")
    # No s3 fetcher registered in the default set.
    assert f is None


def test_register_fetcher_appends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify newly-registered fetchers see traffic without disrupting
    the default HTTP fetcher."""

    class _Custom:
        def supports(self, source: str) -> bool:
            return source.startswith("custom://")

        def fetch(self, spec: AssetSpec, dst: Path) -> Path:
            return dst

    register_fetcher(_Custom())
    try:
        assert fetcher_for("custom://anything").__class__.__name__ == "_Custom"
        # HTTPS still works.
        assert isinstance(fetcher_for("https://x"), HttpAssetFetcher)
    finally:
        # Restore default state for the rest of the suite.
        reset_fetchers()
        register_fetcher(HttpAssetFetcher())


# ──────────────────────────────────────────────────────────────────────────────
# verify_existing
# ──────────────────────────────────────────────────────────────────────────────


def test_verify_existing_matches_and_misses(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    good = AssetSpec(
        id="x", source="https://x", size_bytes=3, sha256=_sha(b"abc"),
    )
    bad = AssetSpec(
        id="x", source="https://x", size_bytes=3, sha256=_sha(b"WRONG"),
    )
    missing = AssetSpec(
        id="x", source="https://x", size_bytes=3, sha256=_sha(b"abc"),
    )
    assert verify_existing(good, p)
    assert not verify_existing(bad, p)
    assert not verify_existing(missing, tmp_path / "absent.bin")


def test_asset_default_root_uses_id() -> None:
    p = asset_default_root("osworld-ubuntu-qcow2")
    assert p == Path("/var/cache/xrlenv/assets/osworld-ubuntu-qcow2")


# ──────────────────────────────────────────────────────────────────────────────
# AssetCacheManager
# ──────────────────────────────────────────────────────────────────────────────


def _spec(
    *, id: str, payload: bytes, source: str,
    extract: str = "none", extract_to: Path,
) -> AssetSpec:
    return AssetSpec(
        id=id, source=source, sha256=_sha(payload),
        size_bytes=len(payload), extract=extract,  # type: ignore[arg-type]
        extract_to=str(extract_to),
    )


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def cache(cache_root: Path) -> AssetCacheManager:
    return AssetCacheManager(
        config=AssetCacheConfig(
            cache_root=cache_root,
            evict_threshold_bytes=2_000_000,
            evict_target_bytes=4_000_000,
        ),
    )


async def test_ensure_present_downloads_then_caches(
    cache: AssetCacheManager,
    cache_root: Path,
    http_server: tuple[str, Path],
) -> None:
    """First ensure_present triggers a fetch; second is served from disk."""
    base_url, docroot = http_server
    payload = b"asset-bytes" * 10
    (docroot / "blob.bin").write_bytes(payload)

    # Use the http-allowing fetcher for the duration of this test.
    reset_fetchers()
    register_fetcher(HttpAssetFetcher(allow_insecure=True))
    spec = _spec(
        id="blob", payload=payload, source=f"{base_url}/blob.bin",
        extract_to=cache_root / "blob",
    )
    try:
        path = await cache.ensure_present(spec)
        assert path.exists()
        # Cached: subsequent call is a no-op (no second fetcher call).
        path2 = await cache.ensure_present(spec)
        assert path2 == path
    finally:
        reset_fetchers()
        register_fetcher(HttpAssetFetcher())


async def test_ensure_present_concurrent_coalesces(
    cache: AssetCacheManager,
    cache_root: Path,
    http_server: tuple[str, Path],
) -> None:
    base_url, docroot = http_server
    payload = b"x" * 4096
    (docroot / "concur.bin").write_bytes(payload)

    reset_fetchers()
    register_fetcher(HttpAssetFetcher(allow_insecure=True))
    spec = _spec(
        id="concur", payload=payload, source=f"{base_url}/concur.bin",
        extract_to=cache_root / "concur",
    )
    try:
        results = await asyncio.gather(*(cache.ensure_present(spec) for _ in range(5)))
        assert len(set(results)) == 1
    finally:
        reset_fetchers()
        register_fetcher(HttpAssetFetcher())


def test_acquire_release_refcount_clamps(cache: AssetCacheManager) -> None:
    cache.acquire("a")
    cache.acquire("a")
    assert cache.in_use_count("a") == 2
    cache.release("a")
    cache.release("a")
    cache.release("a")  # extra is no-op
    assert cache.in_use_count("a") == 0


def test_pin_unpin(cache: AssetCacheManager) -> None:
    cache.pin("p")
    assert "p" in cache.pins
    assert cache.is_pinned("p")
    cache.unpin("p")
    assert not cache.is_pinned("p")


def test_tier_classification(cache: AssetCacheManager) -> None:
    assert cache.tier("nope") == "cold"
    cache.pin("p")
    assert cache.tier("p") == "pinned"
    cache.acquire("u")
    assert cache.tier("u") == "in_use"


def test_ensure_present_without_fetcher_raises_clean_error(
    tmp_path: Path,
) -> None:
    cache = AssetCacheManager(
        config=AssetCacheConfig(cache_root=tmp_path / "c"),
    )
    reset_fetchers()  # no fetchers registered
    spec = AssetSpec(
        id="orphan", source="weird-scheme://x",
        sha256="00" * 32, size_bytes=1,
        extract_to=str(tmp_path / "c" / "orphan"),
    )
    try:
        with pytest.raises(AssetFetchError) as exc_info:
            asyncio.run(cache.ensure_present(spec))
        assert exc_info.value.reason == "no_fetcher"
    finally:
        register_fetcher(HttpAssetFetcher())


def test_report_includes_pin_and_inuse_state(
    cache: AssetCacheManager,
    cache_root: Path,
    http_server: tuple[str, Path],
) -> None:
    base_url, docroot = http_server
    (docroot / "rep.bin").write_bytes(b"abc")
    reset_fetchers()
    register_fetcher(HttpAssetFetcher(allow_insecure=True))
    spec = _spec(
        id="rep", payload=b"abc", source=f"{base_url}/rep.bin",
        extract_to=cache_root / "rep",
    )
    try:
        asyncio.run(cache.ensure_present(spec))
        cache.pin("rep")
        cache.acquire("rep")
        report = cache.report()
        by_id = {a.id: a for a in report.assets}
        assert by_id["rep"].pinned is True
        assert by_id["rep"].in_use_count == 1
        # Pinned wins over in_use? No — spec-15 says in_use wins.
        assert by_id["rep"].tier == "in_use"
        assert "rep" in report.pinned
    finally:
        reset_fetchers()
        register_fetcher(HttpAssetFetcher())


# ──────────────────────────────────────────────────────────────────────────────
# Manifest assets block (loader integration)
# ──────────────────────────────────────────────────────────────────────────────


def test_manifest_loader_parses_assets_block(tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "manifest.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "image": "im/t:1",
                "env_adapter": {"module": "m", "class": "C"},
                "reward": {"mode": "env_step"},
                "assets": [
                    {
                        "id": "qcow",
                        "source": "https://example.com/x.qcow2",
                        "sha256": "00" * 32,
                        "size_bytes": 1024,
                        "extract": "none",
                        "mode": "shared-readonly",
                    },
                ],
            }
        )
    )
    from xrlenv.control.template_catalog import load_manifest

    manifest = load_manifest(p)
    assert len(manifest.assets) == 1
    assert manifest.assets[0].id == "qcow"
    assert manifest.assets[0].mode == "shared-readonly"


def test_manifest_loader_rejects_non_list_assets(tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "manifest.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "image": "im/t:1",
                "env_adapter": {"module": "m", "class": "C"},
                "reward": {"mode": "env_step"},
                "assets": {"not": "a-list"},
            }
        )
    )
    from xrlenv.control.template_catalog import load_manifest

    with pytest.raises(Exception, match="assets"):
        load_manifest(p)
