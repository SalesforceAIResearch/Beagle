"""Unit tests for :mod:`xrlenv.control.scratch_gc` — the scratch-registry GC
decision core (TTL + quota, active-run exemption). Slice 3.

Also covers CLI helpers from ``deploy/registry/scratch_registry_gc.py`` via importlib."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from xrlenv.control.scratch_gc import ScratchImage, select_reclaim_targets


def _load_gc_script() -> types.ModuleType:
    """Load deploy/registry/scratch_registry_gc.py via importlib (non-package script)."""
    script = Path(__file__).resolve().parent.parent.parent.parent / "deploy" / "registry" / "scratch_registry_gc.py"
    spec = importlib.util.spec_from_file_location("scratch_registry_gc", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Prevent sys.path mutation side-effect from polluting other tests
    original_path = sys.path[:]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    sys.path[:] = original_path
    return mod

NOW = 1_000_000.0
GB = 1024**3


def _img(name: str, *, age_s: float, size_gb: float, digest: str | None = None) -> ScratchImage:
    return ScratchImage(
        repo=f"scratch/{name}",
        digest=digest or f"sha256:{name}",
        size_bytes=int(size_gb * GB),
        last_used_at=NOW - age_s,
    )


def _digests(imgs: list[ScratchImage]) -> set[str]:
    return {i.digest for i in imgs}


def test_ttl_reclaims_only_old_images() -> None:
    imgs = [
        _img("old", age_s=100_000, size_gb=1),
        _img("fresh", age_s=10, size_gb=1),
    ]
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=3600, quota_bytes=None, exempt_digests=frozenset(),
    )
    assert _digests(out) == {"sha256:old"}


def test_ttl_none_disables_ttl_pass() -> None:
    imgs = [_img("old", age_s=100_000, size_gb=1)]
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=None, quota_bytes=None, exempt_digests=frozenset(),
    )
    assert out == []


def test_exempt_digest_never_reclaimed_by_ttl() -> None:
    imgs = [_img("old", age_s=100_000, size_gb=1)]
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=3600, quota_bytes=None,
        exempt_digests=frozenset({"sha256:old"}),
    )
    assert out == []


def test_quota_reclaims_oldest_first_until_under() -> None:
    imgs = [
        _img("a", age_s=300, size_gb=4),   # oldest
        _img("b", age_s=200, size_gb=4),
        _img("c", age_s=100, size_gb=4),   # newest
    ]
    # total 12 GB, quota 5 GB → must evict a then b (oldest first) → 4 GB left.
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=None, quota_bytes=5 * GB, exempt_digests=frozenset(),
    )
    assert _digests(out) == {"sha256:a", "sha256:b"}


def test_quota_never_evicts_exempt_even_if_over() -> None:
    imgs = [
        _img("active", age_s=500, size_gb=8, digest="sha256:active"),  # exempt, oldest
        _img("b", age_s=100, size_gb=4),
    ]
    # total 12 GB, quota 5 GB. 'active' is exempt (oldest) → only 'b' evictable.
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=None, quota_bytes=5 * GB,
        exempt_digests=frozenset({"sha256:active"}),
    )
    assert _digests(out) == {"sha256:b"}


def test_quota_shortfall_when_exempt_exceeds_quota() -> None:
    # Exempt images alone (10 GB) exceed the 5 GB quota — can't get under;
    # only the non-exempt image is evicted, quota remains breached (correct).
    imgs = [
        _img("active", age_s=500, size_gb=10, digest="sha256:active"),
        _img("b", age_s=100, size_gb=2),
    ]
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=None, quota_bytes=5 * GB,
        exempt_digests=frozenset({"sha256:active"}),
    )
    assert _digests(out) == {"sha256:b"}


def test_ttl_and_quota_compose_without_double_counting() -> None:
    imgs = [
        _img("veryold", age_s=100_000, size_gb=4),  # TTL-reclaimed
        _img("b", age_s=300, size_gb=4),
        _img("c", age_s=100, size_gb=4),
    ]
    # TTL removes 'veryold' (→ resident b,c = 8 GB). Quota 5 GB → evict oldest
    # remaining 'b'. Result: veryold + b, no duplicates.
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=3600, quota_bytes=5 * GB, exempt_digests=frozenset(),
    )
    assert _digests(out) == {"sha256:veryold", "sha256:b"}
    assert len(out) == len(_digests(out))  # no duplicate entries


def test_under_quota_and_within_ttl_reclaims_nothing() -> None:
    imgs = [_img("a", age_s=10, size_gb=1), _img("b", age_s=20, size_gb=1)]
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=3600, quota_bytes=100 * GB, exempt_digests=frozenset(),
    )
    assert out == []


def test_result_is_oldest_first() -> None:
    imgs = [
        _img("new", age_s=100_000, size_gb=1),
        _img("old", age_s=200_000, size_gb=1),
    ]
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=3600, quota_bytes=None, exempt_digests=frozenset(),
    )
    assert [i.digest for i in out] == ["sha256:old", "sha256:new"]


# ── additional edge-case tests ────────────────────────────────────────────────


def test_empty_images_returns_empty() -> None:
    """No images → no reclaim targets regardless of TTL/quota."""
    out = select_reclaim_targets(
        [], now=NOW, ttl_seconds=3600, quota_bytes=5 * GB, exempt_digests=frozenset(),
    )
    assert out == []


def test_ttl_exact_boundary_not_reclaimed() -> None:
    """An image exactly at the TTL boundary (age == ttl_seconds) is kept.
    The condition is strict ``>``; equal means 'just expired' is not reclaimed."""
    img = _img("edge", age_s=3600, size_gb=1)  # age == ttl_seconds exactly
    out = select_reclaim_targets(
        [img], now=NOW, ttl_seconds=3600, quota_bytes=None, exempt_digests=frozenset(),
    )
    assert out == [], (
        "image at exactly ttl_seconds age must NOT be reclaimed (strict > boundary)"
    )


def test_ttl_one_second_over_boundary_is_reclaimed() -> None:
    """An image one second past the TTL boundary IS reclaimed."""
    img = _img("over", age_s=3601, size_gb=1)
    out = select_reclaim_targets(
        [img], now=NOW, ttl_seconds=3600, quota_bytes=None, exempt_digests=frozenset(),
    )
    assert _digests(out) == {"sha256:over"}


def test_quota_exactly_at_limit_no_eviction() -> None:
    """Total footprint == quota_bytes → no eviction (condition is ``>`` not ``>=``)."""
    imgs = [_img("a", age_s=100, size_gb=2), _img("b", age_s=200, size_gb=3)]
    total = 5 * GB  # exactly matches sum
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=None, quota_bytes=total, exempt_digests=frozenset(),
    )
    assert out == [], "total == quota must NOT trigger eviction"


def test_exempt_digest_never_reclaimed_by_either_pass() -> None:
    """An exempt image old enough for TTL AND present when quota is breached
    must NEVER appear in the reclaim output from either pass."""
    exempt_img = _img("active", age_s=500_000, size_gb=8, digest="sha256:active")
    non_exempt = _img("old", age_s=400_000, size_gb=1)
    exempt_set = frozenset({"sha256:active"})
    out = select_reclaim_targets(
        [exempt_img, non_exempt],
        now=NOW, ttl_seconds=3600, quota_bytes=1 * GB,
        exempt_digests=exempt_set,
    )
    result_digests = _digests(out)
    assert "sha256:active" not in result_digests, (
        "exempt digest must never appear in reclaim output, even when old and quota is over"
    )
    # The non-exempt old image should be reclaimed by TTL
    assert "sha256:old" in result_digests


def test_both_passes_none_reclaims_nothing() -> None:
    """Both ttl_seconds=None and quota_bytes=None → empty output."""
    imgs = [_img("a", age_s=999_999, size_gb=100)]
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=None, quota_bytes=None, exempt_digests=frozenset(),
    )
    assert out == []


def test_quota_only_no_ttl_pass() -> None:
    """With ttl_seconds=None, only quota pass runs (no TTL evictions)."""
    imgs = [
        _img("a", age_s=5000, size_gb=3),  # old but within TTL range (not that TTL runs)
        _img("b", age_s=1000, size_gb=3),
    ]
    # total 6 GB > quota 4 GB → evict oldest 'a'
    out = select_reclaim_targets(
        imgs, now=NOW, ttl_seconds=None, quota_bytes=4 * GB, exempt_digests=frozenset(),
    )
    assert _digests(out) == {"sha256:a"}


# ── CLI helper tests (importlib-loaded, _http monkeypatched) ──────────────────


@pytest.fixture(scope="module")
def gc() -> types.ModuleType:
    return _load_gc_script()


@pytest.mark.parametrize("s,expected", [
    ("3600", 3600.0),
    ("72h", 72 * 3600.0),
    ("30m", 30 * 60.0),
    ("90s", 90.0),
    ("2d", 2 * 86400.0),
    ("1.5h", 1.5 * 3600.0),
    ("0s", 0.0),
    ("72H", 72 * 3600.0),   # case-insensitive
])
def test_parse_duration(gc: types.ModuleType, s: str, expected: float) -> None:
    assert gc._parse_duration(s) == pytest.approx(expected)


def test_manifest_size_leaf_manifest(gc: types.ModuleType) -> None:
    """A leaf manifest (no 'manifests' key) sums config + layer sizes."""
    manifest: dict[str, Any] = {
        "config": {"size": 1000},
        "layers": [{"size": 2000}, {"size": 3000}],
    }
    assert gc._manifest_size("http://r", "scratch/x", manifest) == 6000


def test_manifest_size_oci_index_recurses(gc: types.ModuleType) -> None:
    """OCI index recurses into child manifests, accumulating sizes."""
    child_manifest = {"config": {"size": 100}, "layers": [{"size": 900}]}
    index = {"manifests": [{"digest": "sha256:child"}]}
    responses: dict[str, tuple[int, dict[str, str], bytes]] = {
        "http://r/v2/scratch/x/manifests/sha256:child": (
            200, {}, __import__("json").dumps(child_manifest).encode(),
        ),
    }

    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return responses.get(url, (404, {}, b""))

    with patch.object(gc, "_http", fake_http):
        assert gc._manifest_size("http://r", "scratch/x", index) == 1000


def test_manifest_size_oci_index_child_404_falls_back(gc: types.ModuleType) -> None:
    """When a child manifest 404s, _manifest_size falls back to child['size']."""
    index = {"manifests": [{"digest": "sha256:gone", "size": 500}]}

    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 404, {}, b""

    with patch.object(gc, "_http", fake_http):
        assert gc._manifest_size("http://r", "scratch/x", index) == 500


def test_delete_manifest_404_is_success(gc: types.ModuleType) -> None:
    """_delete_manifest treats 404 as success (idempotent delete)."""
    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 404, {}, b""

    with patch.object(gc, "_http", fake_http):
        assert gc._delete_manifest("http://r", "scratch/x", "sha256:gone") is True


def test_delete_manifest_202_is_success(gc: types.ModuleType) -> None:
    """_delete_manifest treats 202 as success (accepted)."""
    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 202, {}, b""

    with patch.object(gc, "_http", fake_http):
        assert gc._delete_manifest("http://r", "scratch/x", "sha256:digest") is True


def test_delete_manifest_500_is_failure(gc: types.ModuleType) -> None:
    """_delete_manifest returns False on a server error."""
    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 500, {}, b"internal error"

    with patch.object(gc, "_http", fake_http):
        assert gc._delete_manifest("http://r", "scratch/x", "sha256:digest") is False


def test_load_exempt_url_list(gc: types.ModuleType, tmp_path: Path) -> None:
    """_load_exempt with --exempt-url returning a JSON list of digests."""
    import argparse

    body = __import__("json").dumps(["sha256:aaa", "sha256:bbb"]).encode()

    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 200, {}, body

    args = argparse.Namespace(exempt_file=None, exempt_url="http://cp/active")
    with patch.object(gc, "_http", fake_http):
        result = gc._load_exempt(args)
    assert result == frozenset({"sha256:aaa", "sha256:bbb"})


def test_load_exempt_url_dict(gc: types.ModuleType, tmp_path: Path) -> None:
    """_load_exempt with --exempt-url returning {\"digests\": [...]}."""
    import argparse

    body = __import__("json").dumps({"digests": ["sha256:ccc"]}).encode()

    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 200, {}, body

    args = argparse.Namespace(exempt_file=None, exempt_url="http://cp/active")
    with patch.object(gc, "_http", fake_http):
        result = gc._load_exempt(args)
    assert result == frozenset({"sha256:ccc"})


def test_load_exempt_file(gc: types.ModuleType, tmp_path: Path) -> None:
    """_load_exempt reads digests from a file, skipping blanks and comments."""
    import argparse

    f = tmp_path / "active.txt"
    f.write_text("sha256:aaa\n# comment\n\nsha256:bbb\n")
    args = argparse.Namespace(exempt_file=str(f), exempt_url=None)
    result = gc._load_exempt(args)
    assert result == frozenset({"sha256:aaa", "sha256:bbb"})


# ── repo-based exemption + active_scratch_repos (GC endpoint) ──────────────────


def test_exempt_by_repo_not_just_digest() -> None:
    """An image whose REPO is in the exempt set is spared even when its digest
    is not — the control plane exempts by the content-addressed repo it knows."""
    from xrlenv.control.scratch_gc import select_reclaim_targets
    img = _img("old", age_s=100_000, size_gb=8)  # repo=scratch/old, digest=sha256:old
    # exempt by repo, not digest:
    out = select_reclaim_targets(
        [img], now=NOW, ttl_seconds=3600, quota_bytes=1 * GB,
        exempt_digests=frozenset({"scratch/old"}),
    )
    assert out == []


def test_active_scratch_repos_extracts_scratch_repos() -> None:
    from xrlenv.control.scratch_gc import active_scratch_repos
    pairs = [
        ("cp:5012/scratch/aaa:latest", "running"),
        ("cp:5012/scratch/bbb@sha256:xyz", "destroying"),
        ("docker.io/library/busybox:latest", "running"),   # not scratch
        ("cp:5012/scratch/ccc", "destroyed"),               # terminal -> skip
        (None, "running"),                                   # no image
    ]
    assert active_scratch_repos(pairs) == {"scratch/aaa", "scratch/bbb"}


def test_active_scratch_repos_empty() -> None:
    from xrlenv.control.scratch_gc import active_scratch_repos
    assert active_scratch_repos([]) == set()


def test_active_scratch_repos_hyphenated_prefix_excluded() -> None:
    """A repo named 'scratch-foo/abc' (hyphen, not slash) does NOT start with
    'scratch/' and must not be treated as a scratch repo."""
    from xrlenv.control.scratch_gc import active_scratch_repos
    pairs = [
        ("cp:5012/scratch-foo/abc:latest", "running"),   # hyphen — NOT scratch
        ("cp:5012/scratch/real:latest", "running"),       # real scratch
    ]
    result = active_scratch_repos(pairs)
    assert "scratch/real" in result
    assert not any("scratch-foo" in r for r in result), (
        "scratch-foo should not be treated as a scratch namespace"
    )


def test_load_exempt_url_repos_dict(gc: types.ModuleType, tmp_path: Path) -> None:
    """_load_exempt with --exempt-url returning the CP endpoint's {'repos':[...]}."""
    import argparse

    body = __import__("json").dumps({"repos": ["scratch/aaa", "scratch/bbb"]}).encode()

    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 200, {}, body

    args = argparse.Namespace(exempt_file=None, exempt_url="http://cp/api/scratch/active-digests")
    with patch.object(gc, "_http", fake_http):
        result = gc._load_exempt(args)
    assert result == frozenset({"scratch/aaa", "scratch/bbb"})


def test_load_exempt_url_combined_repos_and_digests(gc: types.ModuleType, tmp_path: Path) -> None:
    """_load_exempt merges both 'repos' and 'digests' fields when both present."""
    import argparse

    body = __import__("json").dumps({
        "repos": ["scratch/aaa"], "digests": ["sha256:old"],
    }).encode()

    def fake_http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        return 200, {}, body

    args = argparse.Namespace(exempt_file=None, exempt_url="http://cp/active")
    with patch.object(gc, "_http", fake_http):
        result = gc._load_exempt(args)
    assert result == frozenset({"scratch/aaa", "sha256:old"})
