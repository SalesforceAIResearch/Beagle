"""Unit tests for the pure logic in ``scripts/build_and_push_images.py``.

The script shells out to ``docker``/``git`` for the actual build+push (covered by
the seta-env validation run, not unit tests), but the ref-rewriting, ref-parsing,
size-aware sharding, and registry existence-probe logic are pure and must be
exactly right — a sharding bug silently drops images from a 1000-image fan-out,
and a ref-parse bug makes the skip-if-present probe never match. Those are tested
here without touching Docker or the network.

The script lives under ``scripts/`` (not a package), so it's loaded by file path.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from xrlenv.control.build_plan import (
    BuildEntry,
    EntryPlacement,
    GitSource,
    LocalSource,
    RegistrySource,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "build_and_push_images.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("xrlenv_build_and_push", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (which looks the module up
    # in sys.modules by __module__) resolves rather than seeing None.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bp = _load_module()


def _entry(image_ref: str, size: int, *, subdir: str = ".") -> BuildEntry:
    return BuildEntry(
        image_ref=image_ref,
        context_source=GitSource(
            repo="https://github.com/camel-ai/seta-env", ref="main", subdir=subdir,
        ),
        placement=EntryPlacement(size_hint_bytes=size, size_hint_source="heuristic"),
    )


# ── target_ref ────────────────────────────────────────────────────────────────


def test_target_ref_prefixes_portable_ref() -> None:
    assert bp.target_ref("internal-ip:5011", "xrlenv-seta-env/0:main") == (
        "internal-ip:5011/xrlenv-seta-env/0:main"
    )


def test_target_ref_is_idempotent_when_already_prefixed() -> None:
    already = "internal-ip:5011/xrlenv-seta-env/0:main"
    assert bp.target_ref("internal-ip:5011", already) == already


def test_target_ref_empty_registry_passes_through() -> None:
    assert bp.target_ref("", "xrlenv-seta-env/0:main") == "xrlenv-seta-env/0:main"


# ── split_ref ─────────────────────────────────────────────────────────────────


def test_split_ref_host_repo_tag_with_port() -> None:
    assert bp.split_ref("internal-ip:5011/xrlenv-seta-env/0:main") == (
        "internal-ip:5011", "xrlenv-seta-env/0", "main",
    )


def test_split_ref_defaults_tag_to_latest() -> None:
    assert bp.split_ref("host:5011/repo/name") == ("host:5011", "repo/name", "latest")


def test_split_ref_digest_reference() -> None:
    host, repo, ref = bp.split_ref("host:5011/repo@sha256:abc123")
    assert (host, repo, ref) == ("host:5011", "repo", "sha256:abc123")


def test_split_ref_port_in_host_not_confused_with_tag() -> None:
    # The ':5011' is a port on the host segment, not a tag — only a ':' AFTER the
    # last '/' is a tag.
    host, repo, ref = bp.split_ref("host:5011/a/b/c")
    assert host == "host:5011" and repo == "a/b/c" and ref == "latest"


# ── partition_entries / select_shard ──────────────────────────────────────────


def test_partition_covers_every_entry_exactly_once() -> None:
    entries = [_entry(f"img/{i}:t", size=(i % 5 + 1) * 100) for i in range(37)]
    buckets = bp.partition_entries(entries, 4)
    assert len(buckets) == 4
    flat = [e.image_ref for b in buckets for e in b]
    assert sorted(flat) == sorted(e.image_ref for e in entries)
    assert len(flat) == len(set(flat)) == 37  # no dup, no drop


def test_partition_balances_by_size_not_count() -> None:
    # One 10 GB giant + many tiny ones: the giant's shard should hold far fewer
    # entries, and total bytes per shard should be close.
    entries = [_entry("giant:t", size=10_000_000_000)]
    entries += [_entry(f"tiny/{i}:t", size=100) for i in range(30)]
    buckets = bp.partition_entries(entries, 3)
    loads = [sum(int(e.placement.size_hint_bytes) for e in b) for b in buckets]
    # The giant dominates one bucket; the other two are ~equal and tiny. Assert
    # the lightest two buckets together carry essentially all the tiny entries.
    giant_bucket = max(range(3), key=lambda i: loads[i])
    assert any(e.image_ref == "giant:t" for e in buckets[giant_bucket])
    others = [b for i, b in enumerate(buckets) if i != giant_bucket]
    assert sum(len(b) for b in others) == 30


def test_partition_is_deterministic() -> None:
    entries = [_entry(f"img/{i}:t", size=(i * 7 % 11) * 1000) for i in range(25)]
    a = bp.partition_entries(entries, 5)
    b = bp.partition_entries(entries, 5)
    assert [[e.image_ref for e in bk] for bk in a] == [
        [e.image_ref for e in bk] for bk in b
    ]


def test_select_shard_partitions_are_disjoint_and_complete() -> None:
    entries = [_entry(f"img/{i}:t", size=(i % 4 + 1)) for i in range(20)]
    seen: set[str] = set()
    for idx in range(6):
        for e in bp.select_shard(entries, idx, 6):
            assert e.image_ref not in seen  # disjoint
            seen.add(e.image_ref)
    assert seen == {e.image_ref for e in entries}  # complete


def test_select_shard_single_shard_owns_everything() -> None:
    entries = [_entry(f"img/{i}:t", size=1) for i in range(10)]
    assert len(bp.select_shard(entries, 0, 1)) == 10


def test_select_shard_rejects_out_of_range() -> None:
    entries = [_entry("img/0:t", size=1)]
    with pytest.raises(ValueError):
        bp.select_shard(entries, 3, 3)


def test_partition_rejects_zero_shards() -> None:
    with pytest.raises(ValueError):
        bp.partition_entries([_entry("img/0:t", size=1)], 0)


def test_registry_source_entries_partition_too() -> None:
    # Sharding is context-source-agnostic — registry-source entries (retag+push)
    # shard exactly like git ones.
    entries = [
        BuildEntry(
            image_ref=f"alexgshaw/task-{i}:rev",
            context_source=RegistrySource(),
            placement=EntryPlacement(size_hint_bytes=i + 1, size_hint_source="registry-probe"),
        )
        for i in range(12)
    ]
    buckets = bp.partition_entries(entries, 3)
    assert sum(len(b) for b in buckets) == 12


# ── split_buildable (unified plan: build type: local, skip type: registry) ─────


def _reg_src_entry(image_ref: str, size: int = 100) -> BuildEntry:
    return BuildEntry(
        image_ref=image_ref,
        context_source=RegistrySource(),
        placement=EntryPlacement(size_hint_bytes=size, size_hint_source="registry-probe"),
    )


def _local_src_entry(image_ref: str, size: int = 100) -> BuildEntry:
    return BuildEntry(
        image_ref=image_ref,
        context_source=LocalSource(
            path="/cache/lhtb/chess-mate/environment",
            dockerfile="Dockerfile",
            shared_fs="hyperpod",
        ),
        placement=EntryPlacement(size_hint_bytes=size, size_hint_source="heuristic"),
    )


def test_split_buildable_separates_registry_from_built() -> None:
    # A unified LHTB-shape plan: type: local (built) + type: registry (docker.io,
    # served via the :5010 mirror — nothing to build). git entries are buildable too.
    entries = [
        _local_src_entry("reg:5011/lhtb/chess-mate:main"),
        _reg_src_entry("zli12321/lhtb-2048:x"),
        _entry("reg:5011/seta-env/0:main", size=100),  # GitSource → buildable
        _reg_src_entry("zli12321/lhtb-sokoban:x"),
    ]
    buildable, registry_only = bp.split_buildable(entries)
    assert {e.image_ref for e in buildable} == {
        "reg:5011/lhtb/chess-mate:main", "reg:5011/seta-env/0:main",
    }
    assert {e.image_ref for e in registry_only} == {
        "zli12321/lhtb-2048:x", "zli12321/lhtb-sokoban:x",
    }


def test_split_buildable_all_local_leaves_registry_empty() -> None:
    # terminalworld-shape (all local) → nothing skipped.
    buildable, registry_only = bp.split_buildable(
        [_local_src_entry("a:1"), _entry("b:1", size=100)],
    )
    assert len(buildable) == 2
    assert registry_only == []


# ── manifest_url + registry_has_manifest (injected opener) ─────────────────────


def test_manifest_url_format() -> None:
    assert bp.manifest_url("http", "h:5011", "repo/name", "main") == (
        "http://h:5011/v2/repo/name/manifests/main"
    )


class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *a: object) -> None:
        return None


class _Opener:
    def __init__(self, behavior: Any) -> None:
        self._behavior = behavior
        self.last_url: str | None = None

    def urlopen(self, req: Any, timeout: float | None = None) -> Any:
        self.last_url = req.full_url
        if isinstance(self._behavior, Exception):
            raise self._behavior
        return _FakeResp(self._behavior)


def test_registry_has_manifest_present() -> None:
    opener = _Opener(200)
    assert bp.registry_has_manifest("h:5011/repo:main", opener=opener) is True
    assert opener.last_url == "http://h:5011/v2/repo/manifests/main"


def test_registry_has_manifest_absent_is_false() -> None:
    err = urllib.error.HTTPError("u", 404, "nf", {}, None)  # type: ignore[arg-type]
    assert bp.registry_has_manifest("h:5011/repo:main", opener=_Opener(err)) is False


def test_registry_has_manifest_unreachable_is_none() -> None:
    # Registry down → unknown → caller falls through to build (push surfaces it).
    err = urllib.error.URLError("connection refused")
    assert bp.registry_has_manifest("h:5011/repo:main", opener=_Opener(err)) is None


def test_registry_has_manifest_auth_rejected_is_none() -> None:
    err = urllib.error.HTTPError("u", 401, "unauth", {}, None)  # type: ignore[arg-type]
    assert bp.registry_has_manifest("h:5011/repo:main", opener=_Opener(err)) is None


# ── _should_prune (periodic build-cache prune trigger) ────────────────────────

_GB = 1_000_000_000


def test_should_prune_on_build_count() -> None:
    assert bp._should_prune(builds_since=25, prune_every=25, free_bytes=500 * _GB, min_free_bytes=0)
    assert not bp._should_prune(builds_since=24, prune_every=25, free_bytes=500 * _GB, min_free_bytes=0)


def test_should_prune_on_low_disk() -> None:
    # Plenty of builds left before the count trigger, but disk is low → prune.
    assert bp._should_prune(builds_since=3, prune_every=25, free_bytes=20 * _GB, min_free_bytes=30 * _GB)
    assert not bp._should_prune(builds_since=3, prune_every=25, free_bytes=40 * _GB, min_free_bytes=30 * _GB)


def test_should_prune_thresholds_disabled_with_zero() -> None:
    # both off → never prune, even at high count / zero free
    assert not bp._should_prune(builds_since=10_000, prune_every=0, free_bytes=0, min_free_bytes=0)
    # count off, disk guard on
    assert bp._should_prune(builds_since=10_000, prune_every=0, free_bytes=1 * _GB, min_free_bytes=30 * _GB)
    # disk off, count guard on
    assert bp._should_prune(builds_since=25, prune_every=25, free_bytes=0, min_free_bytes=0)


# ── _CloneCache: clone-once on shared disk (real local git repo, no Docker) ────

_HAS_GIT = shutil.which("git") is not None


def _make_repo(path: Path) -> None:
    path.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    run = lambda *a: subprocess.run(a, cwd=path, env=env, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q", "-b", "main")
    (path / "Dockerfile").write_text("FROM scratch\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "init")


@pytest.mark.skipif(not _HAS_GIT, reason="git not available")
async def test_clone_once_reused_across_shards(tmp_path: Path) -> None:
    # A repo cloned by one "shard" is reused by another shard (a *fresh*
    # _CloneCache over the SAME shared cache root) without re-cloning — the whole
    # point of the shared-FSx clone-once design the user asked about.
    repo = tmp_path / "repo"
    _make_repo(repo)
    cache_root = tmp_path / "cache"
    repo_url = repo.as_uri()  # file:// URL clones cleanly with --branch

    shard0 = bp._CloneCache(root=cache_root)
    checkout0 = await shard0.ensure(repo_url, "main", timeout_s=60.0)
    assert (checkout0 / "Dockerfile").is_file()
    # Drop a sentinel INTO the shared checkout; if a second shard re-cloned, the
    # atomic clone-to-tmp-then-rename would replace the dir and wipe the sentinel.
    (checkout0 / "SENTINEL").write_text("x")

    shard1 = bp._CloneCache(root=cache_root)  # simulates a different node/process
    checkout1 = await shard1.ensure(repo_url, "main", timeout_s=60.0)
    assert checkout1 == checkout0
    assert (checkout1 / "SENTINEL").is_file()  # reused, not re-cloned
    assert (cache_root / f"{bp._safe(repo_url)}__main" / ".complete").is_file()


@pytest.mark.skipif(not _HAS_GIT, reason="git not available")
async def test_refresh_context_reclones(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    cache_root = tmp_path / "cache"
    repo_url = repo.as_uri()

    first = await bp._CloneCache(root=cache_root).ensure(repo_url, "main", timeout_s=60.0)
    (first / "SENTINEL").write_text("x")
    # refresh=True forces a fresh clone → the sentinel is gone.
    refreshed = await bp._CloneCache(root=cache_root, refresh=True).ensure(
        repo_url, "main", timeout_s=60.0,
    )
    assert (refreshed / "Dockerfile").is_file()
    assert not (refreshed / "SENTINEL").is_file()


# ── LocalSource: build a directory in place (no clone, no extract) ─────────────


def _local_entry(image_ref: str, path: str, *, shared_fs: str = "hyperpod") -> BuildEntry:
    return BuildEntry(
        image_ref=image_ref,
        context_source=LocalSource(path=path, shared_fs=shared_fs),
        placement=EntryPlacement(size_hint_bytes=1, size_hint_source="heuristic"),
    )


async def test_build_one_local_source_builds_path_in_place(
    tmp_path: Path, monkeypatch,
) -> None:
    """A ``local`` entry docker-builds ``path`` directly — the build context arg
    is the path itself (no clone-cache dir, no tarball extract dir)."""
    ctx = tmp_path / "environment"
    ctx.mkdir()
    (ctx / "Dockerfile").write_text("FROM scratch\n")

    calls: list[list[str]] = []

    async def fake_run(argv, *, cwd=None, timeout_s, env=None):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        if argv[:2] == ["docker", "inspect"]:
            return 0, "reg:5011/turing-tb2/t:main@sha256:deadbeef\n"
        return 0, ""

    monkeypatch.setattr(bp, "_run", fake_run)
    rec = await bp.build_one(
        _local_entry("turing-tb2/t:main", str(ctx)),
        registry="reg:5011", scheme="http",
        clone_cache=bp._CloneCache(root=tmp_path / "cc"),
        build_timeout_s=60.0, force=True, prune=False, probe_timeout_s=5.0,
    )
    assert rec["status"] == "built", rec
    build_calls = [c for c in calls if c[:2] == ["docker", "build"]]
    assert build_calls, "expected a docker build"
    # Built the local dir in place: last arg (the context) IS the path, and the
    # tag is the registry-prefixed ref.
    assert build_calls[0][-1] == str(ctx)
    assert "reg:5011/turing-tb2/t:main" in build_calls[0]
    # No clone happened (the clone-cache root stays empty).
    assert not (tmp_path / "cc").exists() or not any((tmp_path / "cc").iterdir())


async def test_build_one_local_source_missing_path_fails_clearly(
    tmp_path: Path, monkeypatch,
) -> None:
    """A path absent on this build host fails with a message naming shared_fs —
    the operator's cue that FSx isn't mounted on this node."""
    async def fake_run(argv, *, cwd=None, timeout_s, env=None):  # type: ignore[no-untyped-def]
        return 0, ""

    monkeypatch.setattr(bp, "_run", fake_run)
    rec = await bp.build_one(
        _local_entry("turing-tb2/t:main", str(tmp_path / "nope"), shared_fs="hyperpod"),
        registry="reg:5011", scheme="http",
        clone_cache=bp._CloneCache(root=tmp_path / "cc"),
        build_timeout_s=60.0, force=True, prune=False, probe_timeout_s=5.0,
    )
    assert rec["status"] == "failed"
    assert "hyperpod" in (rec["error"] or "")
    assert "shared filesystem" in (rec["error"] or "")
