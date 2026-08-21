"""Tests for the Slice 6 image cache (spec 15).

Covers:
- ``ImageCacheManager.tier`` classification across all four phase-0 tiers
- ``ensure_present`` cold-pull + cache-hit + concurrent-coalesce + eviction
- LRU ordering: oldest non-pinned, non-in-use cold images go first
- Pinned + in-use images are never evicted
- ``ImageCacheConfig`` thresholds drive the evict/target free-disk math
- Pin loader (``image-pins.yaml``)
- ``NodeAgent`` create/destroy lifecycle bumps + drops the refcount
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

import pytest
import yaml
from xrlenv.backends.base import (
    ExecChunk,
    ImageInUse,
    ImageRecord,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxBackend,
    SandboxCapabilities,
    SandboxHandle,
    ServiceSpec,
    SnapshotID,
    TemplateRef,
)
from xrlenv.errors import ManifestInvalid
from xrlenv.node.agent import NodeAgent, NodeAgentConfig
from xrlenv.node.image_cache import (
    EvictionTier,
    ImageCacheConfig,
    ImageCacheManager,
    default_ownership_classifier,
    default_tier_classifier,
)
from xrlenv.node.image_pins import load_image_pins

# ──────────────────────────────────────────────────────────────────────────────
# Fake backend (records every image-management call)
# ──────────────────────────────────────────────────────────────────────────────


class _FakeImageBackend(SandboxBackend):
    name = "fake-img"
    capabilities = SandboxCapabilities(
        supports_snapshot=False, supports_chainable_snapshot=False,
        live_state_captured=False, supports_port_forward=False,
        supports_gpu=False, isolation_class="container", fast_create_p50_ms=10,
    )

    def __init__(
        self,
        *,
        present: list[ImageRecord] | None = None,
        free_bytes: int = 100 * 1024**3,
        # Default to a fixed, modest total so eviction tests are
        # deterministic. Previously ``None`` fell through to a real
        # ``statvfs`` of the test machine — harmless while eviction
        # ignored total, but once the image cache floors its evict band
        # at the shared disk-policy admit floor (5% of total), a huge
        # host disk made that floor dominate the tests' hand-set
        # thresholds. 100 GiB keeps the admit floor (5% = 5 GiB) well
        # below the adaptive thresholds these tests exercise. Pass an
        # explicit value where a test needs a specific total.
        total_bytes: int | None = 100 * 1024**3,
    ) -> None:
        self._present: dict[str, ImageRecord] = {
            r.name: r for r in (present or [])
        }
        self._free = free_bytes
        # ``None`` keeps the protocol-default sentinel ("unknown total").
        # The adaptive eviction model ignores total disk entirely (it sizes
        # headroom from the largest cached image), so ``total_bytes`` is
        # only here for tests that assert on the reported disk *state*.
        self._total = total_bytes
        self.pulled: list[str] = []
        self.removed: list[str] = []
        self.pull_delay_s: float = 0.0
        self.pull_should_fail: set[str] = set()

    # Image-cache surface ----------------------------------------------------

    async def list_images(self, *, include_shared_size=False) -> list[ImageRecord]:
        return list(self._present.values())

    async def pull_image(self, image: str, *, timeout_s: float = 600.0) -> None:
        if self.pull_delay_s:
            await asyncio.sleep(self.pull_delay_s)
        if image in self.pull_should_fail:
            raise RuntimeError(f"pull boom: {image}")
        self.pulled.append(image)
        # Simulate the pulled image landing on disk.
        self._present[image] = ImageRecord(name=image, size_bytes=1024 * 1024)
        self._free -= self._present[image].size_bytes

    async def remove_image(self, image: str, *, force: bool = False) -> None:
        if image in self._present:
            self._free += self._present[image].size_bytes
            del self._present[image]
        self.removed.append(image)

    async def free_disk_bytes(self) -> int:
        return self._free

    async def total_disk_bytes(self) -> int:
        if self._total is None:
            return await super().total_disk_bytes()
        return self._total

    def set_free_bytes(self, n: int) -> None:
        self._free = n

    # Sandbox-lifecycle abstract methods (no-op for these tests) ------------

    async def create(
        self, template: TemplateRef, resources: ResourceSpec, network_policy: NetworkPolicy,
    ) -> SandboxHandle:
        return SandboxHandle(
            id="sb-fake", backend=self.name,
            backend_ref="cid-fake", stub_endpoint="",
        )

    async def destroy(self, sb: SandboxHandle) -> None:
        return None

    def exec(
        self, sb: SandboxHandle, cmd: list[str], stdin: bytes | None = None,
        env: dict[str, str] | None = None, timeout_s: float | None = None,
    ) -> AsyncIterator[ExecChunk]:
        raise NotImplementedError

    async def read_file(self, sb: SandboxHandle, path: str) -> bytes:
        raise NotImplementedError

    async def write_file(self, sb: SandboxHandle, path: str, data: bytes) -> None:
        raise NotImplementedError

    async def put_archive(
        self, sb: SandboxHandle, target_dir: str, tarball: bytes, *, clean_target: bool = False,
    ) -> None:
        raise NotImplementedError

    def read_file_stream(self, sb: SandboxHandle, path: str) -> AsyncIterator[bytes]:
        raise NotImplementedError

    async def write_file_stream(
        self, sb: SandboxHandle, path: str, src: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError

    async def spawn_service(self, sb: SandboxHandle, spec: ServiceSpec) -> object:
        raise NotImplementedError

    async def spawn_services(
        self, sb: SandboxHandle, specs: list[ServiceSpec],
    ) -> list[object]:
        raise NotImplementedError

    async def port_forward(self, sb: SandboxHandle, internal_port: int) -> str:
        raise NotImplementedError

    async def snapshot(self, sb: SandboxHandle) -> SnapshotID:
        raise NotImplementedError

    async def restore(self, snapshot: SnapshotID) -> SandboxHandle:
        raise NotImplementedError

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Pin loader
# ──────────────────────────────────────────────────────────────────────────────


def test_load_image_pins_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_image_pins(tmp_path / "nope.yaml") == set()


def test_load_image_pins_parses_list(tmp_path: Path) -> None:
    f = tmp_path / "pins.yaml"
    f.write_text(yaml.safe_dump({"pins": ["a:1", "b:2", "c:3"]}))
    assert load_image_pins(f) == {"a:1", "b:2", "c:3"}


def test_load_image_pins_rejects_bad_top_level(tmp_path: Path) -> None:
    f = tmp_path / "pins.yaml"
    f.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(ManifestInvalid):
        load_image_pins(f)


def test_load_image_pins_rejects_non_string_entry(tmp_path: Path) -> None:
    f = tmp_path / "pins.yaml"
    f.write_text(yaml.safe_dump({"pins": ["ok", 42]}))
    with pytest.raises(ManifestInvalid):
        load_image_pins(f)


# ──────────────────────────────────────────────────────────────────────────────
# Tier classification
# ──────────────────────────────────────────────────────────────────────────────


async def test_tier_in_use_wins_over_pinned() -> None:
    cache = ImageCacheManager(backend=_FakeImageBackend(), pins={"img:1"})
    cache.acquire("img:1")
    assert cache.tier("img:1") == "in_use"


async def test_tier_pinned() -> None:
    cache = ImageCacheManager(backend=_FakeImageBackend(), pins={"pin:1"})
    assert cache.tier("pin:1") == "pinned"


async def test_tier_recently_used_within_window() -> None:
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(recent_window_s=10.0),
    )
    cache.acquire("hot:1")
    cache.release("hot:1")
    # Just released — last_used is "now".
    assert cache.tier("hot:1") == "recently_used"


async def test_tier_cold_when_never_seen() -> None:
    cache = ImageCacheManager(backend=_FakeImageBackend())
    assert cache.tier("never-touched:1") == "cold"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-slice 2 — build-time grace window
# ──────────────────────────────────────────────────────────────────────────────


async def test_tier_recently_used_within_build_grace_window() -> None:
    """A freshly-built image with no acquire touch yet sorts as
    ``recently_used`` while inside the grace window. Without this,
    ``ensure_present`` could finish a build seconds before the
    eviction loop runs and the image would be reaped immediately —
    forcing a rebuild on the very next acquire."""
    import time as _time

    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            build_grace_window_s=600.0, recent_window_s=10.0,
        ),
    )
    # Simulate ensure_present's stamp.
    cache._built_at["fresh-build:1"] = _time.monotonic()
    assert cache.tier("fresh-build:1") == "recently_used"


async def test_tier_cold_after_build_grace_window_expires() -> None:
    """Once the grace window passes, an unused freshly-built image
    falls to ``cold`` so it doesn't pin disk forever (the operator
    might have applied a plan they don't actually use)."""
    import time as _time

    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            build_grace_window_s=10.0, recent_window_s=5.0,
        ),
    )
    # Stamp ``built_at`` 1 hour ago — long past the 10s grace.
    cache._built_at["stale-build:1"] = _time.monotonic() - 3600.0
    assert cache.tier("stale-build:1") == "cold"


async def test_acquire_clears_build_grace_window() -> None:
    """First real acquire resets the grace tracker — standard LRU
    semantics take over from there. Without this, the grace
    window would stay attached forever and skew tier sort even
    after the image has been actively used."""
    import time as _time

    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(build_grace_window_s=600.0),
    )
    cache._built_at["just-built:1"] = _time.monotonic()
    cache.acquire("just-built:1")
    cache.release("just-built:1")
    assert "just-built:1" not in cache._built_at


async def test_ensure_present_stamps_build_grace_after_producer() -> None:
    """When ensure_present invokes the builder hook (build-on-
    acquire OR a fresh BuildImageCommand), the grace window is
    stamped automatically."""
    backend = _FakeImageBackend(free_bytes=100 * 1024**3)

    async def fake_producer(_ref: str, _timeout: float) -> None:
        # Mark image present so the cache "sees" the build result.
        backend._present["produced:1"] = ImageRecord(
            name="produced:1", size_bytes=1024,
        )

    cache = ImageCacheManager(
        backend=backend,
        builder_lookup=lambda r: fake_producer if r == "produced:1" else None,
        config=ImageCacheConfig(build_grace_window_s=600.0),
    )
    await cache.ensure_present("produced:1")
    assert "produced:1" in cache._built_at


async def test_ensure_present_does_not_stamp_grace_for_registry_pull() -> None:
    """Registry pulls (not source builds) don't get the grace
    window — they're cheap to redo, and the existing recent_window
    on first cache hit covers the typical re-acquire window."""
    backend = _FakeImageBackend(free_bytes=100 * 1024**3)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(build_grace_window_s=600.0),
    )
    # No builder_lookup → fall through to backend.pull_image.
    await cache.ensure_present("registry-pull:1")
    assert "registry-pull:1" not in cache._built_at


# ──────────────────────────────────────────────────────────────────────────────
# ensure_present
# ──────────────────────────────────────────────────────────────────────────────


async def test_ensure_present_pulls_when_absent() -> None:
    backend = _FakeImageBackend(free_bytes=100 * 1024**3)
    cache = ImageCacheManager(backend=backend)
    await cache.ensure_present("a:1")
    assert backend.pulled == ["a:1"]


async def test_ensure_present_skips_when_already_local() -> None:
    backend = _FakeImageBackend(
        present=[ImageRecord(name="a:1", size_bytes=1024)],
        free_bytes=100 * 1024**3,
    )
    cache = ImageCacheManager(backend=backend)
    await cache.ensure_present("a:1")
    assert backend.pulled == []


async def test_ensure_present_uses_backend_image_exists_for_digest_form() -> None:
    """Regression: locally-built images pinned to a content-addressed
    Id (``<repo>@sha256:<id>``) must not trigger a doomed registry
    pull. ``ImageCacheManager._is_present`` defers to
    ``backend.image_exists()`` which Docker implements via
    ``images.get(ref)`` so both the tag form and the digest form
    resolve to the same locally-built image. Pin the contract with a
    fake backend whose ``image_exists`` knows about the digest form
    even though ``list_images`` only carries the tag form."""

    class _DigestAwareBackend(_FakeImageBackend):
        async def image_exists(self, image: str) -> bool:
            # The catalog-pinned digest form maps to the tag we have locally.
            if image == "xrlenv/hello-shell@sha256:dd24945":
                return True
            return await super().image_exists(image)

    backend = _DigestAwareBackend(
        present=[ImageRecord(name="xrlenv/hello-shell:0.1", size_bytes=1024)],
        free_bytes=100 * 1024**3,
    )
    cache = ImageCacheManager(backend=backend)
    # The cache asks for the digest-form ref the catalog pinned. The
    # backend's image_exists answers "yes, locally" → no pull.
    await cache.ensure_present("xrlenv/hello-shell@sha256:dd24945")
    assert backend.pulled == []


async def test_ensure_present_concurrent_coalesces_into_single_pull() -> None:
    backend = _FakeImageBackend(free_bytes=100 * 1024**3)
    backend.pull_delay_s = 0.05
    cache = ImageCacheManager(backend=backend)
    await asyncio.gather(*(cache.ensure_present("z:1") for _ in range(5)))
    # Only one pull recorded despite five concurrent ensure_present calls.
    assert backend.pulled == ["z:1"]


async def test_ensure_present_evicts_cold_when_under_threshold() -> None:
    cold = ImageRecord(name="cold:1", size_bytes=10 * 1024**3)
    backend = _FakeImageBackend(
        present=[cold],
        free_bytes=5 * 1024**3,  # below the 15 GB default threshold
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_bytes=15 * 1024**3,
            evict_target_bytes=20 * 1024**3,
        ),
    )
    await cache.ensure_present("new:1")
    # Cold image evicted before pulling new:1.
    assert "cold:1" in backend.removed
    assert "new:1" in backend.pulled


async def test_ensure_present_skips_eviction_above_threshold() -> None:
    backend = _FakeImageBackend(
        present=[ImageRecord(name="cold:1", size_bytes=1024)],
        free_bytes=50 * 1024**3,  # well above threshold
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_bytes=15 * 1024**3,
            evict_target_bytes=20 * 1024**3,
        ),
    )
    await cache.ensure_present("new:1")
    assert backend.removed == []
    assert backend.pulled == ["new:1"]


# ──────────────────────────────────────────────────────────────────────────────
# Eviction LRU ordering + safety
# ──────────────────────────────────────────────────────────────────────────────


async def test_evict_skips_pinned_and_in_use() -> None:
    pinned = ImageRecord(name="pin:1", size_bytes=10 * 1024**3)
    inuse = ImageRecord(name="run:1", size_bytes=10 * 1024**3)
    cold = ImageRecord(name="cold:1", size_bytes=10 * 1024**3)
    backend = _FakeImageBackend(
        present=[pinned, inuse, cold], free_bytes=1 * 1024**3,
    )
    cache = ImageCacheManager(
        backend=backend, pins={"pin:1"},
        config=ImageCacheConfig(
            evict_threshold_bytes=20 * 1024**3,
            evict_target_bytes=25 * 1024**3,
        ),
    )
    cache.acquire("run:1")
    await cache.ensure_present("new:1")
    # Only the cold image got evicted; pinned + in-use survive.
    assert backend.removed == ["cold:1"]


async def test_evict_orders_oldest_first() -> None:
    a = ImageRecord(name="a:1", size_bytes=8 * 1024**3)
    b = ImageRecord(name="b:1", size_bytes=8 * 1024**3)
    c = ImageRecord(name="c:1", size_bytes=8 * 1024**3)
    backend = _FakeImageBackend(present=[a, b, c], free_bytes=2 * 1024**3)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_bytes=20 * 1024**3,
            evict_target_bytes=18 * 1024**3,
        ),
    )
    # Touch in order so c:1 is "freshest", a:1 is "oldest".
    cache.acquire("a:1")
    cache.release("a:1")
    await asyncio.sleep(0.001)
    cache.acquire("b:1")
    cache.release("b:1")
    await asyncio.sleep(0.001)
    cache.acquire("c:1")
    cache.release("c:1")
    await cache.ensure_present("new:1")
    # Should have evicted a:1 first (oldest).
    assert backend.removed[0] == "a:1"


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive eviction headroom (2026-06 — replaces fraction-of-disk thresholds)
#
# Headroom = clamp(slots x largest_cached_image x safety, floor, cap). It
# scales with the workload's image size, not the disk size, so a 500 GiB
# and a 1 TiB node reserve the same modest pull-burst buffer instead of a
# fixed fraction of the whole disk.
# ──────────────────────────────────────────────────────────────────────────────


def test_adaptive_headroom_falls_back_to_floor_on_empty_cache() -> None:
    # Cold start: no image observed yet → largest is unknown (0), so the
    # absolute floor applies (a fresh node still keeps a sane buffer
    # rather than reserving nothing).
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            evict_threshold_bytes=15 * 1024**3,
            evict_target_bytes=25 * 1024**3,
        ),
    )
    assert cache._effective_threshold() == 15 * 1024**3
    assert cache._effective_target() == 25 * 1024**3


async def test_adaptive_headroom_scales_with_largest_image() -> None:
    # largest = 4 GiB, slots = 4, safety = 1.5 →
    #   START = 4 x 4 x 1.5 = 24 GiB (between the 15 floor / 50 cap)
    #   STOP  = 5 x 4 x 1.5 = 30 GiB (between the 25 floor / 75 cap)
    backend = _FakeImageBackend(
        present=[ImageRecord(name="big:1", size_bytes=4 * 1024**3)],
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_headroom_slots=4,
            evict_disk_safety_factor=1.5,
        ),
    )
    await cache._refresh_image_stats()
    assert cache._effective_threshold() == 24 * 1024**3
    assert cache._effective_target() == 30 * 1024**3


async def test_adaptive_headroom_clamped_to_cap() -> None:
    # A pathologically large base image must not reserve an unbounded
    # buffer — the per-node cap is the safety ceiling that bounds it.
    backend = _FakeImageBackend(
        present=[ImageRecord(name="huge:1", size_bytes=200 * 1024**3)],
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_cap_bytes=50 * 1024**3,
            evict_target_cap_bytes=75 * 1024**3,
        ),
    )
    await cache._refresh_image_stats()
    assert cache._effective_threshold() == 50 * 1024**3
    assert cache._effective_target() == 75 * 1024**3


async def test_adaptive_headroom_independent_of_total_disk() -> None:
    # The pre-2026-06 model compared free / total; the adaptive model
    # ignores total entirely. Same free + same cached image but wildly
    # different totals (a 296 GiB node vs a 2 TiB node) must reserve
    # identically.
    async def _thresh(total: int) -> tuple[int, int]:
        backend = _FakeImageBackend(
            present=[ImageRecord(name="img:1", size_bytes=4 * 1024**3)],
            free_bytes=40 * 1024**3,
            total_bytes=total,
        )
        cache = ImageCacheManager(backend=backend)
        await cache._refresh_image_stats()
        return cache._effective_threshold(), cache._effective_target()

    assert await _thresh(296 * 1024**3) == await _thresh(2 * 1024**4)


async def test_sweep_loop_evicts_under_pressure_without_acquire() -> None:
    # The sweep is the load-bearing fix: even when no ensure_present
    # is firing (steady-state on cached images), if free disk drops
    # below the threshold the periodic loop should free space.
    total = 296 * 1024**3
    backend = _FakeImageBackend(
        present=[ImageRecord(name="cold:1", size_bytes=10 * 1024**3)],
        free_bytes=5 * 1024**3,  # well below thresholds
        total_bytes=total,
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_bytes=15 * 1024**3,
            evict_target_bytes=20 * 1024**3,
            sweep_interval_s=0.01,
        ),
    )
    task = asyncio.create_task(cache.run_sweep_loop())
    try:
        # Poll for the eviction; bound the wait so a regression
        # doesn't hang the suite.
        for _ in range(200):
            if backend.removed:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert "cold:1" in backend.removed


async def test_sweep_loop_noop_when_above_threshold() -> None:
    # Idle cluster + free disk above threshold → sweep should not
    # touch a single image (matches the user's constraint that an
    # idle, healthy cluster never triggers eviction). largest = 10 GiB →
    # adaptive START headroom = min(4 x 10 x 1.5, 50 cap) = 50 GiB; 200
    # GiB free is comfortably above it.
    total = 296 * 1024**3
    backend = _FakeImageBackend(
        present=[ImageRecord(name="cold:1", size_bytes=10 * 1024**3)],
        free_bytes=200 * 1024**3,
        total_bytes=total,
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            sweep_interval_s=0.01,
        ),
    )
    task = asyncio.create_task(cache.run_sweep_loop())
    try:
        # Let several ticks fire.
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert backend.removed == []


async def test_sweep_loop_returns_immediately_when_disabled() -> None:
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(sweep_interval_s=0.0),
    )
    # No timeout wrapper needed — the loop should exit on its own
    # when the configured interval is zero.
    await cache.run_sweep_loop()


async def test_sweep_loop_survives_backend_exception() -> None:
    # A transient backend failure (docker daemon hiccup) inside one
    # sweep tick should not kill the loop — the next tick must still
    # fire. This protects the long-lived background task from being
    # silently torn down by a single bad call.
    class _FlakyBackend(_FakeImageBackend):
        def __init__(self) -> None:
            super().__init__(
                present=[ImageRecord(name="cold:1", size_bytes=10 * 1024**3)],
                free_bytes=5 * 1024**3,
                total_bytes=296 * 1024**3,
            )
            self.calls = 0

        async def free_disk_bytes(self) -> int:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient docker daemon hiccup")
            return await super().free_disk_bytes()

    backend = _FlakyBackend()
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_bytes=15 * 1024**3,
            evict_target_bytes=20 * 1024**3,
            sweep_interval_s=0.01,
        ),
    )
    task = asyncio.create_task(cache.run_sweep_loop())
    try:
        for _ in range(200):
            if backend.removed:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert "cold:1" in backend.removed


# ──────────────────────────────────────────────────────────────────────────────
# Refcount
# ──────────────────────────────────────────────────────────────────────────────


def test_in_use_refcount_increments_and_clamps() -> None:
    cache = ImageCacheManager(backend=_FakeImageBackend())
    for _ in range(3):
        cache.acquire("x:1")
    assert cache.in_use_count("x:1") == 3
    for _ in range(2):
        cache.release("x:1")
    assert cache.in_use_count("x:1") == 1
    for _ in range(2):
        cache.release("x:1")
    # Extra release is a no-op (idempotent destroy).
    assert cache.in_use_count("x:1") == 0


def test_pin_unpin_runtime() -> None:
    cache = ImageCacheManager(backend=_FakeImageBackend())
    cache.pin("p:1")
    assert "p:1" in cache.pins
    cache.unpin("p:1")
    assert "p:1" not in cache.pins


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────


async def test_report_classifies_each_image_into_a_tier() -> None:
    backend = _FakeImageBackend(
        present=[
            ImageRecord(name="pinned:1", size_bytes=1),
            ImageRecord(name="inuse:1", size_bytes=2),
            ImageRecord(name="cold:1", size_bytes=3),
        ],
        free_bytes=42,
    )
    cache = ImageCacheManager(backend=backend, pins={"pinned:1"})
    cache.acquire("inuse:1")
    rep = await cache.report()
    by_name = {img.name: img for img in rep.images}
    assert by_name["pinned:1"].tier == "pinned"
    assert by_name["pinned:1"].pinned is True
    assert by_name["inuse:1"].tier == "in_use"
    assert by_name["inuse:1"].in_use_count == 1
    assert by_name["cold:1"].tier == "cold"
    assert rep.free_disk_bytes == 42
    assert rep.pinned == ("pinned:1",)


# ──────────────────────────────────────────────────────────────────────────────
# report() daemon-free cache serving (admin /images responsiveness under build)
# ──────────────────────────────────────────────────────────────────────────────


class _CountingListBackend(_FakeImageBackend):
    """Counts ``list_images`` calls so tests can assert which ``report()``
    paths hit the Docker daemon vs serve from the in-memory cache."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.list_calls = 0

    async def list_images(
        self, *, include_shared_size: bool = False,
    ) -> list[ImageRecord]:
        self.list_calls += 1
        return await super().list_images(include_shared_size=include_shared_size)


async def test_report_serves_from_fresh_cache_without_relisting() -> None:
    # The admin /images hot path: a second report() within one sweep
    # interval must NOT hit docker images.list again — that live call is
    # what contended with a running build and stalled the page.
    backend = _CountingListBackend(
        present=[ImageRecord(name="a:1", size_bytes=1)], free_bytes=42,
    )
    cache = ImageCacheManager(backend=backend)
    await cache.report()   # cold → one live list
    await cache.report()   # fresh cache → served from memory
    assert backend.list_calls == 1


async def test_report_relists_when_cache_stale() -> None:
    backend = _CountingListBackend(
        present=[ImageRecord(name="a:1", size_bytes=1)], free_bytes=42,
    )
    cache = ImageCacheManager(
        backend=backend, config=ImageCacheConfig(sweep_interval_s=60.0),
    )
    await cache.report()           # cold → live list (count=1)
    cache._cached_images_at = 0.0  # force "older than a sweep interval"
    await cache.report()           # stale → live list again
    assert backend.list_calls == 2


async def test_report_with_shared_size_always_lists_live() -> None:
    # calibrate / the budget provider need SharedSize, which the cache never
    # carries — so include_shared_size=True must bypass the cache even when
    # it is fresh.
    backend = _CountingListBackend(
        present=[ImageRecord(name="a:1", size_bytes=1)], free_bytes=42,
    )
    cache = ImageCacheManager(backend=backend)
    await cache.report()                          # warms the cache (count=1)
    await cache.report(include_shared_size=True)  # must go live (count=2)
    assert backend.list_calls == 2


async def test_eviction_skips_image_held_by_external_container() -> None:
    # A non-xrlenv container (e.g. a node monitoring sidecar) holds an
    # image's layers; the backend raises ImageInUse. Eviction must skip it
    # quietly and still reclaim the other cold images — not abort the sweep.
    from xrlenv.backends.base import ImageInUse

    class _Backend(_FakeImageBackend):
        async def remove_image(self, image: str, *, force: bool = False) -> None:
            if image == "held:1":
                raise ImageInUse(image)
            await super().remove_image(image, force=force)

    held = ImageRecord(name="held:1", size_bytes=8 * 1024**3)
    reclaimable = ImageRecord(name="free:1", size_bytes=8 * 1024**3)
    backend = _Backend(present=[held, reclaimable], free_bytes=2 * 1024**3)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            # Pin headroom to the floors so exactly one reclaim is needed.
            evict_headroom_slots=0,
            evict_disk_safety_factor=1.0,
            evict_threshold_bytes=20 * 1024**3,
            evict_target_bytes=10 * 1024**3,
        ),
    )
    # held:1 sorts first (insertion order, equal tier/age) → attempted first,
    # raises ImageInUse → skipped → free:1 reclaimed. No exception escapes.
    await cache.ensure_present("new:1")
    assert "free:1" in backend.removed
    assert "held:1" not in backend.removed
    assert "new:1" in backend.pulled


# ──────────────────────────────────────────────────────────────────────────────
# NodeAgent integration
# ──────────────────────────────────────────────────────────────────────────────


async def test_node_agent_report_images_delegates_to_cache() -> None:
    """B7.6 (P1.2.c): ``NodeAgent.report_images()`` returns the wired
    cache's full snapshot — the wire format the admin ``/images`` route
    consumes via ``ReportImagesCommand``. Pinned by exercising
    every observable field (tier classification, size, in-use count,
    pinned flag, free-disk).
    """
    rec_a = ImageRecord(name="bench/task-a:1", size_bytes=2 * 1024**3)
    rec_b = ImageRecord(name="bench-base/task-a:1", size_bytes=8 * 1024**3)
    rec_c = ImageRecord(name="ops/sidecar:1", size_bytes=1 * 1024**3)
    backend = _FakeImageBackend(
        present=[rec_a, rec_b, rec_c], free_bytes=20 * 1024**3,
    )
    cache = ImageCacheManager(backend=backend, pins={"ops/sidecar:1"})
    cache.acquire("bench/task-a:1")  # in_use
    agent = NodeAgent(
        NodeAgentConfig(node_id="t", backends={backend.name: backend}),
        image_cache=cache,
    )
    report = await agent.report_images()

    by_name = {img.name: img for img in report.images}
    assert by_name["bench/task-a:1"].tier == "in_use"
    assert by_name["bench/task-a:1"].in_use_count == 1
    assert by_name["bench-base/task-a:1"].tier == "cold"
    assert by_name["ops/sidecar:1"].tier == "pinned"
    assert by_name["ops/sidecar:1"].pinned is True
    assert report.free_disk_bytes == 20 * 1024**3
    assert "ops/sidecar:1" in report.pinned


async def test_node_agent_report_images_without_cache_returns_empty() -> None:
    """Defensive: NodeAgent constructed without an image cache (test
    fixtures, minimal LocalRuntime) returns an empty report instead
    of crashing — keeps the admin /images path honest about partial
    visibility.
    """
    backend = _FakeImageBackend()
    agent = NodeAgent(
        NodeAgentConfig(node_id="t", backends={backend.name: backend}),
        # image_cache=None
    )
    report = await agent.report_images()
    assert report.images == []
    assert report.free_disk_bytes == 0
    assert report.pinned == ()


async def test_node_agent_acquires_and_releases_through_cache() -> None:
    backend = _FakeImageBackend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="t", backends={backend.name: backend}),
        image_cache=cache,
    )
    handle = await agent.create_sandbox(
        rollout_id="rid-1", backend=backend.name,
        template=TemplateRef(name="hello", image="hello:1"),
        resources=ResourceSpec(
            cpu_request=0.1, cpu_limit=1.0,
            mem_request_bytes=1024, mem_limit_bytes=2048,
            disk_request_bytes=1024,
        ),
        network_policy="open",
    )
    # Cache reflects the sandbox's image as in-use.
    assert cache.in_use_count("hello:1") == 1
    # And ensure_present was called (pull recorded for the missing image).
    assert backend.pulled == ["hello:1"]
    await agent.destroy_sandbox(handle)
    assert cache.in_use_count("hello:1") == 0


async def test_node_agent_releases_image_even_when_destroy_raises() -> None:
    """Refcount must drop on destroy failure, otherwise the image stays in_use
    forever and the LRU sweep can't evict it.
    """

    class _DestroyBoom(_FakeImageBackend):
        async def destroy(self, sb: SandboxHandle) -> None:
            raise RuntimeError("destroy boom")

    backend = _DestroyBoom()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="t", backends={backend.name: backend}),
        image_cache=cache,
    )
    handle = await agent.create_sandbox(
        rollout_id="rid-1", backend=backend.name,
        template=TemplateRef(name="hello", image="hello:1"),
        resources=ResourceSpec(
            cpu_request=0.1, cpu_limit=1.0,
            mem_request_bytes=1024, mem_limit_bytes=2048,
            disk_request_bytes=1024,
        ),
        network_policy="open",
    )
    assert cache.in_use_count("hello:1") == 1
    with pytest.raises(RuntimeError, match="destroy boom"):
        await agent.destroy_sandbox(handle)
    assert cache.in_use_count("hello:1") == 0


async def test_node_agent_releases_image_when_create_raises() -> None:
    """If the backend's create raises, the speculative acquire from
    ensure_present must be undone — otherwise we'd leak a refcount."""

    class _CreateBoom(_FakeImageBackend):
        async def create(
            self, template: TemplateRef, resources: ResourceSpec,
            network_policy: NetworkPolicy,
        ) -> SandboxHandle:
            raise RuntimeError("create boom")

    backend = _CreateBoom()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="t", backends={backend.name: backend}),
        image_cache=cache,
    )
    with pytest.raises(RuntimeError, match="create boom"):
        await agent.create_sandbox(
            rollout_id="rid-1", backend=backend.name,
            template=TemplateRef(name="hello", image="hello:1"),
            resources=ResourceSpec(
                cpu_request=0.1, cpu_limit=1.0,
                mem_request_bytes=1024, mem_limit_bytes=2048,
                disk_request_bytes=1024,
            ),
            network_policy="open",
        )
    # Refcount must be 0 — speculative acquire was undone.
    assert cache.in_use_count("hello:1") == 0


# ──────────────────────────────────────────────────────────────────────────────
# Misc parameter validation
# ──────────────────────────────────────────────────────────────────────────────


def test_image_cache_config_defaults() -> None:
    cfg = ImageCacheConfig()
    assert cfg.evict_threshold_bytes == 15 * 1024**3
    assert cfg.evict_target_bytes == 25 * 1024**3
    assert cfg.evict_headroom_slots == 4
    assert cfg.evict_disk_safety_factor == 1.5
    assert cfg.evict_threshold_cap_bytes == 50 * 1024**3
    assert cfg.evict_target_cap_bytes == 75 * 1024**3
    assert cfg.sweep_interval_s == 60.0
    assert cfg.recent_window_s == 30 * 60.0
    assert cfg.pull_concurrency == 2


# ──────────────────────────────────────────────────────────────────────────────
# D16 — tier-ordered eviction
# ──────────────────────────────────────────────────────────────────────────────


def test_default_tier_classifier_label_role_is_authoritative() -> None:
    # The plug-in's Dockerfile is the source of truth: setting
    # ``LABEL org.xrlenv.role=intermediate`` on a benchmark addon
    # image classifies it as the medium "stub_runtime" tier without
    # the platform needing to know any plug-in's tag-naming
    # convention. Same goes for explicit "final" / "base" roles.
    assert default_tier_classifier(
        "swebench-verified-bench/django__django-11099:0.1",
        {"org.xrlenv.owned": "true", "org.xrlenv.role": "intermediate"},
    ) == "stub_runtime"
    assert default_tier_classifier(
        "swebench-verified/django__django-11099:0.1",
        {"org.xrlenv.owned": "true", "org.xrlenv.role": "final"},
    ) == "final"
    assert default_tier_classifier(
        "some-future-base-image:0.1",
        {"org.xrlenv.owned": "true", "org.xrlenv.role": "base"},
    ) == "base"
    # Label wins over name pattern: if a plug-in explicitly declares
    # role=intermediate, that beats the name-based "-base/" fallback.
    assert default_tier_classifier(
        "weird-base/foo:0.1",
        {"org.xrlenv.role": "intermediate"},
    ) == "stub_runtime"
    # Unknown role values fall through to the name-based classifier.
    assert default_tier_classifier(
        "foo-base/bar:0.1",
        {"org.xrlenv.role": "garbage-value"},
    ) == "base"


def test_default_tier_classifier_recognizes_base_repo_without_label() -> None:
    # Back-compat fallback for upstream-base retags that can't carry
    # xrlenv labels (``docker tag`` doesn't attach LABEL directives;
    # the only way to add them would be a one-line FROM build, which
    # would cost an extra layer per instance for no semantic gain).
    # The "-base/" repo convention lets us classify retags correctly
    # without that overhead.
    assert default_tier_classifier(
        "terminal-bench-2-base/fix-git:0.1", {},
    ) == "base"
    assert default_tier_classifier(
        "swebench-base/django__django-12345:0.1", {},
    ) == "base"


def test_default_tier_classifier_defaults_to_final() -> None:
    # Final task tag, anonymous tag, missing tag — all classify "final"
    # because they're cheap to recreate (one RUN layer on top of base).
    # No label, no recognized name pattern → final.
    assert default_tier_classifier("terminal-bench-2/fix-git:0.1", {}) == "final"
    assert default_tier_classifier("xrlenv/hello-shell:0.1", {}) == "final"
    assert default_tier_classifier("nginx", {}) == "final"


def test_default_tier_classifier_strips_digest_and_tag() -> None:
    # The "-base/" marker lives in the repo segment; tag + digest
    # suffixes must not affect the classification.
    assert default_tier_classifier("foo-base/bar@sha256:abc", {}) == "base"
    assert default_tier_classifier("foo-base/bar:0.1@sha256:abc", {}) == "base"
    # "-base" inside the tag (not the repo) does NOT classify as base.
    assert default_tier_classifier("foo/bar:0.1-base", {}) == "final"
    # "-base" elsewhere in the path does not falsely match either.
    assert default_tier_classifier("base/bar:0.1", {}) == "final"


def test_default_tier_classifier_handles_registry_host_port() -> None:
    # M4 follow-up: a host:port colon earlier in the reference must
    # not steal the tag-stripping. Pre-fix the parser stripped at the
    # FIRST colon, leaving repo="localhost" and silently misclassifying
    # expensive base images as cheap final tags for any private/local
    # registry that exposes a port.
    assert default_tier_classifier(
        "localhost:5000/terminal-bench-2-base/fix-git:0.1", {},
    ) == "base"
    assert default_tier_classifier(
        "registry.example.com:5000/swebench-base/django-12345:0.1", {},
    ) == "base"
    # Same registry shape but a final tag — must stay "final".
    assert default_tier_classifier(
        "localhost:5000/terminal-bench-2/fix-git:0.1", {},
    ) == "final"
    # Host:port without a tag.
    assert default_tier_classifier(
        "localhost:5000/foo-base/bar", {},
    ) == "base"
    # Host:port + tag with "-base" only in the tag — must NOT match.
    assert default_tier_classifier(
        "registry.example.com:5000/foo/bar:0.1-base", {},
    ) == "final"
    # Host:port + digest, no tag.
    assert default_tier_classifier(
        "localhost:5000/foo-base/bar@sha256:abc", {},
    ) == "base"


# ──────────────────────────────────────────────────────────────────────────────
# Ownership classifier (B7.6 admin-filter follow-on)
# ──────────────────────────────────────────────────────────────────────────────


def test_default_ownership_classifier_requires_owned_label() -> None:
    # Without ``org.xrlenv.owned=true`` the image is external —
    # historical name conventions don't earn a free pass.
    assert default_ownership_classifier({}) == "external"
    assert default_ownership_classifier({"foo": "bar"}) == "external"
    # Old name shape, no label → still external. Operators must
    # rebuild after the label rollout for it to surface.
    assert default_ownership_classifier({"some.other.label": "x"}) == "external"


def test_default_ownership_classifier_role_dispatch() -> None:
    # ``role=final`` (or missing role with owned=true) → xrlenv_final.
    assert default_ownership_classifier(
        {"org.xrlenv.owned": "true", "org.xrlenv.role": "final"},
    ) == "xrlenv_final"
    # Missing role defaults to final so an unlabeled-role xrlenv
    # image still surfaces under the default-on filter.
    assert default_ownership_classifier(
        {"org.xrlenv.owned": "true"},
    ) == "xrlenv_final"
    # ``role=intermediate`` → xrlenv_intermediate (filtered out by
    # the "hide intermediates" default).
    assert default_ownership_classifier(
        {"org.xrlenv.owned": "true", "org.xrlenv.role": "intermediate"},
    ) == "xrlenv_intermediate"
    # Any future role value falls through to final so the unknown
    # gets surfaced rather than swallowed as external.
    assert default_ownership_classifier(
        {"org.xrlenv.owned": "true", "org.xrlenv.role": "future-tier"},
    ) == "xrlenv_final"


def test_default_ownership_classifier_owned_must_be_true_string() -> None:
    # The label is a string-typed Docker label; only the literal
    # ``"true"`` opts in. Any other value (case-different, empty,
    # ``"yes"``) classifies as external — keeps the contract
    # operator-auditable.
    assert default_ownership_classifier(
        {"org.xrlenv.owned": "True", "org.xrlenv.role": "final"},
    ) == "external"
    assert default_ownership_classifier(
        {"org.xrlenv.owned": "1", "org.xrlenv.role": "final"},
    ) == "external"
    assert default_ownership_classifier(
        {"org.xrlenv.owned": "", "org.xrlenv.role": "final"},
    ) == "external"


async def test_evict_prefers_final_over_base_when_both_cold() -> None:
    # Mixed cold cache: one final tag + one base tag, equal size and
    # equal LRU age. Disk pressure forces one eviction. The final tag
    # must go first because the base is the expensive layer to rebuild.
    final = ImageRecord(name="bench/task:0.1", size_bytes=8 * 1024**3)
    base = ImageRecord(name="bench-base/task:0.1", size_bytes=8 * 1024**3)
    backend = _FakeImageBackend(present=[final, base], free_bytes=2 * 1024**3)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            # Pin headroom to the absolute floors (disable the
            # image-size-adaptive reserve) so this test isolates tier/LRU
            # *ordering*: the target is reached after exactly one eviction.
            evict_headroom_slots=0,
            evict_disk_safety_factor=1.0,
            evict_threshold_bytes=20 * 1024**3,
            # Target reachable after evicting one 8 GB image; second
            # eviction shouldn't trigger.
            evict_target_bytes=10 * 1024**3,
        ),
    )
    # Touch the final tag *more recently* than the base — under the
    # old pure-LRU policy this would have evicted the base. The new
    # tier-first policy still evicts the final first.
    cache.acquire("bench-base/task:0.1")
    cache.release("bench-base/task:0.1")
    await asyncio.sleep(0.001)
    cache.acquire("bench/task:0.1")
    cache.release("bench/task:0.1")
    await cache.ensure_present("bench/new:0.1")
    assert backend.removed == ["bench/task:0.1"]


async def test_evict_falls_back_to_base_when_final_tier_exhausted() -> None:
    # Disk pressure deeper than the final tier can satisfy: the
    # eviction loop spills into the base tier rather than giving up.
    final = ImageRecord(name="bench/task:0.1", size_bytes=4 * 1024**3)
    base = ImageRecord(name="bench-base/task:0.1", size_bytes=8 * 1024**3)
    backend = _FakeImageBackend(present=[final, base], free_bytes=2 * 1024**3)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_bytes=20 * 1024**3,
            # Even after evicting the 4 GB final, free is 6 GB — still
            # below target 12 GB — so the base evicts too.
            evict_target_bytes=12 * 1024**3,
        ),
    )
    await cache.ensure_present("bench/new:0.1")
    assert backend.removed == ["bench/task:0.1", "bench-base/task:0.1"]


async def test_evict_within_tier_still_oldest_first() -> None:
    # Two final tags, equal size; the older LRU one evicts first. This
    # pins that the secondary sort key (LRU within tier) still works.
    a = ImageRecord(name="bench/a:0.1", size_bytes=8 * 1024**3)
    b = ImageRecord(name="bench/b:0.1", size_bytes=8 * 1024**3)
    backend = _FakeImageBackend(present=[a, b], free_bytes=2 * 1024**3)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            # Pin headroom to the floors (see ordering note above) so
            # exactly one image evicts and the LRU tie-break is observable.
            evict_headroom_slots=0,
            evict_disk_safety_factor=1.0,
            evict_threshold_bytes=20 * 1024**3,
            evict_target_bytes=10 * 1024**3,
        ),
    )
    cache.acquire("bench/a:0.1")
    cache.release("bench/a:0.1")
    await asyncio.sleep(0.001)
    cache.acquire("bench/b:0.1")
    cache.release("bench/b:0.1")
    await cache.ensure_present("bench/new:0.1")
    assert backend.removed == ["bench/a:0.1"]


async def test_custom_tier_classifier_overrides_default() -> None:
    # External plug-ins with different tag conventions can pass their
    # own classifier. Here the custom rule classifies "precious/*"
    # tags as base (preserve last) and everything else as final.
    def custom(image: str, labels: dict[str, str]) -> EvictionTier:
        if image.startswith("precious/"):
            return "base"
        return "final"

    precious = ImageRecord(name="precious/keepme:1", size_bytes=8 * 1024**3)
    plain = ImageRecord(name="ordinary/burnme:1", size_bytes=8 * 1024**3)
    backend = _FakeImageBackend(
        present=[precious, plain], free_bytes=2 * 1024**3,
    )
    cache = ImageCacheManager(
        backend=backend,
        tier_classifier=custom,
        config=ImageCacheConfig(
            # Pin headroom to the floors (see ordering note above) so
            # exactly one image evicts and the classifier choice is
            # observable in isolation.
            evict_headroom_slots=0,
            evict_disk_safety_factor=1.0,
            evict_threshold_bytes=20 * 1024**3,
            evict_target_bytes=10 * 1024**3,
        ),
    )
    # Make precious the OLDER image (under pure LRU it would evict
    # first); the custom classifier promotes it to base so it survives
    # the eviction pass despite being older. The newer "ordinary"
    # tag — same age math but classified "final" — gets evicted
    # instead. This is the test that the classifier is load-bearing,
    # not just shadowing what LRU would have done anyway.
    cache.acquire("precious/keepme:1")
    cache.release("precious/keepme:1")
    await asyncio.sleep(0.001)
    cache.acquire("ordinary/burnme:1")
    cache.release("ordinary/burnme:1")
    await cache.ensure_present("new/img:1")
    assert backend.removed == ["ordinary/burnme:1"]


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 — bounded pull retry
# ──────────────────────────────────────────────────────────────────────────────


class _FailNThenSucceedBackend(_FakeImageBackend):
    """Backend whose ``pull_image`` fails its first ``fail_first``
    attempts, then succeeds — models a flaky registry / auth endpoint."""

    def __init__(self, *, fail_first: int, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._fail_first = fail_first
        self.attempts = 0

    async def pull_image(self, image: str, *, timeout_s: float = 600.0) -> None:
        self.attempts += 1
        if self.attempts <= self._fail_first:
            raise RuntimeError(f"transient pull failure #{self.attempts}")
        await super().pull_image(image, timeout_s=timeout_s)


@pytest.fixture
def _instant_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neuter the retry backoff so the tests don't burn wall-clock."""
    async def _noop(_s: float) -> None:
        return None
    monkeypatch.setattr(
        "xrlenv.node.image_cache.asyncio.sleep", _noop,
    )


@pytest.mark.asyncio
async def test_pull_retries_transient_failure_then_succeeds(
    _instant_sleep: None,
) -> None:
    """A pull that fails transiently is retried within the deadline;
    once an attempt succeeds the image is present and no error
    surfaces (issue #18)."""
    backend = _FailNThenSucceedBackend(fail_first=2)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(pull_max_attempts=3),
    )

    await cache.ensure_present("flaky/img:1", deadline_s=60.0)

    assert backend.attempts == 3  # 2 failures + 1 success
    assert "flaky/img:1" in backend.pulled


@pytest.mark.asyncio
async def test_pull_gives_up_after_max_attempts(
    _instant_sleep: None,
) -> None:
    """A persistently-failing pull raises after exactly
    ``pull_max_attempts`` tries — surfacing the last error."""
    backend = _FailNThenSucceedBackend(fail_first=99)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(pull_max_attempts=3),
    )

    with pytest.raises(RuntimeError, match="transient pull failure"):
        await cache.ensure_present("dead/img:1", deadline_s=60.0)

    assert backend.attempts == 3


@pytest.mark.asyncio
async def test_pull_max_attempts_1_disables_retry(
    _instant_sleep: None,
) -> None:
    """``pull_max_attempts=1`` is the opt-out — one attempt, no retry."""
    backend = _FailNThenSucceedBackend(fail_first=99)
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(pull_max_attempts=1),
    )

    with pytest.raises(RuntimeError):
        await cache.ensure_present("dead/img:1", deadline_s=60.0)

    assert backend.attempts == 1


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive (AIMD) pull concurrency
# ──────────────────────────────────────────────────────────────────────────────


class _ConcurrencyTrackingBackend(_FakeImageBackend):
    """Records the peak number of ``pull_image`` calls in flight at once."""

    def __init__(self, *, free_bytes: int = 100 * 1024**3) -> None:
        super().__init__(free_bytes=free_bytes)
        self.pull_delay_s = 0.05
        self._inflight = 0
        self.max_inflight = 0

    async def pull_image(self, image: str, *, timeout_s: float = 600.0) -> None:
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            await asyncio.sleep(self.pull_delay_s)
            self.pulled.append(image)
            self._present[image] = ImageRecord(name=image, size_bytes=1024 * 1024)
            self._free -= 1024 * 1024
        finally:
            self._inflight -= 1


# --- AdjustableSemaphore ------------------------------------------------------


async def test_adjustable_semaphore_bounds_then_resizes_up() -> None:
    from xrlenv.node.adaptive_pull import AdjustableSemaphore

    sem = AdjustableSemaphore(2)
    await sem.acquire()
    await sem.acquire()
    assert sem.in_flight == 2

    third = asyncio.create_task(sem.acquire())  # blocks at limit
    await asyncio.sleep(0.02)
    assert not third.done()

    await sem.set_limit(3)  # opens a slot → the waiter proceeds
    await asyncio.wait_for(third, timeout=1.0)
    assert sem.in_flight == 3


async def test_adjustable_semaphore_lowering_drains_naturally() -> None:
    from xrlenv.node.adaptive_pull import AdjustableSemaphore

    sem = AdjustableSemaphore(4)
    await sem.acquire()
    await sem.acquire()
    await sem.set_limit(1)  # below in_flight(2) — holders are NOT cancelled
    assert sem.in_flight == 2

    waiter = asyncio.create_task(sem.acquire())
    await asyncio.sleep(0.02)
    assert not waiter.done()
    await sem.release()  # 2→1, still == limit → still blocked
    await asyncio.sleep(0.02)
    assert not waiter.done()
    await sem.release()  # 1→0 → slot frees
    await asyncio.wait_for(waiter, timeout=1.0)
    assert sem.in_flight == 1


# --- PullAimdController --------------------------------------------------------


def test_pull_aimd_controller_multiplicative_decrease_to_floor() -> None:
    from xrlenv.node.adaptive_pull import PullAimdController

    c = PullAimdController(floor=2, ceiling=64, initial=16)
    assert [c.observe(busy=True) for _ in range(4)] == [8, 4, 2, 2]  # clamps


def test_pull_aimd_controller_additive_increase_to_ceiling() -> None:
    from xrlenv.node.adaptive_pull import PullAimdController

    c = PullAimdController(floor=2, ceiling=20, initial=16, additive_step=2)
    assert [c.observe(busy=False) for _ in range(3)] == [18, 20, 20]  # clamps


def test_pull_aimd_controller_clamps_initial_into_range() -> None:
    from xrlenv.node.adaptive_pull import PullAimdController

    assert PullAimdController(floor=2, ceiling=64, initial=1).limit == 2
    assert PullAimdController(floor=2, ceiling=64, initial=999).limit == 64
    assert PullAimdController(floor=2, ceiling=64, initial=16).limit == 16


def test_pull_aimd_controller_set_ceiling_clamps_limit_and_floor() -> None:
    from xrlenv.node.adaptive_pull import PullAimdController

    c = PullAimdController(floor=2, ceiling=64, initial=32)
    assert c.limit == 32
    # Lowering the ceiling below the live limit clamps the limit down too
    # (so a disk-bounded ceiling takes effect immediately, not just on the
    # next tick).
    c.set_ceiling(8)
    assert (c.ceiling, c.limit) == (8, 8)
    # Never below the floor, even when asked for less.
    c.set_ceiling(1)
    assert (c.ceiling, c.limit) == (2, 2)
    # Raising the ceiling again leaves the (already-lower) limit alone —
    # AIMD ramps it back up over subsequent calm ticks, not in one jump.
    c.set_ceiling(64)
    assert (c.ceiling, c.limit) == (64, 2)


# --- Manager integration ------------------------------------------------------


def test_manager_initial_pull_limit_is_clamped_initial() -> None:
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=16,
        ),
    )
    assert cache.pull_concurrency_limit == 16


async def test_pull_limit_caps_concurrent_pulls() -> None:
    """All pulls share the single adaptive limiter at its current bound."""
    backend = _ConcurrencyTrackingBackend()
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=5,
        ),
    )
    await asyncio.gather(*(cache.ensure_present(f"img:{i}") for i in range(10)))
    assert backend.max_inflight == 5  # never exceeds the current limit


async def test_aimd_loop_shrinks_pull_limit_when_busy() -> None:
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=16, pull_aimd_interval_s=0.01,
            pull_busy_threshold=0,
        ),
    )
    cache.acquire("running:1")  # in_use=1 > threshold(0) → busy
    task = asyncio.create_task(cache.run_pull_aimd_loop())
    try:
        await asyncio.sleep(0.1)  # ~10 ticks: 16→8→4→2 then stable
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert cache.pull_concurrency_limit == 2


async def test_aimd_loop_grows_pull_limit_when_idle() -> None:
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=16, pull_aimd_interval_s=0.01,
            pull_aimd_additive_step=2, pull_busy_threshold=0,
        ),
    )
    task = asyncio.create_task(cache.run_pull_aimd_loop())  # idle → grows
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert cache.pull_concurrency_limit > 16


async def test_aimd_loop_disk_bounded_ceiling_caps_growth_when_tight() -> None:
    # An idle node would normally ramp the pull limit toward the static
    # ceiling (64), but a tight disk must cap it so a pull burst can't
    # overrun the small eviction reserve: disk-bounded ceiling =
    # free / (largest x safety) = 30 / (10 x 1.5) = 2. The limit clamps
    # to 2 despite the node being idle.
    backend = _FakeImageBackend(
        present=[ImageRecord(name="big:1", size_bytes=10 * 1024**3)],
        free_bytes=30 * 1024**3,
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=16, pull_aimd_interval_s=0.01,
            pull_busy_threshold=0, evict_disk_safety_factor=1.5,
        ),
    )
    await cache._refresh_image_stats()  # prime the largest-image size
    task = asyncio.create_task(cache.run_pull_aimd_loop())
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert cache.pull_concurrency_limit == 2


class _FakeIoSampler:
    """Duck-typed DiskIoSampler stand-in: fixed saturation verdict so the
    AIMD-integration tests don't depend on /sys. The sampler's own
    math/hysteresis is covered in test_disk_io.py."""

    def __init__(self, *, saturated: bool, util: float | None = None) -> None:
        self._sat = saturated
        self.last_utilization = util

    def saturated(self) -> bool:
        return self._sat


async def test_aimd_loop_backs_off_when_disk_io_saturated() -> None:
    # #1: an idle node (no in-use containers) would normally RAMP the pull
    # limit, but a saturated data-root volume must force back-off so cold
    # pulls don't peg the EBS volume and wedge containerd's teardown path.
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=16, pull_aimd_interval_s=0.01,
            pull_busy_threshold=0,
        ),
        disk_io_sampler=_FakeIoSampler(saturated=True, util=0.99),
    )
    task = asyncio.create_task(cache.run_pull_aimd_loop())
    try:
        await asyncio.sleep(0.1)  # ~10 ticks: 16→8→4→2
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert cache.pull_concurrency_limit == 2  # floor, despite being idle


async def test_aimd_loop_idle_grows_when_disk_io_not_saturated() -> None:
    # The I/O sampler must not interfere when the volume is calm: an idle
    # node still ramps toward the ceiling.
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=16, pull_aimd_interval_s=0.01,
            pull_busy_threshold=0,
        ),
        disk_io_sampler=_FakeIoSampler(saturated=False, util=0.10),
    )
    task = asyncio.create_task(cache.run_pull_aimd_loop())
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert cache.pull_concurrency_limit > 16


async def test_aimd_loop_ignores_io_when_throttle_disabled() -> None:
    # io_throttle_enabled=False ignores the sampler entirely (escape hatch).
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(
            pull_concurrency=2, pull_concurrency_ceiling=64,
            pull_concurrency_initial=16, pull_aimd_interval_s=0.01,
            pull_busy_threshold=0, io_throttle_enabled=False,
        ),
        disk_io_sampler=_FakeIoSampler(saturated=True, util=0.99),
    )
    task = asyncio.create_task(cache.run_pull_aimd_loop())
    try:
        await asyncio.sleep(0.1)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert cache.pull_concurrency_limit > 16  # grew — sampler ignored


async def test_aimd_loop_second_call_is_idempotent() -> None:
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),
        config=ImageCacheConfig(pull_aimd_interval_s=0.01),
    )
    t1 = asyncio.create_task(cache.run_pull_aimd_loop())
    await asyncio.sleep(0.02)
    # A second call returns immediately (guarded) rather than starting
    # a competing loop.
    await asyncio.wait_for(cache.run_pull_aimd_loop(), timeout=1.0)
    t1.cancel()
    with suppress(asyncio.CancelledError):
        await t1


def test_pull_concurrency_knobs_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor / ceiling / initial env knobs flow into ImageCacheConfig."""
    from xrlenv.node.cli import _image_cache_config_from_env

    monkeypatch.setenv("XRLENV_PULL_CONCURRENCY", "3")
    monkeypatch.setenv("XRLENV_PULL_CONCURRENCY_CEILING", "48")
    monkeypatch.setenv("XRLENV_PULL_CONCURRENCY_INITIAL", "12")
    cfg = _image_cache_config_from_env()
    assert isinstance(cfg, ImageCacheConfig)
    assert cfg.pull_concurrency == 3
    assert cfg.pull_concurrency_ceiling == 48
    assert cfg.pull_concurrency_initial == 12


# ── Eviction headroom caps (large-disk over-eviction fix) ───────────────────


async def test_capped_headroom_limits_eviction_on_large_disk() -> None:
    """Regression: the capped adaptive headroom must NOT behave like the
    old disk-fraction (0.30 x 500 GiB = 150 GiB) and drain the whole
    cache. With a 20 GiB largest image the START/STOP headroom clamps to
    the 50/75 GiB caps, so eviction stops well before the cache empties."""
    images = [
        ImageRecord(name=f"cold:{i}", size_bytes=20 * 1024**3)
        for i in range(6)
    ]
    backend = _FakeImageBackend(
        present=images, free_bytes=30 * 1024**3, total_bytes=500 * 1024**3,
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_cap_bytes=50 * 1024**3,
            evict_target_cap_bytes=75 * 1024**3,
        ),
    )
    await cache._refresh_image_stats()
    # START = min(4x20x1.5, 50) = 50 GiB; STOP = min(5x20x1.5, 75) = 75
    # GiB. From 30 GiB free, three 20 GiB evictions reach 90 GiB ≥ 75 and
    # stop — the other three survive. The old 0.30 fraction (150 GiB)
    # would have drained all six.
    await cache._evict_if_needed()
    assert len(backend.removed) == 3
    assert backend._free >= 75 * 1024**3


async def test_adaptive_headroom_floor_dominates_for_small_images() -> None:
    """Small-disk / laptop protection: a tiny-image workload reserves the
    absolute floor, never a smaller adaptive value — the headroom must
    not collapse below the floor on a small node."""
    backend = _FakeImageBackend(
        present=[ImageRecord(name="tiny:1", size_bytes=1 * 1024**3)],
    )
    cache = ImageCacheManager(
        backend=backend,
        config=ImageCacheConfig(
            evict_threshold_bytes=15 * 1024**3,
            evict_target_bytes=25 * 1024**3,
        ),
    )
    await cache._refresh_image_stats()
    # reserve = 4x1x1.5 = 6 GiB (START) / 5x1x1.5 = 7.5 GiB (STOP); both
    # below the floors, so the absolute floors win.
    assert cache._effective_threshold() == 15 * 1024**3
    assert cache._effective_target() == 25 * 1024**3


def test_evict_cap_knobs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``XRLENV_EVICT_{THRESHOLD,TARGET}_CAP_GB`` flow into the config as
    bytes."""
    from xrlenv.node.cli import _image_cache_config_from_env

    monkeypatch.setenv("XRLENV_EVICT_THRESHOLD_CAP_GB", "30")
    monkeypatch.setenv("XRLENV_EVICT_TARGET_CAP_GB", "40")
    cfg = _image_cache_config_from_env()
    assert isinstance(cfg, ImageCacheConfig)
    assert cfg.evict_threshold_cap_bytes == 30 * 1024**3
    assert cfg.evict_target_cap_bytes == 40 * 1024**3


# ── evict_ref (xrlenv image evict) ──────────────────────────────────────────


async def test_evict_ref_matches_registry_qualified_tag() -> None:
    """A node holds the image under its registry-qualified tag (post-pull);
    evicting by the bare plan ref must still match + remove it, and leave
    unrelated images alone. The mutable-tag staleness escape hatch."""
    qualified = "node-host:5011/wai/substrate:1ca77813"
    backend = _FakeImageBackend(present=[
        ImageRecord(name=qualified, size_bytes=1_000_000_000),
        ImageRecord(name="node-host:5011/other/img:9", size_bytes=42),
    ])
    cache = ImageCacheManager(backend=backend)

    outcome = await cache.evict_ref("wai/substrate:1ca77813")

    assert outcome.status == "evicted"
    assert outcome.reclaimed_bytes == 1_000_000_000
    assert outcome.removed == (qualified,)
    assert backend.removed == [qualified]
    assert "node-host:5011/other/img:9" in backend._present


async def test_evict_ref_absent_when_no_match() -> None:
    backend = _FakeImageBackend(present=[
        ImageRecord(name="some/other:1", size_bytes=10),
    ])
    cache = ImageCacheManager(backend=backend)

    outcome = await cache.evict_ref("wai/substrate:1ca77813")

    assert outcome.status == "absent"
    assert outcome.reclaimed_bytes == 0
    assert backend.removed == []


async def test_evict_ref_skips_in_use_unless_force() -> None:
    """An in-use image is skipped (status=in_use) so a live rollout is not
    disrupted — until --force, which removes it."""
    ref = "wai/substrate:1ca77813"
    backend = _FakeImageBackend(present=[
        ImageRecord(name=ref, size_bytes=500),
    ])
    cache = ImageCacheManager(backend=backend)
    cache.acquire(ref)  # bump in-use refcount

    skipped = await cache.evict_ref(ref)
    assert skipped.status == "in_use"
    assert backend.removed == []
    assert ref in backend._present

    forced = await cache.evict_ref(ref, force=True)
    assert forced.status == "evicted"
    assert forced.reclaimed_bytes == 500
    assert backend.removed == [ref]


async def test_evict_ref_skips_pinned_unless_force() -> None:
    ref = "wai/substrate:1ca77813"
    backend = _FakeImageBackend(present=[
        ImageRecord(name=ref, size_bytes=500),
    ])
    cache = ImageCacheManager(backend=backend, pins={ref})

    skipped = await cache.evict_ref(ref)
    assert skipped.status == "in_use"  # pinned counts as blocked
    assert backend.removed == []

    forced = await cache.evict_ref(ref, force=True)
    assert forced.status == "evicted"
    # Force also drops the pin so the cache doesn't show a ghost.
    assert ref not in cache.pins


async def test_evict_ref_backend_in_use_is_skipped_not_failed() -> None:
    """If the daemon refuses removal with ImageInUse (409 — held by a
    non-xrlenv container), evict reports in_use, not failed."""
    ref = "wai/substrate:1ca77813"

    class _RefusingBackend(_FakeImageBackend):
        async def remove_image(self, image: str, *, force: bool = False) -> None:
            raise ImageInUse(image)

    backend = _RefusingBackend(present=[
        ImageRecord(name=ref, size_bytes=500),
    ])
    cache = ImageCacheManager(backend=backend)

    outcome = await cache.evict_ref(ref)
    assert outcome.status == "in_use"
    assert outcome.reclaimed_bytes == 0
