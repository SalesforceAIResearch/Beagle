"""WS2 — RawContainerManager disk-usage enumeration + image refcount.

Two behaviours the disk guard depends on:

  * ``list_disk_usage`` surfaces each running raw container's writable
    layer (``SizeRw``) + owning rollout — the guard's offender list;
  * acquire/destroy now refcount the container's image in the cache, so
    a running container's image is no longer misclassified "cold" (the
    futile per-tick eviction log-spam) and the guard's
    ``evictable_image_bytes`` doesn't over-count in-use images.
"""

from __future__ import annotations

from typing import Any

import pytest
from xrlenv.node.raw_container import RawContainerManager


class _NotFound(Exception):
    pass


class _FakeContainer:
    def __init__(
        self, cid: str, name: str, image: str, labels: dict[str, str],
    ) -> None:
        self.id = cid
        self.name = name
        self.image_ref = image
        self.labels = labels
        self.removed = False
        self.size_rw = 0  # writable-layer bytes the daemon reports under size=True
        self.attrs: dict[str, Any] = {
            "HostConfig": {"CpusetCpus": ""},
            "Config": {"Labels": dict(labels), "Image": image},
        }

    def remove(self, *, force: bool = False) -> None:
        self.removed = True


class _FakeContainers:
    def __init__(self) -> None:
        self._n = 0
        self._reg: dict[str, _FakeContainer] = {}
        self.list_calls: list[dict[str, Any]] = []

    def list(
        self, *, filters: dict[str, str] | None = None,
        all: bool = False, sparse: bool = False,
    ) -> list[_FakeContainer]:
        # NOTE: real docker-py ContainerCollection.list() has NO ``size``
        # param — it raises TypeError if passed one. This fake omits it
        # deliberately so any code that reaches for size via the
        # high-level API (the bug a live smoke caught) fails here too.
        self.list_calls.append({"all": all, "sparse": sparse})
        return [
            c for c in self._reg.values()
            if not c.removed
            and c.labels.get("xrlenv.session_kind") == "raw"
        ]

    def run(
        self, *, image: str, detach: bool, labels: dict[str, str],
        command: list[str] | None = None, name: str | None = None,
        **_extra: Any,
    ) -> _FakeContainer:
        self._n += 1
        cid = f"c-{self._n:04d}"
        c = _FakeContainer(cid, name or cid, image, labels)
        self._reg[cid] = c
        return c

    def get(self, cid: str) -> _FakeContainer:
        c = self._reg.get(cid)
        if c is None or c.removed:
            raise _NotFound(cid)
        return c


class _FakeImages:
    def get(self, image: str) -> Any:
        return object()


class _FakeApi:
    """Low-level docker-py APIClient stand-in. ``containers(size=True)``
    returns the raw ``GET /containers/json?size=1`` dicts (Id / Image /
    Labels / SizeRw) — this is where SizeRw actually comes from; the
    high-level ContainerCollection.list() can't do it."""

    def __init__(self, containers: _FakeContainers) -> None:
        self._containers = containers

    def containers(
        self, *, filters: dict[str, str] | None = None,
        all: bool = False, size: bool = False,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in self._containers._reg.values():
            if c.removed:
                continue
            if c.labels.get("xrlenv.session_kind") != "raw":
                continue
            out.append({
                "Id": c.id,
                "Image": c.image_ref,
                "Labels": dict(c.labels),
                "SizeRw": c.size_rw if size else None,
            })
        return out


class _FakeClient:
    def __init__(self) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers()
        self.api = _FakeApi(self.containers)


class _FakeCache:
    """Records image-refcount acquire/release and serves ensure_present."""

    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.released: list[str] = []

    async def ensure_present(
        self, image: str, *, deadline_s: float | None = None,
    ) -> None:
        return None

    def acquire(self, image: str) -> None:
        self.acquired.append(image)

    def release(self, image: str) -> None:
        self.released.append(image)


@pytest.mark.asyncio
async def test_list_disk_usage_reports_size_rollout_and_image() -> None:
    client = _FakeClient()
    mgr = RawContainerManager(docker_client=client)

    rec_a = await mgr.acquire(rollout_id="roll-A", image="img:1")
    rec_b = await mgr.acquire(rollout_id="roll-B", image="img:2")
    # Writable-layer bytes the daemon reports via the low-level
    # api.containers(size=True) — NOT the high-level list().
    client.containers._reg[rec_a.container_id].size_rw = 510 * 1024**3
    client.containers._reg[rec_b.container_id].size_rw = 2 * 1024**3

    usage = await mgr.list_disk_usage()
    by_id = {u.container_id: u for u in usage}

    assert by_id[rec_a.container_id].size_rw_bytes == 510 * 1024**3
    assert by_id[rec_a.container_id].rollout_id == "roll-A"
    assert by_id[rec_a.container_id].image == "img:1"
    assert by_id[rec_b.container_id].size_rw_bytes == 2 * 1024**3
    assert by_id[rec_b.container_id].rollout_id == "roll-B"


@pytest.mark.asyncio
async def test_acquire_refcounts_image_then_destroy_releases() -> None:
    client = _FakeClient()
    cache = _FakeCache()
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    rec = await mgr.acquire(rollout_id="roll-A", image="img:1")
    assert cache.acquired == ["img:1"]
    assert cache.released == []

    await mgr.destroy(rollout_id="roll-A", container_id=rec.container_id)
    assert cache.released == ["img:1"]


@pytest.mark.asyncio
async def test_destroy_already_gone_still_releases_image() -> None:
    """The benign-race path (docker already removed the container) must
    still release the image refcount so it never leaks."""
    client = _FakeClient()
    cache = _FakeCache()
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    rec = await mgr.acquire(rollout_id="roll-A", image="img:1")
    # Container vanished out from under us before destroy.
    client.containers._reg[rec.container_id].removed = True

    await mgr.destroy(rollout_id="roll-A", container_id=rec.container_id)
    assert cache.released == ["img:1"]


@pytest.mark.asyncio
async def test_force_destroy_releases_tracked_image() -> None:
    client = _FakeClient()
    cache = _FakeCache()
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    rec = await mgr.acquire(rollout_id="roll-A", image="img:1")
    await mgr.force_destroy(container_id=rec.container_id)
    assert cache.released == ["img:1"]


# ── Audit P3 — node-autonomous reap-reason log ──────────────────────────────


def test_note_disk_reaped_is_bounded_lru() -> None:
    mgr = RawContainerManager(docker_client=_FakeClient())
    mgr._disk_reaped_max = 3
    for i in range(5):
        mgr.note_disk_reaped(f"r{i}", f"reason{i}")
    reasons = mgr.disk_reaped_reasons()
    # Oldest two evicted; newest three retained in insertion order.
    assert list(reasons) == ["r2", "r3", "r4"]
    assert reasons["r4"] == "reason4"


def test_note_disk_reaped_ignores_empty_rollout() -> None:
    mgr = RawContainerManager(docker_client=_FakeClient())
    mgr.note_disk_reaped("", "no rollout id")
    assert mgr.disk_reaped_reasons() == {}


def test_disk_reaped_reasons_snapshot_is_a_copy() -> None:
    mgr = RawContainerManager(docker_client=_FakeClient())
    mgr.note_disk_reaped("r1", "boom")
    snap = mgr.disk_reaped_reasons()
    snap["r1"] = "mutated"
    # Mutating the snapshot must not corrupt the manager's log.
    assert mgr.disk_reaped_reasons()["r1"] == "boom"
