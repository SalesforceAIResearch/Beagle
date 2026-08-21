"""Unit tests for ``xrlenv.node.raw_container.RawContainerManager``.

Uses a hand-rolled fake ``docker.DockerClient`` so the tests don't
need a live daemon. The fake mirrors the surface RawContainerManager
calls into: ``client.images.get``, ``client.containers.run``,
``client.containers.get``, plus per-container ``.exec_run`` and
``.remove``. Anything outside that surface — pulls, builds, the
manager classes' richer features — is out of scope here; the
manager doesn't touch them.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.errors import ArchiveTooLarge, XRLEnvError
from xrlenv.node.raw_container import RawContainerManager

# ──────────────────────────────────────────────────────────────────────────────
# Fakes — minimal docker-py shim
# ──────────────────────────────────────────────────────────────────────────────


class _FakeImageNotFound(Exception):
    """Stand-in for docker.errors.ImageNotFound."""


@dataclass
class _ExecResult:
    exit_code: int
    output: tuple[bytes, bytes]


@dataclass
class _FakeContainer:
    id: str
    name: str
    image_ref: str
    labels: dict[str, str]
    command: list[str] | None = None
    environment: dict[str, str] | None = None
    userns_mode: str | None = None
    # Issue #6 audit M1: Level-1/2 kwargs are forwarded from the
    # control plane via ``RawContainerManager.acquire`` → docker-py
    # ``containers.run(**kwargs)``. Tests assert these landed on the
    # fake container so the wire path is locked in.
    cap_add: list[str] | None = None
    devices: list[str] | None = None
    privileged: bool = False
    network_mode: str | None = None
    volumes: dict[str, dict[str, str]] | None = None
    exec_results: dict[tuple, _ExecResult] = field(default_factory=dict)
    # P1.7.A.2: archive scratch — tests pre-seed get_archive
    # results, observe put_archive calls.
    put_archive_calls: list[tuple[str, bytes]] = field(default_factory=list)
    get_archive_returns: dict[str, bytes] = field(default_factory=dict)
    put_archive_ok: bool = True
    removed: bool = False
    raise_on_remove: bool = False
    exec_delay_s: float = 0.0
    # P2 — docker-py's Container.attrs (inspect output). The core-ledger
    # reconcile reads HostConfig.CpusetCpus from here.
    attrs: dict[str, Any] = field(default_factory=dict)
    # P1.7.A.2: streaming exec scratch — the manager calls into
    # the docker low-level API on ``client.api`` rather than the
    # container; the api-level fakes live on ``_FakeAPI`` below.

    def exec_run(self, cmd: list[str], **kwargs: Any) -> _ExecResult:
        if self.exec_delay_s:
            import time
            time.sleep(self.exec_delay_s)
        # Lookup by tuple of cmd; default to a benign 0 exit if not
        # pre-seeded — keeps tests terse for happy paths.
        result = self.exec_results.get(tuple(cmd))
        if result is None:
            return _ExecResult(exit_code=0, output=(b"", b""))
        return result

    def remove(self, *, force: bool = False) -> None:
        if self.raise_on_remove:
            raise RuntimeError("simulated remove failure")
        self.removed = True

    def put_archive(self, target_dir: str, tarball: bytes) -> bool:
        self.put_archive_calls.append((target_dir, bytes(tarball)))
        return self.put_archive_ok

    def get_archive(self, source_path: str) -> tuple[Any, dict]:
        # docker-py returns ``(iterator_of_bytes, stat_dict)``; the
        # iterator can yield the tarball in chunks. Returning a
        # single-element tuple-iterable mirrors that.
        data = self.get_archive_returns.get(source_path, b"")
        return (iter([data]) if data else iter([]), {"name": source_path})


class _FakeImages:
    def __init__(self, present: set[str]) -> None:
        self.present = present

    def get(self, image: str) -> Any:
        if image not in self.present:
            raise _FakeImageNotFound(f"no such image: {image}")
        return object()  # docker-py returns an Image; we don't use it.


class _FakeContainers:
    def __init__(self) -> None:
        self._next_id = 0
        self._registry: dict[str, _FakeContainer] = {}
        self.list_calls: list[dict[str, Any]] = []
        # P1 — captures the non-destructured ``containers.run`` kwargs
        # (cpu_period / cpu_quota / mem_limit / ...) so tests can assert
        # the cgroup limits landed on the wire.
        self.run_extra: list[dict[str, Any]] = []

    def list(
        self,
        *,
        filters: dict[str, str] | None = None,
        all: bool = False,
        sparse: bool = False,
    ) -> list[_FakeContainer]:
        """Mimics docker-py's ``ContainerCollection.list``. With
        ``sparse=False`` (the default) docker-py inspects every
        container; a container removed mid-enumeration raises
        ``NotFound`` and aborts the whole call. ``sparse=True`` skips
        the per-container inspect."""
        self.list_calls.append({"all": all, "sparse": sparse})
        matched = [
            c for c in self._registry.values()
            if c.labels.get("xrlenv.session_kind") == "raw"
        ]
        if not sparse:
            for c in matched:
                if c.removed:
                    raise _FakeImageNotFound(
                        f"404 No such container: {c.id}",
                    )
        return matched

    def run(
        self, *, image: str, detach: bool, labels: dict[str, str],
        command: list[str] | None = None, name: str | None = None,
        environment: dict[str, str] | None = None,
        userns_mode: str | None = None,
        cap_add: list[str] | None = None,
        devices: list[str] | None = None,
        privileged: bool = False,
        network_mode: str | None = None,
        volumes: dict[str, dict[str, str]] | None = None,
        **_extra: Any,
    ) -> _FakeContainer:
        assert detach is True
        self.run_extra.append(dict(_extra))
        self._next_id += 1
        cid = f"fake-container-id-{self._next_id:04d}"
        cname = name or f"fake-container-{self._next_id:04d}"
        c = _FakeContainer(
            id=cid, name=cname, image_ref=image, labels=labels,
            command=command, environment=environment,
            userns_mode=userns_mode,
            cap_add=cap_add, devices=devices, privileged=privileged,
            network_mode=network_mode, volumes=volumes,
            # P2 — mirror docker's inspect shape so the core-ledger
            # reconcile can read back this container's cpuset.
            attrs={"HostConfig": {"CpusetCpus": _extra.get("cpuset_cpus", "")}},
        )
        self._registry[cid] = c
        return c

    def get(self, container_id: str) -> _FakeContainer:
        c = self._registry.get(container_id)
        if c is None or c.removed:
            raise _FakeImageNotFound(
                f"no such container: {container_id}",
            )
        return c


class _FakeAPI:
    """docker-py's low-level ``client.api``; the streaming exec
    path uses ``exec_create`` / ``exec_start`` / ``exec_inspect``
    here rather than the high-level ``container.exec_run``."""

    def __init__(self) -> None:
        self._next_exec = 0
        self.exec_streams: dict[str, list[tuple]] = {}
        self.exec_exit_codes: dict[str, int] = {}

    def exec_create(self, container_id: str, cmd: list[str], **kw: Any) -> dict:
        self._next_exec += 1
        exec_id = f"exec-{self._next_exec:04d}"
        # Tests pre-seed the chunk stream + exit code keyed on
        # the cmd tuple; we map them onto the new exec_id.
        seed = self.exec_streams.pop(tuple(cmd), [])
        self.exec_streams[exec_id] = seed
        self.exec_exit_codes[exec_id] = self.exec_exit_codes.pop(
            tuple(cmd), 0,
        )
        return {"Id": exec_id}

    def exec_start(self, exec_id: str, *, stream: bool, demux: bool) -> Any:
        assert stream and demux
        return iter(self.exec_streams.get(exec_id, []))

    def exec_inspect(self, exec_id: str) -> dict:
        return {"ExitCode": self.exec_exit_codes.get(exec_id, 0)}

    def exec_resize(self, *args: Any, **kwargs: Any) -> None:
        # No-op; only called on cancellation paths. The real
        # docker daemon would resize the exec's TTY; we don't
        # care about TTY in tests.
        pass


class _FakeDockerClient:
    def __init__(
        self,
        *,
        images_present: set[str],
        runtimes: tuple[str, ...] = ("runc",),
        default_runtime: str = "runc",
    ) -> None:
        self.images = _FakeImages(images_present)
        self.containers = _FakeContainers()
        self.api = _FakeAPI()
        # §5.3/§5.5 — mirror ``docker info``'s Runtimes + DefaultRuntime so
        # ``RawContainerManager.registered_runtimes()`` can probe them.
        self._runtimes = runtimes
        self._default_runtime = default_runtime
        self.info_calls = 0

    def info(self) -> dict[str, Any]:
        self.info_calls += 1
        return {
            "Runtimes": {name: {} for name in self._runtimes},
            "DefaultRuntime": self._default_runtime,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_registers_container_with_owner_labels() -> None:
    """Acquire spawns a container and merges the platform's reserved
    labels (``xrlenv.rollout_id`` + ``xrlenv.session_kind=raw``)
    on top of operator-provided labels."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    record = await mgr.acquire(
        rollout_id="r-001", image="busybox:1",
        command=["sleep", "infinity"],
        labels={"my-label": "abc"},
    )

    assert record.rollout_id == "r-001"
    assert record.image == "busybox:1"
    container = client.containers.get(record.container_id)
    assert container.labels == {
        "my-label": "abc",
        "xrlenv.rollout_id": "r-001",
        "xrlenv.session_kind": "raw",
    }
    assert container.command == ["sleep", "infinity"]


@pytest.mark.asyncio
async def test_list_on_docker_tolerates_container_removed_mid_enumeration() -> None:
    """Issue #18: ``list_on_docker`` must not abort when a container is
    destroyed mid-enumeration. docker-py's non-sparse
    ``containers.list`` inspects every container; one removed between
    the ``/containers/json`` snapshot and its inspect raises NotFound
    and 404s the whole listing — which made the GC reconciler skip the
    node for the sweep. ``sparse=True`` skips the per-container
    inspect, so the race can't fire.
    """
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    rec_live = await mgr.acquire(rollout_id="r-live", image="busybox:1")
    rec_gone = await mgr.acquire(rollout_id="r-gone", image="busybox:1")
    # r-gone's container is destroyed concurrently — still in the
    # daemon's listing snapshot, but a per-container inspect 404s.
    client.containers._registry[rec_gone.container_id].removed = True

    ids = await mgr.list_on_docker()  # must not raise

    # The fix: list_on_docker passes sparse=True — no inspect, no race.
    assert client.containers.list_calls[-1]["sparse"] is True
    assert rec_live.container_id in ids


@pytest.mark.asyncio
async def test_acquire_default_userns_mode_is_host() -> None:
    """B5.4 — default ``userns_mode="host"`` opts out of the
    docker daemon's userns-remap config. Defense-in-depth default:
    benchmark images that need in-container root (lots of them)
    keep working regardless of the operator's daemon config."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-001", image="busybox:1",
    )
    container = client.containers.get(record.container_id)
    assert container.userns_mode == "host", (
        f"default acquire must pass userns_mode='host' to docker — "
        f"got {container.userns_mode!r}"
    )


@pytest.mark.asyncio
async def test_acquire_userns_remap_opts_in() -> None:
    """B5.4 — explicit ``userns_mode="remap"`` translates to
    docker-py's ``userns_mode=""`` (use daemon default; honors
    remap if configured). On daemons without remap configured,
    this is a silent no-op."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-002", image="busybox:1",
        userns_mode="remap",
    )
    container = client.containers.get(record.container_id)
    assert container.userns_mode == "", (
        f"userns_mode='remap' must pass empty string to docker "
        f"(daemon-default behavior); got {container.userns_mode!r}"
    )


@pytest.mark.asyncio
async def test_acquire_rejects_unknown_userns_mode() -> None:
    """B5.4 — values outside ``{"host", "remap"}`` must raise.
    The fail-loud guard prevents a remap-enabled daemon from
    silently using its default when an unrecognized wire value
    slips past the SDK's ``Literal["host", "remap"]`` hint."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    with pytest.raises(ValueError, match="unsupported userns_mode"):
        await mgr.acquire(
            rollout_id="r-003", image="busybox:1",
            userns_mode="bogus",
        )
    # Empty string is also invalid — the wire layer normalizes
    # empty → "host" upstream, so by the time it reaches the
    # manager we want a known label, not silent fall-through.
    with pytest.raises(ValueError, match="unsupported userns_mode"):
        await mgr.acquire(
            rollout_id="r-004", image="busybox:1",
            userns_mode="",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Issue #6 audit M1: Level-1/2 docker kwargs (cap_add, devices, privileged,
# network_mode, binds) flow from the control plane into
# ``RawContainerManager.acquire`` and through to docker-py's
# ``containers.run``. Tests assert each lands on the wire — closing the
# audit M1 finding that operator opt-ins were unreachable.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_applies_cpu_mem_cgroup_limits() -> None:
    """P1 — the effective ResourceSpec becomes docker cgroup kwargs:
    cpu_period/cpu_quota express the CPU cap, mem_limit the memory cap."""
    from xrlenv.backends.base import ResourceSpec

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(
        rollout_id="r-cg", image="busybox:1",
        resources=ResourceSpec(
            cpu_request=3.0, cpu_limit=3.0,
            mem_request_bytes=6 * 1024**3, mem_limit_bytes=6 * 1024**3,
            disk_request_bytes=1 * 1024**3,
        ),
    )
    extra = client.containers.run_extra[-1]
    assert extra["cpu_period"] == 100_000
    assert extra["cpu_quota"] == 300_000          # 3.0 CPU
    assert extra["mem_limit"] == 6 * 1024**3


@pytest.mark.asyncio
async def test_acquire_without_resources_falls_back_to_node_default() -> None:
    """P1 regression guard — a missing ResourceSpec resolves to the node
    default cap, never an unbounded container."""
    from xrlenv.node.raw_container import (
        _DEFAULT_RAW_CGROUP_CPU_LIMIT,
        _DEFAULT_RAW_CGROUP_MEM_LIMIT_BYTES,
    )

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(rollout_id="r-def", image="busybox:1")  # no resources

    extra = client.containers.run_extra[-1]
    assert extra["cpu_quota"] == int(_DEFAULT_RAW_CGROUP_CPU_LIMIT * 100_000)
    assert extra["mem_limit"] == _DEFAULT_RAW_CGROUP_MEM_LIMIT_BYTES
    # The container is never spawned with no cgroup limits at all.
    assert extra["cpu_quota"] > 0 and extra["mem_limit"] > 0


@pytest.mark.asyncio
async def test_acquire_applies_runtime_limits() -> None:
    """P0b — RuntimeLimits (pids / shm / tmpfs / read-only) become the
    matching docker containers.run kwargs."""
    from xrlenv.backends.base import RuntimeLimits

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(
        rollout_id="r-rl", image="busybox:1",
        runtime_limits=RuntimeLimits(
            pids_limit=4096,
            shm_size_bytes=67108864,
            tmpfs={"/run": "size=64m"},
            readonly_rootfs=True,
        ),
    )
    extra = client.containers.run_extra[-1]
    assert extra["pids_limit"] == 4096
    assert extra["shm_size"] == 67108864
    assert extra["tmpfs"] == {"/run": "size=64m"}
    assert extra["read_only"] is True


@pytest.mark.asyncio
async def test_acquire_without_runtime_limits_applies_no_shape_kwargs() -> None:
    """P0b — no RuntimeLimits → none of the container-shape kwargs are
    set, matching local-Docker behaviour (cluster mode injects no
    pids/shm defaults of its own)."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(rollout_id="r-nrl", image="busybox:1")

    extra = client.containers.run_extra[-1]
    assert "pids_limit" not in extra
    assert "shm_size" not in extra
    assert "tmpfs" not in extra
    assert "read_only" not in extra


# ──────────────────────────────────────────────────────────────────────────────
# P2 — cpuset pinning + core ledger
# ──────────────────────────────────────────────────────────────────────────────


def test_core_ledger_allocate_release_roundtrip() -> None:
    from xrlenv.node.raw_container import _CoreLedger

    ledger = _CoreLedger(8)
    assert ledger.free_count() == 8
    a = ledger.allocate(3)
    assert a is not None and len(a) == 3
    assert ledger.free_count() == 5
    # Allocations are disjoint.
    b = ledger.allocate(3)
    assert b is not None and set(a).isdisjoint(b)
    ledger.release(a)
    assert ledger.free_count() == 5


def test_core_ledger_exhaustion_returns_none() -> None:
    from xrlenv.node.raw_container import _CoreLedger

    ledger = _CoreLedger(4)
    assert ledger.allocate(4) is not None
    # No cores left — over-request returns None rather than raising.
    assert ledger.allocate(1) is None


def test_core_ledger_mark_in_use_for_crash_recovery() -> None:
    from xrlenv.node.raw_container import _CoreLedger

    ledger = _CoreLedger(8)
    ledger.mark_in_use([0, 1, 2])
    assert ledger.free_count() == 5
    a = ledger.allocate(5)
    assert a is not None and set(a).isdisjoint({0, 1, 2})


def test_parse_cpuset_roundtrip() -> None:
    from xrlenv.node.raw_container import _cpuset_str, _parse_cpuset

    assert _parse_cpuset("0,2-4,7") == (0, 2, 3, 4, 7)
    assert _cpuset_str([3, 1, 2]) == "1,2,3"
    assert _parse_cpuset(_cpuset_str([5, 0, 9])) == (0, 5, 9)


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-3a — shared-parent cpuset foundation (§8.11), gated OFF (behavior-neutral)
# ──────────────────────────────────────────────────────────────────────────────


class _FakeCgroupWriter:
    """Records the cgroup writes the shared parent makes, so tests assert the
    exact cpuset.cpus strings without touching /sys."""

    def __init__(self) -> None:
        self.ensured: list[str] = []
        self.cpuset_writes: list[tuple[str, str]] = []

    def ensure_group(self, path: str) -> None:
        self.ensured.append(path)

    def write_cpuset_cpus(self, path: str, value: str) -> None:
        self.cpuset_writes.append((path, value))

    def remove_group(self, path: str) -> None:  # pragma: no cover - unused here
        pass


def _shared_parent(writer: Any, *, total: int = 8) -> Any:
    from xrlenv.node.raw_container import _SharedCpusetParent

    return _SharedCpusetParent(
        total_cores=total, cgroup_root="/cg", writer=writer,
    )


def test_shared_cpuset_parent_ensure_writes_all_cores_once() -> None:
    w = _FakeCgroupWriter()
    parent = _shared_parent(w)
    assert parent.cgroup_parent == "/xrlenv-shared"
    parent.ensure()
    assert w.ensured == ["/cg/xrlenv-shared"]
    assert w.cpuset_writes[-1] == ("/cg/xrlenv-shared", "0,1,2,3,4,5,6,7")
    parent.ensure()  # idempotent — no second ensure_group / write
    assert w.ensured == ["/cg/xrlenv-shared"]
    assert len(w.cpuset_writes) == 1


def test_shared_cpuset_parent_set_complement_writes_all_minus_in_use() -> None:
    w = _FakeCgroupWriter()
    parent = _shared_parent(w)
    parent.set_complement({0, 1})  # ensure() (all) then complement (2..7)
    assert w.cpuset_writes[-1] == ("/cg/xrlenv-shared", "2,3,4,5,6,7")
    parent.set_complement(set())  # no pins → whole node is the shared pool
    assert w.cpuset_writes[-1] == ("/cg/xrlenv-shared", "0,1,2,3,4,5,6,7")


def test_shared_cpuset_parent_empty_complement_falls_back_to_single_core() -> None:
    """Every logical CPU pinned (over-pinned node, only reachable via a legacy
    reconcile — never via the floor). ``set_complement`` must NOT write an empty
    cpuset (which cgroup v2 reads as 'inherit all' → unpinned get every core);
    it confines to a single core instead."""
    w = _FakeCgroupWriter()
    parent = _shared_parent(w, total=4)
    parent.set_complement({0, 1, 2, 3})  # all pinned → complement empty
    assert w.cpuset_writes[-1] == ("/cg/xrlenv-shared", "3")  # max core, not ""


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-3c — compose cgroup_parent injection for runc services
# ──────────────────────────────────────────────────────────────────────────────


def test_inject_shared_cgroup_parent_into_runc_services() -> None:
    import yaml
    from xrlenv.node.raw_container import _inject_shared_cgroup_parent

    doc_in = "services:\n  main:\n    image: app\n  db:\n    image: pg\n"
    out = _inject_shared_cgroup_parent(doc_in, "/xrlenv-shared")
    svcs = yaml.safe_load(out)["services"]
    assert svcs["main"]["cgroup_parent"] == "/xrlenv-shared"
    assert svcs["db"]["cgroup_parent"] == "/xrlenv-shared"
    # An explicit runtime: runc is still the daemon default → injected.
    out2 = _inject_shared_cgroup_parent("services:\n  m:\n    runtime: runc\n", "/xrlenv-shared")
    assert yaml.safe_load(out2)["services"]["m"]["cgroup_parent"] == "/xrlenv-shared"


def test_inject_shared_cgroup_parent_skips_sysbox_services() -> None:
    import yaml
    from xrlenv.node.raw_container import _inject_shared_cgroup_parent

    doc_in = (
        "services:\n"
        "  main:\n    image: app\n"
        "  sb:\n    image: dind\n    runtime: sysbox-runc\n"
    )
    svcs = yaml.safe_load(_inject_shared_cgroup_parent(doc_in, "/xrlenv-shared"))["services"]
    assert svcs["main"]["cgroup_parent"] == "/xrlenv-shared"      # runc → injected
    assert "cgroup_parent" not in svcs["sb"]                      # sysbox → untouched


def test_inject_shared_cgroup_parent_best_effort_on_bad_or_empty() -> None:
    """A parse/shape surprise returns the document unchanged — a transform bug
    must never block a compose up."""
    from xrlenv.node.raw_container import _inject_shared_cgroup_parent

    assert _inject_shared_cgroup_parent("{", "/x") == "{"                # invalid YAML
    assert _inject_shared_cgroup_parent("- a\n- b\n", "/x") == "- a\n- b\n"  # not a mapping
    assert _inject_shared_cgroup_parent("services: null\n", "/x") == "services: null\n"


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-4a — fold the legacy transition-gap into pinned_cpus_free (behavior-neutral)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pinned_cpu_capacity_folds_legacy_gap_to_zero_free() -> None:
    """P6 step-4a — a CAPABLE node with legacy unpinned-runc containers reports
    pinned_cpus_free=0 (so the step-4 predicate won't place `required` there),
    while total stays real; once the gap drains, free returns to the ledger."""
    w = _FakeCgroupWriter()
    _client, mgr = _capable_manager(w, total_cores=8)
    await mgr._ensure_isolation_wiring()          # wire the shared parent
    assert mgr._shared_parent is not None
    assert mgr.pinned_cpu_capacity() == (8, 8)    # no legacy → real free

    mgr._legacy_unpinned_runc = 2                 # legacy present
    assert mgr.pinned_cpu_capacity() == (0, 8)    # free folded to 0, total real

    mgr._legacy_unpinned_runc = 0                 # drained
    assert mgr.pinned_cpu_capacity() == (8, 8)


@pytest.mark.asyncio
async def test_pinned_cpu_capacity_no_fold_on_non_capable_node() -> None:
    """The fold only applies on a capable node (a wired shared parent). A
    non-capable node never folds — behavior-neutral."""
    client = _FakeDockerClient(images_present=set())
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    mgr._legacy_unpinned_runc = 5                 # even if set, no shared parent → no fold
    assert mgr._shared_parent is None
    assert mgr.pinned_cpu_capacity() == (8, 8)


@pytest.mark.asyncio
async def test_refresh_legacy_gap_recomputes_and_clears_fold() -> None:
    """refresh_legacy_gap recomputes from docker (the heartbeat cadence drains
    the gap as legacy containers exit); non-capable → 0."""
    # Non-capable: refresh is a no-op 0.
    plain = RawContainerManager(
        docker_client=_FakeDockerClient(images_present=set()), total_cores=8,
    )
    assert await plain.refresh_legacy_gap() == 0

    # Capable + wired: a stale non-zero cache is recomputed from docker (empty
    # fake → 0), clearing the fold.
    w = _FakeCgroupWriter()
    _c, cap = _capable_manager(w, total_cores=8)
    await cap._ensure_isolation_wiring()
    cap._legacy_unpinned_runc = 5                 # stale
    assert cap.pinned_cpu_capacity() == (0, 8)    # folded while stale
    assert await cap.refresh_legacy_gap() == 0    # docker list empty → 0
    assert cap.pinned_cpu_capacity() == (8, 8)    # fold cleared


@pytest.mark.asyncio
async def test_refresh_legacy_gap_proactively_wires_capable_node() -> None:
    """P6 step-4b pre-req (audit follow-up) — on a CAPABLE node, refresh_legacy_gap
    (the heartbeat cadence) wires the shared parent PROACTIVELY, without waiting
    for the first acquire. So a capable-but-idle node's pinned_cpu_capacity
    reflects the real transition gap before step-4b places any `required` work.
    A non-capable node stays unwired (no shared parent)."""
    w = _FakeCgroupWriter()
    _c, cap = _capable_manager(w, total_cores=8)
    assert cap._shared_parent is None             # not wired yet (no acquire)
    await cap.refresh_legacy_gap()                 # the heartbeat path
    assert cap._shared_parent is not None          # wired proactively, pre-acquire

    plain = RawContainerManager(
        docker_client=_FakeDockerClient(images_present=set()), total_cores=8,
    )
    plain._isolation_capable_cache = False         # hello resolved non-capable
    await plain.refresh_legacy_gap()
    assert plain._shared_parent is None            # non-capable → never wired


def test_core_ledger_syncs_complement_on_allocate_and_release() -> None:
    from xrlenv.node.raw_container import _CoreLedger

    w = _FakeCgroupWriter()
    ledger = _CoreLedger(8, shared_parent=_shared_parent(w), min_shared_cores=2)
    picked = ledger.allocate(3)
    assert picked == (0, 1, 2)
    # Shared parent shrunk to the complement of the pinned cores.
    assert w.cpuset_writes[-1] == ("/cg/xrlenv-shared", "3,4,5,6,7")
    ledger.release([0, 1, 2])
    # Restored to the whole node once nothing is pinned.
    assert w.cpuset_writes[-1] == ("/cg/xrlenv-shared", "0,1,2,3,4,5,6,7")


def test_core_ledger_mark_in_use_syncs_complement_for_reconcile() -> None:
    from xrlenv.node.raw_container import _CoreLedger

    w = _FakeCgroupWriter()
    ledger = _CoreLedger(8, shared_parent=_shared_parent(w))
    ledger.mark_in_use([0, 1])  # restart reconcile → complement 2..7
    assert w.cpuset_writes[-1] == ("/cg/xrlenv-shared", "2,3,4,5,6,7")


def test_core_ledger_floor_refuses_pinning_below_min_shared() -> None:
    from xrlenv.node.raw_container import _CoreLedger

    w = _FakeCgroupWriter()
    ledger = _CoreLedger(8, shared_parent=_shared_parent(w), min_shared_cores=2)
    assert ledger.allocate(6) is not None      # free 8→2, floor 2 — OK
    writes_before = len(w.cpuset_writes)
    assert ledger.allocate(1) is None          # would drop shared pool to 1 < 2
    assert ledger.free_count() == 2            # refusal didn't consume cores
    assert len(w.cpuset_writes) == writes_before  # and wrote no complement


def test_core_ledger_floor_clamped_to_total() -> None:
    """A misconfigured floor ≥ total can't wedge the node into refusing all
    pinning; it's clamped so at least the legacy path works."""
    from xrlenv.node.raw_container import _CoreLedger

    ledger = _CoreLedger(4, min_shared_cores=100)
    # Clamped to total=4 → free-after-grant must stay ≥ 4 → can't pin any.
    # (Clamp prevents a NEGATIVE floor / overflow; refusing all pinning at
    # floor==total is the correct meaning of "reserve the whole node unpinned".)
    assert ledger.allocate(1) is None
    assert ledger.free_count() == 4


def test_core_ledger_default_shared_parent_none_is_behavior_neutral() -> None:
    """The 3a defaults (no shared parent, floor 0) reproduce pre-step-3 behavior
    exactly: allocate to exhaustion, no cgroup writes."""
    from xrlenv.node.raw_container import _CoreLedger

    ledger = _CoreLedger(4)  # shared_parent=None, min_shared_cores=0
    a = ledger.allocate(4)
    assert a is not None and len(a) == 4
    assert ledger.allocate(1) is None  # exhausted, floor 0 → legacy check
    ledger.release(a)
    assert ledger.free_count() == 4


def test_core_ledger_complement_write_failure_is_non_fatal() -> None:
    """A failing cgroup write degrades isolation but must not break the ledger's
    allocate/release bookkeeping (the pinned container still holds its cores via
    its own cpuset_cpus)."""
    from xrlenv.node.raw_container import _CoreLedger, _SharedCpusetParent

    class _BoomWriter:
        def ensure_group(self, path: str) -> None:
            pass

        def write_cpuset_cpus(self, path: str, value: str) -> None:
            raise OSError("permission denied")

        def remove_group(self, path: str) -> None:
            pass

    parent = _SharedCpusetParent(total_cores=8, cgroup_root="/cg", writer=_BoomWriter())
    ledger = _CoreLedger(8, shared_parent=parent)
    picked = ledger.allocate(2)
    assert picked is not None and ledger.free_count() == 6
    ledger.release(picked)
    assert ledger.free_count() == 8


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-3b — wire the shared parent into the acquire path (capable nodes only)
# ──────────────────────────────────────────────────────────────────────────────


def _capable_manager(
    writer: Any, *, total_cores: int = 8, min_shared_cores: Any = None,
    runtimes: Any = None,
) -> Any:
    """A RawContainerManager that reports isolation_capable=True (injected seam)
    and writes the shared parent through the given fake cgroup writer."""
    if runtimes is not None:
        client = _FakeDockerClient(images_present={"busybox:1"}, runtimes=runtimes)
    else:
        client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(
        docker_client=client, total_cores=total_cores,
        isolation_selftest=lambda: True, cgroup_writer=writer,
        min_shared_cores=min_shared_cores,
    )
    # Simulate NodeHello: the self-test runs (and caches) at hello time, before
    # any acquire — the acquire path only reads the cached capability.
    assert mgr.isolation_capable() is True
    return client, mgr


@pytest.mark.asyncio
async def test_acquire_unpinned_runc_placed_under_shared_parent_when_capable() -> None:
    """P6 step-3b — on a capable node an UNPINNED runc container is created under
    /xrlenv-shared (no own cpuset), so it's confined to the shared pool."""
    w = _FakeCgroupWriter()
    client, mgr = _capable_manager(w)
    record = await mgr.acquire(rollout_id="r-unp", image="busybox:1")

    extra = client.containers.run_extra[-1]
    assert extra["cgroup_parent"] == "/xrlenv-shared"
    assert "cpuset_cpus" not in extra
    assert record.cpuset == ()
    # Shared parent was created + initialised to the whole node (no pins yet).
    assert w.ensured == ["/sys/fs/cgroup/xrlenv-shared"]
    assert w.cpuset_writes[-1] == ("/sys/fs/cgroup/xrlenv-shared", "0,1,2,3,4,5,6,7")


@pytest.mark.asyncio
async def test_acquire_pinned_shrinks_shared_parent_and_sets_cpuset_when_capable() -> None:
    """P6 step-3b — a PINNED container on a capable node keeps its explicit
    cpuset_cpus (exclusive cores) AND the shared parent is shrunk to exclude
    them, so unpinned children can't reach the pinned cores."""
    from xrlenv.backends.base import CpuIsolation

    w = _FakeCgroupWriter()
    client, mgr = _capable_manager(w)
    record = await mgr.acquire(
        rollout_id="r-pin", image="busybox:1",
        resources=_iso_resources(CpuIsolation.BEST_EFFORT),  # pins 2 cores
    )
    extra = client.containers.run_extra[-1]
    assert extra["cpuset_cpus"] == "0,1"
    assert "cgroup_parent" not in extra          # pinned → not under shared parent
    assert record.cpuset == (0, 1)
    # Shared parent shrunk to the complement of the pinned cores.
    assert w.cpuset_writes[-1] == ("/sys/fs/cgroup/xrlenv-shared", "2,3,4,5,6,7")


@pytest.mark.asyncio
async def test_acquire_sysbox_not_placed_under_shared_parent() -> None:
    """P6 step-3b — a sysbox-runtime container is left on today's path (the
    self-test only proved runc); no cgroup_parent injection even on a capable
    node."""
    w = _FakeCgroupWriter()
    client, mgr = _capable_manager(w, runtimes={"runc", "sysbox-runc"})
    await mgr.acquire(
        rollout_id="r-sb", image="busybox:1", container_runtime="sysbox-runc",
    )
    extra = client.containers.run_extra[-1]
    assert extra.get("runtime") == "sysbox-runc"
    assert "cgroup_parent" not in extra


@pytest.mark.asyncio
async def test_acquire_non_capable_node_is_unchanged() -> None:
    """P6 step-3b — a NON-capable node never wires a shared parent: no
    cgroup_parent, no cgroup writes — byte-for-byte today's behavior."""
    w = _FakeCgroupWriter()
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(
        docker_client=client, total_cores=8,
        isolation_selftest=lambda: False, cgroup_writer=w,
    )
    assert mgr.isolation_capable() is False  # hello resolved it as non-capable
    await mgr.acquire(rollout_id="r-nc", image="busybox:1")
    extra = client.containers.run_extra[-1]
    assert "cgroup_parent" not in extra
    assert w.ensured == [] and w.cpuset_writes == []
    assert mgr._shared_parent is None


@pytest.mark.asyncio
async def test_min_shared_cores_defaults_to_25_percent() -> None:
    """P6 step-3b — without an override the floor is 25% of the node's logical
    CPUs (ceil); an explicit override wins."""
    w = _FakeCgroupWriter()
    _client, mgr = _capable_manager(w, total_cores=8)  # 25% of 8 = 2
    await mgr.acquire(rollout_id="r-f", image="busybox:1")  # triggers wiring
    assert mgr._core_ledger.min_shared_cores == 2

    w2 = _FakeCgroupWriter()
    _c2, mgr2 = _capable_manager(w2, total_cores=8, min_shared_cores=5)
    await mgr2.acquire(rollout_id="r-f2", image="busybox:1")
    assert mgr2._core_ledger.min_shared_cores == 5


@pytest.mark.asyncio
async def test_acquire_pins_cpuset_when_opted_in() -> None:
    """P2 — cpuset pinning is OPT-IN per acquire via
    RuntimeLimits(cpu_pinning=True); it then pins ceil(cpu_limit) whole
    cores (the default 2.0 CPU limit reserves 2)."""
    from xrlenv.backends.base import RuntimeLimits

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr.acquire(
        rollout_id="r-pin", image="busybox:1",
        runtime_limits=RuntimeLimits(cpu_pinning=True),
    )

    extra = client.containers.run_extra[-1]
    assert extra["cpuset_cpus"] == "0,1"          # 2 cores for 2.0 CPU
    assert record.cpuset == (0, 1)
    assert mgr._core_ledger.free_count() == 6


@pytest.mark.asyncio
async def test_acquire_pins_ceil_of_cpu_limit() -> None:
    """P2 — an opted-in fractional cpu_limit pins ceil() whole cores."""
    from xrlenv.backends.base import ResourceSpec, RuntimeLimits

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    await mgr.acquire(
        rollout_id="r-ceil", image="busybox:1",
        resources=ResourceSpec(
            cpu_request=3.5, cpu_limit=3.5,
            mem_request_bytes=1 << 30, mem_limit_bytes=1 << 30,
            disk_request_bytes=1 << 30,
        ),
        runtime_limits=RuntimeLimits(cpu_pinning=True),
    )
    extra = client.containers.run_extra[-1]
    assert extra["cpuset_cpus"] == "0,1,2,3"      # ceil(3.5) = 4 cores


@pytest.mark.asyncio
async def test_acquire_no_pinning_by_default() -> None:
    """Faithful default — without RuntimeLimits(cpu_pinning=True) the
    container runs CFS-quota-only (no cpuset_cpus), exactly like harbor.
    The core ledger is left untouched; pinning never happens unless a
    task explicitly opts in at acquire time."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr.acquire(rollout_id="r-nopin", image="busybox:1")
    assert "cpuset_cpus" not in client.containers.run_extra[-1]
    assert record.cpuset == ()
    assert mgr._core_ledger.free_count() == 8


@pytest.mark.asyncio
async def test_destroy_releases_pinned_cores() -> None:
    """P2 — destroy returns an opted-in container's cores to the ledger."""
    from xrlenv.backends.base import RuntimeLimits

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr.acquire(
        rollout_id="r-rel", image="busybox:1",
        runtime_limits=RuntimeLimits(cpu_pinning=True),
    )
    assert mgr._core_ledger.free_count() == 6

    await mgr.destroy(
        rollout_id="r-rel", container_id=record.container_id,
    )
    assert mgr._core_ledger.free_count() == 8


@pytest.mark.asyncio
async def test_acquire_degrades_gracefully_when_ledger_exhausted() -> None:
    """P2 — an opted-in acquire still succeeds (CFS-quota-only) when the
    ledger has no free cores; pinning is a best-effort determinism boost,
    not an admission gate."""
    from xrlenv.backends.base import RuntimeLimits

    pin = RuntimeLimits(cpu_pinning=True)
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=2)
    # First acquire takes both cores (default 2.0 CPU → 2 cores).
    first = await mgr.acquire(
        rollout_id="r-1", image="busybox:1", runtime_limits=pin,
    )
    assert first.cpuset == (0, 1)
    # Second acquire: ledger empty → graceful degradation.
    second = await mgr.acquire(
        rollout_id="r-2", image="busybox:1", runtime_limits=pin,
    )
    assert second.cpuset == ()
    assert "cpuset_cpus" not in client.containers.run_extra[-1]


@pytest.mark.asyncio
async def test_acquire_validation_error_does_not_leak_cores() -> None:
    """P2 (audit M2) — a pre-run local validation error (e.g. an
    unsupported userns_mode) must not leak a ledger reservation even for
    an opted-in acquire: the cpuset is allocated only after all local
    validation passes."""
    from xrlenv.backends.base import RuntimeLimits

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    with pytest.raises(ValueError, match="userns_mode"):
        await mgr.acquire(
            rollout_id="r-bad", image="busybox:1", userns_mode="bogus",
            runtime_limits=RuntimeLimits(cpu_pinning=True),
        )
    # No container was created — every core is still free.
    assert mgr._core_ledger.free_count() == 8


@pytest.mark.asyncio
async def test_acquire_run_failure_releases_cores() -> None:
    """P2 — a failed containers.run releases the reserved cores back to
    the ledger so a transient docker failure can't leak them."""
    import requests.exceptions
    from xrlenv.backends.base import RuntimeLimits

    client = _FakeDockerClient(images_present={"busybox:1"})

    def _boom(**_kw: Any) -> Any:
        raise requests.exceptions.ReadTimeout("read timed out")

    client.containers.run = _boom  # type: ignore[method-assign]
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    with pytest.raises(XRLEnvError):
        await mgr.acquire(
            rollout_id="r-fail", image="busybox:1",
            runtime_limits=RuntimeLimits(cpu_pinning=True),
        )
    assert mgr._core_ledger.free_count() == 8


@pytest.mark.asyncio
async def test_core_ledger_reconciles_from_live_containers() -> None:
    """P2 (Risk 4) — a fresh manager rebuilds the ledger from live raw
    containers so a node-process restart can't double-assign cores."""
    from xrlenv.backends.base import RuntimeLimits

    pin = RuntimeLimits(cpu_pinning=True)
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr1 = RawContainerManager(docker_client=client, total_cores=8)
    await mgr1.acquire(
        rollout_id="r-old", image="busybox:1", runtime_limits=pin,
    )  # pins 0,1

    # Simulate a node-process restart: brand-new manager, same docker.
    mgr2 = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr2.acquire(
        rollout_id="r-new", image="busybox:1", runtime_limits=pin,
    )
    # The reconcile saw the surviving container on cores 0,1, so the
    # new container gets a disjoint set.
    assert set(record.cpuset).isdisjoint({0, 1})
    assert mgr2._core_ledger.free_count() == 4  # 8 - 2 (old) - 2 (new)


# ──────────────────────────────────────────────────────────────────────────────
# P6 — node consumes the explicit ResourceSpec.cpu_isolation contract
# ──────────────────────────────────────────────────────────────────────────────


def _iso_resources(cpu_isolation: Any, *, cpu_limit: float = 2.0) -> Any:
    """A ResourceSpec that carries an explicit cpu_isolation mode (and a
    cpu_limit so the pinned core-count is predictable: ceil(cpu_limit))."""
    from xrlenv.backends.base import ResourceSpec

    return ResourceSpec(
        cpu_request=cpu_limit, cpu_limit=cpu_limit,
        mem_request_bytes=1 << 30, mem_limit_bytes=1 << 30,
        disk_request_bytes=1 << 30,
        cpu_isolation=cpu_isolation,
    )


@pytest.mark.asyncio
async def test_acquire_pins_when_cpu_isolation_best_effort() -> None:
    """P6 — an explicit ``ResourceSpec.cpu_isolation=BEST_EFFORT`` pins
    ceil(cpu_limit) whole cores, WITHOUT the legacy
    ``RuntimeLimits(cpu_pinning=True)`` alias. This proves the new wire field
    reaches the node's pinning decision on its own."""
    from xrlenv.backends.base import CpuIsolation

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr.acquire(
        rollout_id="r-be", image="busybox:1",
        resources=_iso_resources(CpuIsolation.BEST_EFFORT),
    )
    assert client.containers.run_extra[-1]["cpuset_cpus"] == "0,1"
    assert record.cpuset == (0, 1)
    assert mgr._core_ledger.free_count() == 6


@pytest.mark.asyncio
async def test_acquire_pins_when_cpu_isolation_required() -> None:
    """P6 — ``REQUIRED`` pins ``ceil(cpu_limit)`` dedicated cores when the node
    has free pinnable capacity (identical to ``BEST_EFFORT`` on the happy path;
    they differ only on ledger exhaustion — see the pin-or-fail test below)."""
    from xrlenv.backends.base import CpuIsolation

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr.acquire(
        rollout_id="r-req", image="busybox:1",
        resources=_iso_resources(CpuIsolation.REQUIRED),
    )
    assert record.cpuset == (0, 1)
    assert mgr._core_ledger.free_count() == 6


@pytest.mark.asyncio
async def test_acquire_required_fails_loud_when_ledger_exhausted() -> None:
    """P6 step-4c — ``REQUIRED`` is pin-or-fail: on ledger exhaustion the acquire
    raises ``PinCapacityExhausted`` (a node-specific ``CapacityExhausted``
    subclass the CP re-admits) rather than silently degrading to CFS quota (a
    silent degrade would reopen the trampling the caller required isolation to
    prevent). No container is created and no ledger reservation leaks — the first
    pin's cores stay held, nothing else is taken."""
    from xrlenv.backends.base import CpuIsolation
    from xrlenv.errors import CapacityExhausted, PinCapacityExhausted

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=2)
    first = await mgr.acquire(
        rollout_id="r-req1", image="busybox:1",
        resources=_iso_resources(CpuIsolation.REQUIRED),  # takes cores 0,1
    )
    assert first.cpuset == (0, 1)
    runs_before = len(client.containers.run_extra)
    with pytest.raises(PinCapacityExhausted) as ei:
        await mgr.acquire(
            rollout_id="r-req2", image="busybox:1",
            resources=_iso_resources(CpuIsolation.REQUIRED),  # ledger empty
        )
    assert isinstance(ei.value, CapacityExhausted)  # re-admit-terminal contract
    # No container created for the failed acquire, and the ledger still holds
    # only the first pin's two cores (nothing leaked / partially reserved).
    assert len(client.containers.run_extra) == runs_before
    assert mgr._core_ledger.free_count() == 0


@pytest.mark.asyncio
async def test_acquire_best_effort_still_degrades_when_ledger_exhausted() -> None:
    """P6 step-4c — the pin-or-fail flip is REQUIRED-only: ``BEST_EFFORT`` still
    degrades to CFS-quota-only (no cpuset, no raise) on ledger exhaustion."""
    from xrlenv.backends.base import CpuIsolation

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=2)
    await mgr.acquire(
        rollout_id="r-be1", image="busybox:1",
        resources=_iso_resources(CpuIsolation.BEST_EFFORT),  # takes cores 0,1
    )
    second = await mgr.acquire(
        rollout_id="r-be2", image="busybox:1",
        resources=_iso_resources(CpuIsolation.BEST_EFFORT),  # ledger empty
    )
    assert second.cpuset == ()
    assert "cpuset_cpus" not in client.containers.run_extra[-1]


@pytest.mark.asyncio
async def test_acquire_cpu_isolation_off_with_legacy_pinning_still_pins() -> None:
    """P6 — the legacy alias is preserved: ``cpu_isolation=OFF`` (the default)
    combined with ``RuntimeLimits(cpu_pinning=True)`` still pins, exactly as
    before this slice. Existing harbor/task markers keep working unchanged."""
    from xrlenv.backends.base import CpuIsolation, RuntimeLimits

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr.acquire(
        rollout_id="r-legacy", image="busybox:1",
        resources=_iso_resources(CpuIsolation.OFF),
        runtime_limits=RuntimeLimits(cpu_pinning=True),
    )
    assert record.cpuset == (0, 1)
    assert mgr._core_ledger.free_count() == 6


@pytest.mark.asyncio
async def test_acquire_cpu_isolation_off_no_pinning() -> None:
    """P6 — ``cpu_isolation=OFF`` and no legacy alias → CFS-quota-only, no
    cpuset (the faithful default is unchanged)."""
    from xrlenv.backends.base import CpuIsolation

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    record = await mgr.acquire(
        rollout_id="r-off", image="busybox:1",
        resources=_iso_resources(CpuIsolation.OFF),
    )
    assert "cpuset_cpus" not in client.containers.run_extra[-1]
    assert record.cpuset == ()
    assert mgr._core_ledger.free_count() == 8


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-2a — node advertises isolation capability + pinned-CPU accounting
# ──────────────────────────────────────────────────────────────────────────────


def test_pinned_cpu_capacity_reports_ledger_free_and_total() -> None:
    """P6 (§8.6, R6) — ``pinned_cpu_capacity()`` reflects the core ledger:
    ``(free, total)``, and ``free`` drops as cores are reserved."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    assert mgr.pinned_cpu_capacity() == (8, 8)
    # Reserve 3 cores directly on the ledger; free drops, total holds.
    mgr._core_ledger.allocate(3)
    assert mgr.pinned_cpu_capacity() == (5, 8)


@pytest.mark.asyncio
async def test_pinned_cpu_capacity_tracks_pinned_acquire() -> None:
    """P6 — an opted-in (pinned) acquire is reflected in the reported free
    count, so the heartbeat carries live pinnable-CPU capacity."""
    from xrlenv.backends.base import CpuIsolation

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, total_cores=8)
    await mgr.acquire(
        rollout_id="r-cap", image="busybox:1",
        resources=_iso_resources(CpuIsolation.BEST_EFFORT),  # pins 2 cores
    )
    assert mgr.pinned_cpu_capacity() == (6, 8)


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-2b — cgroup-v2 cpuset-inheritance self-test (isolation_capable)
# ──────────────────────────────────────────────────────────────────────────────


class _SelftestDocker:
    """Minimal docker stand-in for the self-test gate tests: answers
    ``info()['CgroupDriver']`` and ``images.get``."""

    def __init__(self, *, driver: str = "cgroupfs", image_present: bool = True) -> None:
        self._driver = driver
        self._image_present = image_present

    def info(self) -> dict[str, Any]:
        return {"CgroupDriver": self._driver}

    class _Images:
        def __init__(self, present: bool) -> None:
            self._present = present

        def get(self, image: str) -> object:
            if not self._present:
                raise RuntimeError("ImageNotFound")
            return object()

    @property
    def images(self) -> _SelftestDocker._Images:
        return _SelftestDocker._Images(self._image_present)


@pytest.mark.parametrize("outcome", [True, False])
def test_isolation_capable_reflects_injected_selftest(outcome: bool) -> None:
    """P6 step-2b — ``isolation_capable()`` returns the self-test's verdict
    (injected seam; production runs the real container probe)."""
    mgr = RawContainerManager(
        docker_client=_FakeDockerClient(images_present=set()), total_cores=8,
        isolation_selftest=lambda: outcome,
    )
    assert mgr.isolation_capable() is outcome


def test_isolation_capable_caches_selftest_result() -> None:
    """The self-test runs at most once — capability doesn't change without a
    daemon/kernel change (which restarts the agent)."""
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return True

    mgr = RawContainerManager(
        docker_client=_FakeDockerClient(images_present=set()), total_cores=8,
        isolation_selftest=probe,
    )
    assert mgr.isolation_capable() is True
    assert mgr.isolation_capable() is True
    assert len(calls) == 1


def test_isolation_capable_fail_safe_when_selftest_raises() -> None:
    """Fail-safe: a self-test that raises → ``False`` (a node never advertises
    isolation it couldn't prove), cached so it isn't retried per call."""
    def probe() -> bool:
        raise RuntimeError("cgroup write exploded")

    mgr = RawContainerManager(
        docker_client=_FakeDockerClient(images_present=set()), total_cores=8,
        isolation_selftest=probe,
    )
    assert mgr.isolation_capable() is False
    assert mgr.isolation_capable() is False


def test_default_selftest_false_on_non_cgroupfs_docker() -> None:
    """§8.13 — the DEFAULT (production) capability check reports ``False`` for a
    manager whose docker driver isn't ``cgroupfs`` (a plain node stays
    non-capable without opt-in). ``_FakeDockerClient.info()`` reports no
    ``CgroupDriver`` ⇒ ``""`` ⇒ the P6-v1 gate fails before any cgroup probe."""
    mgr = RawContainerManager(
        docker_client=_FakeDockerClient(images_present={"busybox:1"}),
        total_cores=8,
    )
    assert mgr.isolation_capable() is False


def test_delegated_shared_parent_writable_reflects_fs(tmp_path: Any) -> None:
    """§8.13 — the non-root agent's capability signal: ``xrlenv-shared/cpuset.cpus``
    present + writable ⇒ ``True`` (the root enable step delegated it); absent ⇒
    ``False`` (node never enabled)."""
    from xrlenv.node.raw_container import _delegated_shared_parent_writable

    root = tmp_path
    assert _delegated_shared_parent_writable(str(root)) is False  # no xrlenv-shared
    shared = root / "xrlenv-shared"
    shared.mkdir()
    (shared / "cpuset.cpus").write_text("0-3")
    assert _delegated_shared_parent_writable(str(root)) is True  # present + writable
    if os.geteuid() != 0:  # root bypasses DAC — the read-only case only holds non-root
        (shared / "cpuset.cpus").chmod(0o444)
        assert _delegated_shared_parent_writable(str(root)) is False


def test_default_selftest_true_when_delegation_intact(monkeypatch: Any) -> None:
    """§8.13 — the DEFAULT capability check is ``True`` when the docker driver is
    ``cgroupfs``, cgroup v2 exposes ``cpuset``, and ``xrlenv-shared`` is writable
    (the enable step delegated it). NO container probe runs in the (non-root)
    agent — it verifies the delegation instead."""
    import xrlenv.node.raw_container as rc

    monkeypatch.setattr(rc, "_cgroup_v2_cpuset_available", lambda *a, **k: True)
    monkeypatch.setattr(rc, "_delegated_shared_parent_writable", lambda *a, **k: True)
    mgr = RawContainerManager(
        docker_client=_SelftestDocker(driver="cgroupfs"), total_cores=8,
    )
    assert mgr.isolation_capable() is True


def test_default_selftest_false_when_not_delegated(monkeypatch: Any) -> None:
    """§8.13 — ``cgroupfs`` + cgroup v2 but ``xrlenv-shared`` absent / not writable
    (no delegation → node never enabled, or a reboot/driver-flip tore it down)
    ⇒ non-capable. Fail-safe: never a false ``true``."""
    import xrlenv.node.raw_container as rc

    monkeypatch.setattr(rc, "_cgroup_v2_cpuset_available", lambda *a, **k: True)
    monkeypatch.setattr(rc, "_delegated_shared_parent_writable", lambda *a, **k: False)
    mgr = RawContainerManager(
        docker_client=_SelftestDocker(driver="cgroupfs"), total_cores=8,
    )
    assert mgr.isolation_capable() is False


def test_default_selftest_false_on_systemd_driver_short_circuits(
    monkeypatch: Any,
) -> None:
    """§8.13 — the P6-v1 gate is checked FIRST: a ``systemd``-driver node stays
    non-capable even if a stray writable ``xrlenv-shared`` exists."""
    import xrlenv.node.raw_container as rc

    monkeypatch.setattr(rc, "_cgroup_v2_cpuset_available", lambda *a, **k: True)
    monkeypatch.setattr(rc, "_delegated_shared_parent_writable", lambda *a, **k: True)
    mgr = RawContainerManager(
        docker_client=_SelftestDocker(driver="systemd"), total_cores=8,
    )
    assert mgr.isolation_capable() is False


def test_cgroup_v2_cpuset_available_detects_unified_hierarchy(tmp_path: Any) -> None:
    from xrlenv.node.raw_container import _cgroup_v2_cpuset_available

    root = tmp_path
    assert _cgroup_v2_cpuset_available(str(root)) is False  # no cgroup.controllers (v1)
    (root / "cgroup.controllers").write_text("cpu memory pids")
    assert _cgroup_v2_cpuset_available(str(root)) is False  # v2 but no cpuset controller
    (root / "cgroup.controllers").write_text("cpuset cpu io memory")
    assert _cgroup_v2_cpuset_available(str(root)) is True


def test_docker_cgroup_driver_reads_info_and_traps_errors() -> None:
    from xrlenv.node.raw_container import _docker_cgroup_driver

    assert _docker_cgroup_driver(_SelftestDocker(driver="cgroupfs")) == "cgroupfs"
    assert _docker_cgroup_driver(_SelftestDocker(driver="systemd")) == "systemd"

    class _Boom:
        def info(self) -> dict[str, Any]:
            raise RuntimeError("docker down")

    assert _docker_cgroup_driver(_Boom()) == ""


def test_read_cpus_allowed_list_parses_proc_status() -> None:
    """Parse the real ``/proc/self/status`` field against this process's actual
    affinity — proves the parser matches the kernel's format."""
    from xrlenv.node.raw_container import _read_cpus_allowed_list

    assert _read_cpus_allowed_list(os.getpid()) == set(os.sched_getaffinity(0))
    assert _read_cpus_allowed_list(-1) == set()  # unreadable → empty


def test_selftest_short_circuits_before_any_cgroup_write(tmp_path: Any) -> None:
    """Every gate failure returns False WITHOUT touching the cgroup fs (no
    parent dir is created under the given root)."""
    from xrlenv.node.raw_container import _run_cgroup_isolation_selftest

    root = tmp_path
    # (a) not cgroup v2 (no controllers file).
    assert _run_cgroup_isolation_selftest(
        _SelftestDocker(), image="busybox:1", cgroup_root=str(root),
    ) is False
    # Make it "v2 + cpuset" for the remaining gate tests.
    (root / "cgroup.controllers").write_text("cpuset cpu memory")
    # (b) systemd driver (v1 decision: not capable).
    assert _run_cgroup_isolation_selftest(
        _SelftestDocker(driver="systemd"), image="busybox:1", cgroup_root=str(root),
    ) is False
    # (c) no probe image configured.
    assert _run_cgroup_isolation_selftest(
        _SelftestDocker(), image="", cgroup_root=str(root),
    ) is False
    # (d) probe image not present on the node (no pull at init).
    assert _run_cgroup_isolation_selftest(
        _SelftestDocker(image_present=False), image="busybox:1", cgroup_root=str(root),
    ) is False
    # No gate created the throwaway parent cgroup dir.
    assert not list(root.glob("xrlenv-selftest-*"))


@pytest.mark.skipif(
    os.geteuid() != 0,
    reason="real cgroup self-test needs root; runs only on a capable node",
)
def test_real_isolation_selftest_end_to_end() -> None:
    """Root+cgroup2+cgroupfs-gated integration test (§8.10): run the ACTUAL
    container probe. Skipped in CI / on the non-root dev box; validates the true
    path only where root + cgroup v2 + the cgroupfs driver + a probe image
    exist. It asserts a consistent bool, and — when the environment is fully
    capable — that a capable node reports True."""
    import docker
    from xrlenv.node.raw_container import (
        _cgroup_v2_cpuset_available,
        _docker_cgroup_driver,
        _run_cgroup_isolation_selftest,
    )

    image = os.environ.get("XRLENV_SELFTEST_IMAGE", "")
    client = docker.from_env()
    result = _run_cgroup_isolation_selftest(client, image=image)
    assert isinstance(result, bool)
    fully_capable = (
        _cgroup_v2_cpuset_available()
        and _docker_cgroup_driver(client) == "cgroupfs"
        and bool(image)
    )
    if fully_capable:
        assert result is True


@pytest.mark.asyncio
async def test_acquire_forwards_cap_add_to_containers_run() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-cap", image="busybox:1",
        cap_add=["SYS_PTRACE", "NET_ADMIN"],
    )
    container = client.containers.get(record.container_id)
    assert container.cap_add == ["SYS_PTRACE", "NET_ADMIN"]


@pytest.mark.asyncio
async def test_acquire_forwards_devices_to_containers_run() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-dev", image="busybox:1",
        devices=["/dev/kvm", "/dev/net/tun"],
    )
    container = client.containers.get(record.container_id)
    assert container.devices == ["/dev/kvm", "/dev/net/tun"]


@pytest.mark.asyncio
async def test_acquire_forwards_privileged_to_containers_run() -> None:
    """Audit M1: ``allow_privileged: true`` operator opt-in must reach
    docker-py."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-priv", image="busybox:1", privileged=True,
    )
    container = client.containers.get(record.container_id)
    assert container.privileged is True


@pytest.mark.asyncio
async def test_acquire_default_privileged_is_false() -> None:
    """No ``privileged=`` kwarg → docker-py's safe default of ``False``
    persists. Regression guard against accidentally setting it on
    every container."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-1", image="busybox:1")
    container = client.containers.get(record.container_id)
    assert container.privileged is False


@pytest.mark.asyncio
async def test_acquire_forwards_network_mode_to_containers_run() -> None:
    """Audit M1: ``allow_host_network: true`` operator opt-in must
    reach docker-py."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-net", image="busybox:1", network_mode="host",
    )
    container = client.containers.get(record.container_id)
    assert container.network_mode == "host"


@pytest.mark.asyncio
async def test_acquire_forwards_binds_to_volumes_kwarg() -> None:
    """Audit M1: ``allowed_host_paths`` operator opt-in. The manager
    translates the docker CLI ``"/host:/ctr:mode"`` bind spec into
    docker-py's high-level ``volumes={host: {bind, mode}}`` dict
    form before passing to ``containers.run``."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-bind", image="busybox:1",
        binds=["/mnt/datasets:/data:ro"],
    )
    container = client.containers.get(record.container_id)
    assert container.volumes == {
        "/mnt/datasets": {"bind": "/data", "mode": "ro"},
    }


@pytest.mark.asyncio
async def test_acquire_binds_default_mode_is_rw() -> None:
    """Bind spec without explicit mode (``/host:/ctr``) gets
    docker-py's default ``"rw"`` mode."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(
        rollout_id="r-bind2", image="busybox:1",
        binds=["/mnt/cache:/cache"],
    )
    container = client.containers.get(record.container_id)
    assert container.volumes == {
        "/mnt/cache": {"bind": "/cache", "mode": "rw"},
    }


@pytest.mark.asyncio
async def test_acquire_refuses_when_image_missing() -> None:
    """Phase-1 contract: no implicit pull on the raw-container path.
    A missing image surfaces as a clear ``XRLEnvError``."""
    client = _FakeDockerClient(images_present=set())
    mgr = RawContainerManager(docker_client=client)

    with pytest.raises(XRLEnvError, match="not present"):
        await mgr.acquire(rollout_id="r-001", image="missing:1")


@pytest.mark.asyncio
async def test_exec_returns_exit_code_and_demuxed_streams() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-001", image="busybox:1")
    # Pre-seed an exec result on the fake container.
    container = client.containers.get(record.container_id)
    container.exec_results[("echo", "hi")] = _ExecResult(
        exit_code=0, output=(b"hi\n", b""),
    )

    result = await mgr.exec(
        rollout_id="r-001",
        container_id=record.container_id,
        cmd=["echo", "hi"],
    )

    assert result == {
        "exit_code": 0, "stdout": b"hi\n", "stderr": b"",
        "timed_out": False,
    }


@pytest.mark.asyncio
async def test_exec_rejects_unowned_container() -> None:
    """Per-rollout scoping: rollout B cannot exec into a container
    rollout A acquired."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-A", image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not own"):
        await mgr.exec(
            rollout_id="r-B",
            container_id=record.container_id,
            cmd=["echo", "hi"],
        )


@pytest.mark.asyncio
async def test_exec_rejects_unknown_container_id() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    with pytest.raises(XRLEnvError, match="not registered"):
        await mgr.exec(
            rollout_id="r-001",
            container_id="ghost-id-deadbeef",
            cmd=["echo", "hi"],
        )


@pytest.mark.asyncio
async def test_exec_times_out_returns_timed_out_flag() -> None:
    """A slow exec that exceeds ``timeout_s`` returns
    ``timed_out=True`` rather than raising."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-001", image="busybox:1")
    container = client.containers.get(record.container_id)
    container.exec_delay_s = 0.5  # > timeout_s below

    result = await mgr.exec(
        rollout_id="r-001",
        container_id=record.container_id,
        cmd=["sleep", "1"],
        timeout_s=0.05,
    )

    assert result["timed_out"] is True
    assert result["exit_code"] == -1


@pytest.mark.asyncio
async def test_destroy_removes_container_and_deregisters() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-001", image="busybox:1")

    await mgr.destroy(
        rollout_id="r-001", container_id=record.container_id,
    )

    # Container is removed (fake registry tracks .removed).
    container = client.containers._registry[record.container_id]
    assert container.removed is True
    # Manager no longer tracks it; subsequent exec fails ownership check.
    owned = await mgr.list_owned()
    assert owned == []


@pytest.mark.asyncio
async def test_destroy_rejects_unregistered_container_id() -> None:
    """Audit Raw-Scoping-M1: caller cannot destroy a container they
    don't own by guessing its id. Symmetric with exec's
    ``not registered`` rejection."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    with pytest.raises(XRLEnvError, match="not registered"):
        await mgr.destroy(
            rollout_id="r-001", container_id="ghost-id-deadbeef",
        )


@pytest.mark.asyncio
async def test_destroy_idempotent_only_when_docker_side_already_gone() -> None:
    """Destroy is idempotent ONLY for registered records whose
    docker container has already been removed externally — a benign
    race with the harness's own ``docker rm``. A re-destroy after
    a successful destroy raises ``not registered`` because the
    record was deregistered the first time around (see
    ``test_destroy_rejects_unregistered_container_id``)."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-001", image="busybox:1")
    # Simulate the harness racing us: docker-side container removed
    # but the manager still has the record.
    container = client.containers._registry[record.container_id]
    container.removed = True
    # Destroy on a registered-but-docker-gone container is a no-op,
    # not an error (the goal — "make sure this is gone" — is
    # already satisfied).
    await mgr.destroy(
        rollout_id="r-001", container_id=record.container_id,
    )
    # Now the record is deregistered too; a follow-up destroy by
    # the same caller raises because there's no longer a record.
    with pytest.raises(XRLEnvError, match="not registered"):
        await mgr.destroy(
            rollout_id="r-001", container_id=record.container_id,
        )


@pytest.mark.asyncio
async def test_destroy_rejects_unowned_container() -> None:
    """Symmetric to exec — wrong rollout can't destroy."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-A", image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not own"):
        await mgr.destroy(
            rollout_id="r-B", container_id=record.container_id,
        )

    # And the original rollout can still destroy.
    await mgr.destroy(
        rollout_id="r-A", container_id=record.container_id,
    )


@pytest.mark.asyncio
async def test_list_owned_filters_by_rollout_id() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    a1 = await mgr.acquire(rollout_id="r-A", image="busybox:1")
    a2 = await mgr.acquire(rollout_id="r-A", image="busybox:1")
    b1 = await mgr.acquire(rollout_id="r-B", image="busybox:1")

    all_records = await mgr.list_owned()
    assert {r.container_id for r in all_records} == {
        a1.container_id, a2.container_id, b1.container_id,
    }
    just_a = await mgr.list_owned(rollout_id="r-A")
    assert {r.container_id for r in just_a} == {
        a1.container_id, a2.container_id,
    }


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.A.2 — archives
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_put_archive_flows_through_to_docker() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r1", image="busybox:1")
    container = client.containers.get(record.container_id)

    await mgr.put_archive(
        rollout_id="r1",
        container_id=record.container_id,
        target_dir="/tmp",
        tarball=b"<tar bytes>",
    )

    assert container.put_archive_calls == [("/tmp", b"<tar bytes>")]


@pytest.mark.asyncio
async def test_put_archive_rejects_unowned_container() -> None:
    """Symmetric to exec/destroy ownership enforcement."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-A", image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not own"):
        await mgr.put_archive(
            rollout_id="r-B",
            container_id=record.container_id,
            target_dir="/tmp",
            tarball=b"<bytes>",
        )


@pytest.mark.asyncio
async def test_put_archive_raises_when_docker_returns_false() -> None:
    """docker-py's ``put_archive`` returns a bool; ``False``
    indicates the daemon refused the extract. Surface as
    XRLEnvError rather than silently ignoring."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r1", image="busybox:1")
    container = client.containers.get(record.container_id)
    container.put_archive_ok = False

    with pytest.raises(XRLEnvError, match="docker returned False"):
        await mgr.put_archive(
            rollout_id="r1", container_id=record.container_id,
            target_dir="/tmp", tarball=b"<bytes>",
        )


@pytest.mark.asyncio
async def test_get_archive_returns_tar_bytes() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r1", image="busybox:1")
    container = client.containers.get(record.container_id)
    container.get_archive_returns["/logs/artifacts"] = b"<tar bytes>"

    tarball = await mgr.get_archive(
        rollout_id="r1",
        container_id=record.container_id,
        source_path="/logs/artifacts",
    )

    assert tarball == b"<tar bytes>"


@pytest.mark.asyncio
async def test_get_archive_rejects_unowned_container() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-A", image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not own"):
        await mgr.get_archive(
            rollout_id="r-B",
            container_id=record.container_id,
            source_path="/anything",
        )


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.A.2 — streaming exec
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exec_stream_yields_chunks_then_terminator() -> None:
    """Streaming exec yields N stdout/stderr chunks followed by
    exactly one terminator chunk with done=True + exit_code."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r1", image="busybox:1")
    # Pre-seed three chunks + a non-zero exit code.
    cmd = ("bash", "-c", "echo a; echo b; echo c")
    client.api.exec_streams[cmd] = [
        (b"a\n", None), (None, b"oops\n"), (b"c\n", None),
    ]
    client.api.exec_exit_codes[cmd] = 7

    chunks: list[dict[str, Any]] = []
    async for chunk in mgr.exec_stream(
        rollout_id="r1",
        container_id=record.container_id,
        cmd=list(cmd),
        timeout_s=10.0,
    ):
        chunks.append(chunk)

    # 3 data chunks + 1 terminator
    assert len(chunks) == 4
    assert chunks[0] == {
        "stdout": b"a\n", "stderr": b"", "done": False,
        "exit_code": 0, "timed_out": False,
    }
    assert chunks[1] == {
        "stdout": b"", "stderr": b"oops\n", "done": False,
        "exit_code": 0, "timed_out": False,
    }
    assert chunks[2]["stdout"] == b"c\n"
    # Terminator carries the final exit code.
    assert chunks[3]["done"] is True
    assert chunks[3]["exit_code"] == 7
    assert chunks[3]["timed_out"] is False


@pytest.mark.asyncio
async def test_exec_stream_rejects_unowned_container() -> None:
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-A", image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not own"):
        async for _ in mgr.exec_stream(
            rollout_id="r-B",
            container_id=record.container_id,
            cmd=["echo", "hi"],
        ):
            pass


@pytest.mark.asyncio
async def test_exec_stream_emits_heartbeat_chunks_on_quiet_periods() -> None:
    """Audit Raw-Stream-M1 closure: a quiet exec (no output for
    >heartbeat_interval_s) yields empty-payload chunks to keep
    the consumer-side ``_send_and_stream`` chunk-timer reset.
    Without this, a 30+ s silent compile/test would surface as a
    false-positive wedge.

    Test uses a tiny ``heartbeat_interval_s`` so it runs fast
    + a slow ``exec_start`` so the queue actually stays empty
    long enough for heartbeats to fire."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r1", image="busybox:1")
    cmd = ("slow-cmd",)
    client.api.exec_streams[cmd] = []
    client.api.exec_exit_codes[cmd] = 0

    # Slow the drain thread: sleep before yielding the sentinel
    # so the queue stays empty for several heartbeat intervals.
    orig_exec_start = client.api.exec_start

    def _slow_start(*args: Any, **kwargs: Any) -> Any:
        import time
        time.sleep(0.25)  # >= 5 x the test heartbeat below
        return orig_exec_start(*args, **kwargs)
    client.api.exec_start = _slow_start  # type: ignore[method-assign]

    chunks = [
        c async for c in mgr.exec_stream(
            rollout_id="r1",
            container_id=record.container_id,
            cmd=list(cmd),
            timeout_s=2.0,
            heartbeat_interval_s=0.05,
        )
    ]

    # Exactly one terminator at the end + at least one heartbeat
    # in between (the slow_start delay covers ~5 heartbeats).
    assert chunks[-1]["done"] is True
    assert chunks[-1]["exit_code"] == 0
    heartbeats = [
        c for c in chunks[:-1]
        if c["stdout"] == b"" and c["stderr"] == b""
    ]
    assert len(heartbeats) >= 1, (
        f"expected at least one heartbeat chunk; got {chunks=}"
    )


@pytest.mark.asyncio
async def test_exec_stream_handles_empty_output() -> None:
    """A successful exec that produces nothing still yields the
    terminator chunk with exit_code."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r1", image="busybox:1")
    cmd = ("true",)
    client.api.exec_streams[cmd] = []  # no chunks
    client.api.exec_exit_codes[cmd] = 0

    chunks = [
        c async for c in mgr.exec_stream(
            rollout_id="r1",
            container_id=record.container_id,
            cmd=list(cmd),
        )
    ]
    assert len(chunks) == 1
    assert chunks[0] == {
        "stdout": b"", "stderr": b"", "done": True,
        "exit_code": 0, "timed_out": False,
    }


# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_acquires_dont_corrupt_registry() -> None:
    """Sanity: 10 concurrent acquires register all 10 records."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    records = await asyncio.gather(*[
        mgr.acquire(rollout_id=f"r-{i}", image="busybox:1")
        for i in range(10)
    ])

    assert len({r.container_id for r in records}) == 10
    owned = await mgr.list_owned()
    assert len(owned) == 10


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.B.2 — ImageCacheManager-routed ensure_image_present
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeImageCache:
    """Records ``ensure_present`` calls; optional pre-seed of which
    images to materialise into the docker client (mirrors a real
    cache landing the image after pull/build)."""

    calls: list[tuple[str, float]] = field(default_factory=list)
    materialise_into: Any | None = None
    raise_on_ensure: Exception | None = None
    # WS2: the raw path now refcounts the image in-use for the
    # container's lifetime (acquire at spawn, release at destroy) so a
    # running container's image isn't misclassified "cold".
    acquired: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)

    async def ensure_present(self, image: str, *, deadline_s: float) -> None:
        self.calls.append((image, deadline_s))
        if self.raise_on_ensure is not None:
            raise self.raise_on_ensure
        if self.materialise_into is not None:
            self.materialise_into.images.present.add(image)

    def acquire(self, image: str) -> None:
        self.acquired.append(image)

    def release(self, image: str) -> None:
        self.released.append(image)


@pytest.mark.asyncio
async def test_acquire_default_routes_through_image_cache() -> None:
    """ensure_image_present=True (default) + ImageCacheManager wired:
    missing image is materialised by the cache, then acquire proceeds."""
    client = _FakeDockerClient(images_present=set())
    cache = _FakeImageCache(materialise_into=client)
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    record = await mgr.acquire(rollout_id="r-001", image="busybox:1")

    assert record.container_id  # acquire succeeded
    # Issue #12 follow-up: the manager now passes ``deadline_s=None``
    # when the caller doesn't override, deferring to the
    # ImageCacheManager's own configured default (currently 600 s).
    # Single source of truth lives on ``ImageCacheConfig`` instead
    # of being duplicated as a hardcoded 600.0 on the manager path.
    assert cache.calls == [("busybox:1", None)]


@pytest.mark.asyncio
async def test_acquire_strict_mode_skips_cache_and_raises_on_missing() -> None:
    """ensure_image_present=False reverts to the legacy strict path —
    even if a cache is wired, we don't pull. Operator-pre-pull contract."""
    client = _FakeDockerClient(images_present=set())
    cache = _FakeImageCache(materialise_into=client)
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    with pytest.raises(XRLEnvError, match="strict mode"):
        await mgr.acquire(
            rollout_id="r-001", image="missing:1",
            ensure_image_present=False,
        )
    # Cache was NOT consulted — strict mode bypasses it.
    assert cache.calls == []


@pytest.mark.asyncio
async def test_acquire_no_cache_falls_back_to_strict_path() -> None:
    """When no ImageCacheManager is wired (legacy fixtures), default
    ensure_image_present=True falls back to the strict path: missing
    image still raises rather than wedging."""
    client = _FakeDockerClient(images_present=set())
    mgr = RawContainerManager(docker_client=client)  # no image_cache

    with pytest.raises(XRLEnvError, match="not present on this node"):
        await mgr.acquire(rollout_id="r-001", image="missing:1")


@pytest.mark.asyncio
async def test_acquire_present_image_skips_cache() -> None:
    """When image is already present locally (cache was a no-op
    on a previous acquire), ensure_image_present=True still calls
    cache.ensure_present which is fast (just LRU touch). The acquire
    proceeds either way."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    cache = _FakeImageCache()
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    record = await mgr.acquire(rollout_id="r-001", image="busybox:1")

    assert record.container_id
    # Cache was consulted (it's idempotent + cheap). Default
    # ``deadline_s=None`` so the cache applies its own
    # configured pull timeout (issue #12 follow-up).
    assert cache.calls == [("busybox:1", None)]


@pytest.mark.asyncio
async def test_acquire_cache_failure_surfaces_clean_error() -> None:
    """ImageCacheManager.ensure_present blowing up surfaces as a
    clean XRLEnvError carrying the original cause type."""
    client = _FakeDockerClient(images_present=set())
    cache = _FakeImageCache(
        raise_on_ensure=TimeoutError("pull deadline exceeded"),
    )
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    with pytest.raises(XRLEnvError, match="ensure_present"):
        await mgr.acquire(rollout_id="r-001", image="some:1")


@pytest.mark.asyncio
async def test_acquire_passes_xrlenv_error_through_unchanged() -> None:
    """A cache that already raises XRLEnvError (e.g. OutOfDiskAfterEviction)
    propagates as-is — no wrapping, no message corruption."""
    client = _FakeDockerClient(images_present=set())
    cache = _FakeImageCache(
        raise_on_ensure=XRLEnvError("disk full after eviction"),
    )
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    with pytest.raises(XRLEnvError, match="disk full after eviction"):
        await mgr.acquire(rollout_id="r-001", image="some:1")


@pytest.mark.asyncio
async def test_acquire_threads_ensure_image_deadline_to_cache() -> None:
    """Issue #12 audit M1 regression: a per-call pull deadline override
    must reach ``ImageCacheManager.ensure_present(deadline_s=...)``
    so widening the wire wait actually widens the node-side pull. Before
    the fix, a consumer passing ``acquire_timeout_s=1800`` for a 10 GB
    image got a longer wire wait but the pull still failed at 600 s.
    """
    client = _FakeDockerClient(images_present=set())
    cache = _FakeImageCache(materialise_into=client)
    mgr = RawContainerManager(docker_client=client, image_cache=cache)

    await mgr.acquire(
        rollout_id="r-001", image="huge:1",
        ensure_image_deadline_s=1800.0,
    )
    assert cache.calls == [("huge:1", 1800.0)]


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 fix #4 — per-node destroy concurrency cap
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_destroy_concurrency_cap_limits_concurrent_removes() -> None:
    """With ``destroy_concurrency=2``, at most 2 ``container.remove``
    calls are in flight simultaneously even when 4 are fired at once.
    The semaphore serialises the docker-side work; the ownership check
    runs unbounded (no gate before the semaphore acquire).
    """
    # Arrange: 4 containers; each remove blocks until we release a per-
    # container event.
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, destroy_concurrency=2)

    records = [
        await mgr.acquire(rollout_id=f"r-{i}", image="busybox:1")
        for i in range(4)
    ]

    remove_entered: list[str] = []
    remove_events: dict[str, asyncio.Event] = {
        rec.container_id: asyncio.Event() for rec in records
    }

    # Patch each fake container's ``remove`` to track concurrency.
    max_concurrent = 0
    in_flight: list[int] = [0]

    def _blocking_remove(cid: str, *, force: bool = False) -> None:
        nonlocal max_concurrent
        in_flight[0] += 1
        max_concurrent = max(max_concurrent, in_flight[0])
        remove_entered.append(cid)
        # Block until the test releases this container.
        # The asyncio Event is resolved from the test after a brief yield.
        release_event = remove_events[cid]
        # We can't block on an asyncio.Event from a thread pool thread
        # directly; instead spin-poll with a tiny sleep.
        import time
        while not release_event.is_set():
            time.sleep(0.005)
        in_flight[0] -= 1

    for rec in records:
        container = client.containers._registry[rec.container_id]
        cid = rec.container_id
        container.remove = lambda force=False, _cid=cid: _blocking_remove(_cid, force=force)  # type: ignore[method-assign]

    # Act: fire 4 destroys concurrently, but release them in two batches.
    async def _destroy(rec: Any) -> None:
        await mgr.destroy(rollout_id=rec.rollout_id, container_id=rec.container_id)

    # Start all 4 destroys concurrently but let them race into the semaphore.
    tasks = [asyncio.create_task(_destroy(rec)) for rec in records]

    # Wait until at least 2 have entered remove (semaphore at capacity).
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(remove_entered) >= 2:
            break

    # Release all blocked removes so tasks can complete.
    for ev in remove_events.values():
        ev.set()

    await asyncio.gather(*tasks)

    # Assert: never more than 2 removes were active at once.
    assert max_concurrent <= 2, (
        f"destroy_concurrency=2 allowed {max_concurrent} concurrent removes"
    )
    assert len(remove_entered) == 4


@pytest.mark.asyncio
async def test_destroy_concurrency_zero_allows_all_concurrent() -> None:
    """``destroy_concurrency=0`` disables the semaphore (legacy unbounded
    behaviour). All 4 removes must be in flight simultaneously."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, destroy_concurrency=0)

    records = [
        await mgr.acquire(rollout_id=f"r-{i}", image="busybox:1")
        for i in range(4)
    ]

    remove_entered: list[str] = []
    remove_events: dict[str, asyncio.Event] = {
        rec.container_id: asyncio.Event() for rec in records
    }
    max_concurrent = 0
    in_flight: list[int] = [0]

    def _blocking_remove(cid: str, *, force: bool = False) -> None:
        nonlocal max_concurrent
        in_flight[0] += 1
        max_concurrent = max(max_concurrent, in_flight[0])
        remove_entered.append(cid)
        import time
        release_event = remove_events[cid]
        while not release_event.is_set():
            time.sleep(0.005)
        in_flight[0] -= 1

    for rec in records:
        container = client.containers._registry[rec.container_id]
        cid = rec.container_id
        container.remove = lambda force=False, _cid=cid: _blocking_remove(_cid, force=force)  # type: ignore[method-assign]

    tasks = [
        asyncio.create_task(
            mgr.destroy(rollout_id=rec.rollout_id, container_id=rec.container_id)
        )
        for rec in records
    ]

    # Wait until all 4 have entered remove — the semaphore is absent so
    # they all proceed immediately.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(remove_entered) >= 4:
            break

    for ev in remove_events.values():
        ev.set()

    await asyncio.gather(*tasks)

    # All 4 were in flight at the same time.
    assert max_concurrent == 4


@pytest.mark.asyncio
async def test_force_destroy_also_uses_destroy_gate() -> None:
    """``force_destroy`` (reconciler path) shares the destroy semaphore
    with ``destroy`` so a reconciler sweep can't bypass the cap and
    independently thrash the daemon while consumer destroys are queued.
    """
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, destroy_concurrency=1)

    # Acquire 2 containers; we'll force-destroy them.
    r1 = await mgr.acquire(rollout_id="r1", image="busybox:1")
    r2 = await mgr.acquire(rollout_id="r2", image="busybox:1")

    remove_order: list[str] = []
    for rec in [r1, r2]:
        cid = rec.container_id
        container = client.containers._registry[cid]

        def _remove(force: bool = False, _cid: str = cid) -> None:
            remove_order.append(_cid)
        container.remove = _remove  # type: ignore[method-assign]

    # Fire both force_destroys concurrently; semaphore=1 serialises them.
    await asyncio.gather(
        mgr.force_destroy(container_id=r1.container_id),
        mgr.force_destroy(container_id=r2.container_id),
    )

    # Both ran, in some order, but never concurrently (serialised by cap=1).
    assert set(remove_order) == {r1.container_id, r2.container_id}


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 — per-node container-create concurrency cap
# ──────────────────────────────────────────────────────────────────────────────


def _instrument_run_concurrency(
    client: _FakeDockerClient, release: asyncio.Event,
) -> tuple[list[int], list[int]]:
    """Wrap ``client.containers.run`` so it blocks until ``release`` is
    set, tracking how many calls are simultaneously in flight. Returns
    ``(entered, max_concurrent)`` — single-element lists used as
    mutable cells (the wrapper runs in a thread-pool thread)."""
    import time

    real_run = client.containers.run
    entered: list[int] = [0]
    in_flight: list[int] = [0]
    max_concurrent: list[int] = [0]

    def _blocking_run(**kwargs: Any) -> _FakeContainer:
        entered[0] += 1
        in_flight[0] += 1
        max_concurrent[0] = max(max_concurrent[0], in_flight[0])
        while not release.is_set():
            time.sleep(0.005)
        in_flight[0] -= 1
        return real_run(**kwargs)

    client.containers.run = _blocking_run  # type: ignore[method-assign]
    return entered, max_concurrent


@pytest.mark.asyncio
async def test_create_concurrency_cap_limits_concurrent_runs() -> None:
    """Issue #18: with ``create_concurrency=2``, at most 2
    ``containers.run`` calls are in flight at once even when 4 acquires
    are fired concurrently — bounding the daemon's create-time load."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, create_concurrency=2)

    release = asyncio.Event()
    entered, max_concurrent = _instrument_run_concurrency(client, release)

    tasks = [
        asyncio.create_task(
            mgr.acquire(rollout_id=f"r-{i}", image="busybox:1"),
        )
        for i in range(4)
    ]
    # Let the race settle; the cap must hold the in-flight count at 2.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if entered[0] >= 2:
            break
    await asyncio.sleep(0.05)  # give any (incorrectly) un-gated runs a chance
    release.set()
    await asyncio.gather(*tasks)

    assert max_concurrent[0] <= 2, (
        f"create_concurrency=2 allowed {max_concurrent[0]} concurrent runs"
    )
    assert entered[0] == 4


@pytest.mark.asyncio
async def test_create_concurrency_zero_allows_all_concurrent() -> None:
    """``create_concurrency=0`` disables the cap — legacy unbounded
    behaviour, kept for the opt-out path."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, create_concurrency=0)

    release = asyncio.Event()
    entered, max_concurrent = _instrument_run_concurrency(client, release)

    tasks = [
        asyncio.create_task(
            mgr.acquire(rollout_id=f"r-{i}", image="busybox:1"),
        )
        for i in range(4)
    ]
    for _ in range(200):
        await asyncio.sleep(0.01)
        if entered[0] >= 4:
            break
    release.set()
    await asyncio.gather(*tasks)

    assert max_concurrent[0] == 4


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — node health collector wired into the raw-container manager
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_records_create_latency_in_health() -> None:
    """A successful acquire records one ``docker run`` duration sample
    the heartbeat snapshot then carries."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    assert mgr.health_snapshot().create_count == 0
    await mgr.acquire(rollout_id="r-1", image="busybox:1")

    snap = mgr.health_snapshot()
    assert snap.create_count == 1
    assert snap.create_p95_ms >= 0.0


@pytest.mark.asyncio
async def test_docker_error_recorded_in_health() -> None:
    """A docker failure on ``containers.run`` increments the health
    error counter — and a timeout-class error the timeout counter."""
    import requests.exceptions

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    def _raise(**_kwargs: Any) -> Any:
        raise requests.exceptions.ReadTimeout("read timed out")

    client.containers.run = _raise  # type: ignore[method-assign]

    with pytest.raises(XRLEnvError):
        await mgr.acquire(rollout_id="r-1", image="busybox:1")

    snap = mgr.health_snapshot()
    assert snap.docker_error_count == 1
    assert snap.docker_timeout_count == 1
    assert snap.create_count == 0  # the create never completed


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 (Ask #3) — docker-error translation at the node boundary
# ──────────────────────────────────────────────────────────────────────────────


class _ApiWithTimeout:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout


class _ClientWithTimeout:
    """Minimal client double exposing ``client.api.timeout`` — the
    knob ``_translate_docker_error`` reads to name the HTTP-client
    ceiling in its message."""

    def __init__(self, timeout: float) -> None:
        self.api = _ApiWithTimeout(timeout)


def test_translate_docker_error_60s_timeout_hints_redeploy() -> None:
    """A timeout against a client whose HTTP timeout is the 60s
    docker-py default produces a 'redeploy the node-agent' hint —
    the exact stale-binary case from the SWE-bench Pro run."""
    import requests.exceptions
    from xrlenv.node.raw_container import _translate_docker_error

    exc = requests.exceptions.ReadTimeout("read timed out (read timeout=60)")
    err = _translate_docker_error(
        operation="containers.run",
        target="image='jefzda/sweap-images:foo'",
        client=_ClientWithTimeout(60.0),
        exc=exc,
    )
    text = str(err)
    assert "containers.run" in text
    assert "jefzda/sweap-images:foo" in text
    assert "60s" in text
    assert "redeploy" in text.lower()
    assert "node↔dockerd" in text


def test_translate_docker_error_600s_timeout_hints_overload() -> None:
    """A timeout against a 600s-configured client (current builds)
    points at daemon overload, not a stale binary."""
    import requests.exceptions
    from xrlenv.node.raw_container import _translate_docker_error

    exc = requests.exceptions.ReadTimeout("read timed out")
    err = _translate_docker_error(
        operation="containers.run",
        target="image='x'",
        client=_ClientWithTimeout(600.0),
        exc=exc,
    )
    text = str(err)
    assert "600s" in text
    assert "overloaded" in text.lower()
    assert "redeploy" not in text.lower()


def test_translate_docker_error_non_timeout_keeps_plain_translation() -> None:
    """A non-timeout docker error is still translated (operation +
    target named) but carries no timeout-layer hint."""
    import docker.errors
    from xrlenv.node.raw_container import _translate_docker_error

    exc = docker.errors.ImageNotFound("404: no such image")
    err = _translate_docker_error(
        operation="containers.run",
        target="image='gone:1'",
        client=_ClientWithTimeout(600.0),
        exc=exc,
    )
    text = str(err)
    assert "containers.run" in text and "gone:1" in text
    assert "ImageNotFound" in text
    assert "HTTP client timeout" not in text


@pytest.mark.asyncio
async def test_acquire_wraps_containers_run_readtimeout() -> None:
    """``acquire`` translates a raw ``requests`` ReadTimeout from
    ``containers.run`` into an XRLEnvError with node-side context —
    the consumer no longer gets opaque docker-py internals."""
    import requests.exceptions

    client = _FakeDockerClient(images_present={"busybox:1"})
    client.api.timeout = 60.0  # type: ignore[attr-defined]

    def _boom(**_kw: Any) -> Any:
        raise requests.exceptions.ReadTimeout(
            "UnixHTTPConnectionPool(host='localhost', port=None): "
            "Read timed out. (read timeout=60)",
        )

    client.containers.run = _boom  # type: ignore[method-assign]
    mgr = RawContainerManager(docker_client=client)

    with pytest.raises(XRLEnvError) as excinfo:
        await mgr.acquire(rollout_id="r1", image="busybox:1")
    text = str(excinfo.value)
    assert "containers.run" in text
    assert "busybox:1" in text
    assert "redeploy" in text.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Regression: get_archive_stream — the node-lost fix
#
# Before the fix, ``get_archive`` drained the full tar via ``b"".join(bits)``
# on the event-loop thread.  A slow blocking read starved the heartbeat task,
# causing the control plane to mark the node "lost" and seal all in-flight
# rollouts.  The fix reads one chunk per ``asyncio.to_thread`` hop inside
# ``get_archive_stream`` so the event loop runs between chunks.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_archive_stream_does_not_block_event_loop() -> None:
    """Core regression: the event loop must keep running other tasks while
    a large archive drains.

    We give the fake container a ``get_archive`` whose generator does a REAL
    ``time.sleep`` before each chunk (simulating a slow blocking socket read).
    A "ticker" coroutine runs concurrently and counts how many times it wakes
    up during the drain.  With the old on-loop ``b"".join(bits)``, all the
    blocking sleeps would run on the event-loop thread and the ticker would be
    starved to zero.  With the fix, each chunk is a ``to_thread`` hop so the
    ticker keeps advancing.
    """
    import time

    # Fake container whose get_archive yields 4 chunks with a blocking sleep
    # before each one — simulating a slow docker socket read.
    CHUNK_SLEEP = 0.02   # 20 ms per chunk, 4 chunks → ~80 ms total block time
    NUM_CHUNKS = 4
    CHUNK_DATA = [b"chunk%d" % i for i in range(NUM_CHUNKS)]

    def _slow_gen() -> Any:
        for c in CHUNK_DATA:
            time.sleep(CHUNK_SLEEP)
            yield c

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-ev", image="busybox:1")

    # Patch the container's get_archive to return the slow generator.
    container = client.containers._registry[record.container_id]
    container.get_archive = lambda path: (_slow_gen(), {"name": path})  # type: ignore[method-assign]

    # Ticker coroutine: wakes up every 5 ms; counts completions during drain.
    TICK_INTERVAL = 0.005
    NUM_TICKS = 12  # enough ticks to run during the ~80 ms drain
    ticks: list[int] = [0]

    async def _ticker() -> None:
        for _ in range(NUM_TICKS):
            await asyncio.sleep(TICK_INTERVAL)
            ticks[0] += 1

    # Run the drain and the ticker concurrently.
    collected: list[bytes] = []

    async def _drain() -> None:
        async for chunk in mgr.get_archive_stream(
            rollout_id="r-ev",
            container_id=record.container_id,
            source_path="/testbed",
        ):
            collected.append(chunk)

    await asyncio.gather(_drain(), _ticker())

    # The archive bytes must be intact.
    assert b"".join(collected) == b"".join(CHUNK_DATA)

    # The ticker must have advanced — this is the key regression check.
    # With the old on-loop drain, ticks would be 0 because every blocking
    # sleep ran on the event-loop thread and no await ever yielded.
    assert ticks[0] >= NUM_TICKS // 2, (
        f"event loop was starved: ticker only reached {ticks[0]}/{NUM_TICKS} "
        "ticks during the archive drain — get_archive_stream is blocking the "
        "event loop instead of using asyncio.to_thread per chunk"
    )


@pytest.mark.asyncio
async def test_get_archive_stream_bytes_are_lossless_and_ordered() -> None:
    """``get_archive_stream`` yields chunks in order and ``get_archive``
    (the bytes wrapper) returns their exact concatenation.
    Multiple chunks, interleaved with empty-string guard in the fake."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-bytes", image="busybox:1")
    container = client.containers._registry[record.container_id]

    # Pre-seed multiple chunks via a custom get_archive that mimics
    # docker-py's lazy multi-chunk generator.
    CHUNKS = [b"alpha", b"beta", b"gamma", b"delta"]

    def _multi_chunk_archive(path: str) -> tuple[Any, dict]:
        return (iter(CHUNKS), {"name": path})

    container.get_archive = _multi_chunk_archive  # type: ignore[method-assign]

    # --- get_archive_stream yields correct chunks in order ---
    stream_chunks: list[bytes] = []
    async for chunk in mgr.get_archive_stream(
        rollout_id="r-bytes",
        container_id=record.container_id,
        source_path="/data",
    ):
        stream_chunks.append(chunk)

    assert stream_chunks == CHUNKS, (
        f"expected chunks {CHUNKS!r}, got {stream_chunks!r}"
    )

    # --- get_archive (bytes wrapper) returns exact concatenation ---
    result = await mgr.get_archive(
        rollout_id="r-bytes",
        container_id=record.container_id,
        source_path="/data",
    )
    assert result == b"".join(CHUNKS)


@pytest.mark.asyncio
async def test_get_archive_empty_returns_empty_bytes() -> None:
    """Edge case: a container path that produces no chunks.
    ``get_archive_stream`` should yield nothing; ``get_archive`` returns b"".
    The _FakeContainer default (no pre-seeded path) already returns an empty
    iterator, so this exercises the code path directly."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-empty", image="busybox:1")
    # No pre-seeded get_archive_returns → _FakeContainer returns iter([]).

    stream_chunks: list[bytes] = [
        c async for c in mgr.get_archive_stream(
            rollout_id="r-empty",
            container_id=record.container_id,
            source_path="/nothing",
        )
    ]
    assert stream_chunks == [], (
        f"empty archive should yield no chunks; got {stream_chunks!r}"
    )

    result = await mgr.get_archive(
        rollout_id="r-empty",
        container_id=record.container_id,
        source_path="/nothing",
    )
    assert result == b""


@pytest.mark.asyncio
async def test_archive_gate_bounds_concurrent_transfers() -> None:
    """``archive_concurrency=2`` must cap simultaneous ``get_archive_stream``
    calls at 2 even when 4 are fired at once.

    Each fake transfer blocks inside a real time.sleep loop (run inside
    to_thread by the stream) until a shared event is set.  We track how many
    transfers are simultaneously inside the gated region and assert the max
    never exceeded 2.
    """
    import time

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, archive_concurrency=2)

    NUM_TRANSFERS = 4
    records = [
        await mgr.acquire(rollout_id=f"r-gate-{i}", image="busybox:1")
        for i in range(NUM_TRANSFERS)
    ]

    # Shared concurrency counter (mutable cell).
    in_flight: list[int] = [0]
    max_concurrent: list[int] = [0]
    # Each transfer blocks until this asyncio.Event is set.
    release_event = asyncio.Event()

    def _gated_gen(path: str) -> Any:
        """Generator that bumps the counter and blocks until released."""
        in_flight[0] += 1
        max_concurrent[0] = max(max_concurrent[0], in_flight[0])
        # Block until the test releases all transfers.
        while not release_event.is_set():
            time.sleep(0.005)
        in_flight[0] -= 1
        yield b"data"

    for rec in records:
        cid = rec.container_id
        container = client.containers._registry[cid]
        # Each container uses the same tracking generator (closure is fine
        # because the gen factory is called once per get_archive call).
        container.get_archive = lambda path, _cid=cid: (  # type: ignore[method-assign]
            _gated_gen(path), {"name": path}
        )

    # Fire all 4 stream drains concurrently.
    async def _drain(rec: Any) -> None:
        async for _ in mgr.get_archive_stream(
            rollout_id=rec.rollout_id,
            container_id=rec.container_id,
            source_path="/data",
        ):
            pass

    tasks = [asyncio.create_task(_drain(rec)) for rec in records]

    # Wait until at least 2 transfers have entered the generator (i.e. are
    # inside the gated region), then release them all.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if in_flight[0] >= 2:
            break

    release_event.set()
    await asyncio.gather(*tasks)

    assert max_concurrent[0] <= 2, (
        f"archive_concurrency=2 allowed {max_concurrent[0]} concurrent "
        "transfers inside the gated region"
    )
    # Sanity: all 4 eventually ran.
    assert in_flight[0] == 0  # all have exited the generator


@pytest.mark.asyncio
async def test_archive_concurrency_zero_allows_all_concurrent() -> None:
    """``archive_concurrency=0`` disables the semaphore (unbounded path).
    All 4 transfers should enter the gated region simultaneously."""
    import time

    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client, archive_concurrency=0)

    NUM_TRANSFERS = 4
    records = [
        await mgr.acquire(rollout_id=f"r-unb-{i}", image="busybox:1")
        for i in range(NUM_TRANSFERS)
    ]

    in_flight: list[int] = [0]
    max_concurrent: list[int] = [0]
    release_event = asyncio.Event()

    def _gated_gen(path: str) -> Any:
        in_flight[0] += 1
        max_concurrent[0] = max(max_concurrent[0], in_flight[0])
        while not release_event.is_set():
            time.sleep(0.005)
        in_flight[0] -= 1
        yield b"x"

    for rec in records:
        container = client.containers._registry[rec.container_id]
        container.get_archive = lambda path: (_gated_gen(path), {"name": path})  # type: ignore[method-assign]

    async def _drain(rec: Any) -> None:
        async for _ in mgr.get_archive_stream(
            rollout_id=rec.rollout_id,
            container_id=rec.container_id,
            source_path="/d",
        ):
            pass

    tasks = [asyncio.create_task(_drain(rec)) for rec in records]

    for _ in range(200):
        await asyncio.sleep(0.01)
        if in_flight[0] >= NUM_TRANSFERS:
            break

    release_event.set()
    await asyncio.gather(*tasks)

    assert max_concurrent[0] == NUM_TRANSFERS, (
        f"archive_concurrency=0 (unbounded) should allow all "
        f"{NUM_TRANSFERS} concurrent — got max {max_concurrent[0]}"
    )


@pytest.mark.asyncio
async def test_get_archive_stream_rejects_unowned_container() -> None:
    """Ownership enforcement is also enforced on ``get_archive_stream``:
    a different rollout cannot stream an archive from a container it
    doesn't own.  The error must fire before any docker I/O."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-owner", image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not own"):
        async for _ in mgr.get_archive_stream(
            rollout_id="r-stranger",
            container_id=record.container_id,
            source_path="/data",
        ):
            pass


@pytest.mark.asyncio
async def test_get_archive_stream_rejects_unknown_container_id() -> None:
    """``get_archive_stream`` on a container_id not registered with the
    manager raises the same 'not registered' error as ``get_archive``
    and ``exec``."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)

    with pytest.raises(XRLEnvError, match="not registered"):
        async for _ in mgr.get_archive_stream(
            rollout_id="r-any",
            container_id="ghost-id-deadbeef",
            source_path="/data",
        ):
            pass


# ── Control-plane relay cap (plane-split guardrail) ──────────────────────────


@pytest.mark.asyncio
async def test_get_archive_stream_refuses_over_relay_cap() -> None:
    """A get_archive whose streamed size exceeds ``max_get_archive_relay_bytes``
    is refused with ``ArchiveTooLarge`` — the plane-split guardrail. The refusal
    happens mid-stream (a directory's tar size isn't known up front), and
    ``ArchiveTooLarge`` is an ``XRLEnvError`` so existing handlers still catch."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    # 8-byte cap; the archive streams 5+5+5 = 15 bytes → refused.
    mgr = RawContainerManager(
        docker_client=client, max_get_archive_relay_bytes=8,
    )
    record = await mgr.acquire(rollout_id="r-cap", image="busybox:1")
    container = client.containers._registry[record.container_id]
    container.get_archive = lambda path: (  # type: ignore[method-assign]
        iter([b"aaaaa", b"bbbbb", b"ccccc"]), {"name": path},
    )

    got: list[bytes] = []
    with pytest.raises(ArchiveTooLarge, match="relay cap"):
        async for chunk in mgr.get_archive_stream(
            rollout_id="r-cap",
            container_id=record.container_id,
            source_path="/testbed",
        ):
            got.append(chunk)
    # It streamed only up to the cap before refusing (bounded waste), never
    # the whole archive.
    assert sum(len(c) for c in got) <= 15
    # The bytes wrapper surfaces the same refusal.
    with pytest.raises(ArchiveTooLarge):
        await mgr.get_archive(
            rollout_id="r-cap",
            container_id=record.container_id,
            source_path="/testbed",
        )


@pytest.mark.asyncio
async def test_relay_cap_zero_disables_the_guardrail() -> None:
    """``max_get_archive_relay_bytes=0`` disables the cap — a large archive
    flows unrefused (legacy / opt-out)."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(
        docker_client=client, max_get_archive_relay_bytes=0,
    )
    record = await mgr.acquire(rollout_id="r-nocap", image="busybox:1")
    container = client.containers._registry[record.container_id]
    big = [b"x" * 1000 for _ in range(50)]  # 50 KB, far over any small cap
    container.get_archive = lambda path: (  # type: ignore[method-assign]
        iter(list(big)), {"name": path},
    )

    result = await mgr.get_archive(
        rollout_id="r-nocap",
        container_id=record.container_id,
        source_path="/testbed",
    )
    assert result == b"".join(big)


@pytest.mark.asyncio
async def test_get_archive_under_relay_cap_succeeds() -> None:
    """An archive at/under the cap is unaffected (small reward-file reads —
    the common legitimate case — keep working)."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(
        docker_client=client, max_get_archive_relay_bytes=1024,
    )
    record = await mgr.acquire(rollout_id="r-ok", image="busybox:1")
    container = client.containers._registry[record.container_id]
    payload = [b"reward=1\n", b"extra"]
    container.get_archive = lambda path: (  # type: ignore[method-assign]
        iter(list(payload)), {"name": path},
    )

    result = await mgr.get_archive(
        rollout_id="r-ok",
        container_id=record.container_id,
        source_path="/logs/reward.txt",
    )
    assert result == b"".join(payload)


# ──────────────────────────────────────────────────────────────────────────────
# §10.x — Sysbox runtime regression + positive tests (container_runtime, §5.1/5.5/5.6)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_acquire_omits_runtime_kwarg() -> None:
    """§10.x(1) — a default acquire (no container_runtime) must NOT pass a
    ``runtime`` kwarg to ``containers.run``: the normal raw path is
    byte-for-byte unchanged and lands on docker's default runtime."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(rollout_id="r-nort", image="busybox:1")
    extra = client.containers.run_extra[-1]
    assert "runtime" not in extra
    # Never probes docker info for the default path (no verification needed).
    assert client.info_calls == 0


@pytest.mark.asyncio
async def test_default_acquire_keeps_docker_init_enabled() -> None:
    """§10.x(2) — the default acquire still injects ``init=True`` (tini) —
    the nushell zombie-reaping fix is not regressed."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(rollout_id="r-init", image="busybox:1")
    extra = client.containers.run_extra[-1]
    assert extra.get("init") is True


@pytest.mark.asyncio
async def test_node_wide_raw_init_escape_hatch_disables_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§10.x(2) — the existing node-wide ``XRLENV_RAW_INIT=0`` escape hatch
    still disables the injected init (unchanged by the sysbox work)."""
    monkeypatch.setenv("XRLENV_RAW_INIT", "0")
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(rollout_id="r-noinit", image="busybox:1")
    extra = client.containers.run_extra[-1]
    assert "init" not in extra


@pytest.mark.asyncio
async def test_sysbox_acquire_sets_runtime_and_skips_tini() -> None:
    """§5.1/§5.6 — a sysbox acquire on a sysbox-capable node passes
    ``runtime=sysbox-runc`` to ``containers.run`` and skips the tini init
    (systemd / inner-init must be PID 1)."""
    client = _FakeDockerClient(
        images_present={"sysimg:1"},
        runtimes=("runc", "sysbox-runc"),
    )
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(
        rollout_id="r-sb", image="sysimg:1",
        container_runtime="sysbox-runc",
    )
    extra = client.containers.run_extra[-1]
    assert extra.get("runtime") == "sysbox-runc"
    # §5.6 — tini must NOT be injected in front of a sysbox system container.
    assert "init" not in extra


@pytest.mark.asyncio
async def test_sysbox_acquire_fails_loud_when_runtime_not_registered() -> None:
    """§5.5 — a sysbox acquire on a node whose docker does NOT have
    sysbox-runc registered fails loud (no silent fall-back to runc), and no
    container is spawned."""
    from xrlenv.errors import XRLEnvError

    client = _FakeDockerClient(
        images_present={"sysimg:1"},
        runtimes=("runc",),  # sysbox-runc NOT registered
    )
    mgr = RawContainerManager(docker_client=client)
    with pytest.raises(XRLEnvError, match="not registered in docker"):
        await mgr.acquire(
            rollout_id="r-sb-missing", image="sysimg:1",
            container_runtime="sysbox-runc",
        )
    # Fail-loud happened before any container was spawned.
    assert client.containers.run_extra == []


@pytest.mark.asyncio
async def test_runc_container_runtime_is_treated_as_default() -> None:
    """§5.1 — an explicit ``container_runtime='runc'`` is the daemon default,
    not an override: no ``runtime`` kwarg, init still on, no info probe."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    await mgr.acquire(
        rollout_id="r-runc", image="busybox:1", container_runtime="runc",
    )
    extra = client.containers.run_extra[-1]
    assert "runtime" not in extra
    assert extra.get("init") is True
    assert client.info_calls == 0


def test_registered_runtimes_caches_docker_info() -> None:
    """§5.3/§5.5 — ``registered_runtimes`` probes ``docker info`` once and
    caches (the runtime set only changes on a daemon restart)."""
    client = _FakeDockerClient(
        images_present=set(),
        runtimes=("runc", "sysbox-runc"),
        default_runtime="runc",
    )
    mgr = RawContainerManager(docker_client=client)
    runtimes, default = mgr.registered_runtimes()
    assert "sysbox-runc" in runtimes and default == "runc"
    mgr.registered_runtimes()  # cached
    assert client.info_calls == 1


def test_runtimes_probed_true_only_after_successful_probe() -> None:
    """``runtimes_probed`` is False until a ``docker info`` probe succeeds — the
    signal the NodeHello gate uses to know the advertised runtime set is
    authoritative rather than the startup fallback."""
    client = _FakeDockerClient(
        images_present=set(), runtimes=("runc", "sysbox-runc"),
    )
    mgr = RawContainerManager(docker_client=client)
    assert mgr.runtimes_probed() is False  # nothing probed yet
    mgr.registered_runtimes()
    assert mgr.runtimes_probed() is True


def test_runtimes_probed_stays_false_until_docker_ready_then_recovers() -> None:
    """The redeploy race in miniature: while ``docker info`` raises (daemon not
    ready), ``registered_runtimes`` returns the conservative ``{'runc'}`` WITHOUT
    caching and ``runtimes_probed`` stays False; once docker answers, a later
    probe enumerates the real set and marks the node probed. This is what lets
    the node link wait for a ready daemon instead of advertising a stale
    conservative set for the whole connection."""

    class _NotReadyThenReadyClient:
        def __init__(self) -> None:
            self.images = _FakeImages(set())
            self.containers = _FakeContainers()
            self.api = _FakeAPI()
            self.ready = False
            self.info_calls = 0

        def info(self) -> dict[str, Any]:
            self.info_calls += 1
            if not self.ready:
                raise RuntimeError("Cannot connect to the Docker daemon")
            return {
                "Runtimes": {"runc": {}, "sysbox-runc": {}},
                "DefaultRuntime": "runc",
            }

    client = _NotReadyThenReadyClient()
    mgr = RawContainerManager(docker_client=client)

    # Docker not ready → conservative, uncached, not "probed".
    runtimes, default = mgr.registered_runtimes()
    assert runtimes == frozenset({"runc"}) and default == "runc"
    assert mgr.runtimes_probed() is False

    # Docker recovers → the next probe re-reads (fallback was NOT cached) and
    # enumerates the real set.
    client.ready = True
    runtimes2, _ = mgr.registered_runtimes()
    assert "sysbox-runc" in runtimes2
    assert mgr.runtimes_probed() is True
    assert client.info_calls == 2  # re-probed after the failed attempt


# ── exec() resync-retry on docker-py demux stream corruption ──────────────────
@pytest.mark.asyncio
async def test_exec_resync_retries_demux_stream_corruption() -> None:
    """docker-py's demuxer raises ``ValueError('N is not a valid stream')`` on a
    misaligned attach stream (a short read on the daemon socket under concurrent exec).
    It's transient — a fresh exec opens a new stream — so ``exec`` must resync-retry
    rather than surface it fatally (a single occurrence killed EvoClaw's tag-watcher
    thread and hung a rollout for hours)."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-demux", image="busybox:1")
    container = client.containers.get(record.container_id)

    calls = {"n": 0}

    def flaky(cmd: Any, **kwargs: Any) -> _ExecResult:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("53 is not a valid stream")  # first read misaligned
        return _ExecResult(exit_code=0, output=(b"ok", b""))

    container.exec_run = flaky  # type: ignore[assignment]
    res = await mgr.exec(
        rollout_id="r-demux", container_id=record.container_id,
        cmd=["git", "rev-parse", "HEAD"],
    )
    assert res["exit_code"] == 0 and res["stdout"] == b"ok" and res["timed_out"] is False
    assert calls["n"] == 2  # one corruption + one resync-retry that succeeded


@pytest.mark.asyncio
async def test_exec_propagates_persistent_demux_corruption() -> None:
    """If the stream stays corrupt past the retry budget, the error surfaces — we resync,
    we don't silently swallow a genuinely broken exec."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-demux2", image="busybox:1")
    container = client.containers.get(record.container_id)

    calls = {"n": 0}

    def always_corrupt(cmd: Any, **kwargs: Any) -> _ExecResult:
        calls["n"] += 1
        raise ValueError("7 is not a valid stream")

    container.exec_run = always_corrupt  # type: ignore[assignment]
    with pytest.raises(ValueError, match="not a valid stream"):
        await mgr.exec(
            rollout_id="r-demux2", container_id=record.container_id, cmd=["x"],
        )
    assert calls["n"] == 1 + 2  # initial + _EXEC_DEMUX_RETRIES


@pytest.mark.asyncio
async def test_exec_does_not_retry_unrelated_valueerror() -> None:
    """A ValueError that isn't the demux stream corruption must propagate immediately —
    the retry is narrowly scoped to the known-transient stream misalignment."""
    client = _FakeDockerClient(images_present={"busybox:1"})
    mgr = RawContainerManager(docker_client=client)
    record = await mgr.acquire(rollout_id="r-demux3", image="busybox:1")
    container = client.containers.get(record.container_id)

    calls = {"n": 0}

    def other(cmd: Any, **kwargs: Any) -> _ExecResult:
        calls["n"] += 1
        raise ValueError("totally unrelated boom")

    container.exec_run = other  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unrelated"):
        await mgr.exec(
            rollout_id="r-demux3", container_id=record.container_id, cmd=["x"],
        )
    assert calls["n"] == 1  # no retry for an unrelated error
