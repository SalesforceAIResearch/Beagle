"""P1.7.A.1 — Tests for ``RawContainerCoordinator``.

Exercises the control-plane fan-out:

- ``acquire`` picks a node, dispatches to its NodeTransport,
  records the session, returns a ``RawContainerSession``.
- ``exec`` looks up the session, validates ``container_id`` matches
  (defends against stale handles), routes to the same node.
- ``destroy`` looks up, dispatches, drops the session.
- Empty / docker-less node-pools surface clean errors.

Uses a fake ``NodeTransport`` so the tests don't need a live
gRPC server. The transport's surface mirrors what
``RawContainerCoordinator`` actually calls, which is a small
subset of the full ``NodeTransport`` Protocol.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.raw_container_service import (
    RAW_SESSION_DEADLINE_DEFAULT_S,
    RawContainerCoordinator,
    RawContainerSession,
)
from xrlenv.errors import CapacityExhausted, XRLEnvError
from xrlenv.types import TerminateRawGroupReport

# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeRecord:
    rollout_id: str
    container_id: str
    container_name: str
    image: str


@dataclass
class _FakeNodeTransport:
    """Minimal NodeTransport stand-in. Captures calls so tests can
    assert which node received which RPC + with what args."""

    node_id: str = "node-A"
    backends: list[str] = field(default_factory=lambda: ["docker"])
    acquire_calls: list[dict] = field(default_factory=list)
    exec_calls: list[dict] = field(default_factory=list)
    destroy_calls: list[dict] = field(default_factory=list)
    next_container_id: str = "container-001"
    exec_result: dict[str, Any] = field(default_factory=lambda: {
        "exit_code": 0, "stdout": b"hi\n", "stderr": b"", "timed_out": False,
    })
    raise_on_acquire: Exception | None = None
    raise_on_exec: Exception | None = None

    has_image: bool = True
    """P1.7.B.2: scheduler runs ``query_image`` per placement and
    pre-flight; default True so existing tests pass without
    additional setup. Override for image-absence tests."""

    def supported_backends(self) -> list[str]:
        return list(self.backends)

    def hardware(self) -> Any:
        """Surface for the real ``Scheduler``'s capacity probe. Returns
        a generously-sized box so resource math never gates a test
        that's exercising another concern (e.g. ``max_runs_per_task``).
        Lazily built so tests that don't touch the real Scheduler
        don't pay the ``xrlenv.node.hw_probe`` import cost."""
        from xrlenv.node.hw_probe import HardwareInfo
        return HardwareInfo(
            vcpus=64, mem_bytes=128 * 1024**3, disk_bytes=2000 * 1024**3,
            has_kvm=True, has_gpu=False, gpu_model=None,
            kernel_version="6.0.0", platform="linux",
        )

    async def query_image(self, image: str) -> _FakeQueryImageReply:
        return _FakeQueryImageReply(present=self.has_image)

    async def acquire_container(self, **kwargs: Any) -> _FakeRecord:
        if self.raise_on_acquire:
            raise self.raise_on_acquire
        self.acquire_calls.append(kwargs)
        return _FakeRecord(
            rollout_id=kwargs["rollout_id"],
            container_id=self.next_container_id,
            container_name=f"name-of-{self.next_container_id}",
            image=kwargs["image"],
        )

    async def container_exec(self, **kwargs: Any) -> dict[str, Any]:
        if self.raise_on_exec:
            raise self.raise_on_exec
        self.exec_calls.append(kwargs)
        return dict(self.exec_result)

    async def destroy_container(self, **kwargs: Any) -> None:
        self.destroy_calls.append(kwargs)

    async def container_put_archive(self, **kwargs: Any) -> None:
        # Test fakes track calls via the ``__getattr__`` dance —
        # tests reach in to .put_archive_calls. Default: no-op.
        if not hasattr(self, "put_archive_calls"):
            self.put_archive_calls = []
        self.put_archive_calls.append(kwargs)

    async def container_get_archive(self, **kwargs: Any) -> bytes:
        if not hasattr(self, "get_archive_calls"):
            self.get_archive_calls = []
        self.get_archive_calls.append(kwargs)
        return getattr(self, "get_archive_return", b"<tar bytes>")

    def container_exec_stream(self, **kwargs: Any) -> Any:
        if not hasattr(self, "exec_stream_calls"):
            self.exec_stream_calls = []
        self.exec_stream_calls.append(kwargs)
        chunks = getattr(self, "exec_stream_chunks", [
            {"stdout": b"hi\n", "stderr": b"", "done": False,
             "exit_code": 0, "timed_out": False},
            {"stdout": b"", "stderr": b"", "done": True,
             "exit_code": 0, "timed_out": False},
        ])

        async def _gen() -> Any:
            for c in chunks:
                yield c
        return _gen()


@dataclass
class _FakeQueryImageReply:
    present: bool


@dataclass
class _FakePlacement:
    """Mirror of ``xrlenv.control.scheduler.Placement`` minimum surface.

    ``reservation_id`` is included so the leak-fix lifecycle
    (``commit_placement`` / ``release_placement`` in
    ``RawContainerCoordinator.acquire``) has a real token to track.
    """

    node: Any
    backend: str = "docker"
    score: float = 1.0
    reservation_id: str = "fake-res-0"


@dataclass
class _FakeScheduler:
    """P1.7.B.2: case-2/3 raw-acquire now goes through
    ``Scheduler.place(image_present=...)`` — same algorithm case-1
    uses. The fake mimics the real scheduler's relevant surface:

    - ``nodes`` — the candidate pool.
    - ``image_aware_placement`` — operator opt-out flag (default
      True so query_image fan-out runs).
    - ``place(...)`` — by default returns the first node that
      supports the docker backend, mirroring the simplest case-1
      placement decision. Tests asserting image-affinity behaviour
      override ``place`` directly.
    - ``commit_placement`` / ``release_placement`` — counters for the
      raw-container ``_pending`` leak fix. Tests assert these are
      called exactly once per acquire (commit on success, release on
      failure) so the real scheduler's ``_pending`` dict can't leak.
    """

    nodes: list[Any]
    image_aware_placement: bool = True
    place_calls: list[dict] = field(default_factory=list)
    commit_calls: list[Any] = field(default_factory=list)
    release_calls: list[Any] = field(default_factory=list)
    raise_on_release: bool = False
    _next_reservation: int = 0

    def place(
        self,
        manifest: Any,
        *,
        task_key: Any = None,
        backend: Any = None,
        container_runtime: Any = None,
        image_present: Any = None,
        preferred_home_node: Any = None,
    ) -> _FakePlacement:
        self.place_calls.append({
            "manifest_image": getattr(manifest, "image", None),
            "manifest_resources": getattr(manifest, "resources", None),
            "task_key": task_key,
            "backend": backend,
            "container_runtime": container_runtime,
            "image_present": image_present,
            "preferred_home_node": preferred_home_node,
        })
        for node in self.nodes:
            if "docker" in node.supported_backends():
                reservation_id = f"fake-res-{self._next_reservation}"
                self._next_reservation += 1
                return _FakePlacement(
                    node=node, backend="docker", score=1.0,
                    reservation_id=reservation_id,
                )
        # No docker-capable node — emulate the real scheduler's
        # ``BackendCapabilityMissing`` raise.
        raise XRLEnvError(
            "no node supports backend 'docker' for raw-container",
        )

    def commit_placement(self, placement: Any) -> None:
        self.commit_calls.append(placement.reservation_id)

    def release_placement(self, placement: Any) -> None:
        if self.raise_on_release:
            raise RuntimeError("simulated release_placement failure")
        self.release_calls.append(placement.reservation_id)


@dataclass
class _CapacityExhaustedScheduler(_FakeScheduler):
    """``place()`` always raises ``CapacityExhausted`` — the pool was full for
    the whole ``queue_timeout_s`` wait (backpressure), the case the acquire
    seal must map to ``capacity_rejected`` rather than ``failed``."""

    def place(self, manifest: Any, **kw: Any) -> Any:
        self.place_calls.append(kw)
        raise CapacityExhausted(
            "queue_timeout_s=240.0 (waited 240.0s) expired waiting for capacity",
        )


@dataclass
class _RaisingScheduler(_FakeScheduler):
    """``place()`` raises a caller-supplied non-capacity fault — used to prove
    only ``CapacityExhausted`` gets the ``capacity_rejected`` carve-out."""

    exc: BaseException = field(default_factory=lambda: XRLEnvError("boom"))

    def place(self, manifest: Any, **kw: Any) -> Any:
        self.place_calls.append(kw)
        raise self.exc


# ──────────────────────────────────────────────────────────────────────────────
# acquire
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_rejected_when_not_raw_reconnect_capable() -> None:
    # audit H11: a deployment with the periodic raw-GC reconciler disabled
    # (gc_reconcile_interval_s=None) cannot inventory reconnecting-node survivors, so it is
    # EXPLICITLY not raw/compose-capable — raw + compose acquires fail loud rather than accrue
    # un-reconcilable load. (gym/step is unaffected — it doesn't use this coordinator.)
    from xrlenv.errors import XRLEnvError
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), raw_reconnect_capable=False,
    )
    with pytest.raises(XRLEnvError, match="cannot restart-safely inventory"):
        await coord.acquire(image="busybox:1", command=["sleep", "inf"])
    assert node.acquire_calls == []   # never dispatched to the node


@pytest.mark.asyncio
async def test_acquire_picks_node_and_records_session() -> None:
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    session = await coord.acquire(image="busybox:1", command=["sleep", "inf"])

    assert isinstance(session, RawContainerSession)
    assert session.node_id == "node-A"
    assert session.image == "busybox:1"
    assert session.container_id == "container-001"
    assert session.container_name == "name-of-container-001"

    # Wire-level call landed on the node with the assigned rollout_id.
    assert len(node.acquire_calls) == 1
    call = node.acquire_calls[0]
    assert call["rollout_id"] == session.rollout_id
    assert call["image"] == "busybox:1"
    assert call["command"] == ["sleep", "inf"]
    assert call["backend"] == "docker"

    # Session is registered.
    sessions = coord.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].rollout_id == session.rollout_id


@pytest.mark.asyncio
async def test_acquire_resolves_tag_to_digest_when_resolver_wired() -> None:
    """Freshness model (Part 2): with a digest resolver wired, the
    registry-qualified tag is pinned to its current digest BEFORE
    dispatch — the node receives the digest ref and the session records
    it (so a re-pushed tag reaches the node, and the run is auditable)."""
    from xrlenv.control.registry_resolver import RegistryDigestResolver

    digest = "sha256:" + "c" * 64

    async def _fixed_digest(host: str, repo: str, tag: str) -> str:
        assert (host, repo, tag) == ("reg:5011", "wai/substrate", "1ca77813")
        return digest

    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]),
        digest_resolver=RegistryDigestResolver(manifest_digest_fn=_fixed_digest),
    )

    session = await coord.acquire(image="reg:5011/wai/substrate:1ca77813")

    pinned = f"reg:5011/wai/substrate@{digest}"
    assert session.image == pinned
    # The node was dispatched the digest ref, not the mutable tag.
    assert node.acquire_calls[0]["image"] == pinned


@pytest.mark.asyncio
async def test_acquire_default_resources_when_no_request() -> None:
    """P0a — with no harness CPU/memory request the scheduler sees the
    raw-container default budget."""
    from xrlenv.control.raw_container_service import _DEFAULT_RAW_RESOURCES

    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1")

    assert sched.place_calls[-1]["manifest_resources"] == _DEFAULT_RAW_RESOURCES


@pytest.mark.asyncio
async def test_acquire_effective_resources_from_harness_request() -> None:
    """P0a — a harness CPU/memory request becomes the effective
    ResourceSpec the scheduler / capacity estimator place against."""
    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(
        image="busybox:1",
        cpu_limit=4.0,
        mem_limit_bytes=8 * 1024 * 1024 * 1024,
    )

    res = sched.place_calls[-1]["manifest_resources"]
    assert res.cpu_limit == 4.0
    assert res.cpu_request == 4.0
    assert res.mem_limit_bytes == 8 * 1024 * 1024 * 1024
    assert res.mem_request_bytes == 8 * 1024 * 1024 * 1024


@pytest.mark.asyncio
async def test_acquire_fractional_cpu_limit_rounds_request_up() -> None:
    """P3(a) integer-core admission — a pinned raw container reserves
    ceil(cpu_limit) whole cores, so the effective cpu_request is the
    integer core count, while cpu_limit keeps the exact fractional
    value for the runtime CFS quota. Without the round-up the estimator
    would under-count and over-admit pinned containers."""
    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1", cpu_limit=3.5)

    res = sched.place_calls[-1]["manifest_resources"]
    assert res.cpu_limit == 3.5            # exact — CFS quota
    assert res.cpu_request == 4.0          # ceil(3.5) — cores reserved


@pytest.mark.asyncio
async def test_acquire_threads_effective_resources_to_node() -> None:
    """P1 — the effective ResourceSpec is passed to node.acquire_container
    so the node stamps AcquireContainerCommand.resources and applies
    cgroup limits. The node receives the same spec used for placement."""
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    await coord.acquire(
        image="busybox:1", cpu_limit=4.0, mem_limit_bytes=8 * 1024**3,
    )

    res = node.acquire_calls[-1]["resources"]
    assert res is not None
    assert res.cpu_limit == 4.0
    assert res.mem_limit_bytes == 8 * 1024**3


@pytest.mark.asyncio
async def test_acquire_stamps_cpu_isolation_same_spec_to_placement_and_node() -> None:
    """P6 (audit TEST GAP) — the 'derive once, same object to placement and
    node' invariant for cpu_isolation. ``coord.acquire(cpu_isolation=REQUIRED)``
    stamps ``ResourceSpec.cpu_isolation`` on the effective spec and hands that
    SAME spec to BOTH ``Scheduler.place`` (capacity/placement) and
    ``node.acquire_container`` (the node command). If the two ever diverged,
    placement and the node would disagree on the isolation contract."""
    from xrlenv.backends.base import CpuIsolation

    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1", cpu_isolation=CpuIsolation.REQUIRED)

    placed = sched.place_calls[-1]["manifest_resources"]
    sent = node.acquire_calls[-1]["resources"]
    assert placed.cpu_isolation is CpuIsolation.REQUIRED
    assert sent.cpu_isolation is CpuIsolation.REQUIRED
    # Derive-once: placement and the node see the same effective spec.
    assert placed == sent


@pytest.mark.asyncio
async def test_acquire_cpu_isolation_defaults_off_on_effective_spec() -> None:
    """P6 — with no cpu_isolation request the effective spec defaults to OFF on
    both surfaces (the coordinator never accidentally stamps isolation)."""
    from xrlenv.backends.base import CpuIsolation

    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1")

    assert sched.place_calls[-1]["manifest_resources"].cpu_isolation is CpuIsolation.OFF
    assert node.acquire_calls[-1]["resources"].cpu_isolation is CpuIsolation.OFF


@pytest.mark.asyncio
async def test_acquire_threads_runtime_limits_to_node() -> None:
    """P0b — RuntimeLimits flow through the coordinator to
    node.acquire_container unchanged (scheduling-neutral pass-through)."""
    from xrlenv.backends.base import RuntimeLimits

    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    await coord.acquire(
        image="busybox:1",
        runtime_limits=RuntimeLimits(pids_limit=1024, readonly_rootfs=True),
    )

    rl = node.acquire_calls[-1]["runtime_limits"]
    assert rl is not None
    assert rl.pids_limit == 1024
    assert rl.readonly_rootfs is True


@pytest.mark.asyncio
async def test_acquire_threads_container_runtime_to_node_and_placement() -> None:
    """§5.1/§5.3 — container_runtime flows through the coordinator to BOTH
    Scheduler.place (so placement filters by runtime) AND node.acquire_container
    (so the node runs under it). Requires the policy to permit the runtime."""
    from xrlenv.control.kwargs_policy import KwargsPolicy

    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(
        scheduler=sched,
        kwargs_policy=KwargsPolicy(allowed_runtimes=("sysbox-runc",)),
    )

    await coord.acquire(image="dind:1", container_runtime="sysbox-runc")

    assert sched.place_calls[-1]["container_runtime"] == "sysbox-runc"
    assert node.acquire_calls[-1]["container_runtime"] == "sysbox-runc"


@pytest.mark.asyncio
async def test_acquire_default_omits_container_runtime() -> None:
    """§10.x — a default acquire (no container_runtime) forwards None to both
    placement and the node: the ordinary runc path is unchanged."""
    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1")

    assert sched.place_calls[-1]["container_runtime"] is None
    assert node.acquire_calls[-1]["container_runtime"] is None


@pytest.mark.asyncio
async def test_acquire_rejects_runtime_without_policy_optin() -> None:
    """§5.2 — with the default policy (empty allowed_runtimes), an acquire that
    requests a non-runc runtime is rejected BEFORE any placement or node call."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)  # DEFAULT_POLICY

    with pytest.raises(KwargsPolicyViolation):
        await coord.acquire(image="dind:1", container_runtime="sysbox-runc")

    # Fail-fast: no placement, no node acquire happened.
    assert sched.place_calls == []
    assert node.acquire_calls == []


@pytest.mark.asyncio
async def test_acquire_assigns_unique_rollout_ids() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    s1 = await coord.acquire(image="busybox:1")
    node.next_container_id = "container-002"
    s2 = await coord.acquire(image="busybox:1")

    assert s1.rollout_id != s2.rollout_id


@pytest.mark.asyncio
async def test_acquire_passes_environment_through() -> None:
    """Audit Raw-Policy-M1 closure: environment flows from the
    coordinator to the node's transport."""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    await coord.acquire(
        image="busybox:1",
        environment={"HF_TOKEN": "abc"},
    )

    assert node.acquire_calls[0]["environment"] == {"HF_TOKEN": "abc"}


@pytest.mark.asyncio
async def test_acquire_threads_acquire_timeout_to_node_transport() -> None:
    """Issue #12: ``acquire_timeout_s`` must flow from the consumer
    SDK all the way to the per-node ``acquire_container`` call so
    consumers with known-huge images can widen the wire deadline.
    """
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    await coord.acquire(image="busybox:1", acquire_timeout_s=1800.0)
    assert node.acquire_calls[0]["acquire_timeout_s"] == 1800.0

    # And the unset case must thread ``None`` so the transport can
    # apply its own default (currently 600 s) without ambiguity.
    await coord.acquire(image="busybox:2")
    assert node.acquire_calls[1]["acquire_timeout_s"] is None


@pytest.mark.asyncio
async def test_acquire_raises_when_no_nodes_attached() -> None:
    """P1.7.B.2: raw-acquire now goes through ``Scheduler.place``,
    which raises ``BackendCapabilityMissing`` (or our fake's
    equivalent) when no backend-capable node exists."""
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[]))

    with pytest.raises(XRLEnvError, match="no node supports backend"):
        await coord.acquire(image="busybox:1")


@pytest.mark.asyncio
async def test_acquire_raises_when_no_docker_capable_node() -> None:
    """P1.7.B.2: same — scheduler rejects the placement request."""
    node = _FakeNodeTransport(backends=["cubesandbox"])
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(XRLEnvError, match="no node supports backend"):
        await coord.acquire(image="busybox:1")


@pytest.mark.asyncio
async def test_acquire_picks_first_docker_node_when_mixed() -> None:
    """First-available with a docker filter — non-docker nodes are
    skipped. (Image-affinity scheduling is P1.7.B territory.)"""
    cube = _FakeNodeTransport(node_id="cube-A", backends=["cubesandbox"])
    docker_node = _FakeNodeTransport(node_id="docker-B", backends=["docker"])
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[cube, docker_node]),
    )

    session = await coord.acquire(image="busybox:1")
    assert session.node_id == "docker-B"


# ──────────────────────────────────────────────────────────────────────────────
# exec
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exec_routes_to_session_node() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    result = await coord.exec(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
        cmd=["echo", "hi"],
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == b"hi\n"
    assert len(node.exec_calls) == 1
    assert node.exec_calls[0]["rollout_id"] == session.rollout_id
    assert node.exec_calls[0]["container_id"] == session.container_id


@pytest.mark.asyncio
async def test_exec_rejects_unknown_rollout() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(XRLEnvError, match="not found"):
        await coord.exec(
            rollout_id="ghost-rollout",
            container_id="ghost-container",
            cmd=["echo", "hi"],
        )


@pytest.mark.asyncio
async def test_exec_rejects_mismatched_container_id() -> None:
    """Stale handle: rollout exists but caller's container_id is
    not the session's. Common when a destroy / re-acquire cycle
    races with a slow exec call held by the consumer."""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not match"):
        await coord.exec(
            rollout_id=session.rollout_id,
            container_id="stale-container-id",
            cmd=["echo", "hi"],
        )


# ──────────────────────────────────────────────────────────────────────────────
# destroy
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_destroy_routes_to_node_and_drops_session() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
    )

    assert len(node.destroy_calls) == 1
    assert node.destroy_calls[0]["rollout_id"] == session.rollout_id
    # Session deregistered.
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_destroy_rejects_unknown_rollout() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(XRLEnvError, match="not found"):
        await coord.destroy(
            rollout_id="ghost", container_id="ghost",
        )


@pytest.mark.asyncio
async def test_destroy_rejects_mismatched_container_id() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not match"):
        await coord.destroy(
            rollout_id=session.rollout_id,
            container_id="stale-container-id",
        )


# ──────────────────────────────────────────────────────────────────────────────
# terminate_raw_group — raw analogue of cancel_group (destroy a run's containers)
# ──────────────────────────────────────────────────────────────────────────────


async def _state_backed_coord() -> tuple[Any, Any, Any]:
    """A coordinator wired to a real ``InMemoryStateStore`` so ``list_raw_rollouts``
    (which ``terminate_raw_group`` sweeps) sees persisted rows. Returns (coord, node, state)."""
    from xrlenv.control.scheduler import Scheduler
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.control.template_catalog import TemplateCatalog

    node = _FakeNodeTransport(node_id="node-A")
    state = InMemoryStateStore()
    scheduler = Scheduler(
        [node], catalog=TemplateCatalog(), state=state, max_runs_per_task=100,
    )
    coord = RawContainerCoordinator(scheduler=scheduler, state=state)
    scheduler.set_raw_session_provider(coord.iter_load_entries)
    return coord, node, state


async def _acquire_in_group(coord: Any, node: Any, group_id: str, cid: str) -> Any:
    node.next_container_id = cid
    return await coord.acquire(image="busybox:1", labels={"xrlenv.group_id": group_id})


@pytest.mark.asyncio
async def test_terminate_raw_group_destroys_only_that_group() -> None:
    coord, node, _state = await _state_backed_coord()
    a = await _acquire_in_group(coord, node, "g1", "c-a")
    b = await _acquire_in_group(coord, node, "g1", "c-b")
    other = await _acquire_in_group(coord, node, "g2", "c-c")

    report = await coord.terminate_raw_group("g1", reason="run_aborted")

    # Both g1 members torn down on the node; g2 untouched.
    assert set(report.terminated) == {a.rollout_id, b.rollout_id}
    assert report.already_terminal == ()
    destroyed = {c["rollout_id"] for c in node.destroy_calls}
    assert destroyed == {a.rollout_id, b.rollout_id}
    assert other.rollout_id not in destroyed
    # g1 sessions deregistered, g2 still live.
    assert {s.rollout_id for s in coord.list_sessions()} == {other.rollout_id}


@pytest.mark.asyncio
async def test_terminate_raw_group_is_idempotent() -> None:
    coord, node, _state = await _state_backed_coord()
    a = await _acquire_in_group(coord, node, "g1", "c-a")

    first = await coord.terminate_raw_group("g1")
    assert first.terminated == (a.rollout_id,)

    # Second sweep: the row is now terminal (released) → reported already_terminal, no re-destroy.
    second = await coord.terminate_raw_group("g1")
    assert second.terminated == ()
    assert a.rollout_id in second.already_terminal
    assert len(node.destroy_calls) == 1  # not destroyed twice


@pytest.mark.asyncio
async def test_terminate_raw_group_empty_group_is_noop() -> None:
    coord, node, _state = await _state_backed_coord()
    report = await coord.terminate_raw_group("nonexistent")
    assert report == TerminateRawGroupReport(
        group_id="nonexistent", terminated=(), already_terminal=(),
    )
    assert node.destroy_calls == []


@pytest.mark.asyncio
async def test_terminate_raw_group_owner_scoped() -> None:
    # A caller scoped to a different owner sweeps nothing (the WHERE owner_id guard) — a tenant
    # can't tear down another's group by guessing its id. Rows are stamped owner "default".
    coord, node, _state = await _state_backed_coord()
    await _acquire_in_group(coord, node, "g1", "c-a")
    report = await coord.terminate_raw_group("g1", owner_id="someone-else")
    assert report.terminated == () and report.already_terminal == ()
    assert node.destroy_calls == []


# ──────────────────────────────────────────────────────────────────────────────
# seal_orphan — coordinator-only orphan teardown (no wire destroy)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seal_orphan_drops_session_without_wire_destroy() -> None:
    """A coordinator-only orphan's container is already gone on the node,
    so ``seal_orphan`` must drop the in-memory session (freeing its
    capacity charge) WITHOUT issuing a wire-level destroy — that RPC would
    only race and, once the node dropped its record, fail benignly."""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    await coord.seal_orphan(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
        reason="disk-guard: reaped runaway raw container",
    )

    assert node.destroy_calls == []  # NO wire destroy
    assert coord.list_sessions() == []  # session deregistered


@pytest.mark.asyncio
async def test_seal_orphan_rejects_mismatched_container_id() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not match"):
        await coord.seal_orphan(
            rollout_id=session.rollout_id,
            container_id="stale-container-id",
        )
    # Session left intact for the caller's fallback to handle.
    assert len(coord.list_sessions()) == 1


class _RecordingState:
    """Captures the latest ``raw_rollouts`` field-set per rollout_id so a
    test can assert how ``seal_orphan`` sealed the durable row."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def record_raw_rollout(self, record: Any) -> None:
        self.rows[record.rollout_id] = {
            "status": getattr(record, "status", None),
        }

    def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
        self.rows.setdefault(rollout_id, {}).update(fields)


@pytest.mark.asyncio
async def test_seal_orphan_seals_reaped_with_reason_else_released() -> None:
    """``seal_orphan`` seals ``reaped`` + ``error=reason`` when the node
    reported a real reap cause (audit P3 — disk-guard), and ``released``
    (no error) when the container simply vanished for some other cause."""
    state = _RecordingState()
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=state,
    )
    reaped = await coord.acquire(image="busybox:1")
    node.next_container_id = "container-002"
    released = await coord.acquire(image="busybox:1")

    disk_reason = (
        "disk-guard: reaped runaway raw container (writable 5e11 bytes)"
    )
    await coord.seal_orphan(
        rollout_id=reaped.rollout_id,
        container_id=reaped.container_id,
        reason=disk_reason,
    )
    await coord.seal_orphan(
        rollout_id=released.rollout_id,
        container_id=released.container_id,
    )

    assert state.rows[reaped.rollout_id]["status"] == "reaped"
    assert state.rows[reaped.rollout_id]["error"] == disk_reason
    assert state.rows[reaped.rollout_id]["finished_at"] > 0.0
    assert state.rows[released.rollout_id]["status"] == "released"
    assert state.rows[released.rollout_id].get("error") is None
    assert node.destroy_calls == []


@pytest.mark.asyncio
async def test_acquire_capacity_exhausted_seals_capacity_rejected() -> None:
    """A scheduler decline (``CapacityExhausted``) seals the durable row
    ``capacity_rejected`` — NOT ``failed`` — so a paced-then-retried acquire
    doesn't paint a green run red (spec 13 admin categorization). The reason
    text keeps the original queue_timeout message and adds the operator levers
    (raise queue_timeout_s / retry caller-side)."""
    state = _RecordingState()
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_CapacityExhaustedScheduler(
            nodes=[node], image_aware_placement=False,
        ),
        state=state,
    )
    with pytest.raises(CapacityExhausted):
        await coord.acquire(image="busybox:1")

    assert len(state.rows) == 1
    (row,) = state.rows.values()
    assert row["status"] == "capacity_rejected"
    assert "queue_timeout_s=240.0" in row["error"]
    assert "backpressure" in row["error"].lower()
    assert "queue_timeout_s" in row["error"]  # names the lever
    assert row["finished_at"] > 0.0
    # Never placed → no scheduler commit/release, no leaked reservation.
    assert coord._scheduler.commit_calls == []  # type: ignore[attr-defined]
    assert coord._scheduler.release_calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_acquire_generic_scheduler_error_still_seals_failed() -> None:
    """Regression guard: ONLY ``CapacityExhausted`` maps to
    ``capacity_rejected``. A generic placement fault still seals ``failed`` so
    real breakage stays visible in the failed tally."""
    state = _RecordingState()
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_RaisingScheduler(
            nodes=[node], image_aware_placement=False,
            exc=XRLEnvError("scheduler exploded"),
        ),
        state=state,
    )
    with pytest.raises(XRLEnvError):
        await coord.acquire(image="busybox:1")

    (row,) = state.rows.values()
    assert row["status"] == "failed"
    assert "scheduler exploded" in row["error"]


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.A.2 — archives
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_put_archive_routes_to_session_node() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    await coord.put_archive(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
        target_dir="/tmp",
        tarball=b"<tar>",
    )

    assert len(node.put_archive_calls) == 1
    call = node.put_archive_calls[0]
    assert call["rollout_id"] == session.rollout_id
    assert call["container_id"] == session.container_id
    assert call["target_dir"] == "/tmp"
    assert call["tarball"] == b"<tar>"


@pytest.mark.asyncio
async def test_put_archive_rejects_mismatched_container_id() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not match"):
        await coord.put_archive(
            rollout_id=session.rollout_id,
            container_id="stale-id",
            target_dir="/tmp",
            tarball=b"<tar>",
        )


@pytest.mark.asyncio
async def test_get_archive_routes_and_returns_bytes() -> None:
    node = _FakeNodeTransport()
    node.get_archive_return = b"<received tar>"  # type: ignore[attr-defined]
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    tarball = await coord.get_archive(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
        source_path="/logs/artifacts",
    )

    assert tarball == b"<received tar>"
    assert len(node.get_archive_calls) == 1
    assert node.get_archive_calls[0]["source_path"] == "/logs/artifacts"


@pytest.mark.asyncio
async def test_exec_stream_routes_to_session_node_and_yields_chunks() -> None:
    """Coordinator's exec_stream resolves the session, dispatches
    to the session's node, and the iterator yields chunks until
    the terminator."""
    node = _FakeNodeTransport()
    node.exec_stream_chunks = [  # type: ignore[attr-defined]
        {"stdout": b"a\n", "stderr": b"", "done": False,
         "exit_code": 0, "timed_out": False},
        {"stdout": b"b\n", "stderr": b"", "done": False,
         "exit_code": 0, "timed_out": False},
        {"stdout": b"", "stderr": b"", "done": True,
         "exit_code": 0, "timed_out": False},
    ]
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    chunks = [
        c async for c in coord.exec_stream(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
            cmd=["bash", "-c", "echo a; echo b"],
        )
    ]

    assert len(chunks) == 3
    assert chunks[0]["stdout"] == b"a\n"
    assert chunks[2]["done"] is True
    # Routed via session's node.
    assert len(node.exec_stream_calls) == 1  # type: ignore[attr-defined]
    assert node.exec_stream_calls[0]["rollout_id"] == session.rollout_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_exec_stream_rejects_mismatched_container_id() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not match"):
        async for _ in coord.exec_stream(
            rollout_id=session.rollout_id,
            container_id="stale-id",
            cmd=["echo", "hi"],
        ):
            pass


@pytest.mark.asyncio
async def test_get_archive_rejects_mismatched_container_id() -> None:
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    with pytest.raises(XRLEnvError, match="does not match"):
        await coord.get_archive(
            rollout_id=session.rollout_id,
            container_id="stale-id",
            source_path="/logs",
        )


# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_destroy_retains_session_when_node_destroy_raises() -> None:
    """audit H8 / invariant 2: a FAILED wire-level destroy is NOT node-confirmed, so the
    session (and its capacity charge) is RETAINED — the raw-GC reconciler / a retry re-attempts
    teardown, and only a confirmed destroy frees capacity. Mirrors the compose teardown path.
    (Was ``..._drops_session_even_when_node_destroy_raises`` — that dropped capacity without a
    node ack, an invariant-2 violation.)"""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    # Make the node-side destroy raise.
    async def _raising_destroy(**kwargs: Any) -> None:
        raise RuntimeError("node went away")
    node.destroy_container = _raising_destroy  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="node went away"):
        await coord.destroy(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
        )
    # Session RETAINED despite the exception — capacity stays charged until node-confirmed.
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]
    # And it still charges the scheduler's capacity gate (invariant 2 regression: a dropped
    # session would emit 0 load entries and let the scheduler reuse the capacity).
    assert len(coord.iter_load_entries()) == 1
    # Not mid-destroy anymore — the deadline/liveness sweep + a retry may re-attempt.
    assert coord.is_destroying(session.rollout_id) is False


# ──────────────────────────────────────────────────────────────────────────────
# Issue #6: cluster-wide docker-kwarg policy is enforced authoritatively
# at the coordinator. The drop-in's fast-fail against DEFAULT_POLICY runs
# first for harness-side callers; this re-check defends in-process callers
# (e.g. ``Client.acquire_container(devices=[...])`` direct from a custom
# harness) and applies operator-specific tweaks (denied_caps, Level-2 opt-
# ins) the drop-in can't see.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_rejects_unlisted_device_under_default_policy() -> None:
    """Coordinator with no explicit ``kwargs_policy`` uses DEFAULT_POLICY;
    a device outside ``[/dev/kvm, /dev/net/tun, /dev/fuse]`` rejects."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(KwargsPolicyViolation) as ei:
        await coord.acquire(
            image="busybox:1", command=["sleep", "inf"],
            devices=["/dev/sda"],
        )
    assert "/dev/sda" in str(ei.value)
    # Coordinator fast-fails BEFORE picking a node or dispatching.
    assert node.acquire_calls == []


@pytest.mark.asyncio
async def test_acquire_allows_kvm_device_under_default_policy() -> None:
    """SCUBA-style ``devices=[/dev/kvm]`` flows through; node receives
    the kwarg verbatim. Regression guard for the unblock that motivated
    issue #6's Option D."""
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    await coord.acquire(
        image="busybox:1", command=["sleep", "inf"],
        devices=["/dev/kvm"], cap_add=["NET_ADMIN"],
    )
    assert len(node.acquire_calls) == 1
    assert node.acquire_calls[0]["devices"] == ["/dev/kvm"]
    assert node.acquire_calls[0]["cap_add"] == ["NET_ADMIN"]


@pytest.mark.asyncio
async def test_acquire_honors_operator_denied_caps() -> None:
    """Cluster policy with ``denied_caps`` rejects matching kwargs even
    when DEFAULT_POLICY would allow them. The operator's restriction
    is the authoritative one — drop-in's pre-check is just a fast-fail
    convenience."""
    from xrlenv.control.kwargs_policy import (
        KwargsPolicy,
        KwargsPolicyViolation,
    )

    policy = KwargsPolicy(denied_caps=("NET_ADMIN",))
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]),
        kwargs_policy=policy,
    )

    with pytest.raises(KwargsPolicyViolation) as ei:
        await coord.acquire(
            image="busybox:1", command=["sleep", "inf"],
            cap_add=["NET_ADMIN"],
        )
    assert "NET_ADMIN" in str(ei.value)
    assert node.acquire_calls == []


@pytest.mark.asyncio
async def test_acquire_honors_operator_extended_device_allowlist() -> None:
    """Operator extending ``allowed_devices`` unlocks a device the
    drop-in's DEFAULT_POLICY allowlist doesn't include. Per audit M1
    the drop-in does NOT pre-validate Level-1 device lists — they all
    flow to the wire, and the coordinator (which sees the cluster
    policy) is the sole authority. Regression guard for the M1 fix."""
    from xrlenv.control.kwargs_policy import KwargsPolicy

    policy = KwargsPolicy(
        allowed_devices=("/dev/kvm", "/dev/net/tun", "/dev/fuse", "/dev/dri/card0"),
    )
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]),
        kwargs_policy=policy,
    )

    await coord.acquire(
        image="busybox:1", command=["sleep", "inf"],
        devices=["/dev/dri/card0"],
    )
    assert node.acquire_calls[0]["devices"] == ["/dev/dri/card0"]


# ──────────────────────────────────────────────────────────────────────────────
# Audit M1: Level-2 operator opt-ins must actually unlock the kwarg
# end-to-end. Previously ``allow_host_network`` / ``allow_privileged`` /
# ``allowed_host_paths`` were documented + schema-supported but the drop-in's
# DEFAULT_POLICY pre-check rejected the request before the cluster policy
# was consulted. Now the drop-in defers, and the coordinator + node honor
# the operator's policy. These tests pin that contract.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_rejects_privileged_under_default_policy() -> None:
    """``privileged=True`` is Level 2 — default policy rejects with
    operator-rationale hint."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(KwargsPolicyViolation) as ei:
        await coord.acquire(
            image="busybox:1", command=["sleep", "inf"], privileged=True,
        )
    assert "privileged" in str(ei.value)
    assert "allow_privileged" in str(ei.value)
    assert node.acquire_calls == []


@pytest.mark.asyncio
async def test_acquire_allows_privileged_when_operator_opts_in() -> None:
    """Operator sets ``allow_privileged: true`` in nodes.yaml → request
    flows through to the node with ``privileged=True``. This is the
    end-to-end proof for audit M1: the Level-2 opt-in is actually
    reachable."""
    from xrlenv.control.kwargs_policy import KwargsPolicy

    policy = KwargsPolicy(allow_privileged=True)
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]),
        kwargs_policy=policy,
    )

    await coord.acquire(
        image="busybox:1", command=["sleep", "inf"], privileged=True,
    )
    assert node.acquire_calls[0]["privileged"] is True


@pytest.mark.asyncio
async def test_acquire_rejects_network_mode_host_under_default_policy() -> None:
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(KwargsPolicyViolation) as ei:
        await coord.acquire(
            image="busybox:1", command=["sleep", "inf"], network_mode="host",
        )
    assert "network_mode" in str(ei.value)
    assert "allow_host_network" in str(ei.value)


@pytest.mark.asyncio
async def test_acquire_allows_network_mode_host_when_operator_opts_in() -> None:
    """Operator sets ``allow_host_network: true`` → ``host`` flows to
    the node. Audit M1 end-to-end proof."""
    from xrlenv.control.kwargs_policy import KwargsPolicy

    policy = KwargsPolicy(allow_host_network=True)
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]),
        kwargs_policy=policy,
    )

    await coord.acquire(
        image="busybox:1", command=["sleep", "inf"], network_mode="host",
    )
    assert node.acquire_calls[0]["network_mode"] == "host"


@pytest.mark.asyncio
async def test_acquire_network_mode_bridge_always_allowed() -> None:
    """``bridge`` / ``none`` / ``default`` aren't security-relevant —
    pass through under default policy."""
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    await coord.acquire(
        image="busybox:1", command=["sleep", "inf"], network_mode="bridge",
    )
    assert node.acquire_calls[0]["network_mode"] == "bridge"


@pytest.mark.asyncio
async def test_acquire_rejects_binds_under_default_policy() -> None:
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(KwargsPolicyViolation) as ei:
        await coord.acquire(
            image="busybox:1", command=["sleep", "inf"],
            binds=["/var/data:/data:ro"],
        )
    assert "binds" in str(ei.value)
    assert "/var/data" in str(ei.value)


@pytest.mark.asyncio
async def test_acquire_allows_binds_when_operator_opts_in() -> None:
    """Operator lists the host path → bind flows to the node. Audit M1
    end-to-end proof for ``allowed_host_paths``."""
    from xrlenv.control.kwargs_policy import KwargsPolicy

    policy = KwargsPolicy(allowed_host_paths=("/mnt/datasets",))
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]),
        kwargs_policy=policy,
    )

    await coord.acquire(
        image="busybox:1", command=["sleep", "inf"],
        binds=["/mnt/datasets:/data:ro"],
    )
    assert node.acquire_calls[0]["binds"] == ["/mnt/datasets:/data:ro"]


# ──────────────────────────────────────────────────────────────────────────────
# Leak-fix lifecycle (Option C) — scheduler ``_pending`` is released
# correctly so the cluster never accumulates phantom load, AND the
# coordinator surfaces active sessions to the scheduler's capacity gate
# via the ``set_raw_session_provider`` plumbing so raw containers count
# toward the operator's parallelism cap.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_commits_placement_on_success() -> None:
    """Successful ``acquire`` calls ``commit_placement`` exactly once
    and never ``release_placement``. Without this, every successful
    acquire leaks a ``_pending`` entry in the real scheduler — the
    bug that caused 358/500 SWE-Verified tasks to fail with
    ``CapacityExhausted`` on a cluster that had nothing actually
    running."""
    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    session = await coord.acquire(image="busybox:1")

    assert len(sched.commit_calls) == 1
    assert sched.release_calls == []
    # The commit token must match the placement minted by ``place``.
    assert sched.commit_calls[0] == "fake-res-0"
    # Session is registered and addressable for the load provider.
    assert session.task_key is None
    assert coord.iter_load_entries() != []


@pytest.mark.asyncio
async def test_acquire_releases_placement_on_node_failure() -> None:
    """When the wire-level ``acquire_container`` raises, the
    coordinator must release the ``_pending`` reservation. Without
    this, a transient node failure permanently leaks one scheduler
    slot per failed attempt — the original report's ``142`` cumulative
    successes before saturation came partly from this path."""
    node = _FakeNodeTransport(node_id="node-A")
    node.raise_on_acquire = RuntimeError("docker pull timed out")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    with pytest.raises(RuntimeError, match="docker pull timed out"):
        await coord.acquire(image="busybox:1")

    assert sched.commit_calls == []
    assert sched.release_calls == ["fake-res-0"]
    # No session left behind to leak load on the next placement.
    assert coord.iter_load_entries() == []


@pytest.mark.asyncio
async def test_iter_load_entries_stores_task_key_for_max_runs_accounting() -> None:
    """``RawContainerSession.task_key`` flows from the acquire call
    through ``iter_load_entries`` so the scheduler's
    ``max_runs_per_task`` accounting can attribute raw containers to
    their task. Without this, raw containers would never count toward
    the per-task cap regardless of how many were active for one task."""
    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched)

    await coord.acquire(image="busybox:1", task_key="task-X")
    node.next_container_id = "container-002"
    await coord.acquire(image="busybox:1", task_key="task-Y")

    entries = coord.iter_load_entries()
    task_keys = sorted(e.task_key for e in entries if e.task_key is not None)
    assert task_keys == ["task-X", "task-Y"]
    # Both entries land on the same node and synthesize the same
    # template name shape as ``_synthetic_manifest_for_raw``.
    assert {e.node_id for e in entries} == {"node-A"}
    assert all(
        e.template_name == "raw-container/busybox:1" for e in entries
    )


@pytest.mark.asyncio
async def test_iter_load_entries_reports_effective_resources() -> None:
    """P0a (audit M1) — a harness CPU/memory override is stored on the
    session and reported by ``iter_load_entries``, so the scheduler's
    steady-state load accounting charges the same footprint the
    placement decision used. Without this, a harness asking for more
    than the default is correctly placed once, then under-counted for
    every subsequent placement."""
    from xrlenv.control.raw_container_service import _DEFAULT_RAW_RESOURCES

    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    await coord.acquire(
        image="busybox:1",
        cpu_limit=6.0,
        mem_limit_bytes=16 * 1024 * 1024 * 1024,
    )
    node.next_container_id = "container-002"
    await coord.acquire(image="busybox:2")  # no override → default

    by_template = {
        e.template_name: e.effective_resources
        for e in coord.iter_load_entries()
    }
    overridden = by_template["raw-container/busybox:1"]
    assert overridden.cpu_limit == 6.0
    assert overridden.cpu_request == 6.0
    assert overridden.mem_limit_bytes == 16 * 1024 * 1024 * 1024
    # The un-overridden session still charges the default budget.
    assert by_template["raw-container/busybox:2"] == _DEFAULT_RAW_RESOURCES


@pytest.mark.asyncio
async def test_parallelism_cap_enforced_via_real_scheduler() -> None:
    """End-to-end Option C: a real ``Scheduler`` wired to the
    coordinator's ``iter_load_entries`` must refuse a ``place()``
    once the operator's parallelism cap is reached, and must accept
    a new acquire once one of the existing sessions is destroyed.

    This is the operator's hard contract: ``parallelism=N`` means at
    most ``N`` concurrent raw containers, enforced by the capacity
    gate — not by hope. Without Option C, raw containers don't appear
    in ``state.list_sandboxes()`` and the scheduler is blind to them;
    the gate would let acquires through until the docker daemon or
    OOM intervenes.

    Uses ``max_runs_per_task`` (=4 here, single shared ``task_key``)
    as the cap because it's deterministic and doesn't depend on
    resource math. The same mechanism feeds the resource-based gate
    via the same load contribution from ``iter_load_entries``.
    """
    from xrlenv.control.scheduler import Scheduler
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.control.template_catalog import TemplateCatalog

    node = _FakeNodeTransport(node_id="node-A")
    state = InMemoryStateStore()
    catalog = TemplateCatalog()
    scheduler = Scheduler(
        [node],
        catalog=catalog,
        state=state,
        max_runs_per_task=4,
    )
    coord = RawContainerCoordinator(
        scheduler=scheduler, state=state,
    )
    scheduler.set_raw_session_provider(coord.iter_load_entries)

    # Fill the cap. Each acquire mints a new container_id; reuse the
    # same task_key so the per-task gate is the one being exercised.
    sessions = []
    for i in range(4):
        node.next_container_id = f"container-{i:03d}"
        sessions.append(
            await coord.acquire(image="busybox:1", task_key="t1"),
        )

    # 5th acquire must be refused — the scheduler sees 4 live raw
    # sessions for ``task_key=t1`` on node-A and ``max_runs_per_task=4``
    # is the ceiling.
    node.next_container_id = "container-004"
    with pytest.raises(CapacityExhausted):
        await coord.acquire(image="busybox:1", task_key="t1")

    # The refused acquire must not have leaked a ``_pending`` entry —
    # the scheduler's load contribution from raw sessions is still
    # exactly 4 (the cap), not 5. If the leak fix regresses we'd see
    # ``_pending`` accumulate here and a subsequent destroy + retry
    # would still fail.
    assert len(scheduler._pending) == 0
    assert len(coord.iter_load_entries()) == 4

    # Destroy one and retry — succeeds now that the cap has headroom.
    await coord.destroy(
        rollout_id=sessions[0].rollout_id,
        container_id=sessions[0].container_id,
    )
    node.next_container_id = "container-005"
    new_session = await coord.acquire(image="busybox:1", task_key="t1")
    assert new_session.rollout_id != sessions[0].rollout_id
    # Steady-state: still exactly 4 alive, ``_pending`` empty.
    assert len(coord.iter_load_entries()) == 4
    assert len(scheduler._pending) == 0


@pytest.mark.asyncio
async def test_failed_acquires_do_not_accumulate_pending_in_real_scheduler() -> None:
    """Direct repro of the SWE-Verified production symptom: 500 failed
    attempts in a row must not saturate ``_pending`` to the point
    where a subsequent successful attempt is impossible.

    Pre-fix, every call to ``RawContainerCoordinator.acquire`` that
    raised in ``acquire_container`` left a ``_pending`` entry behind.
    After ~N attempts (where N depends on the node's capacity) the
    scheduler would refuse every subsequent ``place()`` even though
    no containers were actually running.

    This test asserts the post-fix invariant: pure-failure runs never
    grow ``_pending``, period.
    """
    from xrlenv.control.scheduler import Scheduler
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.control.template_catalog import TemplateCatalog

    node = _FakeNodeTransport(node_id="node-A")
    node.raise_on_acquire = RuntimeError("simulated node failure")
    state = InMemoryStateStore()
    scheduler = Scheduler(
        [node], catalog=TemplateCatalog(), state=state,
        max_runs_per_task=4,
    )
    coord = RawContainerCoordinator(scheduler=scheduler, state=state)
    scheduler.set_raw_session_provider(coord.iter_load_entries)

    # Hammer the same task_key with failing acquires.
    for _ in range(20):
        with pytest.raises(RuntimeError, match="simulated node failure"):
            await coord.acquire(image="busybox:1", task_key="t1")

    assert len(scheduler._pending) == 0
    assert coord.iter_load_entries() == []

    # And a successful acquire after the leak window still works.
    node.raise_on_acquire = None
    session = await coord.acquire(image="busybox:1", task_key="t1")
    assert session is not None
    assert len(coord.iter_load_entries()) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 fix #2 — _destroying state + load visibility during teardown
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_is_destroying_true_during_wire_destroy_false_before_and_after() -> None:
    """``is_destroying(rollout_id)`` returns False before destroy is
    called, True while the wire call is awaiting, and False once
    destroy completes (session dropped from _sessions and _destroying
    cleared in the finally block)."""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    assert coord.is_destroying(session.rollout_id) is False

    destroy_started = asyncio.Event()
    destroy_may_finish = asyncio.Event()

    original_destroy = node.destroy_container

    async def _slow_destroy(**kwargs: Any) -> None:
        destroy_started.set()
        await destroy_may_finish.wait()
        await original_destroy(**kwargs)

    node.destroy_container = _slow_destroy  # type: ignore[method-assign]

    task = asyncio.create_task(
        coord.destroy(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
        )
    )

    await destroy_started.wait()
    # Wire call is in flight — is_destroying must be True.
    assert coord.is_destroying(session.rollout_id) is True

    destroy_may_finish.set()
    await task

    # Destroy complete — is_destroying must be False and session gone.
    assert coord.is_destroying(session.rollout_id) is False
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_session_in_load_entries_during_destroy() -> None:
    """The session remains in ``iter_load_entries()`` for the entire
    duration of the destroy wire call so the scheduler doesn't
    over-place onto the node while teardown is in flight."""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    destroy_started = asyncio.Event()
    destroy_may_finish = asyncio.Event()

    original_destroy = node.destroy_container

    async def _slow_destroy(**kwargs: Any) -> None:
        destroy_started.set()
        await destroy_may_finish.wait()
        await original_destroy(**kwargs)

    node.destroy_container = _slow_destroy  # type: ignore[method-assign]

    task = asyncio.create_task(
        coord.destroy(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
        )
    )

    await destroy_started.wait()
    # Session is still in load entries while wire call is in flight.
    entries = coord.iter_load_entries()
    assert any(True for _ in entries), (
        "session must still appear in iter_load_entries() during destroy"
    )
    assert session.rollout_id in coord._sessions

    destroy_may_finish.set()
    await task

    # After destroy completes the session is gone.
    assert coord.iter_load_entries() == []


@pytest.mark.asyncio
async def test_is_destroying_false_after_destroy_raises() -> None:
    """If the wire-level destroy raises, the finally block still clears
    ``_destroying`` so the flag doesn't get stuck True forever."""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")

    async def _failing_destroy(**kwargs: Any) -> None:
        raise RuntimeError("node disconnected")
    node.destroy_container = _failing_destroy  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="node disconnected"):
        await coord.destroy(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
        )

    assert coord.is_destroying(session.rollout_id) is False


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 audit M1 (round 2) — _acquiring_ids in-flight tracking
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_acquiring_ids_populated_during_acquire_cleared_after() -> None:
    """``list_acquiring_ids()`` contains the rollout_id while the
    acquire's wire call is in flight and is empty once the acquire
    completes (handed off to ``_sessions``). This is the liveness
    signal the raw-GC reconciler unions with ``list_sessions`` so a
    legitimately-queued acquire is never swept as a SQLite ghost."""
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    assert coord.list_acquiring_ids() == set()

    acquire_started = asyncio.Event()
    acquire_may_finish = asyncio.Event()
    original_acquire = node.acquire_container

    async def _slow_acquire(**kwargs: Any) -> Any:
        acquire_started.set()
        await acquire_may_finish.wait()
        return await original_acquire(**kwargs)

    node.acquire_container = _slow_acquire  # type: ignore[method-assign]

    task = asyncio.create_task(coord.acquire(image="busybox:1"))
    await acquire_started.wait()

    # Wire acquire in flight — exactly one rollout_id tracked.
    in_flight = coord.list_acquiring_ids()
    assert len(in_flight) == 1
    assert coord.list_sessions() == [], "no session until acquire completes"

    acquire_may_finish.set()
    session = await task

    # Handed off to _sessions; in-flight set cleared.
    assert coord.list_acquiring_ids() == set()
    assert [s.rollout_id for s in coord.list_sessions()] == [
        session.rollout_id,
    ]
    # The snapshot is a copy — mutating it can't corrupt coordinator state.
    in_flight.add("bogus")
    assert coord.list_acquiring_ids() == set()


@pytest.mark.asyncio
async def test_acquiring_id_cleared_when_acquire_fails() -> None:
    """A failed acquire must drop the rollout_id from
    ``_acquiring_ids`` (the ``finally`` clause), otherwise the row
    would stay permanently protected from reconciliation."""
    node = _FakeNodeTransport(
        raise_on_acquire=RuntimeError("node acquire blew up"),
    )
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    with pytest.raises(RuntimeError, match="node acquire blew up"):
        await coord.acquire(image="busybox:1")

    assert coord.list_acquiring_ids() == set(), (
        "a failed acquire must not leave a stale _acquiring_ids entry"
    )


@pytest.mark.asyncio
async def test_acquire_cancellation_runs_full_cleanup() -> None:
    """Audit M1 (round 3): a cancelled acquire coroutine raises
    ``CancelledError`` — a ``BaseException``, NOT an ``Exception``.
    The cleanup must run the SAME path as an ordinary failure:
    drop the ``_acquiring_ids`` marker, release the ``_pending``
    scheduler reservation, and seal the persisted row terminal.
    Otherwise a cancelled acquire leaks a scheduler slot until
    process restart and leaves a stuck ``acquiring`` row. A clean
    teardown seals ``cancelled`` (the cancel succeeded)."""
    captured: list[dict] = []

    class _TrackingState:
        def find_registered_preferred_home(self, image: str) -> None:
            return None

        def record_raw_rollout(self, record: Any) -> None:
            pass

        def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
            captured.append({"rollout_id": rollout_id, **fields})

    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched, state=_TrackingState())

    acquire_started = asyncio.Event()
    acquire_may_finish = asyncio.Event()
    original_acquire = node.acquire_container

    async def _slow_acquire(**kwargs: Any) -> Any:
        acquire_started.set()
        await acquire_may_finish.wait()  # never set — we cancel instead
        return await original_acquire(**kwargs)

    node.acquire_container = _slow_acquire  # type: ignore[method-assign]

    task = asyncio.create_task(coord.acquire(image="busybox:1"))
    await acquire_started.wait()
    # Mid-flight: placement minted, row in flight.
    assert len(coord.list_acquiring_ids()) == 1
    assert sched.commit_calls == [] and sched.release_calls == []

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # In-flight marker dropped.
    assert coord.list_acquiring_ids() == set(), (
        "a cancelled acquire must not leave a stale _acquiring_ids entry"
    )
    # Scheduler reservation released (not committed) — no leaked slot.
    assert sched.commit_calls == []
    assert sched.release_calls == ["fake-res-0"]
    assert coord.iter_load_entries() == []
    # The cancel was torn down cleanly (reservation released, no leaked
    # slot) → the cancellation SUCCEEDED → status 'cancelled', not
    # 'failed'. The reason is recorded (not the useless bare
    # "CancelledError: " — a CancelledError stringifies to "").
    terminal = [
        u for u in captured if u.get("status") in ("failed", "cancelled")
    ]
    assert len(terminal) == 1
    assert terminal[0]["status"] == "cancelled"
    err = terminal[0]["error"]
    assert "cancelled" in err.lower()
    assert "unwound cleanly" in err.lower()
    assert err.strip() not in ("CancelledError:", "CancelledError")
    # This test cancels AFTER placement was minted (the reservation was
    # released above), so the reason reflects that and reports the wait.
    assert "after placement" in err
    assert "wait" in err


@pytest.mark.asyncio
async def test_acquire_cancellation_with_failed_teardown_is_failed() -> None:
    """If the cancellation's teardown itself errors (releasing the
    scheduler reservation raises → a slot may be leaked), the cancel did
    NOT succeed, so the row is sealed ``failed`` (not ``cancelled``) with
    the teardown error recorded. Status reflects the *outcome* of the
    cancellation, not the fact that a cancel was requested."""
    captured: list[dict] = []

    class _TrackingState:
        def find_registered_preferred_home(self, image: str) -> None:
            return None

        def record_raw_rollout(self, record: Any) -> None:
            pass

        def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
            captured.append({"rollout_id": rollout_id, **fields})

    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node], raise_on_release=True)
    coord = RawContainerCoordinator(scheduler=sched, state=_TrackingState())

    acquire_started = asyncio.Event()
    acquire_may_finish = asyncio.Event()
    original_acquire = node.acquire_container

    async def _slow_acquire(**kwargs: Any) -> Any:
        acquire_started.set()
        await acquire_may_finish.wait()
        return await original_acquire(**kwargs)

    node.acquire_container = _slow_acquire  # type: ignore[method-assign]

    task = asyncio.create_task(coord.acquire(image="busybox:1"))
    await acquire_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    terminal = [
        u for u in captured if u.get("status") in ("failed", "cancelled")
    ]
    assert len(terminal) == 1
    assert terminal[0]["status"] == "failed"  # cancellation teardown failed
    err = terminal[0]["error"]
    assert "cancellation failed" in err.lower()
    assert "leaked" in err.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 fix #2 — wire ceiling: destroy_container passes timeout_s=300.0
# (tested via RemoteNodeTransport + monkeypatch on _send_and_wait)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_destroy_container_passes_300s_timeout_to_send_and_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RemoteNodeTransport.destroy_container`` must pass
    ``timeout_s=300.0`` to ``_send_and_wait``. The original value was
    30 s; under heavy parallel teardowns with overlay-fs serialisation
    that blew the ceiling while docker was still in ``container.remove``.
    """
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.control.grpc_endpoint import RemoteNodeTransport, _MonotonicCounter
    from xrlenv.node.hw_probe import HardwareInfo

    hw = HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )
    transport = RemoteNodeTransport(
        node_id="test-node",
        backends=["docker"],
        hardware=hw,
        outbox=asyncio.Queue(),
        stream_epoch="test-epoch",
        control_instance_id="ctrl-1",
        control_seq=_MonotonicCounter(),
    )

    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(
            command_id=command_id, status=pb.ReplyStatus.OK,
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.destroy_container(
        rollout_id="r-1", container_id="c" * 12,
    )
    assert captured["timeout_s"] == 300.0, (
        f"destroy_container must use 300s wire ceiling, got {captured['timeout_s']}"
    )


@pytest.mark.asyncio
async def test_force_destroy_raw_container_passes_300s_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M2: ``force_destroy_raw_container`` must use the same
    300 s wire ceiling as the regular destroy path. It's a
    ``docker rm -f`` against the same daemon, driven by the raw-GC
    reconciler for node-only orphans after a control-plane restart —
    exactly when teardowns are slowest. The old 30 s ceiling would
    time out the cleanup it most needs to finish.
    """
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.control.grpc_endpoint import RemoteNodeTransport, _MonotonicCounter
    from xrlenv.node.hw_probe import HardwareInfo

    hw = HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )
    transport = RemoteNodeTransport(
        node_id="test-node",
        backends=["docker"],
        hardware=hw,
        outbox=asyncio.Queue(),
        stream_epoch="test-epoch",
        control_instance_id="ctrl-1",
        control_seq=_MonotonicCounter(),
    )

    captured: dict[str, Any] = {}

    async def _fake_send_and_wait(
        msg: Any, command_id: str, *, timeout_s: float | None = None,
    ) -> Any:
        captured["timeout_s"] = timeout_s
        return pb.CommandReply(
            command_id=command_id, status=pb.ReplyStatus.OK,
        )

    monkeypatch.setattr(transport, "_send_and_wait", _fake_send_and_wait)
    await transport.force_destroy_raw_container(container_id="c" * 12)
    assert captured["timeout_s"] == 300.0, (
        f"force_destroy_raw_container must use the 300s wire ceiling "
        f"(audit M2), got {captured['timeout_s']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 fix #1 — AdmissionQueue routing for raw acquires
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeAdmission:
    """Minimal AdmissionQueue surface the coordinator calls."""

    acquire_calls: list[dict] = field(default_factory=list)
    kick_calls: int = 0
    raise_on_acquire: Exception | None = None
    _placement: Any = None  # set by test to control what acquire returns

    async def acquire(
        self,
        *,
        manifest: Any,
        task_key: Any = None,
        request_id: Any = None,
        owner_id: str = "default",
        backend: Any = None,
        timeout_s: float | None = None,
    ) -> Any:
        self.acquire_calls.append({
            "manifest_image": getattr(manifest, "image", None),
            "task_key": task_key,
            "request_id": request_id,
            "owner_id": owner_id,
            "backend": backend,
            "timeout_s": timeout_s,
        })
        if self.raise_on_acquire is not None:
            raise self.raise_on_acquire
        return self._placement

    def kick(self) -> None:
        self.kick_calls += 1

    def queue_status(self, request_id: str) -> tuple[int, int, str]:
        return (0, 0, "not_in_queue")


@pytest.mark.asyncio
async def test_acquire_without_admission_uses_scheduler_place() -> None:
    """When ``admission=None`` (legacy path), acquire calls
    ``scheduler.place`` directly — existing scheduler-based tests
    already verify most of this. This test asserts ``scheduler.place``
    is called and ``admission.acquire`` is NOT called (since no queue
    is wired)."""
    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    coord = RawContainerCoordinator(scheduler=sched, admission=None)

    await coord.acquire(image="busybox:1")

    assert len(sched.place_calls) == 1
    # No admission object was involved.


@pytest.mark.asyncio
async def test_acquire_with_admission_calls_admission_not_scheduler_place() -> None:
    """When ``admission`` is wired, ``acquire`` routes through
    ``admission.acquire`` instead of calling ``scheduler.place``
    directly. The admission queue does its own placement internally;
    the coordinator must not bypass it."""
    node = _FakeNodeTransport(node_id="node-A")
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=0.9)
    admission = _FakeAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    session = await coord.acquire(image="busybox:1", task_key="tk-1")

    assert len(admission.acquire_calls) == 1
    call = admission.acquire_calls[0]
    assert call["task_key"] == "tk-1"
    assert call["backend"] == "docker"
    # Scheduler.place must NOT have been called (admission owns that).
    assert sched.place_calls == []
    assert session.node_id == "node-A"


@pytest.mark.asyncio
async def test_destroy_calls_admission_kick_on_success() -> None:
    """``destroy()`` must call ``admission.kick()`` after popping the
    session so a waiter parked on CapacityExhausted can re-place
    against the freshly-released capacity."""
    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=1.0)
    admission = _FakeAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0  # reset counter after acquire's potential kick

    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
    )

    assert admission.kick_calls == 1


@pytest.mark.asyncio
async def test_destroy_does_not_kick_admission_on_failure() -> None:
    """audit H8 / invariant 2: a FAILED destroy did NOT free capacity (the session is
    retained), so admission must NOT be woken — waking it would let a parked acquire re-place
    against capacity that is still charged. (Was ``..._calls_admission_kick_on_failure`` — the
    old ``finally`` kicked unconditionally.)"""
    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=1.0)
    admission = _FakeAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0

    async def _failing_destroy(**kwargs: Any) -> None:
        raise RuntimeError("node went away")
    node.destroy_container = _failing_destroy  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="node went away"):
        await coord.destroy(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
        )

    # kick must NOT have fired — capacity was not released.
    assert admission.kick_calls == 0
    # A clean, node-confirmed destroy DOES free + kick.
    node.destroy_container = _FakeNodeTransport().destroy_container  # type: ignore[method-assign]
    await coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id,
    )
    assert admission.kick_calls == 1
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_destroy_is_single_flight() -> None:
    # audit M14: two concurrent destroys of the SAME rollout must issue ONE wire call, ONE
    # terminal seal, and ONE admission kick — the second joins/short-circuits the first.
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0

    calls = 0
    gate = asyncio.Event()

    async def _slow_destroy(**_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        await gate.wait()   # hold the wire call so a second destroy races it
    node.destroy_container = _slow_destroy  # type: ignore[method-assign]

    a = asyncio.create_task(coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(1000):   # let A enter + mark _destroying + block on the gate
        await asyncio.sleep(0)
        if coord.is_destroying(session.rollout_id):
            break
    assert coord.is_destroying(session.rollout_id)

    # B (concurrent) JOINS the owner (audit M14): it must NOT issue a second wire call while A
    # holds the gate, and it must park (await A's outcome) rather than optimistically returning.
    b = asyncio.create_task(coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(1000):
        await asyncio.sleep(0)
    assert calls == 1          # B issued no second wire call
    assert not b.done()        # B is parked, joining A's in-flight destroy

    gate.set()
    await asyncio.gather(a, b)
    assert calls == 1                    # still ONE wire call
    assert admission.kick_calls == 1     # ONE kick
    assert coord.list_sessions() == []   # finalized exactly once


@pytest.mark.asyncio
async def test_destroy_finalization_completes_despite_cancellation() -> None:
    # audit M14: if the destroy coroutine is CANCELLED after the wire teardown is node-confirmed
    # but before finalization finishes, finalization MUST still complete (it runs under
    # asyncio.shield) — otherwise the row seals while the session/capacity stay charged and
    # _destroying stays set (which GC + seal_orphan then skip forever: wedged).
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0

    wire_gate = asyncio.Event()

    async def _gated_wire(**_kwargs: Any) -> None:
        await wire_gate.wait()   # hold the (successful) wire call
    node.destroy_container = _gated_wire  # type: ignore[method-assign]

    task = asyncio.create_task(coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(1000):        # let it reach the wire call
        await asyncio.sleep(0)
        if coord.is_destroying(session.rollout_id):
            break
    # Hold the coordinator lock so the (shielded) finalization blocks acquiring it once the wire
    # completes — giving us a deterministic window to cancel mid-finalization.
    await coord._lock.acquire()
    wire_gate.set()              # wire completes → enter shielded finalization → blocks on lock
    for _ in range(200):
        await asyncio.sleep(0)
    task.cancel()               # cancel the destroy while finalization is shielded + blocked
    for _ in range(200):
        await asyncio.sleep(0)
    coord._lock.release()       # let the shielded finalization run to completion
    for _ in range(200):
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await task
    # Despite the cancellation, finalization completed exactly once (NOT wedged):
    assert coord.list_sessions() == []                  # session dropped → capacity freed
    assert not coord.is_destroying(session.rollout_id)  # _destroying cleared
    assert admission.kick_calls == 1                    # admission kicked


@pytest.mark.asyncio
async def test_destroy_join_propagates_owner_hard_failure() -> None:
    # audit M14: a JOINED duplicate reports the owner's REAL outcome — a hard wire failure
    # re-raises for BOTH callers (not an optimistic success), and the session is retained.
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")

    gate = asyncio.Event()

    async def _slow_fail(**_kwargs: Any) -> None:
        await gate.wait()
        raise RuntimeError("wire boom")
    node.destroy_container = _slow_fail  # type: ignore[method-assign]

    a = asyncio.create_task(coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(1000):
        await asyncio.sleep(0)
        if coord.is_destroying(session.rollout_id):
            break
    b = asyncio.create_task(coord.destroy(     # joins the in-flight owner
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(200):
        await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(a, b, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)     # BOTH see the failure
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]  # retained
    assert not coord.is_destroying(session.rollout_id)           # cleared for a retry


@pytest.mark.asyncio
async def test_cancelled_joiner_does_not_poison_siblings() -> None:
    # audit M14: joiners await the shared owner-future under asyncio.shield, so cancelling ONE
    # joiner must NOT cancel the shared future out from under the owner + sibling joiners.
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")

    gate = asyncio.Event()

    async def _slow(**_kwargs: Any) -> None:
        await gate.wait()
    node.destroy_container = _slow  # type: ignore[method-assign]

    a = asyncio.create_task(coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(1000):
        await asyncio.sleep(0)
        if coord.is_destroying(session.rollout_id):
            break
    b = asyncio.create_task(coord.destroy(     # joiner 1
        rollout_id=session.rollout_id, container_id=session.container_id))
    c = asyncio.create_task(coord.destroy(     # joiner 2
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(200):
        await asyncio.sleep(0)
    b.cancel()                                  # cancel ONE joiner
    for _ in range(200):
        await asyncio.sleep(0)
    gate.set()
    await a
    with pytest.raises(asyncio.CancelledError):
        await b
    await c                                     # sibling still sees the owner's SUCCESS
    assert coord.list_sessions() == []          # finalized exactly once


@pytest.mark.asyncio
async def test_node_loss_resolves_joiner_with_node_loss_outcome() -> None:
    # audit M14: when a node is lost while a destroy is in flight, a JOINED caller must see the
    # node-loss outcome (the CP's authoritative terminal decision), not a false "success".
    from xrlenv.errors import NodeLost
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")

    gate = asyncio.Event()

    async def _slow(**_kwargs: Any) -> None:
        await gate.wait()
    node.destroy_container = _slow  # type: ignore[method-assign]

    owner = asyncio.create_task(coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(1000):
        await asyncio.sleep(0)
        if coord.is_destroying(session.rollout_id):
            break
    joiner = asyncio.create_task(coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id))
    for _ in range(200):
        await asyncio.sleep(0)

    await coord.handle_node_lost(session.node_id)   # node lost mid-destroy

    with pytest.raises(NodeLost):
        await joiner                                 # joiner sees node-loss, not false success
    gate.set()                                       # release the owner's wire
    with pytest.raises(NodeLost):
        await owner                                  # owner CONVERGES on the same node-loss (M14)
    assert coord.list_sessions() == []               # session sealed node_lost exactly once


@pytest.mark.asyncio
async def test_finalize_converges_on_node_loss_when_session_finalized_mid_wire() -> None:
    # audit M14 (generation-safe + owner convergence): if the session is torn down by node loss
    # between the wire confirm and finalization, finalize must NOT re-seal / re-kick, and the
    # OWNER must raise NodeLost (converge) rather than return a stale success.
    from xrlenv.errors import NodeLost
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0

    async def _wire_then_lose(**_kwargs: Any) -> None:
        # simulate handle_node_lost racing: pop the session mid-wire, before finalization.
        async with coord._lock:
            coord._sessions.pop(session.rollout_id, None)
    node.destroy_container = _wire_then_lose  # type: ignore[method-assign]

    with pytest.raises(NodeLost):
        await coord.destroy(rollout_id=session.rollout_id, container_id=session.container_id)
    assert admission.kick_calls == 0        # finalize saw the session gone → no second kick
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_destroy_converges_on_node_loss_when_wire_FAILS_after_loss() -> None:
    # audit M14 (late-exception convergence): a racing handle_node_lost can pop the session +
    # resolve the single-flight future with NodeLost WHILE the owner's wire destroy is in flight;
    # the wire then surfaces its OWN hard failure. The owner must CONVERGE on NodeLost (matching
    # every joiner) rather than re-raise its now-moot failure.
    from xrlenv.errors import NodeLost
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")

    async def _lose_then_fail(**_kwargs: Any) -> None:
        await coord.handle_node_lost(session.node_id)   # node lost mid-wire (pops + resolves)
        raise RuntimeError("wire teardown failed after the node was already lost")
    node.destroy_container = _lose_then_fail  # type: ignore[method-assign]

    with pytest.raises(NodeLost):                        # converges — NOT RuntimeError
        await coord.destroy(rollout_id=session.rollout_id, container_id=session.container_id)
    assert coord.list_sessions() == []                  # sealed node_lost exactly once


@pytest.mark.asyncio
async def test_destroy_converges_on_node_loss_when_wire_TIMES_OUT_after_loss() -> None:
    # audit M14 (late-timeout convergence): a CONSUMER destroy (reason=None) normally returns
    # success on a bare timeout, but if node loss already superseded the teardown the owner must
    # raise NodeLost — never a false success while joiners saw NodeLost.
    from xrlenv.errors import NodeCommandTimeout, NodeLost
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")

    async def _lose_then_timeout(**_kwargs: Any) -> None:
        await coord.handle_node_lost(session.node_id)
        raise NodeCommandTimeout("node x: command y timed out after 300.0s")
    node.destroy_container = _lose_then_timeout  # type: ignore[method-assign]

    with pytest.raises(NodeLost):                        # consumer path converges, no false success
        await coord.destroy(rollout_id=session.rollout_id, container_id=session.container_id)
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_seal_orphan_noop_when_session_already_finalized() -> None:
    # audit M14 (generation-safe): if a concurrent destroy already finalized the rollout
    # (session popped, row sealed) between the reconciler's snapshot and now, seal_orphan must
    # be a NO-OP — not re-seal the terminal row or fire a second admission kick.
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")
    await coord.destroy(rollout_id=session.rollout_id, container_id=session.container_id)
    admission.kick_calls = 0
    # a stale reconciler snapshot calls seal_orphan for the already-gone session:
    await coord.seal_orphan(
        rollout_id=session.rollout_id, container_id=session.container_id,
    )
    assert admission.kick_calls == 0        # no second kick
    assert coord.list_sessions() == []      # still gone (not re-added / re-sealed)


@pytest.mark.asyncio
async def test_drop_orphan_session_generation_safe_with_cleanup() -> None:
    # audit Low: the raw-GC fallback drop must be generation-safe (only the same container) AND
    # do proper cleanup + admission kick — not a bare _sessions pop that leaks fleet/admission.
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0

    # a DIFFERENT container_id (newer generation reused the rollout_id) → NOT dropped.
    await coord.drop_orphan_session(session.rollout_id, "different-container")
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]
    assert admission.kick_calls == 0

    # the matching container_id → dropped WITH an admission kick.
    await coord.drop_orphan_session(session.rollout_id, session.container_id)
    assert coord.list_sessions() == []
    assert admission.kick_calls == 1


@pytest.mark.asyncio
async def test_seal_orphan_skips_when_destroy_in_flight() -> None:
    # audit M14: the reconciler's confirmed-absence seal must NOT double-finalize a rollout a
    # destroy is already tearing down (would double seal/pop/kick).
    node = _FakeNodeTransport()
    admission = _FakeAdmission(_placement=_FakePlacement(node=node, backend="docker", score=1.0))
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]), admission=admission)
    session = await coord.acquire(image="busybox:1")
    admission.kick_calls = 0

    # simulate mid-destroy: register an unresolved single-flight future (audit M14 — _destroying
    # is now a rollout_id -> Future owner map, not a set).
    coord._destroying[session.rollout_id] = (   # type: ignore[attr-defined]
        asyncio.get_running_loop().create_future())
    await coord.seal_orphan(
        rollout_id=session.rollout_id, container_id=session.container_id,
    )
    # skipped — session retained, no kick (the in-flight destroy owns finalization).
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]
    assert admission.kick_calls == 0


@pytest.mark.asyncio
async def test_acquire_propagates_capacity_exhausted_from_admission() -> None:
    """When the admission queue raises ``CapacityExhausted`` (queue
    timeout), the raw ``acquire`` propagates it. The ``acquiring`` row must be
    flipped to ``status=capacity_rejected`` — the platform declined to place
    (backpressure), distinct from ``failed`` so a retried acquire isn't scored
    as a failure. This is the production path the sweep's infra-retry loop
    hits under load."""
    from xrlenv.errors import CapacityExhausted

    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    admission = _FakeAdmission(
        raise_on_acquire=CapacityExhausted("queue timed out"),
    )

    update_calls: list[dict] = []

    class _TrackingState:
        def find_registered_preferred_home(self, image: str) -> None:
            return None

        def record_build_plan(self, **kwargs: Any) -> None:
            pass

        def record_assignment(self, record: Any) -> None:
            pass

        def record_raw_rollout(self, record: Any) -> None:
            pass

        def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
            update_calls.append({"rollout_id": rollout_id, **fields})

    coord = RawContainerCoordinator(
        scheduler=sched, admission=admission, state=_TrackingState(),
    )

    with pytest.raises(CapacityExhausted, match="queue timed out"):
        await coord.acquire(image="busybox:1")

    # The acquiring row must have been sealed capacity_rejected (NOT failed),
    # carrying the original CapacityExhausted text in the reason.
    rejected = [
        u for u in update_calls if u.get("status") == "capacity_rejected"
    ]
    assert len(rejected) == 1, (
        f"expected one status=capacity_rejected update; got {update_calls}"
    )
    assert not [u for u in update_calls if u.get("status") == "failed"]
    assert "queue timed out" in rejected[0]["error"]


@pytest.mark.asyncio
async def test_acquire_passes_queue_timeout_s_to_admission() -> None:
    """``RawContainerCoordinator.acquire(queue_timeout_s=...)`` must
    propagate to ``admission.acquire(timeout_s=...)`` so consumers can
    bound how long they're willing to wait for capacity."""
    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=1.0)
    admission = _FakeAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    await coord.acquire(image="busybox:1", queue_timeout_s=1234.5)

    assert admission.acquire_calls[0]["timeout_s"] == 1234.5


@pytest.mark.asyncio
async def test_acquire_default_queue_timeout_is_24h() -> None:
    """Stage 2: the default queue timeout is the 24 h backstop
    (``DEFAULT_QUEUE_TIMEOUT_S``). Waiting in the admission queue is
    not a failure — a queued request consumes no cluster resources —
    so the default is a long backstop, not a deadline. A consumer
    that wants fail-fast passes a small ``queue_timeout_s`` explicitly.
    """
    from xrlenv.control.admission import DEFAULT_QUEUE_TIMEOUT_S

    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=1.0)
    admission = _FakeAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    await coord.acquire(image="busybox:1")

    assert admission.acquire_calls[0]["timeout_s"] == DEFAULT_QUEUE_TIMEOUT_S
    assert DEFAULT_QUEUE_TIMEOUT_S == 86_400.0


@pytest.mark.asyncio
async def test_session_deadline_starts_after_queue_wait() -> None:
    """Stage 2 (P2): the session-lifetime clock starts at admission,
    not at request submit — time spent waiting in the admission queue
    must not erode ``session_deadline_s``."""
    import time

    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=1.0)

    class _SlowAdmission(_FakeAdmission):
        async def acquire(self, **kwargs: Any) -> Any:
            await asyncio.sleep(0.3)  # simulate an admission-queue wait
            return await super().acquire(**kwargs)

    admission = _SlowAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    t_before = time.time()
    session = await coord.acquire(image="busybox:1", session_deadline_s=100.0)

    # ``deadline_at`` is computed at session creation — AFTER the 0.3 s
    # queue wait — so the full 100 s budget sits on top of the wait. If
    # queue-wait eroded the deadline it would be ~t_before+100; instead
    # it is ~t_before+0.3+100.
    assert session.deadline_at - t_before > 100.0


@pytest.mark.asyncio
async def test_acquire_warns_when_queued_wait_exceeds_threshold(
    caplog: Any,
) -> None:
    """When the queue wait crosses 1 s, the coordinator logs a WARN
    so the operator sees that the consumer is over-requesting
    concurrency relative to cluster capacity. Below the threshold,
    log level stays INFO."""
    import logging

    node = _FakeNodeTransport()
    sched = _FakeScheduler(nodes=[node])
    fake_placement = _FakePlacement(node=node, backend="docker", score=1.0)

    # Slow admission — sleeps 1.2 s before returning, simulating real
    # queue contention.
    class _SlowAdmission(_FakeAdmission):
        async def acquire(self, **kwargs: Any) -> Any:
            await asyncio.sleep(1.2)
            return await super().acquire(**kwargs)

    admission = _SlowAdmission(_placement=fake_placement)
    coord = RawContainerCoordinator(scheduler=sched, admission=admission)

    with caplog.at_level(
        logging.WARNING, logger="xrlenv.control.raw_container_service",
    ):
        await coord.acquire(image="busybox:1")

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("admit-queued" in r.message for r in warnings), (
        f"expected admit-queued WARN, got {[r.message for r in warnings]}"
    )
    assert any(
        "consumer concurrency exceeds cluster capacity" in r.message
        for r in warnings
    )


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 — raw-session deadline (lifetime cap)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_sets_default_session_deadline() -> None:
    """With no explicit ``session_deadline_s``, acquire stamps the
    coordinator's default cap onto the session — the leak backstop
    the raw-GC reconciler enforces."""
    import time as _t

    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    before = _t.time()
    session = await coord.acquire(image="busybox:1")
    after = _t.time()

    assert (before + RAW_SESSION_DEADLINE_DEFAULT_S
            <= session.deadline_at
            <= after + RAW_SESSION_DEADLINE_DEFAULT_S)


@pytest.mark.asyncio
async def test_acquire_honors_explicit_session_deadline_s() -> None:
    """A consumer-supplied ``session_deadline_s`` overrides the
    default cap."""
    import time as _t

    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))

    before = _t.time()
    session = await coord.acquire(
        image="busybox:1", session_deadline_s=120.0,
    )
    after = _t.time()

    assert before + 120.0 <= session.deadline_at <= after + 120.0


@pytest.mark.asyncio
async def test_acquire_default_session_deadline_default_is_4h() -> None:
    """Pin the default cap at 4 h — a leak backstop well above any
    single grading task, not a limit consumers normally hit."""
    assert RAW_SESSION_DEADLINE_DEFAULT_S == 4 * 60 * 60.0


@pytest.mark.asyncio
async def test_destroy_with_reason_seals_row_reaped() -> None:
    """``destroy(reason=...)`` on a clean teardown seals the
    raw_rollouts row ``reaped`` (NOT ``failed``) with the reason — so a
    force-reap is distinguishable from a consumer-initiated destroy
    (which seals ``released``) and from a genuine work failure."""
    updates: list[dict] = []

    class _State:
        def find_registered_preferred_home(self, image: str) -> None:
            return None

        def record_raw_rollout(self, record: Any) -> None:
            pass

        def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
            updates.append({"rollout_id": rollout_id, **fields})

    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=_State(),
    )
    session = await coord.acquire(image="busybox:1")

    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
        reason="session deadline exceeded (overdue 12s)",
    )

    terminal = [
        u for u in updates if u.get("status") in ("reaped", "failed", "released")
    ]
    assert terminal and terminal[-1]["status"] == "reaped"
    assert "deadline exceeded" in terminal[-1]["error"]


@pytest.mark.asyncio
async def test_destroy_without_reason_still_seals_released() -> None:
    """A normal consumer destroy (no reason) still seals ``released``."""
    updates: list[dict] = []

    class _State:
        def find_registered_preferred_home(self, image: str) -> None:
            return None

        def record_raw_rollout(self, record: Any) -> None:
            pass

        def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
            updates.append({"rollout_id": rollout_id, **fields})

    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=_State(),
    )
    session = await coord.acquire(image="busybox:1")

    await coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id,
    )

    terminal = [u for u in updates if u.get("status") in ("failed", "released")]
    assert terminal and terminal[-1]["status"] == "released"


# ──────────────────────────────────────────────────────────────────────────────
# #2 — resilient teardown: a consumer destroy that *times out* (node briefly
# I/O-wedged) must NOT false-fail the rollout. Seal ``released``, swallow the
# timeout, defer the actual container removal to the raw-GC orphan reconciler.
# A reaper-driven destroy (reason set) keeps the old raise/seal-failed path.
# ──────────────────────────────────────────────────────────────────────────────


def _state_recording(updates: list[dict]) -> Any:
    class _State:
        def find_registered_preferred_home(self, image: str) -> None:
            return None

        def record_raw_rollout(self, record: Any) -> None:
            pass

        def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
            updates.append({"rollout_id": rollout_id, **fields})

    return _State()


@pytest.mark.asyncio
async def test_consumer_destroy_timeout_retains_session_and_does_not_raise() -> None:
    # audit H8: a consumer destroy that TIMES OUT is not node-confirmed. It still doesn't raise
    # to the consumer (work is done), but the session + capacity charge are RETAINED (not
    # sealed released, not dropped) — teardown is deferred to the raw-GC reconciler, which
    # frees capacity only on confirmed absence. (Was: sealed released + dropped the session.)
    from xrlenv.errors import NodeCommandTimeout

    updates: list[dict] = []
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=_state_recording(updates),
    )
    session = await coord.acquire(image="busybox:1")

    async def _timeout_destroy(**kwargs: Any) -> None:
        raise NodeCommandTimeout("node x: command y timed out after 300.0s")
    node.destroy_container = _timeout_destroy  # type: ignore[method-assign]

    # No exception propagates to the consumer — the work is done.
    await coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id,
    )

    # Session RETAINED (capacity held) — NOT sealed terminal here.
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]
    assert len(coord.iter_load_entries()) == 1
    assert not [u for u in updates if u.get("status") in ("failed", "released", "reaped")]


@pytest.mark.asyncio
async def test_reaper_destroy_timeout_reraises_and_retains_session() -> None:
    # audit H8: a reaper-driven destroy (reason set) that times out re-raises (so the reaper
    # records it) AND retains the session + capacity — not node-confirmed, so no seal/free.
    from xrlenv.errors import NodeCommandTimeout

    updates: list[dict] = []
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=_state_recording(updates),
    )
    session = await coord.acquire(image="busybox:1")

    async def _timeout_destroy(**kwargs: Any) -> None:
        raise NodeCommandTimeout("node x: command y timed out after 300.0s")
    node.destroy_container = _timeout_destroy  # type: ignore[method-assign]

    with pytest.raises(NodeCommandTimeout):
        await coord.destroy(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
            reason="session deadline exceeded",
        )
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]
    assert not [u for u in updates if u.get("status") in ("failed", "released", "reaped")]


@pytest.mark.asyncio
async def test_consumer_destroy_hard_failure_reraises_and_retains_session() -> None:
    # audit H8: a non-timeout failure (transport error) is not node-confirmed — re-raise AND
    # RETAIN the session + capacity (was: re-raise + seal ``failed`` + drop the session, an
    # invariant-2 violation). The raw-GC reconciler / a retry re-attempts teardown.
    updates: list[dict] = []
    node = _FakeNodeTransport()
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node]), state=_state_recording(updates),
    )
    session = await coord.acquire(image="busybox:1")

    async def _raising_destroy(**kwargs: Any) -> None:
        raise RuntimeError("transport boom")
    node.destroy_container = _raising_destroy  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="transport boom"):
        await coord.destroy(
            rollout_id=session.rollout_id, container_id=session.container_id,
        )
    assert [s.rollout_id for s in coord.list_sessions()] == [session.rollout_id]
    assert len(coord.iter_load_entries()) == 1
    assert not [u for u in updates if u.get("status") in ("failed", "released", "reaped")]


@pytest.mark.asyncio
async def test_handle_node_lost_discards_heartbeat_membership() -> None:
    # audit Low: node loss must leave NO trace in _heartbeated (the finalize/seal paths already
    # discard it), else the id lingers and could falsely mark a (hypothetically reused) rollout as
    # already heartbeated.
    node = _FakeNodeTransport(node_id="node-A")
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[node]))
    session = await coord.acquire(image="busybox:1")
    coord._heartbeated.add(session.rollout_id)

    await coord.handle_node_lost("node-A")
    assert session.rollout_id not in coord._heartbeated


# ──────────────────────────────────────────────────────────────────────────────
# Open item 2 — RawContainerCoordinator.handle_node_lost seals raw sessions on a
# lost node so their rows go terminal instead of lingering 'running' forever
# (only gym/step rollouts were sealed on node loss pre-fix).
# ──────────────────────────────────────────────────────────────────────────────


def _raw_session(rid: str, node: Any, node_id: str) -> RawContainerSession:
    import datetime as _dt
    return RawContainerSession(
        rollout_id=rid, node=node, node_id=node_id,
        container_id=f"c-{rid}", container_name=f"name-{rid}",
        image="busybox:1", created_at=_dt.datetime.now(_dt.UTC),
    )


def _node_lost_coord() -> tuple[Any, list[dict]]:
    captured: list[dict] = []

    class _TrackingState:
        def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
            captured.append({"rollout_id": rollout_id, **fields})

    node_a = _FakeNodeTransport(node_id="node-A")
    node_b = _FakeNodeTransport(node_id="node-B")
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(nodes=[node_a, node_b]),
        state=_TrackingState(),
    )
    coord._sessions["a1"] = _raw_session("a1", node_a, "node-A")
    coord._sessions["a2"] = _raw_session("a2", node_a, "node-A")
    coord._sessions["b1"] = _raw_session("b1", node_b, "node-B")
    return coord, captured


@pytest.mark.asyncio
async def test_handle_node_lost_seals_only_that_nodes_sessions() -> None:
    coord, captured = _node_lost_coord()

    sealed = await coord.handle_node_lost("node-A")

    assert sealed == 2
    # node-A sessions dropped (frees scheduler load via iter_load_entries);
    # node-B untouched.
    assert set(coord._sessions) == {"b1"}
    assert {u["rollout_id"] for u in captured} == {"a1", "a2"}
    for u in captured:
        assert u["status"] == "failed"
        assert "node_lost" in u["error"]
        assert u.get("finished_at") is not None


@pytest.mark.asyncio
async def test_handle_node_lost_is_transport_scoped() -> None:
    # audit H11: a STALE old-stream close (transport=t_old) must seal only t_old's sessions —
    # a reconnected replacement (t_new, same node_id) that already re-adopted the node must NOT
    # have its live sessions sealed by the old stream's teardown.
    t_old = _FakeNodeTransport(node_id="node-A")
    t_new = _FakeNodeTransport(node_id="node-A")   # same node id, new stream generation
    coord = RawContainerCoordinator(scheduler=_FakeScheduler(nodes=[t_old]))
    coord._sessions["old"] = _raw_session("old", t_old, "node-A")
    coord._sessions["new"] = _raw_session("new", t_new, "node-A")

    sealed = await coord.handle_node_lost("node-A", transport=t_old)
    assert sealed == 1
    assert set(coord._sessions) == {"new"}         # replacement's session untouched

    # the watchdog path (no transport) still seals every session by node_id.
    sealed2 = await coord.handle_node_lost("node-A")
    assert sealed2 == 1
    assert coord._sessions == {}


@pytest.mark.asyncio
async def test_handle_node_lost_frees_scheduler_load() -> None:
    coord, _ = _node_lost_coord()
    # Before: node-A carries 2 raw sessions of load.
    before = {e.node_id for e in coord.iter_load_entries()}
    assert "node-A" in before

    await coord.handle_node_lost("node-A")

    after = [e for e in coord.iter_load_entries() if e.node_id == "node-A"]
    assert after == []  # node-A no longer charged


@pytest.mark.asyncio
async def test_handle_node_lost_is_idempotent() -> None:
    coord, _ = _node_lost_coord()
    first = await coord.handle_node_lost("node-A")
    second = await coord.handle_node_lost("node-A")
    assert (first, second) == (2, 0)


@pytest.mark.asyncio
async def test_handle_node_lost_unknown_node_is_noop() -> None:
    coord, captured = _node_lost_coord()
    sealed = await coord.handle_node_lost("node-ZZZ")
    assert sealed == 0
    assert captured == []
    assert set(coord._sessions) == {"a1", "a2", "b1"}
