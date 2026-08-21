"""Tests for the capacity-aware scheduler (spec 03 + spec 10)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.scheduler import (
    NODE_TIMEOUT_COOLDOWN_S,
    Placement,
    Scheduler,
)
from xrlenv.control.state import InMemoryStateStore, SandboxRecord
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.errors import BackendCapabilityMissing, CapacityExhausted
from xrlenv.node.hw_probe import HardwareInfo


def _manifest(
    name: str = "t",
    cpu_request: float = 0.25,
    mem_gb: int = 1,
    disk_gb: int = 1,
) -> TemplateManifest:
    return TemplateManifest(
        name=name,
        version="0.1",
        digest=f"sha256:{name}",
        image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=cpu_request,
            cpu_limit=cpu_request,
            mem_request_bytes=mem_gb * 1024**3,
            mem_limit_bytes=mem_gb * 1024**3,
            disk_request_bytes=disk_gb * 1024**3,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _hw(*, vcpus: int = 8, mem_gb: int = 32, disk_gb: int = 200) -> HardwareInfo:
    return HardwareInfo(
        vcpus=vcpus,
        mem_bytes=mem_gb * 1024**3,
        disk_bytes=disk_gb * 1024**3,
        has_kvm=True,
        has_gpu=False,
        gpu_model=None,
        kernel_version="6.0.0",
        platform="linux",
    )


def _node(
    node_id: str,
    backends: tuple[str, ...] = ("docker",),
    *,
    vcpus: int = 8,
    mem_gb: int = 32,
    disk_gb: int = 200,
    disk_state: tuple[int, int] | None = None,
    seconds_since_timeout: float | None = None,
) -> Any:
    n = MagicMock()
    n.node_id = node_id
    n.supported_backends.return_value = list(backends)
    n.hardware.return_value = _hw(vcpus=vcpus, mem_gb=mem_gb, disk_gb=disk_gb)
    # Issue #14 — explicit disk_state lets tests exercise the
    # placement gate; default leaves the MagicMock's auto-attr
    # (which the gate's strict tuple check classifies as "unknown
    # / healthy"), preserving legacy test behaviour.
    if disk_state is not None:
        n.disk_state.return_value = disk_state
    # Issue #18 (Ask #2) — node-health timeout gate. Default ``None``
    # = "never timed out" (healthy). A float exercises the gate:
    # < NODE_TIMEOUT_COOLDOWN_S → excluded, >= → recovered.
    n.seconds_since_last_command_timeout.return_value = seconds_since_timeout
    return n


def _scheduler(
    *nodes: Any,
    manifests: list[TemplateManifest] | None = None,
    state: InMemoryStateStore | None = None,
) -> Scheduler:
    catalog = TemplateCatalog()
    for m in manifests or []:
        catalog.register(m)
    return Scheduler(list(nodes), catalog=catalog, state=state or InMemoryStateStore())


# ──────────────────────────────────────────────────────────────────────────────
# Constructor + capability filter
# ──────────────────────────────────────────────────────────────────────────────


def test_scheduler_requires_at_least_one_node() -> None:
    with pytest.raises(ValueError):
        Scheduler([], catalog=TemplateCatalog(), state=InMemoryStateStore())


def test_place_rejects_when_no_node_supports_backend() -> None:
    """Backend is per-rollout user policy now (run-config), not a
    manifest field. Pass it explicitly to ``place()``."""
    m = _manifest()
    sched = _scheduler(_node("a", ("docker",)), manifests=[m])
    with pytest.raises(BackendCapabilityMissing):
        sched.place(m, backend="cubesandbox")


def test_place_filters_to_backend_capable_nodes() -> None:
    m = _manifest()
    sched = _scheduler(
        _node("cube-only", ("cubesandbox",)),
        _node("docker-node", ("docker",)),
        manifests=[m],
    )
    placement = sched.place(m, backend="docker")
    assert isinstance(placement, Placement)
    assert placement.node.node_id == "docker-node"
    assert placement.backend == "docker"


# ──────────────────────────────────────────────────────────────────────────────
# §10.x / §5.3 — runtime-aware placement
# ──────────────────────────────────────────────────────────────────────────────


def test_default_placement_uses_ordinary_docker_nodes() -> None:
    """§10.x(4) — with no container_runtime requested, an ordinary docker
    node (no sysbox advertised) is eligible; the runtime filter never fires
    so the normal path is unchanged."""
    m = _manifest()
    node = _node("docker-node", ("docker",))
    # An ordinary node that doesn't even implement supported_runtimes still
    # places fine (defensive default = runc).
    node.supported_runtimes.side_effect = AttributeError
    sched = _scheduler(node, manifests=[m])
    placement = sched.place(m)  # container_runtime defaults to None
    assert placement.node.node_id == "docker-node"


def test_runtime_filter_only_applies_when_requested() -> None:
    """§10.x(5) — ``container_runtime='sysbox-runc'`` places ONLY on nodes
    advertising it; an ordinary docker node is skipped. A default (None)
    placement still sees every backend-capable node."""
    plain = _node("plain-docker", ("docker",))
    plain.supported_runtimes.return_value = ["runc"]
    sysbox = _node("sysbox-node", ("docker",))
    sysbox.supported_runtimes.return_value = ["runc", "sysbox-runc"]
    m = _manifest()
    sched = _scheduler(plain, sysbox, manifests=[m])

    # sysbox request → only the sysbox node is eligible.
    p = sched.place(m, container_runtime="sysbox-runc")
    assert p.node.node_id == "sysbox-node"

    # default request → the plain node is still eligible (filter didn't fire).
    p2 = sched.place(m)
    assert p2.node.node_id in ("plain-docker", "sysbox-node")


def test_place_raises_when_no_node_supports_requested_runtime() -> None:
    """§5.3 — requesting a runtime no node advertises fails with a precise
    capability error (not an opaque capacity error), before any reservation."""
    plain = _node("plain-docker", ("docker",))
    plain.supported_runtimes.return_value = ["runc"]
    m = _manifest()
    sched = _scheduler(plain, manifests=[m])
    with pytest.raises(BackendCapabilityMissing, match="runtime 'sysbox-runc'"):
        sched.place(m, container_runtime="sysbox-runc")


# ──────────────────────────────────────────────────────────────────────────────
# Capacity-aware placement
# ──────────────────────────────────────────────────────────────────────────────


def test_place_picks_node_with_more_remaining_slack() -> None:
    """Two nodes both fit; the one with more headroom wins."""
    m = _manifest(cpu_request=1.0, mem_gb=1, disk_gb=1)
    big = _node("big", vcpus=32, mem_gb=128, disk_gb=400)
    small = _node("small", vcpus=4, mem_gb=8, disk_gb=50)
    sched = _scheduler(small, big, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "big"


def test_place_raises_capacity_exhausted_when_no_node_fits() -> None:
    """Saturate the only node — second placement must raise."""
    m = _manifest(cpu_request=4.0, mem_gb=1, disk_gb=1)
    state = InMemoryStateStore()
    state.insert_sandbox(
        SandboxRecord(
            sandbox_id="sb1",
            backend="docker",
            backend_ref="cid1",
            stub_endpoint="tcp://127.0.0.1:5000",
            template="t",
            node_id="only",
        )
    )
    sched = _scheduler(_node("only", vcpus=4, mem_gb=8, disk_gb=50), manifests=[m], state=state)
    with pytest.raises(CapacityExhausted):
        sched.place(m)


def test_place_uses_per_sandbox_effective_resources_for_load() -> None:
    """Slice 9b — Pattern A: a heavy per-task instance pinned to a
    node must be charged its effective resources, not the outer
    template's defaults. SandboxRecord.effective_resources_json
    carries the post-resolver-overlay snapshot; the scheduler reads
    it via _gather_cluster_load.

    Setup: outer manifest declares 0.5 CPU. A heavy 4 CPU instance is
    already running on the node (recorded with effective_resources_json
    = 4 CPU). Placing another heavy instance must fail; without the
    snapshot, the outer-manifest 0.5 CPU accounting would happily
    over-admit.
    """
    from xrlenv.backends.base import ResourceSpec

    outer = _manifest("tb2", cpu_request=0.5, mem_gb=1, disk_gb=1)
    heavy = ResourceSpec(
        cpu_request=4.0, cpu_limit=4.0,
        mem_request_bytes=4 * 1024**3, mem_limit_bytes=4 * 1024**3,
        disk_request_bytes=4 * 1024**3,
    )
    state = InMemoryStateStore()
    state.insert_sandbox(
        SandboxRecord(
            sandbox_id="sb1",
            backend="docker",
            backend_ref="cid1",
            stub_endpoint="tcp://127.0.0.1:5000",
            template="tb2",
            node_id="n1",
            effective_resources_json=heavy.model_dump_json(),
        )
    )
    # Build the candidate manifest with the same heavy resources —
    # simulating the post-overlay manifest the coordinator hands to
    # ``place()`` for the second per-task placement.
    heavy_outer = outer.model_copy(update={"resources": heavy})
    sched = _scheduler(
        _node("n1", vcpus=8, mem_gb=16, disk_gb=200),
        manifests=[outer], state=state,
    )
    # 4 CPU committed (effective) + headroom → second 4 CPU instance
    # exceeds the budget.
    with pytest.raises(CapacityExhausted):
        sched.place(heavy_outer)


def test_place_excludes_destroyed_sandboxes_from_load() -> None:
    """Capacity is released only on confirmed destroy; status='destroyed' frees it.

    Sized so that *one* sandbox of t fully consumes node n1's CPU; if
    destroyed sandboxes were still counted, the second placement would fail.
    """
    m = _manifest(cpu_request=4.0, mem_gb=1, disk_gb=1)
    state = InMemoryStateStore()
    state.insert_sandbox(
        SandboxRecord(
            sandbox_id="sb1",
            backend="docker",
            backend_ref="cid1",
            stub_endpoint="tcp://127.0.0.1:5000",
            template="t",
            node_id="n1",
            status="destroyed",
        )
    )
    sched = _scheduler(_node("n1", vcpus=8, mem_gb=8, disk_gb=50), manifests=[m], state=state)
    placement = sched.place(m)
    assert placement.node.node_id == "n1"


# ──────────────────────────────────────────────────────────────────────────────
# Per-task fairness cap
# ──────────────────────────────────────────────────────────────────────────────


def test_place_routes_around_per_task_cap() -> None:
    """Node A has 4 of task X already → scheduler picks node B even though A is bigger."""
    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=1)
    state = InMemoryStateStore()
    # 4 sandboxes on A, all running rollouts that share task_key="prompt-X".
    from xrlenv.control.state import RolloutRecord
    from xrlenv.types import RolloutStatus

    for i in range(4):
        rid = f"r{i}"
        state.insert_rollout(
            RolloutRecord(
                rollout_id=rid,
                template="t",
                status=RolloutStatus.RUNNING,
                task_key="prompt-X",
                sandbox_id=f"sb{i}",
                node_id="A",
            )
        )
        state.insert_sandbox(
            SandboxRecord(
                sandbox_id=f"sb{i}",
                backend="docker",
                backend_ref=f"cid{i}",
                stub_endpoint="tcp://127.0.0.1:5000",
                template="t",
                node_id="A",
                rollout_id=rid,
            )
        )

    sched = _scheduler(
        _node("A", vcpus=64, mem_gb=512, disk_gb=2000),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[m],
        state=state,
    )
    placement = sched.place(m, task_key="prompt-X")
    assert placement.node.node_id == "B"


def test_place_ignores_per_task_cap_when_task_key_is_none() -> None:
    """Without a task_key the scheduler doesn't apply max_runs_per_task."""
    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=1)
    state = InMemoryStateStore()
    for i in range(8):
        state.insert_sandbox(
            SandboxRecord(
                sandbox_id=f"sb{i}",
                backend="docker",
                backend_ref=f"cid{i}",
                stub_endpoint="tcp://127.0.0.1:5000",
                template="t",
                node_id="A",
            )
        )
    sched = _scheduler(
        _node("A", vcpus=64, mem_gb=512, disk_gb=2000),
        manifests=[m],
        state=state,
    )
    placement = sched.place(m)  # no task_key
    assert placement.node.node_id == "A"


# ──────────────────────────────────────────────────────────────────────────────
# Concurrent placements + commit/release lifecycle
#
# The scheduler must respect ``max_runs_per_task`` even when N placements race
# in before any sandbox has been recorded in state. Pre-fix, each concurrent
# ``place()`` read ``state.list_sandboxes()`` while the registry was empty and
# every call picked the same node — defeating anti-affinity.
# ──────────────────────────────────────────────────────────────────────────────


def test_pending_placements_count_against_max_runs_per_task() -> None:
    """Two consecutive ``place()`` calls with the same task_key, no
    intervening state insert: the second call must respect the first's
    pending reservation and pick a different node when ``max_runs_per_task=1``.
    """
    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=1)
    catalog = TemplateCatalog()
    catalog.register(m)
    sched = Scheduler(
        [_node("A", vcpus=8, mem_gb=16), _node("B", vcpus=8, mem_gb=16)],
        catalog=catalog,
        state=InMemoryStateStore(),
        max_runs_per_task=1,
    )
    p1 = sched.place(m, task_key="t1")
    p2 = sched.place(m, task_key="t1")
    assert {p1.node.node_id, p2.node.node_id} == {"A", "B"}
    assert p1.reservation_id != p2.reservation_id


def test_release_placement_frees_pending_slot() -> None:
    """A released reservation should free up its slot so a follow-up
    placement with the same task_key can land on the same node again."""
    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=1)
    catalog = TemplateCatalog()
    catalog.register(m)
    sched = Scheduler(
        [_node("A", vcpus=8, mem_gb=16)],
        catalog=catalog,
        state=InMemoryStateStore(),
        max_runs_per_task=1,
    )
    p1 = sched.place(m, task_key="t1")
    with pytest.raises(CapacityExhausted):
        sched.place(m, task_key="t1")
    sched.release_placement(p1)
    p2 = sched.place(m, task_key="t1")
    assert p2.node.node_id == "A"


def test_commit_placement_then_state_insert_avoids_double_count() -> None:
    """After ``commit_placement`` the reservation is dropped; the state
    record then carries the load instead. A follow-up placement should
    see the same total load as before commit (state record exists,
    pending reservation gone) — not double-counted."""
    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=1)
    state = InMemoryStateStore()
    catalog = TemplateCatalog()
    catalog.register(m)
    # Pre-register a rollout so we can attach a sandbox + task_key to it.
    from xrlenv.control.state import RolloutRecord
    from xrlenv.types import RolloutStatus
    state.insert_rollout(RolloutRecord(
        rollout_id="r1", template="t", status=RolloutStatus.STARTING,
        node_id="A", task_key="t1", init_params={},
    ))
    sched = Scheduler(
        [_node("A", vcpus=8, mem_gb=16)],
        catalog=catalog,
        state=state,
        max_runs_per_task=2,
    )
    p1 = sched.place(m, task_key="t1")
    # Coordinator's bootstrap order: insert_sandbox THEN commit.
    state.insert_sandbox(SandboxRecord(
        sandbox_id="sb1", backend="docker", backend_ref="c1",
        stub_endpoint="tcp://127.0.0.1:5000", template="t", node_id="A",
        rollout_id="r1",
    ))
    sched.commit_placement(p1)
    # Total load on A is 1 (from state). With max=2, we still have room
    # for one more.
    p2 = sched.place(m, task_key="t1")
    assert p2.node.node_id == "A"
    # Now max is hit (1 from state + 1 pending). Third call must fail.
    with pytest.raises(CapacityExhausted):
        sched.place(m, task_key="t1")


def test_concurrent_place_distributes_under_max_runs_per_task() -> None:
    """The original race: 4 concurrent threads call ``place(task_key="t")``
    against 2 nodes with max_runs_per_task=2. Pre-fix, all 4 raced to read
    ``state.list_sandboxes()=[]`` and every decision picked the highest-
    capacity node. Post-fix, the in-flight reservation count means each
    decision sees the previous reservations.
    """
    import threading

    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=1)
    catalog = TemplateCatalog()
    catalog.register(m)
    # Both nodes are 16 vCPU so capacity ties; with the lock the
    # second concurrent caller sees the first's reservation and lands
    # on the other node deterministically (alphabetical tiebreak).
    sched = Scheduler(
        [_node("A", vcpus=16, mem_gb=32), _node("B", vcpus=16, mem_gb=32)],
        catalog=catalog,
        state=InMemoryStateStore(),
        max_runs_per_task=2,
    )

    placements: list[Placement] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            p = sched.place(m, task_key="shared")
            with lock:
                placements.append(p)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"unexpected errors: {errors}"
    assert len(placements) == 4
    counts = {"A": 0, "B": 0}
    for p in placements:
        counts[p.node.node_id] += 1
    assert counts == {"A": 2, "B": 2}, (
        f"max_runs_per_task=2 should force 2/2 split; got {counts}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# A7 / D13 (P1.1) — load-vector placement scoring
# ──────────────────────────────────────────────────────────────────────────────


def test_placement_picks_node_with_more_aggregate_slack() -> None:
    """Two nodes, identical hardware, but node-A is already running a
    heavy template. The scheduler must pick node-B for the next
    placement because B has more aggregate slack (CPU + mem + disk),
    not just because B has more "slots of this template" remaining.

    Pre-D13 (count-based score) this passed only by coincidence
    because the heavy template was the same as the candidate. The
    test below uses a *different* heavy template so the count-based
    metric would call A and B equal — only the load-vector metric
    correctly prefers B.
    """
    light = _manifest("light", cpu_request=0.25, mem_gb=1, disk_gb=1)
    heavy = _manifest("heavy", cpu_request=4.0, mem_gb=12, disk_gb=20)

    state = InMemoryStateStore()
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=32, disk_gb=200),
        _node("B", vcpus=8, mem_gb=32, disk_gb=200),
        manifests=[light, heavy],
        state=state,
    )
    # Pre-load node-A with a committed heavy sandbox.
    state.insert_sandbox(SandboxRecord(
        sandbox_id="sb-heavy", backend="docker", backend_ref="cid-heavy",
        stub_endpoint="tcp://127.0.0.1:0", template="heavy",
        node_id="A", rollout_id="r-heavy", status="running",
        effective_resources_json=heavy.resources.model_dump_json(),
    ))

    placement = sched.place(light)
    assert placement.node.node_id == "B", (
        "load-vector cost should pick the unloaded node-B over the "
        f"heavy-loaded node-A; got {placement.node.node_id}"
    )


def test_placement_distributes_evenly_across_identical_nodes() -> None:
    """Three identical-size nodes, 6 light placements → 2 each. The
    load-vector cost's "every placement reduces the picked node's
    slack" property gives natural round-robin behaviour as soon as
    the ties shift; a degenerate scoring (e.g. always picks the
    same node) would land 6/0/0.

    This is the basic D13 sanity check. The asymmetric-cluster
    behaviour (where the right answer is *not* equal but biased to
    larger nodes) is harder to assert deterministically because it
    interacts with alphabetic tie-break + integer-rounded slack;
    the first test in this section pins the heterogeneous case via
    a controlled scenario instead.
    """
    m = _manifest("light", cpu_request=0.25, mem_gb=1, disk_gb=1)
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        _node("C", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[m],
    )
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for i in range(6):
        p = sched.place(m)
        sched.commit_placement(p)
        sched._state.insert_sandbox(SandboxRecord(  # type: ignore[attr-defined]
            sandbox_id=f"sb-{i}", backend="docker",
            backend_ref=f"cid-{i}", stub_endpoint="tcp://127.0.0.1:0",
            template="light", node_id=p.node.node_id,
            rollout_id=f"r-{i}", status="running",
        ))
        counts[p.node.node_id] += 1

    assert counts == {"A": 2, "B": 2, "C": 2}, (
        f"identical nodes must round-robin under D13 load-vector cost; "
        f"got {counts}"
    )


def test_slack_after_placement_excludes_opted_out_axes() -> None:
    """A template with ``cpu_request <= 0`` opts out of CPU
    accounting; the score must only consider the axes the template
    actually requests so a CPU-only template doesn't see disk slack
    drag the score up artificially.
    """
    from xrlenv.control.capacity import (
        NodeProfile,
        StaticCapacityEstimator,
    )

    estimator = StaticCapacityEstimator()
    profile = NodeProfile(
        node_id="N", hardware=_hw(vcpus=8, mem_gb=32, disk_gb=200),
        backends=("docker",),
    )

    # Template requests ONLY mem; cpu and disk are opted out.
    mem_only = TemplateManifest(
        name="mem-only", version="0.1", digest="sha256:m",
        image="im:1",
        resources=ResourceSpec(
            cpu_request=0.0, cpu_limit=0.0,
            mem_request_bytes=8 * 1024**3,
            mem_limit_bytes=8 * 1024**3,
            disk_request_bytes=0,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    slack = estimator.slack_after_placement(profile, mem_only, [])
    # Only the mem axis is in play; expect (32 GB usable - mem_headroom
    # discount - 8 GB) / (32 GB usable - mem_headroom discount).
    # The exact value depends on the default headroom; we just pin
    # that the result is the mem-only slack (no disk contribution
    # making it look near 1.0).
    assert 0.0 < slack < 1.0


def test_slack_after_placement_zero_when_candidate_overshoots() -> None:
    """If placing the candidate would push utilization past the
    headroom-discounted cap on some requested axis, the per-axis
    slack hits the 0.0 floor (the caller's ``fits`` should already
    have rejected this case, but the score must not go negative)."""
    from xrlenv.control.capacity import (
        NodeProfile,
        StaticCapacityEstimator,
    )

    estimator = StaticCapacityEstimator()
    profile = NodeProfile(
        node_id="N", hardware=_hw(vcpus=2, mem_gb=4, disk_gb=10),
        backends=("docker",),
    )
    # Template requests far more memory than the node has.
    huge = TemplateManifest(
        name="huge", version="0.1", digest="sha256:h", image="im:h",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=100 * 1024**3,
            mem_limit_bytes=100 * 1024**3,
            disk_request_bytes=1 * 1024**3,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )
    slack = estimator.slack_after_placement(profile, huge, [])
    assert slack == 0.0


def test_slack_after_placement_returns_one_when_all_axes_opted_out() -> None:
    """A pure-Python adapter with all requests=0 has no signal to
    rank by — ``slack_after_placement`` returns 1.0 so the
    scheduler's node_id tiebreak takes over."""
    from xrlenv.control.capacity import (
        NodeProfile,
        StaticCapacityEstimator,
    )

    estimator = StaticCapacityEstimator()
    profile = NodeProfile(
        node_id="N", hardware=_hw(),
        backends=("docker",),
    )
    free_template = TemplateManifest(
        name="free", version="0.1", digest="sha256:f", image="im:f",
        resources=ResourceSpec(
            cpu_request=0.0, cpu_limit=0.0,
            mem_request_bytes=0, mem_limit_bytes=0,
            disk_request_bytes=0,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )
    assert estimator.slack_after_placement(profile, free_template, []) == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# A1 / D18 (P1.2) — image-affinity placement scoring
# ──────────────────────────────────────────────────────────────────────────────


def test_placement_prefers_node_that_has_the_image() -> None:
    """A1 / D18 (P1.2): with image-affinity on, placing 5 rollouts whose
    image is only on node A → all 5 land on A even though B has the
    same load-vector slack. The bonus is a *score weight* (default
    +500), not a hard filter — A keeps winning until either A's
    slack drops far enough OR the bonus can't compensate for B's
    relative slack lead.
    """
    m = _manifest("light", cpu_request=0.25, mem_gb=1, disk_gb=1)
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[m],
    )
    image_present = {"A": True, "B": False}
    counts: dict[str, int] = {"A": 0, "B": 0}
    for i in range(5):
        p = sched.place(m, image_present=image_present)
        sched.commit_placement(p)
        sched._state.insert_sandbox(SandboxRecord(  # type: ignore[attr-defined]
            sandbox_id=f"sb-{i}", backend="docker",
            backend_ref=f"cid-{i}", stub_endpoint="tcp://127.0.0.1:0",
            template="light", node_id=p.node.node_id,
            rollout_id=f"r-{i}", status="running",
        ))
        counts[p.node.node_id] += 1
    assert counts["A"] == 5, (
        f"image-affinity should pin all rollouts to the node that has the "
        f"image; got {counts}"
    )


def test_placement_falls_through_to_b_when_a_does_not_fit() -> None:
    """The affinity bonus is a weight, not a hard filter. When the
    image-affine node A can't fit the candidate (capacity gate
    rejects it), B gets the work even though B doesn't have the
    image. Pin this so the affinity behavior never starves a
    saturated cluster.

    Setup: identical-size A and B; pre-seed A with state that
    consumes its CPU. The candidate template no longer fits on A;
    ``estimator.fits()`` filters A out before the score loop, so
    the affinity bonus is irrelevant for A. B wins by being the
    only fitting node.
    """
    heavy = _manifest("heavy", cpu_request=4.0, mem_gb=4, disk_gb=20)

    state = InMemoryStateStore()
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[heavy],
        state=state,
    )
    # Pre-seed A with two committed heavy sandboxes — that leaves
    # ~0 cpu slack; a third heavy candidate doesn't fit.
    for i in range(2):
        state.insert_sandbox(SandboxRecord(
            sandbox_id=f"sb-pre-{i}", backend="docker",
            backend_ref=f"cid-pre-{i}",
            stub_endpoint="tcp://127.0.0.1:0", template="heavy",
            node_id="A", rollout_id=f"r-pre-{i}", status="running",
            effective_resources_json=heavy.resources.model_dump_json(),
        ))

    image_present = {"A": True, "B": False}
    placement = sched.place(heavy, image_present=image_present)
    assert placement.node.node_id == "B", (
        f"affinity must NOT pin work to a saturated node; got "
        f"{placement.node.node_id}"
    )


def test_placement_no_image_present_arg_falls_back_to_slack_only() -> None:
    """When admission doesn't pre-fetch image presence (e.g. Pattern A
    manifest without an instance yet, or test fixture), ``place()``
    must score on slack only — no bonus, no penalty. Pin so the
    image-affinity feature stays opt-in via the kwarg."""
    m = _manifest("light", cpu_request=0.25, mem_gb=1, disk_gb=1)
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[m],
    )
    counts: dict[str, int] = {"A": 0, "B": 0}
    for i in range(4):
        p = sched.place(m)  # no image_present kwarg
        sched.commit_placement(p)
        sched._state.insert_sandbox(SandboxRecord(  # type: ignore[attr-defined]
            sandbox_id=f"sb-{i}", backend="docker",
            backend_ref=f"cid-{i}", stub_endpoint="tcp://127.0.0.1:0",
            template="light", node_id=p.node.node_id,
            rollout_id=f"r-{i}", status="running",
        ))
        counts[p.node.node_id] += 1
    # Identical-size nodes round-robin 2/2 under D13 load-vector cost.
    assert counts == {"A": 2, "B": 2}


def test_placement_prefers_planner_preferred_home_when_image_absent_everywhere(
) -> None:
    """Audit P1.6.g-H2 (2026-05-05): when the candidate's image isn't
    materialized on any node yet (deferred / lazy), the scheduler
    biases first-rollout placement toward the build coordinator's
    recorded ``preferred_home`` so the lazy build lands where the
    bin-packer planned the spread. Soft signal — only meaningful
    when ``image_present`` is uniformly False; once the lazy build
    materializes the image somewhere, real ``image_present`` takes
    over."""
    m = _manifest("light", cpu_request=0.25, mem_gb=1, disk_gb=1)
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[m],
    )
    image_present = {"A": False, "B": False}
    counts: dict[str, int] = {"A": 0, "B": 0}
    for i in range(5):
        p = sched.place(
            m, image_present=image_present, preferred_home_node="B",
        )
        sched.commit_placement(p)
        sched._state.insert_sandbox(SandboxRecord(  # type: ignore[attr-defined]
            sandbox_id=f"sb-{i}", backend="docker",
            backend_ref=f"cid-{i}", stub_endpoint="tcp://127.0.0.1:0",
            template="light", node_id=p.node.node_id,
            rollout_id=f"r-{i}", status="running",
        ))
        counts[p.node.node_id] += 1
    # First rollout lands on B (preferred_home); subsequent rollouts
    # may either keep landing on B or rebalance toward A as B's
    # slack drops. The contract is that B wins the first one.
    assert counts["B"] >= 1, (
        f"first-rollout placement must honor the planner's "
        f"preferred_home; counts={counts}"
    )


def test_placement_image_present_beats_preferred_home_bonus() -> None:
    """The preferred_home bonus is gated on ``not image_present``
    everywhere — once any node has the image, the existing image-
    affinity bonus dominates and pins rollouts to the materialized
    node, regardless of where the planner originally pointed.
    Otherwise we'd ship cold rollouts to a not-yet-loaded
    preferred_home while a warm node sits idle."""
    m = _manifest("light", cpu_request=0.25, mem_gb=1, disk_gb=1)
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[m],
    )
    # The planner's preferred_home was B, but the lazy build
    # actually landed the image on A (e.g. rollout 0 took the bonus
    # to B but failed → fell back to A which materialized it).
    image_present = {"A": True, "B": False}
    placement = sched.place(
        m, image_present=image_present, preferred_home_node="B",
    )
    assert placement.node.node_id == "A", (
        f"image_present=A must beat preferred_home=B; got "
        f"{placement.node.node_id}"
    )


def test_placement_preferred_home_suppressed_when_any_node_has_image() -> None:
    """Audit P1.6.g-M3 (2026-05-05): the preferred_home bonus must be
    suppressed globally — for ALL candidates — once any node has the
    image. Otherwise a high-slack preferred_home can outscore a
    loaded-but-warm node solely because of the planner hint and
    trigger a duplicate lazy build.

    The contract this test pins is narrow: "warm beats the
    preferred_home half-bonus regardless of load delta." It does
    NOT claim warm always wins; the resource weight (W_R=2/3) is
    intentionally larger than the image weight (W_I=1/3), so a
    sufficiently-idle cold node DOES beat a sufficiently-loaded
    warm node. That calibration is pinned by the companion test
    test_placement_warm_node_yields_to_cold_when_slack_delta_is_large.

    Setup: A has the image and carries pre-existing load; B is
    the preferred_home, image-cold, with full slack. The slack
    delta is intentionally modest (delta < 0.5) so the test
    isolates the preferred_home gating from the resource-vs-
    image-affinity calibration.

    Math (default weights w_R=2/3, w_I=1/3):
      A: slack≈0.625, image_present=True
        -> 2/3 * 0.625 + 1/3 * 1.0 ≈ 0.750
      B (with bug — preferred_home bonus alive):
        slack≈1.0, affinity 0.5
        -> 2/3 * 1.0 + 1/3 * 0.5 ≈ 0.833 -> B wins (BUG)
      B (with fix — bonus suppressed because A has the image):
        affinity 0.0
        -> 2/3 * 1.0 ≈ 0.667 -> A wins (correct)
    """
    light = _manifest("light", cpu_request=1.0, mem_gb=1, disk_gb=5)

    state = InMemoryStateStore()
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[light],
        state=state,
    )
    # Pre-seed A with two light sandboxes so its slack drops below
    # B's. A still fits another candidate (3 of 8 cpu used after
    # placement) so it stays in the candidate pool.
    for i in range(2):
        state.insert_sandbox(SandboxRecord(
            sandbox_id=f"sb-pre-{i}", backend="docker",
            backend_ref=f"cid-pre-{i}",
            stub_endpoint="tcp://127.0.0.1:0", template="light",
            node_id="A", rollout_id=f"r-pre-{i}", status="running",
            effective_resources_json=light.resources.model_dump_json(),
        ))

    placement = sched.place(
        light,
        image_present={"A": True, "B": False},
        preferred_home_node="B",
    )
    assert placement.node.node_id == "A", (
        "warm cache must win over preferred_home + extra slack; "
        f"got {placement.node.node_id} (regression in M3 gating)"
    )


def test_placement_warm_node_yields_to_cold_when_slack_delta_is_large() -> None:
    """Audit P1.6.g-M4 calibration (2026-05-05): "warm beats
    preferred_home" is NOT "warm always wins". The default weights
    (W_R=2/3 resource, W_I=1/3 image) deliberately let a
    sufficiently-idle cold node beat a heavily-loaded warm node —
    this is the D18 design decision (resource is twice as
    important as image presence) carried forward through the H2
    work. Without this test, weight tuning could quietly cross the
    calibration threshold and starve workloads on a saturated
    warm node.

    Math (default weights):
      Warm wins when:  slack_cold - slack_warm <= 0.5
      Cold wins when:  slack_cold - slack_warm >= 0.5

    Setup: A is warm but heavily loaded (slack ≈ 0.125); B is
    cold and idle (slack ≈ 0.875). Slack delta ≈ 0.75 > 0.5 so
    cold should win.

      A: 2/3 * 0.125 + 1/3 * 1.0 ≈ 0.417
      B: 2/3 * 0.875 + 0           ≈ 0.583  -> B wins (correct)

    The companion
    test_placement_preferred_home_suppressed_when_any_node_has_image
    pins the inverse case (small slack delta -> warm wins).
    """
    light = _manifest("light", cpu_request=1.0, mem_gb=1, disk_gb=5)

    state = InMemoryStateStore()
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[light],
        state=state,
    )
    # Pre-seed A heavily — 6 sandboxes leaves slack ≈ 0.125 after
    # placing the candidate (7/8 cpu used). A still fits.
    for i in range(6):
        state.insert_sandbox(SandboxRecord(
            sandbox_id=f"sb-pre-{i}", backend="docker",
            backend_ref=f"cid-pre-{i}",
            stub_endpoint="tcp://127.0.0.1:0", template="light",
            node_id="A", rollout_id=f"r-pre-{i}", status="running",
            effective_resources_json=light.resources.model_dump_json(),
        ))

    placement = sched.place(
        light,
        image_present={"A": True, "B": False},
    )
    assert placement.node.node_id == "B", (
        "saturated warm node must yield to idle cold node when "
        "slack delta exceeds the W_I weight; got "
        f"{placement.node.node_id} "
        "(calibration drift in default weights?)"
    )


def test_placement_preferred_home_yields_to_loaded_node() -> None:
    """Soft preference, not a hard constraint: when the
    preferred_home is meaningfully more loaded than another fitting
    node, the load score eventually overrides the bonus. With
    default weights (2/3 resource, 1/3 image) the bonus is
    half-strength image-affinity ≈ 0.17, so a ≥ 0.25 slack delta
    flips the decision. Pin the failover so an offline / saturated
    preferred_home doesn't wedge work."""
    heavy = _manifest("heavy", cpu_request=4.0, mem_gb=4, disk_gb=20)

    state = InMemoryStateStore()
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=16, disk_gb=100),
        _node("B", vcpus=8, mem_gb=16, disk_gb=100),
        manifests=[heavy],
        state=state,
    )
    # Pre-seed B (the planner's preferred_home) with two heavy
    # sandboxes — leaves ~0 cpu slack on B; a third heavy
    # candidate doesn't fit.
    for i in range(2):
        state.insert_sandbox(SandboxRecord(
            sandbox_id=f"sb-pre-{i}", backend="docker",
            backend_ref=f"cid-pre-{i}",
            stub_endpoint="tcp://127.0.0.1:0", template="heavy",
            node_id="B", rollout_id=f"r-pre-{i}", status="running",
            effective_resources_json=heavy.resources.model_dump_json(),
        ))

    placement = sched.place(
        heavy,
        image_present={"A": False, "B": False},
        preferred_home_node="B",
    )
    assert placement.node.node_id == "A", (
        f"preferred_home bonus must NOT pin work to a saturated "
        f"node; got {placement.node.node_id}"
    )


def test_default_weights_preserve_pre_p1_2_behavior() -> None:
    """Pin the default weights ``(2/3, 1/3)`` — chosen so the
    scoring formula's relative-importance contract is explicit
    (resource is twice as important as image presence). Changing
    the defaults is a behavior change worth a separate, audited
    diff; this test catches accidental drift.
    """
    from xrlenv.control.scheduler import (
        DEFAULT_IMAGE_AFFINITY_WEIGHT,
        DEFAULT_RESOURCE_WEIGHT,
    )

    assert pytest.approx(DEFAULT_RESOURCE_WEIGHT) == 2.0 / 3.0
    assert pytest.approx(DEFAULT_IMAGE_AFFINITY_WEIGHT) == 1.0 / 3.0
    assert pytest.approx(
        DEFAULT_RESOURCE_WEIGHT + DEFAULT_IMAGE_AFFINITY_WEIGHT
    ) == 1.0


def test_weighted_sum_score_factory_validates_weight_ranges() -> None:
    """Each weight in ``[0, 1]``; reject negative + above-1 values
    at factory-call time so misconfiguration surfaces at boot."""
    from xrlenv.control.scheduler import weighted_sum_score

    with pytest.raises(ValueError, match="resource_weight"):
        weighted_sum_score(
            resource_weight=-0.1, image_affinity_weight=1.1,
        )
    with pytest.raises(ValueError, match="image_affinity_weight"):
        weighted_sum_score(
            resource_weight=0.5, image_affinity_weight=1.5,
        )


def test_weighted_sum_score_factory_validates_weight_sum() -> None:
    """``resource_weight + image_affinity_weight == 1`` is the
    framework's relative-importance contract — silently allowing
    ``0.5 + 0.4`` would let the operator's intent drift invisibly.
    Reject violations at factory-call time."""
    from xrlenv.control.scheduler import weighted_sum_score

    with pytest.raises(ValueError, match="must sum to 1"):
        weighted_sum_score(
            resource_weight=0.5, image_affinity_weight=0.4,
        )


def test_custom_score_fn_overrides_default() -> None:
    """``Scheduler(score_fn=...)`` plugs a custom scoring function
    in. Pin the contract: the scheduler delegates to whatever
    callable the operator supplies, with the per-node
    :class:`PlacementFeatures` shape as input.
    """
    from xrlenv.control.scheduler import PlacementFeatures

    captured: list[PlacementFeatures] = []

    def picky_score(features: PlacementFeatures) -> float:
        # Always pick node-B by giving it a higher score regardless
        # of slack / affinity. Pure plug-in; no defaults.
        captured.append(features)
        return 1.0 if features.node_id == "B" else 0.0

    m = _manifest()
    catalog = TemplateCatalog()
    catalog.register(m)
    sched = Scheduler(
        [_node("A"), _node("B")],
        catalog=catalog, state=InMemoryStateStore(),
        score_fn=picky_score,
    )
    placement = sched.place(m, image_present={"A": True, "B": False})

    assert placement.node.node_id == "B"
    # Both candidates were scored.
    assert {f.node_id for f in captured} == {"A", "B"}
    # Each features struct carries the manifest + slack + presence.
    for f in captured:
        assert f.manifest is m
        assert 0.0 <= f.resource_slack <= 1.0
        assert isinstance(f.image_present, bool)


def test_default_score_fn_is_weighted_sum_with_default_weights() -> None:
    """``DEFAULT_SCORE_FN`` is built from
    ``weighted_sum_score()`` with the documented default weights.
    Pin both halves: that the module-level constant exists and
    that it's wired correctly."""
    from xrlenv.control.scheduler import (
        DEFAULT_IMAGE_AFFINITY_WEIGHT,
        DEFAULT_RESOURCE_WEIGHT,
        DEFAULT_SCORE_FN,
        PlacementFeatures,
    )

    m = _manifest()
    f_warm = PlacementFeatures(
        node_id="N", manifest=m,
        resource_slack=0.5, image_present=True,
    )
    expected_warm = (
        DEFAULT_RESOURCE_WEIGHT * 0.5
        + DEFAULT_IMAGE_AFFINITY_WEIGHT * 1.0
    )
    assert pytest.approx(DEFAULT_SCORE_FN(f_warm)) == expected_warm

    f_cold_full = PlacementFeatures(
        node_id="N", manifest=m,
        resource_slack=1.0, image_present=False,
    )
    assert pytest.approx(DEFAULT_SCORE_FN(f_cold_full)) == DEFAULT_RESOURCE_WEIGHT


def test_score_is_a_float_in_unit_interval() -> None:
    """Both inputs to the score formula are normalised to [0, 1]
    and the weights sum to 1, so the placement score is in [0, 1]
    by construction. Pin so a future refactor that re-introduces
    integer scaling at the wrong layer surfaces here."""
    m = _manifest()
    sched = _scheduler(_node("A"), manifests=[m])
    placement = sched.place(m, image_present={"A": True})
    assert isinstance(placement.score, float)
    assert 0.0 <= placement.score <= 1.0


def test_image_aware_off_ignores_image_present_arg() -> None:
    """Constructor flag ``image_aware_placement=False`` makes ``place()``
    ignore any pre-fetched image-presence map. Belt-and-suspenders
    for operators who explicitly opt out (e.g., when a registry
    is so reliable that affinity scoring is wasted overhead)."""
    m = _manifest("light", cpu_request=0.25, mem_gb=1, disk_gb=1)
    catalog = TemplateCatalog()
    catalog.register(m)
    sched = Scheduler(
        [_node("A", vcpus=8, mem_gb=16, disk_gb=100),
         _node("B", vcpus=8, mem_gb=16, disk_gb=100)],
        catalog=catalog,
        state=InMemoryStateStore(),
        image_aware_placement=False,
    )
    image_present = {"A": True, "B": False}
    counts: dict[str, int] = {"A": 0, "B": 0}
    for i in range(4):
        p = sched.place(m, image_present=image_present)
        sched.commit_placement(p)
        sched._state.insert_sandbox(SandboxRecord(  # type: ignore[attr-defined]
            sandbox_id=f"sb-{i}", backend="docker",
            backend_ref=f"cid-{i}", stub_endpoint="tcp://127.0.0.1:0",
            template="light", node_id=p.node.node_id,
            rollout_id=f"r-{i}", status="running",
        ))
        counts[p.node.node_id] += 1
    # Affinity ignored → identical nodes round-robin.
    assert counts == {"A": 2, "B": 2}


# Issue #14 — disk-pressure placement gate
# ──────────────────────────────────────────────────────────────────────────────


def test_place_skips_disk_pressured_node_in_favour_of_healthy() -> None:
    # The pressured node would otherwise win on slack (more capacity);
    # the gate should send placement to the healthy peer instead.
    m = _manifest()
    pressured = _node(
        "p", vcpus=32, mem_gb=128, disk_gb=400,
        # 1 GiB free of 296 GB total → critical (well below 5 % AND 5 GiB).
        disk_state=(1 * 1024**3, 296 * 1024**3),
    )
    healthy = _node(
        "h", vcpus=8, mem_gb=32, disk_gb=200,
        disk_state=(150 * 1024**3, 200 * 1024**3),
    )
    sched = _scheduler(pressured, healthy, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "h"


def test_place_raises_when_all_nodes_disk_pressured() -> None:
    m = _manifest()
    sched = _scheduler(
        _node("a", disk_state=(1 * 1024**3, 296 * 1024**3)),
        _node("b", disk_state=(2 * 1024**3, 296 * 1024**3)),
        manifests=[m],
    )
    with pytest.raises(CapacityExhausted, match="disk pressure"):
        sched.place(m)


def test_place_treats_unknown_disk_state_as_healthy() -> None:
    # ``(0, 0)`` is the documented sentinel for "node hasn't reported
    # yet" — must NOT trip the gate, otherwise freshly-attached nodes
    # are blackholed before their first heartbeat.
    m = _manifest()
    fresh = _node("fresh", disk_state=(0, 0))
    sched = _scheduler(fresh, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "fresh"


def test_place_uses_fraction_threshold_for_large_disks() -> None:
    # 10 GiB free on a 500 GiB disk = 2 % — below the 5 % fraction
    # threshold even though above the 5 GiB absolute floor.
    m = _manifest()
    pressured = _node(
        "p", disk_state=(10 * 1024**3, 500 * 1024**3),
    )
    healthy = _node(
        "h", disk_state=(50 * 1024**3, 200 * 1024**3),
    )
    sched = _scheduler(pressured, healthy, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "h"


def test_place_tolerates_legacy_transport_without_disk_state() -> None:
    # MagicMock without an explicit return_value still produces some
    # mock object on ``disk_state()``; the gate's strict tuple check
    # must classify that as "unknown / healthy" so legacy transports
    # opt into the gate cleanly without code changes.
    m = _manifest()
    sched = _scheduler(_node("legacy"), manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "legacy"


# Issue #18 (Ask #2) — node-health timeout placement gate
# ──────────────────────────────────────────────────────────────────────────────


def test_place_skips_node_with_recent_command_timeout() -> None:
    # A node that missed a command reply 10 s ago (well within the
    # 120 s cooldown) must be excluded even though it would otherwise
    # win — placement goes to the healthy peer.
    m = _manifest()
    wedged = _node(
        "wedged", vcpus=32, mem_gb=128, disk_gb=400,
        seconds_since_timeout=10.0,
    )
    healthy = _node("healthy", vcpus=8, mem_gb=32, disk_gb=200)
    sched = _scheduler(wedged, healthy, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "healthy"


def test_place_readmits_node_after_cooldown_elapses() -> None:
    # A node whose last timeout is older than the cooldown window is
    # healthy again — self-healing, no explicit requalification.
    m = _manifest()
    recovered = _node(
        "recovered", vcpus=32, mem_gb=128, disk_gb=400,
        seconds_since_timeout=NODE_TIMEOUT_COOLDOWN_S + 1.0,
    )
    sched = _scheduler(recovered, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "recovered"


def test_place_treats_never_timed_out_node_as_healthy() -> None:
    # ``None`` from ``seconds_since_last_command_timeout`` is the
    # "never timed out" sentinel — must not trip the gate.
    m = _manifest()
    fresh = _node("fresh", seconds_since_timeout=None)
    sched = _scheduler(fresh, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "fresh"


def test_place_raises_when_all_nodes_in_timeout_cooldown() -> None:
    # Whole cluster wedged → CapacityExhausted, which the admission
    # queue catches and retries once a node recovers.
    m = _manifest()
    sched = _scheduler(
        _node("a", seconds_since_timeout=5.0),
        _node("b", seconds_since_timeout=30.0),
        manifests=[m],
    )
    with pytest.raises(CapacityExhausted, match="missed a command reply"):
        sched.place(m)


def test_place_tolerates_legacy_transport_without_timeout_probe() -> None:
    # A transport predating the protocol extension has no
    # ``seconds_since_last_command_timeout`` method — the gate's
    # ``getattr(..., None)`` guard classifies it as healthy.
    m = _manifest()
    legacy = _node("legacy")
    del legacy.seconds_since_last_command_timeout
    sched = _scheduler(legacy, manifests=[m])
    placement = sched.place(m)
    assert placement.node.node_id == "legacy"


# ──────────────────────────────────────────────────────────────────────────────
# Raw-session load provider (Option C — raw containers don't appear in
# ``state.list_sandboxes()``; the scheduler reads them via an injected
# callable so the capacity gate accounts for them just like managed
# sandboxes).
# ──────────────────────────────────────────────────────────────────────────────


def test_raw_session_provider_contributes_to_max_runs_per_task() -> None:
    """An active raw-container session for ``task_key="t1"`` counts
    toward that task's ``max_runs_per_task`` cap on its node. Without
    the provider plumbing the scheduler would treat the node as empty
    and over-place.
    """
    from xrlenv.control.scheduler import RawSessionLoad

    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=1)
    catalog = TemplateCatalog()
    catalog.register(m)
    sched = Scheduler(
        [_node("A", vcpus=8, mem_gb=16)],
        catalog=catalog,
        state=InMemoryStateStore(),
        max_runs_per_task=1,
    )

    # Pretend a raw container is already running for task "t1" on A.
    raw_load = [RawSessionLoad(
        node_id="A",
        template_name="raw-container/busybox:1",
        effective_resources=m.resources,
        task_key="t1",
    )]
    sched.set_raw_session_provider(lambda: raw_load)

    # ``place(task_key="t1")`` must see the raw session counting
    # toward the cap and refuse — the only node is full from this
    # task's perspective.
    with pytest.raises(CapacityExhausted):
        sched.place(m, task_key="t1")

    # A different task_key isn't constrained by the cap, just by
    # resources — which still fit.
    p = sched.place(m, task_key="t2")
    assert p.node.node_id == "A"


def test_set_raw_session_provider_replaces_previous_provider() -> None:
    """The setter is idempotent / replaceable so tests (and any
    future re-wiring at runtime) can swap providers safely. Caught a
    real bug shape during dev: an initial empty provider stuck around
    after the runtime tried to swap in the real coordinator."""
    from xrlenv.control.scheduler import RawSessionLoad

    m = _manifest()
    catalog = TemplateCatalog()
    catalog.register(m)
    sched = Scheduler(
        [_node("A")], catalog=catalog, state=InMemoryStateStore(),
        max_runs_per_task=1,
    )

    # Stage 1: provider returns one entry for t1 → cap is hit.
    sched.set_raw_session_provider(lambda: [RawSessionLoad(
        node_id="A", template_name="raw-container/x",
        effective_resources=m.resources, task_key="t1",
    )])
    with pytest.raises(CapacityExhausted):
        sched.place(m, task_key="t1")

    # Stage 2: replace with an empty provider → placement now succeeds.
    sched.set_raw_session_provider(lambda: [])
    p = sched.place(m, task_key="t1")
    assert p.node.node_id == "A"


# ──────────────────────────────────────────────────────────────────────────────
# place(reserve=…) — fleet-footprint reservation (phase 1, opt-in)
# ──────────────────────────────────────────────────────────────────────────────


def _footprint(cpu: float, mem_gb: int = 1, disk_gb: int = 0) -> ResourceSpec:
    return ResourceSpec(
        cpu_request=cpu,
        cpu_limit=cpu,
        mem_request_bytes=mem_gb * 1024**3,
        mem_limit_bytes=mem_gb * 1024**3,
        disk_request_bytes=disk_gb * 1024**3,
    )


def test_place_reserve_charges_footprint_not_container_own() -> None:
    """A fleet-opening ``place(reserve=footprint)`` reserves the whole
    footprint in ``_pending`` — not the lead container's tiny own request.
    A small lead (0.25 CPU) with a footprint of 6 CPU must leave the pending
    reservation holding 6, so the node's remaining capacity reflects the
    whole fleet immediately (the anti-starvation guarantee)."""
    m = _manifest("lead", cpu_request=0.25, mem_gb=1, disk_gb=1)
    sched = _scheduler(_node("A", vcpus=8, mem_gb=32, disk_gb=200), manifests=[m])

    p = sched.place(m, reserve=_footprint(cpu=6, mem_gb=8))

    assert p.node.node_id == "A"
    assert len(sched._pending) == 1
    pending = next(iter(sched._pending.values()))
    # The pending reservation holds the FOOTPRINT, not the lead's 0.25.
    assert pending.effective_resources.cpu_request == 6.0
    assert pending.effective_resources.mem_request_bytes == 8 * 1024**3
    # name/image preserved — the substitution only swaps resources.
    assert pending.template_name == "lead"


def test_place_reserve_footprint_gates_subsequent_placement() -> None:
    """The reserved footprint occupies the node for concurrent placements:
    after a 6-CPU footprint is reserved on an 8-CPU node, a second placement
    needing 4 CPU must be refused (6 + 4 > 8 - headroom) — proving the
    reservation is real capacity, not just bookkeeping on the lead."""
    lead = _manifest("lead", cpu_request=0.25, mem_gb=1, disk_gb=1)
    other = _manifest("other", cpu_request=4.0, mem_gb=1, disk_gb=1)
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=32, disk_gb=200), manifests=[lead, other],
    )

    sched.place(lead, reserve=_footprint(cpu=6, mem_gb=8))
    with pytest.raises(CapacityExhausted):
        sched.place(other)


def test_place_reserve_none_is_legacy_per_container() -> None:
    """``reserve=None`` (the default) reserves only the container's own
    resources — the byte-for-byte legacy path. The same small lead that
    reserved 6 CPU above reserves only 0.25 here, leaving ample room for the
    4-CPU placement that was refused under the footprint reservation."""
    lead = _manifest("lead", cpu_request=0.25, mem_gb=1, disk_gb=1)
    other = _manifest("other", cpu_request=4.0, mem_gb=1, disk_gb=1)
    sched = _scheduler(
        _node("A", vcpus=8, mem_gb=32, disk_gb=200), manifests=[lead, other],
    )

    sched.place(lead)  # no reserve → charges 0.25
    pending = next(iter(sched._pending.values()))
    assert pending.effective_resources.cpu_request == 0.25
    # Room remains for the 4-CPU placement (unlike the footprint case).
    p = sched.place(other)
    assert p.node.node_id == "A"


def test_place_reserve_infeasible_footprint_raises() -> None:
    """A footprint larger than any node can hold fails fast with
    ``CapacityExhausted`` (single-node reservation, MVP — no splitting) and
    leaves no pending reservation behind."""
    m = _manifest("lead", cpu_request=0.25, mem_gb=1, disk_gb=1)
    sched = _scheduler(_node("A", vcpus=8, mem_gb=32, disk_gb=200), manifests=[m])

    with pytest.raises(CapacityExhausted):
        sched.place(m, reserve=_footprint(cpu=64, mem_gb=8))
    assert len(sched._pending) == 0


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-4b — REQUIRED cpu-isolation placement predicate + pinned reservations
# ──────────────────────────────────────────────────────────────────────────────


def _iso_manifest(
    mode: Any, *, name: str = "t", cpu_limit: float = 2.0, cpu_request: float = 0.25,
) -> TemplateManifest:
    """A manifest with a given `cpu_isolation` mode + `cpu_limit`. `cpu_request`
    (what the estimator charges) is separate from `cpu_limit` (the pin count),
    so tests can isolate the pinned-core logic (small request) or force a slack
    gap (large request)."""
    m = _manifest(name=name, cpu_request=cpu_request)
    return m.model_copy(
        update={
            "resources": m.resources.model_copy(
                update={"cpu_isolation": mode, "cpu_limit": cpu_limit},
            ),
        },
    )


def _required_manifest(name: str = "t", cpu_limit: float = 2.0) -> TemplateManifest:
    """A manifest whose ResourceSpec REQUIRES cpu isolation (ceil(cpu_limit)
    dedicated cores). cpu_request is kept small so the ordinary cpu/mem estimator
    fits — isolating the test to the pinned-core predicate."""
    from xrlenv.backends.base import CpuIsolation

    return _iso_manifest(CpuIsolation.REQUIRED, name=name, cpu_limit=cpu_limit)


def _iso_node(
    node_id: str, *, capable: bool = True, free: int = 8, total: int = 8, **kw: Any,
) -> Any:
    """A node that advertises P6 isolation capability + a pinned-CPU state."""
    n = _node(node_id, **kw)
    n.isolation_capable.return_value = capable
    n.pinned_cpu_state.return_value = (free, total)
    return n


def test_required_isolation_places_on_capable_node_with_free_cores() -> None:
    """A REQUIRED-isolation rollout lands on an isolation-capable node that has
    enough free pinnable cores, and reserves ceil(cpu_limit) pinned cores."""
    m = _required_manifest(cpu_limit=2.0)
    sched = _scheduler(_iso_node("A", free=8, total=8), manifests=[m])

    p = sched.place(m)

    assert p.node.node_id == "A"
    pending = next(iter(sched._pending.values()))
    assert pending.pinned_cores_reserved == 2  # ceil(2.0)


def test_required_isolation_skips_non_capable_node() -> None:
    """A REQUIRED rollout is never placed on a non-isolation-capable node; when
    a capable sibling exists it wins, when none exists it raises."""
    m = _required_manifest(cpu_limit=2.0)
    # Only a non-capable node → refused (the predicate empties the pool).
    only_plain = _scheduler(_iso_node("P", capable=False, free=8), manifests=[m])
    with pytest.raises(CapacityExhausted):
        only_plain.place(m)
    # Capable sibling present → placement routes to it, not the plain node.
    mixed = _scheduler(
        _iso_node("P", capable=False, free=8),
        _iso_node("C", capable=True, free=8),
        manifests=[m],
    )
    assert mixed.place(m).node.node_id == "C"


def test_required_isolation_skips_node_without_enough_free_cores() -> None:
    """A capable node whose free pinnable cores can't cover ceil(cpu_limit) is
    refused (need 2, only 1 free)."""
    m = _required_manifest(cpu_limit=2.0)
    sched = _scheduler(_iso_node("A", capable=True, free=1, total=8), manifests=[m])
    with pytest.raises(CapacityExhausted):
        sched.place(m)


def test_required_isolation_unknown_pinned_state_is_refused() -> None:
    """A capable node reporting the (0, 0) unknown sentinel (no heartbeat yet)
    can't cover a REQUIRED pin — refused, never a false placement."""
    m = _required_manifest(cpu_limit=1.0)
    sched = _scheduler(_iso_node("A", capable=True, free=0, total=0), manifests=[m])
    with pytest.raises(CapacityExhausted):
        sched.place(m)


def test_required_isolation_pending_reservation_prevents_overcommit() -> None:
    """Two concurrent REQUIRED placements can't each read the same free cores:
    the first reserves 2 of the node's 2 free pinnable cores, so the second is
    refused before any heartbeat reflects the first pin."""
    m = _required_manifest(cpu_limit=2.0)
    sched = _scheduler(_iso_node("A", capable=True, free=2, total=8), manifests=[m])

    first = sched.place(m)
    assert first.node.node_id == "A"
    with pytest.raises(CapacityExhausted):  # free 2 - pending 2 = 0 < 2
        sched.place(m)

    # Releasing the first reservation frees the pinnable cores again.
    sched.release_placement(first)
    assert sched.place(m).node.node_id == "A"


def test_best_effort_and_off_not_gated_by_isolation_predicate() -> None:
    """OFF / BEST_EFFORT rollouts are NOT filtered by the predicate — they place
    on a non-isolation-capable node (best_effort degrades on the node) and
    reserve no pinned cores (behavior-neutral for the non-REQUIRED path)."""
    from xrlenv.backends.base import CpuIsolation

    # BEST_EFFORT on a plainly non-capable node → still placed, 0 reserved.
    be = _manifest(name="be", cpu_request=0.25)
    be = be.model_copy(
        update={
            "resources": be.resources.model_copy(
                update={"cpu_isolation": CpuIsolation.BEST_EFFORT},
            ),
        },
    )
    sched = _scheduler(_iso_node("P", capable=False, free=0, total=0), manifests=[be])
    p = sched.place(be)
    assert p.node.node_id == "P"
    assert next(iter(sched._pending.values())).pinned_cores_reserved == 0

    # OFF (the default manifest) likewise unaffected.
    off = _manifest(name="off", cpu_request=0.25)
    sched2 = _scheduler(_iso_node("P", capable=False, free=0, total=0), manifests=[off])
    assert sched2.place(off).node.node_id == "P"


def test_best_effort_soft_nudge_prefers_capable_node_on_tie() -> None:
    """P6 — a BEST_EFFORT rollout gets a small score bonus on an isolation-capable
    node with free pinnable cores, so among otherwise-equal candidates it lands
    where it will actually pin. The bonus must OVERCOME the alphabetical tiebreak:
    the capable node is `Z` (loses ties normally to `A`), yet wins here."""
    from xrlenv.backends.base import CpuIsolation

    be = _iso_manifest(CpuIsolation.BEST_EFFORT, cpu_limit=2.0)
    # Identical hardware/load → identical resource slack; only isolation differs.
    plain_a = _iso_node("A", capable=False, free=0, total=0)  # alphabetically first
    cap_z = _iso_node("Z", capable=True, free=8, total=8)     # capable + free
    sched = _scheduler(plain_a, cap_z, manifests=[be])
    assert sched.place(be).node.node_id == "Z"  # bonus beats the A-wins-tie default


def test_best_effort_no_nudge_when_capable_node_has_no_free_cores() -> None:
    """P6 — the nudge only fires when the capable node can actually pin. A capable
    node with 0 free pinnable cores gets NO bonus, so the tie falls back to the
    alphabetical default (`A`)."""
    from xrlenv.backends.base import CpuIsolation

    be = _iso_manifest(CpuIsolation.BEST_EFFORT, cpu_limit=2.0)
    plain_a = _iso_node("A", capable=False, free=0, total=0)
    cap_z_full = _iso_node("Z", capable=True, free=1, total=8)  # capable but <2 free
    sched = _scheduler(plain_a, cap_z_full, manifests=[be])
    assert sched.place(be).node.node_id == "A"  # no bonus → tie → A wins


def test_best_effort_nudge_is_subordinate_to_resource_slack() -> None:
    """P6 — the nudge is SMALL: a roomy but non-capable node beats a tight
    capable one. best_effort runs fine unpinned on the roomy node; it would run
    worse pinned on the saturated one, so slack must dominate the bonus."""
    from xrlenv.backends.base import CpuIsolation

    # Large cpu_request so the slack gap between an 8-vcpu and a 2-vcpu node is
    # far larger than the ~0.1 isolation bonus.
    be = _iso_manifest(CpuIsolation.BEST_EFFORT, cpu_limit=2.0, cpu_request=2.0)
    cap_tight = _iso_node("A", capable=True, free=2, total=2, vcpus=2)  # can pin, no slack
    plain_roomy = _iso_node("B", capable=False, free=0, total=0, vcpus=8)  # empty, incapable
    sched = _scheduler(cap_tight, plain_roomy, manifests=[be])
    assert sched.place(be).node.node_id == "B"  # slack wins over the small nudge


def test_off_gets_no_isolation_nudge() -> None:
    """P6 — OFF rollouts never get the nudge (they don't pin). The capable `Z`
    node has no advantage, so the tie falls to `A`."""
    off = _manifest(name="off", cpu_request=0.25)
    plain_a = _iso_node("A", capable=False, free=0, total=0)
    cap_z = _iso_node("Z", capable=True, free=8, total=8)
    sched = _scheduler(plain_a, cap_z, manifests=[off])
    assert sched.place(off).node.node_id == "A"


def test_best_effort_nudge_is_owned_by_score_fn() -> None:
    """P6 (audit follow-up) — the best_effort preference is a `PlacementFeatures`
    signal (`best_effort_pinnable`) the score function OWNS, not a
    scheduler-imposed bonus. A custom `score_fn` that ignores the feature produces
    NO nudge, so the tie falls to the alphabetical default (`A`) — whereas the
    DEFAULT scorer prefers the capable `Z` (see the tie test above). Proves the
    scheduler no longer imposes a fixed bonus on arbitrary custom scorers."""
    from xrlenv.backends.base import CpuIsolation
    from xrlenv.control.scheduler import Scheduler

    def slack_only(f: Any) -> float:  # ignores best_effort_pinnable
        return f.resource_slack

    be = _iso_manifest(CpuIsolation.BEST_EFFORT, cpu_limit=2.0)
    catalog = TemplateCatalog()
    catalog.register(be)
    sched = Scheduler(
        [
            _iso_node("A", capable=False, free=0, total=0),
            _iso_node("Z", capable=True, free=8, total=8),
        ],
        catalog=catalog,
        state=InMemoryStateStore(),
        score_fn=slack_only,
    )
    assert sched.place(be).node.node_id == "A"  # custom scorer → no isolation nudge
