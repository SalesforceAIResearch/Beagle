"""P1.7.A.2 — Tests for ``RawGCReconciler`` (Raw-GC-M1 closure).

Closes the audit finding from P1.7.A.1: the existing GC layer-3
reconciler doesn't cover raw containers. The new reconciler
diffs ``RawContainerCoordinator.list_sessions()`` (in-memory
truth) against each node's ``list_raw_container_ids`` reply
(docker's truth, label-filtered) and acts on each side's orphans.

Tests use fake transports / coordinator / registry — no docker
or gRPC.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass, field
from dataclasses import dataclass as _dc
from dataclasses import field as _field
from typing import Any
from typing import Any as _Any

import pytest
from xrlenv.control import raw_gc_reconciler as _raw_gc_reconciler_module
from xrlenv.control.raw_container_service import (
    RAW_LIVENESS_QUARANTINE_DEFAULT_S,
    RawContainerCoordinator,
)
from xrlenv.control.raw_gc_reconciler import RawGCReconciler
from xrlenv.observability.metrics import MetricsRegistry

# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeNodeTransport:
    node_id: str = "node-A"
    backends: list[str] = field(default_factory=lambda: ["docker"])
    docker_container_ids: list[str] = field(default_factory=list)
    force_destroyed: list[str] = field(default_factory=list)
    raise_on_list: Exception | None = None
    raise_on_force_destroy: Exception | None = None
    list_delay_s: float = 0.0

    def supported_backends(self) -> list[str]:
        return list(self.backends)

    async def query_image(self, image: str) -> Any:
        # P1.7.B.2: scheduler runs query_image per placement and
        # the coordinator runs a pre-flight query_image on the
        # winner. Treat as always-present for these GC tests so
        # acquire goes through.
        @dataclass
        class _R:
            present: bool = True
        return _R()

    async def list_raw_container_ids(self, **kwargs: Any) -> list[str]:
        if self.list_delay_s:
            await asyncio.sleep(self.list_delay_s)
        if self.raise_on_list:
            raise self.raise_on_list
        return list(self.docker_container_ids)

    async def force_destroy_raw_container(
        self, *, container_id: str,
    ) -> None:
        if self.raise_on_force_destroy:
            raise self.raise_on_force_destroy
        self.force_destroyed.append(container_id)

    # The coordinator path uses these for coordinator-only orphan
    # cleanup; reuse the smallest acquire/destroy stubs.
    async def acquire_container(self, **kwargs: Any) -> Any:
        @dataclass
        class _Rec:
            rollout_id: str
            container_id: str
            container_name: str
            image: str
        return _Rec(
            rollout_id=kwargs["rollout_id"],
            container_id=f"c-{kwargs['rollout_id']}",
            container_name=f"name-c-{kwargs['rollout_id']}",
            image=kwargs["image"],
        )

    async def destroy_container(self, **kwargs: Any) -> None:
        # Coordinator-only orphan path calls coord.destroy →
        # transport.destroy_container → this method. No-op for
        # the test (the docker container is already gone).
        pass


@dataclass
class _FakeRegistry:
    transports: dict[str, _FakeNodeTransport]

    @property
    def node_ids(self) -> list[str]:
        return list(self.transports.keys())

    def get(self, node_id: str) -> _FakeNodeTransport | None:
        return self.transports.get(node_id)


def _make_coord(transports: list[_FakeNodeTransport]) -> RawContainerCoordinator:
    """P1.7.B.2: ``RawContainerCoordinator.acquire`` now goes through
    ``Scheduler.place(...)``. The fake mimics enough of that surface
    to pick the first docker-capable transport — equivalent to the
    pre-P1.7.B.2 first-available behaviour for the GC tests' purposes
    (these tests don't exercise image-affinity scoring)."""
    @dataclass
    class _Placement:
        node: Any
        backend: str = "docker"
        score: float = 1.0
        reservation_id: str = "fake-res-0"

    @dataclass
    class _S:
        """P1.7.A leak-fix surface: ``commit_placement`` /
        ``release_placement`` are no-ops here — the GC reconciler tests
        don't assert on the lifecycle, they just need the production
        ``acquire`` not to AttributeError."""

        nodes: list[Any]
        image_aware_placement: bool = True

        def place(
            self,
            manifest: Any,
            *,
            task_key: Any = None,
            backend: Any = None,
            image_present: Any = None,
            preferred_home_node: Any = None,
        ) -> _Placement:
            for node in self.nodes:
                if "docker" in node.supported_backends():
                    return _Placement(node=node)
            from xrlenv.errors import XRLEnvError as _Err
            raise _Err("no docker-capable node")

        def commit_placement(self, placement: Any) -> None:
            pass

        def release_placement(self, placement: Any) -> None:
            pass

    return RawContainerCoordinator(scheduler=_S(nodes=transports))


# ──────────────────────────────────────────────────────────────────────────────
# Sweep behavior
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_orphans_when_docker_and_coordinator_agree() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    report = await reconciler.reconcile_once()

    assert report["node-A"] == {"node_only": 0, "coordinator_only": 0}
    # The deadline sweep also runs every reconcile; the just-acquired
    # session is far from its deadline so nothing is reaped.
    assert report["__deadlines__"] == {"reaped": 0}
    # Nothing destroyed.
    assert transport.force_destroyed == []


@pytest.mark.asyncio
async def test_node_only_orphan_force_destroyed() -> None:
    """Container present on docker, not in coordinator. Common
    after a CP restart wipes the in-memory session map."""
    transport = _FakeNodeTransport(
        node_id="node-A",
        docker_container_ids=["orphan-id-A", "orphan-id-B"],
    )
    coord = _make_coord([transport])
    # Coordinator has no sessions.

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    # Issue #18 fix A — two-sweep confirmation: a node-only container
    # is only force-destroyed once observed on two consecutive sweeps.
    await reconciler.reconcile_once()
    report = await reconciler.reconcile_once()

    assert report["node-A"] == {"node_only": 2, "coordinator_only": 0}
    assert sorted(transport.force_destroyed) == ["orphan-id-A", "orphan-id-B"]


@pytest.mark.asyncio
async def test_coordinator_only_orphan_session_dropped() -> None:
    """Coordinator session whose docker container is gone — the
    harness or someone else removed it externally. Drop the
    in-memory session."""
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    # Docker says nothing — the container is gone.
    transport.docker_container_ids = []

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    report = await reconciler.reconcile_once()

    assert report["node-A"] == {"node_only": 0, "coordinator_only": 1}
    # Session is no longer tracked.
    assert coord.list_sessions() == []
    # Force-destroy was NOT called for the coordinator-only path
    # (the docker container's already gone).
    assert transport.force_destroyed == []
    # Sanity: session was the one we acquired.
    assert session.rollout_id is not None


@pytest.mark.asyncio
async def test_mixed_orphans_each_side() -> None:
    """One coordinator-only session + two node-only orphans on
    the same node."""
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    await coord.acquire(image="busybox:1")
    transport.docker_container_ids = ["orphan-1", "orphan-2"]
    # Coordinator session's container_id is NOT in the docker list.

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    # Sweep 1 observes both orphan classes; the coordinator-only
    # session is dropped immediately, the node-only orphans wait for
    # the two-sweep confirmation.
    report = await reconciler.reconcile_once()
    assert report["node-A"] == {"node_only": 2, "coordinator_only": 1}

    await reconciler.reconcile_once()  # sweep 2 confirms the node-only orphans
    assert sorted(transport.force_destroyed) == ["orphan-1", "orphan-2"]
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_per_node_failure_doesnt_stop_other_nodes() -> None:
    """One node's list_raw_container_ids raises — the reconciler
    skips that node and proceeds with the others."""
    bad = _FakeNodeTransport(
        node_id="node-bad",
        raise_on_list=RuntimeError("node went away"),
    )
    good = _FakeNodeTransport(
        node_id="node-good",
        docker_container_ids=["orphan-on-good"],
    )
    coord = _make_coord([bad, good])

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-bad": bad, "node-good": good}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    await reconciler.reconcile_once()
    report = await reconciler.reconcile_once()  # two-sweep confirmation

    assert "node-bad" not in report  # skipped
    assert report["node-good"] == {"node_only": 1, "coordinator_only": 0}
    assert good.force_destroyed == ["orphan-on-good"]


@pytest.mark.asyncio
async def test_per_node_timeout_logs_warning_not_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #18: a node too slow to answer ``list_raw_container_ids``
    (docker daemon swamped by concurrent cold pulls) is an expected,
    self-healing condition — skip + retry next sweep. It must log at
    WARNING without a traceback, and must not stop the sweep on other
    nodes.
    """
    slow = _FakeNodeTransport(node_id="node-slow", list_delay_s=10.0)
    fast = _FakeNodeTransport(node_id="node-fast", docker_container_ids=[])
    coord = _make_coord([slow, fast])

    reconciler = RawGCReconciler(
        registry=_FakeRegistry(  # type: ignore[arg-type]
            {"node-slow": slow, "node-fast": fast},
        ),
        coordinator=coord,
        per_node_timeout_s=0.05,
    )
    with caplog.at_level("WARNING", logger="xrlenv.control.raw_gc_reconciler"):
        report = await reconciler.reconcile_once()

    # Slow node skipped; fast node still swept this sweep.
    assert "node-slow" not in report
    assert report["node-fast"] == {"node_only": 0, "coordinator_only": 0}

    slow_recs = [r for r in caplog.records if "node-slow" in r.getMessage()]
    assert slow_recs, "expected a log record naming the timed-out node"
    assert all(r.levelname == "WARNING" for r in slow_recs)
    # A handled timeout must not carry a traceback.
    assert all(r.exc_info is None for r in slow_recs)
    assert any("timed out" in r.getMessage() for r in slow_recs)


@pytest.mark.asyncio
async def test_node_only_orphan_needs_two_sweeps_to_destroy() -> None:
    """Issue #18 fix A — two-sweep confirmation: a node-only container
    is NOT force-destroyed on the first sweep it is observed; only
    after it persists into a second consecutive sweep."""
    transport = _FakeNodeTransport(
        node_id="node-A", docker_container_ids=["c1"],
    )
    coord = _make_coord([transport])
    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )

    await reconciler.reconcile_once()  # sweep 1: observed only
    assert transport.force_destroyed == []

    await reconciler.reconcile_once()  # sweep 2: confirmed
    assert transport.force_destroyed == ["c1"]


@pytest.mark.asyncio
async def test_node_only_container_that_gains_a_session_is_spared() -> None:
    """Issue #18 fix A — the case-2 race. A container created during
    an in-flight acquire shows node-only until the acquire reply
    registers its session. If it stops being node-only before the
    second sweep, it must never be force-destroyed.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )

    # Sweep 1: the node has created a container for an in-flight
    # acquire; the coordinator has no session for it yet.
    transport.docker_container_ids = ["c-inflight"]
    await reconciler.reconcile_once()
    assert transport.force_destroyed == []

    # The acquire reply lands — the coordinator registers a session,
    # and docker now reports the container under its tracked id.
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    # Sweep 2: the once-node-only container is no longer node-only,
    # so the two-sweep intersection is empty — nothing destroyed.
    await reconciler.reconcile_once()
    assert transport.force_destroyed == []


@pytest.mark.asyncio
async def test_node_only_classifies_terminal_rollout_as_deferred_teardown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #18 fix B — a node-only container whose ``raw_rollouts``
    row is terminal is a leftover from a ``destroy`` whose node-side
    removal didn't finish: logged at INFO as routine deferred
    teardown. A container with no row is genuinely unexplained:
    logged at WARNING. Both are still reaped."""
    transport = _FakeNodeTransport(
        node_id="node-A",
        docker_container_ids=["c-terminal", "c-unknown"],
    )
    state = _FakeStateStore(rows=[
        _FakeRow(
            rollout_id="rr-done", status="released",
            created_at=_time.time(), container_id="c-terminal",
        ),
    ])
    reconciler = _make_reconciler_with_state([transport], state)

    await reconciler.reconcile_once()  # sweep 1: observe
    with caplog.at_level("INFO", logger="xrlenv.control.raw_gc_reconciler"):
        await reconciler.reconcile_once()  # sweep 2: confirm + classify

    # Both are reaped regardless of classification.
    assert sorted(transport.force_destroyed) == ["c-terminal", "c-unknown"]

    term = [r for r in caplog.records if "c-terminal" in r.getMessage()]
    unknown = [r for r in caplog.records if "c-unknown" in r.getMessage()]
    assert term and all(r.levelname == "INFO" for r in term)
    assert any("deferred teardown" in r.getMessage() for r in term)
    assert unknown and all(r.levelname == "WARNING" for r in unknown)
    assert any("orphan" in r.getMessage() for r in unknown)


@pytest.mark.asyncio
async def test_force_destroy_failure_logged_not_propagated() -> None:
    """A force_destroy failure on one orphan doesn't stop the
    reconciler from processing siblings or the next sweep."""
    transport = _FakeNodeTransport(
        node_id="node-A",
        docker_container_ids=["orphan-X"],
        raise_on_force_destroy=RuntimeError("docker daemon hiccup"),
    )
    coord = _make_coord([transport])

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    # Doesn't raise; report still records the orphan count
    # (whether or not the destroy succeeded — count reflects what
    # was found, not what was successfully cleaned). Two sweeps so
    # the orphan is confirmed and force_destroy is actually attempted.
    await reconciler.reconcile_once()
    report = await reconciler.reconcile_once()
    assert report["node-A"] == {"node_only": 1, "coordinator_only": 0}


@pytest.mark.asyncio
async def test_coordinator_only_destroy_failure_doesnt_abort_sweep() -> None:
    """Audit Raw-GC-M2 closure: a non-XRLEnvError from the
    wire-level destroy on a coordinator-only orphan must not
    abort ``reconcile_once`` — sibling orphans + later nodes
    still need to be processed. Symmetric with
    ``_handle_node_only``'s log-and-swallow of force_destroy
    failures."""
    transport_a = _FakeNodeTransport(node_id="node-A")
    transport_b = _FakeNodeTransport(
        node_id="node-B",
        docker_container_ids=["orphan-on-B"],
    )
    coord = _make_coord([transport_a])
    # Acquire on node-A so coordinator has a session there.
    session = await coord.acquire(image="busybox:1")
    # Make node-A's docker say "no such container" so the session
    # becomes a coordinator-only orphan, AND make the
    # transport's destroy_container raise a non-XRLEnvError to
    # trigger the bug's escape path.
    transport_a.docker_container_ids = []

    async def _raising_destroy(**kwargs: Any) -> None:
        raise RuntimeError("transient gRPC blip")
    transport_a.destroy_container = _raising_destroy  # type: ignore[method-assign]

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({  # type: ignore[arg-type]
            "node-A": transport_a, "node-B": transport_b,
        }),
        coordinator=coord,
    )
    # Without the fix this raises and skips node-B; with the fix
    # the sweep completes both nodes.
    report = await reconciler.reconcile_once()  # sweep 1
    assert report["node-A"]["coordinator_only"] == 1
    assert report["node-B"] == {"node_only": 1, "coordinator_only": 0}

    # Critically: node-B still got swept + (after the two-sweep
    # confirmation) its orphan got force-destroyed despite the
    # failure on node-A.
    await reconciler.reconcile_once()  # sweep 2 confirms node-B's orphan
    assert transport_b.force_destroyed == ["orphan-on-B"]
    # Coordinator session was best-effort dropped so the next
    # sweep doesn't see the same orphan immediately.
    assert coord.list_sessions() == []
    # Sanity: session existed before.
    assert session.rollout_id


@pytest.mark.asyncio
async def test_orphans_scoped_by_node() -> None:
    """A coordinator session on node-A's container_id should NOT
    be flagged as a coordinator-only orphan when a different node
    (node-B) is being swept."""
    transport_a = _FakeNodeTransport(node_id="node-A")
    transport_b = _FakeNodeTransport(node_id="node-B")
    coord = _make_coord([transport_a])  # node-A is the picked one
    session = await coord.acquire(image="busybox:1")
    # node-A's docker matches the session.
    transport_a.docker_container_ids = [session.container_id]
    # node-B's docker has its own orphan.
    transport_b.docker_container_ids = ["orphan-on-B"]

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({  # type: ignore[arg-type]
            "node-A": transport_a, "node-B": transport_b,
        }),
        coordinator=coord,
    )
    await reconciler.reconcile_once()
    report = await reconciler.reconcile_once()  # two-sweep confirmation

    # node-A clean: session matches.
    assert report["node-A"] == {"node_only": 0, "coordinator_only": 0}
    # node-B has the orphan. The session on node-A is NOT
    # mistakenly counted as a coordinator-only orphan on node-B.
    assert report["node-B"] == {"node_only": 1, "coordinator_only": 0}
    assert transport_b.force_destroyed == ["orphan-on-B"]


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 fix #3 — SQLite ghost reconciler (_reconcile_sqlite)
# ──────────────────────────────────────────────────────────────────────────────


@_dc
class _FakeRow:
    """Minimal raw_rollout row double."""
    rollout_id: str
    status: str
    created_at: float  # epoch seconds
    container_id: str | None = None


@_dc
class _FakeStateStore:
    """Minimal StateStore surface that _reconcile_sqlite needs."""

    rows: list[_FakeRow] = _field(default_factory=list)
    updates: list[dict] = _field(default_factory=list)

    def list_raw_rollouts(
        self, *, status: str | None = None,
    ) -> list[_FakeRow]:
        if status is None:
            return list(self.rows)
        return [r for r in self.rows if r.status == status]

    def update_raw_rollout(self, rollout_id: str, **fields: _Any) -> None:
        self.updates.append({"rollout_id": rollout_id, **fields})
        for row in self.rows:
            if row.rollout_id == rollout_id:
                for k, v in fields.items():
                    if hasattr(row, k):
                        setattr(row, k, v)


def _make_reconciler_with_state(
    transports: list[_FakeNodeTransport],
    state: _FakeStateStore,
    running_stale_s: float = 60.0,
    coord: RawContainerCoordinator | None = None,
) -> RawGCReconciler:
    coord = coord if coord is not None else _make_coord(transports)
    return RawGCReconciler(
        registry=_FakeRegistry({t.node_id: t for t in transports}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        running_stale_s=running_stale_s,
    )


@pytest.mark.asyncio
async def test_startup_sweep_flips_acquiring_row_not_in_flight() -> None:
    """startup sweep (_reconcile_sqlite in start()) marks an
    ``acquiring`` row as ``failed`` when its rollout_id is in neither
    the live-session set NOR the in-flight acquiring set. After a
    control-plane restart the acquiring set is empty (the coroutines
    died with the process), so every leftover ``acquiring`` row is a
    genuine ghost — regardless of age."""
    transport = _FakeNodeTransport(node_id="node-A")
    state = _FakeStateStore(rows=[
        _FakeRow(
            rollout_id="r-ghost",
            status="acquiring",
            # Even a *young* row is a ghost post-restart: the acquire
            # coroutine that would own it can't have survived the
            # restart. No age grace for acquiring rows.
            created_at=_time.time() - 5.0,
        ),
    ])
    reconciler = _make_reconciler_with_state([transport], state)

    await reconciler.start()
    await reconciler.shutdown()

    flipped = [u for u in state.updates if u["rollout_id"] == "r-ghost"]
    assert len(flipped) == 1
    assert flipped[0]["status"] == "failed"
    assert "lost-on-restart" in flipped[0]["error"]


@pytest.mark.asyncio
async def test_acquiring_row_in_flight_is_not_swept() -> None:
    """Audit M1 (round 2): an ``acquiring`` row whose rollout_id is in
    the coordinator's in-flight acquiring set must NOT be swept — no
    matter how old the row is. This is the case a fixed stale-age
    window could not handle: a consumer that raised ``queue_timeout_s``
    to e.g. 7200s leaves a legitimately-queued acquire parked far
    longer than any reasonable window, but the set tracks the live
    coroutine directly so there's no time proxy to outgrow."""
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    # Simulate an acquire parked in the admission queue: the row is
    # written ``acquiring`` and the rollout_id sits in the
    # coordinator's in-flight set, but no session exists yet.
    coord._acquiring_ids.add("r-queued")
    state = _FakeStateStore(rows=[
        _FakeRow(
            rollout_id="r-queued",
            status="acquiring",
            # 2h old — far past any plausible fixed window.
            created_at=_time.time() - 7200.0,
        ),
    ])
    reconciler = _make_reconciler_with_state(
        [transport], state, coord=coord,
    )

    # Periodic sweep (same process — the in-flight set is populated).
    report = await reconciler.reconcile_once()

    assert state.updates == [], (
        "an acquire still in flight (in _acquiring_ids) must never be "
        "swept, regardless of row age"
    )
    assert report["__sqlite__"]["ghosts"] == 0

    # Once the acquire finishes (rollout_id leaves the in-flight set),
    # a subsequent sweep treats the now-orphaned row as a ghost.
    coord._acquiring_ids.discard("r-queued")
    await reconciler.reconcile_once()
    flipped = [u for u in state.updates if u["rollout_id"] == "r-queued"]
    assert len(flipped) == 1 and flipped[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_startup_sweep_flips_stale_running_row() -> None:
    """startup sweep flips a ``running`` row older than ``running_stale_s``."""
    transport = _FakeNodeTransport(node_id="node-A")
    state = _FakeStateStore(rows=[
        _FakeRow(
            rollout_id="r-stale-run",
            status="running",
            created_at=_time.time() - 120.0,  # older than 60s grace
        ),
    ])
    reconciler = _make_reconciler_with_state(
        [transport], state, running_stale_s=60.0,
    )

    await reconciler.start()
    await reconciler.shutdown()

    flipped = [u for u in state.updates if u["rollout_id"] == "r-stale-run"]
    assert len(flipped) == 1
    assert flipped[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_young_running_row_within_grace_is_not_swept() -> None:
    """A ``running`` row younger than ``running_stale_s`` is given the
    benefit of the doubt — guards the sub-second race between the
    ``_sessions`` insert and the ``status="running"`` SQLite write."""
    transport = _FakeNodeTransport(node_id="node-A")
    state = _FakeStateStore(rows=[
        _FakeRow(
            rollout_id="r-young-run",
            status="running",
            created_at=_time.time() - 5.0,  # within the 60s grace
        ),
    ])
    reconciler = _make_reconciler_with_state(
        [transport], state, running_stale_s=60.0,
    )

    await reconciler.start()
    await reconciler.shutdown()

    assert state.updates == [], (
        "a running row within the grace window must not be swept"
    )


@pytest.mark.asyncio
async def test_startup_sweep_skips_row_in_live_sessions() -> None:
    """A row whose rollout_id IS in the coordinator's live sessions must
    not be flipped — it's genuinely in flight, not a ghost."""
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")

    state = _FakeStateStore(rows=[
        _FakeRow(
            rollout_id=session.rollout_id,
            status="running",
            created_at=_time.time() - 120.0,  # age would qualify
        ),
    ])
    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        running_stale_s=60.0,
    )

    await reconciler.start()
    await reconciler.shutdown()

    assert state.updates == [], (
        "live-session row must not be flipped by startup sweep"
    )


@pytest.mark.asyncio
async def test_state_none_is_noop() -> None:
    """``state=None`` (legacy path) — _reconcile_sqlite does nothing,
    reconcile_once returns no ``__sqlite__`` key."""
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=None,
    )
    report = await reconciler.reconcile_once()

    assert "__sqlite__" not in report


@pytest.mark.asyncio
async def test_periodic_sweep_writes_sqlite_ghost_count_to_report() -> None:
    """``reconcile_once`` populates ``report["__sqlite__"] = {"ghosts": N}``
    when a StateStore is wired. Verifies the periodic call path (not
    startup) stores the ghost count for dashboards."""
    transport = _FakeNodeTransport(
        node_id="node-A", docker_container_ids=[],
    )
    state = _FakeStateStore(rows=[
        _FakeRow(
            rollout_id="ghost-1",
            status="acquiring",
            created_at=_time.time() - 1000.0,
        ),
        _FakeRow(
            rollout_id="ghost-2",
            status="running",
            created_at=_time.time() - 200.0,
        ),
    ])
    reconciler = _make_reconciler_with_state(
        [transport], state, running_stale_s=60.0,
    )

    report = await reconciler.reconcile_once()

    assert "__sqlite__" in report
    assert report["__sqlite__"]["ghosts"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 — deadline sweep (_reconcile_deadlines)
# ──────────────────────────────────────────────────────────────────────────────


from dataclasses import replace as _replace  # noqa: E402


@pytest.mark.asyncio
async def test_deadline_sweep_reaps_expired_session() -> None:
    """A session past its ``deadline_at`` is force-destroyed by the
    reconciler — the only path that cleans up a container abandoned
    by a dead consumer (the docker diff + SQLite sweep both see it
    as healthy)."""
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    # docker side agrees — keeps the docker-diff from reaping it as a
    # coordinator-only orphan before the deadline sweep runs.
    transport.docker_container_ids = [session.container_id]
    # Force the session's deadline into the past.
    coord._sessions[session.rollout_id] = _replace(
        session, deadline_at=_time.time() - 30.0,
    )

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    report = await reconciler.reconcile_once()

    assert report["__deadlines__"] == {"reaped": 1}
    # Session dropped from the coordinator → capacity released.
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_deadline_sweep_leaves_session_within_deadline() -> None:
    """A session whose deadline is still in the future is untouched."""
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    # docker side agrees — keeps the docker-diff from reaping it as a
    # coordinator-only orphan before the deadline sweep runs.
    transport.docker_container_ids = [session.container_id]
    # Default cap (4h) leaves deadline_at far in the future.
    assert session.deadline_at > _time.time() + 3600.0

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    report = await reconciler.reconcile_once()

    assert report["__deadlines__"] == {"reaped": 0}
    assert [s.rollout_id for s in coord.list_sessions()] == [
        session.rollout_id,
    ]


@pytest.mark.asyncio
async def test_deadline_reap_seals_row_reaped_with_reason() -> None:
    """The reaped session's raw_rollouts row is sealed ``reaped`` (NOT
    ``failed``) with a reason that names the deadline overrun — so an
    operator can tell a force-reap apart from a clean consumer destroy,
    and reaps don't inflate the rollout failure rate."""
    updates: list[dict] = []

    class _State:
        def find_registered_preferred_home(self, image: str) -> None:
            return None

        def record_raw_rollout(self, record: _Any) -> None:
            pass

        def update_raw_rollout(self, rollout_id: str, **fields: _Any) -> None:
            updates.append({"rollout_id": rollout_id, **fields})

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    coord._state = _State()  # type: ignore[assignment]
    session = await coord.acquire(image="busybox:1")
    # docker side agrees — keeps the docker-diff from reaping it as a
    # coordinator-only orphan before the deadline sweep runs.
    transport.docker_container_ids = [session.container_id]
    coord._sessions[session.rollout_id] = _replace(
        session, deadline_at=_time.time() - 5.0,
    )

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )
    await reconciler.reconcile_once()

    reaped = [u for u in updates if u.get("status") == "reaped"]
    assert len(reaped) == 1
    assert "session deadline exceeded" in reaped[0]["error"]
    # A reap is not a failure — the failure rate must not be inflated.
    assert not [u for u in updates if u.get("status") == "failed"]


# ──────────────────────────────────────────────────────────────────────────────
# Consumer-liveness reaper (heartbeat-based; reaps sessions a dead consumer
# abandoned, without touching healthy long-running ones)
# ──────────────────────────────────────────────────────────────────────────────


def _liveness_reconciler(
    coord: RawContainerCoordinator, transport: _FakeNodeTransport,
) -> RawGCReconciler:
    return RawGCReconciler(
        registry=_FakeRegistry({transport.node_id: transport}),  # type: ignore[arg-type]
        coordinator=coord,
    )


@pytest.mark.asyncio
async def test_liveness_not_reaped_without_heartbeat() -> None:
    # A session whose consumer never heartbeated falls back to the deadline
    # cap — the liveness sweep ignores it even when its clock is stale.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord._last_seen_at[session.rollout_id] = 0.0  # stale, but never heartbeat
    report = await _liveness_reconciler(coord, transport).reconcile_once()
    assert report["__liveness__"] == {"reaped": 0, "suspect": 0}
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_liveness_reaped_when_heartbeated_and_stale() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = 0.0  # silent past the quarantine
    report = await _liveness_reconciler(coord, transport).reconcile_once()
    assert report["__liveness__"] == {"reaped": 1, "suspect": 1}
    assert coord.list_sessions() == []  # force-reaped


@pytest.mark.asyncio
async def test_liveness_not_reaped_with_inflight_rpc() -> None:
    # An open session-scoped RPC (e.g. a long blocking exec) exempts the
    # session even if heartbeats lapsed — the consumer is connected + waiting.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = 0.0
    coord._inflight_rpcs[session.rollout_id] = 1  # RPC in flight
    report = await _liveness_reconciler(coord, transport).reconcile_once()
    assert report["__liveness__"] == {"reaped": 0, "suspect": 0}
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_liveness_not_reaped_when_fresh() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])  # fresh beat (now)
    report = await _liveness_reconciler(coord, transport).reconcile_once()
    assert report["__liveness__"] == {"reaped": 0, "suspect": 0}
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_mark_heartbeat_and_session_rpc_touch() -> None:
    coord = _make_coord([_FakeNodeTransport(node_id="node-A")])
    session = await coord.acquire(image="busybox:1")
    rid = session.rollout_id
    # mark_heartbeat: recognizes a known session, ignores unknown ids.
    assert coord.mark_heartbeat([rid, "not-a-session"]) == 1
    assert rid in coord._heartbeated
    # _session_rpc: in-flight during the body, drained after, clock bumped.
    coord._last_seen_at[rid] = 0.0
    with coord._session_rpc(rid):
        assert coord._inflight_rpcs[rid] == 1
    assert coord._inflight_rpcs.get(rid, 0) == 0
    assert coord._last_seen_at[rid] > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Two-phase liveness: TTL marks SUSPECT, only the quarantine horizon destroys.
#
# The distinction this pins is the whole point of the split. Silence is not
# death: a consumer whose host stalls (memory reclaim, a frozen VM) is alive and
# will come back, but at the TTL it is indistinguishable from one that exited.
# Destroying at the TTL threw away live work — 243 sessions in the 2026-08-19 cn
# run. See notes/design-consumer-liveness-contract.md.
# ──────────────────────────────────────────────────────────────────────────────


def _quarantine_coord(
    transports: list[_FakeNodeTransport], *, ttl: float = 120.0, quarantine: float = 900.0,
) -> RawContainerCoordinator:
    coord = _make_coord(transports)
    coord._liveness_ttl_s = ttl
    coord._liveness_quarantine_s = quarantine
    return coord


@pytest.mark.asyncio
async def test_stale_past_ttl_is_suspect_but_not_destroyed() -> None:
    # THE regression this exists to prevent: past the TTL, inside the horizon.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0  # >120s, <900s

    report = await _liveness_reconciler(coord, transport).reconcile_once()

    assert report["__liveness__"] == {"reaped": 0, "suspect": 1}
    assert len(coord.list_sessions()) == 1        # container survives
    assert coord.suspect_count() == 1


@pytest.mark.asyncio
async def test_suspect_session_is_destroyed_once_past_quarantine() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    reconciler = _liveness_reconciler(coord, transport)

    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    assert (await reconciler.reconcile_once())["__liveness__"]["reaped"] == 0

    # Still silent when the horizon passes — now it really is abandoned.
    coord._last_seen_at[session.rollout_id] = _time.time() - 1000.0
    report = await reconciler.reconcile_once()
    assert report["__liveness__"]["reaped"] == 1
    assert coord.list_sessions() == []


@pytest.mark.asyncio
async def test_heartbeat_during_quarantine_clears_suspicion() -> None:
    # A stalled consumer that comes back keeps its session AND its suspect mark
    # is retired, so it starts from a clean clock rather than a countdown.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    reconciler = _liveness_reconciler(coord, transport)

    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    await reconciler.reconcile_once()
    assert coord.suspect_count() == 1

    coord.mark_heartbeat([session.rollout_id])          # consumer resumes
    assert coord.suspect_count() == 0

    report = await reconciler.reconcile_once()
    assert report["__liveness__"] == {"reaped": 0, "suspect": 0}
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_session_rpc_during_quarantine_clears_suspicion() -> None:
    # An ordinary session RPC is as good a liveness signal as a heartbeat.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    reconciler = _liveness_reconciler(coord, transport)

    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    await reconciler.reconcile_once()
    assert coord.suspect_count() == 1

    with coord._session_rpc(session.rollout_id):
        pass
    assert coord.suspect_count() == 0


@pytest.mark.asyncio
async def test_suspect_marking_is_idempotent_across_sweeps() -> None:
    # Repeated sweeps inside the horizon must not re-count or reset the clock.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    reconciler = _liveness_reconciler(coord, transport)

    first = await reconciler.reconcile_once()
    marked_at = coord.suspect_since(session.rollout_id)
    second = await reconciler.reconcile_once()

    assert first["__liveness__"]["suspect"] == 1
    assert second["__liveness__"]["suspect"] == 0     # already marked
    assert coord.suspect_since(session.rollout_id) == marked_at
    assert coord.suspect_count() == 1


@pytest.mark.asyncio
async def test_destroy_clears_all_liveness_bookkeeping() -> None:
    # The suspect map must be popped wherever the other liveness dicts are, or
    # it leaks an entry per session for the control plane's lifetime.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    await _liveness_reconciler(coord, transport).reconcile_once()
    assert coord.suspect_count() == 1

    await coord.destroy(
        rollout_id=session.rollout_id, container_id=session.container_id,
    )

    assert coord._suspect_since == {}
    assert coord._last_seen_at == {}
    assert coord._inflight_rpcs == {}
    assert coord._heartbeated == set()


@pytest.mark.asyncio
async def test_node_lost_clears_suspect_bookkeeping() -> None:
    # Same leak class as test_destroy_clears_all_liveness_bookkeeping, but
    # through handle_node_lost — the OTHER path (besides destroy) that pops
    # _last_seen_at, and one the audit flagged by name as easy to miss.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    await _liveness_reconciler(coord, transport).reconcile_once()
    assert coord.suspect_count() == 1

    await coord.handle_node_lost("node-A")

    assert coord._suspect_since == {}
    assert coord._last_seen_at == {}
    assert coord._inflight_rpcs == {}
    assert coord._heartbeated == set()


@pytest.mark.asyncio
async def test_seal_orphan_clears_suspect_bookkeeping() -> None:
    # seal_orphan is the coordinator-only-orphan path (docker already dropped
    # the container) — another _last_seen_at-popping site the suspect map
    # must track.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    await _liveness_reconciler(coord, transport).reconcile_once()
    assert coord.suspect_count() == 1

    await coord.seal_orphan(
        rollout_id=session.rollout_id, container_id=session.container_id,
    )

    assert coord._suspect_since == {}
    assert coord._last_seen_at == {}


@pytest.mark.asyncio
async def test_drop_orphan_session_clears_suspect_bookkeeping() -> None:
    # drop_orphan_session is the seal_orphan fallback (audit Low) — verify it
    # carries the same suspect-map bookkeeping the others do.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0
    await _liveness_reconciler(coord, transport).reconcile_once()
    assert coord.suspect_count() == 1

    await coord.drop_orphan_session(session.rollout_id, session.container_id)

    assert coord._suspect_since == {}
    assert coord._last_seen_at == {}


@pytest.mark.asyncio
async def test_never_heartbeated_session_exempt_from_suspect_phase() -> None:
    # Symmetric with test_liveness_not_reaped_without_heartbeat: a session
    # that never heartbeats must be exempt from BOTH phases, not just reap.
    # The shared _liveness_stale() helper is what's supposed to guarantee
    # this — pin it directly against phase 1.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord._last_seen_at[session.rollout_id] = 0.0  # ancient, but never heartbeat

    report = await _liveness_reconciler(coord, transport).reconcile_once()

    assert report["__liveness__"] == {"reaped": 0, "suspect": 0}
    assert coord.suspect_count() == 0
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_inflight_rpc_exempts_from_suspect_phase() -> None:
    # Symmetric with test_liveness_not_reaped_with_inflight_rpc: an open
    # session-scoped RPC must exempt from BOTH phases via the same shared
    # helper, not just reap.
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0  # > TTL
    coord._inflight_rpcs[session.rollout_id] = 1  # RPC in flight

    report = await _liveness_reconciler(coord, transport).reconcile_once()

    assert report["__liveness__"] == {"reaped": 0, "suspect": 0}
    assert coord.suspect_count() == 0
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_liveness_reap_batch_paces_destroys_but_marks_all_suspects() -> None:
    # _mark_liveness_suspects is documented "cheap and uncapped" (no I/O), so
    # every stale session should be flagged in one sweep even when the PACED
    # destroy pass (_LIVENESS_REAP_BATCH) can only take a few of them this
    # round. The remainder must stay correctly suspect (not lost, not
    # double-counted) until a later sweep's batch reaches them.
    original_batch = _raw_gc_reconciler_module._LIVENESS_REAP_BATCH
    _raw_gc_reconciler_module._LIVENESS_REAP_BATCH = 2
    try:
        transport = _FakeNodeTransport(node_id="node-A")
        coord = _quarantine_coord([transport])
        sessions = [await coord.acquire(image="busybox:1") for _ in range(3)]
        transport.docker_container_ids = [s.container_id for s in sessions]
        coord.mark_heartbeat([s.rollout_id for s in sessions])
        for s in sessions:
            # Already past the quarantine horizon on first observation (e.g.
            # a slow reconciler / just-readopted session) — legitimately
            # both suspect-eligible and reap-eligible in the same sweep.
            coord._last_seen_at[s.rollout_id] = _time.time() - 1000.0
        reconciler = _liveness_reconciler(coord, transport)

        first = await reconciler.reconcile_once()
        assert first["__liveness__"] == {"reaped": 2, "suspect": 3}
        assert len(coord.list_sessions()) == 1       # batch cap held one back
        assert coord.suspect_count() == 1            # the held-back one

        second = await reconciler.reconcile_once()
        assert second["__liveness__"] == {"reaped": 1, "suspect": 0}
        assert coord.list_sessions() == []
        assert coord.suspect_count() == 0
    finally:
        _raw_gc_reconciler_module._LIVENESS_REAP_BATCH = original_batch


@pytest.mark.asyncio
async def test_liveness_reap_skips_a_session_that_recovers_mid_sweep() -> None:
    """Round-3 audit finding: ``_reconcile_liveness`` snapshots
    ``liveness_reap_candidates()`` ONCE at the top of the sweep, then
    sequentially ``await``s a real wire ``destroy()`` per candidate. A
    session captured in that snapshot can have its consumer heartbeat (or
    issue any session RPC) WHILE an *earlier* sibling's destroy is still in
    flight — i.e. before its own turn in the loop is ever reached. Without a
    fresh re-check right before firing each destroy, that session gets
    force-destroyed anyway even though it just proved it is alive, silently
    breaking the feature's central promise: "any liveness signal ... clears
    suspicion." A real fleet-wide die-off (the scenario the quarantine exists
    for) can put many sessions in one batch, each destroy taking real wall
    time, giving plenty of room for this to bite.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    a = await coord.acquire(image="busybox:1")
    b = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [a.container_id, b.container_id]
    coord.mark_heartbeat([a.rollout_id, b.rollout_id])
    stale = _time.time() - 1000.0  # both well past the quarantine horizon
    coord._last_seen_at[a.rollout_id] = stale
    coord._last_seen_at[b.rollout_id] = stale

    # b's consumer heartbeats WHILE a's destroy is in flight — simulated by
    # hooking the wire call a's destroy makes. Insertion order (a before b)
    # means the sweep's sequential loop reaches a first.
    original_destroy_container = transport.destroy_container

    async def _destroy_container(**kwargs: Any) -> None:
        if kwargs["rollout_id"] == a.rollout_id:
            coord.mark_heartbeat([b.rollout_id])
        await original_destroy_container(**kwargs)

    transport.destroy_container = _destroy_container  # type: ignore[method-assign]

    report = await _liveness_reconciler(coord, transport).reconcile_once()

    live_ids = {s.rollout_id for s in coord.list_sessions()}
    assert a.rollout_id not in live_ids  # a never recovered — correctly reaped
    assert b.rollout_id in live_ids      # b recovered before its turn — must survive
    assert report["__liveness__"]["reaped"] == 1


@pytest.mark.asyncio
async def test_liveness_reap_retries_after_a_failed_destroy_and_stays_suspect() -> None:
    """Round-3 audit check (clean, not a defect — pinned as a regression test):
    a quarantine destroy whose wire call FAILS must retain the session
    (invariant 2 — capacity released only on node-confirmed destroy), clear
    ``is_destroying`` so a retry isn't wedged forever, and leave
    ``_suspect_since`` untouched so the session stays correctly suspect. The
    NEXT sweep must retry and succeed — a failed reap must never become a
    silent permanent no-op.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 1000.0

    calls = {"n": 0}
    original_destroy_container = transport.destroy_container

    async def _flaky_destroy_container(**kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient node hiccup")
        await original_destroy_container(**kwargs)

    transport.destroy_container = _flaky_destroy_container  # type: ignore[method-assign]
    reconciler = _liveness_reconciler(coord, transport)

    first = await reconciler.reconcile_once()
    assert first["__liveness__"]["reaped"] == 0          # destroy failed, not counted
    assert len(coord.list_sessions()) == 1                # session retained (invariant 2)
    assert coord.is_destroying(session.rollout_id) is False  # not wedged
    assert coord.suspect_since(session.rollout_id) is not None  # still suspect

    second = await reconciler.reconcile_once()
    assert second["__liveness__"]["reaped"] == 1
    assert coord.list_sessions() == []                     # retried and confirmed-destroyed


def _metric_value(reg: MetricsRegistry, name: str) -> float:
    """Read a single counter/gauge sample's value by (family) name."""
    suffixes = ("_total",)
    family_candidates = {name}
    for suffix in suffixes:
        if name.endswith(suffix):
            family_candidates.add(name.removesuffix(suffix))
    for family in reg.collector_registry.collect():
        if family.name not in family_candidates:
            continue
        for sample in family.samples:
            if sample.name == name:
                return float(sample.value)
    return 0.0


@pytest.mark.asyncio
async def test_suspect_total_and_gauge_increment_on_mark() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    metrics = MetricsRegistry()
    coord._metrics = metrics
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0  # suspect only

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        metrics=metrics,
    )
    await reconciler.reconcile_once()

    assert _metric_value(metrics, "xrlenv_raw_liveness_suspect_total") == 1.0
    assert _metric_value(metrics, "xrlenv_raw_sessions_suspect") == 1.0

    # Idempotent across sweeps — the second sweep must not re-count the same
    # still-suspect session.
    await reconciler.reconcile_once()
    assert _metric_value(metrics, "xrlenv_raw_liveness_suspect_total") == 1.0


@pytest.mark.asyncio
async def test_recovered_total_increments_once_per_real_recovery() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    metrics = MetricsRegistry()
    coord._metrics = metrics
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        metrics=metrics,
    )
    await reconciler.reconcile_once()
    assert coord.suspect_count() == 1

    coord.mark_heartbeat([session.rollout_id])  # consumer resumes: 1 real recovery
    assert _metric_value(metrics, "xrlenv_raw_liveness_recovered_total") == 1.0

    # A second heartbeat on an already-healthy session is not a NEW
    # recovery — _clear_suspect must not double count it.
    coord.mark_heartbeat([session.rollout_id])
    assert _metric_value(metrics, "xrlenv_raw_liveness_recovered_total") == 1.0


@pytest.mark.parametrize("quarantine_s", [30.0, 120.0, 0.0, -50.0])
@pytest.mark.asyncio
async def test_quarantine_at_or_below_ttl_is_clamped_to_ttl(
    quarantine_s: float,
) -> None:
    # Lower, equal, zero and negative all collapse to the same "would destroy
    # on the marking sweep" hazard. Clamping to the TTL does NOT fix it —
    # both phases read one clock, so quarantine == ttl still makes a session a
    # suspect AND a reap candidate in the same sweep. The clamp must restore a
    # real grace window.
    coord = RawContainerCoordinator(
        scheduler=_make_coord([])._scheduler,
        liveness_ttl_s=120.0,
        liveness_quarantine_s=quarantine_s,
    )
    assert coord._liveness_quarantine_s > 120.0
    assert coord._liveness_quarantine_s == RAW_LIVENESS_QUARANTINE_DEFAULT_S


@pytest.mark.asyncio
async def test_clamped_quarantine_still_grants_a_grace_window() -> None:
    """The clamp must be behavioural, not merely numeric.

    Asserting the clamped *number* is what let the equal-to-TTL bug through: a
    quarantine clamped to the TTL reads fine as an attribute while restoring
    destroy-on-TTL in practice. So drive an actual sweep on a misconfigured
    coordinator and require the session to survive being marked suspect.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    coord._liveness_ttl_s = 120.0
    coord._liveness_quarantine_s = 120.0        # the pre-fix clamp result
    coord._liveness_quarantine_s = max(
        RAW_LIVENESS_QUARANTINE_DEFAULT_S, 120.0 * 2.0,
    )                                            # what the clamp now yields
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 200.0   # past TTL

    report = await _liveness_reconciler(coord, transport).reconcile_once()

    assert report["__liveness__"] == {"reaped": 0, "suspect": 1}
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_reaped_total_counts_liveness_reaps_not_deadline_reaps() -> None:
    """`raw_liveness_reaped_total` must count the QUARANTINE sweep.

    Regression: the increment originally sat in `_reconcile_deadlines` (the
    unrelated 4 h wall-clock sweep), so a liveness reap reported
    `__liveness__.reaped == 1` while the counter stayed at zero — the metric the
    feature is judged by was wired to the wrong code path entirely.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    metrics = MetricsRegistry()
    coord._metrics = metrics
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 1000.0   # past horizon

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        metrics=metrics,
    )
    report = await reconciler.reconcile_once()

    assert report["__liveness__"]["reaped"] == 1
    assert _metric_value(metrics, "xrlenv_raw_liveness_reaped_total") == 1.0


@pytest.mark.asyncio
async def test_suspect_gauge_is_synced_after_the_destroy_pass() -> None:
    """The gauge must not read high once the reap has popped the suspect map.

    Regression: it was `.set()` inside the marking pass, which runs BEFORE the
    destroy pass, so a session marked-and-reaped in one sweep left the gauge
    reading 1 until the next sweep re-marked — stale for a whole interval on the
    metric operators watch.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    metrics = MetricsRegistry()
    coord._metrics = metrics
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 1000.0

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        metrics=metrics,
    )
    # Drive it in BOTH directions: asserting only the final 0.0 would also pass
    # if the gauge were never set at all.
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0   # past TTL only
    await reconciler.reconcile_once()
    assert _metric_value(metrics, "xrlenv_raw_sessions_suspect") == 1.0

    coord._last_seen_at[session.rollout_id] = _time.time() - 1000.0  # past horizon
    await reconciler.reconcile_once()
    assert coord.suspect_count() == 0
    assert _metric_value(metrics, "xrlenv_raw_sessions_suspect") == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SessionReaped — a platform teardown must not read as a stale handle.
#
# The consumer usually learns of a reap long afterwards, at its next session RPC
# (15 minutes later in the 2026-08-19 incident), so the message it gets is the
# only explanation it will ever have. "Acquire first." described a caller
# bookkeeping bug and was raised for both cases, which is why that incident read
# as a client bug for so long.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reaped_session_raises_typed_error_with_reason() -> None:
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped

    transport = _FakeNodeTransport(node_id="node-A")
    state = InMemoryStateStore()
    coord = _quarantine_coord([transport])
    coord._state = state
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 1000.0

    await _liveness_reconciler(coord, transport).reconcile_once()

    with pytest.raises(SessionReaped) as excinfo:
        coord._require_session(session.rollout_id)
    exc = excinfo.value
    assert "reaped by the control plane" in str(exc)
    assert "quarantine horizon" in exc.reason      # the recorded teardown reason
    assert exc.retryable is True                   # a fresh acquire will work
    assert exc.reaped_at is not None


@pytest.mark.asyncio
async def test_unknown_rollout_still_raises_the_stale_handle_error() -> None:
    # The other half of the contract: an id we never knew is NOT a reap, and
    # must not be dressed up as one.
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped, XRLEnvError

    coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
    coord._state = InMemoryStateStore()
    with pytest.raises(XRLEnvError) as excinfo:
        coord._require_session("never-existed")
    assert not isinstance(excinfo.value, SessionReaped)
    assert "Acquire first" in str(excinfo.value)


@pytest.mark.asyncio
async def test_missing_session_falls_back_when_state_read_fails() -> None:
    # The explanation lookup is best-effort: a broken state store must surface
    # the original "session is gone" error, never its own failure.
    from xrlenv.errors import XRLEnvError

    class _ExplodingState:
        def get_raw_rollout(self, rollout_id: str) -> Any:
            raise RuntimeError("state store is down")

    coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
    coord._state = _ExplodingState()  # type: ignore[assignment]
    with pytest.raises(XRLEnvError, match="Acquire first"):
        coord._require_session("some-id")


@pytest.mark.asyncio
async def test_missing_session_falls_back_when_no_state_store_is_wired() -> None:
    # A coordinator constructed without a state store at all (self._state is
    # None, the constructor default) must fall back the same as a broken one —
    # not every deployment wires the durable raw_rollouts store.
    from xrlenv.errors import SessionReaped, XRLEnvError

    coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
    assert coord._state is None
    with pytest.raises(XRLEnvError) as excinfo:
        coord._require_session("some-id")
    assert not isinstance(excinfo.value, SessionReaped)
    assert "Acquire first" in str(excinfo.value)


@pytest.mark.asyncio
async def test_released_session_does_not_raise_session_reaped() -> None:
    # A session the CONSUMER destroyed normally seals the row "released", not
    # "reaped" (raw_container_service.py: status is "reaped" only when the
    # destroy carried a reaper ``reason``). A miss on a released rollout id is
    # a stale handle, not a platform teardown, and must get the generic
    # message — not be misreported as a reap.
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped, XRLEnvError

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    coord._state = InMemoryStateStore()
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    await coord.destroy(rollout_id=session.rollout_id, container_id=session.container_id)

    record = coord._state.get_raw_rollout(session.rollout_id)
    assert record is not None
    assert record.status == "released"

    with pytest.raises(XRLEnvError) as excinfo:
        coord._require_session(session.rollout_id)
    assert not isinstance(excinfo.value, SessionReaped)
    assert "Acquire first" in str(excinfo.value)


@pytest.mark.asyncio
async def test_double_destroy_second_call_is_not_reported_as_reaped() -> None:
    # A consumer that calls destroy() twice (e.g. cleanup-on-error racing a
    # normal teardown) must see the ordinary stale-handle error on the second
    # call — not SessionReaped, which would misrepresent its OWN double-
    # destroy as a platform-initiated teardown.
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped, XRLEnvError

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    coord._state = InMemoryStateStore()
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    await coord.destroy(rollout_id=session.rollout_id, container_id=session.container_id)

    with pytest.raises(XRLEnvError) as excinfo:
        await coord.destroy(
            rollout_id=session.rollout_id, container_id=session.container_id,
        )
    assert not isinstance(excinfo.value, SessionReaped)
    assert "Acquire first" in str(excinfo.value)


@pytest.mark.asyncio
async def test_reaped_session_with_empty_recorded_reason_uses_default_text() -> None:
    # The reap path always writes a reason today, but ``_raise_for_missing_session``
    # treats ``error`` as best-effort text (``str(... or "consumer liveness
    # lost")``) — pin the fallback for a row whose error came back falsy
    # (None or "") so a future reap path that forgets to set it doesn't
    # silently raise SessionReaped with an empty/`"None"` explanation.
    from xrlenv.errors import SessionReaped

    class _Record:
        def __init__(self, error: str | None) -> None:
            self.status = "reaped"
            self.error = error
            self.finished_at = None

    class _FakeState:
        def __init__(self, record: _Record) -> None:
            self._record = record

        def get_raw_rollout(self, rollout_id: str) -> _Record:
            return self._record

    for empty_error in (None, ""):
        coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
        coord._state = _FakeState(_Record(empty_error))  # type: ignore[assignment]
        with pytest.raises(SessionReaped) as excinfo:
            coord._require_session("some-id")
        assert excinfo.value.reason == "consumer liveness lost"
        assert excinfo.value.reaped_at is None


@pytest.mark.asyncio
async def test_quarantine_at_or_beyond_the_deadline_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A horizon past the wall-clock deadline silently disables the feature.

    The liveness reaper only earns its keep by reclaiming SOONER than the
    deadline; at or beyond it the deadline sweep always destroys first, so the
    whole quarantine path becomes unreachable. Easy to hit by accident, because
    an invalid quarantine is derived from the TTL (``ttl * 2``), so merely
    raising the TTL past half the deadline produces it.
    """
    with caplog.at_level(logging.WARNING):
        RawContainerCoordinator(
            scheduler=_make_coord([])._scheduler,
            liveness_ttl_s=8000.0,          # -> synthesized quarantine 16000s
            liveness_quarantine_s=10.0,     # invalid, forces the derivation
            session_deadline_default_s=14400.0,
        )
    assert "effectively disabled" in caplog.text


@pytest.mark.asyncio
async def test_suspect_gauge_is_resynced_even_when_the_sweep_raises() -> None:
    """A raising sweep must still resync the gauge.

    It used to live inside the try, so an exception skipped it. That is not
    "stale for one interval" as the docstring claimed — if the failure repeats
    every sweep the gauge stays wrong indefinitely, on the metric this feature
    is judged by.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    metrics = MetricsRegistry()
    coord._metrics = metrics
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        metrics=metrics,
    )
    await reconciler.reconcile_once()
    assert _metric_value(metrics, "xrlenv_raw_sessions_suspect") == 1.0

    # Consumer comes back, then the next sweep blows up partway through.
    coord.mark_heartbeat([session.rollout_id])
    assert coord.suspect_count() == 0

    def _boom() -> list[Any]:
        raise RuntimeError("coordinator hiccup")

    coord.liveness_reap_candidates = _boom  # type: ignore[method-assign]
    report = await reconciler.reconcile_once()

    assert "__liveness__" in report          # never a KeyError trap
    assert _metric_value(metrics, "xrlenv_raw_sessions_suspect") == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SessionReaped audit round 3 — ``drop_orphan_session`` used to drop the
# in-memory session WITHOUT sealing the durable row at all, so a rollout that
# fell through the ``seal_orphan``-raised fallback could never be reported as
# ``SessionReaped`` again (any later touch got the generic "not found. Acquire
# first." message — the exact misreport this whole feature exists to fix).
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drop_orphan_session_seals_row_reaped_when_reason_given() -> None:
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    coord._state = InMemoryStateStore()
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    # drop_orphan_session is the FALLBACK the reconciler calls when
    # seal_orphan itself raised (defensive container-id mismatch, or a
    # swallowed state-store hiccup) — exercise it directly, the same way
    # RawGCReconciler._handle_coordinator_only does in its except branches.
    await coord.drop_orphan_session(
        session.rollout_id, session.container_id, reason="disk-pressure",
    )

    record = coord._state.get_raw_rollout(session.rollout_id)
    assert record is not None
    assert record.status == "reaped"
    assert record.error == "disk-pressure"

    # The whole point: a later touch must be reported as a platform teardown,
    # not misdiagnosed as a stale/unknown handle.
    with pytest.raises(SessionReaped) as excinfo:
        coord._require_session(session.rollout_id)
    assert excinfo.value.reason == "disk-pressure"


@pytest.mark.asyncio
async def test_drop_orphan_session_seals_row_released_when_no_reason() -> None:
    # Symmetric with seal_orphan: no reason (container vanished for some other
    # cause — OOM / external ``docker rm``) seals the clean-teardown status,
    # not ``reaped`` — a stale handle must still get the generic message, not
    # be misreported as a platform-initiated reap.
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped, XRLEnvError

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    coord._state = InMemoryStateStore()
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    await coord.drop_orphan_session(session.rollout_id, session.container_id)

    record = coord._state.get_raw_rollout(session.rollout_id)
    assert record is not None
    assert record.status == "released"

    with pytest.raises(XRLEnvError) as excinfo:
        coord._require_session(session.rollout_id)
    assert not isinstance(excinfo.value, SessionReaped)
    assert "Acquire first" in str(excinfo.value)


@pytest.mark.asyncio
async def test_reconciler_coordinator_only_fallback_seals_reaped_end_to_end() -> None:
    # End-to-end through RawGCReconciler._handle_coordinator_only itself
    # (not just the coordinator API): force seal_orphan to raise something
    # OTHER than the container-id-mismatch XRLEnvError — a bare, unexpected
    # failure inside its locked lookup, the scenario the reconciler's
    # ``except Exception:`` (the "state-store hiccup" catch-all) branch
    # documents. That branch falls back to drop_orphan_session with the SAME
    # (correct, matching) container_id. Before the fix, this path dropped the
    # in-memory session WITHOUT sealing the durable row at all, leaving it
    # exactly as it was pre-reap (``running``) — silently making
    # ``SessionReaped`` unreachable for this rollout forever. Note: the
    # sibling ``except XRLEnvError`` (container-id mismatch) branch passes a
    # deliberately STALE container_id to the fallback, so
    # ``drop_orphan_session``'s own generation-safety guard (only drop if the
    # id still matches) correctly no-ops there — covered separately at the
    # coordinator level above, not duplicated here.
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    coord._state = InMemoryStateStore()
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    class _ExplodeOnce(dict):
        """A ``_sessions`` stand-in whose first ``.get`` raises (simulating an
        unrelated internal hiccup mid-``seal_orphan``), then behaves normally —
        so the reconciler's own fallback ``drop_orphan_session`` call still
        finds the real session under its real (matching) container_id."""

        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self._boomed = False

        def get(self, *a: Any, **kw: Any) -> Any:
            if not self._boomed:
                self._boomed = True
                raise RuntimeError("simulated coordinator-internal hiccup")
            return super().get(*a, **kw)

    # Bypass the (irrelevant here) compose-project probe, which also reads
    # ``_sessions.get`` — so the single ``_ExplodeOnce`` trip wire lands on
    # seal_orphan's own lookup, not this unrelated one.
    coord.is_compose_project = lambda rollout_id: False  # type: ignore[method-assign]
    coord._sessions = _ExplodeOnce(coord._sessions)  # type: ignore[assignment]

    reconciler = _make_reconciler_with_state([transport], coord._state, coord=coord)

    await reconciler._handle_coordinator_only(
        "node-A", session.container_id, session.rollout_id, reason="disk-pressure",
    )

    record = coord._state.get_raw_rollout(session.rollout_id)
    assert record is not None
    assert record.status == "reaped"
    assert record.error == "disk-pressure"
    assert coord.list_sessions() == []

    with pytest.raises(SessionReaped) as excinfo:
        coord._require_session(session.rollout_id)
    assert excinfo.value.reason == "disk-pressure"


@pytest.mark.asyncio
async def test_session_reaped_survives_in_process_transport_with_reaped_at_intact() -> None:
    """Audit round 3 — angle: the IN-PROCESS transport never crosses gRPC, so it
    must not be conflated with the wire path. Round 1/2 verified SessionReaped
    over a REAL grpc.aio server (where ``reaped_at`` is documented as NOT
    surviving — no metadata key for it) and drove the coordinator directly
    in-process (never through ``InProcessTransport``/``CoordinatorRolloutService``
    at all). Neither covers this exact path: ``RawContainerCoordinator`` ->
    ``CoordinatorRolloutService.container_exec`` -> ``InProcessTransport`` ->
    caller — a direct in-memory call chain with no serialization boundary. This
    pins that ``reaped_at`` (which the wire path documents as always ``None``
    client-side) is preserved intact here, since a caller of the LOCAL runtime
    (no gRPC configured) reaches this exact chain.
    """
    from xrlenv.client.transport import InProcessTransport
    from xrlenv.control.service import CoordinatorRolloutService
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import SessionReaped

    transport_a = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport_a])
    coord._state = InMemoryStateStore()
    session = await coord.acquire(image="busybox:1")
    transport_a.docker_container_ids = [session.container_id]

    # The reap: consumer-liveness quarantine → destroy with a reason, exactly
    # the ``status="reaped"`` durable seal ``_raise_for_missing_session`` reads.
    await coord.destroy(
        rollout_id=session.rollout_id,
        container_id=session.container_id,
        reason="consumer went silent past quarantine",
    )

    # object() stands in for the case-1 RolloutCoordinator — never touched by
    # any raw-container call, per _require_raw_coordinator's implementation.
    service = CoordinatorRolloutService(
        coordinator=object(),  # type: ignore[arg-type]
        raw_container_coordinator=coord,
    )
    in_process = InProcessTransport(service)  # type: ignore[arg-type]

    with pytest.raises(SessionReaped) as excinfo:
        await in_process.container_exec(
            rollout_id=session.rollout_id,
            container_id=session.container_id,
            cmd=["echo", "hi"],
        )
    assert excinfo.value.reason == "consumer went silent past quarantine"
    # The load-bearing distinction from the wire path (round 1/2's real-gRPC
    # tests): reaped_at has no wire metadata key and is documented as always
    # None client-side there. In-process, there's no serialization boundary at
    # all — it must survive.
    assert excinfo.value.reaped_at is not None


@pytest.mark.asyncio
async def test_signal_during_its_own_destroy_is_not_counted_as_a_recovery() -> None:
    """A doomed session must not inflate the rescue metric.

    Candidacy is re-checked immediately before each destroy, but a consumer can
    still signal while its own wire call is in flight — and then it dies anyway,
    because a node-confirmed teardown cannot be rolled back. That residual is
    accepted; reporting it as a save is not. Counting it would inflate the very
    metric that measures work this feature rescued, and the "not reaped" log line
    would be false about a session being reaped.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    metrics = MetricsRegistry()
    coord._metrics = metrics
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 1000.0   # past horizon

    original = transport.destroy_container

    async def _destroy_with_late_signal(**kwargs: Any) -> Any:
        # The consumer wakes up mid-teardown, after the pre-destroy re-check.
        coord.mark_heartbeat([session.rollout_id])
        return await original(**kwargs)

    transport.destroy_container = _destroy_with_late_signal  # type: ignore[method-assign]

    report = await _liveness_reconciler(coord, transport).reconcile_once()

    assert report["__liveness__"]["reaped"] == 1        # the residual: it still dies
    assert coord.list_sessions() == []
    # ...but it is NOT reported as rescued work.
    assert _metric_value(metrics, "xrlenv_raw_liveness_recovered_total") == 0.0


@pytest.mark.asyncio
async def test_session_being_destroyed_is_not_marked_suspect() -> None:
    """A session already being torn down must not be reported as suspect.

    The suspect phase issues no destroy of its own, so it had no is_destroying
    guard — unlike both destroy-issuing sweeps. A session mid-teardown for an
    unrelated reason (slow consumer destroy, or the deadline sweep) that is also
    past the TTL would be logged "NOT destroying yet (quarantine)" — false — and
    would inflate the suspect counter and the gauge operators watch.
    """
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    metrics = MetricsRegistry()
    coord._metrics = metrics
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = _time.time() - 300.0   # suspect-only

    gate = asyncio.Event()
    original = transport.destroy_container

    async def _blocked_destroy(**kwargs: Any) -> Any:
        await gate.wait()
        return await original(**kwargs)

    transport.destroy_container = _blocked_destroy  # type: ignore[method-assign]
    destroying = asyncio.create_task(
        coord.destroy(rollout_id=session.rollout_id, container_id=session.container_id),
    )
    await asyncio.sleep(0)                      # let the destroy reach the wire call
    assert coord.is_destroying(session.rollout_id)

    reconciler = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord, metrics=metrics,
    )
    report = await reconciler.reconcile_once()

    assert report["__liveness__"]["suspect"] == 0
    assert coord.suspect_count() == 0
    assert _metric_value(metrics, "xrlenv_raw_liveness_suspect_total") == 0.0

    gate.set()
    await destroying


@pytest.mark.asyncio
async def test_node_lost_session_raises_typed_node_lost_not_stale_handle() -> None:
    """A node-lost session is a platform teardown, not a stale handle.

    handle_node_lost seals `failed` (nothing was destroyed — the node is gone),
    so the `reaped` gate missed it and every later RPC got the generic
    "Acquire first." — the exact misreporting SessionReaped exists to end, just
    in a different status column.
    """
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import NodeLost

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _quarantine_coord([transport])
    coord._state = InMemoryStateStore()
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    await coord.handle_node_lost("node-A", transport=transport)

    with pytest.raises(NodeLost) as excinfo:
        coord._require_session(session.rollout_id)
    assert "node was lost" in str(excinfo.value)
    assert "Acquire first" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_control_plane_ghost_raises_typed_control_plane_lost() -> None:
    """A row the CP lost track of is also the platform's doing, not the caller's."""
    from xrlenv.control.raw_container_service import RAW_LOST_CP_MARKERS
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import ControlPlaneLost

    coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
    state = InMemoryStateStore()
    coord._state = state
    session = await coord.acquire(image="busybox:1")
    rid = session.rollout_id
    coord._sessions.pop(rid, None)                     # CP lost its in-memory session
    state.update_raw_rollout(
        rid, status="failed",
        error=f"raw-gc-reconciler: {RAW_LOST_CP_MARKERS[0]} (no in-memory session)",
        finished_at=_time.time(),
    )

    with pytest.raises(ControlPlaneLost, match="lost track"):
        coord._require_session(rid)


@pytest.mark.asyncio
async def test_genuine_acquire_failure_still_reads_as_a_stale_handle() -> None:
    """`failed` is shared with real acquire/workload failures.

    Those are NOT platform teardowns and must not be dressed up as retryable
    platform events — matching on the status alone would do exactly that.
    """
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import ControlPlaneLost, NodeLost, XRLEnvError

    coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
    state = InMemoryStateStore()
    coord._state = state
    session = await coord.acquire(image="busybox:1")
    rid = session.rollout_id
    coord._sessions.pop(rid, None)
    state.update_raw_rollout(
        rid, status="failed",
        error="acquire_compose_project failed: image pull denied",
        finished_at=_time.time(),
    )

    with pytest.raises(XRLEnvError) as excinfo:
        coord._require_session(rid)
    assert not isinstance(excinfo.value, (NodeLost, ControlPlaneLost))
    assert "Acquire first" in str(excinfo.value)


@pytest.mark.asyncio
async def test_acquire_failure_quoting_a_marker_like_image_is_not_a_platform_teardown() -> None:
    """User-supplied text must not be able to forge a platform teardown.

    A genuine acquire failure stringifies the wire error verbatim, and dockerd
    echoes the caller's image ref back inside it. Hyphens are legal in refs, so an
    unanchored search for "lost-mid-run" let an ordinary bad-image pull masquerade
    as ControlPlaneLost — telling the caller "the platform tore this down, retry"
    about a failure that will recur identically forever. The marker is only
    trustworthy anchored to the control-plane-authored prefix at position 0.
    """
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import ControlPlaneLost, NodeLost, XRLEnvError

    coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
    state = InMemoryStateStore()
    coord._state = state
    session = await coord.acquire(image="busybox:1")
    rid = session.rollout_id
    coord._sessions.pop(rid, None)
    state.update_raw_rollout(
        rid, status="failed",
        error=(
            "acquire_compose_project failed: ImagePullError: pull access denied "
            "for myrepo/lost-mid-run-fix:latest, repository does not exist"
        ),
        finished_at=_time.time(),
    )

    with pytest.raises(XRLEnvError) as excinfo:
        coord._require_session(rid)
    assert not isinstance(excinfo.value, (ControlPlaneLost, NodeLost))
    assert "Acquire first" in str(excinfo.value)


@pytest.mark.asyncio
async def test_control_plane_ghost_still_recognised_when_properly_anchored() -> None:
    """The anchoring must not break the real case it exists to detect."""
    from xrlenv.control.raw_container_service import (
        RAW_LOST_CP_MARKERS,
        RAW_LOST_CP_PREFIX,
    )
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import ControlPlaneLost

    coord = _quarantine_coord([_FakeNodeTransport(node_id="node-A")])
    state = InMemoryStateStore()
    coord._state = state
    for marker in RAW_LOST_CP_MARKERS:
        session = await coord.acquire(image="busybox:1")
        rid = session.rollout_id
        coord._sessions.pop(rid, None)
        state.update_raw_rollout(
            rid, status="failed",
            error=f"{RAW_LOST_CP_PREFIX} {marker} (no in-memory session; row age 700s)",
            finished_at=_time.time(),
        )
        with pytest.raises(ControlPlaneLost, match="lost track"):
            coord._require_session(rid)


@pytest.mark.asyncio
async def test_control_plane_ghost_end_to_end_through_real_reconcile_sqlite() -> None:
    """Full pipeline, not a hand-written row: the real ``_reconcile_sqlite``
    ghost sweep (the SAME method the startup sweep and the periodic sweep call
    in production, at ``RAW_LOST_CP_MARKERS[0]``/``[1]``) writes the ``failed``
    row, and the real ``_require_session`` reads it back.

    The other CP-marker tests hand-construct the ``error`` string to check the
    reader's anchoring in isolation; this closes the gap that the WRITER's
    actual f-string (``f"{RAW_LOST_CP_PREFIX} {reason} (no in-memory session; "
    f"row age {age_s:.0f}s, prior status {status!r})"``, raw_gc_reconciler.py)
    produces a string the reader still recognises — so a rename/reword of
    either side that breaks the shared-constant contract fails HERE, not just
    in a test that already assumes the contract holds.
    """
    from xrlenv.control.raw_container_service import RAW_LOST_CP_MARKERS
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.errors import ControlPlaneLost

    for marker in RAW_LOST_CP_MARKERS:
        transport = _FakeNodeTransport(node_id="node-A")
        coord = _make_coord([transport])
        state = InMemoryStateStore()
        coord._state = state
        session = await coord.acquire(image="busybox:1")
        rid = session.rollout_id
        # The CP "lost its in-memory session" — the exact precondition
        # _reconcile_sqlite diffs against (row present, no live session/
        # acquiring-id for it).
        coord._sessions.pop(rid, None)

        recon = RawGCReconciler(
            registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
            coordinator=coord,
            state=state,  # type: ignore[arg-type]
            running_stale_s=0.0,  # the freshly-acquired row is "stale" immediately
            # readopt_grace_s left at its default — irrelevant here since
            # _started_at defaults to 0.0 until .start() runs, so the grace
            # window (now - 0.0) is already long expired, exactly as the
            # real startup sweep behaves before the reconciler task starts.
        )

        flipped = recon._reconcile_sqlite(reason=marker)
        assert flipped == 1

        row = state.get_raw_rollout(rid)
        assert row is not None
        assert row.status == "failed"
        assert row.error is not None and row.error.startswith(
            f"raw-gc-reconciler: {marker}",
        )

        with pytest.raises(ControlPlaneLost, match="lost track"):
            coord._require_session(rid)


# ──────────────────────────────────────────────────────────────────────────────
# The incident, as an executable statement of the contract.
#
# This is the A/B probe that reproduced the 2026-08-19 cn failure against the
# live cluster, reduced to a unit test. It is deliberately the ONLY test here
# that blocks the event loop for real rather than backdating `_last_seen_at`:
# the incident was not a session that looked stale in a dict, it was a consumer
# whose whole process was frozen and therefore could not beat. Everything else
# in this file asserts the state machine; this asserts the thing the state
# machine exists for.
# ──────────────────────────────────────────────────────────────────────────────


class _DirectHeartbeatTransport:
    """Minimal client transport wired straight into a coordinator.

    Real enough for the probe: the SDK keepalive's beats land on the same
    `mark_heartbeat` the wire would reach, so a blocked consumer loop stops the
    control plane hearing from it exactly as it did in production.
    """

    def __init__(self, coord: RawContainerCoordinator) -> None:
        self._coord = coord

    async def heartbeat_many(self, rollout_ids: list[str]) -> None:
        self._coord.mark_heartbeat(rollout_ids)


async def _probe(*, block_the_loop: bool) -> tuple[int, int]:
    """Acquire, go quiet for longer than the horizon — one way or the other."""
    import time as _t

    from xrlenv.client.client import _RawSessionKeepalive

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    coord._liveness_ttl_s = 0.05
    coord._liveness_quarantine_s = 0.30
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    keepalive = _RawSessionKeepalive(
        _DirectHeartbeatTransport(coord), interval_s=0.02,  # type: ignore[arg-type]
    )
    keepalive.register(session.rollout_id)
    await asyncio.sleep(0.05)                    # let the opt-in beat land

    if block_the_loop:
        _t.sleep(0.5)                            # the incident: process frozen
    else:
        await asyncio.sleep(0.5)                 # the control: loop free to beat

    report = await _liveness_reconciler(coord, transport).reconcile_once()
    await keepalive.close()
    return report["__liveness__"]["reaped"], len(coord.list_sessions())


@pytest.mark.asyncio
async def test_incident_probe_blocked_consumer_loses_its_session() -> None:
    """Above the horizon with a frozen loop: reaped, as in production.

    On cn this exact shape died at 166 s with the message the harness reported;
    243 sessions went the same way in one run.
    """
    reaped, alive = await _probe(block_the_loop=True)
    assert reaped == 1
    assert alive == 0


@pytest.mark.asyncio
async def test_incident_probe_healthy_consumer_keeps_its_session() -> None:
    """Same elapsed time, loop free to beat: survives.

    The control half of the A/B. Without it the test above would also pass if the
    reaper simply destroyed everything, which is precisely the pre-quarantine
    behaviour being replaced.
    """
    reaped, alive = await _probe(block_the_loop=False)
    assert reaped == 0
    assert alive == 1


@pytest.mark.asyncio
async def test_incident_probe_briefly_blocked_consumer_survives() -> None:
    """The case the quarantine exists for, and the only one that proves it.

    A consumer frozen for longer than the TTL but less than the horizon is
    exactly the 2026-08-19 population: alive, stalled, about to come back. The
    other two probes here pass with or without the quarantine — blocked past both
    thresholds is reaped either way, and a beating consumer is stale under
    neither — so this is the one that fails if the reaper ever returns to
    destroying at the TTL.

    Asserted through the coordinator's own queries rather than a sweep: once the
    loop unblocks, the keepalive resumes and refreshes the clock, so an async
    sweep would race the very recovery being measured.
    """
    import time as _t

    from xrlenv.client.client import _RawSessionKeepalive

    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    coord._liveness_ttl_s = 0.05
    coord._liveness_quarantine_s = 0.50
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]

    keepalive = _RawSessionKeepalive(
        _DirectHeartbeatTransport(coord), interval_s=0.02,  # type: ignore[arg-type]
    )
    keepalive.register(session.rollout_id)
    await asyncio.sleep(0.05)

    _t.sleep(0.20)      # frozen: past the TTL, well inside the horizon

    assert coord.liveness_suspect_candidates(), "should be flagged as suspect"
    assert not coord.liveness_reap_candidates(), (
        "a consumer inside the quarantine horizon must NOT be a reap candidate — "
        "this is destroy-at-TTL, the behaviour that cost 243 sessions"
    )

    # And it recovers: the next beat clears suspicion, session intact.
    await asyncio.sleep(0.10)
    assert coord.suspect_count() == 0
    assert len(coord.list_sessions()) == 1
    await keepalive.close()
