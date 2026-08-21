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
import time as _time
from dataclasses import dataclass, field
from dataclasses import dataclass as _dc
from dataclasses import field as _field
from typing import Any
from typing import Any as _Any

import pytest
from xrlenv.control.raw_container_service import (
    RawContainerCoordinator,
)
from xrlenv.control.raw_gc_reconciler import RawGCReconciler

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
    assert report["__liveness__"] == {"reaped": 0}
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_liveness_reaped_when_heartbeated_and_stale() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])
    coord._last_seen_at[session.rollout_id] = 0.0  # consumer went silent
    report = await _liveness_reconciler(coord, transport).reconcile_once()
    assert report["__liveness__"] == {"reaped": 1}
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
    assert report["__liveness__"] == {"reaped": 0}
    assert len(coord.list_sessions()) == 1


@pytest.mark.asyncio
async def test_liveness_not_reaped_when_fresh() -> None:
    transport = _FakeNodeTransport(node_id="node-A")
    coord = _make_coord([transport])
    session = await coord.acquire(image="busybox:1")
    transport.docker_container_ids = [session.container_id]
    coord.mark_heartbeat([session.rollout_id])  # fresh beat (now)
    report = await _liveness_reconciler(coord, transport).reconcile_once()
    assert report["__liveness__"] == {"reaped": 0}
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
