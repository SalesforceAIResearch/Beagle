"""D-AR-2026-07-07-B — control-plane re-admit on node-saturation create fault.

The node-local create-cap+retry (``raw_container.py``) absorbs a *transient*
create burst on ONE node. It cannot help when a single node is *sustainedly*
overloaded (sysbox-fs FUSE wedges → every create returns ``pre-register with
sysbox-fs: DeadlineExceeded``). The right recovery is to steer to a *sibling*
node. The AdmissionQueue only re-queues on a *proactive* ``CapacityExhausted``
(the scheduler declined to place); it never sees the *reactive* create-time
saturation 5xx that surfaces from ``node.acquire_container`` AFTER placement.
``RawContainerCoordinator.acquire`` closes that gap with a re-admit loop.

These tests exercise (numbering follows ``notes/design-cp-readmit-saturation.md``
"Tests (required)"):

1. A→B rebalance — saturation on A → re-admit lands on B, A excluded.
2. Both-fail relax — A then B fail → exclusion relaxes to ∅ (no hard-fail/spin).
3. Single-node wait-or-bound — stops at ``_CP_REQUEUE_MAX`` WITHOUT leaking a
   ``_pending`` reservation (placement released every attempt).
4. Terminal errors propagate — 4xx / name-conflict / image-not-found /
   capability → unretried (classifier returns False).
5. Budgets are total — the total wall-clock cap gives up early and clamps each
   attempt's node-wire ``acquire_timeout_s`` to the remainder.
6. Fleet-opener retry — a re-admitted opener doesn't leak ``_fleet_opening`` /
   ``_pending``; final failure clears both.
7. Scheduler exclusion unit — ``place(exclude_node_ids=…)`` drops the node;
   all-capable-excluded → ``CapacityExhausted``; empty-before-exclude →
   ``BackendCapabilityMissing``; ``capable_node_ids`` returns the right set.

Test 8 ("ambiguous timeout / late orphan raw-GC") splits: the ambiguous-create
timeout is a classifier case (covered in ``test_classifier_*``), and the
node-only raw-GC by ``container_id`` that reaps such an orphan is already
covered by ``test_raw_gc_reconciler.py::test_node_only_orphan_force_destroyed``.
"""

from __future__ import annotations

import pytest
from xrlenv.compat.metadata import (
    LABEL_FLEET_CPU_REQUEST,
    LABEL_FLEET_ID,
    LABEL_FLEET_MEM_REQUEST,
)
from xrlenv.control import raw_container_service as rcs
from xrlenv.control.raw_container_service import (
    RawContainerCoordinator,
    RawContainerSession,
    _is_retriable_acquire_saturation,
)
from xrlenv.control.scheduler import Scheduler
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import TemplateCatalog
from xrlenv.errors import (
    AuthDenied,
    BackendCapabilityMissing,
    CapacityExhausted,
    ImageMissingOnNode,
    ImagePullFailed,
    MountDenied,
    NodeCommandTimeout,
    PinCapacityExhausted,
    XRLEnvError,
)

from tests.unit.control.test_raw_container_coordinator import _FakeNodeTransport
from tests.unit.control.test_scheduler import _manifest, _node

_GIB = 1024**3

# A realistic reconstructed create-saturation error (what the CP sees on the
# wire: ``node <id>: remote command <kind>: <node-side translated message>``).
_SYSBOX_SATURATION = XRLEnvError(
    "node node-A: remote command APIError: node-side docker create failed for "
    "busybox:1: APIError: 500 Server Error: pre-register with sysbox-fs: "
    "context DeadlineExceeded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Harness — real Scheduler (has capable_node_ids + exclude_node_ids) + fake nodes
# ──────────────────────────────────────────────────────────────────────────────


def _make_cluster(
    node_ids: list[str],
    *,
    fail_nodes: set[str] | None = None,
    fail_exc: Exception | None = None,
) -> tuple[RawContainerCoordinator, Scheduler, dict[str, _FakeNodeTransport]]:
    """A coordinator wired to a REAL Scheduler over ``node_ids``. Nodes in
    ``fail_nodes`` raise ``fail_exc`` from every ``acquire_container``."""
    nodes = [_FakeNodeTransport(node_id=nid) for nid in node_ids]
    for n in nodes:
        if fail_nodes and n.node_id in fail_nodes:
            n.raise_on_acquire = fail_exc
    state = InMemoryStateStore()
    scheduler = Scheduler(nodes, catalog=TemplateCatalog(), state=state)
    coord = RawContainerCoordinator(scheduler=scheduler, state=state)
    scheduler.set_raw_session_provider(coord.iter_load_entries)
    return coord, scheduler, {n.node_id: n for n in nodes}


def _spy_place_excludes(scheduler: Scheduler) -> list[frozenset[str] | None]:
    """Record the ``exclude_node_ids`` passed to each ``place()`` call."""
    excludes: list[frozenset[str] | None] = []
    orig = scheduler.place

    def spy(*args: object, **kwargs: object) -> object:
        excludes.append(kwargs.get("exclude_node_ids"))  # type: ignore[arg-type]
        return orig(*args, **kwargs)  # type: ignore[arg-type]

    scheduler.place = spy  # type: ignore[method-assign,assignment]
    return excludes


def _opener_labels(fleet_id: str, cpu: float, mem_gb: int) -> dict[str, str]:
    return {
        LABEL_FLEET_ID: fleet_id,
        LABEL_FLEET_CPU_REQUEST: str(cpu),
        LABEL_FLEET_MEM_REQUEST: str(mem_gb * _GIB),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Classifier — the core correctness gate (covers test 4 + test 8 ambiguous case)
# ──────────────────────────────────────────────────────────────────────────────


def test_classifier_retriable_saturation() -> None:
    # Wire-level timeout — the canonical node-overload / wedged-agent signal.
    assert _is_retriable_acquire_saturation(
        NodeCommandTimeout("node node-A: command c1 timed out after 60.0s")
    )
    # The motivating case: sysbox-fs pre-register DeadlineExceeded (a 5xx).
    assert _is_retriable_acquire_saturation(_SYSBOX_SATURATION)
    # Other saturation shapes: plain docker 5xx, bare deadline, generic timeout.
    for msg in (
        "node node-A: remote command APIError: 503 Server Error: Service "
        "Unavailable",
        "node node-A: remote command X: context deadline exceeded",
        "node node-A: remote command X: read timed out",
    ):
        assert _is_retriable_acquire_saturation(XRLEnvError(msg)), msg


def test_classifier_terminal_errors_not_retried() -> None:
    # Specific terminal subclasses — reproduce identically on any node.
    for exc in (
        BackendCapabilityMissing("no node supports runtime 'sysbox-runc'"),
        ImageMissingOnNode("image not present and ensure=False"),
        ImagePullFailed("pull failed: manifest unknown"),
        MountDenied("host path not allowlisted"),
        AuthDenied("registry auth failed"),
        # CapacityExhausted comes from the PLACE step (queue already waited /
        # pool full); re-admit can't help. Its message even contains "timeout"
        # — the terminal-subclass check must win over the message match.
        CapacityExhausted("queue_timeout_s=5.0 expired waiting for capacity"),
    ):
        assert not _is_retriable_acquire_saturation(exc), type(exc).__name__
    # 4xx create faults reconstructed as a bare XRLEnvError — terminal.
    for msg in (
        "node node-A: remote command APIError: 409 Client Error: Conflict "
        '("container name "/x" is already in use")',
        "node node-A: remote command APIError: 404 Client Error: No such "
        "image: nope:latest",
    ):
        assert not _is_retriable_acquire_saturation(XRLEnvError(msg)), msg
    # A non-XRLEnvError (programming error) never re-admits.
    assert not _is_retriable_acquire_saturation(ValueError("boom"))


def test_classifier_readmits_node_pin_or_fail() -> None:
    """P6 step-4c follow-up — node-side REQUIRED pin-or-fail
    (``PinCapacityExhausted``) is node-specific (a stale-heartbeat / ledger
    race), so it IS retriable on a sibling capable node — even though it
    subclasses the terminal ``CapacityExhausted``. Both transport shapes match."""
    # In-process: the real exception instance reaches the classifier.
    assert _is_retriable_acquire_saturation(
        PinCapacityExhausted(
            "cpu_isolation=required but the node core ledger is exhausted "
            "(0/2 free, wanted 2) — refusing to degrade a REQUIRED pin to CFS "
            "quota (P6 §8.7 pin-or-fail)"
        )
    )
    # Over the gRPC wire: reconstructed as a bare XRLEnvError carrying the node's
    # error_kind (the class name "PinCapacityExhausted") in its message.
    assert _is_retriable_acquire_saturation(
        XRLEnvError(
            "node node-A: remote command AcquireContainerCommand: "
            "PinCapacityExhausted: cpu_isolation=required but the node core "
            "ledger is exhausted (0/2 free, wanted 2) — refusing to degrade a "
            "REQUIRED pin to CFS quota (P6 §8.7 pin-or-fail)"
        )
    )
    # But a PLAIN CapacityExhausted (from the place/queue step) stays terminal —
    # the pin-or-fail check must not widen it.
    assert not _is_retriable_acquire_saturation(
        CapacityExhausted("queue_timeout_s=5.0 expired waiting for capacity")
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. A→B rebalance
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_readmit_rebalances_from_saturated_node_to_sibling() -> None:
    coord, scheduler, nodes = _make_cluster(
        ["node-A", "node-B"], fail_nodes={"node-A"}, fail_exc=_SYSBOX_SATURATION,
    )
    excludes = _spy_place_excludes(scheduler)

    session = await coord.acquire(image="busybox:1", command=["sleep", "inf"])

    # Landed on the sibling.
    assert isinstance(session, RawContainerSession)
    assert session.node_id == "node-B"
    # A raised before recording a call; B took exactly one create.
    assert nodes["node-A"].acquire_calls == []
    assert len(nodes["node-B"].acquire_calls) == 1
    # First place had no exclusion; the re-admit excluded the failed node A.
    assert excludes[0] is None or excludes[0] == frozenset()
    assert frozenset({"node-A"}) in excludes
    # No leaked reservation — A's placement was released before the retry.
    assert len(scheduler._pending) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Both-fail relax
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_both_nodes_fail_relaxes_exclusion_and_bounds() -> None:
    coord, scheduler, _nodes = _make_cluster(
        ["node-A", "node-B"],
        fail_nodes={"node-A", "node-B"},
        fail_exc=_SYSBOX_SATURATION,
    )
    excludes = _spy_place_excludes(scheduler)

    # It must NOT hard-fail with an "all excluded" CapacityExhausted, and must
    # NOT spin forever — it relaxes and bounds, surfacing the saturation error.
    with pytest.raises(XRLEnvError) as ei:
        await coord.acquire(image="busybox:1", command=["sleep", "inf"])
    assert "sysbox-fs" in str(ei.value)
    # Node A was excluded on the first re-admit...
    assert any(e == frozenset({"node-A"}) for e in excludes)
    # ...and once the failed set covered the WHOLE capable pool the exclusion
    # relaxed — a later place ran with NO exclusion (an empty exclusion is
    # omitted, so the spy sees None/∅) rather than place() raising the
    # "all excluded" CapacityExhausted. That proves the relax, not a hard-fail.
    first_excluded = next(i for i, e in enumerate(excludes) if e)
    assert any(not e for e in excludes[first_excluded + 1:])
    # Bounded: 1 initial + _CP_REQUEUE_MAX re-admits.
    assert len(excludes) == rcs._CP_REQUEUE_MAX + 1
    # Every abandoned placement was released — no leak.
    assert len(scheduler._pending) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Single-node wait-or-bound (no _pending leak)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_node_bounds_without_leaking_pending() -> None:
    coord, scheduler, _nodes = _make_cluster(
        ["node-A"], fail_nodes={"node-A"}, fail_exc=_SYSBOX_SATURATION,
    )
    excludes = _spy_place_excludes(scheduler)

    with pytest.raises(XRLEnvError) as ei:
        await coord.acquire(image="busybox:1", command=["sleep", "inf"])
    assert "sysbox-fs" in str(ei.value)
    # Single capable node → the exclusion always relaxes to ∅ (nowhere else to
    # go), so every attempt re-tries the same node; bounded by _CP_REQUEUE_MAX.
    assert len(excludes) == rcs._CP_REQUEUE_MAX + 1
    assert all(e in (None, frozenset()) for e in excludes)
    # The crucial invariant: no scheduler reservation leaked across the retries.
    assert len(scheduler._pending) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Terminal errors propagate unretried
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_exc",
    [
        XRLEnvError(
            "node node-A: remote command APIError: 409 Client Error: Conflict "
            '("container name is already in use")',
        ),
        ImageMissingOnNode("image not present and ensure_image_present=False"),
        BackendCapabilityMissing("no node supports runtime 'sysbox-runc'"),
    ],
)
async def test_terminal_error_propagates_without_readmit(
    terminal_exc: Exception,
) -> None:
    coord, scheduler, nodes = _make_cluster(
        ["node-A", "node-B"], fail_nodes={"node-A"}, fail_exc=terminal_exc,
    )
    excludes = _spy_place_excludes(scheduler)

    with pytest.raises(type(terminal_exc)):
        await coord.acquire(image="busybox:1", command=["sleep", "inf"])
    # Exactly one placement attempt — no re-admit on a terminal fault.
    assert len(excludes) == 1
    # The outer handler released the one placement; no leak, no B fallback.
    assert len(scheduler._pending) == 0
    assert nodes["node-B"].acquire_calls == []


# ──────────────────────────────────────────────────────────────────────────────
# 5. Budgets are total — wall-clock cap gives up early + clamps node-wire timeout
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_total_cap_gives_up_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A zero cap means: after the FIRST saturation failure arms the deadline
    # (now + 0), the next loop iteration sees remaining_cap <= 0 and gives up —
    # re-raising the last saturation error rather than exhausting _CP_REQUEUE_MAX.
    monkeypatch.setattr(rcs, "_CP_REQUEUE_TOTAL_CAP_S", 0.0)
    coord, scheduler, _nodes = _make_cluster(
        ["node-A", "node-B"], fail_nodes={"node-A", "node-B"},
        fail_exc=_SYSBOX_SATURATION,
    )
    excludes = _spy_place_excludes(scheduler)

    with pytest.raises(XRLEnvError) as ei:
        await coord.acquire(image="busybox:1", command=["sleep", "inf"])
    assert "sysbox-fs" in str(ei.value)
    # Only the first attempt ran; the cap short-circuited the re-admit.
    assert len(excludes) == 1
    assert len(scheduler._pending) == 0


@pytest.mark.asyncio
async def test_total_cap_bounds_caller_but_floors_committed_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The re-admit cap bounds the caller's huge acquire_timeout_s — but must NOT
    # starve the COMMITTED create below the floor. Cap 100s < floor 180s: the
    # create on B is floored at 180 (not the caller's 10_000s, not the leftover
    # ~100s cap). The starving-to-leftover-cap behaviour was the conc-32 bug
    # (tw_650591 / tw_709166 timed out at ~30s mid-sysbox-create).
    monkeypatch.setattr(rcs, "_CP_REQUEUE_TOTAL_CAP_S", 100.0)
    coord, _scheduler, nodes = _make_cluster(
        ["node-A", "node-B"], fail_nodes={"node-A"}, fail_exc=_SYSBOX_SATURATION,
    )

    session = await coord.acquire(
        image="busybox:1", command=["sleep", "inf"],
        acquire_timeout_s=10_000.0,  # huge — must be bounded, but create not starved
    )
    assert session.node_id == "node-B"
    b_timeout = nodes["node-B"].acquire_calls[0]["acquire_timeout_s"]
    assert b_timeout is not None
    # Floored at _MIN_CREATE_DEADLINE_S — bounded well below the caller's 10_000s,
    # but NOT starved to the ~100s leftover cap.
    assert b_timeout == pytest.approx(rcs._MIN_CREATE_DEADLINE_S)


@pytest.mark.asyncio
async def test_floor_never_raises_an_explicit_tight_create_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The floor undoes the re-admit budget's OVER-reduction; it must never raise
    # the create deadline ABOVE a caller's explicit (tight) acquire_timeout_s. Cap
    # 100s, floor 180s, but the caller asked for 60s → the create gets 60s.
    monkeypatch.setattr(rcs, "_CP_REQUEUE_TOTAL_CAP_S", 100.0)
    coord, _scheduler, nodes = _make_cluster(
        ["node-A", "node-B"], fail_nodes={"node-A"}, fail_exc=_SYSBOX_SATURATION,
    )

    session = await coord.acquire(
        image="busybox:1", command=["sleep", "inf"],
        acquire_timeout_s=60.0,  # deliberately tight — honoured, not raised to 180
    )
    assert session.node_id == "node-B"
    b_timeout = nodes["node-B"].acquire_calls[0]["acquire_timeout_s"]
    assert b_timeout is not None
    assert b_timeout == pytest.approx(60.0)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Fleet-opener retry
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fleet_opener_readmits_to_sibling_no_leak() -> None:
    coord, scheduler, _nodes = _make_cluster(
        ["node-A", "node-B"], fail_nodes={"node-A"}, fail_exc=_SYSBOX_SATURATION,
    )

    lead = await coord.acquire(
        image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
    )

    # The opener re-admitted onto B and opened the fleet there.
    assert lead.fleet_id == "f1"
    assert "f1" in coord._fleets
    assert coord._fleets["f1"].node_id == "node-B"
    # The footprint (18) is reserved once on B, not the lead's own 2.
    assert coord._fleets["f1"].footprint.cpu_request == 18.0
    # No double-marked / stuck opening marker; no leaked footprint reservation.
    assert coord._fleet_opening == set()
    assert len(scheduler._pending) == 0


@pytest.mark.asyncio
async def test_fleet_opener_all_fail_clears_opening_marker() -> None:
    coord, scheduler, _nodes = _make_cluster(
        ["node-A", "node-B"],
        fail_nodes={"node-A", "node-B"},
        fail_exc=_SYSBOX_SATURATION,
    )

    with pytest.raises(XRLEnvError):
        await coord.acquire(
            image="busybox:1", cpu_limit=2, labels=_opener_labels("f1", 18, 32),
        )
    # Final failure rolled everything back: no reservation, no opening marker,
    # no leaked footprint pending.
    assert "f1" not in coord._fleets
    assert coord._fleet_opening == set()
    assert len(scheduler._pending) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 7. Scheduler exclusion unit
# ──────────────────────────────────────────────────────────────────────────────


def _sched(*nodes: object) -> Scheduler:
    return Scheduler(
        list(nodes), catalog=TemplateCatalog(), state=InMemoryStateStore(),
    )


def test_scheduler_place_drops_excluded_node() -> None:
    a, b = _node("node-A"), _node("node-B")
    sched = _sched(a, b)
    p = sched.place(_manifest(), backend="docker",
                    exclude_node_ids=frozenset({"node-A"}))
    assert p.node.node_id == "node-B"


def test_scheduler_all_capable_excluded_raises_capacity_exhausted() -> None:
    a, b = _node("node-A"), _node("node-B")
    sched = _sched(a, b)
    # Both capable nodes excluded → CapacityExhausted (they CAN serve it; they
    # were just all excluded this attempt), NOT BackendCapabilityMissing.
    with pytest.raises(CapacityExhausted):
        sched.place(_manifest(), backend="docker",
                    exclude_node_ids=frozenset({"node-A", "node-B"}))


def test_scheduler_empty_before_exclusion_raises_capability_missing() -> None:
    # No docker node at all → the capability check fires BEFORE the exclusion
    # gate, so it's a BackendCapabilityMissing regardless of exclude set.
    only_fc = _node("node-fc", backends=("function_call",))
    sched = _sched(only_fc)
    with pytest.raises(BackendCapabilityMissing):
        sched.place(_manifest(), backend="docker",
                    exclude_node_ids=frozenset({"node-A"}))


def test_scheduler_capable_node_ids() -> None:
    a = _node("node-A")
    b = _node("node-B")
    fc = _node("node-fc", backends=("function_call",))
    sched = _sched(a, b, fc)
    assert sched.capable_node_ids(backend="docker") == frozenset(
        {"node-A", "node-B"}
    )
    # A runtime nothing advertises (fakes default to ["runc"]) → empty set.
    assert sched.capable_node_ids(
        backend="docker", container_runtime="sysbox-runc",
    ) == frozenset()
