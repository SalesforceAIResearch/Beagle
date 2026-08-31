"""Tests for the in-memory StateStore (spec 20)."""

from __future__ import annotations

import pytest
from xrlenv.control.state import (
    InMemoryStateStore,
    RolloutRecord,
    SandboxRecord,
)
from xrlenv.types import RolloutStatus


@pytest.fixture
def store() -> InMemoryStateStore:
    return InMemoryStateStore()


def _rollout(rollout_id: str = "r1") -> RolloutRecord:
    return RolloutRecord(
        rollout_id=rollout_id,
        template="hello-shell",
        status=RolloutStatus.STARTING,
    )


def _sandbox(sandbox_id: str = "sb1") -> SandboxRecord:
    return SandboxRecord(
        sandbox_id=sandbox_id,
        backend="docker",
        backend_ref="docker-cid",
        stub_endpoint="unix:///tmp/stub.sock",
        template="hello-shell",
        node_id="local-laptop",
    )


def test_insert_get_update_rollout(store: InMemoryStateStore) -> None:
    store.insert_rollout(_rollout())
    assert store.get_rollout("r1").status == RolloutStatus.STARTING

    store.update_rollout("r1", status=RolloutStatus.RUNNING)
    assert store.get_rollout("r1").status == RolloutStatus.RUNNING


def test_duplicate_rollout_is_rejected(store: InMemoryStateStore) -> None:
    store.insert_rollout(_rollout())
    with pytest.raises(KeyError):
        store.insert_rollout(_rollout())


def test_unknown_rollout_raises(store: InMemoryStateStore) -> None:
    with pytest.raises(KeyError):
        store.get_rollout("missing")


def test_sandbox_lifecycle(store: InMemoryStateStore) -> None:
    store.insert_sandbox(_sandbox())
    assert store.get_sandbox("sb1").status == "running"

    store.update_sandbox("sb1", status="destroying")
    assert store.get_sandbox("sb1").status == "destroying"

    store.remove_sandbox("sb1")
    with pytest.raises(KeyError):
        store.get_sandbox("sb1")


def test_idempotency(store: InMemoryStateStore) -> None:
    store.record_idempotent("req-1", "rollout-A")
    assert store.lookup_idempotent("req-1") == "rollout-A"
    assert store.lookup_idempotent("req-2") is None


def test_event_seq_monotonic(store: InMemoryStateStore) -> None:
    e1 = store.append_event("kind1", rollout_id="r1")
    e2 = store.append_event("kind2", rollout_id="r2")
    assert e2.seq > e1.seq
    assert [e.seq for e in store.events_since(0)] == [e1.seq, e2.seq]
    assert [e.seq for e in store.events_since(e1.seq)] == [e2.seq]


def test_find_registered_preferred_home_returns_node_for_deferred_row(
    store: InMemoryStateStore,
) -> None:
    """Audit P1.6.g-H2 (2026-05-05): the lookup must return the
    most-recent applied plan's preferred_home for a row still in
    ``status="registered"`` matching the image_ref."""
    from xrlenv.control.state import BuildAssignmentRecord

    store.record_build_plan(
        plan_id="p1", applied_by="cli", plan_json="{}",
    )
    store.record_assignment(BuildAssignmentRecord(
        plan_id="p1", node_id="n2", image_ref="bench/x:1",
        benchmark="bench", status="registered",
    ))
    # A 'done' row for a different image must not pollute the lookup.
    store.record_assignment(BuildAssignmentRecord(
        plan_id="p1", node_id="n1", image_ref="bench/y:1",
        benchmark="bench", status="done",
    ))

    assert store.find_registered_preferred_home("bench/x:1") == "n2"
    # Image without any registered row → None.
    assert store.find_registered_preferred_home("bench/y:1") is None
    # Unknown image → None (no false positives).
    assert store.find_registered_preferred_home("never-built:1") is None


def test_find_registered_preferred_home_picks_latest_plan(
    store: InMemoryStateStore,
) -> None:
    """When two plans both have a registered row for the same image,
    the lookup returns the more-recently-applied plan's preferred_home
    (operators re-apply plans to update placement)."""
    import time as _time

    from xrlenv.control.state import BuildAssignmentRecord

    store.record_build_plan(
        plan_id="p_old", applied_by="cli", plan_json="{}",
    )
    _time.sleep(0.01)  # ensure applied_at differs
    store.record_build_plan(
        plan_id="p_new", applied_by="cli", plan_json="{}",
    )
    store.record_assignment(BuildAssignmentRecord(
        plan_id="p_old", node_id="old-home", image_ref="bench/x:1",
        benchmark="bench", status="registered",
    ))
    store.record_assignment(BuildAssignmentRecord(
        plan_id="p_new", node_id="new-home", image_ref="bench/x:1",
        benchmark="bench", status="registered",
    ))

    assert store.find_registered_preferred_home("bench/x:1") == "new-home"


def test_find_registered_preferred_home_skips_non_registered_rows(
    store: InMemoryStateStore,
) -> None:
    """A row that's already ``done`` or ``failed`` shouldn't return —
    the image is materialized (or known-failed) and the scheduler's
    image-affinity / failure handling takes over from there."""
    from xrlenv.control.state import BuildAssignmentRecord

    store.record_build_plan(
        plan_id="p1", applied_by="cli", plan_json="{}",
    )
    store.record_assignment(BuildAssignmentRecord(
        plan_id="p1", node_id="n1", image_ref="bench/x:1",
        benchmark="bench", status="done",
    ))
    assert store.find_registered_preferred_home("bench/x:1") is None


def test_seal_trajectory_status_round_trip(store: InMemoryStateStore) -> None:
    store.insert_rollout(_rollout())
    store.update_rollout(
        "r1",
        status=RolloutStatus.FINISHED,
        reason=None,
        final_reward=1.5,
    )
    traj = store.seal_trajectory("r1")
    assert traj.status == RolloutStatus.FINISHED
    assert traj.final_reward == 1.5
    assert traj.template == "hello-shell"
