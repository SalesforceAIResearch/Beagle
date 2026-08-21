"""Multi-user (Slice C) — pure fair-share model + StateStore accessors.

Two layers, both no-network / no-port:

1. ``FairnessPolicy.cap_for`` — the per-owner cap arithmetic
   (disabled / default cap / owner cap / uncapped / blocked).
2. ``SqliteStateStore`` fair-share accessors against a real on-disk db under
   ``tmp_path`` — round-trip + upsert/clear + ``running_counts_by_owner``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv.control.state import (
    FairnessOwnerOverride,
    FairnessPolicy,
    RawRolloutRecord,
    RolloutRecord,
    SqliteStateStore,
)
from xrlenv.types import RolloutStatus

# ── Pure model: FairnessPolicy.cap_for ─────────────────────────────────────────


def test_cap_for_disabled_returns_none() -> None:
    # capacity_basis None → fairness off → uncapped for any owner.
    pol = FairnessPolicy()
    assert pol.enabled is False
    assert pol.cap_for("alice", {"alice", "bob"}) is None


def test_cap_for_capacity_is_per_owner_default() -> None:
    pol = FairnessPolicy(capacity_basis=10)
    assert pol.cap_for("alice", {"alice", "bob"}) == 10
    assert pol.cap_for("bob", {"alice", "bob"}) == 10


def test_cap_for_blocked_owner_is_zero() -> None:
    pol = FairnessPolicy(
        capacity_basis=10,
        overrides={"alice": FairnessOwnerOverride(owner_id="alice", blocked=True)},
    )
    assert pol.cap_for("alice", {"alice", "bob"}) == 0


def test_cap_for_blocked_owner_does_not_affect_other_owner_cap() -> None:
    pol = FairnessPolicy(
        capacity_basis=10,
        overrides={"alice": FairnessOwnerOverride(owner_id="alice", blocked=True)},
    )
    assert pol.cap_for("bob", {"alice", "bob"}) == 10


def test_cap_for_uncapped_owner_returns_none() -> None:
    pol = FairnessPolicy(
        capacity_basis=10,
        overrides={"alice": FairnessOwnerOverride(owner_id="alice", uncapped=True)},
    )
    assert pol.cap_for("alice", {"alice", "bob"}) is None
    assert pol.cap_for("bob", {"alice", "bob"}) == 10


def test_cap_for_owner_cap_overrides_default_capacity() -> None:
    pol = FairnessPolicy(
        capacity_basis=10,
        overrides={
            "alice": FairnessOwnerOverride(
                owner_id="alice", hard_cap=256,
            ),
        },
    )
    assert pol.cap_for("alice", {"alice", "bob"}) == 256
    assert pol.cap_for("bob", {"alice", "bob"}) == 10


def test_cap_for_owner_cap_overrides_default_capacity_downward() -> None:
    # --owner alice --cap 5 with --default-cap 32 → alice's effective cap is 5,
    # not 32; override works in both directions.
    pol = FairnessPolicy(
        capacity_basis=32,
        overrides={
            "alice": FairnessOwnerOverride(owner_id="alice", hard_cap=5),
        },
    )
    assert pol.cap_for("alice", {"alice", "bob"}) == 5
    # bob has no override → still gets the default.
    assert pol.cap_for("bob", {"alice", "bob"}) == 32


def test_cap_for_blocked_owner_with_cap_and_uncap_still_returns_zero() -> None:
    # blocked=True takes priority over cap/uncapped; an operator cannot
    # accidentally re-admit a blocked owner by setting another override.
    pol = FairnessPolicy(
        capacity_basis=10,
        overrides={
            "alice": FairnessOwnerOverride(
                owner_id="alice", blocked=True, uncapped=True, hard_cap=100,
            ),
        },
    )
    assert pol.cap_for("alice", {"alice"}) == 0


def test_cap_for_disabled_with_blocked_override_still_returns_none() -> None:
    # When fairness is disabled (capacity_basis=None) the policy is completely
    # off; even a blocked owner override must return None (uncapped), not 0.
    pol = FairnessPolicy(
        overrides={
            "alice": FairnessOwnerOverride(owner_id="alice", blocked=True),
        },
    )
    assert pol.cap_for("alice", {"alice"}) is None


def test_cap_for_active_owner_set_does_not_change_default_cap() -> None:
    pol = FairnessPolicy(capacity_basis=10)
    assert pol.cap_for("carol", {"alice"}) == 10


# ── StateStore accessors (real on-disk SqliteStateStore) ───────────────────────


@pytest.fixture
def store(tmp_path: Path) -> SqliteStateStore:
    s = SqliteStateStore(tmp_path / "s.db")
    yield s
    s.close()


def test_default_policy_is_disabled(store: SqliteStateStore) -> None:
    pol = store.get_fairness_policy()
    assert pol.enabled is False
    assert pol.capacity_basis is None
    assert pol.floor == 1
    assert pol.overrides == {}


def test_set_fairness_global_round_trips(store: SqliteStateStore) -> None:
    store.set_fairness_global(capacity_basis=12, floor=3)
    pol = store.get_fairness_policy()
    assert pol.enabled is True
    assert pol.capacity_basis == 12
    assert pol.floor == 3


def test_set_fairness_global_none_disables(store: SqliteStateStore) -> None:
    store.set_fairness_global(capacity_basis=12, floor=3)
    store.set_fairness_global(capacity_basis=None)
    pol = store.get_fairness_policy()
    assert pol.enabled is False
    assert pol.capacity_basis is None
    # floor resets to the default (1) when not passed on the disabling call.
    assert pol.floor == 1


def test_set_fairness_owner_upserts(store: SqliteStateStore) -> None:
    store.set_fairness_owner("alice", hard_cap=5, blocked=False)
    store.set_fairness_owner("alice", hard_cap=None, uncapped=True, blocked=True)
    pol = store.get_fairness_policy()
    # Last write wins; exactly one row for alice.
    assert set(pol.overrides) == {"alice"}
    ov = pol.overrides["alice"]
    assert ov.hard_cap is None
    assert ov.uncapped is True
    assert ov.blocked is True


def test_clear_fairness_owner_removes_row(store: SqliteStateStore) -> None:
    store.set_fairness_owner("alice", hard_cap=2)
    store.clear_fairness_owner("alice")
    assert store.get_fairness_policy().overrides == {}


def test_clear_unknown_owner_is_noop(store: SqliteStateStore) -> None:
    # Clearing an owner that was never set must not raise.
    store.clear_fairness_owner("ghost")
    assert store.get_fairness_policy().overrides == {}


# ── running_counts_by_owner ────────────────────────────────────────────────────


def _gym(store: SqliteStateStore, rid: str, owner: str, status: RolloutStatus) -> None:
    store.insert_rollout(RolloutRecord(
        rollout_id=rid, template="hello-shell", status=status, owner_id=owner,
    ))


def _raw(store: SqliteStateStore, rid: str, owner: str, status: str) -> None:
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id=rid, status=status, image="busybox:1", owner_id=owner,
    ))


def test_running_counts_empty_store(store: SqliteStateStore) -> None:
    assert store.running_counts_by_owner() == {}


def test_running_counts_sums_active_across_both_tables(
    store: SqliteStateStore,
) -> None:
    # Gym side: alice has one running + one starting (both active, both
    # post-placement), bob has one finished (NOT counted).
    _gym(store, "g-a1", "alice", RolloutStatus.RUNNING)
    _gym(store, "g-a2", "alice", RolloutStatus.STARTING)
    _gym(store, "g-b1", "bob", RolloutStatus.FINISHED)
    # Raw side: alice one acquiring (PRE-admission, NOT counted — audit M3) +
    # one running (counted); bob one running (counted) + one released (NOT).
    _raw(store, "r-a1", "alice", "acquiring")
    _raw(store, "r-a2", "alice", "running")
    _raw(store, "r-b1", "bob", "running")
    _raw(store, "r-b2", "bob", "released")

    counts = store.running_counts_by_owner()
    # alice: 2 gym (running+starting) + 1 raw running = 3. The raw 'acquiring'
    # row is excluded so a raw candidate never counts against its own cap.
    assert counts["alice"] == 3
    # bob: 0 gym active + 1 raw running = 1 (finished gym + released raw excluded).
    assert counts["bob"] == 1


def test_running_counts_excludes_terminal_only_owner(
    store: SqliteStateStore,
) -> None:
    # An owner whose only rollouts are terminal does not appear at all.
    _gym(store, "g-c1", "carol", RolloutStatus.FINISHED)
    _raw(store, "r-c1", "carol", "released")
    assert "carol" not in store.running_counts_by_owner()
