"""Multi-user (Slice C) — AdmissionQueue per-owner fair-share gate.

``AdmissionQueue._owner_at_cap`` reads the live ``FairnessPolicy`` + cluster
running counts from the StateStore and answers whether an owner is at/over its
effective cap. It is off-by-default and fail-open: a disabled policy, a missing
store hook, or any error resolves to ``False`` (never blocks).

The gate only touches ``self._state`` (the StateStore), so these tests
construct ``AdmissionQueue(scheduler=object(), state=...)`` with a dummy
scheduler — no event loop, no port bind, no real placement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.state import RawRolloutRecord, SqliteStateStore


def _queue(state: object) -> AdmissionQueue:
    # The fair-share gate never calls the scheduler; a bare object() is enough.
    return AdmissionQueue(scheduler=object(), state=state)  # type: ignore[arg-type]


def _seed_raw_running(store: SqliteStateStore, owner: str, n: int) -> None:
    for i in range(n):
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=f"r-{owner}-{i}", status="running",
            image="busybox:1", owner_id=owner,
        ))


@pytest.fixture
def store(tmp_path: Path) -> SqliteStateStore:
    s = SqliteStateStore(tmp_path / "s.db")
    yield s
    s.close()


def test_gate_off_by_default(store: SqliteStateStore) -> None:
    # No policy configured → fairness disabled → nobody is ever at cap.
    _seed_raw_running(store, "alice", 5)
    q = _queue(store)
    assert q._owner_at_cap("alice") is False
    assert q._owner_at_cap("bob") is False


def test_owner_at_cap_when_running_meets_cap(store: SqliteStateStore) -> None:
    # capacity 2 is a per-owner cap; alice has 2 running → at cap.
    store.set_fairness_global(capacity_basis=2)
    _seed_raw_running(store, "alice", 2)
    q = _queue(store)
    assert q._owner_at_cap("alice") is True


def test_owner_below_cap_when_no_running(store: SqliteStateStore) -> None:
    # bob has 0 running; alice contending does not lower bob's per-owner cap.
    store.set_fairness_global(capacity_basis=2)
    _seed_raw_running(store, "alice", 2)
    q = _queue(store)
    assert q._owner_at_cap("bob") is False


def test_blocked_owner_always_at_cap(store: SqliteStateStore) -> None:
    # Blocking alice → cap 0 → at cap even with capacity raised high and
    # only one sandbox running.
    store.set_fairness_global(capacity_basis=100)
    store.set_fairness_owner("alice", blocked=True)
    _seed_raw_running(store, "alice", 1)
    q = _queue(store)
    assert q._owner_at_cap("alice") is True


def test_raising_capacity_lifts_the_gate(store: SqliteStateStore) -> None:
    # alice at cap with per-owner capacity 2 (2 running). Raise capacity so
    # cap > running.
    store.set_fairness_global(capacity_basis=2)
    _seed_raw_running(store, "alice", 2)
    q = _queue(store)
    assert q._owner_at_cap("alice") is True

    store.set_fairness_global(capacity_basis=20)
    # cap is re-read live on each call — now well above alice's 2 running.
    assert q._owner_at_cap("alice") is False


def test_owner_cap_override_can_exceed_default_capacity(
    store: SqliteStateStore,
) -> None:
    # --default-cap is the default owner cap; --owner alice --cap 4 overrides
    # it upward when real scheduler resources exist.
    store.set_fairness_global(capacity_basis=2)
    store.set_fairness_owner("alice", hard_cap=4)
    _seed_raw_running(store, "alice", 3)
    q = _queue(store)
    assert q._owner_at_cap("alice") is False


def test_uncapped_owner_is_never_at_fairshare_cap(
    store: SqliteStateStore,
) -> None:
    store.set_fairness_global(capacity_basis=2)
    store.set_fairness_owner("alice", uncapped=True)
    _seed_raw_running(store, "alice", 100)
    q = _queue(store)
    assert q._owner_at_cap("alice") is False


def test_raw_acquiring_row_does_not_self_block(store: SqliteStateStore) -> None:
    # Audit M3: a raw acquire writes its row as 'acquiring' BEFORE the gate
    # runs. With default cap 1 an otherwise-idle owner's first
    # acquire must NOT count itself and park forever — 'acquiring' is excluded.
    store.set_fairness_global(capacity_basis=1, floor=1)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-alice-acq", status="acquiring",
        image="busybox:1", owner_id="alice",
    ))
    q = _queue(store)
    assert q._owner_at_cap("alice") is False
    # Once it is actually placed (running), it counts and the owner is at cap.
    store.update_raw_rollout("r-alice-acq", status="running")
    assert q._owner_at_cap("alice") is True


def test_blocked_owner_at_cap_with_zero_running(store: SqliteStateStore) -> None:
    # cap=0 means no new admissions; cap=0, running=0 → 0 >= 0 is True → gated.
    # This guards the edge case where a blocked owner has not yet started any
    # sandboxes — they should still be blocked, not admitted for "free".
    store.set_fairness_global(capacity_basis=100)
    store.set_fairness_owner("alice", blocked=True)
    q = _queue(store)
    assert q._owner_at_cap("alice") is True


def test_owner_cap_override_downward_blocks_at_lower_threshold(
    store: SqliteStateStore,
) -> None:
    # --default-cap is the default owner cap; --owner alice --cap 2 overrides
    # it downward: alice is at cap after 2 running even though the default is 10.
    store.set_fairness_global(capacity_basis=10)
    store.set_fairness_owner("alice", hard_cap=2)
    _seed_raw_running(store, "alice", 2)
    q = _queue(store)
    assert q._owner_at_cap("alice") is True
    # bob has no override and only 1 running → well under default cap of 10.
    _seed_raw_running(store, "bob", 1)
    assert q._owner_at_cap("bob") is False


def test_fail_open_when_store_lacks_fairness_hooks() -> None:
    # A bare stub with no get_fairness_policy / running_counts_by_owner must
    # not raise — the gate fails open (returns False).
    class _Stub:
        pass

    q = _queue(_Stub())
    assert q._owner_at_cap("alice") is False


def test_blocked_owner_with_disabled_policy_fails_open(
    store: SqliteStateStore,
) -> None:
    # When the policy is disabled (capacity_basis=None), a blocked override has
    # no effect on the gate — FairnessPolicy.cap_for returns None (uncapped) and
    # the gate must return False (fail-open), not True.
    # This guards the invariant: fairness is strictly opt-in; disabling the
    # policy is a hard off-switch, even for owners that have a blocked flag.
    store.set_fairness_owner("alice", blocked=True)
    # policy NOT enabled (no set_fairness_global call)
    q = _queue(store)
    assert q._owner_at_cap("alice") is False


def test_uncapped_owner_not_at_cap_even_with_high_running_count(
    store: SqliteStateStore,
) -> None:
    # Already covered by test_uncapped_owner_is_never_at_fairshare_cap but this
    # variant seeds many running sandboxes to confirm the boundary clearly.
    store.set_fairness_global(capacity_basis=1)
    store.set_fairness_owner("alice", uncapped=True)
    _seed_raw_running(store, "alice", 50)
    q = _queue(store)
    assert q._owner_at_cap("alice") is False
    # bob with no override and 1 running is exactly at the default cap.
    _seed_raw_running(store, "bob", 1)
    assert q._owner_at_cap("bob") is True


def test_owner_at_cap_is_re_read_after_recap(store: SqliteStateStore) -> None:
    # After a recap (hard_cap cleared, uncapped=False, blocked=False), the gate
    # must re-read the live policy and fall back to the default cap.
    store.set_fairness_global(capacity_basis=2)
    store.set_fairness_owner("alice", uncapped=True)
    _seed_raw_running(store, "alice", 10)
    q = _queue(store)
    # While uncapped, the gate is open.
    assert q._owner_at_cap("alice") is False
    # Simulate a recap (clear uncapped).
    store.set_fairness_owner("alice", uncapped=False, hard_cap=None, blocked=False)
    # Now alice has 10 running and default cap is 2 → at cap.
    assert q._owner_at_cap("alice") is True


def test_owner_cap_state_reports_running_and_cap(store: SqliteStateStore) -> None:
    # _owner_cap_state exposes (running, cap) when a cap applies, None when uncapped —
    # the numbers the throttle warning surfaces.
    store.set_fairness_global(capacity_basis=2)
    _seed_raw_running(store, "alice", 3)
    q = _queue(store)
    assert q._owner_cap_state("alice") == (3, 2)   # over cap: running 3 >= cap 2
    assert q._owner_cap_state("bob") == (0, 2)      # capped policy, 0 running
    # A store with no fairness hooks (bare stub) → no cap → None (fail-open).
    assert _queue(object())._owner_cap_state("alice") is None


def test_over_cap_warning_is_deduped_per_owner(
    store: SqliteStateStore, caplog: pytest.LogCaptureFixture,
) -> None:
    # A burst of over-cap acquires must NOT flood the log: at most one warning per owner per
    # interval. alice warned twice back-to-back → 1 line; bob once → 1 line.
    import logging

    store.set_fairness_global(capacity_basis=2)
    _seed_raw_running(store, "alice", 3)
    _seed_raw_running(store, "bob", 3)
    q = _queue(store)
    with caplog.at_level(logging.WARNING, logger="xrlenv.control.admission"):
        q._warn_over_cap("alice")
        q._warn_over_cap("alice")   # within the dedup window → suppressed
        q._warn_over_cap("alice")   # ditto
        q._warn_over_cap("bob")     # different owner → its own single warning
    warns = [r.getMessage() for r in caplog.records if "fair-share cap" in r.getMessage()]
    assert len(warns) == 2
    assert any("owner=alice" in m and "running=3 >= cap=2" in m for m in warns)
    assert any("owner=bob" in m for m in warns)


def test_over_cap_warning_refires_after_interval(
    store: SqliteStateStore, caplog: pytest.LogCaptureFixture,
) -> None:
    # After the dedup window elapses the warning fires again (a periodic "still throttled"
    # reminder), so a long throttle isn't silently forgotten.
    import logging

    import xrlenv.control.admission as adm

    store.set_fairness_global(capacity_basis=2)
    _seed_raw_running(store, "alice", 3)
    q = _queue(store)
    with caplog.at_level(logging.WARNING, logger="xrlenv.control.admission"):
        q._warn_over_cap("alice")
        # Backdate the last-warn stamp past the dedup window to simulate elapsed time.
        q._over_cap_warned_at["alice"] -= adm._OVER_CAP_WARN_INTERVAL_S + 1.0
        q._warn_over_cap("alice")
    warns = [r.getMessage() for r in caplog.records if "fair-share cap" in r.getMessage()]
    assert len(warns) == 2
