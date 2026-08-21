"""Tests for StateStore.raw_rollouts_created_span (Change 2).

Verifies the new method on BOTH InMemoryStateStore and SqliteStateStore:

1. Empty store → (None, None).
2. Single rollout → (ts, ts) — min equals max.
3. Multiple rollouts with distinct created_at values → (min, max) correct.
4. Non-NOT-NULL columns (status, image, owner_id, created_at) are always set.

The parametrized ``store`` fixture mirrors the pattern in test_sqlite_state.py
so both backends run through every assertion in a single pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from xrlenv.control.state import (
    InMemoryStateStore,
    RawRolloutRecord,
    SqliteStateStore,
    StateStore,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture — both backends
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["inmem", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StateStore]:
    if request.param == "inmem":
        s: StateStore = InMemoryStateStore()
        yield s
    else:
        s = SqliteStateStore(tmp_path / "state.db")
        try:
            yield s
        finally:
            s.close()  # type: ignore[attr-defined]


def _raw(
    rollout_id: str,
    *,
    created_at: float,
    status: str = "released",
    image: str = "busybox:1",
    owner_id: str = "default",
) -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status=status,  # type: ignore[arg-type]
        image=image,
        owner_id=owner_id,
        created_at=created_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_created_span_empty_store_returns_none_none(store: StateStore) -> None:
    """An empty raw_rollouts table / dict returns (None, None)."""
    lo, hi = store.raw_rollouts_created_span()
    assert lo is None
    assert hi is None


def test_created_span_single_rollout_min_equals_max(store: StateStore) -> None:
    """With a single row the span is (ts, ts) — both bounds equal the one value."""
    ts = 1_700_000_000.0
    store.record_raw_rollout(_raw("r1", created_at=ts))

    lo, hi = store.raw_rollouts_created_span()
    assert lo == ts
    assert hi == ts


def test_created_span_multiple_rollouts_returns_correct_min_max(
    store: StateStore,
) -> None:
    """With three rows at known timestamps, (min, max) is returned correctly."""
    t1 = 1_000_000.0
    t2 = 2_000_000.0
    t3 = 1_500_000.0  # middle — ensures we don't just use insertion order

    store.record_raw_rollout(_raw("r1", created_at=t1))
    store.record_raw_rollout(_raw("r2", created_at=t2))
    store.record_raw_rollout(_raw("r3", created_at=t3))

    lo, hi = store.raw_rollouts_created_span()
    assert lo == t1  # minimum
    assert hi == t2  # maximum


def test_created_span_result_is_float_not_none_when_present(
    store: StateStore,
) -> None:
    """Return values are floats (not ints or other types) when rows exist."""
    ts = 1_700_000_500.123
    store.record_raw_rollout(_raw("r1", created_at=ts))

    lo, hi = store.raw_rollouts_created_span()
    assert lo is not None and hi is not None
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    # Sub-second precision is preserved (SQLite stores REAL; in-memory stores float).
    assert abs(lo - ts) < 1e-3
    assert abs(hi - ts) < 1e-3


def test_created_span_two_rollouts_identical_timestamps(store: StateStore) -> None:
    """Two rows with the same created_at → (ts, ts)."""
    ts = 9_000_000.0
    store.record_raw_rollout(_raw("r1", created_at=ts))
    store.record_raw_rollout(_raw("r2", created_at=ts))

    lo, hi = store.raw_rollouts_created_span()
    assert lo == ts
    assert hi == ts


def test_created_span_works_across_statuses(store: StateStore) -> None:
    """Span covers all rows regardless of status (released, running, failed, etc.)."""
    t_old = 500_000.0
    t_new = 999_000_000.0
    store.record_raw_rollout(_raw("r1", created_at=t_old, status="running"))
    store.record_raw_rollout(_raw("r2", created_at=t_new, status="failed"))

    lo, hi = store.raw_rollouts_created_span()
    assert lo == t_old
    assert hi == t_new
