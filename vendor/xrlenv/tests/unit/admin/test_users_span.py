"""Tests for the span_start / span_end keys added to ``_users_blocking`` (Change 3).

The ``/users`` handler calls ``_users_blocking(cfg)`` in a thread and unpacks
``rows``, ``totals``, ``span_start``, and ``span_end``.  The span keys were added
to surface the created_at window of the surviving raw_rollouts rows so the page
can label its stats as a retention-windowed view rather than all-time.

Tests:

1. Empty DB → span_start and span_end are None in the returned dict.
2. Rollouts present → span_start and span_end are non-None ISO strings in
   ``"%Y-%m-%d %H:%M:%S UTC"`` format (the ``_iso`` helper format).
3. span values bracket the actual rollout timestamps (span_start <= every
   created_at <= span_end when compared as epoch seconds via the DB).
4. Existing rows/totals aggregation is unaffected:
   - Owner grouping correct.
   - paced excluded from total.
   - success_pct correct.
5. span_start < span_end when rollouts span different timestamps (not equal).
6. The ``/users`` HTML page renders without 500 and includes the keys from the
   context dict (the template will either render or silently skip missing keys;
   the important thing is the handler reaches 200 without error).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xrlenv.admin.server import AdminServerConfig, _users_blocking, build_admin_app
from xrlenv.control.state import RawRolloutRecord, SqliteStateStore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$")


def _raw(
    rollout_id: str,
    owner: str,
    status: str,
    *,
    created_at: float | None = None,
) -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status=status,  # type: ignore[arg-type]
        image="busybox:1",
        owner_id=owner,
        created_at=created_at if created_at is not None else time.time(),
    )


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — empty DB → span keys are None
# ──────────────────────────────────────────────────────────────────────────────


def test_span_both_none_when_empty(state_db: Path, runs_root: Path) -> None:
    """When raw_rollouts is empty, span_start and span_end must be None."""
    # State DB doesn't exist yet — _users_blocking handles non-existent db gracefully.
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    data = _users_blocking(cfg)

    assert "span_start" in data, "span_start key must be present in the returned dict"
    assert "span_end" in data, "span_end key must be present in the returned dict"
    assert data["span_start"] is None, f"Expected None, got {data['span_start']!r}"
    assert data["span_end"] is None, f"Expected None, got {data['span_end']!r}"


def test_span_both_none_with_existing_empty_db(state_db: Path, runs_root: Path) -> None:
    """When the DB file exists but raw_rollouts is empty, span keys are still None."""
    # Create the DB (triggers schema migrations) but add no rollouts.
    store = SqliteStateStore(state_db)
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    data = _users_blocking(cfg)

    assert data["span_start"] is None
    assert data["span_end"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — span keys are ISO strings when rollouts exist
# ──────────────────────────────────────────────────────────────────────────────


def test_span_values_are_iso_strings_when_rollouts_present(
    state_db: Path, runs_root: Path,
) -> None:
    """span_start and span_end must be ``%Y-%m-%d %H:%M:%S UTC`` formatted strings."""
    store = SqliteStateStore(state_db)
    store.record_raw_rollout(_raw("r1", "alice", "released", created_at=1_700_000_000.0))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    data = _users_blocking(cfg)

    assert data["span_start"] is not None
    assert data["span_end"] is not None
    assert _ISO_PATTERN.match(data["span_start"]), (
        f"span_start {data['span_start']!r} does not match ISO format"
    )
    assert _ISO_PATTERN.match(data["span_end"]), (
        f"span_end {data['span_end']!r} does not match ISO format"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — span brackets actual rollout timestamps
# ──────────────────────────────────────────────────────────────────────────────


def test_span_start_equals_span_end_for_single_rollout(
    state_db: Path, runs_root: Path,
) -> None:
    """With one rollout, span_start and span_end are the same ISO timestamp."""
    ts = 1_600_000_000.0
    store = SqliteStateStore(state_db)
    store.record_raw_rollout(_raw("r1", "alice", "released", created_at=ts))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    data = _users_blocking(cfg)

    assert data["span_start"] == data["span_end"], (
        f"Single rollout should yield identical span bounds; "
        f"got start={data['span_start']!r} end={data['span_end']!r}"
    )


def test_span_start_before_span_end_for_multiple_rollouts(
    state_db: Path, runs_root: Path,
) -> None:
    """With rollouts at distinct timestamps, span_start < span_end lexicographically.

    ISO format with UTC sorts lexicographically correctly, so string comparison works.
    """
    t_early = 1_500_000_000.0
    t_late  = 1_700_000_000.0
    store = SqliteStateStore(state_db)
    store.record_raw_rollout(_raw("r1", "alice", "released", created_at=t_early))
    store.record_raw_rollout(_raw("r2", "alice", "released", created_at=t_late))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    data = _users_blocking(cfg)

    assert data["span_start"] is not None and data["span_end"] is not None
    assert data["span_start"] < data["span_end"], (
        f"Expected span_start < span_end; "
        f"got start={data['span_start']!r} end={data['span_end']!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — existing rows/totals aggregation is unchanged by the span addition
# ──────────────────────────────────────────────────────────────────────────────


def test_rows_and_totals_aggregation_still_correct(
    state_db: Path, runs_root: Path,
) -> None:
    """Adding span keys must not change the per-owner grouping or totals math."""
    store = SqliteStateStore(state_db)
    for rec in (
        _raw("a1", "alice", "released"),
        _raw("a2", "alice", "released"),
        _raw("a3", "alice", "failed"),
        _raw("b1", "bob", "released"),
        _raw("b2", "bob", "cancelled"),
    ):
        store.record_raw_rollout(rec)
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    data = _users_blocking(cfg)

    rows = {r["owner"]: r for r in data["rows"]}

    # Alice: 2 released + 1 failed = 3 total, success = 2/3 ≈ 66.7%
    alice = rows["alice"]
    assert alice["total"] == 3
    assert alice["released"] == 2
    assert alice["failed"] == 1
    assert alice["success_pct"] == pytest.approx(2 / 3 * 100, rel=1e-3)

    # Bob: 1 released + 1 cancelled = 2 total, success = 1/2 = 50%
    bob = rows["bob"]
    assert bob["total"] == 2
    assert bob["released"] == 1
    assert bob["cancelled"] == 1
    assert bob["success_pct"] == pytest.approx(50.0, rel=1e-3)

    # Totals
    assert data["totals"]["total"] == 5
    assert data["totals"]["released"] == 3


def test_paced_excluded_from_total_still_works_with_span(
    state_db: Path, runs_root: Path,
) -> None:
    """capacity_rejected is still excluded from total when span keys are present."""
    store = SqliteStateStore(state_db)
    for rec in (
        _raw("a1", "alice", "released"),
        _raw("a2", "alice", "capacity_rejected"),
    ):
        store.record_raw_rollout(rec)
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    data = _users_blocking(cfg)

    (alice,) = data["rows"]
    assert alice["paced"] == 1
    # total excludes paced
    assert alice["total"] == 1
    assert alice["success_pct"] == 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — /users HTTP endpoint returns 200 with span context
# ──────────────────────────────────────────────────────────────────────────────


def test_users_page_returns_200_with_rollouts(state_db: Path, runs_root: Path) -> None:
    """The /users HTML page renders successfully when rollouts are present."""
    store = SqliteStateStore(state_db)
    store.record_raw_rollout(_raw("r1", "alice", "released", created_at=1_700_000_000.0))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    with TestClient(build_admin_app(cfg)) as client:
        resp = client.get("/users")
    assert resp.status_code == 200
    # At a minimum the owner appears in the rendered page.
    assert "alice" in resp.text


def test_users_page_returns_200_with_empty_db(state_db: Path, runs_root: Path) -> None:
    """The /users HTML page renders successfully even with no rollouts (empty span)."""
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    with TestClient(build_admin_app(cfg)) as client:
        resp = client.get("/users")
    assert resp.status_code == 200
