"""Tests for StateRetentionJanitor (spec 20 retention GC matrix).

The janitor is a thin scheduler around ``StateStore.prune_expired`` (whose
per-table logic is covered in test_sqlite_state.py). These tests pin the
janitor's own contract: construction validation + that ``sweep_once`` wires the
configured windows through to a prune and runs off the event loop.
"""

from __future__ import annotations

import time

import pytest
from xrlenv.control.state import InMemoryStateStore, RawRolloutRecord
from xrlenv.control.state_retention_janitor import StateRetentionJanitor


def test_rejects_nonpositive_retention() -> None:
    with pytest.raises(ValueError, match="audit_retention_days"):
        StateRetentionJanitor(InMemoryStateStore(), audit_retention_days=0)
    with pytest.raises(ValueError, match="raw_rollout_retention_days"):
        StateRetentionJanitor(InMemoryStateStore(), raw_rollout_retention_days=-1)


def test_none_windows_are_allowed() -> None:
    # A fully-disabled janitor is legal (all windows None) — it just no-ops.
    j = StateRetentionJanitor(
        InMemoryStateStore(),
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=None,
    )
    assert j is not None


async def test_sweep_once_prunes_old_terminal_raw_rollouts() -> None:
    state = InMemoryStateStore()
    old = time.time() - 100 * 86400
    state.record_raw_rollout(
        RawRolloutRecord(
            rollout_id="old", status="released", image="i",
            created_at=old, finished_at=old,
        )
    )
    state.record_raw_rollout(  # active + old — must survive
        RawRolloutRecord(rollout_id="run", status="running", image="i", created_at=old)
    )
    janitor = StateRetentionJanitor(
        state,
        audit_retention_days=None,
        events_retention_days=None,
        raw_rollout_retention_days=1,
    )
    counts = await janitor.sweep_once()

    assert counts["raw_rollouts"] == 1
    assert state.get_raw_rollout("old") is None
    assert state.get_raw_rollout("run") is not None
