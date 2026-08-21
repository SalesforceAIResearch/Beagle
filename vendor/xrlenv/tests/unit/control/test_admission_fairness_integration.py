"""End-to-end fair-share gate wiring through the live AdmissionQueue.

The unit tests in ``test_admission_fairness_gate.py`` pin ``_owner_at_cap`` in
isolation; this file drives the two real call sites — the ``acquire`` fast-path
(gated owner skips the immediate place and parks) and the drain loop (a gated
waiter stays queued while a *different* owner's request is admitted ahead of it,
and a ``kick`` after the owner's slot frees re-admits the parked waiter).

Uses a real ``SqliteStateStore`` (the in-memory store has no fairness hooks, so
the gate would fail-open) and a fake scheduler whose ``place`` always succeeds —
so the only thing that can hold a request back is the fairness gate, not
capacity.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.scheduler import Placement
from xrlenv.control.state import RolloutRecord, SqliteStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.types import RolloutStatus

pytestmark = pytest.mark.asyncio


def _manifest(name: str = "t") -> TemplateManifest:
    return TemplateManifest(
        name=name, version="0.1", digest=f"sha256:{name}", image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _always_places() -> Any:
    """A scheduler whose ``place`` always succeeds — capacity is never the
    constraint, so only the fairness gate can hold a request."""
    sched = MagicMock()
    sched.image_aware_placement = False
    sched.nodes = []

    def _place(_m, *, task_key=None, backend=None,
               image_present=None, preferred_home_node=None):
        node = MagicMock()
        node.node_id = "A"
        return Placement(node=node, backend="docker", score=1)

    sched.place.side_effect = _place
    return sched


def _seed_running(state: SqliteStateStore, owner: str, rollout_id: str) -> None:
    state.insert_rollout(RolloutRecord(
        rollout_id=rollout_id, template="t",
        status=RolloutStatus.RUNNING, owner_id=owner,
    ))


async def test_gate_parks_hog_lets_other_through_then_kick_readmits(
    tmp_path: Any,
) -> None:
    state = SqliteStateStore(tmp_path / "s.db")
    # capacity_basis=1 → each active owner's cap floors to 1. alice already
    # holds 1 running sandbox, so she is exactly at cap and cannot admit more.
    state.set_fairness_global(capacity_basis=1, floor=1)
    _seed_running(state, "alice", "alice-run")

    q = AdmissionQueue(scheduler=_always_places(), state=state, poll_interval_s=0.05)
    await q.start()
    try:
        # alice is at cap → her acquire parks (does NOT place even though the
        # scheduler has room). Run it as a background task.
        alice_task = asyncio.create_task(
            q.acquire(manifest=_manifest(), owner_id="alice", timeout_s=3.0),
        )
        await asyncio.sleep(0.15)
        assert not alice_task.done(), "alice should be gated + parked, not placed"

        # A *different* owner with no running sandboxes is admitted immediately,
        # despite alice sitting in the queue — no head-of-line block.
        bob_placement = await q.acquire(
            manifest=_manifest(), owner_id="bob", timeout_s=1.0,
        )
        assert bob_placement.node.node_id == "A"
        assert not alice_task.done(), "bob draining must not release alice's gate"

        # Free alice's running slot (simulate node-confirmed destroy) and kick:
        # the drain loop re-checks the gate, finds alice now under cap, and
        # admits the parked waiter.
        state.update_rollout("alice-run", status=RolloutStatus.FINISHED)
        q.kick()
        alice_placement = await asyncio.wait_for(alice_task, timeout=2.0)
        assert alice_placement.node.node_id == "A"
    finally:
        await q.stop()
        state.close()


async def test_gate_off_by_default_admits_immediately(tmp_path: Any) -> None:
    """With no policy configured, an owner already holding many sandboxes is
    still admitted immediately — fairness is strictly opt-in."""
    state = SqliteStateStore(tmp_path / "s.db")
    for i in range(5):
        _seed_running(state, "alice", f"alice-{i}")
    q = AdmissionQueue(scheduler=_always_places(), state=state, poll_interval_s=0.05)
    await q.start()
    try:
        placement = await q.acquire(
            manifest=_manifest(), owner_id="alice", timeout_s=1.0,
        )
        assert placement.node.node_id == "A"
        assert state.list_pending() == []
    finally:
        await q.stop()
        state.close()
