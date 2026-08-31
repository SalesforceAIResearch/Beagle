"""A3 / D15 (P1.1) — spec-09 GC layer 3 reconciler tests.

Pin both directions of the orphan diff:

  - **Node-only** (node has a sandbox the state store doesn't know
    about) → reconciler emits ``gc.reconcile.orphan_sandbox`` and
    issues ``destroy_sandbox`` on the node.
  - **State-only** (state store has a ``running`` sandbox attached
    to a rollout but the node's reply omits the ID) → reconciler
    emits ``gc.reconcile.lost_sandbox`` and seals the owning rollout
    ``failed/sandbox_lost`` via ``coordinator.handle_sandbox_lost``.

The reconciler operates against a tiny ``_NodeLister`` Protocol
surface so tests can supply a minimal fake without dragging in the
full :class:`~xrlenv.control.node_transport.NodeTransport`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.gc_reconciler import GCReconciler
from xrlenv.control.state import (
    InMemoryStateStore,
    RolloutRecord,
    SandboxRecord,
)
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.types import RolloutStatus

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_manifest() -> TemplateManifest:
    return TemplateManifest(
        name="t", version="0.1", digest="sha256:abc", image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


class _FakeRegistry:
    """Minimal NodeRegistry stand-in. ``transports`` is a
    ``{node_id: lister}`` dict so tests can pre-seed and rotate."""

    def __init__(self, transports: dict[str, Any]) -> None:
        self._transports = transports

    @property
    def node_ids(self) -> list[str]:
        return list(self._transports)

    def get(self, node_id: str) -> Any:
        return self._transports.get(node_id)


class _FakeNodeLister:
    """Implements the GCReconciler-private ``_NodeLister`` Protocol."""

    def __init__(self, node_id: str, *, sandbox_ids: list[str]) -> None:
        self.node_id = node_id
        self._sandbox_ids = sandbox_ids
        self.destroy_calls: list[str] = []

    async def list_sandbox_ids(
        self, *, backend: str | None = None,
    ) -> list[str]:
        return list(self._sandbox_ids)

    async def destroy_sandbox(self, sb: Any) -> None:
        self.destroy_calls.append(sb.id)


def _seed_running_sandbox_for_rollout(
    state: InMemoryStateStore,
    *,
    rollout_id: str,
    sandbox_id: str,
    node_id: str,
    template: str = "t",
) -> None:
    """Pre-create the rollout + sandbox rows the reconciler needs to
    diff against. Mirrors what :py:meth:`RolloutCoordinator._bootstrap`
    sets up but skips the EnvAdapter side so the test stays unit-level.
    """
    state.insert_rollout(RolloutRecord(
        rollout_id=rollout_id,
        template=template,
        status=RolloutStatus.RUNNING,
        node_id=node_id,
        sandbox_id=sandbox_id,
    ))
    state.insert_sandbox(SandboxRecord(
        sandbox_id=sandbox_id,
        backend="docker",
        backend_ref=f"cid-{sandbox_id}",
        stub_endpoint="tcp://127.0.0.1:0",
        template=template,
        node_id=node_id,
        rollout_id=rollout_id,
        status="running",
    ))


def _make_coordinator(state: InMemoryStateStore) -> RolloutCoordinator:
    catalog = TemplateCatalog()
    catalog.register(_make_manifest())
    sched = MagicMock()
    sched.nodes = []
    return RolloutCoordinator(catalog=catalog, scheduler=sched, state=state)


# ── Direction 1: node-only orphan ─────────────────────────────────────────────


async def test_reconcile_destroys_node_only_orphan() -> None:
    """Node has sandbox X, state doesn't know about X → reconciler
    issues ``destroy_sandbox`` on the node and emits
    ``gc.reconcile.orphan_sandbox``.
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    lister = _FakeNodeLister("node-A", sandbox_ids=["sb-orphan-on-node"])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    assert report["node-A"]["node_only"] == 1
    assert report["node-A"]["state_only"] == 0
    assert lister.destroy_calls == ["sb-orphan-on-node"]
    events = [e for e in state.events_since(0)
              if e.kind == "gc.reconcile.orphan_sandbox"]
    assert len(events) == 1
    assert events[0].payload == {
        "node_id": "node-A", "sandbox_id": "sb-orphan-on-node",
    }


async def test_reconcile_destroy_failure_does_not_break_sweep() -> None:
    """A destroy that raises must NOT stop the sweep — the next
    orphan still gets handled. The reconciler explicitly suppresses
    transport errors per the design comment in
    ``_handle_node_only``.
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)

    class _BlowingLister(_FakeNodeLister):
        async def destroy_sandbox(self, sb: Any) -> None:
            self.destroy_calls.append(sb.id)
            raise RuntimeError("destroy failed")

    lister = _BlowingLister("node-A", sandbox_ids=["sb-1", "sb-2"])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    await reconciler.reconcile_once()
    # Both orphans were attempted, even though every destroy raised.
    assert sorted(lister.destroy_calls) == ["sb-1", "sb-2"]


# ── Direction 2: state-only orphan ────────────────────────────────────────────


async def test_reconcile_seals_state_only_orphan() -> None:
    """State store has a running sandbox/rollout pair but the node
    no longer reports the sandbox → reconciler emits
    ``gc.reconcile.lost_sandbox`` and the rollout seals
    ``failed/sandbox_lost``.
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_running_sandbox_for_rollout(
        state, rollout_id="r-lost", sandbox_id="sb-lost", node_id="node-A",
    )
    lister = _FakeNodeLister("node-A", sandbox_ids=[])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    assert report["node-A"]["state_only"] == 1
    assert report["node-A"]["node_only"] == 0
    rec = state.get_rollout("r-lost")
    assert rec.status == RolloutStatus.FAILED
    assert rec.reason == "sandbox_lost"
    # Sandbox row is gone.
    with pytest.raises(KeyError):
        state.get_sandbox("sb-lost")
    # Reconcile event was recorded.
    events = [e for e in state.events_since(0)
              if e.kind == "gc.reconcile.lost_sandbox"]
    assert len(events) == 1
    assert events[0].payload == {
        "node_id": "node-A", "sandbox_id": "sb-lost",
    }


# ── No-op path ────────────────────────────────────────────────────────────────


async def test_reconcile_in_sync_emits_zero_events() -> None:
    """Both sides agree → no destroy, no seal, no event."""
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_running_sandbox_for_rollout(
        state, rollout_id="r-1", sandbox_id="sb-1", node_id="node-A",
    )
    lister = _FakeNodeLister("node-A", sandbox_ids=["sb-1"])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    assert report["node-A"] == {
        "node_only": 0, "state_only": 0, "destroy_pending_cleanup": 0,
    }
    assert lister.destroy_calls == []
    assert state.get_rollout("r-1").status == RolloutStatus.RUNNING
    reconcile_events = [
        e for e in state.events_since(0)
        if e.kind.startswith("gc.reconcile.")
    ]
    assert reconcile_events == []


# ── Per-node failure isolation ────────────────────────────────────────────────


async def test_reconcile_skips_node_when_list_raises() -> None:
    """A node whose ``list_sandbox_ids`` raises is skipped for this
    sweep; other nodes are still reconciled. State-only orphans on
    the failing node are NOT swept (we cannot tell whether the
    sandbox is missing or the RPC just failed).
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_running_sandbox_for_rollout(
        state, rollout_id="r-on-failing-node",
        sandbox_id="sb-fail", node_id="node-fail",
    )

    class _RaisingLister(_FakeNodeLister):
        async def list_sandbox_ids(
            self, *, backend: str | None = None,
        ) -> list[str]:
            raise ConnectionError("rpc failed")

    failing = _RaisingLister("node-fail", sandbox_ids=[])
    healthy = _FakeNodeLister("node-ok", sandbox_ids=["sb-orphan"])
    registry = _FakeRegistry({"node-fail": failing, "node-ok": healthy})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    # Failing node is absent from the report (skipped early).
    assert "node-fail" not in report
    assert report["node-ok"]["node_only"] == 1
    # Healthy node's orphan was destroyed.
    assert healthy.destroy_calls == ["sb-orphan"]
    # Failing node's rollout was NOT sealed — we can't tell whether
    # the sandbox actually died or the RPC was just transiently
    # broken; only a successful list followed by absence proves loss.
    assert state.get_rollout("r-on-failing-node").status == RolloutStatus.RUNNING


async def test_reconcile_per_node_timeout_skips_hung_node() -> None:
    """A node whose ``list_sandbox_ids`` hangs past
    ``per_node_timeout_s`` is treated like a transport failure —
    skipped, sweep continues for other nodes.
    """
    import asyncio

    state = InMemoryStateStore()
    coord = _make_coordinator(state)

    class _HangingLister(_FakeNodeLister):
        async def list_sandbox_ids(
            self, *, backend: str | None = None,
        ) -> list[str]:
            await asyncio.sleep(60)  # well past the 0.05 s timeout below
            return []

    hanging = _HangingLister("node-hang", sandbox_ids=[])
    healthy = _FakeNodeLister("node-ok", sandbox_ids=[])
    registry = _FakeRegistry({"node-hang": hanging, "node-ok": healthy})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state, per_node_timeout_s=0.05,
    )

    report = await reconciler.reconcile_once()

    assert "node-hang" not in report
    assert "node-ok" in report


# ── Rollout state guards on handle_sandbox_lost ──────────────────────────────


async def test_handle_sandbox_lost_skips_terminal_rollouts() -> None:
    """Defensive: if the rollout is already terminal (e.g. another
    code path raced the reconciler), the seal is a no-op — only the
    sandbox row is dropped.
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_running_sandbox_for_rollout(
        state, rollout_id="r-terminal", sandbox_id="sb-1", node_id="node-A",
    )
    state.update_rollout("r-terminal", status=RolloutStatus.FINISHED)

    await coord.handle_sandbox_lost("node-A", "sb-1", reason="sandbox_lost")

    # Status unchanged.
    assert state.get_rollout("r-terminal").status == RolloutStatus.FINISHED
    # Sandbox row dropped.
    with pytest.raises(KeyError):
        state.get_sandbox("sb-1")


async def test_handle_sandbox_lost_unknown_sandbox_is_noop() -> None:
    """Calling with an ID the state store never knew about is a
    no-op — the reconciler doesn't need to coordinate races against
    other GC layers.
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    # Should NOT raise.
    await coord.handle_sandbox_lost("node-A", "never-existed")


# ── Audit H1: destroy_pending row cleanup ─────────────────────────────────────


def _seed_destroy_pending_sandbox(
    state: InMemoryStateStore,
    *,
    sandbox_id: str,
    node_id: str,
    template: str = "t",
) -> None:
    """Mirrors the state shape the coordinator's ``_terminate`` leaves
    behind when ``destroy_sandbox`` raises / times out: the sandbox
    row stays at ``status='destroy_pending'`` but the rollout has
    already been terminalised. The reconciler is responsible for
    cleaning up the row."""
    state.insert_sandbox(SandboxRecord(
        sandbox_id=sandbox_id,
        backend="docker",
        backend_ref=f"cid-{sandbox_id}",
        stub_endpoint="tcp://127.0.0.1:0",
        template=template,
        node_id=node_id,
        rollout_id=None,  # rollout has been sealed; sandbox row orphaned
        status="destroy_pending",
    ))


async def test_reconcile_retries_destroy_when_node_still_has_destroy_pending() -> None:
    """Audit H1: state has a ``destroy_pending`` row and the node
    still reports the sandbox → reconciler retries ``destroy_sandbox``
    and on success removes the state row so the scheduler stops
    counting it against capacity. Pre-fix, this row leaked forever
    (state_set was filtered to ``running``-only, so the sandbox got
    classified as a node-only orphan and destroyed but the state row
    was untouched; on the next sweep neither side flagged the row).
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_destroy_pending_sandbox(state, sandbox_id="sb-stuck", node_id="node-A")
    lister = _FakeNodeLister("node-A", sandbox_ids=["sb-stuck"])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    # The retry was attempted on the node side.
    assert lister.destroy_calls == ["sb-stuck"]
    # And the row is gone — capacity is finally released.
    with pytest.raises(KeyError):
        state.get_sandbox("sb-stuck")
    # Counted under destroy_pending_cleanup, not node_only.
    assert report["node-A"]["destroy_pending_cleanup"] == 1
    assert report["node-A"]["node_only"] == 0
    # Reconcile event recorded.
    events = [e for e in state.events_since(0)
              if e.kind == "gc.reconcile.destroy_pending_retry"]
    assert len(events) == 1
    assert events[0].payload == {
        "node_id": "node-A", "sandbox_id": "sb-stuck",
    }


async def test_reconcile_destroy_pending_row_kept_when_retry_fails() -> None:
    """Audit H1 follow-up: if the retry destroy ALSO fails, leave the
    state row so a subsequent sweep retries. We must not silently
    drop a row whose actual node-side destroy didn't happen."""
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_destroy_pending_sandbox(state, sandbox_id="sb-stuck", node_id="node-A")

    class _BlowingLister(_FakeNodeLister):
        async def destroy_sandbox(self, sb: Any) -> None:
            self.destroy_calls.append(sb.id)
            raise RuntimeError("node still wedged")

    lister = _BlowingLister("node-A", sandbox_ids=["sb-stuck"])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    await reconciler.reconcile_once()

    # Retry was attempted but raised — the row stays for the next sweep.
    assert lister.destroy_calls == ["sb-stuck"]
    rec = state.get_sandbox("sb-stuck")
    assert rec.status == "destroy_pending"


async def test_reconcile_drops_destroy_pending_row_when_node_no_longer_has_it() -> None:
    """Audit H1: destroy actually completed (node's reply doesn't
    include the sandbox), but the CP timed out / errored before
    recording it as destroyed. The row would otherwise leak forever.
    Reconciler drops it.
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_destroy_pending_sandbox(state, sandbox_id="sb-leaked", node_id="node-A")
    lister = _FakeNodeLister("node-A", sandbox_ids=[])  # node no longer has it
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    # No destroy retry attempted — node already cleaned up.
    assert lister.destroy_calls == []
    # State row dropped → capacity released.
    with pytest.raises(KeyError):
        state.get_sandbox("sb-leaked")
    # Counted under destroy_pending_cleanup, NOT state_only (which
    # is reserved for ``running`` rows whose sandbox died unexpectedly).
    assert report["node-A"]["destroy_pending_cleanup"] == 1
    assert report["node-A"]["state_only"] == 0
    events = [e for e in state.events_since(0)
              if e.kind == "gc.reconcile.destroy_pending_cleared"]
    assert len(events) == 1


async def test_reconcile_destroy_pending_does_not_count_as_node_only() -> None:
    """Audit H1 regression: a node-reported sandbox whose state row
    is ``destroy_pending`` must NOT be classified as a node-only
    genuine orphan (the pre-fix bug). It belongs in the
    destroy_pending_cleanup bucket so the matching state row gets
    removed after a successful destroy.
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_destroy_pending_sandbox(state, sandbox_id="sb-pending", node_id="node-A")
    lister = _FakeNodeLister("node-A", sandbox_ids=["sb-pending"])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    # Pre-fix would have shown {"node_only": 1, "state_only": 0};
    # post-fix: {"destroy_pending_cleanup": 1, ...}.
    assert report["node-A"]["node_only"] == 0
    assert report["node-A"]["destroy_pending_cleanup"] == 1
    # And the corresponding event kind is destroy_pending_retry, not
    # orphan_sandbox.
    kinds = [e.kind for e in state.events_since(0)
             if e.kind.startswith("gc.reconcile.")]
    assert "gc.reconcile.orphan_sandbox" not in kinds
    assert "gc.reconcile.destroy_pending_retry" in kinds


async def test_reconcile_running_sandbox_takes_precedence_over_destroy_pending() -> None:
    """Defensive cross-check: a node-reported sandbox whose state row
    is ``running`` is in-sync (no action). Make sure the new
    destroy_pending bucketing logic doesn't double-count rows whose
    sandbox_id appears under multiple state statuses (which can't
    actually happen — sandbox_id is the PK — but the test pins the
    invariant that a running row + node-present means in-sync).
    """
    state = InMemoryStateStore()
    coord = _make_coordinator(state)
    _seed_running_sandbox_for_rollout(
        state, rollout_id="r-1", sandbox_id="sb-active", node_id="node-A",
    )
    lister = _FakeNodeLister("node-A", sandbox_ids=["sb-active"])
    registry = _FakeRegistry({"node-A": lister})
    reconciler = GCReconciler(
        registry=registry,  # type: ignore[arg-type]
        coordinator=coord, state=state,
    )

    report = await reconciler.reconcile_once()

    assert report["node-A"] == {
        "node_only": 0, "state_only": 0, "destroy_pending_cleanup": 0,
    }
    assert lister.destroy_calls == []
    # Rollout is untouched.
    assert state.get_rollout("r-1").status == RolloutStatus.RUNNING
