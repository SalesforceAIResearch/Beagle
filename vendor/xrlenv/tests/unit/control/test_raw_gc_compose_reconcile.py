"""P1.7.C.2 — raw-GC reconciler restart-rebuild + orphan routing for compose.

The reconciler must treat a multi-service compose PROJECT's ``main`` container
(session_kind=raw, so it shows up on the raw node-truth diff) differently from a
plain raw container:

  * **restart survivor** (persisted project row + node reports its main) →
    re-adopt the *whole project* (rebuild the ``main`` session + ``_compose_projects``)
    instead of force-destroying it and sealing the rollout lost-on-restart;
  * **genuine orphan** (compose label but no re-adoptable row) → down the *whole
    stack* via ``destroy_compose_project`` rather than a bare single-container
    force-destroy that would leak the session_kind=compose sidecars (they're off
    the raw diff, so nothing else reaps them);
  * **stale row reclaim** — a persisted compose row whose project is gone past the
    TTL is reclaimed so ``compose_projects`` doesn't accumulate ghosts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.raw_gc_reconciler import RawGCReconciler
from xrlenv.control.state import (
    ComposeProjectStateRecord,
    RawRolloutRecord,
)
from xrlenv.errors import XRLEnvError


@dataclass
class _FakeTransport:
    node_id: str = "node-A"
    docker_container_ids: list[str] = field(default_factory=list)
    # container_id -> (rollout_id, compose_project); the real RemoteNodeTransport
    # populates this as a side effect of list_raw_container_ids.
    container_info: dict[str, tuple[str, str]] = field(default_factory=dict)
    force_destroyed: list[str] = field(default_factory=list)
    composed_down: list[tuple[str, str]] = field(default_factory=list)
    down_raises: bool = False
    down_attempts: int = 0
    # audit H11 — the broader managed inventory: (cid, rollout_id, compose_project, session_kind),
    # incl compose sidecars (session_kind=compose). None ⇒ transport doesn't support it (old node).
    managed_info: list[tuple[str, str, str, str]] | None = None

    def supported_backends(self) -> list[str]:
        return ["docker"]

    async def list_raw_container_ids(self, **_: Any) -> list[str]:
        self._last_container_info = dict(self.container_info)
        return list(self.docker_container_ids)

    async def list_managed_container_info(
        self, **_: Any,
    ) -> list[tuple[str, str, str, str]]:
        if self.managed_info is None:
            # An old node without the broader listing falls back to the raw set (session_kind "").
            return [(cid, rid, proj, "") for cid, (rid, proj) in self.container_info.items()]
        return list(self.managed_info)

    async def force_destroy_raw_container(self, *, container_id: str) -> None:
        self.force_destroyed.append(container_id)
        # reflect the reap: a destroyed container disappears from subsequent listings
        self.docker_container_ids = [c for c in self.docker_container_ids if c != container_id]
        self.container_info.pop(container_id, None)
        if self.managed_info is not None:
            self.managed_info = [m for m in self.managed_info if m[0] != container_id]

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str, force: bool = True,
    ) -> None:
        self.down_attempts += 1
        if self.down_raises:
            raise XRLEnvError("project not registered on this node")
        self.composed_down.append((rollout_id, project_name))
        # reflect the whole-project reap: every member of this rollout disappears
        self.docker_container_ids = [
            c for c in self.docker_container_ids
            if self.container_info.get(c, ("", ""))[0] != rollout_id
        ]
        self.container_info = {
            c: v for c, v in self.container_info.items() if v[0] != rollout_id
        }
        if self.managed_info is not None:
            self.managed_info = [m for m in self.managed_info if m[1] != rollout_id]


@dataclass
class _FakeRegistry:
    transports: dict[str, _FakeTransport]

    @property
    def node_ids(self) -> list[str]:
        return list(self.transports.keys())

    def get(self, node_id: str) -> _FakeTransport | None:
        return self.transports.get(node_id)


@dataclass
class _FakeState:
    rows: list[RawRolloutRecord] = field(default_factory=list)
    compose_rows: list[ComposeProjectStateRecord] = field(default_factory=list)
    updates: list[dict[str, Any]] = field(default_factory=list)

    # raw_rollouts surface (subset used by the sweep)
    def list_raw_rollouts(
        self, *, status: str | None = None,
    ) -> list[RawRolloutRecord]:
        if status is None:
            return list(self.rows)
        return [r for r in self.rows if r.status == status]

    def get_raw_rollout(self, rollout_id: str) -> RawRolloutRecord | None:
        return next((r for r in self.rows if r.rollout_id == rollout_id), None)

    def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
        self.updates.append({"rollout_id": rollout_id, **fields})
        for i, row in enumerate(self.rows):
            if row.rollout_id == rollout_id:
                self.rows[i] = row.model_copy(update=fields)

    def record_raw_rollout(self, record: RawRolloutRecord) -> None:
        self.rows.append(record)

    # compose_projects surface
    def list_compose_projects(self) -> list[ComposeProjectStateRecord]:
        return list(self.compose_rows)

    def record_compose_project(self, rec: ComposeProjectStateRecord) -> None:
        self.compose_rows = [
            r for r in self.compose_rows if r.rollout_id != rec.rollout_id
        ]
        self.compose_rows.append(rec)

    def delete_compose_project(self, rollout_id: str) -> None:
        self.compose_rows = [
            r for r in self.compose_rows if r.rollout_id != rollout_id
        ]


@dataclass
class _FakeScheduler:
    nodes: list[Any] = field(default_factory=list)
    image_aware_placement: bool = False

    def place(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
        raise XRLEnvError("placement not used in these tests")

    def commit_placement(self, *_: Any) -> None:  # pragma: no cover
        pass

    def release_placement(self, *_: Any) -> None:  # pragma: no cover
        pass


def _footprint_json() -> str:
    from xrlenv.control.raw_container_service import _DEFAULT_RAW_RESOURCES

    return _DEFAULT_RAW_RESOURCES.model_dump_json()


def _compose_row(rollout_id: str, project: str, **kw: Any) -> ComposeProjectStateRecord:
    base: dict[str, Any] = {
        "rollout_id": rollout_id,
        "project_name": project,
        "node_id": "node-A",
        "footprint_json": _footprint_json(),
        "subnet_claims_json": '["172.16.9.0/24"]',
    }
    base.update(kw)
    return ComposeProjectStateRecord(**base)


def _recon(transport: _FakeTransport, state: _FakeState) -> RawGCReconciler:
    # coordinator + reconciler share the StateStore, as in production.
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(), state=state,  # type: ignore[arg-type]
    )
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
    )
    recon._started_at = time.time()  # restart grace active
    return recon


# ──────────────────────────────────────────────────────────────────────────────
# restart survivor → re-adopt whole project
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_readopts_compose_survivor_not_force_destroy() -> None:
    transport = _FakeTransport(
        docker_container_ids=["c-main"],
        container_info={"c-main": ("r1", "proj")},
    )
    state = _FakeState(compose_rows=[_compose_row("r1", "proj")])
    recon = _recon(transport, state)

    await recon.reconcile_once()

    coord = recon._coordinator
    # session rebuilt WITH the compose marker (so teardown downs the project)
    sessions = coord.list_sessions()  # type: ignore[attr-defined]
    assert [s.rollout_id for s in sessions] == ["r1"]
    assert sessions[0].compose_project_name == "proj"
    # not force-destroyed, row kept
    assert transport.force_destroyed == []
    assert state.list_compose_projects()[0].rollout_id == "r1"


# ──────────────────────────────────────────────────────────────────────────────
# genuine orphan → whole-project down, never a bare main force-destroy
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orphan_compose_main_routes_to_whole_project_down() -> None:
    # compose-main label present but NO persisted row → not re-adoptable → orphan.
    transport = _FakeTransport(
        docker_container_ids=["c-main"],
        container_info={"c-main": ("r9", "orphan-proj")},
    )
    state = _FakeState()  # no compose row, no raw row
    recon = _recon(transport, state)

    await recon.reconcile_once()  # observe node-only
    await recon.reconcile_once()  # confirm → act

    # downed the WHOLE stack, and never force-destroyed just main (leaks sidecars)
    assert transport.composed_down == [("r9", "orphan-proj")]
    assert transport.force_destroyed == []


@pytest.mark.asyncio
async def test_orphan_compose_main_retains_and_retries_on_down_failure() -> None:
    # audit H10: when the whole-project down FAILS, the reconciler must NOT force-destroy just
    # main — that strands the session_kind=compose sidecars (off the raw diff → never re-observed).
    # It RETAINS main as the sentinel and re-attempts the whole-project down on the next sweep.
    transport = _FakeTransport(
        docker_container_ids=["c-main"],
        container_info={"c-main": ("r9", "orphan-proj")},
        down_raises=True,
    )
    state = _FakeState()
    recon = _recon(transport, state)

    await recon.reconcile_once()  # observe node-only
    await recon.reconcile_once()  # confirm → attempt whole-project down (raises)
    assert transport.force_destroyed == []          # main RETAINED, not force-destroyed
    assert transport.down_attempts == 1             # attempted once so far

    # main is still reported (sentinel retained) → the next confirmed sweep retries the down.
    await recon.reconcile_once()
    assert transport.force_destroyed == []          # still never force-destroys main
    assert transport.down_attempts == 2             # retried the whole-project teardown


# ──────────────────────────────────────────────────────────────────────────────
# H10 — COORDINATOR-ONLY compose orphan (main absent from the node) → whole-project
# down (node-confirmed), never a bare capacity seal
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_coordinator_only_compose_orphan_downs_whole_project() -> None:
    # audit H10: the coordinator HOLDS a compose session but the node no longer reports its
    # main container (coordinator-only orphan). Main-absence is NOT confirmation the
    # session_kind=compose sidecars are gone (they're off the raw diff), so a bare seal_orphan
    # would release the whole project's AGGREGATE capacity while sidecars run. The reconciler
    # must route to a node-confirmed whole-project down, freeing capacity only on confirmation.
    import datetime as _dt

    from xrlenv.control.raw_container_service import (
        RawContainerSession,
        _ComposeProjectRecord,
    )

    transport = _FakeTransport(docker_container_ids=[])   # node reports NO raw containers
    state = _FakeState()
    recon = _recon(transport, state)
    coord = recon._coordinator
    coord._sessions["rC"] = RawContainerSession(   # type: ignore[attr-defined]
        rollout_id="rC", node=transport,  # type: ignore[arg-type]
        node_id="node-A", container_id="c-main", container_name="proj-main",
        image="img", created_at=_dt.datetime.now(_dt.UTC),
        compose_project_name="proj",
    )
    coord._compose_projects["rC"] = _ComposeProjectRecord(   # type: ignore[attr-defined]
        project_name="proj", node_id="node-A",
        service_container_ids={"main": "c-main", "pg": "c-pg"},
    )

    await recon.reconcile_once()

    # whole-project down (node-confirmed) — NOT a bare single-container seal; on success the
    # session + project record are dropped (capacity freed only after confirmation).
    assert transport.composed_down == [("rC", "proj")]
    assert transport.force_destroyed == []
    assert [s.rollout_id for s in coord.list_sessions()] == []   # type: ignore[attr-defined]
    assert "rC" not in coord._compose_projects                    # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_coordinator_only_compose_orphan_failed_down_retains_capacity() -> None:
    # audit H10 + invariant 2: a failed whole-project down RETAINS the session + aggregate
    # capacity (sidecars may still run) so the next sweep retries — never a premature release.
    import datetime as _dt

    from xrlenv.control.raw_container_service import (
        RawContainerSession,
        _ComposeProjectRecord,
    )

    transport = _FakeTransport(docker_container_ids=[], down_raises=True)
    state = _FakeState()
    recon = _recon(transport, state)
    coord = recon._coordinator
    coord._sessions["rC"] = RawContainerSession(   # type: ignore[attr-defined]
        rollout_id="rC", node=transport,  # type: ignore[arg-type]
        node_id="node-A", container_id="c-main", container_name="proj-main",
        image="img", created_at=_dt.datetime.now(_dt.UTC),
        compose_project_name="proj",
    )
    coord._compose_projects["rC"] = _ComposeProjectRecord(   # type: ignore[attr-defined]
        project_name="proj", node_id="node-A",
        service_container_ids={"main": "c-main"},
    )

    await recon.reconcile_once()   # down raises → swallowed by the reconciler, session RETAINED

    assert [s.rollout_id for s in coord.list_sessions()] == ["rC"]  # type: ignore[attr-defined]
    assert "rC" in coord._compose_projects                          # type: ignore[attr-defined]
    assert transport.force_destroyed == []   # NOT force-destroyed (would leak sidecars)


# ──────────────────────────────────────────────────────────────────────────────
# H11 — readopt-on-connect re-adopts surviving load before the node is schedulable
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_readopt_on_connect_readopts_surviving_raw_session() -> None:
    # audit H11: after a CP restart _sessions is empty; on (re)connect the node's surviving
    # durable raw session must be re-adopted BEFORE the node is admitted for placement, so
    # iter_load_entries accounts for it and admission can't over-place. (distributed_runtime
    # gates scheduler.add_node behind this call.)
    transport = _FakeTransport(
        docker_container_ids=["c-live"],
        container_info={"c-live": ("rlive", "")},   # non-compose container
    )
    row = RawRolloutRecord(
        rollout_id="rlive", status="running", image="img:1", node_id="node-A",
        container_id="c-live", container_name="n-live", created_at=1000.0,
    )
    state = _FakeState(rows=[row])
    recon = _recon(transport, state)
    coord = recon._coordinator
    assert coord.list_sessions() == []                 # CP restart: nothing in memory yet

    ok = await recon.readopt_node_on_connect(transport)

    assert ok is True                                  # clean pass → caller may admit
    # re-adopted → iter_load_entries now charges the surviving container (no over-place window).
    assert [s.rollout_id for s in coord.list_sessions()] == ["rlive"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_readopt_on_connect_reroutes_stale_generation_session() -> None:
    # audit H11 (ownership transfer): a node-agent reconnect can leave a session routed through
    # the OLD transport (its _on_disconnected no-ops once the registry points at the replacement).
    # readopt-on-connect against the NEW transport must RE-ROUTE the survivor and still succeed —
    # NOT fail closed forever (the old session is otherwise never re-adoptable → deadlock).
    row = RawRolloutRecord(
        rollout_id="rlive", status="running", image="img:1", node_id="node-A",
        container_id="c-live", container_name="n-live", created_at=1000.0,
    )
    old = _FakeTransport(node_id="node-A")
    new = _FakeTransport(
        node_id="node-A",
        docker_container_ids=["c-live"],
        container_info={"c-live": ("rlive", "")},
    )
    recon = _recon(new, _FakeState(rows=[row]))
    coord = recon._coordinator
    await coord.readopt(row, old)                       # seed the stale-generation session
    assert coord.list_sessions()[0].node is old         # type: ignore[attr-defined]

    ok = await recon.readopt_node_on_connect(new)

    assert ok is True                                   # re-routed → admit, not a deadlock
    sessions = coord.list_sessions()                    # type: ignore[attr-defined]
    assert len(sessions) == 1                            # not duplicated
    assert sessions[0].node is new                       # now routes through the new transport


@pytest.mark.asyncio
async def test_readopt_on_connect_fails_closed_when_readopt_cannot_transfer() -> None:
    # audit H11: if readopt returns False (an in-flight acquire owns the rollout), the node
    # reports a live managed container we could NOT transfer ownership of → fail closed rather
    # than admit while it routes through a stale/absent generation.
    row = RawRolloutRecord(
        rollout_id="racq", status="acquiring", image="img:1", node_id="node-A",
        container_id="c-acq", container_name="n", created_at=1000.0,
    )
    transport = _FakeTransport(
        docker_container_ids=["c-acq"],
        container_info={"c-acq": ("racq", "")},
    )
    recon = _recon(transport, _FakeState(rows=[row]))
    coord = recon._coordinator
    coord._acquiring_ids.add("racq")                    # an acquire is in flight for this rollout

    ok = await recon.readopt_node_on_connect(transport)
    assert ok is False
    assert coord.list_sessions() == []                  # type: ignore[attr-defined]  # not adopted


@pytest.mark.asyncio
async def test_readopt_on_connect_fails_closed_on_fleet_rebuild_typeerror() -> None:
    # audit H11: a real TypeError raised INSIDE the production fleet rebuild must FAIL the
    # reconnect pass — not be mistaken for an "old test-double signature" and masked by a
    # suppressed best-effort retry. The rebuild is called by capability (signature inspection),
    # so an accepting-but-raising rebuild fails closed.
    transport = _FakeTransport()   # no containers — the loop is empty; only the rebuild runs
    recon = _recon(transport, _FakeState())
    coord = recon._coordinator

    async def _boom(*, now: Any = None, reclaim_after_s: Any = None,
                    allow_reclaim: Any = None, raise_on_error: bool = False) -> Any:
        raise TypeError("a genuine bug inside rebuild — must not be masked")
    coord.rebuild_fleets_from_state = _boom  # type: ignore[method-assign]

    assert await recon.readopt_node_on_connect(transport) is False


@pytest.mark.asyncio
async def test_readopt_on_connect_readopts_surviving_compose_project() -> None:
    # H11 for compose: a surviving compose main + persisted row is re-adopted (session +
    # _compose_projects) on connect, so teardown later routes to a whole-project down.
    transport = _FakeTransport(
        docker_container_ids=["c-main"],
        container_info={"c-main": ("rc", "proj")},
    )
    state = _FakeState(compose_rows=[_compose_row("rc", "proj")])
    recon = _recon(transport, state)
    coord = recon._coordinator

    await recon.readopt_node_on_connect(transport)

    sessions = coord.list_sessions()  # type: ignore[attr-defined]
    assert [s.rollout_id for s in sessions] == ["rc"]
    assert sessions[0].compose_project_name == "proj"


@pytest.mark.asyncio
async def test_readopt_on_connect_reaps_sidecar_only_survivor() -> None:
    # audit H11 (deploy-safety): a compose project whose MAIN is gone but whose SIDECARS
    # (session_kind=compose) are still alive is invisible to the raw-only inventory (empty
    # docker_container_ids). The broader managed inventory surfaces the live sidecar → its rollout
    # has no session → REAP the whole project (by project+rollout label) and ADMIT, rather than
    # quarantine the node forever.
    transport = _FakeTransport(
        docker_container_ids=[],            # main gone → raw inventory empty
        container_info={},
        managed_info=[("c-side", "rc", "proj", "compose")],   # a surviving sidecar
    )
    recon = _recon(transport, _FakeState())   # no compose row (deleted on disconnect)

    ok = await recon.readopt_node_on_connect(transport)
    assert ok is True
    assert transport.composed_down == [("rc", "proj")]        # whole-project reap
    assert recon._coordinator.list_sessions() == []   # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_readopt_on_connect_admits_healthy_project_with_sidecars() -> None:
    # audit H11: the sidecar sweep must NOT false-quarantine a HEALTHY project — main present
    # (re-adopted from its row) covers its sidecars (same rollout_id) → all accounted → admit.
    transport = _FakeTransport(
        docker_container_ids=["c-main"],
        container_info={"c-main": ("rc", "proj")},
        managed_info=[
            ("c-main", "rc", "proj", "raw"),       # the main (also in the raw inventory)
            ("c-side", "rc", "proj", "compose"),   # its sidecar — same rollout_id
        ],
    )
    recon = _recon(transport, _FakeState(compose_rows=[_compose_row("rc", "proj")]))

    ok = await recon.readopt_node_on_connect(transport)
    assert ok is True
    assert [s.rollout_id for s in recon._coordinator.list_sessions()] == ["rc"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_readopt_on_connect_fails_closed_when_managed_inventory_rpc_fails() -> None:
    # audit H11: if the broader managed-inventory RPC fails we can't rule out sidecar-only
    # survivors → fail closed rather than admit on the raw-only view.
    class _T(_FakeTransport):
        async def list_managed_container_info(self, **_: Any) -> list:
            raise RuntimeError("node wedged on the managed listing")

    transport = _T()
    recon = _recon(transport, _FakeState())
    assert await recon.readopt_node_on_connect(transport) is False


@pytest.mark.asyncio
async def test_readopt_on_connect_fails_closed_on_old_agent_without_capability() -> None:
    # audit H11: an OLDER node agent that can't return the broad managed inventory
    # (ManagedInventoryUnsupported) must NOT be trusted as "clean" — its raw-only view hides
    # sidecar-only survivors. Fail reconnect closed (quarantine) until it's upgraded.
    from xrlenv.errors import ManagedInventoryUnsupported

    class _OldAgent(_FakeTransport):
        async def list_managed_container_info(self, **_: Any) -> list:
            raise ManagedInventoryUnsupported("older agent — no include_all_managed")

    transport = _OldAgent()
    recon = _recon(transport, _FakeState())
    assert await recon.readopt_node_on_connect(transport) is False


@pytest.mark.asyncio
async def test_readopt_on_connect_fails_closed_on_inventory_failure() -> None:
    # audit H11 (fail closed): if the node's inventory RPC fails we can't determine surviving
    # load — readopt returns False so the caller does NOT admit the node (it retries), instead of
    # admitting with load unaccounted.
    class _BrokenTransport(_FakeTransport):
        async def list_raw_container_ids(self, **_: Any) -> list[str]:
            raise RuntimeError("node wedged")

    recon = _recon(_BrokenTransport(), _FakeState())
    ok = await recon.readopt_node_on_connect(_BrokenTransport())
    assert ok is False


@pytest.mark.asyncio
async def test_readopt_on_connect_fails_closed_on_node_ownership_mismatch() -> None:
    # audit H11: a persisted row whose node_id != the connecting transport is corrupt/stale
    # inventory — don't route a session through the wrong node; fail closed (return False, no
    # session created).
    transport = _FakeTransport(
        docker_container_ids=["c-live"],
        container_info={"c-live": ("rlive", "")},
    )
    row = RawRolloutRecord(
        rollout_id="rlive", status="running", image="img:1", node_id="node-OTHER",
        container_id="c-live", container_name="n-live", created_at=1000.0,
    )
    recon = _recon(transport, _FakeState(rows=[row]))
    coord = recon._coordinator

    ok = await recon.readopt_node_on_connect(transport)
    assert ok is False
    assert coord.list_sessions() == []   # type: ignore[attr-defined]  # not re-adopted


@pytest.mark.asyncio
async def test_readopt_on_connect_reaps_unmatched_raw_survivor() -> None:
    # audit H11 (deploy-safety): a managed raw container with NO durable row is a leaked orphan.
    # readopt REAPS it (force-destroy) and ADMITS — not quarantine (which deadlocked deploys,
    # since readopt only re-runs on reconnect and never re-tried after the periodic sweep reaped).
    transport = _FakeTransport(
        docker_container_ids=["c-orphan"],
        container_info={"c-orphan": ("rgone", "")},   # managed (rollout label), no row
    )
    recon = _recon(transport, _FakeState())
    ok = await recon.readopt_node_on_connect(transport)
    assert ok is True
    assert transport.force_destroyed == ["c-orphan"]   # reaped, node admitted
    assert recon._coordinator.list_sessions() == []   # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_readopt_on_connect_reaps_terminal_row_survivor() -> None:
    # THE prod deploy case: a container whose rollout row is TERMINAL (a CP-restart node-loss
    # sealed it failed/released, then the container lingered). It's an unambiguous orphan → REAP
    # (force-destroy) + ADMIT.
    transport = _FakeTransport(
        docker_container_ids=["c-dead"],
        container_info={"c-dead": ("rdone", "")},
    )
    row = RawRolloutRecord(
        rollout_id="rdone", status="failed", image="img:1", node_id="node-A",
        container_id="c-dead", container_name="n", created_at=1000.0,
    )
    recon = _recon(transport, _FakeState(rows=[row]))
    ok = await recon.readopt_node_on_connect(transport)
    assert ok is True
    assert transport.force_destroyed == ["c-dead"]     # reaped, node admitted


@pytest.mark.asyncio
async def test_readopt_on_connect_fails_closed_when_reap_fails() -> None:
    # if the orphan reap can't be CONFIRMED (force-destroy RPC errors), we can't admit a node
    # that may still carry uncharged load → fail closed + retry.
    class _T(_FakeTransport):
        async def force_destroy_raw_container(self, *, container_id: str) -> None:
            raise RuntimeError("force-destroy wire error")

    transport = _T(
        docker_container_ids=["c-orphan"],
        container_info={"c-orphan": ("rgone", "")},
    )
    recon = _recon(transport, _FakeState())
    assert await recon.readopt_node_on_connect(transport) is False


@pytest.mark.asyncio
async def test_readopt_on_connect_reaps_compose_main_without_project_row() -> None:
    # a compose MAIN whose project row is missing/terminal is a leaked compose orphan → REAP the
    # whole project (main + sidecars) via whole-project teardown + ADMIT.
    transport = _FakeTransport(
        docker_container_ids=["c-main"],
        container_info={"c-main": ("rc", "proj")},   # compose-labeled, but no compose row
    )
    recon = _recon(transport, _FakeState())
    ok = await recon.readopt_node_on_connect(transport)
    assert ok is True
    assert transport.composed_down == [("rc", "proj")]   # whole-project reap, node admitted


@pytest.mark.asyncio
async def test_plain_raw_orphan_still_single_container_force_destroy() -> None:
    # regression guard: a NON-compose orphan (no project label) keeps the plain
    # single-container force-destroy path — routing only applies to compose-mains.
    transport = _FakeTransport(
        docker_container_ids=["c-plain"],
        container_info={"c-plain": ("r2", "")},  # empty project = not compose
    )
    state = _FakeState()
    recon = _recon(transport, state)

    await recon.reconcile_once()
    await recon.reconcile_once()

    assert transport.force_destroyed == ["c-plain"]
    assert transport.composed_down == []


# ──────────────────────────────────────────────────────────────────────────────
# stale-row reclaim wired into the sweep
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_reclaims_stale_compose_row_after_grace() -> None:
    # a persisted row whose project the node no longer reports (not re-adopted),
    # created long ago, is reclaimed once the re-adoption grace has lapsed.
    transport = _FakeTransport(docker_container_ids=[])  # node reports nothing
    state = _FakeState(
        compose_rows=[_compose_row("gone", "p", created_ts=1_000.0)],
    )
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(), state=state,  # type: ignore[arg-type]
    )
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        readopt_grace_s=0.0,  # grace already elapsed
        fleet_reservation_ttl_s=1.0,
    )
    recon._started_at = time.time() - 10.0

    report = await recon.reconcile_once()

    assert state.list_compose_projects() == []  # reclaimed
    assert report.get("__compose__", {}).get("reclaimed") == 1


@pytest.mark.asyncio
async def test_sweep_keeps_stale_row_within_grace() -> None:
    transport = _FakeTransport(docker_container_ids=[])
    state = _FakeState(
        compose_rows=[_compose_row("gone", "p", created_ts=1_000.0)],
    )
    coord = RawContainerCoordinator(
        scheduler=_FakeScheduler(), state=state,  # type: ignore[arg-type]
    )
    recon = RawGCReconciler(
        registry=_FakeRegistry({"node-A": transport}),  # type: ignore[arg-type]
        coordinator=coord,
        state=state,  # type: ignore[arg-type]
        readopt_grace_s=300.0,  # grace still active
        fleet_reservation_ttl_s=1.0,
    )
    recon._started_at = time.time()

    await recon.reconcile_once()

    assert len(state.list_compose_projects()) == 1  # not reclaimed within grace
