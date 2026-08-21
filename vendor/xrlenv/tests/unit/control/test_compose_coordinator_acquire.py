"""Coordinator compose-project acquire (multi-service step 3a-3b-ii).

Exercises ``RawContainerCoordinator.acquire_compose_project`` end-to-end with fake
scheduler / node / digest-resolver: the vet→digest→acquiring-row→prepare→place→node
command→session/commit lifecycle, the digest-pin threaded through the compose sent
to the node, the footprint carried on the session, and failure cleanup. No docker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import yaml
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.kwargs_policy import KwargsPolicy, KwargsPolicyViolation
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.errors import CapacityExhausted

FOOTPRINT = ResourceSpec(
    cpu_request=6.0, cpu_limit=6.0,
    mem_request_bytes=6 * 1024**3, mem_limit_bytes=6 * 1024**3,
    disk_request_bytes=10 * 1024**3,
)


@dataclass
class _Placement:
    node: Any
    score: float = 1.0
    reservation_id: str = "res-1"


class _Scheduler:
    # image_aware_placement=False short-circuits query_image_presence (the direct
    # no-admission place path) so the fake needs no node/image machinery.
    image_aware_placement = False

    def __init__(self, node: Any) -> None:
        self._node = node
        self.committed: list[Any] = []
        self.released: list[Any] = []
        self.place_kwargs: dict[str, Any] = {}

    def place(self, manifest: Any, **kwargs: Any) -> _Placement:
        self.place_kwargs = kwargs
        return _Placement(node=self._node)

    def commit_placement(self, p: Any) -> None:
        self.committed.append(p)

    def release_placement(self, p: Any) -> None:
        self.released.append(p)


@dataclass
class _Record:
    main_container_id: str
    main_container_name: str
    service_container_ids: dict[str, str]
    project_name: str = "proj"
    project_dir: str = "/tmp/proj"


class _Node:
    def __init__(self, *, fail: bool = False, node_id: str = "node-A") -> None:
        self.node_id = node_id
        self.acquire_calls: list[dict] = []
        self.down_calls: list[dict] = []
        self._fail = fail

    async def acquire_compose_project(self, **kwargs: Any) -> _Record:
        self.acquire_calls.append(kwargs)
        if self._fail:
            raise RuntimeError("compose up failed")
        return _Record(
            main_container_id="cidmainfull",
            main_container_name="proj-main",
            service_container_ids={"main": "cidmainfull", "postgres": "cidpgfull"},
        )

    async def destroy_compose_project(self, **kwargs: Any) -> None:
        self.down_calls.append(kwargs)


class _Resolver:
    async def resolve(self, ref: str) -> str:
        # tag -> digest
        return f"{ref.split(':')[0]}@sha256:deadbeef"


def _coord(node: _Node, *, resolver: bool = True, policy: KwargsPolicy | None = None):
    return RawContainerCoordinator(
        scheduler=_Scheduler(node),  # type: ignore[arg-type]
        digest_resolver=_Resolver() if resolver else None,  # type: ignore[arg-type]
        kwargs_policy=policy,
    )


_COMPOSE = (
    "services:\n"
    "  main: {image: 'reg/ns/tw:main'}\n"
    "  postgres: {image: 'postgres:14'}\n"
)


async def test_acquire_happy_path_creates_session_and_project() -> None:
    node = _Node()
    coord = _coord(node)
    result = await coord.acquire_compose_project(
        compose_yaml=_COMPOSE,
        images=["reg/ns/tw:main", "postgres:14"],
        footprint=FOOTPRINT,
        main_service="main",
    )
    rid = result.rollout_id
    # result shape
    assert result.node_id == "node-A"
    assert result.main_container_id == "cidmainfull"
    assert result.project_name.startswith("xrlenv-")
    assert result.service_container_ids == {
        "main": "cidmainfull", "postgres": "cidpgfull",
    }
    # a main session was created carrying the footprint + the compose marker
    session = coord._sessions[rid]
    assert session.compose_project_name == result.project_name
    assert session.container_id == "cidmainfull"  # full id
    assert session.effective_resources == FOOTPRINT
    # project record tracks the members
    proj = coord._compose_projects[rid]
    assert proj.service_container_ids == {"main": "cidmainfull", "postgres": "cidpgfull"}
    # placement committed, in-flight marker cleared
    assert coord._scheduler.committed  # type: ignore[attr-defined]
    assert rid not in coord._acquiring_ids


async def test_acquire_pins_main_digest_in_compose_and_images() -> None:
    node = _Node()
    coord = _coord(node)  # resolver on → tag becomes digest
    await coord.acquire_compose_project(
        compose_yaml=_COMPOSE,
        images=["reg/ns/tw:main", "postgres:14"],
        footprint=FOOTPRINT,
    )
    sent = node.acquire_calls[0]
    sent_compose = yaml.safe_load(sent["compose_yaml"])
    # main image pinned to the resolved digest in the compose the node runs...
    assert sent_compose["services"]["main"]["image"] == "reg/ns/tw@sha256:deadbeef"
    # ...and the images list pinned to the SAME ref (no tag/digest split)
    assert "reg/ns/tw@sha256:deadbeef" in sent["images"]
    assert "reg/ns/tw:main" not in sent["images"]
    assert "postgres:14" in sent["images"]
    # reserved labels stamped: main=raw, sidecar=compose
    labels_main = sent_compose["services"]["main"]["labels"]
    assert labels_main["xrlenv.session_kind"] == "raw"
    assert labels_main["xrlenv.rollout_id"]  # non-empty
    assert sent_compose["services"]["postgres"]["labels"]["xrlenv.session_kind"] == "compose"
    # place() reserved the whole-stack footprint
    assert coord._scheduler.place_kwargs.get("reserve") == FOOTPRINT  # type: ignore[attr-defined]
    # ...and the direct (no-admission) path scores by main-image affinity +
    # preferred-home (like acquire()), not a bare place.
    assert "image_present" in coord._scheduler.place_kwargs  # type: ignore[attr-defined]
    assert "preferred_home_node" in coord._scheduler.place_kwargs  # type: ignore[attr-defined]


async def test_acquire_vet_rejection_leaves_no_session() -> None:
    node = _Node()
    coord = _coord(node)  # DEFAULT_POLICY → privileged denied
    compose = "services:\n  main: {image: 'app:1', privileged: true}\n"
    with pytest.raises(KwargsPolicyViolation):
        await coord.acquire_compose_project(
            compose_yaml=compose, images=["app:1"], footprint=FOOTPRINT,
        )
    # failed before any placement / node command / row
    assert node.acquire_calls == []
    assert coord._sessions == {}
    assert coord._acquiring_ids == set()


async def test_acquire_node_failure_cleans_up() -> None:
    node = _Node(fail=True)
    coord = _coord(node)
    with pytest.raises(RuntimeError, match="compose up failed"):
        await coord.acquire_compose_project(
            compose_yaml=_COMPOSE, images=["reg/ns/tw:main"], footprint=FOOTPRINT,
        )
    # no session/project left; placement released; in-flight cleared; best-effort down
    assert coord._sessions == {}
    assert coord._compose_projects == {}
    assert coord._acquiring_ids == set()
    assert coord._scheduler.released  # type: ignore[attr-defined]
    assert node.down_calls  # best-effort teardown attempted


async def test_acquire_unconfirmed_teardown_retains_capacity_charge() -> None:
    # audit H10: once the wire `up` was attempted the stack MAY be live. If the acquire fails AND
    # the cleanup down can't be CONFIRMED, capacity must NOT be released into a possibly-live
    # stack — the coordinator retains a GC-reclaimable compose session (iter_load_entries charges
    # it) + commits the reservation, and the raw-GC reaper reclaims it later.
    class _BothFailNode(_Node):
        async def destroy_compose_project(self, **kwargs: Any) -> None:
            self.down_calls.append(kwargs)
            raise RuntimeError("down also failed — teardown UNCONFIRMED")

    node = _BothFailNode(fail=True)
    coord = _coord(node)
    with pytest.raises(RuntimeError, match="compose up failed"):
        await coord.acquire_compose_project(
            compose_yaml=_COMPOSE, images=["reg/ns/tw:main"], footprint=FOOTPRINT,
        )
    # a session + project are RETAINED so capacity stays charged (invariant 2)…
    assert len(coord._sessions) == 1
    (sess,) = coord.list_sessions()
    assert sess.compose_project_name  # a compose session
    assert sess.effective_resources == FOOTPRINT
    assert len(coord._compose_projects) == 1
    assert len(coord.iter_load_entries()) == 1     # still charged to the node
    # …the reservation was COMMITTED (not released) so it converts to the session charge…
    assert coord._scheduler.committed          # type: ignore[attr-defined]
    assert not coord._scheduler.released       # type: ignore[attr-defined]
    assert node.down_calls                      # confirmed-teardown was attempted (and failed)
    assert coord._acquiring_ids == set()


class _RecordingComposeState:
    """Captures raw_rollouts field-sets so a compose test can read the sealed
    status (the default ``_coord`` wires ``state=None``)."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def record_raw_rollout(self, record: Any) -> None:
        self.rows[record.rollout_id] = {"status": record.status}

    def update_raw_rollout(self, rollout_id: str, **fields: Any) -> None:
        self.rows.setdefault(rollout_id, {}).update(fields)


class _CapacityScheduler(_Scheduler):
    """``place()`` always declines with ``CapacityExhausted`` (pool full)."""

    def place(self, manifest: Any, **kwargs: Any) -> Any:
        raise CapacityExhausted(
            "queue_timeout_s=240.0 expired waiting for capacity",
        )


async def test_acquire_capacity_exhausted_seals_capacity_rejected() -> None:
    """A compose acquire the scheduler declines (``CapacityExhausted``) seals
    ``capacity_rejected`` — the same backpressure carve-out as the single-
    container path (spec 13). ``place()`` raised before a node was chosen, so
    no node-side ``down`` runs and no placement is released."""
    node = _Node()
    state = _RecordingComposeState()
    coord = RawContainerCoordinator(
        scheduler=_CapacityScheduler(node),  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
    )
    with pytest.raises(CapacityExhausted):
        await coord.acquire_compose_project(
            compose_yaml=_COMPOSE,
            images=["reg/ns/tw:main", "postgres:14"],
            footprint=FOOTPRINT,
            main_service="main",
        )

    (row,) = state.rows.values()
    assert row["status"] == "capacity_rejected"
    assert "backpressure" in row["error"].lower()
    assert row["finished_at"] > 0.0
    assert node.down_calls == []  # never placed → no compose down
    assert coord._sessions == {}
    assert coord._compose_projects == {}
    assert coord._acquiring_ids == set()


_STATIC_IP_COMPOSE = (
    "services:\n"
    "  main: {image: 'reg/ns/tw:main'}\n"
    "networks:\n"
    "  twnet: {ipam: {config: [{subnet: '172.16.70.0/24'}]}}\n"
)


def _inject_existing_project(
    coord: RawContainerCoordinator, *, node_id: str, subnet: str,
) -> None:
    from xrlenv.control.raw_container_service import _ComposeProjectRecord

    coord._compose_projects["existing"] = _ComposeProjectRecord(
        project_name="existing", node_id=node_id,
        service_container_ids={"main": "x"}, subnet_claims=(subnet,),
    )


async def test_acquire_excludes_node_with_overlapping_subnet() -> None:
    # 3b: a project pinning 172.16.70.0/24 must not co-locate with another already
    # claiming an overlapping subnet — that node is excluded from placement.
    node = _Node()
    coord = _coord(node, resolver=False)
    _inject_existing_project(coord, node_id="node-A", subnet="172.16.70.0/24")
    await coord.acquire_compose_project(
        compose_yaml=_STATIC_IP_COMPOSE, images=["reg/ns/tw:main"],
        footprint=FOOTPRINT,
    )
    exclude = coord._scheduler.place_kwargs.get("exclude_node_ids")  # type: ignore[attr-defined]
    assert exclude == frozenset({"node-A"})
    # the claim is stored on the new project record
    (rec,) = [r for r in coord._compose_projects.values() if r.project_name != "existing"]
    assert rec.subnet_claims == ("172.16.70.0/24",)


async def test_acquire_no_exclusion_for_disjoint_subnet() -> None:
    node = _Node()
    coord = _coord(node, resolver=False)
    _inject_existing_project(coord, node_id="node-A", subnet="internal-ip/8")
    await coord.acquire_compose_project(
        compose_yaml=_STATIC_IP_COMPOSE, images=["reg/ns/tw:main"],
        footprint=FOOTPRINT,
    )
    # disjoint subnet → no exclusion
    assert coord._scheduler.place_kwargs.get("exclude_node_ids") is None  # type: ignore[attr-defined]


async def test_acquire_dns_only_never_excludes() -> None:
    node = _Node()
    coord = _coord(node, resolver=False)
    _inject_existing_project(coord, node_id="node-A", subnet="172.16.70.0/24")
    # a service-DNS-only compose (no pinned subnet) → empty claims → no exclusion
    await coord.acquire_compose_project(
        compose_yaml=_COMPOSE, images=["reg/ns/tw:main"], footprint=FOOTPRINT,
    )
    assert coord._scheduler.place_kwargs.get("exclude_node_ids") is None  # type: ignore[attr-defined]


async def test_acquire_explicit_project_name_is_sanitized() -> None:
    node = _Node()
    coord = _coord(node, resolver=False)
    result = await coord.acquire_compose_project(
        compose_yaml=_COMPOSE, images=["reg/ns/tw:main"], footprint=FOOTPRINT,
        project_name="My_Project/42",
    )
    assert result.project_name == "my_project-42"


async def test_acquire_persists_project_row_deleted_on_destroy() -> None:
    # 3c: the tiny compose-project row persists the footprint + subnet claims (the
    # node can't re-derive them) for CP-restart recovery; deleted on teardown.
    import json

    from xrlenv.control.state import InMemoryStateStore

    node = _Node()
    state = InMemoryStateStore()
    coord = RawContainerCoordinator(
        scheduler=_Scheduler(node),  # type: ignore[arg-type]
        digest_resolver=None, state=state,
    )
    result = await coord.acquire_compose_project(
        compose_yaml=_STATIC_IP_COMPOSE, images=["reg/ns/tw:main"],
        footprint=FOOTPRINT,
    )
    (row,) = state.list_compose_projects()
    assert row.rollout_id == result.rollout_id
    assert row.node_id == "node-A"
    assert json.loads(row.subnet_claims_json) == ["172.16.70.0/24"]
    assert json.loads(row.footprint_json)["cpu_request"] == 6.0
    # confirmed destroy deletes the row
    await coord.destroy_compose_project(rollout_id=result.rollout_id)
    assert state.list_compose_projects() == []


async def test_compose_destroy_converges_on_node_loss_when_wire_fails_after_loss() -> None:
    # audit M14 (compose late-failure convergence): a racing handle_node_lost can pop the session
    # + resolve the single-flight future with NodeLost WHILE the owner's wire compose-down is in
    # flight; the wire then surfaces its own failure. The owner must CONVERGE on NodeLost (like
    # the raw path), not re-raise the now-moot wire error — else joiners see NodeLost and the
    # owner sees the wire failure.
    from xrlenv.errors import NodeLost

    node = _Node(node_id="node-A")
    coord = _coord(node, resolver=False)
    result = await coord.acquire_compose_project(
        compose_yaml=_COMPOSE, images=["reg/ns/tw:main", "postgres:14"], footprint=FOOTPRINT,
    )

    async def _lose_then_fail(**_kwargs: Any) -> None:
        await coord.handle_node_lost("node-A")   # node lost mid-wire (pops + resolves NodeLost)
        raise RuntimeError("compose down failed after the node was already lost")
    node.destroy_compose_project = _lose_then_fail  # type: ignore[method-assign]

    with pytest.raises(NodeLost):                    # converges — NOT RuntimeError
        await coord.destroy_compose_project(rollout_id=result.rollout_id)
    assert coord.list_sessions() == []               # sealed node_lost exactly once


async def test_node_loss_deletes_project_row() -> None:
    from xrlenv.control.state import InMemoryStateStore

    node = _Node(node_id="node-A")
    state = InMemoryStateStore()
    coord = RawContainerCoordinator(
        scheduler=_Scheduler(node),  # type: ignore[arg-type]
        digest_resolver=None, state=state,
    )
    await coord.acquire_compose_project(
        compose_yaml=_STATIC_IP_COMPOSE, images=["reg/ns/tw:main"],
        footprint=FOOTPRINT,
    )
    assert len(state.list_compose_projects()) == 1
    await coord.handle_node_lost("node-A")
    assert state.list_compose_projects() == []


async def test_reap_stale_compose_projects_reclaims_only_dead_rows() -> None:
    # 3c-3: the reclaim backstop. A persisted row with no live _compose_projects
    # entry past the TTL is deleted; a row still within the TTL, or with a live
    # project, is kept; and allow_reclaim=False (grace not elapsed) never reaps.
    from xrlenv.control.state import ComposeProjectStateRecord, InMemoryStateStore

    state = InMemoryStateStore()
    node = _Node()
    coord = RawContainerCoordinator(
        scheduler=_Scheduler(node),  # type: ignore[arg-type]
        digest_resolver=None, state=state,
    )
    # dead+old (no live project, created 1h ago), dead+fresh (created "now"),
    # and live (has a _compose_projects entry, old ts — must survive).
    state.record_compose_project(ComposeProjectStateRecord(
        rollout_id="dead-old", project_name="p1", node_id="node-A",
        footprint_json=FOOTPRINT.model_dump_json(), created_ts=1_000.0,
    ))
    state.record_compose_project(ComposeProjectStateRecord(
        rollout_id="dead-fresh", project_name="p2", node_id="node-A",
        footprint_json=FOOTPRINT.model_dump_json(), created_ts=9_500.0,
    ))
    state.record_compose_project(ComposeProjectStateRecord(
        rollout_id="live", project_name="p3", node_id="node-A",
        footprint_json=FOOTPRINT.model_dump_json(), created_ts=1_000.0,
    ))
    from xrlenv.control.raw_container_service import _ComposeProjectRecord
    coord._compose_projects["live"] = _ComposeProjectRecord(
        project_name="p3", node_id="node-A",
        service_container_ids={"main": "cid"}, subnet_claims=(),
    )

    # grace not elapsed → no reclaim regardless of age
    assert await coord.reap_stale_compose_projects(
        now=10_000.0, reclaim_after_s=3_600.0, allow_reclaim=False,
    ) == 0
    assert len(state.list_compose_projects()) == 3

    # grace elapsed → only dead-old (no live project + past TTL) is reclaimed
    assert await coord.reap_stale_compose_projects(
        now=10_000.0, reclaim_after_s=3_600.0, allow_reclaim=True,
    ) == 1
    remaining = {r.rollout_id for r in state.list_compose_projects()}
    assert remaining == {"dead-fresh", "live"}


async def test_readopt_compose_project_rebuilds_session_and_project() -> None:
    # 3c-3: after a CP restart, re-adopt a live project from its persisted row +
    # the node's main container — rebuilding the session (compose marker +
    # footprint) AND _compose_projects so teardown downs the whole project.
    from xrlenv.control.state import ComposeProjectStateRecord

    node = _Node()
    coord = _coord(node, resolver=False)
    row = ComposeProjectStateRecord(
        rollout_id="r1", project_name="proj", node_id="node-A",
        footprint_json=FOOTPRINT.model_dump_json(),
        subnet_claims_json='["172.16.70.0/24"]',
    )
    assert await coord.readopt_compose_project(row, "cidmainfull", node) is True  # type: ignore[arg-type]
    session = coord._sessions["r1"]
    assert session.compose_project_name == "proj"
    assert session.container_id == "cidmainfull"  # full id from the label sweep
    assert session.effective_resources == FOOTPRINT  # charges the project again
    proj = coord._compose_projects["r1"]
    assert proj.project_name == "proj"
    assert proj.subnet_claims == ("172.16.70.0/24",)  # restored for anti-affinity
    # idempotent — a second readopt on the SAME transport is a no-op but reports SUCCESS (True):
    # readopt-on-connect treats False as "couldn't transfer ownership → fail closed", so an
    # already-adopted project must report True (audit H11).
    assert await coord.readopt_compose_project(row, "cidmainfull", node) is True  # type: ignore[arg-type]
    assert len(coord._sessions) == 1  # not duplicated


class _MultiScheduler:
    """2-node scheduler that honors exclude_node_ids (returns the first
    non-excluded node) — for the concurrent-placement race test."""

    image_aware_placement = False

    def __init__(self, nodes: list[Any]) -> None:
        self._nodes = nodes
        self.committed: list[Any] = []
        self.released: list[Any] = []

    def place(self, manifest: Any, *, exclude_node_ids: Any = None, **kwargs: Any) -> _Placement:
        excl = exclude_node_ids or frozenset()
        for n in self._nodes:
            if n.node_id not in excl:
                return _Placement(node=n)
        raise AssertionError("no free node (both excluded)")

    def commit_placement(self, p: Any) -> None:
        self.committed.append(p)

    def release_placement(self, p: Any) -> None:
        self.released.append(p)


class _MultiAdmission:
    """Admission queue that honors exclude_node_ids, with an await boundary so two
    concurrent acquires genuinely interleave — proving the per-subnet guard (not
    luck) serializes them."""

    def __init__(self, nodes: list[Any]) -> None:
        self._nodes = nodes

    async def acquire(self, *, exclude_node_ids: Any = None, **kwargs: Any) -> _Placement:
        import asyncio

        await asyncio.sleep(0)  # yield: without the guard both would see empty excl
        excl = exclude_node_ids or frozenset()
        for n in self._nodes:
            if n.node_id not in excl:
                return _Placement(node=n)
        raise AssertionError("no free node (both excluded)")

    def kick(self) -> None:
        pass


async def _assert_concurrent_land_on_different_nodes(coord: RawContainerCoordinator) -> None:
    import asyncio

    r1, r2 = await asyncio.gather(
        coord.acquire_compose_project(
            compose_yaml=_STATIC_IP_COMPOSE, images=["reg/ns/tw:main"],
            footprint=FOOTPRINT,
        ),
        coord.acquire_compose_project(
            compose_yaml=_STATIC_IP_COMPOSE, images=["reg/ns/tw:main"],
            footprint=FOOTPRINT,
        ),
    )
    # different nodes — the same-subnet collision was avoided
    assert {r1.node_id, r2.node_id} == {"node-A", "node-B"}
    # both committed as projects; no pending reservations leaked
    assert len(coord._compose_projects) == 2
    assert coord._pending_subnet_claims == {}


async def test_concurrent_same_subnet_direct_path_different_nodes() -> None:
    # P2 audit (direct path): the per-subnet guard + pending reservation prevent
    # two same-subnet acquires from both picking the same node.
    node_a, node_b = _Node(node_id="node-A"), _Node(node_id="node-B")
    coord = RawContainerCoordinator(
        scheduler=_MultiScheduler([node_a, node_b]),  # type: ignore[arg-type]
        digest_resolver=None,
    )
    await _assert_concurrent_land_on_different_nodes(coord)


async def test_concurrent_same_subnet_admission_path_different_nodes() -> None:
    # P2 audit (admission path): the per-subnet guard serializes exclude→admit→
    # reserve, so the admission queue can't fast-path both onto the same node.
    node_a, node_b = _Node(node_id="node-A"), _Node(node_id="node-B")
    coord = RawContainerCoordinator(
        scheduler=_MultiScheduler([node_a, node_b]),  # type: ignore[arg-type]
        admission=_MultiAdmission([node_a, node_b]),  # type: ignore[arg-type]
        digest_resolver=None,
    )
    await _assert_concurrent_land_on_different_nodes(coord)
