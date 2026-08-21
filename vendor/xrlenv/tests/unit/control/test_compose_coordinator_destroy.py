"""Coordinator compose-project teardown + reap routing (multi-service step 3a-3b-i).

A compose ``main`` session is modeled in ``_sessions`` (carrying the footprint), so
capacity + deadline/liveness sweeps cover it for free; its member ids live in
``_compose_projects``. Teardown must route to the node's **strict**
``destroy_compose_project`` (down the whole project) — a failed down RETAINS the
session + capacity (invariant 2). The reap path (``destroy(container_id=main)``)
routes there too. No docker; a fake node transport records the calls.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.raw_container_service import (
    RawContainerCoordinator,
    RawContainerSession,
    _ComposeProjectRecord,
)
from xrlenv.errors import XRLEnvError


class _Scheduler:
    """Bare scheduler stub — the teardown path never calls it."""


class _FakeNode:
    node_id = "node-A"

    def __init__(self, *, fail_down: bool = False) -> None:
        self.down_calls: list[dict] = []
        self.destroy_container_calls: list[dict] = []
        self._fail = fail_down

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str, force: bool = True,
    ) -> None:
        self.down_calls.append(
            {"rollout_id": rollout_id, "project_name": project_name, "force": force},
        )
        if self._fail:
            raise RuntimeError("docker compose down failed")

    async def destroy_container(self, **kwargs: Any) -> None:
        self.destroy_container_calls.append(kwargs)


def _coord() -> RawContainerCoordinator:
    return RawContainerCoordinator(scheduler=_Scheduler())  # type: ignore[arg-type]


def _inject(
    coord: RawContainerCoordinator, node: _FakeNode,
    *, rollout_id: str = "r1", project: str = "proj", main_id: str = "cidmain",
) -> None:
    coord._sessions[rollout_id] = RawContainerSession(
        rollout_id=rollout_id,
        node=node,  # type: ignore[arg-type]
        node_id=node.node_id,
        container_id=main_id,
        container_name="proj-main",
        image="reg/ns/tw@sha256:abc",
        created_at=_dt.datetime.now(_dt.UTC),
        effective_resources=ResourceSpec(
            cpu_request=6.0, cpu_limit=6.0,
            mem_request_bytes=6 * 1024**3, mem_limit_bytes=6 * 1024**3,
            disk_request_bytes=10 * 1024**3,
        ),
        compose_project_name=project,
    )
    coord._compose_projects[rollout_id] = _ComposeProjectRecord(
        project_name=project,
        node_id=node.node_id,
        service_container_ids={"main": main_id, "postgres": "cidpg"},
    )


async def test_destroy_compose_project_downs_and_frees_on_success() -> None:
    coord = _coord()
    node = _FakeNode()
    _inject(coord, node)
    await coord.destroy_compose_project(rollout_id="r1", project_name="proj")
    # the WHOLE project was downed (not a single-container destroy)
    assert node.down_calls == [
        {"rollout_id": "r1", "project_name": "proj", "force": True},
    ]
    assert node.destroy_container_calls == []
    # session + project record dropped → capacity freed
    assert "r1" not in coord._sessions
    assert "r1" not in coord._compose_projects


async def test_failed_down_retains_session_and_capacity() -> None:
    # invariant 2: a failed / unconfirmed down must NOT free capacity.
    coord = _coord()
    node = _FakeNode(fail_down=True)
    _inject(coord, node)
    with pytest.raises(RuntimeError, match="down failed"):
        await coord.destroy_compose_project(rollout_id="r1", project_name="proj")
    # session + project record RETAINED for a retry
    assert "r1" in coord._sessions
    assert "r1" in coord._compose_projects
    # and it's no longer marked destroying (so a retry can proceed)
    assert "r1" not in coord._destroying


async def test_destroy_routes_compose_session_to_project_down() -> None:
    # the reap path calls destroy(rollout_id, container_id=main) — it must route
    # to the whole-project down, never a single-container destroy.
    coord = _coord()
    node = _FakeNode()
    _inject(coord, node)
    await coord.destroy(rollout_id="r1", container_id="cidmain", reason="deadline")
    assert node.down_calls and node.down_calls[0]["project_name"] == "proj"
    assert node.destroy_container_calls == []
    assert "r1" not in coord._sessions


async def test_destroy_compose_project_unknown_rollout_raises() -> None:
    coord = _coord()
    with pytest.raises(XRLEnvError, match="not an active compose project"):
        await coord.destroy_compose_project(rollout_id="ghost")


async def test_destroy_compose_project_wrong_project_name_raises() -> None:
    coord = _coord()
    node = _FakeNode()
    _inject(coord, node, project="proj")
    with pytest.raises(XRLEnvError, match="owns project"):
        await coord.destroy_compose_project(rollout_id="r1", project_name="other")
    assert node.down_calls == []  # nothing torn down on the rejected call


async def test_node_loss_drops_compose_project_metadata() -> None:
    # P2 audit: a lost node takes its whole compose project — no stale
    # _compose_projects row may linger (3b/3c read it for subnet anti-affinity +
    # reconcile; a leftover could falsely block placement).
    coord = _coord()
    node = _FakeNode()
    _inject(coord, node, rollout_id="r1", project="proj")
    assert "r1" in coord._compose_projects
    reaped = await coord.handle_node_lost("node-A")
    assert reaped == 1
    assert "r1" not in coord._sessions
    assert "r1" not in coord._compose_projects


async def test_is_compose_project() -> None:
    coord = _coord()
    node = _FakeNode()
    _inject(coord, node, rollout_id="r1")
    assert coord.is_compose_project("r1") is True
    # a plain single-container session is NOT a compose project
    coord._sessions["r2"] = RawContainerSession(
        rollout_id="r2", node=node,  # type: ignore[arg-type]
        node_id=node.node_id, container_id="c2", container_name="c2",
        image="img", created_at=_dt.datetime.now(_dt.UTC),
    )
    assert coord.is_compose_project("r2") is False
    assert coord.is_compose_project("ghost") is False


async def test_destroy_compose_orphan_downs_whole_project_and_frees() -> None:
    # audit H10: a coordinator-only compose orphan (main gone, sidecars off the raw diff) must
    # down the WHOLE project (node-confirmed) — freeing capacity only then, never a bare seal
    # that would release the aggregate footprint while sidecars still run.
    coord = _coord()
    node = _FakeNode()
    _inject(coord, node)
    await coord.destroy_compose_orphan(rollout_id="r1", reason="disk-guard")
    assert node.down_calls and node.down_calls[0]["project_name"] == "proj"
    assert node.destroy_container_calls == []      # whole-project down, not single-container
    assert "r1" not in coord._sessions
    assert "r1" not in coord._compose_projects


async def test_destroy_compose_orphan_failed_down_retains_capacity() -> None:
    # audit H10 + invariant 2: a failed whole-project down RETAINS the session + aggregate
    # capacity (sidecars may still be running) and re-raises so the reconciler retries.
    coord = _coord()
    node = _FakeNode(fail_down=True)
    _inject(coord, node)
    with pytest.raises(RuntimeError, match="down failed"):
        await coord.destroy_compose_orphan(rollout_id="r1")
    assert "r1" in coord._sessions
    assert "r1" in coord._compose_projects
    assert "r1" not in coord._destroying          # cleared so a retry can proceed


async def test_destroy_compose_orphan_noop_when_already_finalized() -> None:
    # generation-safe: if a concurrent destroy already finalized the session, no down is issued.
    coord = _coord()
    node = _FakeNode()
    await coord.destroy_compose_orphan(rollout_id="gone")
    assert node.down_calls == []


async def test_destroy_compose_orphan_rejects_non_compose() -> None:
    coord = _coord()
    node = _FakeNode()
    coord._sessions["r2"] = RawContainerSession(
        rollout_id="r2", node=node,  # type: ignore[arg-type]
        node_id=node.node_id, container_id="c2", container_name="c2",
        image="img", created_at=_dt.datetime.now(_dt.UTC),
    )
    with pytest.raises(XRLEnvError, match="not a compose project"):
        await coord.destroy_compose_orphan(rollout_id="r2")
    assert node.down_calls == []


async def test_compose_destroy_skips_stale_session_no_second_down() -> None:
    # audit M14: destroy_compose_orphan snapshots the session BEFORE single-flight; if a
    # concurrent finalize already removed it, _destroy_compose_session must re-check under the
    # lock and NOT issue a second wire down for the already-gone project.
    coord = _coord()
    node = _FakeNode()
    _inject(coord, node)
    stale = coord._sessions["r1"]
    # a concurrent finalize already tore it down + dropped the session/project record:
    coord._sessions.pop("r1", None)
    coord._compose_projects.pop("r1", None)
    await coord._destroy_compose_session(stale, force=True, reason=None)
    assert node.down_calls == []            # NO second wire down
    assert "r1" not in coord._destroying    # no stranded single-flight marker


async def test_single_container_destroy_unaffected() -> None:
    # a non-compose session takes the ordinary single-container destroy path.
    coord = _coord()
    node = _FakeNode()
    coord._sessions["r2"] = RawContainerSession(
        rollout_id="r2", node=node,  # type: ignore[arg-type]
        node_id=node.node_id, container_id="c2", container_name="c2",
        image="img", created_at=_dt.datetime.now(_dt.UTC),
    )
    await coord.destroy(rollout_id="r2", container_id="c2")
    assert node.destroy_container_calls  # single-container path
    assert node.down_calls == []
    assert "r2" not in coord._sessions
