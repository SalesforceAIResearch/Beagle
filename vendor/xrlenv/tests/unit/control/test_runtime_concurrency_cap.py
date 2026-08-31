"""Per-node per-runtime concurrency cap — sysbox-fs wedge prevention.

notes/design-per-node-runtime-concurrency-cap.md. The scheduler counts running
(raw-session provider) + in-flight (`_pending`) containers of a requested
runtime per node and refuses to place past the node's cap; overflow rides the
existing admission-queue hold/kick path. Uncapped runtimes (runc / None / no
nodes.yaml entry) are byte-for-byte unchanged.
"""

from __future__ import annotations

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.scheduler import RawSessionLoad, Scheduler
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import TemplateCatalog
from xrlenv.errors import CapacityExhausted

from tests.unit.control.test_scheduler import _manifest, _node

_RT = "sysbox-runc"


def _sysbox_node(node_id: str) -> object:
    """A MagicMock node that advertises sysbox-runc (so the §5.3 runtime filter
    lets a sysbox acquire through to the cap gate)."""
    n = _node(node_id)
    n.supported_runtimes.return_value = ["runc", _RT]
    return n


def _sched(
    *nodes: object,
    caps: dict[str, dict[str, int]] | None = None,
    sessions: list[RawSessionLoad] | None = None,
) -> Scheduler:
    s = Scheduler(
        list(nodes), catalog=TemplateCatalog(), state=InMemoryStateStore(),
        runtime_caps=caps,
    )
    if sessions is not None:
        s.set_raw_session_provider(lambda: list(sessions))
    return s


def _running(node_id: str, runtime: str | None = _RT) -> RawSessionLoad:
    return RawSessionLoad(
        node_id=node_id,
        template_name="raw-container/img",
        effective_resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=0.25,
            mem_request_bytes=1 << 30, mem_limit_bytes=1 << 30,
            disk_request_bytes=1 << 30,
        ),
        task_key=None,
        container_runtime=runtime,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. At-cap excludes the node
# ──────────────────────────────────────────────────────────────────────────────


def test_running_sessions_at_cap_exclude_node() -> None:
    a = _sysbox_node("node-A")
    # A holds 2 running sysbox sessions; cap is 2 → A is full.
    sched = _sched(
        a, caps={"node-A": {_RT: 2}},
        sessions=[_running("node-A"), _running("node-A")],
    )
    with pytest.raises(CapacityExhausted, match="concurrency cap"):
        sched.place(_manifest(), backend="docker", container_runtime=_RT)


def test_at_cap_steers_to_sibling_under_cap() -> None:
    a, b = _sysbox_node("node-A"), _sysbox_node("node-B")
    sched = _sched(
        a, b, caps={"node-A": {_RT: 2}, "node-B": {_RT: 2}},
        sessions=[_running("node-A"), _running("node-A")],  # A full, B empty
    )
    p = sched.place(_manifest(), backend="docker", container_runtime=_RT)
    assert p.node.node_id == "node-B"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pending counts toward the cap (over-place race guard)
# ──────────────────────────────────────────────────────────────────────────────


def test_pending_placements_count_toward_cap() -> None:
    a = _sysbox_node("node-A")
    sched = _sched(a, caps={"node-A": {_RT: 2}}, sessions=[])
    # Two placements with NO session registered yet → 2 in-flight _pending.
    sched.place(_manifest(), backend="docker", container_runtime=_RT)
    sched.place(_manifest(), backend="docker", container_runtime=_RT)
    # The 3rd must be refused on _pending alone (the race the cap must close).
    with pytest.raises(CapacityExhausted, match="concurrency cap"):
        sched.place(_manifest(), backend="docker", container_runtime=_RT)


def test_releasing_pending_frees_a_slot() -> None:
    a = _sysbox_node("node-A")
    sched = _sched(a, caps={"node-A": {_RT: 1}}, sessions=[])
    p = sched.place(_manifest(), backend="docker", container_runtime=_RT)
    with pytest.raises(CapacityExhausted):
        sched.place(_manifest(), backend="docker", container_runtime=_RT)
    # Abandon the first placement → its _pending slot frees → next place lands.
    sched.release_placement(p)
    p2 = sched.place(_manifest(), backend="docker", container_runtime=_RT)
    assert p2.node.node_id == "node-A"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Overflow queues then drains (end-to-end via the admission queue)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overflow_holds_in_queue_then_drains_on_slot_free() -> None:
    a = _sysbox_node("node-A")
    live: list[RawSessionLoad] = [_running("node-A")]  # A at cap 1
    sched = _sched(a, caps={"node-A": {_RT: 1}})
    sched.set_raw_session_provider(lambda: list(live))
    state = InMemoryStateStore()
    q = AdmissionQueue(scheduler=sched, state=state, poll_interval_s=0.05)
    await q.start()
    try:
        import asyncio

        async def free_slot_soon() -> None:
            await asyncio.sleep(0.05)
            live.clear()          # the running sysbox container is destroyed…
            q.kick()              # …and the destroy kicks the queue.

        freer = asyncio.create_task(free_slot_soon())
        placement = await q.acquire(
            manifest=_manifest(), timeout_s=2.0, container_runtime=_RT,
        )
        await freer
        assert placement.node.node_id == "node-A"
        assert state.list_pending() == []  # drained, not leaked
    finally:
        await q.stop()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Non-sysbox placement is unaffected
# ──────────────────────────────────────────────────────────────────────────────


def test_runc_placement_ignores_sysbox_cap() -> None:
    a = _sysbox_node("node-A")
    # A holds 5 running RUNC containers; the sysbox cap is 1. A runc/None
    # acquire must NOT be gated by the sysbox-runc cap.
    sched = _sched(
        a, caps={"node-A": {_RT: 1}},
        sessions=[_running("node-A", runtime="runc") for _ in range(5)],
    )
    p = sched.place(_manifest(), backend="docker")  # runtime None
    assert p.node.node_id == "node-A"


def test_uncapped_node_is_unlimited() -> None:
    a = _sysbox_node("node-A")  # no caps at all
    sched = _sched(
        a, caps=None,
        sessions=[_running("node-A") for _ in range(20)],
    )
    p = sched.place(_manifest(), backend="docker", container_runtime=_RT)
    assert p.node.node_id == "node-A"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Multi-node spread up to NxM before overflow
# ──────────────────────────────────────────────────────────────────────────────


def test_spreads_across_nodes_up_to_cap_then_overflows() -> None:
    a, b = _sysbox_node("node-A"), _sysbox_node("node-B")
    sched = _sched(a, b, caps={"node-A": {_RT: 2}, "node-B": {_RT: 2}}, sessions=[])
    placed = []
    for _ in range(4):  # 2 nodes x cap 2 = 4 slots
        placed.append(
            sched.place(_manifest(), backend="docker", container_runtime=_RT)
            .node.node_id
        )
    assert placed.count("node-A") == 2
    assert placed.count("node-B") == 2
    # 5th exhausts every node's cap (via _pending) → overflow.
    with pytest.raises(CapacityExhausted, match="concurrency cap"):
        sched.place(_manifest(), backend="docker", container_runtime=_RT)


# ──────────────────────────────────────────────────────────────────────────────
# 6. nodes.yaml plumb-through
# ──────────────────────────────────────────────────────────────────────────────


def test_nodes_yaml_parses_and_plumbs_the_cap(tmp_path: object) -> None:
    from pathlib import Path

    from xrlenv.control.nodes_yaml import load_nodes_yaml

    p = Path(str(tmp_path)) / "nodes.yaml"
    p.write_text(
        "version: 1\n"
        "nodes:\n"
        "  - id: aws-sbx\n"
        "    backends: [docker]\n"
        "    sysbox: true\n"
        "    max_concurrent_by_runtime:\n"
        "      sysbox-runc: 8\n"
    )
    inv = load_nodes_yaml(p)
    entry = inv.by_id()["aws-sbx"]
    assert entry.max_concurrent_by_runtime == {"sysbox-runc": 8}
    # And the distributed-runtime shape: {node_id: {runtime: cap}}.
    caps = {
        n.id: dict(n.max_concurrent_by_runtime)
        for n in inv.nodes
        if n.max_concurrent_by_runtime
    }
    assert caps == {"aws-sbx": {"sysbox-runc": 8}}
    # A node without the field contributes nothing (unlimited).
    assert load_nodes_yaml(p).by_id()["aws-sbx"].max_concurrent_by_runtime


def test_default_empty_cap_map_is_unlimited() -> None:
    from xrlenv.control.nodes_yaml import NodeEntry

    e = NodeEntry(id="n1")
    assert e.max_concurrent_by_runtime == {}
