"""Integration test: heartbeat → NodeRegistry watchdog → coordinator seals
in-flight rollouts as failed/node_lost.

Drives the coordinator side end-to-end with a fake gRPC transport so we
exercise the same handle_node_lost code path the distributed runtime uses,
without standing up a real gRPC server.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

from xrlenv.backends.base import (
    ExecResult,
    ResourceSpec,
    ResourceUsage,
    SandboxHandle,
)
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.node_registry import NodeRegistry
from xrlenv.control.scheduler import Placement
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.types import RolloutStatus


def _manifest() -> TemplateManifest:
    return TemplateManifest(
        name="t",
        version="0.1",
        digest="sha256:t",
        image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="6.0.0", platform="linux",
    )


class _FakeRemoteTransport:
    """Minimal NodeTransport stand-in that mimics RemoteNodeTransport's
    last_heartbeat_at attribute so the registry can mark it dead.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.last_heartbeat_at = time.monotonic()
        self.created = 0
        self.destroyed = 0

    def supported_backends(self) -> list[str]:
        return ["docker"]

    def hardware(self) -> HardwareInfo:
        return _hw()

    async def create_sandbox(self, **_: Any) -> SandboxHandle:
        self.created += 1
        return SandboxHandle(
            id=f"sb-{self.node_id}-{self.created}",
            backend="docker",
            backend_ref=f"cid-{self.node_id}-{self.created}",
            stub_endpoint="tcp://127.0.0.1:0",
        )

    async def destroy_sandbox(self, _sb: SandboxHandle) -> None:
        self.destroyed += 1

    async def env_setup(
        self, _sb: SandboxHandle, *, adapter_module: str, adapter_class: str,
        init_params: dict[str, Any], **_kw: Any,
    ) -> dict[str, Any]:
        return {"obs": "first"}

    async def env_step(
        self, _sb: SandboxHandle, _action: Any, **_kw: Any,
    ) -> dict[str, Any]:
        return {"obs": {}, "reward": 0.0, "done": False, "truncated": False, "info": {}}

    async def env_teardown(
        self, _sb: SandboxHandle, **_kw: Any,
    ) -> dict[str, Any]:
        return {"status": "ok"}

    async def run_in_sandbox(
        self, _sb: SandboxHandle, cmd: list[str], **_: Any
    ) -> ExecResult:
        return ExecResult(exit_code=0)

    async def stats(self, _sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=0.0, rss_bytes=0, disk_bytes=0, rx_bytes=0, tx_bytes=0)

    async def query_image(self, _image: str) -> Any:
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _build_coord(node: _FakeRemoteTransport) -> tuple[RolloutCoordinator, InMemoryStateStore]:
    catalog = TemplateCatalog()
    catalog.register(_manifest())
    sched = MagicMock()
    sched.place.return_value = Placement(node=node, backend="docker", score=1)  # type: ignore[arg-type]
    sched.nodes = [node]
    state = InMemoryStateStore()
    coord = RolloutCoordinator(catalog=catalog, scheduler=sched, state=state)
    return coord, state


# ──────────────────────────────────────────────────────────────────────────────
# coordinator.handle_node_lost
# ──────────────────────────────────────────────────────────────────────────────


async def test_handle_node_lost_seals_running_rollouts_as_failed() -> None:
    node = _FakeRemoteTransport("n1")
    coord, state = _build_coord(node)
    rid, _obs = await coord.start_rollout(template_name="t", init={})

    # Sanity: rollout is RUNNING and bound to node n1.
    record = state.get_rollout(rid)
    assert record.status == RolloutStatus.RUNNING
    assert record.node_id == "n1"

    await coord.handle_node_lost("n1")

    sealed = state.get_rollout(rid)
    assert sealed.status == RolloutStatus.FAILED
    assert sealed.reason == "node_lost"
    # Sandbox row was dropped (node-confirmed-destroy invariant relaxed for
    # node-loss: we can't talk to the node, so we declare its sandboxes gone).
    assert state.list_sandboxes() == []


async def test_handle_node_lost_skips_terminal_rollouts() -> None:
    node = _FakeRemoteTransport("n1")
    coord, state = _build_coord(node)
    rid, _obs = await coord.start_rollout(template_name="t", init={})
    await coord.finish(rid)
    finished_status = state.get_rollout(rid).status

    await coord.handle_node_lost("n1")

    # Already-terminal rollout's status must not be flipped to FAILED.
    assert state.get_rollout(rid).status == finished_status


async def test_handle_node_lost_does_not_touch_other_nodes_rollouts() -> None:
    n1 = _FakeRemoteTransport("n1")
    n2 = _FakeRemoteTransport("n2")
    catalog = TemplateCatalog()
    catalog.register(_manifest())

    sched = MagicMock()
    sched.place.side_effect = [
        Placement(node=n1, backend="docker", score=1),
        Placement(node=n2, backend="docker", score=1),
    ]
    sched.nodes = [n1, n2]
    state = InMemoryStateStore()
    coord = RolloutCoordinator(catalog=catalog, scheduler=sched, state=state)

    rid_a, _ = await coord.start_rollout(template_name="t", init={})
    rid_b, _ = await coord.start_rollout(template_name="t", init={})

    await coord.handle_node_lost("n1")

    a = state.get_rollout(rid_a)
    b = state.get_rollout(rid_b)
    assert a.status == RolloutStatus.FAILED
    assert a.reason == "node_lost"
    assert b.status == RolloutStatus.RUNNING


# ──────────────────────────────────────────────────────────────────────────────
# Registry → coordinator end-to-end (real watchdog, fake transport)
# ──────────────────────────────────────────────────────────────────────────────


async def test_registry_watchdog_invokes_coordinator_handler() -> None:
    node = _FakeRemoteTransport("n-stale")
    # Fast-forward heartbeat so it's already dead.
    node.last_heartbeat_at = time.monotonic() - 10.0

    coord, state = _build_coord(node)
    rid, _ = await coord.start_rollout(template_name="t", init={})

    seen: list[str] = []

    async def handler(node_id: str) -> None:
        seen.append(node_id)
        await coord.handle_node_lost(node_id)

    registry = NodeRegistry(
        on_node_lost=handler,
        disconnect_grace_s=0.05,
        check_interval_s=0.02,
    )
    registry.register(node)
    await registry.start()
    try:
        await asyncio.sleep(0.2)
    finally:
        await registry.shutdown()

    assert seen == ["n-stale"]
    assert state.get_rollout(rid).status == RolloutStatus.FAILED
    assert state.get_rollout(rid).reason == "node_lost"


async def test_registry_heartbeat_refreshes_state_last_seen_at() -> None:
    """The registry installs a per-heartbeat callback on the transport
    that mirrors ``transport.touch()`` into ``state.update_node_seen``.
    Without this hook, ``nodes.last_seen_at`` was frozen at register
    time and ``xrlenv nodes`` showed ever-growing "Xm ago" while the
    in-memory watchdog (which reads ``transport.last_heartbeat_at``)
    correctly held status=connected. Pin the wiring so that bug
    doesn't regress.
    """

    class _HBTransport:
        """Minimal stand-in: only the bits NodeRegistry touches at
        register time + the new ``set_on_heartbeat`` setter +
        ``touch()`` semantics."""

        def __init__(self, node_id: str) -> None:
            self.node_id = node_id
            self.last_heartbeat_at = time.monotonic()
            self.backends = ["docker"]
            self.stream_epoch = "ep-1"
            self.control_instance_id = "inst-1"
            self._cb: Any = None

        def set_on_heartbeat(self, cb: Any) -> None:
            self._cb = cb

        def touch(self) -> None:
            self.last_heartbeat_at = time.monotonic()
            if self._cb is not None:
                self._cb(self.node_id)

    node = _HBTransport("n-hb")
    state = InMemoryStateStore()

    async def _no_op(_node_id: str) -> None:
        return None

    registry = NodeRegistry(on_node_lost=_no_op, state=state)
    registry.register(node)

    # State row exists at register time (seeded by record_node_connected).
    seeded_ts = state.list_nodes()[0].last_seen_at
    assert seeded_ts > 0

    # Simulate a heartbeat arriving after some wall-clock has passed.
    # Sleep 10 ms; the live timestamp must move forward.
    await asyncio.sleep(0.01)
    node.touch()

    refreshed_ts = state.list_nodes()[0].last_seen_at
    assert refreshed_ts > seeded_ts

    # After deregister the callback is dropped — a stray late
    # heartbeat (rare but possible during teardown) doesn't write
    # to state after disconnect.
    registry.deregister("n-hb")
    pre_stray_ts = state.list_nodes()[0].last_seen_at  # set by record_node_disconnected
    await asyncio.sleep(0.01)
    node.touch()  # would be a stray heartbeat; callback must be a no-op
    assert state.list_nodes()[0].last_seen_at == pre_stray_ts


async def test_registry_mirrors_node_health_to_state() -> None:
    """Stage 1: the registry's per-heartbeat callback mirrors the
    transport's ``health_json`` into ``state.list_node_health()`` so the
    admin "Cluster health" page can read it out-of-process."""

    class _HealthTransport:
        """Minimal transport stand-in carrying a Stage-1 health blob."""

        node_id = "n-health"
        stream_epoch = "ep-1"
        control_instance_id = "inst-1"
        health_json = '{"create_p95_ms": 99.0, "create_count": 5}'

        def __init__(self) -> None:
            self.backends = ["docker"]
            self.last_heartbeat_at = time.monotonic()
            self._cb: Any = None

        def set_on_heartbeat(self, cb: Any) -> None:
            self._cb = cb

        def touch(self) -> None:
            if self._cb is not None:
                self._cb(self.node_id)

    node = _HealthTransport()
    state = InMemoryStateStore()

    async def _no_op(_node_id: str) -> None:
        return None

    registry = NodeRegistry(on_node_lost=_no_op, state=state)
    registry.register(node)
    # No health mirrored until a heartbeat fires the callback.
    assert state.list_node_health() == {}

    node.touch()
    assert state.list_node_health() == {
        "n-health": '{"create_p95_ms": 99.0, "create_count": 5}',
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shutdown-race guard: closed state store
# ──────────────────────────────────────────────────────────────────────────────


async def test_handle_node_lost_tolerates_closed_state_store(
    tmp_path: Any, caplog: Any,
) -> None:
    """Operator-reported regression (2026-05-04): during
    ``DistributedRuntime.shutdown`` the gRPC stop unwound bidi streams
    which spawned ``handle_node_lost`` tasks; ``state.close()`` then
    ran before those tasks finished their ``list_rollouts`` query, so
    the shutdown logs ended with a noisy ``sqlite3.ProgrammingError:
    Cannot operate on a closed database`` per attached node. The seal
    path now catches that one specific signal and logs an info-level
    "skipping" line instead of crashing the task.
    """
    import logging

    from xrlenv.control.state import SqliteStateStore

    node = _FakeRemoteTransport("n1")
    catalog = TemplateCatalog()
    catalog.register(_manifest())
    sched = MagicMock()
    sched.place.return_value = Placement(node=node, backend="docker", score=1)  # type: ignore[arg-type]
    sched.nodes = [node]
    state = SqliteStateStore(tmp_path / "state.db")
    coord = RolloutCoordinator(catalog=catalog, scheduler=sched, state=state)

    # Close the store under the coordinator's feet — exactly the
    # ordering the pre-fix shutdown path produced.
    state.close()

    caplog.set_level(logging.INFO, logger="xrlenv.control.coordinator")
    # Must NOT raise.
    await coord.handle_node_lost("n1")

    assert any(
        "state store closed" in r.message and "n1" in r.message
        for r in caplog.records
    ), "expected an info-level 'skipping' log when the store was closed"


async def _noop_loss(node_id: str, transport: object = None) -> None:
    return None


def test_deregister_is_identity_conditional() -> None:
    # audit H11: deregister(expected=...) only removes if that transport is STILL current, so a
    # stale stream can't evict a reconnected replacement under the same node id.
    registry = NodeRegistry(on_node_lost=_noop_loss)
    old = _FakeRemoteTransport("n-x")
    registry.register(old)
    new = _FakeRemoteTransport("n-x")
    registry.register(new)                                  # replaces `old`
    assert registry.deregister("n-x", expected=old) is False   # stale gen → no-op
    assert registry.get("n-x") is new                          # replacement intact
    assert registry.deregister("n-x", expected=new) is True     # current gen → removed
    assert registry.get("n-x") is None


async def test_watchdog_skips_replacement_reconnected_during_sweep() -> None:
    # audit H11: while the sweep awaits node A's loss handler, node B reconnects under the same
    # id. The snapshot's STALE node-B entry must not tear down the reconnected replacement.
    registry = NodeRegistry(
        on_node_lost=_noop_loss, disconnect_grace_s=0.05, check_interval_s=999.0,
    )
    t_a = _FakeRemoteTransport("A")
    t_a.last_heartbeat_at = time.monotonic() - 10.0        # stale
    t_b_old = _FakeRemoteTransport("B")
    t_b_old.last_heartbeat_at = time.monotonic() - 10.0    # stale
    registry.register(t_a)
    registry.register(t_b_old)
    t_b_new = _FakeRemoteTransport("B")                    # fresh replacement, same id

    lost: list[tuple[str, object]] = []

    async def handler(node_id: str, transport: object = None) -> None:
        lost.append((node_id, transport))
        if node_id == "A":
            registry.register(t_b_new)                     # B reconnects mid-sweep
    registry._on_node_lost = handler  # type: ignore[assignment]
    registry._loss_handler_wants_transport = True

    await registry._sweep()

    # A torn down; B's stale generation skipped, replacement preserved + not sealed.
    assert ("A", t_a) in lost
    assert all(nid != "B" for nid, _ in lost)
    assert registry.get("B") is t_b_new
