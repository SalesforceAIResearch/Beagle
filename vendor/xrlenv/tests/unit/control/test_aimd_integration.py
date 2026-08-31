"""Stage 3 — AIMD controller wired into the scheduler + control loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.aimd_loop import AimdControlLoop
from xrlenv.control.capacity import AimdConfig, HealthAimdController
from xrlenv.control.scheduler import Scheduler
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.errors import CapacityExhausted
from xrlenv.node.hw_probe import HardwareInfo


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=8, mem_bytes=32 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="6.0.0", platform="linux",
    )


def _node(node_id: str) -> Any:
    n = MagicMock()
    n.node_id = node_id
    n.supported_backends.return_value = ["docker"]
    n.hardware.return_value = _hw()
    return n


def _manifest() -> TemplateManifest:
    return TemplateManifest(
        name="t", version="0.1", digest="sha256:t", image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _scheduler(node: Any, controller: HealthAimdController | None) -> Scheduler:
    catalog = TemplateCatalog()
    catalog.register(_manifest())
    return Scheduler(
        [node], catalog=catalog, state=InMemoryStateStore(),
        aimd_controller=controller,
    )


# ── scheduler enforcement ─────────────────────────────────────────────────────


def test_scheduler_excludes_node_at_adaptive_limit() -> None:
    """A node at its AIMD limit is filtered out of placement; with the
    sole node over-limit, `place` raises CapacityExhausted so the
    request queues."""
    # initial_limit=0 → every node is "at limit" (load 0 >= limit 0).
    controller = HealthAimdController(AimdConfig(initial_limit=0))
    sched = _scheduler(_node("node-A"), controller)

    with pytest.raises(CapacityExhausted, match="adaptive admission limit"):
        sched.place(_manifest(), backend="docker")


def test_scheduler_places_when_under_adaptive_limit() -> None:
    """A generous AIMD limit does not exclude an otherwise-eligible
    node — placement proceeds normally."""
    controller = HealthAimdController(AimdConfig(initial_limit=8))
    sched = _scheduler(_node("node-A"), controller)

    placement = sched.place(_manifest(), backend="docker")
    assert placement.node.node_id == "node-A"


def test_node_load_snapshot_counts_per_node() -> None:
    """`node_load_snapshot` returns one entry per node — 0 on an idle
    cluster with no sandboxes / raw sessions."""
    sched = _scheduler(_node("node-A"), None)
    assert sched.node_load_snapshot() == {"node-A": 0}


# ── control loop ──────────────────────────────────────────────────────────────


class _FakeTransport:
    def __init__(self, health: dict[str, Any] | None) -> None:
        self._last_health = health


class _FakeRegistry:
    def __init__(self, transports: dict[str, _FakeTransport]) -> None:
        self._transports = transports

    @property
    def node_ids(self) -> list[str]:
        return list(self._transports)

    def get(self, node_id: str) -> _FakeTransport | None:
        return self._transports.get(node_id)


class _FakeScheduler:
    def __init__(self, load: dict[str, int]) -> None:
        self._load = load

    def node_load_snapshot(self) -> dict[str, int]:
        return dict(self._load)


def test_control_loop_tick_contracts_on_bad_health() -> None:
    """`AimdControlLoop.tick()` reads each node's stashed health + the
    scheduler's load and runs one AIMD round — a slow node contracts."""
    controller = HealthAimdController(AimdConfig(initial_limit=16))
    registry = _FakeRegistry({
        "n": _FakeTransport({
            "create_p95_ms": 90_000.0,  # > 60s default → bad
            "docker_error_count": 0,
            "docker_timeout_count": 0,
        }),
    })
    loop = AimdControlLoop(
        controller=controller,
        registry=registry,  # type: ignore[arg-type]
        scheduler=_FakeScheduler({"n": 16}),  # type: ignore[arg-type]
    )

    loop.tick()

    assert controller.limit_for("n") == 8  # 16 halved on the bad tick


def test_control_loop_tick_holds_on_missing_health() -> None:
    """A node with no stashed health (pre-Stage-1 agent) → the tick
    neither grows nor contracts its limit."""
    controller = HealthAimdController(AimdConfig(initial_limit=16))
    registry = _FakeRegistry({"n": _FakeTransport(None)})
    loop = AimdControlLoop(
        controller=controller,
        registry=registry,  # type: ignore[arg-type]
        scheduler=_FakeScheduler({"n": 16}),  # type: ignore[arg-type]
    )

    loop.tick()

    assert controller.limit_for("n") == 16


def test_control_loop_mirrors_limit_to_state() -> None:
    """Stage 3 (3c): the loop mirrors each node's adaptive limit into
    the state store so the admin page can read it out-of-process."""
    state = InMemoryStateStore()
    controller = HealthAimdController(AimdConfig(initial_limit=16))
    registry = _FakeRegistry({
        "n": _FakeTransport({
            "create_p95_ms": 90_000.0,  # bad → contract 16 -> 8
            "docker_error_count": 0,
            "docker_timeout_count": 0,
        }),
    })
    loop = AimdControlLoop(
        controller=controller,
        registry=registry,  # type: ignore[arg-type]
        scheduler=_FakeScheduler({"n": 16}),  # type: ignore[arg-type]
        state=state,
    )

    loop.tick()

    assert state.list_node_aimd_limits() == {"n": 8}
