"""Tests for the heartbeat-driven NodeRegistry (Slice 3.5)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.control.node_registry import NodeRegistry


@dataclass
class _FakeTransport:
    node_id: str
    last_heartbeat_at: float = field(default_factory=time.monotonic)
    backends: list[str] = field(default_factory=lambda: ["docker"])
    stream_epoch: str = "epoch-fake"
    control_instance_id: str = "instance-fake"


# ──────────────────────────────────────────────────────────────────────────────
# Membership
# ──────────────────────────────────────────────────────────────────────────────


def test_register_deregister_membership() -> None:
    async def noop(_node_id: str) -> None:
        return None

    reg = NodeRegistry(on_node_lost=noop)
    a = _FakeTransport("a")
    b = _FakeTransport("b")
    reg.register(a)
    reg.register(b)
    assert sorted(reg.node_ids) == ["a", "b"]
    reg.deregister("a")
    assert reg.node_ids == ["b"]
    reg.deregister("missing")  # idempotent
    assert reg.node_ids == ["b"]


def test_get_returns_none_for_unknown() -> None:
    async def noop(_node_id: str) -> None:
        return None

    reg = NodeRegistry(on_node_lost=noop)
    assert reg.get("never") is None


# ──────────────────────────────────────────────────────────────────────────────
# Watchdog
# ──────────────────────────────────────────────────────────────────────────────


async def test_watchdog_marks_node_lost_after_grace() -> None:
    lost: list[str] = []

    async def on_lost(node_id: str) -> None:
        lost.append(node_id)

    reg = NodeRegistry(
        on_node_lost=on_lost,
        disconnect_grace_s=0.2,
        check_interval_s=0.02,
    )
    await reg.start()
    try:
        # Heartbeat-fresh-now node; should NOT be marked lost on first sweep.
        fresh = _FakeTransport("fresh")
        reg.register(fresh)
        # Stale node: last heartbeat far in the past.
        stale = _FakeTransport("stale", last_heartbeat_at=time.monotonic() - 10.0)
        reg.register(stale)

        # Tick "fresh" while we wait for the watchdog to find "stale".
        for _ in range(8):
            await asyncio.sleep(0.02)
            fresh.last_heartbeat_at = time.monotonic()
            if lost:
                break
        assert lost == ["stale"]
        assert reg.node_ids == ["fresh"]
    finally:
        await reg.shutdown()


async def test_fresh_heartbeat_keeps_node_alive() -> None:
    lost: list[str] = []

    async def on_lost(node_id: str) -> None:
        lost.append(node_id)

    reg = NodeRegistry(
        on_node_lost=on_lost,
        disconnect_grace_s=0.1,
        check_interval_s=0.02,
    )
    await reg.start()
    try:
        node = _FakeTransport("alive")
        reg.register(node)
        # Tick the heartbeat regularly.
        for _ in range(8):
            node.last_heartbeat_at = time.monotonic()
            await asyncio.sleep(0.03)
        assert lost == []
        assert reg.node_ids == ["alive"]
    finally:
        await reg.shutdown()


async def test_node_lost_handler_exception_is_isolated() -> None:
    """A buggy handler shouldn't kill the watchdog or other nodes."""
    cleared = asyncio.Event()

    async def boom(_node_id: str) -> None:
        cleared.set()
        raise RuntimeError("handler exploded")

    reg = NodeRegistry(
        on_node_lost=boom,
        disconnect_grace_s=0.05,
        check_interval_s=0.02,
    )
    await reg.start()
    try:
        reg.register(_FakeTransport("stale", last_heartbeat_at=time.monotonic() - 5.0))
        await asyncio.wait_for(cleared.wait(), timeout=1.0)
        # Watchdog still running — register a fresh node and confirm we can
        # still observe it without the loop being dead.
        fresh = _FakeTransport("fresh")
        reg.register(fresh)
        assert "fresh" in reg.node_ids
    finally:
        await reg.shutdown()


# ──────────────────────────────────────────────────────────────────────────────
# nodes.yaml loader
# ──────────────────────────────────────────────────────────────────────────────


def test_nodes_yaml_loader_returns_empty_when_file_absent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    inv = load_nodes_yaml(tmp_path / "absent.yaml")
    assert inv.nodes == ()
    assert inv.version == 1


def test_nodes_yaml_loader_parses_entries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    p = tmp_path / "nodes.yaml"
    p.write_text(
        "version: 1\n"
        "nodes:\n"
        "  - id: gcp-1\n"
        "    address: 10.0.0.1\n"
        "    cloud: gcp\n"
        "    backends: [docker]\n"
        "    auth_token_env: NODE_TOKEN_GCP_1\n"
        "  - id: aws-1\n"
        "    address: 10.0.0.2\n"
        "    cloud: aws\n"
        "    backends: [docker]\n"
    )
    inv = load_nodes_yaml(p)
    assert sorted(inv.by_id().keys()) == ["aws-1", "gcp-1"]
    assert inv.by_id()["gcp-1"].auth_token_env == "NODE_TOKEN_GCP_1"


def test_nodes_yaml_sysbox_pool_and_allowed_runtimes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Sysbox pool — ``sysbox: true`` marks pool membership and
    ``policy.allowed_runtimes`` opts the cluster into the runtime override.
    Nodes default to ``sysbox=False`` (the ordinary docker pool)."""
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    p = tmp_path / "nodes.yaml"
    p.write_text(
        "version: 1\n"
        "nodes:\n"
        "  - id: n-sysbox\n"
        "    address: 10.0.0.1\n"
        "    backends: [docker]\n"
        "    sysbox: true\n"
        "  - id: n-plain\n"
        "    address: 10.0.0.2\n"
        "    backends: [docker]\n"
        "policy:\n"
        "  allowed_runtimes: [sysbox-runc]\n"
    )
    inv = load_nodes_yaml(p)
    assert [n.id for n in inv.sysbox_pool()] == ["n-sysbox"]
    assert inv.by_id()["n-plain"].sysbox is False
    assert "sysbox-runc" in inv.policy.allowed_runtimes


def test_nodes_yaml_default_pool_is_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """No node marked ``sysbox`` → an empty pool and the default (empty)
    ``allowed_runtimes`` (every runtime override rejected) — the normal
    cluster is unchanged."""
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    p = tmp_path / "nodes.yaml"
    p.write_text(
        "version: 1\n"
        "nodes:\n"
        "  - id: n-1\n"
        "    backends: [docker]\n"
    )
    inv = load_nodes_yaml(p)
    assert inv.sysbox_pool() == ()
    assert inv.policy.allowed_runtimes == ()


def test_nodes_yaml_rejects_unknown_field(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from xrlenv.control.nodes_yaml import load_nodes_yaml
    from xrlenv.errors import ManifestInvalid

    p = tmp_path / "nodes.yaml"
    p.write_text("version: 1\nnodes:\n  - id: x\n    weird_field: 42\n")
    with pytest.raises(ManifestInvalid):
        load_nodes_yaml(p)


# ──────────────────────────────────────────────────────────────────────────────
# Cluster docker-kwarg policy section (issue #6) — loaded from nodes.yaml,
# enforced by the control plane on every acquire_container request.
# ──────────────────────────────────────────────────────────────────────────────


def test_nodes_yaml_policy_defaults_when_section_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """nodes.yaml without a ``policy:`` section falls back to
    DEFAULT_POLICY. Important so existing deployments don't regress when
    they upgrade past this slice without editing their file."""
    from xrlenv.control.kwargs_policy import DEFAULT_POLICY
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    p = tmp_path / "nodes.yaml"
    p.write_text("version: 1\nnodes: []\n")
    inv = load_nodes_yaml(p)
    assert inv.policy == DEFAULT_POLICY


def test_nodes_yaml_policy_section_loads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A populated ``policy:`` section parses into the KwargsPolicy
    model with all five operator-tunable knobs honored."""
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    p = tmp_path / "nodes.yaml"
    p.write_text(
        "version: 1\n"
        "nodes: []\n"
        "policy:\n"
        "  allowed_devices: [/dev/kvm, /dev/dri/card0]\n"
        "  denied_caps: [SYS_MODULE]\n"
        "  allow_host_network: true\n"
        "  allow_privileged: false\n"
        "  allowed_host_paths: [/mnt/datasets]\n"
    )
    inv = load_nodes_yaml(p)
    assert inv.policy.allowed_devices == ("/dev/kvm", "/dev/dri/card0")
    assert inv.policy.denied_caps == ("SYS_MODULE",)
    assert inv.policy.allow_host_network is True
    assert inv.policy.allow_privileged is False
    assert inv.policy.allowed_host_paths == ("/mnt/datasets",)


def test_nodes_yaml_policy_rejects_unknown_field(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Typos in policy keys fail loudly at load time — otherwise
    operators would silently get DEFAULT_POLICY when they intended a
    tweak."""
    from xrlenv.control.nodes_yaml import load_nodes_yaml
    from xrlenv.errors import ManifestInvalid

    p = tmp_path / "nodes.yaml"
    p.write_text(
        "version: 1\n"
        "nodes: []\n"
        "policy:\n"
        "  allow_priviledged: true\n"  # typo: priviledged
    )
    with pytest.raises(ManifestInvalid):
        load_nodes_yaml(p)


def test_nodes_yaml_policy_partial_section_keeps_other_defaults(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Operator overrides one knob; the rest of the policy retains
    DEFAULT_POLICY's values (no need to re-declare every default)."""
    from xrlenv.control.kwargs_policy import DEFAULT_POLICY
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    p = tmp_path / "nodes.yaml"
    p.write_text(
        "version: 1\n"
        "nodes: []\n"
        "policy:\n"
        "  denied_caps: [SYS_MODULE]\n"
    )
    inv = load_nodes_yaml(p)
    assert inv.policy.denied_caps == ("SYS_MODULE",)
    # Other knobs still at DEFAULT_POLICY values.
    assert inv.policy.allowed_devices == DEFAULT_POLICY.allowed_devices
    assert inv.policy.allow_host_network is False
    assert inv.policy.allow_privileged is False


def test_repo_nodes_yaml_example_loads_cleanly() -> None:
    """``examples/nodes.yaml.example`` is the canonical template the
    team copies; it must parse against the schema and its ``policy:``
    section must round-trip with ``DEFAULT_POLICY``. Guards against
    drift between the docs/template and the loader contract."""
    from pathlib import Path

    from xrlenv.control.kwargs_policy import DEFAULT_POLICY
    from xrlenv.control.nodes_yaml import load_nodes_yaml

    repo_yaml = (
        Path(__file__).resolve().parents[3]
        / "examples" / "nodes.yaml.example"
    )
    if not repo_yaml.exists():  # pragma: no cover — defensive
        pytest.skip(f"example template not found at {repo_yaml}")
    inv = load_nodes_yaml(repo_yaml)
    # Template ships defaults verbatim — round-trip equality is the
    # guarantee that drift is impossible.
    assert inv.policy == DEFAULT_POLICY


# ──────────────────────────────────────────────────────────────────────────────
# Slice-4 follow-up: NodeRegistry mirrors register/deregister to state.db
# so out-of-process callers (`xrlenv nodes`, future admin RPC) can see who's
# attached without going through gRPC.
# ──────────────────────────────────────────────────────────────────────────────


def test_register_persists_node_row_to_state() -> None:
    from xrlenv.control.state import InMemoryStateStore

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    reg.register(_FakeTransport("aws-1"))

    rows = store.list_nodes()
    assert [r.node_id for r in rows] == ["aws-1"]
    assert rows[0].status == "connected"
    assert rows[0].backends == ["docker"]
    assert rows[0].stream_epoch == "epoch-fake"
    assert rows[0].instance_id == "instance-fake"


def test_deregister_marks_node_lost_in_state() -> None:
    from xrlenv.control.state import InMemoryStateStore

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    reg.register(_FakeTransport("gcp-1"))
    reg.deregister("gcp-1")

    rows = store.list_nodes()
    assert len(rows) == 1
    assert rows[0].status == "lost"

    connected = store.list_nodes(status="connected")
    assert connected == []


def test_register_without_state_is_a_no_op() -> None:
    """Backwards-compat: NodeRegistry constructed without ``state`` keeps
    Slice-3.5 behaviour and never raises."""
    async def noop(_node_id: str) -> None:
        return None

    reg = NodeRegistry(on_node_lost=noop)
    reg.register(_FakeTransport("x"))
    reg.deregister("x")


def test_deregister_skips_state_mirror_when_store_closed() -> None:
    """Shutdown race: ``deregister`` runs from the gRPC stream's
    ``finally`` block, which can fire after the runtime already closed
    the state store. ``deregister`` must skip the mirror write quietly
    (no ``sqlite3.ProgrammingError``, no scary ERROR log) when the
    store reports itself closed."""
    from xrlenv.control.state import SqliteStateStore

    async def noop(_node_id: str) -> None:
        return None

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        store = SqliteStateStore(Path(td) / "state.db")
        reg = NodeRegistry(on_node_lost=noop, state=store)
        reg.register(_FakeTransport("gcp-1"))
        # Simulate the runtime closing the store mid-shutdown, before
        # the gRPC stream finally-block reaches deregister.
        store.close()
        assert store.is_closed is True

        # Must not raise — the closed-store write is skipped quietly.
        reg.deregister("gcp-1")


# ──────────────────────────────────────────────────────────────────────────────
# Backends-extraction shape compatibility (operator-reported regression
# 2026-05-04: the prior registry called ``getattr(transport, "backends",
# [])`` but RemoteNodeTransport exposes its backends via the method
# ``supported_backends()``, so every gRPC-attached node landed in state
# with backends=[] and the SDK's wait_for_nodes(backend="docker")
# filter rejected them all)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _MethodTransport:
    """Mirrors the real :class:`RemoteNodeTransport` interface — backends
    are exposed via a method (``supported_backends()``), not an attribute.
    """

    node_id: str
    last_heartbeat_at: float = field(default_factory=time.monotonic)
    stream_epoch: str = "epoch-method"
    control_instance_id: str = "instance-method"
    _backends: list[str] = field(default_factory=lambda: ["docker"])

    def supported_backends(self) -> list[str]:
        return list(self._backends)


def test_register_extracts_backends_via_supported_backends_method() -> None:
    """Real RemoteNodeTransport exposes backends via ``supported_backends()``
    (a method), NOT a ``.backends`` attribute. The registry must call
    the method when persisting the per-node row to state — pre-fix it
    used ``getattr(transport, "backends", [])`` which silently
    returned ``[]`` for every gRPC-attached node, breaking
    wait_for_nodes(backend="docker") filtering.
    """
    from xrlenv.control.state import InMemoryStateStore

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    reg.register(_MethodTransport("gcp-1"))

    rows = store.list_nodes()
    assert len(rows) == 1
    # The whole point: backends list survives via the method, not "[]".
    assert rows[0].backends == ["docker"], (
        f"expected ['docker']; got {rows[0].backends!r} — registry is "
        "probably reading the wrong interface (attribute vs method)"
    )


def test_register_extracts_backends_via_method_with_multiple_entries() -> None:
    """Method-shape transport with > 1 backend (phase-2 mixed-backend
    nodes will list both ``docker`` and ``cube``)."""
    from xrlenv.control.state import InMemoryStateStore

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    reg.register(_MethodTransport("aws-1", _backends=["docker", "cube"]))

    rows = store.list_nodes()
    assert rows[0].backends == ["docker", "cube"]


def test_register_falls_back_to_backends_attribute_when_no_method() -> None:
    """Defense in depth: a future / test transport that exposes a
    ``backends`` list directly (no ``supported_backends`` method) is
    still accepted. The original ``_FakeTransport`` in this file uses
    that shape — ensure backward compatibility is preserved."""
    from xrlenv.control.state import InMemoryStateStore

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    # The existing _FakeTransport at the top of this file uses the
    # attribute shape: ``backends: list[str] = ["docker"]``.
    reg.register(_FakeTransport("attr-shape"))

    rows = store.list_nodes()
    assert rows[0].backends == ["docker"]


def test_register_with_minimal_transport_lands_empty_backends() -> None:
    """Transport that implements only the watchdog surface (no
    backends method or attribute) lands ``backends=[]`` cleanly —
    mirrors the existing tolerance for test fakes that don't carry
    the persistence fields."""
    from xrlenv.control.state import InMemoryStateStore

    @dataclass
    class _MinimalTransport:
        node_id: str
        last_heartbeat_at: float = field(default_factory=time.monotonic)

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    reg.register(_MinimalTransport("minimal"))

    rows = store.list_nodes()
    assert rows[0].backends == []
    assert rows[0].status == "connected"


def test_supported_backends_method_raising_does_not_crash_register() -> None:
    """If a transport's ``supported_backends`` method raises, the
    registry falls back to the attribute path / empty list rather
    than letting the exception abort registration. Defensive
    against a buggy / partial transport implementation."""
    from xrlenv.control.state import InMemoryStateStore

    @dataclass
    class _BrokenMethodTransport:
        node_id: str
        last_heartbeat_at: float = field(default_factory=time.monotonic)
        backends: list[str] = field(default_factory=lambda: ["docker"])

        def supported_backends(self) -> list[str]:
            raise RuntimeError("intentionally broken")

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    reg.register(_BrokenMethodTransport("broken"))

    rows = store.list_nodes()
    # Falls through to the attribute path — backends survives.
    assert rows[0].backends == ["docker"]


# ── P6 step-2c — isolation capability + pinned-CPU mirror (observability) ──────


def test_register_mirrors_isolation_capable_and_heartbeat_mirrors_pinned_cpus() -> None:
    """P6 step-2c — register persists the advertised isolation capability
    (NodeHello), and the per-heartbeat mirror persists the last-known
    pinnable-CPU counts (heartbeat). Observability only."""
    from xrlenv.control.state import InMemoryStateStore

    @dataclass
    class _IsoTransport:
        node_id: str
        last_heartbeat_at: float = field(default_factory=time.monotonic)
        backends: list[str] = field(default_factory=lambda: ["docker"])
        stream_epoch: str = "e"
        control_instance_id: str = "i"
        on_hb: Any = None

        def supported_backends(self) -> list[str]:
            return ["docker"]

        def isolation_capable(self) -> bool:
            return True

        def pinned_cpu_state(self) -> tuple[int, int]:
            return (6, 8)

        def set_on_heartbeat(self, cb: Any) -> None:
            self.on_hb = cb

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    t = _IsoTransport("iso-1")
    reg.register(t)

    row = {r.node_id: r for r in store.list_nodes()}["iso-1"]
    assert row.isolation_capable is True
    # No heartbeat yet → pinned counts still unknown.
    assert (row.pinned_cpus_free, row.pinned_cpus_total) == (0, 0)

    # The registry installed ``_note_heartbeat`` via set_on_heartbeat; invoking
    # it mirrors the transport's live pinned_cpu_state onto the row.
    assert t.on_hb is not None
    t.on_hb("iso-1")
    row = {r.node_id: r for r in store.list_nodes()}["iso-1"]
    assert (row.pinned_cpus_free, row.pinned_cpus_total) == (6, 8)


def test_register_defaults_isolation_capable_false_for_pre_p6_transport() -> None:
    """A transport with no isolation_capable() (a pre-P6 fake) → False on the
    persisted row, no crash."""
    from xrlenv.control.state import InMemoryStateStore

    @dataclass
    class _NoIsoTransport:
        node_id: str
        last_heartbeat_at: float = field(default_factory=time.monotonic)
        backends: list[str] = field(default_factory=lambda: ["docker"])

    async def noop(_node_id: str) -> None:
        return None

    store = InMemoryStateStore()
    reg = NodeRegistry(on_node_lost=noop, state=store)
    reg.register(_NoIsoTransport("noiso"))
    assert store.list_nodes()[0].isolation_capable is False


# ──────────────────────────────────────────────────────────────────────────────
# Stall-aware eviction (2026-08-21) — a control-plane event-loop freeze must not
# false-mark the fleet lost. These drive ``_sweep`` directly for determinism.
# ──────────────────────────────────────────────────────────────────────────────


def _stale(node_id: str, grace: float) -> _FakeTransport:
    return _FakeTransport(node_id, last_heartbeat_at=time.monotonic() - grace * 5)


async def test_blackout_defers_eviction_then_evicts_after_window() -> None:
    """A sweep that runs far later than its cadence (the loop was frozen) must
    NOT evict stale nodes on that first post-thaw sweep — it opens a fresh grace
    window instead. Only nodes still stale after the window are evicted."""
    lost: list[str] = []

    async def on_lost(node_id: str) -> None:
        lost.append(node_id)

    reg = NodeRegistry(
        on_node_lost=on_lost, disconnect_grace_s=1.0, check_interval_s=0.05,
    )
    reg.register(_stale("a", 1.0))
    # Simulate a prior sweep 1s ago (>> blackout threshold max(0.15, 0.5)=0.5).
    reg._last_tick = time.monotonic() - 1.0
    await reg._sweep()
    assert lost == []                      # deferred, not evicted
    assert reg.node_ids == ["a"]           # still registered
    assert reg._resume_deadline > time.monotonic()   # fresh window opened

    # Window closed, no new blackout, node still stale → evicted now.
    reg._last_tick = time.monotonic()
    reg._resume_deadline = time.monotonic() - 0.01
    await reg._sweep()
    assert lost == ["a"]
    assert reg.node_ids == []


async def test_mass_loss_circuit_breaker_defers_one_cycle() -> None:
    """Even without a detected blackout, a single sweep that would evict a large
    fraction of the fleet defers one grace cycle (synchronized loss ⇒ CP-side),
    fires the on_mass_loss hook, then evicts if still stale."""
    lost: list[str] = []
    mass: list[tuple[int, int]] = []

    async def on_lost(node_id: str) -> None:
        lost.append(node_id)

    reg = NodeRegistry(
        on_node_lost=on_lost,
        disconnect_grace_s=1.0,
        check_interval_s=0.05,
        mass_loss_min_fleet=3,
        on_mass_loss=lambda n, total: mass.append((n, total)),
    )
    for nid in ("a", "b", "c"):
        reg.register(_stale(nid, 1.0))
    reg.register(_FakeTransport("d"))       # healthy — 3/4 stale is > 50%
    reg._last_tick = time.monotonic()       # NO blackout this sweep

    await reg._sweep()
    assert lost == []                       # mass loss deferred, not evicted
    assert mass == [(3, 4)]                  # hook fired with (lost, registered)
    assert set(reg.node_ids) == {"a", "b", "c", "d"}

    # Still stale after the window → accepted as real and evicted.
    reg._last_tick = time.monotonic()
    reg._resume_deadline = time.monotonic() - 0.01
    await reg._sweep()
    assert sorted(lost) == ["a", "b", "c"]
    assert reg.node_ids == ["d"]


async def test_single_stale_node_evicted_immediately() -> None:
    """Regression: a lone stale node in a healthy fleet is NOT a mass loss and
    is evicted on the first sweep (the deferral must not swallow normal loss)."""
    lost: list[str] = []

    async def on_lost(node_id: str) -> None:
        lost.append(node_id)

    reg = NodeRegistry(
        on_node_lost=on_lost, disconnect_grace_s=1.0, check_interval_s=0.05,
        mass_loss_min_fleet=3,
    )
    reg.register(_stale("stale", 1.0))
    reg.register(_FakeTransport("h1"))
    reg.register(_FakeTransport("h2"))
    # Fresh registry (first sweep) ⇒ _last_tick is None ⇒ no blackout.
    await reg._sweep()
    assert lost == ["stale"]
    assert sorted(reg.node_ids) == ["h1", "h2"]


async def test_healthy_sweep_rearms_deferral_flag() -> None:
    """After the fleet recovers (no stale nodes), the one-shot deferral flag
    re-arms so the NEXT mass-loss episode also gets its grace window."""
    reg = NodeRegistry(
        on_node_lost=_noop_on_lost, disconnect_grace_s=1.0, check_interval_s=0.05,
    )
    reg.register(_FakeTransport("a"))
    reg._post_stall_window_used = True
    await reg._sweep()                       # no stale nodes
    assert reg._post_stall_window_used is False


async def _noop_on_lost(_node_id: str) -> None:
    return None
