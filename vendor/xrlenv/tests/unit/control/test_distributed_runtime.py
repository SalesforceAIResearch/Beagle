"""Tests for DistributedRuntime admin_server lifecycle wiring.

Verifies that when admin_port is supplied to build_distributed_runtime,
the AdminServer is instantiated, starts on runtime.start(), and is stopped
on runtime.shutdown() without error.
"""

from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

from xrlenv.control.distributed_runtime import build_distributed_runtime
from xrlenv.control.state import InMemoryStateStore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_distributed_runtime_admin_server_starts_and_serves(
    tmp_path: Path,
) -> None:
    grpc_port = _free_port()
    admin_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    state_db = tmp_path / "state.db"

    state = InMemoryStateStore()
    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state=state,
        state_db_path=state_db,
        admin_port=admin_port,
        run_dir_retention_days=None,
    )
    assert runtime.admin_server is not None
    try:
        await runtime.start()
        assert runtime.admin_server.port > 0
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{runtime.admin_server.port}/healthz", timeout=5.0,
        ).read().decode()
        assert body == "ok"
    finally:
        await runtime.shutdown()


async def test_distributed_runtime_wires_resolved_token_store_to_admin(
    tmp_path: Path,
) -> None:
    """Regression for the 2026-05-11 operator-found bug: when
    ``xrlenv up`` was invoked without an explicit ``token_store=``,
    ``build_distributed_runtime`` resolved a TokenStore for the
    gRPC server (via ``TokenStore.load()``) but handed the admin
    server the raw ``token_store=None`` input parameter. Result:
    gRPC enforced bearer auth while the admin panel silently let
    every request through.

    The fix is to pass the *resolved* store to both servers. This
    test asserts the admin server's config holds the same store
    instance the gRPC interceptor sees.
    """
    from xrlenv.control.security import TokenStore

    secrets_root = tmp_path / "secrets"
    secrets_root.mkdir()
    token_path = secrets_root / "operator.token"
    token_path.write_text("write_op-tok", encoding="utf-8")
    token_path.chmod(0o600)

    grpc_port = _free_port()
    admin_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    state = InMemoryStateStore()
    # No explicit token_store= — exactly the ``xrlenv up`` path.
    # Steer the resolver at our test secrets dir so we don't depend
    # on the developer's ~/.xrlenv/secrets/.
    store = TokenStore.load(secrets_root=secrets_root, env={})
    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state=state,
        admin_port=admin_port,
        token_store=store,
        run_dir_retention_days=None,
    )
    try:
        assert runtime.admin_server is not None
        admin_store = runtime.admin_server.config.token_store
        assert admin_store is store, (
            "admin server received a different TokenStore than the one "
            "wired into the gRPC interceptor — auth middleware would "
            "silently no-op."
        )
        assert "operator" in admin_store.known_roles
    finally:
        await runtime.shutdown()


async def test_distributed_runtime_shutdown_stops_admin_server(
    tmp_path: Path,
) -> None:
    grpc_port = _free_port()
    admin_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    state = InMemoryStateStore()
    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state=state,
        admin_port=admin_port,
        run_dir_retention_days=None,
    )
    await runtime.start()
    bound_port = runtime.admin_server.port  # type: ignore[union-attr]
    assert bound_port > 0

    await runtime.shutdown()

    # After shutdown, admin_server._server is None — the port is released.
    assert runtime.admin_server._server is None  # type: ignore[union-attr]


async def test_startup_sweeps_stale_connected_node_rows(tmp_path: Path) -> None:
    """Audit H1 (2026-05-01): a control-plane crash / kill / reboot
    leaves the ``nodes`` table with rows pinned at
    ``status='connected'`` even though no transport is attached
    anywhere. Without a sweep, ``Client.list_nodes()`` /
    ``Client.wait_for_nodes()`` would satisfy readiness immediately
    on the next ``xrlenv up`` — before any node had reattached.

    Pin the fix: pre-seed ``state.db`` with a ``connected`` node
    row, build a fresh distributed runtime against the same db,
    assert the sweep marked the row ``lost``. New connections still
    flip back to ``connected`` via the registry's normal
    ``record_node_connected`` UPSERT; this test only pins the
    startup-sweep contract.
    """
    from xrlenv.control.state import SqliteStateStore

    grpc_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    state_db = tmp_path / "state.db"

    # Pre-seed the state store with a connected node row, then close.
    pre = SqliteStateStore(state_db)
    pre.record_node_connected(
        "stale-from-prior-process",
        backends=["docker"],
        stream_epoch="ep-1",
        instance_id="inst-1",
    )
    rows_before = pre.list_nodes(status="connected")
    pre.close()
    assert len(rows_before) == 1
    assert rows_before[0].status == "connected"

    # Build a fresh runtime against the same db. The startup sweep
    # runs synchronously inside build_distributed_runtime.
    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state_db_path=state_db,
        run_dir_retention_days=None,
    )
    try:
        # The previously-``connected`` row must now be ``lost``.
        rows_after = runtime.state.list_nodes()
        stale_row = next(
            (r for r in rows_after if r.node_id == "stale-from-prior-process"),
            None,
        )
        assert stale_row is not None
        assert stale_row.status == "lost", (
            f"startup sweep should mark stale rows as lost; got "
            f"status={stale_row.status!r}"
        )
        # And no rows should be ``connected`` after the sweep
        # (no real transports attached yet).
        connected = [r for r in rows_after if r.status == "connected"]
        assert connected == [], (
            f"no rows should be connected before any node attaches; "
            f"got {[r.node_id for r in connected]}"
        )
    finally:
        await runtime.shutdown()


def test_startup_sweep_then_reattach_flips_back_to_connected(tmp_path: Path) -> None:
    """Completes the masquerade→recover contract the sweep docstring promises but
    the test above only half-covers: after a stale prior-process ``connected``
    row is swept to ``lost`` (so it can no longer satisfy ``wait_for_nodes``),
    the REAL node reattaching must flip it back to ``connected`` via
    ``record_node_connected``'s UPSERT — otherwise a real, healthy node would be
    stuck ``lost`` after every CP restart.

    Calls the sweep function directly (no full runtime needed) so the
    lost→connected recovery leg is pinned in isolation.
    """
    from xrlenv.control.distributed_runtime import _mark_stale_connected_nodes_lost
    from xrlenv.control.state import SqliteStateStore

    store = SqliteStateStore(tmp_path / "state.db")
    try:
        store.record_node_connected(
            "n1", backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
        )
        assert [r.node_id for r in store.list_nodes(status="connected")] == ["n1"]

        # CP restart → the sweep marks the orphaned row lost (no masquerade).
        assert _mark_stale_connected_nodes_lost(store) == 1
        assert store.list_nodes(status="connected") == []
        assert (
            next(r for r in store.list_nodes() if r.node_id == "n1").status == "lost"
        )

        # The real node reattaches on a NEW stream epoch → UPSERT flips it back.
        store.record_node_connected(
            "n1", backends=["docker"], stream_epoch="ep-2", instance_id="inst-2",
        )
        connected = store.list_nodes(status="connected")
        assert [r.node_id for r in connected] == ["n1"], (
            "a real node reattaching after the startup sweep must return to "
            "'connected' — a lost row must not be sticky"
        )
        assert connected[0].status == "connected"
    finally:
        store.close()


def test_prune_unrostered_lost_nodes_skips_empty_roster() -> None:
    """Safety guard: an empty roster (missing / failed-to-load nodes.yaml) must
    never prune — otherwise a bad nodes.yaml would nuke the whole registry."""
    from xrlenv.control.distributed_runtime import _prune_unrostered_lost_nodes
    from xrlenv.control.state import InMemoryStateStore

    s = InMemoryStateStore()
    s.record_node_connected("gone", backends=["docker"])
    s.record_node_disconnected("gone")
    assert _prune_unrostered_lost_nodes(s, set()) == []
    assert [r.node_id for r in s.list_nodes()] == ["gone"]


def test_prune_unrostered_lost_nodes_reaps_with_roster() -> None:
    from xrlenv.control.distributed_runtime import _prune_unrostered_lost_nodes
    from xrlenv.control.state import InMemoryStateStore

    s = InMemoryStateStore()
    s.record_node_connected("gone", backends=["docker"])
    s.record_node_disconnected("gone")
    s.record_node_connected("keep", backends=["docker"])
    s.record_node_disconnected("keep")
    assert _prune_unrostered_lost_nodes(s, {"keep"}) == ["gone"]
    assert [r.node_id for r in s.list_nodes()] == ["keep"]


async def test_startup_prunes_unrostered_lost_nodes(tmp_path: Path) -> None:
    """CP-startup reconciliation reaps ``lost`` rows for node_ids absent from
    the nodes.yaml roster (a decommissioned / reboot-orphaned host), while
    keeping rostered nodes. Pins the build_distributed_runtime call-site wiring
    (roster from nodes.yaml -> prune_lost_nodes)."""
    from xrlenv.control.state import SqliteStateStore

    grpc_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    state_db = tmp_path / "state.db"
    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text("nodes:\n  - id: keep-me\n", encoding="utf-8")

    pre = SqliteStateStore(state_db)
    # 'gone' — lost + NOT in the roster -> pruned.
    pre.record_node_connected("gone", backends=["docker"])
    pre.record_node_disconnected("gone")
    # 'keep-me' — lost but IN the roster -> survives.
    pre.record_node_connected("keep-me", backends=["docker"])
    pre.record_node_disconnected("keep-me")
    pre.close()

    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state_db_path=state_db,
        admin_nodes_yaml=nodes_yaml,
        run_dir_retention_days=None,
    )
    try:
        ids = {r.node_id for r in runtime.state.list_nodes()}
        assert "gone" not in ids, "unrostered lost node should be pruned at startup"
        assert "keep-me" in ids, "rostered node must survive the prune"
    finally:
        await runtime.shutdown()


async def test_adaptive_admission_flag_wires_controller_with_config(
    tmp_path: Path,
) -> None:
    """Stage 3: ``adaptive_admission=True`` wires a HealthAimdController
    carrying the supplied AimdConfig into the scheduler, and starts the
    control loop. Off (default) → neither is present."""
    from xrlenv.control.capacity import AimdConfig

    grpc_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state=InMemoryStateStore(),
        run_dir_retention_days=None,
        adaptive_admission=True,
        aimd_config=AimdConfig(initial_limit=7, max_limit=20),
    )
    try:
        assert runtime.aimd_loop is not None
        controller = runtime.scheduler._aimd
        assert controller is not None
        # The supplied config flowed through to the controller.
        assert controller.limit_for("any-node") == 7  # slow-start seed
        assert controller.config.max_limit == 20
    finally:
        await runtime.shutdown()


async def test_adaptive_admission_off_by_default(tmp_path: Path) -> None:
    """Default (flag off) → no controller, no loop; the scheduler's
    AIMD filter is a no-op and behaviour matches the static estimator."""
    grpc_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state=InMemoryStateStore(),
        run_dir_retention_days=None,
    )
    try:
        assert runtime.aimd_loop is None
        assert runtime.scheduler._aimd is None
    finally:
        await runtime.shutdown()


async def test_watchdog_node_loss_removes_node_from_scheduler(
    tmp_path: Path,
) -> None:
    """Issue #18: the heartbeat watchdog's node-loss callback must drop
    the node from the *scheduler*, not only the registry + state store.

    Before the fix the watchdog path (``NodeRegistry`` →
    ``on_node_lost``) ran ``registry.deregister`` + ``handle_node_lost``
    but never ``scheduler.remove_node`` — only the stream-disconnect
    path did. A watchdog-lost node therefore kept receiving placements
    while every operator view showed it ``lost``. This exercises the
    exact ``_on_node_lost`` callback ``build_distributed_runtime``
    wires into the registry.
    """
    grpc_port = _free_port()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    runtime = await build_distributed_runtime(
        grpc_port=grpc_port,
        runs_root=runs_root,
        state=InMemoryStateStore(),
        run_dir_retention_days=None,
    )
    try:

        class _FakeNode:
            node_id = "watchdog-victim"

            def supported_backends(self) -> list[str]:
                return ["docker"]

        runtime.scheduler.add_node(_FakeNode())  # type: ignore[arg-type]
        assert any(
            n.node_id == "watchdog-victim" for n in runtime.scheduler.nodes
        )

        # The literal callback the watchdog invokes on node loss.
        await runtime.registry._on_node_lost("watchdog-victim")

        assert not any(
            n.node_id == "watchdog-victim" for n in runtime.scheduler.nodes
        ), "watchdog node-loss must drop the node from the scheduler"
    finally:
        await runtime.shutdown()


# ── _DistributedBudgetProvider: budgets from heartbeat-cached disk ──────────


async def test_get_budgets_uses_cached_disk_state_not_report_images() -> None:
    """``get_budgets`` reads the heartbeat-cached free disk
    (``disk_state``) and does NOT call the heavy ``report_images``
    (``docker system df``) when a sample exists — avoiding containerd
    metadata-lock contention under a pull storm."""
    from xrlenv.control.distributed_runtime import _DistributedBudgetProvider

    report_calls = {"n": 0}

    class _T:
        def disk_state(self) -> tuple[int, int]:
            return (200 * 1024**3, 500 * 1024**3)

        async def report_images(self) -> object:
            report_calls["n"] += 1
            raise AssertionError(
                "report_images must not run when disk_state is populated",
            )

    class _Reg:
        node_ids = ("n1",)

        def get(self, node_id: str) -> object | None:
            return _T() if node_id == "n1" else None

    prov = _DistributedBudgetProvider(registry=_Reg())  # type: ignore[arg-type]
    budgets = await prov.get_budgets(
        reserved_runtime_gb=10, buffer_gb=5, cap_per_node_gb=None,
    )
    assert report_calls["n"] == 0
    assert len(budgets) == 1
    assert budgets[0].node_id == "n1"
    assert budgets[0].available_bytes == (200 - 15) * 1024**3


async def test_get_budgets_falls_back_to_report_images_without_heartbeat(
) -> None:
    """A just-connected node with no heartbeat sample (``disk_state``
    returns ``(0, 0)``) falls back to a direct ``report_images`` probe."""
    from types import SimpleNamespace

    from xrlenv.control.distributed_runtime import _DistributedBudgetProvider

    class _T:
        def disk_state(self) -> tuple[int, int]:
            return (0, 0)

        async def report_images(self) -> object:
            return SimpleNamespace(free_disk_bytes=300 * 1024**3)

    class _Reg:
        node_ids = ("n1",)

        def get(self, node_id: str) -> object | None:
            return _T()

    prov = _DistributedBudgetProvider(registry=_Reg())  # type: ignore[arg-type]
    budgets = await prov.get_budgets(
        reserved_runtime_gb=0, buffer_gb=0, cap_per_node_gb=None,
    )
    assert len(budgets) == 1
    assert budgets[0].available_bytes == 300 * 1024**3


# ──────────────────────────────────────────────────────────────────────────────
# _is_wire_timeout: distinguish a control-plane reply timeout (node may still be
# working) from a real failure, so a timed-out dispatch isn't counted as failed.
# ──────────────────────────────────────────────────────────────────────────────


def test_is_wire_timeout_detects_timeout_cause_and_message() -> None:
    from xrlenv.control.distributed_runtime import _is_wire_timeout
    from xrlenv.errors import XRLEnvError

    # How _send_and_wait re-raises: XRLEnvError chained from TimeoutError.
    try:
        try:
            raise TimeoutError("reply timeout")
        except TimeoutError as te:
            raise XRLEnvError(
                "node x: command y timed out after 60.0s waiting for reply",
            ) from te
    except XRLEnvError as exc:
        assert _is_wire_timeout(exc) is True

    # Bare TimeoutError is also a wire timeout.
    assert _is_wire_timeout(TimeoutError("boom")) is True
    # Message fallback (no TimeoutError in the cause chain).
    assert _is_wire_timeout(RuntimeError("... timed out after 5s ...")) is True
    # A genuine failure is NOT a timeout.
    assert _is_wire_timeout(ValueError("image not found")) is False


def test_resolve_advertise_endpoint_keeps_explicit_host() -> None:
    """A concrete bind host is shown verbatim — it's already dialable."""
    from xrlenv.control.distributed_runtime import _resolve_advertise_endpoint

    assert _resolve_advertise_endpoint("192.168.1.10", 50051) == "192.168.1.10:50051"


def test_resolve_advertise_endpoint_substitutes_primary_ip_for_wildcard(
    monkeypatch,
) -> None:
    """A wildcard / loopback bind isn't dialable from another box, so the
    overview shows the detected primary IP instead. Falls back to the bind
    host verbatim when detection fails (never invents an address)."""
    import xrlenv.control.distributed_runtime as dr

    monkeypatch.setattr(dr, "_primary_outbound_ip", lambda: "10.0.0.7")
    for wildcard in ("0.0.0.0", "::", "", "127.0.0.1", "localhost", "::1"):
        assert dr._resolve_advertise_endpoint(wildcard, 50051) == "10.0.0.7:50051"

    # Detection unavailable → keep the configured host rather than guess.
    monkeypatch.setattr(dr, "_primary_outbound_ip", lambda: None)
    assert dr._resolve_advertise_endpoint("0.0.0.0", 50051) == "0.0.0.0:50051"
