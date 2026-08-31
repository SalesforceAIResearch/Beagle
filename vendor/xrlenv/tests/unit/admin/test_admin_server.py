"""Tests for the Slice 7a admin server (spec 13).

Covers each phase-0 view via FastAPI's TestClient + the spec-19
admin-bind guard. The server reads ``state.db`` + ``runs/`` directly,
so each test seeds a temp pair of those and asserts the rendered
content / status code.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from xrlenv.admin import (
    ADMIN_PAGE_OWNED_FACTS,
    AdminBindError,
    AdminServer,
    AdminServerConfig,
    build_admin_app,
)
from xrlenv.admin.server import _SESSION_COOKIE
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.state import (
    RawRolloutRecord,
    RolloutRecord,
    SandboxRecord,
    SqliteStateStore,
)
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.control.trajectory_cache import TrajectoryCacheConfig
from xrlenv.control.trajectory_sink import PlatformJsonlSink
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.types import RolloutStatus, Step

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_admin_trajectory_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-route the trajectory cache to ``tmp_path`` so tests don't read or
    write the operator's real ``~/.xrlenv/admin-cache``.

    We replace the ``TrajectoryCache`` symbol the admin module sees with a
    wrapper that injects a per-test cache_root when the caller didn't
    supply one. Tests that pass an explicit ``trajectory_cache_config``
    still take precedence.
    """
    from xrlenv.admin import server as admin_server
    from xrlenv.control.trajectory_cache import (
        TrajectoryCache as _RealCache,
    )
    from xrlenv.control.trajectory_cache import (
        TrajectoryCacheConfig as _RealCfg,
    )

    cache_root = tmp_path / "admin-cache-default"

    def _wrap(cfg: _RealCfg | None = None) -> _RealCache:
        if cfg is None:
            cfg = _RealCfg(cache_root=cache_root)
        return _RealCache(cfg)

    monkeypatch.setattr(admin_server, "TrajectoryCache", _wrap)


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def cfg(state_db: Path, runs_root: Path, tmp_path: Path) -> AdminServerConfig:
    return AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="127.0.0.1", port=0,
        trajectory_cache_config=TrajectoryCacheConfig(
            cache_root=tmp_path / "admin-cache",
        ),
    )


@pytest.fixture
def client(cfg: AdminServerConfig) -> TestClient:
    return TestClient(build_admin_app(cfg))


def _seed_rollout(
    store: SqliteStateStore,
    *,
    rollout_id: str,
    template: str = "obs-t",
    status: RolloutStatus = RolloutStatus.RUNNING,
    node_id: str = "node-A",
    final_reward: float = 0.0,
    created_offset_s: float = 0.0,
    last_touched_offset_s: float | None = None,
    sandbox_id: str | None = None,
    reason: str | None = None,
) -> RolloutRecord:
    if last_touched_offset_s is None:
        last_touched_offset_s = created_offset_s
    record = RolloutRecord(
        rollout_id=rollout_id, template=template, status=status,
        reason=reason, node_id=node_id, sandbox_id=sandbox_id,
        final_reward=final_reward,
        created_at=time.time() - created_offset_s,
        last_touched_at=time.time() - last_touched_offset_s,
    )
    store.insert_rollout(record)
    return record


def _seed_sandbox(
    store: SqliteStateStore,
    *,
    sandbox_id: str,
    node_id: str,
    image: str | None = "im/obs-t:1",
) -> None:
    store.insert_sandbox(
        SandboxRecord(
            sandbox_id=sandbox_id, backend="docker",
            backend_ref=f"cid-{sandbox_id}",
            stub_endpoint="tcp://127.0.0.1:0",
            template="obs-t", image=image, node_id=node_id,
        )
    )


def _manifest() -> TemplateManifest:
    return TemplateManifest(
        name="obs-t", version="0.1", digest="sha256:obs-t",
        image="im/obs-t:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def _open_and_seal_run(
    runs_root: Path, rollout_id: str, *, n_steps: int = 2, final_reward: float = 0.5,
) -> None:
    sink = PlatformJsonlSink(runs_root)
    sink.open(rollout_id=rollout_id, manifest=_manifest(), init={}, node_id="node-A")
    for idx in range(n_steps):
        sink.record_step(rollout_id, Step(
            index=idx, action={"a": idx}, obs={"o": idx}, reward=0.0,
            done=(idx == n_steps - 1), truncated=False,
            info={"info_key": "info_val"}, ts=float(idx),
        ))
    sink.seal(
        rollout_id=rollout_id, status=RolloutStatus.FINISHED,
        reason=None, final_reward=final_reward, metadata={},
    )


def _write_coordinator_log(
    runs_root: Path, rollout_id: str, events: list[tuple[str, str]],
) -> None:
    run_dir = runs_root / "2026-01-01" / rollout_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f'{{"ts":"{ts}","event":"{event}","payload":{{}}}}\n'
        for event, ts in events
    ]
    (run_dir / "coordinator.log").write_text("".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Bind guard (spec 19)
# ──────────────────────────────────────────────────────────────────────────────


def test_admin_bind_guard_rejects_public_without_opt_in(
    state_db: Path, runs_root: Path,
) -> None:
    bad = AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="0.0.0.0", port=0, allow_public=False,
    )
    with pytest.raises(AdminBindError, match="public address"):
        AdminServer(config=bad).start()


def test_admin_bind_guard_refuses_no_auth_public_even_with_opt_in(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.3 (P1.x slice 3): a public bind needs an auth surface. The
    bind guard still refuses ``--admin-allow-public`` when no
    :class:`TokenStore` is wired (or it's empty) — the operator must
    issue at least one credential before exposing the panel.
    """
    bad = AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="0.0.0.0", port=0, allow_public=True,
    )
    with pytest.raises(AdminBindError, match="no auth configured"):
        AdminServer(config=bad).start()


def test_admin_bind_guard_allows_public_with_token_store(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.3: with ``--admin-allow-public`` AND a populated TokenStore,
    the public bind is now allowed. Basic auth gates every non-static
    request via the middleware."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="127.0.0.1", port=0, allow_public=True,
        token_store=store,
    )
    # Use loopback here so the test still binds in CI; the guard
    # short-circuits on loopback. The real assertion is that a public
    # host *with* a token store would also pass — exercised by
    # `_enforce_bind_guard` directly.
    from xrlenv.admin.server import AdminServerConfig as _Cfg
    from xrlenv.admin.server import _enforce_bind_guard
    public_cfg = _Cfg(
        state_db=state_db, runs_root=runs_root,
        host="0.0.0.0", port=0, allow_public=True,
        token_store=store,
    )
    _enforce_bind_guard(public_cfg)  # does not raise
    _ = cfg  # silence unused — kept for parity with other tests.


def test_admin_loopback_bind_serves_dual_stack(
    state_db: Path, runs_root: Path,
) -> None:
    """Operator-found regression (2026-05-11, Slurm cluster + VS Code
    Remote): with the admin server bound on IPv4 ``127.0.0.1`` only,
    VS Code's port-forward dialed ``::1`` and got ``ERR_EMPTY_RESPONSE``
    because nothing listened on IPv6. The fix is dual-stack: bind on
    BOTH ``127.0.0.1`` AND ``::1`` for any loopback host.

    This test starts a real AdminServer with the default loopback
    host and asserts ``/healthz`` is reachable via both families.
    """
    import socket
    import urllib.request

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="127.0.0.1", port=0,
    )
    srv = AdminServer(config=cfg)
    try:
        srv.start()
        port = srv.port
        assert port > 0

        # IPv4 (the primary bind).
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=5.0,
        ).read().decode()
        assert body == "ok"

        # IPv6 (the sibling bind) — skip if the host has no IPv6 stack.
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            s.connect(("::1", port))
            s.close()
            ipv6_reachable = True
        except OSError:
            ipv6_reachable = False
        if ipv6_reachable:
            body = urllib.request.urlopen(
                f"http://[::1]:{port}/healthz", timeout=5.0,
            ).read().decode()
            assert body == "ok"
    finally:
        srv.stop()


def test_admin_bind_guard_refuses_public_with_only_rpc_tokens(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.3 follow-up: a TokenStore that holds only ``node`` /
    ``consumer`` tokens cannot satisfy any admin route — every request
    would 401. The guard refuses up front so the operator gets a clear
    error at startup, not a mysteriously locked-out browser."""
    from xrlenv.admin.server import AdminServerConfig as _Cfg
    from xrlenv.admin.server import _enforce_bind_guard
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("node", "n-tok")
    store.add("consumer", "c-tok")
    public_cfg = _Cfg(
        state_db=state_db, runs_root=runs_root,
        host="0.0.0.0", port=0, allow_public=True,
        token_store=store,
    )
    with pytest.raises(AdminBindError, match="no admin-capable tokens"):
        _enforce_bind_guard(public_cfg)


def test_admin_bind_guard_accepts_public_with_viewer_only(
    state_db: Path, runs_root: Path,
) -> None:
    """A viewer-only TokenStore is enough to pass the guard — read-only
    public exposure is a valid posture for a status-page deployment."""
    from xrlenv.admin.server import AdminServerConfig as _Cfg
    from xrlenv.admin.server import _enforce_bind_guard
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    public_cfg = _Cfg(
        state_db=state_db, runs_root=runs_root,
        host="0.0.0.0", port=0, allow_public=True,
        token_store=store,
    )
    _enforce_bind_guard(public_cfg)  # does not raise.


def test_admin_bind_guard_allows_loopback(state_db: Path, runs_root: Path) -> None:
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="127.0.0.1", port=0,
    )
    srv = AdminServer(config=cfg)
    try:
        srv.start()
        assert srv.port > 0
    finally:
        srv.stop()


def test_admin_bind_guard_allows_localhost(state_db: Path, runs_root: Path) -> None:
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="localhost", port=0,
    )
    srv = AdminServer(config=cfg)
    try:
        srv.start()
    finally:
        srv.stop()


# ──────────────────────────────────────────────────────────────────────────────
# /healthz
# ──────────────────────────────────────────────────────────────────────────────


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.text == "ok"


# ──────────────────────────────────────────────────────────────────────────────
# /builds (P1.6.d)
# ──────────────────────────────────────────────────────────────────────────────


def test_builds_with_no_state_db_renders_hint(cfg: AdminServerConfig) -> None:
    """When state.db doesn't exist, the page renders a populate hint
    instead of a 500."""
    cfg = cfg.model_copy(update={"state_db": cfg.state_db.parent / "nope.db"})
    client = TestClient(build_admin_app(cfg))
    r = client.get("/builds")
    assert r.status_code == 200
    assert "No <code>state.db</code>" in r.text


def test_api_build_apply_returns_503_when_no_coordinator(
    state_db: Path, runs_root: Path,
) -> None:
    """When ``cfg.build_coordinator`` is None, the apply endpoint
    returns 503 — the admin server is reachable but no cluster-wide
    dispatch is wired."""
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "benchmarks": [{
                "name": "x", "selection": {"smoke": True},
            }],
        },
    })
    assert r.status_code == 503
    assert "build coordinator not wired" in r.text


def test_api_build_apply_dry_run_returns_placement(
    state_db: Path, runs_root: Path,
) -> None:
    """Dry-run path returns the placement synchronously without
    persisting anything."""
    from xrlenv.control.image_planner import PlacementResult, PlanAssignment

    placement = PlacementResult(
        assignments=(
            PlanAssignment(
                image_ref="x:1", node_id="n1",  # type: ignore[arg-type]
                benchmark="b", size_bytes=1024,
            ),
        ),
        assignments_by_node={"n1": (  # type: ignore[dict-item]
            PlanAssignment(
                image_ref="x:1", node_id="n1",  # type: ignore[arg-type]
                benchmark="b", size_bytes=1024,
            ),
        )},
    )

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            from xrlenv.control.build_coordinator import BuildOutcome

            return BuildOutcome(
                plan_id="abcd1234", status="dry_run",
                placement=placement,
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        build_coordinator=_FakeCoordinator(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "benchmarks": [{
                "name": "b", "selection": {"smoke": True},
            }],
        },
        "dry_run": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["plan_id"] == "abcd1234"
    assert body["status"] == "dry_run"
    assert len(body["placement"]) == 1
    assert body["placement"][0]["image_ref"] == "x:1"


def test_api_build_apply_threads_eager_to_coordinator(
    state_db: Path, runs_root: Path,
) -> None:
    """Audit P1.6.g-M1 fix: ``eager`` from the request body must reach
    ``coordinator.apply()`` on both the dry-run path and the
    background apply task. Without this, ``xrlenv build apply
    --connect-host --eager`` silently degrades to opportunistic mode.
    """
    from xrlenv.control.image_planner import PlacementResult

    captured: dict[str, Any] = {}

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            captured.update(kw)
            from xrlenv.control.build_coordinator import BuildOutcome

            return BuildOutcome(
                plan_id="abcd1234", status="dry_run",
                placement=PlacementResult(
                    assignments=(), assignments_by_node={},
                ),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        build_coordinator=_FakeCoordinator(),
    )
    client = TestClient(build_admin_app(cfg))

    # Dry-run with eager=True.
    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "benchmarks": [{"name": "b", "selection": {"smoke": True}}],
        },
        "dry_run": True,
        "eager": True,
    })
    assert r.status_code == 200
    assert captured.get("eager") is True

    # Dry-run default (eager omitted) should be opportunistic.
    captured.clear()
    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "benchmarks": [{"name": "b", "selection": {"smoke": True}}],
        },
        "dry_run": True,
    })
    assert r.status_code == 200
    assert captured.get("eager") is False


def test_api_build_apply_threads_fill_missing_to_dry_run_coordinator(
    state_db: Path, runs_root: Path,
) -> None:
    """Audit M2 (2026-05-12): ``fill_missing`` from the request body
    must reach ``coordinator.apply()`` on the DRY-RUN path. Without
    this, ``xrlenv build apply --connect-host --fill-missing
    --dry-run`` previews the normal placement instead of the
    missing-only subset, misleading the operator about what the
    real apply would do."""
    from xrlenv.control.image_planner import PlacementResult

    captured: dict[str, Any] = {}

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            captured.update(kw)
            from xrlenv.control.build_coordinator import BuildOutcome

            return BuildOutcome(
                plan_id="abcd1234", status="dry_run",
                placement=PlacementResult(
                    assignments=(), assignments_by_node={},
                ),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        build_coordinator=_FakeCoordinator(),
    )
    client = TestClient(build_admin_app(cfg))

    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "entries": [
                {
                    "image_ref": "x/y:1",
                    "context_source": {"type": "registry"},
                    "placement": {
                        "size_hint_bytes": 1_000_000_000,
                        "size_hint_source": "heuristic",
                    },
                },
            ],
        },
        "dry_run": True,
        "fill_missing": True,
    })
    assert r.status_code == 200
    assert captured.get("fill_missing") is True


def test_api_build_apply_completed_plan_with_fill_missing_bypasses_no_op(
    state_db: Path, runs_root: Path,
) -> None:
    """Audit M2 (2026-05-12): a completed plan_id whose images were
    later evicted must still be reconcilable via ``--fill-missing``.
    The completed-plan idempotency short-circuit (which returns
    ``no_op_already_completed``) must NOT fire when fill_missing=True,
    because the operator's explicit intent is "the cluster drifted;
    reconcile."""
    import asyncio

    from xrlenv.control.state import SqliteStateStore

    captured: dict[str, Any] = {}
    apply_called = asyncio.Event()

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            captured.update(kw)
            apply_called.set()
            from xrlenv.control.build_coordinator import BuildOutcome
            from xrlenv.control.image_planner import PlacementResult

            return BuildOutcome(
                plan_id="dummy", status="completed",
                placement=PlacementResult(
                    assignments=(), assignments_by_node={},
                ),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        build_coordinator=_FakeCoordinator(),
    )

    # Pre-seed a ``completed`` plan record for the plan_id our body
    # will hash to. We need a stable plan_id, so write the plan
    # first then compute the id, then write the row.
    from xrlenv.control.build_plan import BuildPlan, compute_plan_id
    plan_body = {
        "version": 1,
        "entries": [
            {
                "image_ref": "x/y:1",
                "context_source": {"type": "registry"},
                "placement": {
                    "size_hint_bytes": 1_000_000_000,
                    "size_hint_source": "heuristic",
                },
            },
        ],
    }
    plan = BuildPlan.model_validate(plan_body)
    plan_id = compute_plan_id(plan)

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id=plan_id, applied_by="x",
        plan_json=plan.model_dump_json(exclude_none=True),
        name=plan.name,
    )
    state.update_build_plan_status(plan_id, "completed")
    state.close()

    client = TestClient(build_admin_app(cfg))

    # WITHOUT fill_missing: completed → no-op (existing contract).
    r = client.post("/api/build/apply", json={
        "plan": plan_body,
    })
    assert r.status_code == 200
    assert r.json()["status"] == "no_op_already_completed"
    assert not apply_called.is_set()

    # WITH fill_missing: bypasses the no-op; spawns the background
    # task; returns 202.
    r = client.post("/api/build/apply", json={
        "plan": plan_body,
        "fill_missing": True,
    })
    assert r.status_code == 202
    # The background task will eventually fire coordinator.apply
    # with fill_missing=True. We don't await it (the TestClient
    # would block), but we can verify it was kicked off.


def test_api_build_apply_returns_202_and_kicks_off_background_task(
    state_db: Path, runs_root: Path,
) -> None:
    """Non-dry-run POST returns 202 immediately + the plan_id; the
    coordinator runs in a background asyncio task."""
    import anyio

    apply_called = anyio.Event()

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            apply_called.set()
            from xrlenv.control.build_coordinator import BuildOutcome

            return BuildOutcome(
                plan_id="deadbeef", status="completed", placement=None,
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        build_coordinator=_FakeCoordinator(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "benchmarks": [{
                "name": "b", "selection": {"smoke": True},
            }],
        },
    })
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "in_flight"
    assert "plan_id" in body
    # The TestClient drives the event loop synchronously, so by the
    # time we get here the background task has scheduled. anyio's
    # Event was set inside _FakeCoordinator.apply.


def test_api_build_apply_rejects_invalid_plan(
    state_db: Path, runs_root: Path,
) -> None:
    """Schema validation failures surface as 400, not 500."""

    class _FakeCoordinator:
        async def apply(self, *a, **k):
            raise AssertionError("apply should not be invoked")

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        build_coordinator=_FakeCoordinator(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/build/apply", json={
        "plan": {"version": 1, "replication": -1, "benchmarks": []},
    })
    assert r.status_code == 400
    assert "validation" in r.text.lower()


def test_api_build_plan_status_404_on_unknown(
    state_db: Path, runs_root: Path,
) -> None:
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/api/build/plans/no-such-plan-id")
    assert r.status_code == 404


def test_api_build_plan_status_returns_persisted_snapshot(
    state_db: Path, runs_root: Path,
) -> None:
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="aabbccdd", applied_by="op", plan_json="{}",
    )
    state.record_assignment(BuildAssignmentRecord(
        plan_id="aabbccdd", node_id="n1", image_ref="x:1",
        benchmark="b", status="done",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="aabbccdd", node_id="n1", image_ref="x:2",
        benchmark="b", status="failed", error="oops",
    ))
    state.update_build_plan_status("aabbccdd", "partial_failure")
    state.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/api/build/plans/aabbccdd")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "partial_failure"
    assert body["assignment_count"] == 2
    assert body["per_status"] == {"done": 1, "failed": 1}
    refs = {a["image_ref"]: a for a in body["assignments"]}
    assert refs["x:1"]["status"] == "done"
    assert refs["x:2"]["error"] == "oops"


def test_api_build_cancel_404_on_unknown_plan(
    state_db: Path, runs_root: Path,
) -> None:
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/cancel", json={"plan_id": "no-such-plan-id"},
    )
    assert r.status_code == 404


def test_api_build_cancel_400_on_short_prefix(
    state_db: Path, runs_root: Path,
) -> None:
    """Prefix <4 chars is rejected — same UX guard as the CLI's
    ``_resolve_plan_id`` so the operator can't accidentally fan
    a cluster-wide cancel out from a 2-char typo."""
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/build/cancel", json={"plan_id": "ab"})
    assert r.status_code == 400
    assert "at least 4 chars" in r.text


def test_api_build_cancel_already_terminal_is_noop(
    state_db: Path, runs_root: Path,
) -> None:
    """Cancel against a plan that is already ``completed`` is a no-op
    — returns 200 with cancelled_count=0 + a note. No cluster-side
    fanout happens (which would be wrong, since there are no
    in-flight builds to interrupt)."""
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="completed-plan-aabb", applied_by="op", plan_json="{}",
    )
    state.update_build_plan_status("completed-plan-aabb", "completed")
    state.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/cancel", json={"plan_id": "completed-plan-aabb"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["cancelled_count"] == 0
    assert "already terminal" in body["note"]


def test_api_build_cancel_dispatches_to_each_in_flight_node(
    state_db: Path, runs_root: Path,
) -> None:
    """Cancel of an in_flight plan with mixed assignment statuses:

    - ``pending`` rows are marked ``cancelled`` directly (no wire fanout).
    - ``building`` rows trigger a CancelBuildImageCommand to the owning
      node via cfg.node_lookup; on success they're marked ``cancelled``.
    - ``done`` rows are untouched.
    - The plan record itself transitions to ``cancelled``.
    """
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="in-flight-plan-bbcc", applied_by="op",
        plan_json="{}", name="cancel-me",
    )
    state.update_build_plan_status("in-flight-plan-bbcc", "in_flight")
    state.record_assignment(BuildAssignmentRecord(
        plan_id="in-flight-plan-bbcc", node_id="n1", image_ref="x:1",
        benchmark="b", status="building",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="in-flight-plan-bbcc", node_id="n2", image_ref="x:2",
        benchmark="b", status="pending",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="in-flight-plan-bbcc", node_id="n3", image_ref="x:3",
        benchmark="b", status="done",
    ))
    state.close()

    # node_lookup returns a fake transport for n1; n2/n3 don't need
    # a transport (n2 is pending → no wire call; n3 is done → not
    # touched).
    fake_calls: list[str] = []

    class _FakeTransport:
        async def cancel_build_image(
            self, *, image_ref: str, timeout_s: float = 30.0,
        ) -> tuple[str, str]:
            fake_calls.append(image_ref)
            return ("ok", "")

    def _node_lookup(node_id: str):
        if node_id == "n1":
            return _FakeTransport()
        return None

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=_node_lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/cancel", json={"plan_id": "in-flight-plan-bbcc"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["plan_id"] == "in-flight-plan-bbcc"
    assert body["status"] == "cancelled"
    # 1 pending → cancelled, 1 building → cancelled (via wire) = 2.
    assert body["cancelled_count"] == 2
    assert body["errors"] == []
    # The wire-call list confirms only the building image_ref was
    # dispatched (not the done one, not the pending one).
    assert fake_calls == ["x:1"]

    # Reload state and check rows landed.
    state = SqliteStateStore(state_db)
    try:
        plan = state.get_build_plan("in-flight-plan-bbcc")
        assert plan is not None
        assert plan.status == "cancelled"
        rows = {a.image_ref: a for a in state.list_assignments("in-flight-plan-bbcc")}
        assert rows["x:1"].status == "cancelled"
        assert rows["x:2"].status == "cancelled"
        assert rows["x:3"].status == "done"   # untouched
    finally:
        state.close()


def test_api_build_cancel_reconciles_orphaned_rows_on_already_cancelled_plan(
    state_db: Path, runs_root: Path,
) -> None:
    """Re-cancelling an already-cancelled plan reconciles rows a prior
    cancel's dispatch race left non-terminal — the "cancelled plan still
    shows N building" bug. ``building`` / ``registered`` rows are swept to
    ``cancelled`` (terminal rows untouched), with NO wire fanout."""
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="stuck-cancelled-1234", applied_by="op", plan_json="{}",
    )
    state.update_build_plan_status("stuck-cancelled-1234", "cancelled")
    state.record_assignment(BuildAssignmentRecord(
        plan_id="stuck-cancelled-1234", node_id="n1", image_ref="x:1",
        benchmark="b", status="building",   # orphaned by the prior race
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="stuck-cancelled-1234", node_id="n2", image_ref="x:2",
        benchmark="b", status="registered",  # overflow never dispatched
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="stuck-cancelled-1234", node_id="n3", image_ref="x:3",
        benchmark="b", status="done",
    ))
    state.close()

    wire_calls: list[str] = []

    class _FakeTransport:
        async def cancel_build_image(
            self, *, image_ref: str, timeout_s: float = 30.0,
        ) -> tuple[str, str]:
            wire_calls.append(image_ref)
            return ("ok", "")

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _n: _FakeTransport(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/cancel", json={"plan_id": "stuck-cancelled-1234"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cancelled"
    assert body["cancelled_count"] == 2          # building + registered
    assert "reconciled" in body["note"]
    assert wire_calls == []                       # reconcile-only, no fanout

    state = SqliteStateStore(state_db)
    try:
        rows = {
            a.image_ref: a
            for a in state.list_assignments("stuck-cancelled-1234")
        }
        assert rows["x:1"].status == "cancelled"
        assert rows["x:2"].status == "cancelled"
        assert rows["x:3"].status == "done"       # terminal untouched
    finally:
        state.close()


def test_api_build_cancel_sweeps_registered_overflow_rows(
    state_db: Path, runs_root: Path,
) -> None:
    """The main cancel flow's reconcile sweep also cancels ``registered``
    overflow rows (deferred entries never dispatched) that the
    pending/building-only handling would otherwise leave dangling under a
    cancelled plan."""
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="overflow-cancel-5678", applied_by="op", plan_json="{}",
    )
    state.update_build_plan_status("overflow-cancel-5678", "in_flight")
    state.record_assignment(BuildAssignmentRecord(
        plan_id="overflow-cancel-5678", node_id="n1", image_ref="r:1",
        benchmark="b", status="registered",
    ))
    state.close()

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _n: None,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/cancel", json={"plan_id": "overflow-cancel-5678"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"

    state = SqliteStateStore(state_db)
    try:
        rows = {
            a.image_ref: a
            for a in state.list_assignments("overflow-cancel-5678")
        }
        assert rows["r:1"].status == "cancelled"  # reconciled, not dangling
    finally:
        state.close()


def test_api_build_cancel_records_per_node_errors_but_marks_plan_cancelled(
    state_db: Path, runs_root: Path,
) -> None:
    """When a node's transport raises (or its cancel returns failed),
    the per-(node, image) error is captured in the response but the
    overall cancel still completes (plan + other assignments transition
    to ``cancelled``). HTTP status is 200 — the operator-visible state
    is the JSON body, not the HTTP code.

    Audit fix (post-2ebdaab): the failed-fanout assignment row MUST
    NOT stay at ``building`` — that would leave a row claiming the
    cluster is doing work it isn't doing. It transitions to
    ``failed`` with an error message naming the cancel-dispatch
    reason, so state.db stays self-consistent (no ``building`` rows
    under a ``cancelled`` plan).
    """
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="mixed-failure-ccdd", applied_by="op", plan_json="{}",
    )
    state.update_build_plan_status("mixed-failure-ccdd", "in_flight")
    state.record_assignment(BuildAssignmentRecord(
        plan_id="mixed-failure-ccdd", node_id="n1", image_ref="x:1",
        benchmark="b", status="building",
    ))
    state.close()

    class _BrokenTransport:
        async def cancel_build_image(
            self, *, image_ref: str, timeout_s: float = 30.0,
        ) -> tuple[str, str]:
            raise RuntimeError("network blip mid-cancel")

    def _node_lookup(node_id: str):
        if node_id == "n1":
            return _BrokenTransport()
        return None

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=_node_lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/cancel", json={"plan_id": "mixed-failure-ccdd"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cancelled"   # plan-level still flips
    assert body["cancelled_count"] == 0     # no assignment was cancelled
    assert len(body["errors"]) == 1
    err = body["errors"][0]
    assert err["node_id"] == "n1"
    assert err["image_ref"] == "x:1"
    assert "RuntimeError" in err["error"]
    assert "network blip" in err["error"]

    # State.db: the previously-``building`` row is now ``failed``,
    # NOT lingering at ``building``.
    state2 = SqliteStateStore(state_db)
    try:
        rows = {a.image_ref: a for a in state2.list_assignments("mixed-failure-ccdd")}
        assert rows["x:1"].status == "failed"
        assert rows["x:1"].error is not None
        assert "cancel dispatch raised" in rows["x:1"].error
        assert "network blip" in rows["x:1"].error
    finally:
        state2.close()


def test_api_build_cancel_persists_failed_row_when_node_disconnected(
    state_db: Path, runs_root: Path,
) -> None:
    """Audit fix (post-2ebdaab): when the node has no live transport
    (``node_lookup`` returns None — node disconnected), the cancel
    dispatch can't be issued, but the assignment row still moves
    off ``building``. Persisted as ``failed`` with the
    "no live transport" reason."""
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="disconnected-ddee", applied_by="op", plan_json="{}",
    )
    state.update_build_plan_status("disconnected-ddee", "in_flight")
    state.record_assignment(BuildAssignmentRecord(
        plan_id="disconnected-ddee", node_id="ghost-node",
        image_ref="x:gone", benchmark="b", status="building",
    ))
    state.close()

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _node_id: None,  # always disconnected
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/cancel", json={"plan_id": "disconnected-ddee"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cancelled"
    assert body["cancelled_count"] == 0
    assert len(body["errors"]) == 1
    assert "no live transport" in body["errors"][0]["error"]

    state2 = SqliteStateStore(state_db)
    try:
        rows = {a.image_ref: a for a in state2.list_assignments("disconnected-ddee")}
        assert rows["x:gone"].status == "failed"
        assert rows["x:gone"].error is not None
        assert "cancel could not be issued" in rows["x:gone"].error
        assert "no live transport" in rows["x:gone"].error
    finally:
        state2.close()


def test_api_build_calibrate_aggregates_max_size_per_image_ref(
    state_db: Path, runs_root: Path,
) -> None:
    """Calibrate walks each connected node's ``report_images``
    snapshot, takes the max size across nodes per image_ref, and
    leaves unmeasured image_refs in the ``unmeasured`` list."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    # Two connected nodes in state.db.
    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.record_node_connected("n2", backends=["docker"])
    state.close()

    # n1 has both images at slightly different sizes; n2 has one larger.
    class _Transport:
        def __init__(self, report: NodeImageReport) -> None:
            self._report = report

        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return self._report

    n1 = _Transport(NodeImageReport(
        images=[
            ImageStateRecord(
                name="my/a:1", tier="cold", size_bytes=1_000_000,
                in_use_count=0, last_used_at=None, pinned=False,
            ),
            ImageStateRecord(
                name="my/b:1", tier="cold", size_bytes=2_000_000,
                in_use_count=0, last_used_at=None, pinned=False,
            ),
        ],
    ))
    n2 = _Transport(NodeImageReport(
        images=[
            ImageStateRecord(
                # Bigger on n2 (e.g. base layer differs); calibrate
                # should pick the max.
                name="my/a:1", tier="cold", size_bytes=1_500_000,
                in_use_count=0, last_used_at=None, pinned=False,
            ),
        ],
    ))

    def _node_lookup(node_id: str):
        return {"n1": n1, "n2": n2}.get(node_id)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=_node_lookup,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {
        "version": 1,
        "entries": [
            {
                "image_ref": "my/a:1",
                "context_source": {"type": "registry"},
                "placement": {
                    "size_hint_bytes": 999_999,
                    "size_hint_source": "heuristic",
                },
            },
            {
                "image_ref": "my/b:1",
                "context_source": {"type": "registry"},
                "placement": {
                    "size_hint_bytes": 999_999,
                    "size_hint_source": "heuristic",
                },
            },
            {
                "image_ref": "my/c-not-built-yet:1",
                "context_source": {"type": "registry"},
                "placement": {
                    "size_hint_bytes": 999_999,
                    "size_hint_source": "heuristic",
                },
            },
        ],
    }
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    # max(1_000_000, 1_500_000) = 1_500_000 for my/a:1; my/b:1 only on n1.
    assert body["calibrated"]["my/a:1"] == 1_500_000
    assert body["calibrated"]["my/b:1"] == 2_000_000
    assert body["unmeasured"] == ["my/c-not-built-yet:1"]
    assert body["nodes_queried"] == 2
    assert body["nodes_unreachable"] == []


def test_registry_agnostic_ref_strips_registry_host_only() -> None:
    """The normalization that lets calibrate match a plan's bare
    ``image_ref`` against a node's registry-qualified pulled tag.
    A leading segment is a registry host only when it carries ``.`` /
    ``:`` / is ``localhost`` (Docker's own rule) — a Docker-Hub-relative
    repo component (``library/…``) must survive untouched."""
    from xrlenv.admin.server import _registry_agnostic_ref as norm

    # host:port private registry → stripped (the webarena incident).
    assert norm(
        "ip-10-0-5-6:5011/xrlenv-webarena-infinity/substrate:1ca77813",
    ) == "xrlenv-webarena-infinity/substrate:1ca77813"
    # localhost + dotted-host registries → stripped.
    assert norm("localhost:5000/foo/bar:0.1") == "foo/bar:0.1"
    assert norm("registry.example.com/foo/bar:1") == "foo/bar:1"
    # Docker-Hub-relative repo: first segment is NOT a registry → kept.
    assert norm("library/python:3.12-slim") == "library/python:3.12-slim"
    # Already registry-agnostic (the plan ref) and bare name:tag → kept.
    assert norm(
        "xrlenv-webarena-infinity/substrate:1ca77813",
    ) == "xrlenv-webarena-infinity/substrate:1ca77813"
    assert norm("python:3.12") == "python:3.12"


def test_api_build_calibrate_matches_registry_qualified_node_tag(
    state_db: Path, runs_root: Path,
) -> None:
    """Regression: a node that *pulled* an image from a private registry
    reports it under the registry-qualified tag (``host:5011/repo:tag``),
    while the build plan carries the bare, registry-agnostic
    ``image_ref`` (``repo:tag``). Calibrate must match the two and credit
    the measured size to the plan's bare ref.

    Before the registry-agnostic normalization the strict
    ``img.name not in wanted_refs`` membership test missed every
    pulled-from-registry image, so calibrate reported "0 measured /
    1 unmeasured" while the admin /images page plainly listed the image
    on the node (the ``xrlenv-webarena-infinity/substrate:1ca77813``
    incident, 2026-06-28)."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    bare = "xrlenv-webarena-infinity/substrate:1ca77813"
    qualified = "ip-10-0-5-6:5011/" + bare

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[
                ImageStateRecord(
                    name=qualified, tier="cold", size_bytes=1_700_000_000,
                    in_use_count=1, last_used_at=None, pinned=False,
                ),
                # An unrelated registry-qualified image must NOT match
                # the plan ref just because both are registry-stripped.
                ImageStateRecord(
                    name="ip-10-0-5-6:5011/some/other:9", tier="cold",
                    size_bytes=42, in_use_count=0, last_used_at=None,
                    pinned=False,
                ),
            ])

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {
        "version": 1,
        "entries": [{
            "image_ref": bare,
            "context_source": {"type": "registry"},
            "placement": {
                "size_hint_bytes": 1_089_748_433,
                "size_hint_source": "registry-probe",
            },
        }],
    }
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    # Credited to the *bare* plan ref (the key the CLI walks to rewrite
    # the YAML), not the node's qualified tag; nothing unmeasured.
    assert body["calibrated"] == {bare: 1_700_000_000}
    assert body["unmeasured"] == []
    assert body["nodes_queried"] == 1
    assert body["nodes_unreachable"] == []


def test_api_build_calibrate_matches_digest_pulled_image_by_repo(
    state_db: Path, runs_root: Path,
) -> None:
    """Regression: the control plane digest-pins ``:tag`` → ``@sha256:...``
    (invariant 4), so a node PULLS by digest and holds the image under a
    registry-qualified ``@sha256`` (untagged) ref — never the plan's ``:main``.
    The tag-preserving match misses it, so calibrate reported it ``unmeasured``
    despite the image plainly being on disk (the 7-measured / 195-unmeasured
    terminalworld case, 2026-07-17). Calibrate now falls back to a repo-path
    match when the repo is unambiguous, crediting the plan's tagged ref."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    plan_ref = "terminalworld-verified/tw_99185:main"
    node_digest = (
        "ip-10-0-5-6:5011/terminalworld-verified/tw_99185@sha256:7cd3964f"
    )

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[ImageStateRecord(
                name=node_digest, tier="cold", size_bytes=1_500_000_000,
                in_use_count=0, last_used_at=None, pinned=False,
            )])

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [{
        "image_ref": plan_ref,
        "context_source": {"type": "registry"},
        "placement": {"size_hint_bytes": 1, "size_hint_source": "registry-probe"},
    }]}
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] == {plan_ref: 1_500_000_000}  # via repo-path fallback
    assert body["unmeasured"] == []


def test_api_build_calibrate_digest_fallback_skips_ambiguous_repo(
    state_db: Path, runs_root: Path,
) -> None:
    """The digest-pull fallback fires ONLY when the repo is unambiguous. If the
    plan lists two tags of the same repo, a digest-pulled node image can't be
    attributed to one of them, so it stays unmeasured — no over-crediting."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[ImageStateRecord(
                name="reg:5011/ns/img@sha256:abc", tier="cold", size_bytes=999,
                in_use_count=0, last_used_at=None, pinned=False,
            )])

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": "ns/img:v1", "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "registry-probe"}},
        {"image_ref": "ns/img:v2", "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "registry-probe"}},
    ]}
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    body = r.json()
    # No per-image digest reported (older node / no RepoDigests) → the digest
    # index stays empty, the digest-match fallback can't fire, and the ambiguous
    # repo credits neither. The resolved-digest case is covered by
    # ``test_api_build_calibrate_digest_match_resolves_ambiguous_repo`` below.
    assert body["calibrated"] == {}  # ambiguous repo → credits neither
    assert sorted(body["unmeasured"]) == ["ns/img:v1", "ns/img:v2"]


_ECR_REPO = "public.ecr.aws/d3j8x8q7/swe-bench-202605"


def test_api_build_calibrate_digest_match_resolves_ambiguous_repo(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest-match fallback attributes digest-pinned images to their plan
    entries even when many tags share ONE repository — the SWE-bench / DeepSWE
    shape (113 tags of ``d3j8x8q7/swe-bench-202605``) that disables the
    repo-path fallback (ambiguous repo). Each still-unmeasured plan ref is
    resolved to its manifest digest and looked up in the digests the nodes
    reported; digest is the canonical identity, so the shared-repo ambiguity
    the tag/repo matchers choke on doesn't arise."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    ref_a = f"{_ECR_REPO}:tagAAA-v1.1"
    ref_b = f"{_ECR_REPO}:tagBBB-v1.1"
    ref_c = f"{_ECR_REPO}:tagCCC-v1.1"  # resolvable, but on no node → unmeasured
    dig_a = f"{_ECR_REPO}@sha256:{'a' * 64}"
    dig_b = f"{_ECR_REPO}@sha256:{'b' * 64}"
    dig_c = f"{_ECR_REPO}@sha256:{'c' * 64}"

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            # Held digest-pinned/untagged: the node names the image by its
            # config id (no repo), but the manifest digest rides in the
            # plumbed-through ``digest`` field (Docker RepoDigests).
            return NodeImageReport(images=[
                ImageStateRecord(
                    name=f"sha256:{'0' * 64}", tier="cold",
                    size_bytes=2_000_000_000, shared_size_bytes=500_000_000,
                    in_use_count=0, last_used_at=None, pinned=False,
                    digest=dig_a,
                ),
                ImageStateRecord(
                    # unique == 0 (every layer shared) — load-bearing: must
                    # still land as measured, not dropped.
                    name=f"sha256:{'1' * 64}", tier="cold",
                    size_bytes=1_800_000_000, shared_size_bytes=1_800_000_000,
                    in_use_count=0, last_used_at=None, pinned=False,
                    digest=dig_b,
                ),
            ])

    class _FakeResolver:
        async def resolve(self, ref: str) -> str:
            return {ref_a: dig_a, ref_b: dig_b, ref_c: dig_c}[ref]

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: _FakeResolver())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": r, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}}
        for r in (ref_a, ref_b, ref_c)
    ]}
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    # ref_a: unique = 2.0G - 0.5G = 1.5G; ref_b: unique 0 (fully shared).
    assert body["calibrated"][ref_a] == 1_500_000_000
    assert body["calibrated"][ref_b] == 0
    # ref_c resolves but no node holds its digest → stays unmeasured.
    assert body["unmeasured"] == [ref_c]


def test_api_build_calibrate_same_digest_dedup_counts_once(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When multiple plan refs resolve to the SAME manifest digest (two tags of
    one byte-identical image), all are marked measured, but the physical unique
    footprint is counted ONCE — the size lands on the lexicographically-first
    ref and the duplicates get 0, so downstream FFD never double-counts one
    image's storage."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    ref_x = f"{_ECR_REPO}:x-v1.1"
    ref_y = f"{_ECR_REPO}:y-v1.1"
    same = f"{_ECR_REPO}@sha256:{'e' * 64}"  # both tags → this one image

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[ImageStateRecord(
                name=f"sha256:{'0' * 64}", tier="cold",
                size_bytes=1_200_000_000, shared_size_bytes=0,
                in_use_count=0, last_used_at=None, pinned=False,
                digest=same,
            )])

    class _FakeResolver:
        async def resolve(self, ref: str) -> str:
            return {ref_x: same, ref_y: same}[ref]

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: _FakeResolver())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": r, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}}
        for r in (ref_x, ref_y)
    ]}
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    # Both measured (neither unmeasured), but the 1.2G is counted once:
    # min(ref_x, ref_y) == ref_x carries it, ref_y is the zeroed duplicate.
    assert body["unmeasured"] == []
    assert body["calibrated"][ref_x] == 1_200_000_000
    assert body["calibrated"][ref_y] == 0


def test_api_build_calibrate_digest_match_skips_when_resolver_disabled(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the digest resolver is disabled (``resolver_from_env`` → ``None``,
    e.g. ``XRLENV_REGISTRY_DIGEST_RESOLVE=off``), the digest-match fallback is
    skipped and ambiguous-repo images stay unmeasured — a graceful degrade to
    the prior tag/repo-only behavior, never a hard failure."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    ref_a = f"{_ECR_REPO}:tagAAA-v1.1"
    ref_b = f"{_ECR_REPO}:tagBBB-v1.1"

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[ImageStateRecord(
                name=f"sha256:{'0' * 64}", tier="cold",
                size_bytes=2_000_000_000, shared_size_bytes=0,
                in_use_count=0, last_used_at=None, pinned=False,
                digest=f"{_ECR_REPO}@sha256:{'a' * 64}",
            )])

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: None)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": r, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}}
        for r in (ref_a, ref_b)
    ]}
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] == {}
    assert sorted(body["unmeasured"]) == sorted([ref_a, ref_b])


def test_api_build_calibrate_repo_fallback_skips_different_explicit_tag(
    state_db: Path, runs_root: Path,
) -> None:
    """Regression (2026-07-17 tb2.1 mixed-tag over-credit): the repo-path
    fallback must fire ONLY for a digest / untagged node ref (a real digest
    pull), never for a *different explicit tag* of the same repo. Here the node
    holds a stale ``:20251031`` (from a prior build) while the plan pins the
    single, unambiguous ``:20260403``. Without the ``has_explicit_tag`` guard the
    fallback credited the newer entry with the stale image's size (often 0-byte,
    fully layer-shared). It must stay unmeasured — the newer tag isn't on disk."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            # Stale sibling tag present; the plan's newer tag is NOT on disk.
            return NodeImageReport(images=[ImageStateRecord(
                name="alexgshaw/compile-compcert:20251031", tier="cold",
                size_bytes=0, in_use_count=0, last_used_at=None, pinned=False,
            )])

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_ref = "alexgshaw/compile-compcert:20260403"
    plan_body = {"version": 1, "entries": [{
        "image_ref": plan_ref,
        "context_source": {"type": "registry"},
        "placement": {"size_hint_bytes": 42, "size_hint_source": "registry-probe"},
    }]}
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] == {}          # stale sibling tag NOT credited
    assert body["unmeasured"] == [plan_ref]  # newer tag genuinely absent


def test_api_build_calibrate_writes_unique_size_when_shared_known(
    state_db: Path, runs_root: Path,
) -> None:
    """When ``report_images()`` carries ``shared_size_bytes`` for an
    entry, calibrate writes the **unique** size (= size - shared) — the
    incremental disk a node pays to cache the image when its base
    layers are already present from a sibling image. The legacy
    behavior (write ``size_bytes`` verbatim) over-counts shared layers
    for plans where many images share a common base (swebench-verified
    Python base, harbor runtime, etc.) and inflates FFD reservation."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    # my/a:1: 1.5 GB total, 1.0 GB shared → unique 0.5 GB. That's
    # the number calibrate should emit, not the 1.5 GB legacy value.
    # my/b:1: 800 MB total, no shared info → falls back to size_bytes
    # (legacy behavior for backends that don't surface SharedSize).
    class _Transport:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[
                ImageStateRecord(
                    name="my/a:1", tier="cold",
                    size_bytes=1_500_000_000,
                    shared_size_bytes=1_000_000_000,
                    in_use_count=0, last_used_at=None, pinned=False,
                ),
                ImageStateRecord(
                    name="my/b:1", tier="cold",
                    size_bytes=800_000_000,
                    shared_size_bytes=None,
                    in_use_count=0, last_used_at=None, pinned=False,
                ),
            ])

    def _node_lookup(node_id: str):
        return _Transport() if node_id == "n1" else None

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=_node_lookup,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {
        "version": 1,
        "entries": [
            {
                "image_ref": "my/a:1",
                "context_source": {"type": "registry"},
                "placement": {
                    "size_hint_bytes": 999_999_999,
                    "size_hint_source": "heuristic",
                },
            },
            {
                "image_ref": "my/b:1",
                "context_source": {"type": "registry"},
                "placement": {
                    "size_hint_bytes": 999_999_999,
                    "size_hint_source": "heuristic",
                },
            },
        ],
    }
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    # Unique = 1.5 GB - 1.0 GB = 500 MB. NOT 1.5 GB.
    assert body["calibrated"]["my/a:1"] == 500_000_000
    # No shared info → legacy size_bytes verbatim.
    assert body["calibrated"]["my/b:1"] == 800_000_000


def test_api_build_calibrate_records_zero_unique_when_image_is_pure_shared(
    state_db: Path, runs_root: Path,
) -> None:
    """An image whose every layer is shared with a sibling has
    ``unique = size - shared = 0``. The node still reports it via
    ``report_images``; calibrate must record it (with measured=0) so
    it lands in ``calibrated`` rather than slipping into
    ``unmeasured``. Regression for the
    'docker-image-inspect-finds-it-but-calibrate-says-unmeasured'
    bug: the prior ``measured > prior`` check with ``prior``
    defaulting to 0 silently dropped these refs from the calibrated
    set. The fix uses ``prior is None`` so a measured-zero result
    is still recorded as "node has this image; its incremental disk
    cost is zero."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    # my/thin-wrapper:1 is 800 MB of layers, ALL shared with siblings.
    # unique = 800 MB - 800 MB = 0. Must still land in calibrated.
    class _Transport:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[
                ImageStateRecord(
                    name="my/thin-wrapper:1", tier="cold",
                    size_bytes=800_000_000,
                    shared_size_bytes=800_000_000,
                    in_use_count=0, last_used_at=None, pinned=False,
                ),
            ])

    def _node_lookup(node_id: str):
        return _Transport() if node_id == "n1" else None

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=_node_lookup,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {
        "version": 1,
        "entries": [
            {
                "image_ref": "my/thin-wrapper:1",
                "context_source": {"type": "registry"},
                "placement": {
                    "size_hint_bytes": 999_999_999,
                    "size_hint_source": "heuristic",
                },
            },
        ],
    }
    r = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert r.status_code == 200
    body = r.json()
    # Must be in `calibrated` (with measured=0), NOT in `unmeasured`.
    assert "my/thin-wrapper:1" in body["calibrated"]
    assert body["calibrated"]["my/thin-wrapper:1"] == 0
    assert body["unmeasured"] == []


def test_api_build_calibrate_503_when_no_node_lookup(
    state_db: Path, runs_root: Path,
) -> None:
    """Calibrate without a node_lookup wired returns 503 — the
    admin server is reachable but can't talk to nodes (typical
    standalone-admin process spec, not a usable runtime for
    calibrate)."""
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.post(
        "/api/build/calibrate",
        json={"plan": {
            "version": 1,
            "entries": [
                {
                    "image_ref": "my/a:1",
                    "context_source": {"type": "registry"},
                    "placement": {"size_hint_bytes": 1024},
                },
            ],
        }},
    )
    assert r.status_code == 503


def test_api_build_calibrate_400_on_invalid_plan(
    state_db: Path, runs_root: Path,
) -> None:
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _n: None,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/build/calibrate", json={"plan": "not-a-dict"})
    assert r.status_code == 400


# ── New edge-case tests for the digest-match fallback (2026-07-21) ────────────


def test_api_build_calibrate_digest_index_takes_max_across_nodes(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``digest_sizes`` index (which feeds the digest-match fallback) must
    take the **max** unique size across nodes, not the last-seen value.

    Two nodes both hold the same digest-pinned (untagged) image, but report
    different unique sizes (e.g. their shared base layers differ). The max
    should be the value that lands in the calibrated output — same conservative
    principle as the tag-match path."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.record_node_connected("n2", backends=["docker"])
    state.close()

    # Both tags of one repo → ambiguous repo, forces the digest-match fallback.
    ref_a = f"{_ECR_REPO}:tag-alpha-v1"
    ref_b = f"{_ECR_REPO}:tag-beta-v1"
    dig_a = f"{_ECR_REPO}@sha256:{'a' * 64}"
    dig_b = f"{_ECR_REPO}@sha256:{'b' * 64}"

    # n1 reports dig_a at 1.5 GB unique; n2 reports the SAME digest at 2.0 GB.
    # The digest_sizes index must resolve to 2.0 GB (max), not 1.5 GB (first seen).
    n1_report = NodeImageReport(images=[ImageStateRecord(
        name=f"sha256:{'0' * 64}", tier="cold",
        size_bytes=1_500_000_000, shared_size_bytes=0,
        in_use_count=0, last_used_at=None, pinned=False,
        digest=dig_a,
    )])
    n2_report = NodeImageReport(images=[
        ImageStateRecord(
            # Same digest as n1 but larger unique (heavier shared base).
            name=f"sha256:{'0' * 64}", tier="cold",
            size_bytes=2_000_000_000, shared_size_bytes=0,
            in_use_count=0, last_used_at=None, pinned=False,
            digest=dig_a,
        ),
        ImageStateRecord(
            name=f"sha256:{'1' * 64}", tier="cold",
            size_bytes=800_000_000, shared_size_bytes=0,
            in_use_count=0, last_used_at=None, pinned=False,
            digest=dig_b,
        ),
    ])

    transports: dict[str, Any] = {}
    for nid, rep in (("n1", n1_report), ("n2", n2_report)):
        class _T:
            def __init__(self, r: NodeImageReport) -> None:
                self._r = r
            async def report_images(
                self, *, include_shared_size: bool = False,
            ) -> NodeImageReport:
                return self._r
        transports[nid] = _T(rep)

    class _FakeResolver:
        async def resolve(self, ref: str) -> str:
            return {ref_a: dig_a, ref_b: dig_b}[ref]

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: _FakeResolver())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: transports.get(nid),
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": r, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}}
        for r in (ref_a, ref_b)
    ]}
    resp = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert resp.status_code == 200
    body = resp.json()
    # dig_a max is 2.0 GB (n2 > n1); dig_b only on n2 → 800 MB.
    # Both plan refs are distinct digests → no same-digest dedup applies.
    assert body["calibrated"][ref_a] == 2_000_000_000
    assert body["calibrated"][ref_b] == 800_000_000
    assert body["unmeasured"] == []
    assert body["nodes_queried"] == 2


def test_api_build_calibrate_no_double_count_when_tag_match_and_digest_share(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan ref already measured by TAG match is NOT re-entered by the
    digest-match fallback even if it shares a digest with a still-unmeasured
    sibling.

    Scenario: ref_A is matched by tag (lands in max_size immediately). ref_B
    shares the same manifest digest as ref_A but is not held by any node under
    a tag that matches ref_B's plan entry. After the node loop:
      - still_unmeasured = {ref_B}  (ref_A already measured → excluded)
      - digest-match fallback fires for ref_B only, giving it digest_sizes[dig]

    The concern is that ref_A might be credited TWICE: once from tag match and
    once from the same-digest dedup zeroing logic. Confirm it is not."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    # Unambiguous repo for ref_A — tag-match will credit it.
    ref_a = "ns/single-repo:tag-a"
    # ref_B has a DIFFERENT repo but happens to resolve to the same digest as ref_A.
    ref_b = "ns/alias-repo:tag-b"
    shared_dig = f"ns/single-repo@sha256:{'c' * 64}"

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            # Node holds ref_A by tag (direct tag match fires).
            # It also carries the same image under its config-id ref with
            # a digest that would also match ref_B.
            return NodeImageReport(images=[
                ImageStateRecord(
                    name="ns/single-repo:tag-a", tier="cold",
                    size_bytes=1_000_000_000, shared_size_bytes=0,
                    in_use_count=0, last_used_at=None, pinned=False,
                    digest=shared_dig,
                ),
            ])

    class _FakeResolver:
        async def resolve(self, ref: str) -> str:
            # Both plan refs resolve to the same manifest digest.
            return {ref_a: shared_dig, ref_b: shared_dig}[ref]

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: _FakeResolver())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": r, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}}
        for r in (ref_a, ref_b)
    ]}
    resp = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert resp.status_code == 200
    body = resp.json()
    # ref_A was measured by tag match at 1.0 GB.
    assert body["calibrated"][ref_a] == 1_000_000_000
    # ref_B was measured by the digest-match fallback (same digest → 1.0 GB);
    # same-digest dedup between ref_A and ref_B does NOT fire here because
    # ref_A is NOT in still_unmeasured — it was already in max_size.
    # The dedup only groups refs that ALL went through the digest-fallback path.
    assert body["calibrated"][ref_b] == 1_000_000_000
    # Crucially: ref_A must NOT be zeroed out. Both are present at full size.
    assert body["unmeasured"] == []


def test_api_build_calibrate_resolver_exception_leaves_one_ref_unmeasured(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the resolver raises for one plan ref, that ref stays unmeasured
    while other refs whose resolver calls succeed are correctly attributed.

    Best-effort contract: a single registry timeout / 404 must not abort
    the entire fallback or leave other refs unmeasured."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    ref_ok = f"{_ECR_REPO}:tag-ok-v1"
    ref_fail = f"{_ECR_REPO}:tag-fail-v1"
    dig_ok = f"{_ECR_REPO}@sha256:{'d' * 64}"

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[ImageStateRecord(
                name=f"sha256:{'9' * 64}", tier="cold",
                size_bytes=900_000_000, shared_size_bytes=0,
                in_use_count=0, last_used_at=None, pinned=False,
                digest=dig_ok,
            )])

    class _FlakyResolver:
        async def resolve(self, ref: str) -> str:
            if ref == ref_fail:
                raise RuntimeError("registry unreachable for this ref")
            return {ref_ok: dig_ok}[ref]

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: _FlakyResolver())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": r, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}}
        for r in (ref_ok, ref_fail)
    ]}
    resp = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert resp.status_code == 200
    body = resp.json()
    # ref_ok resolved successfully → credited from digest_sizes.
    assert body["calibrated"][ref_ok] == 900_000_000
    # ref_fail resolver raised → stays unmeasured (graceful degrade).
    assert body["unmeasured"] == [ref_fail]


def test_api_build_calibrate_unmatched_node_digest_is_ignored(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node image carrying a digest that no plan ref resolves to must be
    silently ignored — it accumulates in digest_sizes but never ends up
    in the calibrated output."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    ref_plan = f"{_ECR_REPO}:tag-plan-v1"
    dig_plan = f"{_ECR_REPO}@sha256:{'f' * 64}"
    # A "foreign" digest present on the node, completely unrelated to the plan.
    dig_foreign = f"{_ECR_REPO}@sha256:{'9' * 64}"

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[
                ImageStateRecord(
                    name=f"sha256:{'a' * 64}", tier="cold",
                    size_bytes=500_000_000, shared_size_bytes=0,
                    in_use_count=0, last_used_at=None, pinned=False,
                    digest=dig_plan,
                ),
                # Foreign image: huge, but not in the plan at all.
                ImageStateRecord(
                    name=f"sha256:{'b' * 64}", tier="cold",
                    size_bytes=99_000_000_000, shared_size_bytes=0,
                    in_use_count=0, last_used_at=None, pinned=False,
                    digest=dig_foreign,
                ),
            ])

    class _FakeResolver:
        async def resolve(self, ref: str) -> str:
            return {ref_plan: dig_plan}[ref]

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: _FakeResolver())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": ref_plan, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}},
    ]}
    resp = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert resp.status_code == 200
    body = resp.json()
    # Only the plan ref appears in calibrated, not the foreign image.
    assert set(body["calibrated"].keys()) == {ref_plan}
    assert body["calibrated"][ref_plan] == 500_000_000
    assert body["unmeasured"] == []


def test_api_build_calibrate_digest_path_falls_back_to_size_bytes_when_shared_none(
    state_db: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the digest-match path, when ``shared_size_bytes`` is None
    (backend doesn't surface it), the calibrated size falls back to ``size_bytes``
    — the same legacy behavior as the tag-match path.

    This verifies that the ``include_shared_size=False`` / None branch is
    correctly handled on the digest-index accrual path, not just the tag-match
    path already tested by ``test_api_build_calibrate_writes_unique_size_when_shared_known``."""
    from xrlenv.control import registry_resolver as _rr
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.close()

    # Two tags of one repo → ambiguous, forces the digest-match fallback.
    ref_a = f"{_ECR_REPO}:no-shared-a"
    ref_b = f"{_ECR_REPO}:no-shared-b"
    dig_a = f"{_ECR_REPO}@sha256:{'7' * 64}"
    dig_b = f"{_ECR_REPO}@sha256:{'8' * 64}"

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[
                ImageStateRecord(
                    name=f"sha256:{'c' * 64}", tier="cold",
                    size_bytes=1_100_000_000,
                    shared_size_bytes=None,  # backend doesn't supply it
                    in_use_count=0, last_used_at=None, pinned=False,
                    digest=dig_a,
                ),
                ImageStateRecord(
                    name=f"sha256:{'d' * 64}", tier="cold",
                    size_bytes=600_000_000,
                    shared_size_bytes=None,
                    in_use_count=0, last_used_at=None, pinned=False,
                    digest=dig_b,
                ),
            ])

    class _FakeResolver:
        async def resolve(self, ref: str) -> str:
            return {ref_a: dig_a, ref_b: dig_b}[ref]

    monkeypatch.setattr(_rr, "resolver_from_env", lambda: _FakeResolver())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _Transport() if nid == "n1" else None,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {"version": 1, "entries": [
        {"image_ref": r, "context_source": {"type": "registry"},
         "placement": {"size_hint_bytes": 1, "size_hint_source": "heuristic"}}
        for r in (ref_a, ref_b)
    ]}
    resp = client.post("/api/build/calibrate", json={"plan": plan_body})
    assert resp.status_code == 200
    body = resp.json()
    # No shared_size_bytes → falls back to size_bytes verbatim.
    assert body["calibrated"][ref_a] == 1_100_000_000
    assert body["calibrated"][ref_b] == 600_000_000
    assert body["unmeasured"] == []


def test_api_image_evict_fans_out_and_aggregates(
    state_db: Path, runs_root: Path,
) -> None:
    """``POST /api/image/evict`` fans an eviction to every connected node
    and aggregates per-node outcomes. Each node matches the (possibly
    bare) ref against the registry-qualified tag it holds."""
    from xrlenv.control.state import SqliteStateStore
    from xrlenv.node.image_cache import EvictOutcome

    state = SqliteStateStore(state_db)
    state.record_node_connected("n1", backends=["docker"])
    state.record_node_connected("n2", backends=["docker"])
    state.record_node_connected("n3", backends=["docker"])
    state.close()

    bare = "wai/substrate:1ca77813"
    qualified = "reg:5011/" + bare

    class _Transport:
        def __init__(self, outcome: EvictOutcome) -> None:
            self._outcome = outcome
            self.calls: list[tuple[str, bool]] = []

        async def evict_image(
            self, *, image_ref: str, force: bool = False,
            timeout_s: float = 30.0,
        ) -> EvictOutcome:
            self.calls.append((image_ref, force))
            return self._outcome

    n1 = _Transport(EvictOutcome(
        status="evicted", reclaimed_bytes=1_000_000_000,
        removed=(qualified,),
    ))
    n2 = _Transport(EvictOutcome(status="absent"))
    # n3 disconnected mid-flight → no live transport.
    transports = {"n1": n1, "n2": n2}

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: transports.get(nid),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/image/evict", json={"image_ref": bare})
    assert r.status_code == 200
    body = r.json()
    assert body["nodes_queried"] == 3
    assert body["nodes_evicted"] == 1
    assert body["total_reclaimed_bytes"] == 1_000_000_000
    by_node = {res["node_id"]: res for res in body["results"]}
    assert by_node["n1"]["status"] == "evicted"
    assert by_node["n1"]["removed"] == [qualified]
    assert by_node["n2"]["status"] == "absent"
    assert by_node["n3"]["status"] == "unreachable"
    # The bare plan ref + default force=False reached the node verbatim
    # (the node does the registry-agnostic matching, not the admin).
    assert n1.calls == [(bare, False)]


def test_api_image_evict_400_on_missing_ref(
    state_db: Path, runs_root: Path,
) -> None:
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _n: None,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.post("/api/image/evict", json={"force": True})
    assert r.status_code == 400


def _basic_header(username: str, password: str) -> str:
    """A legacy ``Authorization: Basic ...`` header. B7.4 stopped honoring
    basic auth for the browser (it replays cached creds with no logout); this
    helper now only feeds the regression test proving such a cached header is
    ignored in favor of the cookie session."""
    import base64
    return "Basic " + base64.b64encode(
        f"{username}:{password}".encode(),
    ).decode("ascii")


def _session_cookie(token: str) -> dict[str, str]:
    """Browser auth via the B7.4 cookie session (set by ``POST /login``)."""
    return {"Cookie": f"{_SESSION_COOKIE}={token}"}


def test_admin_browse_open_when_token_store_empty(
    state_db: Path, runs_root: Path,
) -> None:
    """Loopback bind always bypasses auth (the SSH-tunnel is the
    protection boundary). Empty TokenStore on loopback: open."""
    from xrlenv.control.security import TokenStore
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        token_store=TokenStore(),  # empty.
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/")
    assert r.status_code == 200
    r = client.get("/nodes")
    assert r.status_code == 200


def test_admin_loopback_bypasses_auth_even_with_tokens_issued(
    state_db: Path, runs_root: Path,
) -> None:
    """Refined contract (2026-05-11): loopback binds bypass auth
    regardless of TokenStore state. The protection boundary is the
    operator's SSH tunnel, not HTTP basic auth — adding basic auth
    on top of loopback adds zero security uplift and inflicts a real
    UX cost (Chrome / Safari auto-upgrade ``http://localhost`` to
    https, which the HTTP-only admin server can't speak)."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        # Default host = "127.0.0.1" → loopback bypass engages.
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    # No credentials at all — still 200 on a loopback bind.
    r = client.get("/nodes")
    assert r.status_code == 200
    r = client.get("/")
    assert r.status_code == 200
    # Even malformed/garbage credentials don't trip a 401 on loopback.
    r = client.get(
        "/nodes",
        headers={"Authorization": "Basic this-is-not-base64!"},
    )
    assert r.status_code == 200


def test_admin_read_route_rejects_missing_credentials(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.4 (public bind): a credential-less *API* GET (Accept: */*) returns
    401 with a ``Bearer`` challenge — never ``Basic`` (which would pop the
    browser's cached-credential dialog the cookie session replaces)."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/nodes", headers={"accept": "*/*"})
    assert r.status_code == 401
    challenge = r.headers.get("www-authenticate", "")
    assert "Bearer" in challenge
    assert "Basic" not in challenge


def test_admin_browser_get_redirects_to_login(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.4 (public bind): a credential-less *browser* GET (Accept: text/html)
    is 303-redirected to the sign-in page with the original target preserved in
    ``?next=`` — not a JSON 401 or a basic-auth popup."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.get("/nodes", headers={"accept": "text/html"})
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=%2Fnodes"
    # The sign-in page itself is reachable without credentials.
    assert client.get("/login", headers={"accept": "text/html"}).status_code == 200


def test_admin_login_sets_session_and_grants_read(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.4: POST /login with a viewer token mints the session cookie and the
    subsequent read request (cookie carried by the jar) is authorized."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.post(
        "/login", content="token=read_view-tok",
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "accept": "text/html",
        },
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    assert _SESSION_COOKIE in r.cookies
    # The jar now carries the cookie → read route authorized.
    assert client.get("/nodes", headers={"accept": "text/html"}).status_code == 200


def test_admin_login_rejects_unknown_token(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.4: POST /login with a token the store doesn't know re-renders the
    form with 401 and sets no session cookie."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.post(
        "/login", content="token=bogus",
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "accept": "text/html",
        },
    )
    assert r.status_code == 401
    assert _SESSION_COOKIE not in r.cookies


def test_admin_logout_clears_session_and_switches_token(
    state_db: Path, runs_root: Path,
) -> None:
    """The core fix: after logout the session is gone, so the operator can sign
    in again as a *different* token. (Under browser-cached basic auth this was
    impossible — the browser kept replaying the first token.)"""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    form = {
        "content-type": "application/x-www-form-urlencoded",
        "accept": "text/html",
    }
    # Sign in as viewer, confirm session works.
    client.post("/login", content="token=read_view-tok", headers=form)
    assert client.get("/nodes", headers={"accept": "text/html"}).status_code == 200
    # Log out → cookie cleared, browser GET bounces back to /login.
    r = client.post("/logout")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
    assert _SESSION_COOKIE not in client.cookies
    bounced = client.get("/nodes", headers={"accept": "text/html"})
    assert bounced.status_code == 303
    assert bounced.headers["location"].startswith("/login")
    # Now sign in as a *different* token (operator) — the switch works.
    client.post("/login", content="token=write_op-tok", headers=form)
    assert client.get("/nodes", headers={"accept": "text/html"}).status_code == 200


def test_admin_nav_shows_identity_and_logout_when_signed_in(
    state_db: Path, runs_root: Path,
) -> None:
    """The nav renders the signed-in identity (`owner · role`) and a POST
    /logout button on a cookie session, and shows neither on a loopback bind
    (no identity stashed) — so the logout affordance is present exactly when
    there's a session to end."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    public_cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True, token_store=store,
    )
    client = TestClient(build_admin_app(public_cfg))
    body = client.get("/nodes", headers=_session_cookie("write_op-tok")).text
    assert "· operator" in body  # identity label
    assert 'action="/logout"' in body and ">log out<" in body

    # Loopback bind: auth bypassed → no identity stashed → no logout affordance.
    loop_cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    loop_client = TestClient(build_admin_app(loop_cfg))
    loop_body = loop_client.get("/nodes").text
    assert 'action="/logout"' not in loop_body


def test_admin_login_next_is_open_redirect_guarded(
    state_db: Path, runs_root: Path,
) -> None:
    """``?next=`` only honors same-site absolute paths. A protocol-relative
    ``//evil.example`` (which a browser resolves to another host) or an absolute
    URL falls back to ``/`` — no open redirect off the post-login bounce."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    for evil in ("//evil.example/x", "https://evil.example", "javascript:alert(1)"):
        # On the form (GET): a hostile next never lands in the redirect target.
        r = client.get("/login", params={"next": evil}, headers={"accept": "text/html"})
        assert r.status_code == 200  # not signed in → renders the form
        assert evil not in r.text  # the hidden next field is sanitized to "/"
        # On submit (POST): the post-login 303 goes to "/", never off-site.
        r = client.post(
            "/login", content=f"token=write_op-tok&next={evil}",
            headers={
                "content-type": "application/x-www-form-urlencoded",
                "accept": "text/html",
            },
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/"
        client.cookies.clear()  # reset for the next iteration


def test_admin_read_route_accepts_viewer_cookie_session(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.4 (public bind): a viewer token in the session cookie authorizes a
    read route — the browser transport, replacing HTTP basic auth."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/nodes", headers=_session_cookie("read_view-tok"))
    assert r.status_code == 200


def test_admin_read_route_accepts_operator_bearer(
    state_db: Path, runs_root: Path,
) -> None:
    """Operator tokens have full access — they satisfy read routes too. The
    CLI / programmatic transport is ``Authorization: Bearer``."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get(
        "/nodes", headers={"Authorization": "Bearer write_op-tok"},
    )
    assert r.status_code == 200


def test_admin_write_route_rejects_viewer(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.4 (public bind): a viewer token cannot POST to write routes
    — the middleware returns 403. This is the load-bearing two-tier
    guarantee: sharing a `read_xxx` token never grants cluster-
    mutating power. Holds for both transports (cookie + bearer)."""
    from xrlenv.control.security import TokenStore

    class _FakeCoordinator:
        async def apply(self, *a, **k):
            from xrlenv.control.build_coordinator import BuildOutcome
            return BuildOutcome(
                plan_id="abc", status="dry_run", placement=None,
            )

    store = TokenStore()
    store.add("viewer", "read_view-tok")
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        build_coordinator=_FakeCoordinator(),
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {
        "plan": {
            "version": 1,
            "benchmarks": [{"name": "b", "selection": {"smoke": True}}],
        },
        "dry_run": True,
    }
    # Viewer (cookie session) → 403.
    r = client.post(
        "/api/build/apply", json=plan_body,
        headers=_session_cookie("read_view-tok"),
    )
    assert r.status_code == 403
    # The operator (bearer) can still write.
    r = client.post(
        "/api/build/apply", json=plan_body,
        headers={"Authorization": "Bearer write_op-tok"},
    )
    assert r.status_code == 200


def test_admin_session_rejects_unknown_cookie_token(
    state_db: Path, runs_root: Path,
) -> None:
    """A session cookie holding a token the store doesn't know is rejected
    (an API GET → 401)."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get(
        "/nodes",
        headers={**_session_cookie("not-a-real-token"), "accept": "*/*"},
    )
    assert r.status_code == 401


def test_admin_cached_basic_auth_is_ignored(
    state_db: Path, runs_root: Path,
) -> None:
    """Regression for the reported bug: a browser replaying creds it cached
    from the pre-B7.4 basic-auth flow must NOT authenticate. With no session
    cookie a cached ``Authorization: Basic`` header is ignored — an API GET
    gets 401, a browser GET is bounced to /login — so logout stays
    authoritative and the operator can switch tokens."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    cached = {"Authorization": _basic_header("operator", "write_op-tok")}
    # API caller with only the cached basic header → 401 (basic no longer honored).
    assert client.get(
        "/nodes", headers={**cached, "accept": "*/*"},
    ).status_code == 401
    # Browser caller with only the cached basic header → bounced to /login.
    r = client.get("/nodes", headers={**cached, "accept": "text/html"})
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


def test_admin_healthz_open_without_credentials(
    state_db: Path, runs_root: Path,
) -> None:
    """/healthz is open on public binds even with auth wired — load
    balancers need it."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/healthz")
    assert r.status_code == 200


def test_api_build_apply_requires_operator_token_when_store_wired(
    state_db: Path, runs_root: Path,
) -> None:
    """When TokenStore is wired and non-empty, build-apply requires
    Authorization: Bearer <operator-token>."""
    from xrlenv.control.security import TokenStore

    class _FakeCoordinator:
        async def apply(self, *a, **k):
            from xrlenv.control.build_coordinator import BuildOutcome

            return BuildOutcome(
                plan_id="abc", status="dry_run", placement=None,
            )

    store = TokenStore()
    store.add("operator", "op-secret-123")
    store.add("consumer", "consumer-secret-456")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        build_coordinator=_FakeCoordinator(),
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    plan_body = {
        "plan": {
            "version": 1,
            "benchmarks": [{
                "name": "b", "selection": {"smoke": True},
            }],
        },
        "dry_run": True,
    }

    # No auth header → 401.
    r = client.post("/api/build/apply", json=plan_body)
    assert r.status_code == 401

    # Wrong-role token (consumer instead of operator) → 403.
    r = client.post(
        "/api/build/apply", json=plan_body,
        headers={"Authorization": "Bearer consumer-secret-456"},
    )
    assert r.status_code == 403

    # Correct operator token → 200.
    r = client.post(
        "/api/build/apply", json=plan_body,
        headers={"Authorization": "Bearer op-secret-123"},
    )
    assert r.status_code == 200


def test_builds_renders_persisted_plan(
    state_db: Path, runs_root: Path,
) -> None:
    """A persisted plan + assignments render with status counts +
    failure expansion."""
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="abc123def456ghi", applied_by="cli", plan_json="{}",
    )
    state.record_assignment(BuildAssignmentRecord(
        plan_id="abc123def456ghi", node_id="n1", image_ref="x:1",
        benchmark="b", status="done",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="abc123def456ghi", node_id="n1", image_ref="x:2",
        benchmark="b", status="failed", error="nope",
    ))
    state.update_build_plan_status("abc123def456ghi", "partial_failure")
    state.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))

    # List page: short id + status. Per-row failure detail moved to
    # the detail page (the list view now only shows the high-level
    # benchmark + selection summary + progress count).
    r = client.get("/builds")
    assert r.status_code == 200
    list_body = r.text
    assert "abc123def456" in list_body
    assert "partial_failure" in list_body

    # Detail page: full per-assignment breakdown.
    r = client.get("/builds/abc123def456ghi")
    assert r.status_code == 200
    detail_body = r.text
    assert "n1" in detail_body
    assert "x:2" in detail_body
    assert "nope" in detail_body
    # Detail page exposes the canonical plan JSON.
    assert "show canonical plan JSON" in detail_body


def test_builds_renders_registered_and_evicted_states(
    state_db: Path, runs_root: Path,
) -> None:
    """P1.6.g step 5 (#79): the list and detail pages must surface the
    new ``registered`` (deferred — lazy on first rollout) and
    ``evicted`` (cache reclaim) states from the F1 enum, otherwise
    operators only see done/failed/building/pending and can't tell
    why a plan transitioned to ``completed`` while some images
    weren't pre-built."""
    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    state.record_build_plan(
        plan_id="def789abc012ghi", applied_by="cli", plan_json="{}",
    )
    state.record_assignment(BuildAssignmentRecord(
        plan_id="def789abc012ghi", node_id="n1", image_ref="x:1",
        benchmark="b", status="done",
    ))
    # Deferred (overflow) row + an evicted row to exercise both paths.
    state.record_assignment(BuildAssignmentRecord(
        plan_id="def789abc012ghi", node_id="n1", image_ref="x:huge",
        benchmark="b", status="registered",
    ))
    state.record_assignment(BuildAssignmentRecord(
        plan_id="def789abc012ghi", node_id="n2", image_ref="x:old",
        benchmark="b", status="evicted",
    ))
    state.update_build_plan_status("def789abc012ghi", "completed")
    state.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))

    list_body = client.get("/builds").text
    assert "1 deferred" in list_body
    assert "1 evicted" in list_body

    detail_body = client.get("/builds/def789abc012ghi").text
    assert "deferred" in detail_body
    assert "evicted" in detail_body
    assert "lazy on first rollout" in detail_body
    assert "cache reclaim" in detail_body


def test_build_detail_404_on_unknown_plan(
    state_db: Path, runs_root: Path,
) -> None:
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/builds/no-such-plan")
    assert r.status_code == 404


def test_builds_list_shows_plan_name_and_image_count(
    state_db: Path, runs_root: Path,
) -> None:
    """The list page surfaces the operator-supplied plan name in the
    main row column so plans associate with their YAML files at a
    glance instead of by opaque plan_id."""
    import json as _json

    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    plan_json = _json.dumps({
        "version": 1,
        "name": "terminal-bench-2-phase-0",
        "replication": 1,
        "benchmarks": [
            {"name": "terminal-bench-2", "selection": {"smoke": True}},
        ],
    })
    state.record_build_plan(
        plan_id="readable123", applied_by="cli", plan_json=plan_json,
        name="terminal-bench-2-phase-0",
    )
    for i in range(8):
        state.record_assignment(BuildAssignmentRecord(
            plan_id="readable123", node_id="n1",
            image_ref=f"terminal-bench-2/task-{i}:0.1",
            benchmark="terminal-bench-2", status="done",
        ))
    state.update_build_plan_status("readable123", "completed")
    state.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/builds")
    assert r.status_code == 200
    body = r.text
    # Plan name visible in the row instead of the (now-removed)
    # benchmarks-summary cell.
    assert "terminal-bench-2-phase-0" in body
    assert "8 image(s)" in body
    assert "8 / 8" in body
    # Timestamp rendered as readable absolute + relative form, not
    # raw Unix epoch.
    assert "2026-" in body or "2025-" in body
    assert any(s in body for s in ["s ago", "min ago", "h ago", "d ago"])


def test_builds_list_unnamed_plan_renders_placeholder(
    state_db: Path, runs_root: Path,
) -> None:
    """Plans that don't set a ``name`` (e.g. legacy rows persisted
    before the field landed) render as ``(unnamed)`` so the column
    never shows up empty."""
    import json as _json

    from xrlenv.control.state import (
        BuildAssignmentRecord,
        SqliteStateStore,
    )

    state = SqliteStateStore(state_db)
    plan_json = _json.dumps({
        "version": 1,
        "replication": 1,
        "benchmarks": [
            {"name": "x", "selection": {"smoke": True}},
        ],
    })
    state.record_build_plan(
        plan_id="legacy123", applied_by="cli", plan_json=plan_json,
        # no name=
    )
    state.record_assignment(BuildAssignmentRecord(
        plan_id="legacy123", node_id="n1",
        image_ref="x/y:0.1", benchmark="x", status="done",
    ))
    state.update_build_plan_status("legacy123", "completed")
    state.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/builds")
    assert r.status_code == 200
    assert "(unnamed)" in r.text


def test_admin_page_owned_facts_are_pairwise_disjoint() -> None:
    """The page ownership matrix is the executable IA contract."""
    pages = sorted(ADMIN_PAGE_OWNED_FACTS)
    for idx, left in enumerate(pages):
        for right in pages[idx + 1:]:
            overlap = ADMIN_PAGE_OWNED_FACTS[left] & ADMIN_PAGE_OWNED_FACTS[right]
            assert overlap == frozenset(), f"{left} and {right} both own {overlap}"


# ──────────────────────────────────────────────────────────────────────────────
# / overview
# ──────────────────────────────────────────────────────────────────────────────


def test_overview_handles_missing_state_db(cfg: AdminServerConfig) -> None:
    """When state.db doesn't exist yet, overview renders zeros, not 500s."""
    client = TestClient(build_admin_app(cfg))
    r = client.get("/")
    assert r.status_code == 200
    assert "Cluster overview" in r.text
    assert "No <code>state.db</code>" in r.text
    assert "images" in r.text
    assert 'href="/sandboxes"' in r.text
    assert 'href="/nodes"' in r.text
    assert "capacity" in r.text
    assert 'href="/images/cache"' in r.text
    assert 'href="/images/catalog"' in r.text
    assert "disk pressure by node" in r.text
    assert "warm images by cluster" in r.text


def test_overview_shows_cluster_info_when_configured(
    state_db: Path, runs_root: Path,
) -> None:
    """When the control-plane endpoint / registries are wired into the config,
    the overview renders a Cluster banner showing them. Holds even with no
    state.db (the values come from config, not the store)."""
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        control_plane_endpoint="10.0.1.5:50051",
        registry_mirror="http://10.0.1.5:5010",
        private_registry="10.0.1.5:5011",
    )
    client = TestClient(build_admin_app(cfg))
    body = client.get("/").text
    assert "<h2>Cluster</h2>" in body
    assert "10.0.1.5:50051" in body
    assert "http://10.0.1.5:5010" in body
    assert "10.0.1.5:5011" in body


def test_overview_hides_cluster_info_when_unset(cfg: AdminServerConfig) -> None:
    """With none of the cluster-info fields set (the default), the overview
    omits the Cluster banner entirely — no empty section."""
    client = TestClient(build_admin_app(cfg))
    assert "<h2>Cluster</h2>" not in client.get("/").text


def test_overview_cluster_info_is_partial(
    state_db: Path, runs_root: Path,
) -> None:
    """Only the configured rows appear; an unset registry is not rendered as a
    blank row."""
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        control_plane_endpoint="10.0.1.5:50051",  # only this one set
    )
    client = TestClient(build_admin_app(cfg))
    body = client.get("/").text
    assert "<h2>Cluster</h2>" in body
    assert "10.0.1.5:50051" in body
    assert "registry mirror" not in body
    assert "private registry" not in body


def test_nav_identity_prefers_display_name_over_owner_id(
    state_db: Path, runs_root: Path,
) -> None:
    """The nav shows the token's human-readable display_name when set, and
    falls back to the raw owner_id when it isn't."""
    from xrlenv.control.security import TokenStore, write_user_record
    secrets = runs_root.parent / "secrets"
    write_user_record(
        secrets / "users.json", token="named-tok", role="consumer",
        owner_id="alice", display_name="Alice Zhang",
    )
    store = TokenStore.load(secrets_root=secrets, env={})
    store.add("operator", "write_op-tok")  # no display_name → owner_id "default"
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True, token_store=store,
    )
    client = TestClient(build_admin_app(cfg))
    named = client.get("/nodes", headers=_session_cookie("named-tok")).text
    assert "Alice Zhang · consumer" in named
    assert "alice · consumer" not in named  # owner_id not used when name present
    # A token with no display_name falls back to its owner_id.
    unnamed = client.get("/nodes", headers=_session_cookie("write_op-tok")).text
    assert "default · operator" in unnamed


def test_active_page_nav_link_is_marked_active(cfg: AdminServerConfig) -> None:
    """Each top-level page must mark exactly its own nav link as active.

    The ``active_page`` context variable drives ``class="active"`` on the
    matching ``<a>`` element in base.html. This test pins the contract so
    a missing ``active_page`` assignment in a route handler is caught.
    """
    client = TestClient(build_admin_app(cfg))

    cases = [
        ("/", 'href="/" class="active"'),
        ("/health", 'href="/health" class="active"'),
        ("/nodes", 'href="/nodes" class="active"'),
        ("/capacity", 'href="/capacity" class="active"'),
    ]
    for path, expected_fragment in cases:
        r = client.get(path)
        assert r.status_code == 200, path
        assert expected_fragment in r.text, (
            f"{path}: expected nav fragment '{expected_fragment}' not found"
        )
        # Sanity: the compact images menu still renders on every page.
        assert "images" in r.text, path
        assert 'href="/images/cache"' in r.text, path
        assert 'href="/images/catalog"' in r.text, path


def test_overview_refresh_can_be_configured_from_query(
    cfg: AdminServerConfig,
) -> None:
    client = TestClient(build_admin_app(cfg))

    off = client.get("/", params={"refresh": "off"})
    assert off.status_code == 200
    assert 'http-equiv="refresh"' not in off.text
    assert '<option value="off" selected>off</option>' in off.text

    thirty = client.get("/", params={"refresh": "30"})
    assert thirty.status_code == 200
    assert 'http-equiv="refresh" content="30"' in thirty.text
    assert '<option value="30" selected>30s</option>' in thirty.text


def test_top_level_pages_default_to_no_auto_refresh(
    cfg: AdminServerConfig,
) -> None:
    """Auto-refresh is OFF by default on every page (``refresh_interval_s=0``).
    No ``<meta http-equiv=refresh>`` tag is emitted, the selector shows ``off``
    selected, and the note reads ``default off`` — the operator opts in per
    page via the selector (or ``?refresh=``)."""
    client = TestClient(build_admin_app(cfg))
    for path in (
        "/", "/health", "/rollouts", "/sandboxes", "/nodes",
        "/images/cache", "/capacity", "/images/catalog",
    ):
        r = client.get(path)
        assert r.status_code == 200, path
        assert 'http-equiv="refresh"' not in r.text, path
        assert '<option value="off" selected>off</option>' in r.text, path
        assert "default off" in r.text, path


def test_admin_refresh_interval_config_overrides_default(
    state_db: Path, runs_root: Path,
) -> None:
    """An operator can still make pages auto-refresh by default by setting a
    positive ``refresh_interval_s`` — then the meta tag is emitted and the
    matching option is pre-selected."""
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, refresh_interval_s=10,
    )
    client = TestClient(build_admin_app(cfg))
    body = client.get("/").text
    assert 'http-equiv="refresh" content="10' in body
    assert '<option value="10" selected>10s</option>' in body
    assert "default 10s" in body


def test_overview_counts_rollouts_and_sandboxes(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="r1", status=RolloutStatus.RUNNING)
    _seed_rollout(store, rollout_id="r2", status=RolloutStatus.FINISHED)
    _seed_rollout(
        store, rollout_id="r3", status=RolloutStatus.FAILED,
        reason="reward_failed",
    )
    _seed_sandbox(store, sandbox_id="sb1", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb2", node_id="node-B")
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/api/overview")
    payload = r.json()
    assert payload["rollout_running"] == 1
    assert payload["rollout_finished_1h"] == 1
    assert payload["rollout_failed_1h"] == 1
    # 2 case-1 sandboxes, 0 raw rollouts → 2 live containers.
    assert payload["container_count"] == 2
    assert payload["node_active"] == 2
    assert payload["state_db_present"] is True


def test_overview_node_count_includes_connected_but_idle_nodes(
    state_db: Path, runs_root: Path,
) -> None:
    """Operator-found regression (2026-05-11, fresh-cluster overview
    on a Slurm GPU node): the overview's ``node_count`` denominator
    (rendered as "X / Y (active / known)") only summed sandbox-active
    + rostered nodes, ignoring connected-but-idle nodes recorded in
    the NodeRegistry shadow. Operators saw "0 / 0 nodes" on overview
    even though the Nodes page listed two connected nodes. Fix
    unions the registry shadow into the denominator, matching what
    ``_nodes_blocking`` already does for the page rows.
    """
    store = SqliteStateStore(state_db)
    # Two nodes connected to the registry but neither owning a
    # sandbox + no nodes.yaml — the fresh-cluster shape.
    store.record_node_connected(
        node_id="gcp-node-1",
        stream_epoch="epoch-1", instance_id="inst-1",
        backends=["docker"],
    )
    store.record_node_connected(
        node_id="gcp-node-2",
        stream_epoch="epoch-2", instance_id="inst-2",
        backends=["docker"],
    )
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    payload = client.get("/api/overview").json()
    # Both nodes count in the "known" denominator, neither in the
    # active numerator (no sandboxes).
    assert payload["node_count"] == 2, (
        f"connected-but-idle nodes should count in node_count; "
        f"got payload={payload!r}"
    )
    assert payload["node_active"] == 0


def test_overview_counts_include_raw_rollouts(
    state_db: Path, runs_root: Path,
) -> None:
    """Operator-found regression (2026-05-13): the cluster overview's
    ``rollout_running`` / ``rollout_finished_1h`` / ``rollout_failed_1h``
    counted only case-1 sandbox-driven rollouts. With the slim pivot
    moving the dominant audience to case-2/3 raw harnesses
    (xrlenv.from_env() drop-in), operators ran 4 raw rollouts and saw
    the overview report 0 running, 0 finished, 0 failed while
    /rollouts/raw showed all 4 alive. Fix unions raw-rollout counts
    into the overview totals AND marks nodes hosting in-flight raw
    rollouts as active.
    """
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)

    # Case-1 baseline (the existing flow) — one running.
    _seed_rollout(store, rollout_id="r1", status=RolloutStatus.RUNNING)

    # Case-2/3 raw rollouts that the old overview ignored:
    now = time.time()
    cutoff = now - 1800.0  # 30 min ago — inside the 1h window.
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-running", status="running", image="busybox:1",
        node_id="gcp-node-1", created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-acquiring", status="acquiring", image="busybox:1",
        node_id="gcp-node-1", created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-released", status="released", image="busybox:1",
        node_id="gcp-node-2",
        created_at=cutoff, finished_at=cutoff + 10,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-failed", status="failed", image="busybox:1",
        node_id="gcp-node-2",
        created_at=cutoff, finished_at=cutoff + 5,
    ))
    # Released >1h ago — must NOT count in the rolling window.
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-old-released", status="released", image="busybox:1",
        node_id="gcp-node-3",
        created_at=now - 7200.0, finished_at=now - 7200.0 + 10,
    ))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    payload = client.get("/api/overview").json()

    # running = 1 case-1 RUNNING + 2 raw (running + acquiring).
    assert payload["rollout_running"] == 3, payload
    # containers = jobs with a live container now: 0 case-1 sandboxes
    # + 1 raw `running` (rr-running). rr-acquiring is still
    # cold-pulling — no container — so it is excluded.
    assert payload["container_count"] == 1, payload
    # finished_1h = 0 case-1 (test only seeded a RUNNING) + 1 raw
    # (rr-released within window). rr-old-released is excluded.
    assert payload["rollout_finished_1h"] == 1, payload
    # failed_1h = 0 case-1 + 1 raw (rr-failed within window).
    assert payload["rollout_failed_1h"] == 1, payload
    # Nodes hosting in-flight raw rollouts are active — gcp-node-1
    # (acquiring + running). gcp-node-2 only had a released/failed
    # so it's NOT active. The case-1 rollout in this fixture has no
    # sandbox so it doesn't contribute a node.
    assert payload["node_active"] == 1, payload


def test_overview_excludes_reaped_from_raw_failures(
    state_db: Path, runs_root: Path,
) -> None:
    """A ``reaped`` raw rollout (raw-GC deadline / liveness reclaim) must
    NOT count as a failure in the overview. Conflating reaps with failures
    was the 2026-06-10..12 false alarm the ``reaped`` status split fixes:
    123 GC reclaims looked like workload failures and inflated the rate."""
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)
    now = time.time()
    cutoff = now - 1800.0  # 30 min ago — inside the 1h window.
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-failed", status="failed", image="busybox:1",
        node_id="n1", created_at=cutoff, finished_at=cutoff + 5,
        error="exec-timeout cascade",
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-reaped", status="reaped", image="busybox:1",
        node_id="n1", created_at=cutoff, finished_at=cutoff + 5,
        error="session deadline exceeded (overdue 12s) — force-reaped",
    ))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    payload = client.get("/api/overview").json()
    # Only the genuine failure counts; the reap is excluded.
    assert payload["rollout_failed_1h"] == 1, payload
    # And the reap is not silently dropped as "finished" either.
    assert payload["rollout_finished_1h"] == 0, payload


def test_raw_list_renders_reaped_as_terminal(
    state_db: Path, runs_root: Path,
) -> None:
    """Audit P2: a ``reaped`` row carries ``finished_at`` and must render
    as terminal (finite duration, no ``live`` badge), not a forever-live
    session. Also pins that ``reaped`` is an accepted raw-status filter."""
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)
    now = time.time()
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-running", status="running", image="busybox:1",
        node_id="n1", created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-reaped", status="reaped", image="busybox:1",
        node_id="n1", created_at=now - 600.0, finished_at=now - 60.0,
        error="session deadline exceeded — force-reaped",
    ))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))

    # Unfiltered: both rows render, but exactly ONE is live (the running
    # one). Pre-fix the reaped row was also tagged live → count == 2.
    body = client.get("/rollouts/raw").text
    assert "status-reaped" in body
    assert body.count("live</span>") == 1, body

    # Filter to reaped: accepted as a status, returns only the reaped row,
    # rendered terminal (no live badge).
    resp = client.get("/rollouts/raw?raw_status=reaped")
    assert resp.status_code == 200
    filtered = resp.text
    assert "status-reaped" in filtered
    assert "status-running" not in filtered


def test_overview_excludes_capacity_rejected_from_raw_failures(
    state_db: Path, runs_root: Path,
) -> None:
    """A ``capacity_rejected`` raw rollout (scheduler declined to place within
    queue_timeout_s) must NOT count as a failure in the overview — it's
    backpressure the consumer typically retries. It gets its own tile instead.
    Same illusion the ``reaped`` split fixed, one class over: a paced-then-
    retried sweep would otherwise show a wall of ``failed``."""
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)
    now = time.time()
    cutoff = now - 1800.0  # 30 min ago — inside the 1h window.
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-failed", status="failed", image="busybox:1",
        node_id="n1", created_at=cutoff, finished_at=cutoff + 5,
        error="exec-timeout cascade",
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-paced-1", status="capacity_rejected", image="busybox:1",
        node_id="n1", created_at=cutoff, finished_at=cutoff + 5,
        error="capacity_rejected: pool at capacity …",
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-paced-2", status="capacity_rejected", image="busybox:1",
        node_id="n1", created_at=cutoff, finished_at=cutoff + 6,
        error="capacity_rejected: pool at capacity …",
    ))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    payload = client.get("/api/overview").json()
    # Only the genuine failure counts; the two paced declines are excluded …
    assert payload["rollout_failed_1h"] == 1, payload
    # … and surfaced on their own tile instead of vanishing.
    assert payload["rollout_capacity_rejected_1h"] == 2, payload
    assert payload["rollout_finished_1h"] == 0, payload


def test_raw_list_renders_capacity_rejected_as_terminal(
    state_db: Path, runs_root: Path,
) -> None:
    """A ``capacity_rejected`` row carries ``finished_at`` and renders terminal
    (finite duration, no ``live`` badge) with the neutral amber badge, and is
    an accepted raw-status filter."""
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)
    now = time.time()
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-running", status="running", image="busybox:1",
        node_id="n1", created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-paced", status="capacity_rejected", image="busybox:1",
        node_id="n1", created_at=now - 600.0, finished_at=now - 60.0,
        error="capacity_rejected: pool at capacity …",
    ))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))

    body = client.get("/rollouts/raw").text
    assert "status-capacity_rejected" in body
    assert body.count("live</span>") == 1, body  # only the running row is live

    resp = client.get("/rollouts/raw?raw_status=capacity_rejected")
    assert resp.status_code == 200
    filtered = resp.text
    assert "status-capacity_rejected" in filtered
    assert "status-running" not in filtered
    assert "live</span>" not in filtered


def test_resolve_image_include_defaults_to_all() -> None:
    """Operator-found UX gap (2026-05-11): the node-image-cache page
    defaulted to ``include=default`` (= "xrlenv-only-no-intermediates-
    no-foreign"), which on operator-driven docker-pull workloads
    rendered as "Showing 0 of N images" — operators thought the
    panel was broken. Fix: no-filter default is now ``"all"``;
    operators filter down via the dropdown when they want a slice.

    Explicit ``include=default`` still works for backwards-compat —
    any bookmarked filtered URL stays valid.
    """
    from xrlenv.admin.server import _resolve_image_include

    # No flags at all → show everything.
    assert _resolve_image_include(
        None, show_intermediate=0, show_external=0,
    ) == ("all", False, False)

    # Explicit ``include=default`` still resolves to the
    # xrlenv-only-no-extras view, so old bookmarks keep working.
    assert _resolve_image_include(
        "default", show_intermediate=0, show_external=0,
    ) == ("default", True, True)


# ──────────────────────────────────────────────────────────────────────────────
# /nodes
# ──────────────────────────────────────────────────────────────────────────────


def test_nodes_renders_rostered_plus_active(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(yaml.safe_dump({
        "nodes": [
            {"id": "node-A", "cloud": "gcp", "expected_address": "10.0.0.1"},
            {"id": "node-X", "cloud": "aws", "address": "10.0.0.2"},  # legacy key
        ]
    }))
    store = SqliteStateStore(state_db)
    _seed_sandbox(store, sandbox_id="sb1", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb2", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb3", node_id="node-only-active")
    store.close()

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, nodes_yaml=nodes_yaml, port=0,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/nodes")
    assert r.status_code == 200
    body = r.text
    # All three nodes (rostered + active) render.
    assert "node-A" in body and "node-X" in body and "node-only-active" in body
    # Legacy `address` key surfaces under the same column.
    assert "10.0.0.2" in body
    # node-A has 2 active sandboxes.
    assert "2" in body


def test_nodes_view_renders_attached_node_without_sandboxes(
    state_db: Path, runs_root: Path,
) -> None:
    """A node that just connected via gRPC (registry-mirrored to
    ``state.nodes`` but no sandbox rows yet) MUST appear in the admin
    /nodes view. Pre-fix, the page queried ``list_sandboxes()`` only,
    so freshly-attached idle nodes silently vanished and the page
    showed "No nodes recorded yet" while the CLI's ``xrlenv nodes``
    listed the node correctly. This test pins the registry-mirror
    union the fix introduces.
    """
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "node-fresh", backends=["docker"],
        stream_epoch="ep-1", instance_id="inst-1",
    )
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/nodes")
    assert r.status_code == 200
    body = r.text
    # Node appears even though no sandbox row exists.
    assert "node-fresh" in body
    # Status column carries the registry's value, not "absent".
    assert "connected" in body
    # The empty-state message must NOT render when nodes are present.
    assert "No nodes recorded yet" not in body


def test_nodes_view_marks_lost_nodes_distinct_from_connected(
    state_db: Path, runs_root: Path,
) -> None:
    """When the heartbeat watchdog marked a node lost, the admin view
    surfaces ``status=lost`` so the operator sees that node ≠ healthy
    even though it's still in the table.
    """
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "node-alive", backends=["docker"],
        stream_epoch="ep-1", instance_id="inst-1",
    )
    store.record_node_connected(
        "node-dead", backends=["docker"],
        stream_epoch="ep-1", instance_id="inst-1",
    )
    store.record_node_disconnected("node-dead")
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/nodes")
    assert r.status_code == 200
    body = r.text
    # Both nodes render; status differs.
    assert "node-alive" in body and "node-dead" in body
    # The status column reflects the registry mirror — ``connected``
    # for the live one, ``lost`` for the watchdog-tripped one.
    assert ">connected<" in body
    assert ">lost<" in body


def test_nodes_page_renders_cache_detail_links(
    state_db: Path, runs_root: Path,
) -> None:
    """Each node row must carry a 'cache detail' link to /images/nodes/{id}.

    The nodes.html template gained a ``cache`` column in the IA update.
    This test pins the new column so a template regression is caught.
    """
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "node-q", backends=["docker"], stream_epoch="ep-q", instance_id="inst-q",
    )
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/nodes")
    assert r.status_code == 200
    body = r.text
    assert 'href="/images/nodes/node-q"' in body
    assert "cache detail" in body
    # The case-1-only "active sandboxes" column was removed from /nodes
    # (always 0 for raw-container workloads; /health's per-node "running"
    # count is the accurate, comprehensive source).
    assert "active sandboxes" not in body
    assert 'href="/sandboxes?node=node-q"' not in body


# ──────────────────────────────────────────────────────────────────────────────
# /rollouts
# ──────────────────────────────────────────────────────────────────────────────


def test_rollouts_filters_by_status(state_db: Path, runs_root: Path) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="r-run", status=RolloutStatus.RUNNING)
    _seed_rollout(store, rollout_id="r-done", status=RolloutStatus.FINISHED)
    _seed_rollout(store, rollout_id="r-fail", status=RolloutStatus.FAILED)
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/api/rollouts", params={"status": "failed"})
    payload = r.json()
    assert {rec["rollout_id"] for rec in payload["records"]} == {"r-fail"}


def test_rollouts_api_paginates_newest_first(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    for i in range(70):
        _seed_rollout(store, rollout_id=f"r-{i}", created_offset_s=float(i))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/api/rollouts", params={"page_size": "32", "page": "2"})
    payload = r.json()
    assert len(payload["records"]) == 32
    assert payload["records"][0]["rollout_id"] == "r-32"
    assert payload["records"][-1]["rollout_id"] == "r-63"
    assert payload["page"] == 2
    assert payload["page_size"] == 32
    assert payload["has_next"] is True


def test_rollouts_html_renders_filter_form(
    state_db: Path, runs_root: Path,
) -> None:
    SqliteStateStore(state_db).close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/template")
    assert r.status_code == 200
    # Filter form is rendered with all status enum options.
    for s in ("running", "finished", "failed", "truncated", "cancelled"):
        assert f'value="{s}"' in r.text


def test_rollouts_template_points_empty_state_at_raw(
    state_db: Path, runs_root: Path,
) -> None:
    """The template rollouts page is empty for raw-container workloads; its
    subtitle + empty-state must say so and point to /rollouts/raw rather than
    reading as "nothing is running"."""
    SqliteStateStore(state_db).close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/template")
    assert r.status_code == 200
    body = r.text
    assert "gym/step (template) rollouts" in body
    assert 'href="/rollouts/raw"' in body
    assert "No rollouts match the current filters" in body


def test_rollouts_html_defaults_to_first_32_records(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    for i in range(40):
        _seed_rollout(
            store, rollout_id=f"rollout-{i:02d}", created_offset_s=float(i),
        )
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/template")
    assert r.status_code == 200
    assert "Page 1" in r.text
    assert "showing 32+ records" in r.text
    assert "rollout-00" in r.text
    assert "rollout-31" in r.text
    assert "rollout-32" not in r.text
    assert 'aria-label="next page"' in r.text
    assert "&rarr;" in r.text


def test_rollouts_html_uses_configurable_page_size(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    for i in range(12):
        _seed_rollout(
            store, rollout_id=f"rollout-{i:02d}", created_offset_s=float(i),
        )
    store.close()

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, rollout_page_size=64,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/template")
    assert r.status_code == 200
    assert "Page 1" in r.text
    assert "showing 12 records" in r.text
    assert '<option value="64" selected>64</option>' in r.text
    assert "rollout-00" in r.text
    assert "rollout-11" in r.text
    # With auto-refresh enabled, the configured page size is preserved in the
    # refresh URL so a reload stays on the same page/size. (Default refresh is
    # off, so the meta tag only appears once an interval is selected.)
    refreshed = client.get("/rollouts/template", params={"refresh": "30"})
    assert 'http-equiv="refresh" content="30' in refreshed.text
    assert "page_size=64" in refreshed.text


def test_rollouts_api_last_page_has_next_false(
    state_db: Path, runs_root: Path,
) -> None:
    """The final page must carry has_next=False so clients know there are
    no more records. Regression guard for the N+1 probe logic."""
    store = SqliteStateStore(state_db)
    for i in range(65):
        _seed_rollout(store, rollout_id=f"r-{i}", created_offset_s=float(i))
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    # page_size=32: page 3 has one record, so has_next must be False.
    r = client.get("/api/rollouts", params={"page_size": "32", "page": "3"})
    payload = r.json()
    assert len(payload["records"]) == 1
    assert payload["has_next"] is False


def test_rollouts_api_filters_by_template(
    state_db: Path, runs_root: Path,
) -> None:
    """?template=... restricts the response to rollouts of that template only."""
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="r-alpha", template="alpha")
    _seed_rollout(store, rollout_id="r-beta", template="beta")
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/api/rollouts", params={"template": "alpha"})
    ids = {rec["rollout_id"] for rec in r.json()["records"]}
    assert ids == {"r-alpha"}


def test_rollouts_html_renders_duration_column(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store,
        rollout_id="r-duration",
        status=RolloutStatus.FINISHED,
        created_offset_s=120.0,
        last_touched_offset_s=5.0,
    )
    _seed_rollout(
        store,
        rollout_id="r-live",
        status=RolloutStatus.RUNNING,
        created_offset_s=5.0,
    )
    store.close()
    _write_coordinator_log(
        runs_root,
        "r-duration",
        [
            ("rollout.start", "2026-01-01T00:00:00+00:00"),
            ("rollout.finish", "2026-01-01T00:01:30+00:00"),
        ],
    )

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/template")
    assert r.status_code == 200
    assert "<th>duration</th>" in r.text
    assert "1.5 min" in r.text
    assert "duration source: coordinator.log" in r.text
    assert "00:02:00" in r.text
    assert "live" in r.text


# ──────────────────────────────────────────────────────────────────────────────
# /rollouts/{id}
# ──────────────────────────────────────────────────────────────────────────────


def test_rollout_detail_404_for_unknown(
    state_db: Path, runs_root: Path,
) -> None:
    SqliteStateStore(state_db).close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/never-existed")
    assert r.status_code == 404


def test_rollout_detail_renders_step_list_from_disk(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-detail", status=RolloutStatus.FINISHED,
        sandbox_id="sb-detail", final_reward=0.42,
    )
    store.append_event(
        "rollout.start", rollout_id="r-detail", payload={"k": "v"},
    )
    _seed_sandbox(store, sandbox_id="sb-detail", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-detail", n_steps=3, final_reward=0.42)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-detail")
    assert r.status_code == 200
    body = r.text
    assert "r-detail" in body
    assert "obs-t" in body
    # All 3 steps render in the DOM (the navigator hides non-current
    # ones via CSS but the source HTML still has each <li>).
    assert body.count("step ") >= 3
    # Lifecycle events were dropped from the rollout-detail page in
    # the polish pass — they're noisy on success and the
    # coordinator.log on disk has the full lifecycle trail. Confirm
    # we don't accidentally re-introduce the section.
    assert "Lifecycle events" not in body


def test_rollout_detail_renders_verifier_files_when_present(
    state_db: Path, runs_root: Path,
) -> None:
    """When ``<run_dir>/verifier/`` exists (the platform persisted
    /logs/verifier/ from the sandbox), the rollout-detail page lists
    each file with its size and inlines the content of small text
    files. Distinguishes a real agent failure (test.log shows
    assertion errors) from a verifier misfire (no test.log)."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-verifier", status=RolloutStatus.FINISHED,
        sandbox_id="sb-v", final_reward=1.0,
    )
    _seed_sandbox(store, sandbox_id="sb-v", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-verifier", n_steps=1, final_reward=1.0)

    # Plant a harbor-shape verifier dir under <run_dir>/verifier/.
    run_dir = next((runs_root).rglob("r-verifier"))
    verifier = run_dir / "verifier" / "logs" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.txt").write_text("1\n")
    (verifier / "test.log").write_text("== test session ==\n2 passed in 0.01s\n")

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-verifier")
    assert r.status_code == 200
    body = r.text
    assert "Verifier output" in body
    # File names listed.
    assert "logs/verifier/reward.txt" in body
    assert "logs/verifier/test.log" in body
    # Inline preview rendered for small text files.
    assert "2 passed in 0.01s" in body


def test_rollout_detail_no_verifier_dir_renders_hint(
    state_db: Path, runs_root: Path,
) -> None:
    """When the rollout's run dir exists but has no ``verifier/``
    subdir (e.g. hello-shell, or any benchmark that doesn't follow
    harbor's convention), the page shows a hint explaining why
    rather than omitting the section silently."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-no-verifier", status=RolloutStatus.FINISHED,
        sandbox_id="sb-nv",
    )
    _seed_sandbox(store, sandbox_id="sb-nv", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-no-verifier", n_steps=1, final_reward=0.0)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-no-verifier")
    assert r.status_code == 200
    assert "Verifier output" in r.text
    # The empty-state copy is benchmark-agnostic — must not name
    # harbor or any specific path convention; the redesigned copy
    # talks generally about benchmarks that don't ship verifier
    # output.
    assert "No verifier output captured" in r.text
    assert "harbor" not in r.text.lower()


def test_rollout_detail_default_no_auto_refresh(
    state_db: Path, runs_root: Path,
) -> None:
    """Rollout-detail is an "inspect an artifact" page; auto-refresh
    would lose scroll/search/step state. Default ``?refresh`` is OFF
    — the ``<meta http-equiv=refresh>`` element must NOT be in the
    rendered HTML."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-noref", status=RolloutStatus.FINISHED,
        sandbox_id="sb-noref",
    )
    _seed_sandbox(store, sandbox_id="sb-noref", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-noref", n_steps=2, final_reward=1.0)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-noref")
    assert r.status_code == 200
    assert 'http-equiv="refresh"' not in r.text


def test_rollout_detail_refresh_query_param_is_ignored(
    state_db: Path, runs_root: Path,
) -> None:
    """The polish pass dropped the per-page refresh toggle entirely
    — rollout-detail is always auto-refresh=OFF, regardless of any
    ``?refresh=`` value an operator might paste in. No surprise
    reload cycles."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-noref-ig", status=RolloutStatus.RUNNING,
        sandbox_id="sb-noref-ig",
    )
    _seed_sandbox(store, sandbox_id="sb-noref-ig", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-noref-ig", n_steps=1, final_reward=0.0)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    for choice in ("5", "30", "9999", "foo"):
        r = client.get(f"/rollouts/r-noref-ig?refresh={choice}")
        assert r.status_code == 200, choice
        assert 'http-equiv="refresh"' not in r.text, choice


def test_rollout_detail_step_navigator_present(
    state_db: Path, runs_root: Path,
) -> None:
    """Step-nav controls render: prev/next buttons + jump-to-N input
    + the ``<ol class="steps" data-step-indices=...>`` annotation
    the inline JS reads. Pins the contract the JS depends on."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-nav", status=RolloutStatus.FINISHED,
        sandbox_id="sb-nav",
    )
    _seed_sandbox(store, sandbox_id="sb-nav", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-nav", n_steps=4, final_reward=0.0)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-nav")
    body = r.text
    assert 'id="step-nav"' in body
    assert 'data-action="prev"' in body
    assert 'data-action="next"' in body
    assert 'id="step-jump"' in body
    assert 'data-step-indices="' in body


def test_rollout_detail_no_lifecycle_events_section(
    state_db: Path, runs_root: Path,
) -> None:
    """Lifecycle events were dropped from the rollout-detail page
    in the polish pass — they're noise on a successful inspection
    and the coordinator.log on disk has the full lifecycle trail.
    Make sure no future "let's surface events" change accidentally
    re-introduces the block."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-no-evt", status=RolloutStatus.FINISHED,
        sandbox_id="sb-no-evt",
    )
    store.append_event("rollout.start", rollout_id="r-no-evt", payload={})
    store.append_event("rollout.finished", rollout_id="r-no-evt", payload={})
    _seed_sandbox(store, sandbox_id="sb-no-evt", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-no-evt", n_steps=1, final_reward=1.0)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-no-evt")
    body = r.text
    assert "Lifecycle events" not in body
    assert "rollout.start" not in body  # event payloads also out


def test_rollout_detail_verifier_section_collapsed_by_default(
    state_db: Path, runs_root: Path,
) -> None:
    """The Verifier output block contains potentially large file
    previews; collapsed by default keeps the page compact and lets
    operators expand when debugging. Pin the contract: the
    `<details>` wrapper has no ``open`` attribute."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-vfold", status=RolloutStatus.FINISHED,
        sandbox_id="sb-vfold", final_reward=1.0,
    )
    _seed_sandbox(store, sandbox_id="sb-vfold", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-vfold", n_steps=1, final_reward=1.0)
    run_dir = next((runs_root).rglob("r-vfold"))
    (run_dir / "verifier").mkdir(parents=True)
    (run_dir / "verifier" / "reward.txt").write_text("1\n")

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-vfold")
    body = r.text
    assert "Verifier output" in body
    import re
    # Find the outer <details> wrapping the file count summary.
    pattern = re.compile(
        r"<details(?P<attrs>[^>]*)>\s*<summary>\s*\d+ file",
        re.IGNORECASE,
    )
    match = pattern.search(body)
    assert match is not None, "Outer verifier-files <details> not found"
    assert "open" not in match.group("attrs"), (
        "Verifier output wrapper should be collapsed by default"
    )


def test_rollout_detail_rollout_info_at_bottom_collapsed(
    state_db: Path, runs_root: Path,
) -> None:
    """The platform plumbing (template/status/node/...) was moved
    to a collapsed foldout at the very bottom of the page in the
    polish pass — it's noise for the inspect-a-trajectory use case.
    Pin the layout: the Rollout-info block lives below the steps
    and below the verifier section, in a closed <details>."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-info-fold", status=RolloutStatus.FINISHED,
        sandbox_id="sb-info-fold",
    )
    _seed_sandbox(store, sandbox_id="sb-info-fold", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-info-fold", n_steps=1, final_reward=1.0)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-info-fold")
    body = r.text
    assert "Rollout info" in body
    info_idx = body.index("Rollout info")
    steps_idx = body.index("Steps")
    assert steps_idx < info_idx, (
        f"Steps section ({steps_idx}) must come before Rollout info "
        f"({info_idx}); the platform plumbing belongs at the bottom."
    )
    import re
    pattern = re.compile(
        r"<details(?P<attrs>[^>]*)>\s*<summary[^>]*>Rollout info",
        re.IGNORECASE,
    )
    match = pattern.search(body)
    assert match is not None, "Rollout-info <details> not found"
    assert "open" not in match.group("attrs"), (
        "Rollout info should be collapsed by default"
    )


def test_rollout_detail_task_section_appears_above_steps(
    state_db: Path, runs_root: Path,
) -> None:
    """Metadata is the first thing operators want to see (final reward,
    grader breakdown, task-specific keys); it must render BEFORE the
    Steps section in the rendered HTML."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-mdord", status=RolloutStatus.FINISHED,
        sandbox_id="sb-mdord",
    )
    # Plant some metadata so the Task section actually renders —
    # the polish pass hides the section when ``metadata`` is empty
    # to avoid an empty card on a freshly-started rollout.
    store.update_rollout(
        "r-mdord",
        metadata={"rewards": {"default": {"score": 1.0, "weight": 1.0}}},
    )
    _seed_sandbox(store, sandbox_id="sb-mdord", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-mdord", n_steps=1, final_reward=1.0)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-mdord")
    body = r.text
    # Section heading text — the polish pass renamed "Metadata" to
    # "Task" because it carries task-specific info (tb2's reward
    # breakdown, init_obs excerpt, etc.) for the operator's eye.
    task_idx = body.index("<h2>Task</h2>")
    steps_idx = body.index("<h2>Steps</h2>")
    assert task_idx < steps_idx, (
        "Task section should render above Steps; the polish pass "
        "puts task-related metadata at the top so operators see "
        f"the interesting bits first (task at {task_idx}, steps "
        f"at {steps_idx})."
    )


def test_verifier_file_endpoint_streams_file(
    state_db: Path, runs_root: Path,
) -> None:
    """``GET /rollouts/<id>/verifier/<path>`` returns the raw bytes
    so an operator can download a binary or large text file the
    inline preview truncated."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-stream", status=RolloutStatus.FINISHED,
        sandbox_id="sb-s",
    )
    _seed_sandbox(store, sandbox_id="sb-s", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-stream", n_steps=1, final_reward=0.0)
    run_dir = next((runs_root).rglob("r-stream"))
    (run_dir / "verifier").mkdir(parents=True)
    (run_dir / "verifier" / "reward.txt").write_text("0.875")

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-stream/verifier/reward.txt")
    assert r.status_code == 200
    assert r.text == "0.875"


def test_verifier_file_endpoint_rejects_path_traversal(
    state_db: Path, runs_root: Path,
) -> None:
    """An operator-supplied path that escapes the verifier root
    via ``..`` must be rejected with 400 — otherwise we'd be
    happy to serve arbitrary files under the runs root."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-trav", status=RolloutStatus.FINISHED,
        sandbox_id="sb-t",
    )
    _seed_sandbox(store, sandbox_id="sb-t", node_id="node-A")
    store.close()
    _open_and_seal_run(runs_root, "r-trav", n_steps=1, final_reward=0.0)
    run_dir = next((runs_root).rglob("r-trav"))
    (run_dir / "verifier").mkdir(parents=True)
    (run_dir / "verifier" / "ok.txt").write_text("ok")
    # Plant a sentinel one level up that the traversal would target.
    (run_dir / "trajectory.jsonl").write_text("should not be served")

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-trav/verifier/../trajectory.jsonl")
    # FastAPI / Starlette normalize ``..`` in paths before our handler
    # sees them; the resolved guard inside the handler is a
    # belt-and-suspenders against future config changes.
    assert r.status_code in (400, 404)


def test_rollout_detail_handles_missing_trajectory_on_disk(
    state_db: Path, runs_root: Path,
) -> None:
    """When the StateStore has the rollout but the run dir has been pruned
    (RunDirJanitor) or never written, the page renders a hint instead of 500.
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-no-disk", status=RolloutStatus.FINISHED,
    )
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-no-disk")
    assert r.status_code == 200
    assert "No trajectory body on disk" in r.text


def test_rollout_detail_distinguishes_pruned_from_partial_state(
    state_db: Path, runs_root: Path,
) -> None:
    """User-reported bug shape (2026-05-01): a rollout whose run dir
    is intact (verifier files survived) but whose ``trajectory.jsonl``
    is missing used to render the misleading hint "most likely pruned
    by the run-dir janitor". The pruning hypothesis is provably wrong
    when the verifier dir survives — janitor prunes the whole run dir
    atomically.

    Pin the post-fix behaviour: when ``verifier_files is not None``
    (run dir intact) AND ``step_count > 0`` AND ``trajectory is
    None``, the hint must NOT mention pruning; it must explain that
    in phase-0/1 the body normally lives next to meta.json on the
    control plane (the platform-jsonl sink runs there) and surface
    the real candidate causes (sink=none at start, external delete,
    runs_root mismatch).
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-no-jsonl-but-verifier-here",
        status=RolloutStatus.FINISHED,
        node_id="gcp-osworld-exp-1",
    )
    # ``record.step_count`` in the template is computed as
    # ``len(r.steps)``; append two synthetic steps so the
    # branching condition (step_count > 0) fires.
    for idx in range(2):
        store.append_step("r-no-jsonl-but-verifier-here", Step(
            index=idx, action={"a": idx}, obs={"o": idx},
            reward=0.0, done=(idx == 1), truncated=False,
            info={}, ts=float(idx),
        ))
    store.close()

    # Run dir exists locally with a verifier subdir but NO trajectory.jsonl
    # — the exact shape the user reported.
    run_dir = runs_root / "2026-05-01" / "r-no-jsonl-but-verifier-here"
    (run_dir / "verifier").mkdir(parents=True)
    (run_dir / "verifier" / "test.log").write_text("PASS\n")
    (run_dir / "verifier" / "reward.txt").write_text("1.0\n")
    (run_dir / "verifier" / "ctrf.json").write_text('{"summary":{}}\n')
    (run_dir / "coordinator.log").write_text(
        '{"ts":"2026-05-01T12:00:00+00:00","event":"rollout.start","payload":{}}\n'
    )

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-no-jsonl-but-verifier-here")
    assert r.status_code == 200
    body = r.text

    # The page acknowledges body absence.
    assert "No trajectory body on disk" in body
    # The misleading "pruned by janitor" hint MUST NOT fire — the
    # surviving run dir disproves it.
    assert "pruned by the run-dir janitor" not in body
    # The architecture-correct framing IS surfaced — phase-0/1's
    # sink lives on the control plane next to meta.json. The hint
    # names sink=none / external delete / runs_root mismatch.
    assert "control plane" in body
    assert "sink=none" in body or "sink set to" in body
    # The verifier section (with its 3 files) is rendered alongside,
    # which is what made the bug visible.
    assert "test.log" in body
    assert "reward.txt" in body


def test_rollout_detail_failure_callout_renders_for_aborted_rollout(
    state_db: Path, runs_root: Path,
) -> None:
    """A rollout that ended in ``cancelled`` / ``failed`` / ``truncated``
    with a non-trivial reason gets a top-of-page Failure callout so the
    user sees the cause without spelunking coordinator.log.

    Specifically pins the user-reported bug shape: a rollout that
    aborted at step 0 with an exception summary in ``reason`` (the
    SDK packs ``aborted_with_exception: <type>: <message>`` into the
    cancel reason). The page must surface that string somewhere
    visible, not just inside the bottom platform-info foldout.
    """
    store = SqliteStateStore(state_db)
    reason_text = (
        "aborted_with_exception: TimeoutError: stub /env/step timed out"
    )
    _seed_rollout(
        store, rollout_id="r-aborted",
        status=RolloutStatus.CANCELLED,
        reason=reason_text,
    )
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-aborted")
    assert r.status_code == 200
    body = r.text
    # The Failure section is present.
    assert "<h2>Failure</h2>" in body
    # The full reason (including exception type+message) is rendered
    # in the warn callout, not just inside the bottom info-table.
    failure_section = body.split("<h2>Failure</h2>")[1].split("</section>")[0]
    assert reason_text in failure_section
    # The reason renders inside a ``warn`` element so it's visually
    # distinct from regular muted prose.
    assert 'class="warn"' in failure_section


def test_rollout_detail_no_failure_callout_for_finished_rollout(
    state_db: Path, runs_root: Path,
) -> None:
    """A successfully finished rollout with no reason carries no Failure
    section — the callout is reserved for terminal-and-failed states.
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-ok", status=RolloutStatus.FINISHED,
    )
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-ok")
    assert r.status_code == 200
    assert "<h2>Failure</h2>" not in r.text


def test_rollout_detail_step_zero_hint_distinguishes_abort_from_prune(
    state_db: Path, runs_root: Path,
) -> None:
    """A rollout that aborted before its first step (``step_count == 0``)
    gets the "ended before completing its first step" hint, not the
    misleading "may have been pruned by the run-dir janitor" copy.
    Same shape applies to the verifier-output hint — no point telling
    the user "the verifier didn't run because the benchmark didn't ship
    the convention" when the real reason is "the rollout never reached
    reward time".
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-step0",
        status=RolloutStatus.CANCELLED,
        reason="aborted_with_exception: RuntimeError: boom",
    )
    store.close()
    # Create the run dir so the verifier section renders (with an
    # empty verifier_files list — the platform did create the run dir
    # at sandbox-create time but the rollout aborted before reward).
    run_dir = runs_root / "2026-04-30" / "r-step0"
    run_dir.mkdir(parents=True)
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-step0")
    assert r.status_code == 200
    body = r.text
    # No-trajectory hint mentions step-zero abort, not pruning.
    assert "before completing its first step" in body
    assert "pruned by the run-dir janitor" not in body
    # No-verifier hint mentions never-reached-reward, not benchmark
    # convention.
    assert "ended before completing its first step" in body


def test_rollout_detail_renders_coordinator_log_tail_when_present(
    state_db: Path, runs_root: Path,
) -> None:
    """The control plane writes lifecycle events to
    ``<run_dir>/coordinator.log`` (one JSON-line per event). The
    rollout-detail page inlines a tail so operators can see why a
    rollout failed without SSHing into the host. For
    failed/cancelled/truncated rollouts the section is open by
    default; for finished rollouts it's collapsed.
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-coord",
        status=RolloutStatus.CANCELLED,
        reason="aborted_with_exception: TimeoutError: stub /env/step",
    )
    store.close()
    run_dir = runs_root / "2026-04-30" / "r-coord"
    run_dir.mkdir(parents=True)
    log_lines = [
        '{"ts":"2026-04-30T06:32:31","event":"rollout.start","payload":{"node_id":"local"}}',
        '{"ts":"2026-04-30T06:33:32","event":"rollout.cancel","payload":{"status":"cancelled","reason":"aborted_with_exception"}}',
    ]
    (run_dir / "coordinator.log").write_text("\n".join(log_lines) + "\n")

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-coord")
    assert r.status_code == 200
    body = r.text
    assert "<h2>Coordinator log</h2>" in body
    # Both event lines render.
    assert "rollout.start" in body
    assert "rollout.cancel" in body
    # Section is open by default for failed/cancelled status — the
    # operator is here to diagnose, not to hunt for a click.
    coord_section = body.split("<h2>Coordinator log</h2>")[1].split("</section>")[0]
    assert "<details open>" in coord_section
    # Download link to the full file is present.
    assert "/rollouts/r-coord/coordinator.log" in body


def test_rollout_detail_coordinator_log_collapsed_for_finished(
    state_db: Path, runs_root: Path,
) -> None:
    """Finished rollouts get the Coordinator log section collapsed —
    the lifecycle trail is just noise when nothing failed."""
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-ok-coord", status=RolloutStatus.FINISHED,
    )
    store.close()
    run_dir = runs_root / "2026-04-30" / "r-ok-coord"
    run_dir.mkdir(parents=True)
    (run_dir / "coordinator.log").write_text(
        '{"ts":"2026-04-30T06:32:31","event":"rollout.start","payload":{}}\n'
    )

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-ok-coord")
    assert r.status_code == 200
    body = r.text
    assert "<h2>Coordinator log</h2>" in body
    coord_section = body.split("<h2>Coordinator log</h2>")[1].split("</section>")[0]
    # Closed by default — no ``open`` attribute. Template's Jinja
    # ``<details {% if ... %}open{% endif %}>`` collapses to
    # ``<details >`` when the condition is false, so accept either
    # ``<details>`` or ``<details >``.
    assert "<details open>" not in coord_section
    assert "<details>" in coord_section or "<details >" in coord_section


def test_rollout_detail_coordinator_log_section_omitted_when_file_absent(
    state_db: Path, runs_root: Path,
) -> None:
    """A rollout with no run dir (or a run dir without coordinator.log)
    doesn't render the section at all — an empty stub is worse than
    omitting it.
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-no-log", status=RolloutStatus.FINISHED,
    )
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-no-log")
    assert r.status_code == 200
    assert "<h2>Coordinator log</h2>" not in r.text


def test_rollout_coordinator_log_route_streams_full_file(
    state_db: Path, runs_root: Path,
) -> None:
    """``/rollouts/{id}/coordinator.log`` streams the full file so the
    operator can grab events older than the inline tail's 64 KiB
    window. Plain text so the browser displays it inline by default.
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="r-dl", status=RolloutStatus.CANCELLED)
    store.close()
    run_dir = runs_root / "2026-04-30" / "r-dl"
    run_dir.mkdir(parents=True)
    full_log = "line-{}\n".format("x" * 100) * 50
    (run_dir / "coordinator.log").write_text(full_log)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-dl/coordinator.log")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == full_log


def test_rollout_coordinator_log_route_404_when_missing(
    state_db: Path, runs_root: Path,
) -> None:
    """The route 404s when the rollout has no run dir, and 404s with a
    distinct detail when the run dir exists but has no
    ``coordinator.log`` (helpful for diagnosing a bad sink config).
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="r-nodir", status=RolloutStatus.FINISHED)
    _seed_rollout(store, rollout_id="r-emptydir", status=RolloutStatus.FINISHED)
    store.close()
    (runs_root / "2026-04-30" / "r-emptydir").mkdir(parents=True)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))

    r1 = client.get("/rollouts/r-nodir/coordinator.log")
    assert r1.status_code == 404
    assert "not found" in r1.json()["detail"]

    r2 = client.get("/rollouts/r-emptydir/coordinator.log")
    assert r2.status_code == 404
    assert "no coordinator.log" in r2.json()["detail"]


def test_rollouts_list_terminal_status_never_renders_live(
    state_db: Path, runs_root: Path,
) -> None:
    """A rollout in a terminal state (cancelled/finished/failed/...)
    must NOT render with ``live=True`` even when the per-rollout
    ``coordinator.log`` has a ``rollout.start`` event but no terminal
    event. Pre-fix this happened for any rollout force-sealed via:

      - the startup sweep (``coordinator.sweep_stuck_transients``),
      - direct SQL cleanup (operator clearing a wedged dashboard),
      - a future admin RPC that bypasses the normal terminate path.

    All three write to state.db without emitting a
    ``rollout.cancel`` to the log; the duration snapshot trusted
    the log's "no terminal event yet" reading and rendered "X min
    live" for a row whose state-store status was ``cancelled``.
    Source-of-truth pivot: state-store decides liveness; the log
    only contributes timestamps for accuracy.
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-stale-live",
        status=RolloutStatus.CANCELLED,
        reason="user_cancel/swept_at_startup",
    )
    store.close()

    # Write a coordinator.log with ONLY a start event — mimicking the
    # force-sealed-by-sweep / direct-SQL case. Use the helper that
    # mirrors the resolver's run-dir layout.
    _write_coordinator_log(
        runs_root, "r-stale-live",
        events=[
            ("rollout.start", "2026-04-30T12:00:00+00:00"),
            # NB: NO rollout.cancel — that's the bug condition.
        ],
    )

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/template")
    assert r.status_code == 200
    body = r.text
    # The row for r-stale-live must NOT carry the "live" suffix —
    # state-store says cancelled, that wins.
    assert "r-stale-live" in body
    # Probe the snapshot computation directly to make the assertion
    # robust against future template-class renames. The HTML
    # template renders ``"X min live"`` only when
    # ``duration.live is True``; we go to the source.
    from xrlenv.admin.server import (
        _rollout_duration_snapshot,
    )
    from xrlenv.control.state import SqliteStateStore as _Store
    s2 = _Store(state_db)
    try:
        rec = s2.get_rollout("r-stale-live")
    finally:
        s2.close()
    snapshot = _rollout_duration_snapshot(cfg, rec, time.time())
    assert snapshot["live"] is False, (
        f"terminal rollout rendered as live: status={rec.status}, "
        f"snapshot={snapshot}"
    )


def test_rollout_detail_coordinator_log_tail_truncates_head_at_cap(
    state_db: Path, runs_root: Path,
) -> None:
    """A coordinator.log larger than the inline cap (64 KiB) gets the
    head clipped on a line boundary, with a "download full log"
    affordance so the operator can pull the missing prefix.
    """
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-bigfile", status=RolloutStatus.CANCELLED,
        reason="hard_deadline",
    )
    store.close()
    run_dir = runs_root / "2026-04-30" / "r-bigfile"
    run_dir.mkdir(parents=True)
    # 200 KiB of well-formed JSON-lines, well over the 64 KiB cap.
    line = '{"ts":"x","event":"image_cache.miss","payload":{"image":"a/b:1"}}\n'
    (run_dir / "coordinator.log").write_bytes(line.encode() * 5000)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-bigfile")
    assert r.status_code == 200
    body = r.text
    # Truncation summary on the <summary> mentions both the tail size
    # and the total file size.
    assert "of " in body and " B" in body
    # Download link affordance for the full file.
    assert "download full log" in body
    # The tail starts on a JSON-line boundary — no leading partial
    # ``"a","event"`` style fragment.
    coord_section = body.split("<h2>Coordinator log</h2>")[1].split("</section>")[0]
    pre_text = coord_section.split("<pre")[1].split(">", 1)[1].split("</pre>")[0]
    first_line = pre_text.split("\n", 1)[0]
    # Either empty (full first line was clipped) or starts with ``{``.
    assert first_line == "" or first_line.lstrip().startswith("{")


# ──────────────────────────────────────────────────────────────────────────────
# Static asset
# ──────────────────────────────────────────────────────────────────────────────


def test_static_css_served(client: TestClient) -> None:
    r = client.get("/static/xrlenv.css")
    assert r.status_code == 200
    assert "body" in r.text


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle smoke (uvicorn-backed)
# ──────────────────────────────────────────────────────────────────────────────


def test_admin_server_lifecycle_round_trips(
    state_db: Path, runs_root: Path,
) -> None:
    """``start()`` binds + serves; ``stop()`` releases the port. Uses the
    real uvicorn-backed server (not just TestClient) to exercise the
    threaded lifecycle the runtime hooks rely on.
    """
    import urllib.request

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root,
        host="127.0.0.1", port=0,
    )
    srv = AdminServer(config=cfg)
    try:
        srv.start()
        assert srv.port > 0
        body = urllib.request.urlopen(
            f"http://{srv.host}:{srv.port}/healthz", timeout=5.0,
        ).read().decode()
        assert body == "ok"
    finally:
        srv.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher wiring
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# /sandboxes (Slice 7b)
# ──────────────────────────────────────────────────────────────────────────────


def test_sandboxes_renders_active_table(state_db: Path, runs_root: Path) -> None:
    long_image = (
        "terminal-bench-2/build-pov-ray@"
        "sha256:c6258d2b405ca95c0c9d03928e0b34370b53c557983dd7f835b694c5844bcbdd"
    )
    store = SqliteStateStore(state_db)
    _seed_sandbox(store, sandbox_id="sb-1", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb-2", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb-3", node_id="node-B", image=long_image)
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/sandboxes")
    assert r.status_code == 200
    body = r.text
    assert "sb-1" in body and "sb-2" in body and "sb-3" in body
    assert "<th>image</th>" in body
    assert "im/obs-t:1" in body
    assert f'title="{long_image}"' in body
    assert "terminal-bench-2/build-pov-ray@sh..." in body


def test_sandboxes_points_empty_state_at_raw(
    state_db: Path, runs_root: Path,
) -> None:
    """Sandboxes are a case-1 (template) concept; raw workloads create none, so
    the page's subtitle + empty-state must explain that and point to
    /rollouts/raw instead of reading as a broken/empty cluster."""
    SqliteStateStore(state_db).close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/sandboxes")
    assert r.status_code == 200
    body = r.text
    assert "don't create sandboxes" in body
    assert 'href="/rollouts/raw"' in body
    assert "No active sandboxes match the current filters" in body


def test_sandboxes_filters_by_node(state_db: Path, runs_root: Path) -> None:
    store = SqliteStateStore(state_db)
    _seed_sandbox(store, sandbox_id="sb-A", node_id="node-A")
    _seed_sandbox(store, sandbox_id="sb-B", node_id="node-B")
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/sandboxes", params={"node": "node-A"})
    assert "sb-A" in r.text and "sb-B" not in r.text


# ──────────────────────────────────────────────────────────────────────────────
# /capacity (Slice 7b)
# ──────────────────────────────────────────────────────────────────────────────


def _fake_hw(
    *, vcpus: int = 192, mem_gb: int = 744, disk_gb: int = 500,
) -> HardwareInfo:
    return HardwareInfo(
        vcpus=vcpus, mem_bytes=mem_gb * 1024**3, disk_bytes=disk_gb * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


class _FakeNodeTransport:
    """Minimal stand-in for the live transport the scheduler reads."""

    def __init__(self, hw: HardwareInfo) -> None:
        self._hw = hw

    def hardware(self) -> HardwareInfo:
        return self._hw


def _capacity_client(
    state_db: Path, runs_root: Path, tmp_path: Path,
    *, hardware: dict[str, HardwareInfo] | None = None,
    node_ids: tuple[str, ...] = ("node-A", "node-B"),
) -> TestClient:
    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(yaml.safe_dump({
        "nodes": [{"id": n, "cloud": "aws"} for n in node_ids],
    }))
    transports = {
        nid: _FakeNodeTransport(hw) for nid, hw in (hardware or {}).items()
    }
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, nodes_yaml=nodes_yaml, port=0,
        node_lookup=(transports.get if hardware is not None else None),
    )
    return TestClient(build_admin_app(cfg))


def _record_live_raw(
    state_db: Path, *, node_id: str, rollout_id: str,
    cpu: float = 2.0, mem_gb: int = 8, disk_gb: int = 2,
) -> None:
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)
    try:
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=rollout_id,
            status="running",
            image="im/deepswe:1",
            node_id=node_id,
            created_at=1.0,
            effective_resources_json=ResourceSpec(
                cpu_request=cpu, cpu_limit=cpu,
                mem_request_bytes=mem_gb * 1024**3,
                mem_limit_bytes=mem_gb * 1024**3,
                disk_request_bytes=disk_gb * 1024**3,
            ).model_dump_json(),
        ))
    finally:
        store.close()


def test_capacity_renders_cells_from_live_reported_hardware(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Cells come from what the node actually reported, not a fixed profile."""
    SqliteStateStore(state_db).close()
    _record_live_raw(state_db, node_id="node-A", rollout_id="r-1")
    client = _capacity_client(
        state_db, runs_root, tmp_path,
        hardware={"node-A": _fake_hw(), "node-B": _fake_hw()},
    )

    r = client.get("/capacity")

    assert r.status_code == 200
    body = r.text
    assert "node-A" in body and "node-B" in body
    assert "192 vCPU" in body            # the real reported figure
    assert "2 cpu / 8 GiB / 2 GiB disk" in body   # the live raw footprint


def test_capacity_surfaces_raw_container_workloads(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """A cluster running ONLY raw containers must not render a blank page.

    Raw containers carry no template manifest, so a template-catalog-only
    matrix showed nothing at all for every real benchmark harness.
    """
    SqliteStateStore(state_db).close()
    _record_live_raw(state_db, node_id="node-A", rollout_id="r-1")

    r = _capacity_client(
        state_db, runs_root, tmp_path, hardware={"node-A": _fake_hw()},
    ).get("/capacity")

    assert "Cluster ceiling by workload profile" in r.text
    assert "2 cpu / 8 GiB / 2 GiB disk" in r.text
    assert "Nothing to compute yet" not in r.text


def test_capacity_exposes_a_disk_bound_node(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """The regression this view exists to catch.

    A node reporting a small disk (a ``/``-probing agent on a host whose
    data-root is a separate volume) binds on ``disk:sandbox_writable`` at a
    fraction of its cpu/mem ceiling. The page must SAY so.
    """
    SqliteStateStore(state_db).close()
    _record_live_raw(state_db, node_id="node-A", rollout_id="r-1")

    r = _capacity_client(
        state_db, runs_root, tmp_path,
        hardware={"node-A": _fake_hw(disk_gb=97)},
    ).get("/capacity")

    assert "disk:sandbox_writable" in r.text
    # cpu (84) and mem (78) both dwarf the disk axis (23) — the tell.
    assert "<td>23</td>" in r.text
    assert "<td>84</td>" in r.text


def test_capacity_admits_no_hardware_rather_than_inventing_it(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """An unreachable node is listed WITHOUT cells and flagged.

    The prior page fabricated an 8 vCPU / 32 GiB / 200 GB profile for every
    node, so its numbers described no real cluster.
    """
    SqliteStateStore(state_db).close()
    client = _capacity_client(
        state_db, runs_root, tmp_path, hardware={"node-A": _fake_hw()},
    )

    body = client.get("/capacity").text

    assert "No live hardware for" in body and "node-B" in body
    assert "hardware unknown" in body


def test_capacity_says_so_when_cluster_is_unreachable(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    SqliteStateStore(state_db).close()
    client = _capacity_client(state_db, runs_root, tmp_path, hardware=None)

    body = client.get("/capacity").text

    assert "no cluster reachability wired" in body


def test_capacity_handles_no_nodes(state_db: Path, runs_root: Path) -> None:
    SqliteStateStore(state_db).close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/capacity")
    assert r.status_code == 200
    assert "Nothing to compute yet" in r.text


# ──────────────────────────────────────────────────────────────────────────────
# /images (B7.6 / P1.2.c — image cache snapshot)
# ──────────────────────────────────────────────────────────────────────────────


def test_images_renders_explanatory_hint_when_node_lookup_unwired(
    state_db: Path, runs_root: Path,
) -> None:
    """Standalone admin (no embedded runtime → no live node_lookup)
    can't fetch per-node image reports. Page must render an
    operator-facing explanation instead of crashing or showing
    misleading "no data" with no context.
    """
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images")
    assert r.status_code == 200
    assert "ReportImagesCommand" in r.text
    assert "xrlenv up" in r.text


def test_images_route_aliases_are_first_class(
    state_db: Path, runs_root: Path,
) -> None:
    from xrlenv.node.image_cache import NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "node-a", backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
    )
    store.close()

    class _FakeTransport:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=100 * 1024**3)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _node_id: _FakeTransport(),
    )
    client = TestClient(build_admin_app(cfg))

    cache = client.get("/images/cache")
    assert cache.status_code == 200
    assert "Node storage" in cache.text
    assert "Shows live Docker image-cache disk usage" in cache.text
    assert 'action="/images/cache"' in cache.text
    assert 'href="/images/catalog"' in cache.text

    catalog = client.get("/images/catalog")
    assert catalog.status_code == 200
    assert "Image coverage" in catalog.text
    assert "Shows which images are already present" in catalog.text
    assert 'action="/images/catalog"' in catalog.text
    assert 'href="/images/cache"' in catalog.text


def test_images_query_view_alias_warns_until_phase_boundary(
    state_db: Path, runs_root: Path,
) -> None:
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))

    with pytest.warns(DeprecationWarning, match="/images/cache or /images/catalog"):
        r = client.get("/images", params={"view": "images"})
    assert r.status_code == 200
    assert "Image coverage" in r.text


def test_images_renders_per_node_histogram_via_node_lookup(
    state_db: Path, runs_root: Path,
) -> None:
    """B7.6 acceptance: the /images page fans out via cfg.node_lookup
    and renders summary cards plus the dense per-node table. The full
    per-node image list now lives behind a detail route; the cluster
    page still surfaces tier histogram numbers and free disk.
    """
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    # Seed two nodes in the state store so _list_known_node_ids
    # returns them in a deterministic order.
    store = SqliteStateStore(state_db)
    for nid in ("node-a", "node-b"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
        )
    store.close()

    reports = {
        "node-a": NodeImageReport(
            images=[
                ImageStateRecord(
                    name="bench/task-a:1", tier="in_use",
                    size_bytes=2 * 1024**3, in_use_count=1,
                    last_used_at=100.0, pinned=False,
                    owner="xrlenv_final",
                ),
                ImageStateRecord(
                    name="bench-base/task-a:1", tier="cold",
                    size_bytes=8 * 1024**3, in_use_count=0,
                    last_used_at=50.0, pinned=False,
                    owner="xrlenv_final",
                ),
            ],
            free_disk_bytes=20 * 1024**3,
            pinned=(),
        ),
        "node-b": NodeImageReport(
            images=[
                ImageStateRecord(
                    name="ops/sidecar:1", tier="pinned",
                    size_bytes=1 * 1024**3, in_use_count=0,
                    last_used_at=42.0, pinned=True,
                    owner="xrlenv_final",
                ),
            ],
            free_disk_bytes=15 * 1024**3,
            pinned=("ops/sidecar:1",),
        ),
    }

    class _FakeTransport:
        def __init__(self, node_id: str) -> None:
            self._nid = node_id

        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return reports[self._nid]

    def lookup(node_id: str) -> object | None:
        if node_id in reports:
            return _FakeTransport(node_id)
        return None

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images")
    assert r.status_code == 200
    body = r.text

    # Both nodes show up in the per-node table.
    assert "node-a" in body and "node-b" in body
    # Summary-first IA: cluster totals are cards, not long prose/table
    # pairs, and node IDs link to demand-loaded details.
    assert "cached copies" in body
    assert "lowest free disk" in body
    assert "/images/nodes/node-a" in body
    # Free disk per node renders.
    assert "20.0 GiB" in body  # node-a free disk
    assert "15.0 GiB" in body  # node-b free disk
    # The dead final/stub_runtime/base rows are gone — the histogram
    # only carries ImageTier values now.
    assert "<code>final</code>" not in body
    assert "<code>stub_runtime</code>" not in body
    assert "<code>base</code>" not in body


def test_images_aggregates_distinct_vs_summed(
    state_db: Path, runs_root: Path,
) -> None:
    """When the same image is present on multiple nodes, the two
    cluster-total tables disagree by design — summed counts every
    node·image instance; distinct dedupes by tag. This test pins both.
    """
    from xrlenv.node.image_cache import ImageStateRecord, ImageTier, NodeImageReport

    store = SqliteStateStore(state_db)
    for nid in ("n1", "n2", "n3"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
        )
    store.close()

    # Same shared image present on 3 nodes; one per-node unique image.
    def _shared(tier: ImageTier) -> ImageStateRecord:
        return ImageStateRecord(
            name="shared/img:1", tier=tier,
            size_bytes=4 * 1024**3, in_use_count=0,
            last_used_at=10.0, pinned=False,
            owner="xrlenv_final",
        )

    reports = {
        "n1": NodeImageReport(
            images=[_shared("cold"), ImageStateRecord(
                name="solo/n1:1", tier="cold", size_bytes=1 * 1024**3,
                in_use_count=0, last_used_at=10.0, pinned=False,
                owner="xrlenv_final",
            )],
            free_disk_bytes=10 * 1024**3, pinned=(),
        ),
        # Node n2 has the shared image *in use* — distinct view should
        # promote the shared image to in_use (highest tier wins).
        "n2": NodeImageReport(
            images=[_shared("in_use")],
            free_disk_bytes=10 * 1024**3, pinned=(),
        ),
        "n3": NodeImageReport(
            images=[_shared("cold")],
            free_disk_bytes=10 * 1024**3, pinned=(),
        ),
    }

    class _T:
        def __init__(self, nid: str) -> None:
            self._nid = nid

        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return reports[self._nid]

    def lookup(node_id: str) -> object | None:
        return _T(node_id) if node_id in reports else None

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lookup,
    )

    from xrlenv.admin.server import _aggregate_image_totals, _fetch_image_row_for

    async def _run() -> tuple[Any, Any]:
        rows = [await _fetch_image_row_for(cfg, nid) for nid in ("n1", "n2", "n3")]
        return _aggregate_image_totals(rows)

    summed, distinct = asyncio.run(_run())

    # Summed: 4 image-instances cold (1 shared on n1, 1 solo on n1,
    # 1 shared on n3) + 1 in_use (shared on n2). Wait — recount:
    # n1 has 2 cold, n2 has 1 in_use, n3 has 1 cold. So 3 cold + 1 in_use.
    assert summed["cold"]["count"] == 3
    assert summed["in_use"]["count"] == 1

    # Distinct: 2 unique tags. shared/img:1 has been seen as in_use
    # (highest tier) → it lands in in_use. solo/n1:1 stays cold.
    assert distinct["in_use"]["count"] == 1
    assert distinct["cold"]["count"] == 1


def test_images_renders_unreachable_marker_when_transport_missing(
    state_db: Path, runs_root: Path,
) -> None:
    """If a node is registered but ``node_lookup`` returns None for it
    (transport not in the live registry — e.g. node disconnected
    between page renders), the row is marked unreachable rather than
    silently dropped.
    """
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "ghost", backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
    )
    store.close()

    def lookup(node_id: str) -> object | None:
        return None  # always unreachable

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images")
    assert r.status_code == 200
    assert "ghost" in r.text
    assert "unreachable" in r.text


def test_images_filters_out_lost_and_rostered_nodes(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Operator-reported regression (2026-05-04): the /images panel
    fanned out to every node it could find — including
    historical ``status='lost'`` rows and rostered-but-never-connected
    entries from ``nodes.yaml`` — producing rows of "no live transport
    for this node" noise that obscured the actually-connected nodes'
    real data. The filter now scopes to ``status='connected'`` only.
    """
    from xrlenv.node.image_cache import NodeImageReport

    store = SqliteStateStore(state_db)
    # Connected node — should appear.
    store.record_node_connected(
        "live", backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
    )
    # Lost node — connected once, then dropped. Should NOT appear.
    store.record_node_connected(
        "lost", backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
    )
    store.record_node_disconnected("lost")
    store.close()

    # Rostered-only node (never connected). Should NOT appear.
    nodes_yaml = tmp_path / "nodes.yaml"
    nodes_yaml.write_text(yaml.safe_dump({
        "nodes": [
            {"id": "rostered-only", "cloud": "gcp", "expected_address": "10.0.0.9"},
        ],
    }))

    class _Live:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=0, pinned=())

    def lookup(node_id: str) -> object | None:
        return _Live() if node_id == "live" else None

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        nodes_yaml=nodes_yaml, node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images")
    assert r.status_code == 200
    body = r.text
    assert "live" in body
    # Lost + rostered-only nodes never appear in the per-row table.
    assert "lost</" not in body and ">lost<" not in body
    assert "rostered-only" not in body


def test_images_large_inventory_is_paginated_on_cluster_page(
    state_db: Path, runs_root: Path,
) -> None:
    """The cluster page stays bounded for SWE-bench-shaped inventories:
    100 connected nodes with 100 cached images each. The initial page
    renders node summaries only; distinct image rows are paginated under
    the image-catalog view.
    """
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    node_ids = [f"node-{idx:03d}" for idx in range(100)]
    for nid in node_ids:
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
        )
    store.close()

    reports: dict[str, NodeImageReport] = {}
    for node_idx, nid in enumerate(node_ids):
        images = [
            ImageStateRecord(
                name=f"bench/{nid}/task-{img_idx:03d}:1",
                tier="in_use" if node_idx == 0 and img_idx == 0 else "cold",
                size_bytes=(1 + img_idx % 5) * 1024**2,
                in_use_count=1 if node_idx == 0 and img_idx == 0 else 0,
                last_used_at=float(img_idx),
                pinned=False,
                owner="xrlenv_final",
            )
            for img_idx in range(100)
        ]
        reports[nid] = NodeImageReport(
            images=images,
            free_disk_bytes=(5 if node_idx == 0 else 60) * 1024**3,
            pinned=(),
        )

    class _T:
        def __init__(self, nid: str) -> None:
            self._nid = nid

        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return reports[self._nid]

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _T(nid) if nid in reports else None,
    )
    client = TestClient(build_admin_app(cfg))

    node_page = client.get("/images")
    assert node_page.status_code == 200
    node_body = node_page.text
    assert "Showing 1-50 of 100 nodes" in node_body
    assert "node-000" in node_body
    assert "node-099" not in node_body
    assert "bench/node-099/task-099:1" not in node_body
    assert "critical free disk" in node_body

    image_page = client.get(
        "/images/catalog",
        params={"sort": "name", "page_size": "25"},
    )
    assert image_page.status_code == 200
    image_body = image_page.text
    assert "Showing 1-25 of 10000 images" in image_body
    assert "bench/node-000/task-000:1" in image_body
    assert "bench/node-099/task-099:1" not in image_body


def test_image_node_detail_fetches_single_node_and_paginates_images(
    state_db: Path, runs_root: Path,
) -> None:
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    for nid in ("node-a", "node-b"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
        )
    store.close()

    reports = {
        "node-a": NodeImageReport(
            images=[
                ImageStateRecord(
                    name=f"bench/a-{idx:02d}:1", tier="cold",
                    size_bytes=1024**3, in_use_count=0,
                    last_used_at=float(idx), pinned=False,
                    owner="xrlenv_final",
                )
                for idx in range(30)
            ],
            free_disk_bytes=100 * 1024**3,
            pinned=(),
        ),
        "node-b": NodeImageReport(
            images=[
                ImageStateRecord(
                    name="bench/b-only:1", tier="cold",
                    size_bytes=1024**3, in_use_count=0,
                    last_used_at=0.0, pinned=False,
                    owner="xrlenv_final",
                )
            ],
            free_disk_bytes=100 * 1024**3,
            pinned=(),
        ),
    }
    calls: list[str] = []

    class _T:
        def __init__(self, nid: str) -> None:
            self._nid = nid

        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            calls.append(self._nid)
            return reports[self._nid]

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _T(nid) if nid in reports else None,
    )
    client = TestClient(build_admin_app(cfg))

    r = client.get("/images/nodes/node-a", params={"page_size": "25"})
    assert r.status_code == 200
    assert calls == ["node-a"]
    assert "Showing 1-25 of 30 images" in r.text
    assert "bench/a-00:1" in r.text
    assert "bench/a-29:1" not in r.text
    assert "bench/b-only:1" not in r.text


def test_image_detail_shows_cross_node_coverage(
    state_db: Path, runs_root: Path,
) -> None:
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    for nid in ("node-a", "node-b"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="inst-1",
        )
    store.close()

    reports = {
        "node-a": NodeImageReport(
            images=[
                ImageStateRecord(
                    name="shared/img:1", tier="cold", size_bytes=2 * 1024**3,
                    in_use_count=0, last_used_at=1.0, pinned=False,
                    owner="xrlenv_final",
                )
            ],
            free_disk_bytes=100 * 1024**3,
            pinned=(),
        ),
        "node-b": NodeImageReport(
            images=[
                ImageStateRecord(
                    name="shared/img:1", tier="in_use", size_bytes=2 * 1024**3,
                    in_use_count=2, last_used_at=2.0, pinned=True,
                    owner="xrlenv_final",
                )
            ],
            free_disk_bytes=8 * 1024**3,
            pinned=("shared/img:1",),
        ),
    }

    class _T:
        def __init__(self, nid: str) -> None:
            self._nid = nid

        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return reports[self._nid]

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _T(nid) if nid in reports else None,
    )
    client = TestClient(build_admin_app(cfg))

    r = client.get("/images/image", params={"ref": "shared/img:1"})
    assert r.status_code == 200
    body = r.text
    assert "shared/img:1" in body
    assert "node-a" in body and "node-b" in body
    assert "in_use" in body
    assert "critical free disk" in body


def test_blobs_route_is_gone(
    state_db: Path, runs_root: Path,
) -> None:
    """``/blobs`` was renamed to ``/images`` — the old URL must 404
    so any operator bookmark surfaces immediately instead of silently
    rendering a stale page."""
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/blobs")
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# /health (Slice 7b)
# ──────────────────────────────────────────────────────────────────────────────


def test_health_flags_failure_rate(state_db: Path, runs_root: Path) -> None:
    store = SqliteStateStore(state_db)
    # 4 starts of "noisy", 2 failed → 50% failure rate (above 25% threshold).
    for i in range(4):
        _seed_rollout(
            store,
            rollout_id=f"r-noisy-{i}",
            template="noisy",
            status=RolloutStatus.FAILED if i < 2 else RolloutStatus.FINISHED,
            reason="reward_failed" if i < 2 else None,
        )
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "noisy" in r.text
    assert "50%" in r.text


def test_health_excludes_reaped_from_raw_failure_rate(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """`reaped` raw rollouts must not feed the /health high-failure-rate
    alert. With 3 released + 1 reaped in-window a reap counted as a
    failure is exactly 25% — right on the threshold — so it would trip the
    "raw containers (all)" alert if counted; excluded, it must not. The
    contrast set (3 released + 1 failed) confirms the alert *does* fire on
    a genuine failure, so the exclusion assertion isn't vacuous."""
    from xrlenv.control.state import RawRolloutRecord

    now = time.time()

    def _health_text(db: Path, fourth_status: str, error: str | None) -> str:
        store = SqliteStateStore(db)
        for i in range(3):  # 3 clean releases, recent
            store.record_raw_rollout(RawRolloutRecord(
                rollout_id=f"rr-ok-{i}", status="released", image="busybox:1",
                node_id="n1", created_at=now - 60.0, finished_at=now - 30.0,
            ))
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id="rr-fourth", status=fourth_status, image="busybox:1",
            node_id="n1", created_at=now - 60.0, finished_at=now - 30.0,
            error=error,
        ))
        store.close()
        cfg = AdminServerConfig(state_db=db, runs_root=runs_root, port=0)
        return TestClient(build_admin_app(cfg)).get("/health").text

    # reaped → raw_failed 0/4 = 0% → no alert.
    reaped_html = _health_text(
        state_db, "reaped",
        "session deadline exceeded (overdue 12s) — force-reaped",
    )
    assert "raw containers (all)" not in reaped_html

    # contrast: genuine failure → 1/4 = 25% → alert fires (sensitivity check).
    failed_html = _health_text(
        tmp_path / "state_failed.db", "failed", "exec-timeout cascade",
    )
    assert "raw containers (all)" in failed_html


def test_health_flags_stuck_sandbox(
    state_db: Path, runs_root: Path,
) -> None:
    """A sandbox older than 2x the default hard deadline (1h) shows in the
    long-running / queued triage table (an age heuristic, not a failure)."""
    store = SqliteStateStore(state_db)
    sb = SandboxRecord(
        sandbox_id="sb-stuck", backend="docker",
        backend_ref="cid-stuck", stub_endpoint="tcp://127.0.0.1:0",
        template="obs-t", node_id="node-A",
        created_at=time.time() - 3 * 3600.0,  # 3h old
    )
    store.insert_sandbox(sb)
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "Long-running" in r.text
    assert "sb-stuck" in r.text
    # The old alarming heading is gone.
    assert "Stuck sessions" not in r.text


def test_health_handles_empty_state(state_db: Path, runs_root: Path) -> None:
    SqliteStateStore(state_db).close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "All clear, last evaluated" in r.text
    assert "3 checks run" in r.text
    assert "Totals:" not in r.text


def test_health_node_signal_table_shows_docker_signals(
    state_db: Path, runs_root: Path,
) -> None:
    """Stage 1: the per-node signal table renders the docker-run p95 +
    create-gate depth mirrored from the heartbeat (`node_health`)."""
    import json as _json

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "node-A", backends=["docker"], stream_epoch="e", instance_id="i",
    )
    store.update_node_health("node-A", _json.dumps({
        "window_s": 120, "create_p50_ms": 900.0, "create_p95_ms": 4200.0,
        "create_count": 9, "docker_error_count": 0,
        "docker_timeout_count": 0, "create_inflight": 2, "create_queued": 1,
    }))
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "Node signals" in r.text
    assert "node-A" in r.text
    assert "4200 ms" in r.text  # docker-run p95 rendered from node_health


def test_health_node_signal_table_shows_adaptive_limit(
    state_db: Path, runs_root: Path,
) -> None:
    """Stage 3: the per-node signal table shows the AIMD adaptive
    admission limit mirrored by the control loop."""
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "node-A", backends=["docker"], stream_epoch="e", instance_id="i",
    )
    store.update_node_aimd_limit("node-A", 12)
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "adaptive limit" in r.text  # the Stage-3 column header
    assert "12" in r.text              # the mirrored limit value


def test_health_stuck_includes_raw_rollouts(
    state_db: Path, runs_root: Path,
) -> None:
    """Stage 1: a long-lived case-2/3 raw rollout appears in the long-running
    section with state 'running' — the triage is no longer case-1-only, and a
    long-running session alone does NOT flip the health banner to 'Findings'."""
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-stuck", status="running", image="busybox:1",
        node_id="node-A", created_at=time.time() - 3 * 3600.0,  # 3h old
    ))
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "rr-stuck" in r.text
    assert "Long-running" in r.text
    # Age alone is informational — a healthy cluster still reads "All clear".
    assert "All clear" in r.text


def test_health_labels_queued_acquiring_rollout(
    state_db: Path, runs_root: Path,
) -> None:
    """An `acquiring` raw rollout parked in the admission queue is surfaced as
    'queued — awaiting capacity' (backpressure), not as a stuck/failure, and it
    does not flip the health banner. Regression for the operator confusion where
    long-queued admission requests were mislabelled 'Stuck sessions'."""
    from xrlenv.control.state import RawRolloutRecord

    store = SqliteStateStore(state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="rr-queued", status="acquiring", image="busybox:1",
        node_id=None, created_at=time.time() - 3 * 3600.0,  # 3h queued
    ))
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "rr-queued" in r.text
    assert "queued — awaiting capacity" in r.text
    assert "Stuck sessions" not in r.text
    assert "All clear" in r.text


def test_health_with_flags_shows_findings_banner(
    state_db: Path, runs_root: Path,
) -> None:
    """When at least one health check fires, the page must show the
    'Findings below were last evaluated ...' banner rather than 'All clear'.

    This pins the all_clear=False rendering branch added in the IA update.
    """
    store = SqliteStateStore(state_db)
    for i in range(4):
        _seed_rollout(
            store,
            rollout_id=f"r-noisy-{i}",
            template="noisy",
            status=RolloutStatus.FAILED if i < 2 else RolloutStatus.FINISHED,
            reason="reward_failed" if i < 2 else None,
        )
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    assert "Findings below were last evaluated" in r.text
    assert "3 checks" in r.text
    assert "All clear" not in r.text


# ──────────────────────────────────────────────────────────────────────────────
# Rollout-detail search + download (Slice 7b)
# ──────────────────────────────────────────────────────────────────────────────


def test_rollout_detail_search_marks_matching_steps(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-search", status=RolloutStatus.FINISHED,
    )
    store.close()

    sink = PlatformJsonlSink(runs_root)
    sink.open(rollout_id="r-search", manifest=_manifest(), init={}, node_id="n")
    sink.record_step("r-search", Step(
        index=0, action={"cmd": "echo hello"}, obs={"stdout": "hello"},
        reward=0.0, done=False, truncated=False, info={}, ts=0.0,
    ))
    sink.record_step("r-search", Step(
        index=1, action={"cmd": "ls /tmp"}, obs={"stdout": "files"},
        reward=0.0, done=True, truncated=False, info={}, ts=1.0,
    ))
    sink.seal(
        rollout_id="r-search", status=RolloutStatus.FINISHED,
        reason=None, final_reward=0.5, metadata={},
    )

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-search", params={"find": "hello"})
    assert r.status_code == 200
    # Search picked up step 0 (action contains "hello"); 1 match recorded.
    assert "1 match" in r.text
    assert 'class="match"' in r.text


def test_rollout_detail_download_returns_jsonl(
    state_db: Path, runs_root: Path,
) -> None:
    store = SqliteStateStore(state_db)
    _seed_rollout(store, rollout_id="r-dl", status=RolloutStatus.FINISHED)
    store.close()
    _open_and_seal_run(runs_root, "r-dl", n_steps=2)

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-dl/download")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert "r-dl.jsonl" in r.headers["content-disposition"]
    lines = [line for line in r.text.splitlines() if line.strip()]
    assert len(lines) == 2


def test_rollout_detail_download_404_when_no_run_dir(
    state_db: Path, runs_root: Path,
) -> None:
    SqliteStateStore(state_db).close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-missing/download")
    assert r.status_code == 404


def test_rollout_detail_uses_cache_for_trajectory_fetch(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """The rollout-detail handler funnels trajectory reads through the
    spec-17 cache. Resolution order is local-first then bidi-fallback
    (post-2026-05-01 fix); when no local trajectory.jsonl exists the
    bidi RPC path supplies the body, and subsequent fetches hit the
    cache.
    """
    from xrlenv.control.trajectory_cache import TrajectoryCacheConfig

    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-via-bidi", status=RolloutStatus.FINISHED,
        node_id="node-X",
    )
    store.close()

    fetched: list[str] = []

    class _FakeTransport:
        node_id = "node-X"

        async def fetch_trajectory(
            self, rollout_id: str, *,
            range_kind: str = "whole",
            step_start: int = 0,
            step_end: int | None = None,
        ) -> Any:
            from xrlenv.types import RolloutStatus as _RS
            from xrlenv.types import Step as _Step
            from xrlenv.types import Trajectory as _Traj
            fetched.append(rollout_id)
            return _Traj(
                rollout_id=rollout_id, template="obs-t",
                steps=[_Step(
                    index=0, action={"a": 0}, obs={"o": 0}, reward=0.0,
                    done=True, truncated=False, info={}, ts=0.0,
                )],
                status=_RS.FINISHED, final_reward=0.7, metadata={},
            )

    # Isolate the cache to ``tmp_path`` so the test doesn't read or
    # pollute the operator's real ~/.xrlenv/admin-cache.
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _FakeTransport() if nid == "node-X" else None,
        trajectory_cache_config=TrajectoryCacheConfig(
            cache_root=tmp_path / "admin-cache",
        ),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-via-bidi")
    assert r.status_code == 200
    assert "r-via-bidi" in r.text
    # Bidi path was taken because there's no local trajectory.jsonl
    # under runs_root — local read raised FileNotFoundError, fallback
    # to bidi.
    assert fetched == ["r-via-bidi"]
    # Second click is served from the cache — no second bidi fetch.
    r2 = client.get("/rollouts/r-via-bidi")
    assert r2.status_code == 200
    assert fetched == ["r-via-bidi"]


def test_rollout_detail_local_trajectory_wins_over_bidi(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Post-2026-05-01 fix: when the trajectory.jsonl IS present on
    the control plane's local disk (the architecture-correct location
    for phase-0 / phase-1 — the platform-jsonl sink runs on the
    control plane), the admin reads it locally even when
    ``cfg.node_lookup`` is wired. A bidi RPC error must NOT throw
    away a perfectly good local file.

    Bug shape (user-reported 2026-05-01): the admin showed "No
    trajectory body on disk" for a rollout whose ``meta.json`` +
    ``coordinator.log`` + ``<run_dir>/verifier/`` were all rendered
    on the same page; ``trajectory.jsonl`` was on disk and readable
    via the sink, but the admin had bidi-fetched first, the bidi
    call had erred, and the local-disk fallback never ran.
    """
    from xrlenv.control.trajectory_cache import TrajectoryCacheConfig
    from xrlenv.types import Step as _Step

    rid = "r-local-wins"
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id=rid, status=RolloutStatus.FINISHED,
        node_id="node-X",
    )
    # Append two synthetic steps via the StateStore so the page's
    # step_count check is satisfied.
    for idx in range(2):
        store.append_step(rid, _Step(
            index=idx, action={"a": idx}, obs={"o": idx},
            reward=0.0, done=(idx == 1), truncated=False,
            info={}, ts=float(idx),
        ))
    store.close()

    # Write a real trajectory.jsonl via the sink — same way the
    # production code does on every step.
    _open_and_seal_run(runs_root, rid, n_steps=2, final_reward=0.5)

    bidi_calls: list[str] = []

    class _ExplodingTransport:
        node_id = "node-X"

        async def fetch_trajectory(self, rollout_id: str, **_: Any) -> Any:
            bidi_calls.append(rollout_id)
            raise RuntimeError("bidi shouldn't be called when local file is present")

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _ExplodingTransport(),
        trajectory_cache_config=TrajectoryCacheConfig(
            cache_root=tmp_path / "admin-cache",
        ),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get(f"/rollouts/{rid}")
    assert r.status_code == 200

    # Bidi was NEVER called — local file was found first.
    assert bidi_calls == [], (
        f"bidi fetch should not be reached when local trajectory.jsonl "
        f"exists; got {bidi_calls}"
    )
    # The page renders steps (proves the body was actually read,
    # not just "the page didn't 500"). The seed wrote 2 steps with
    # ``info_key`` field; assert at least the step heading text
    # surfaces.
    assert "step 0" in r.text or "Step 0" in r.text or "data-index=\"0\"" in r.text


def test_rollout_detail_bidi_failure_falls_back_to_local(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Companion to the local-wins test. When local read fails AND
    bidi raises, the cache surfaces the original FileNotFoundError
    so the admin's missing-trajectory branch fires (instead of the
    generic "fetch failed" log). Pins the error-translation
    contract.
    """
    from xrlenv.control.trajectory_cache import TrajectoryCacheConfig

    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id="r-both-fail", status=RolloutStatus.FINISHED,
        node_id="node-X",
    )
    store.close()
    # No trajectory.jsonl written on the control plane (local read
    # will FileNotFoundError) AND the bidi transport raises.

    class _ExplodingTransport:
        node_id = "node-X"

        async def fetch_trajectory(self, _rollout_id: str, **_: Any) -> Any:
            raise RuntimeError("node unreachable")

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _ExplodingTransport(),
        trajectory_cache_config=TrajectoryCacheConfig(
            cache_root=tmp_path / "admin-cache",
        ),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/rollouts/r-both-fail")
    assert r.status_code == 200
    # Page renders the missing-trajectory hint (the run-dir IS
    # missing from local + bidi failed — that's the pruned/missing
    # state from the admin's perspective).
    assert "No trajectory body on disk" in r.text


def test_rollout_detail_bypasses_cache_for_non_terminal_rollouts(
    state_db: Path, runs_root: Path, tmp_path: Path,
) -> None:
    """Audit M1 (2026-05-01): the spec-17 trajectory cache assumes
    bodies are immutable once written. That holds for sealed
    rollouts (``finished`` / ``failed`` / ``truncated`` /
    ``cancelled``) but NOT for in-flight ones — ``record_step``
    keeps appending to ``trajectory.jsonl`` until seal. Caching a
    partial body would lock in a stale step count for the cache
    TTL even after the live file grows.

    Pin the bypass: when the rollout's status is non-terminal
    (e.g. ``running``), the admin reads ``trajectory.jsonl``
    directly via ``fetch_fn`` and never writes the cache. A second
    detail-page render after the file grows must see the new
    steps, not the cached older snapshot.
    """
    from xrlenv.control.trajectory_cache import TrajectoryCacheConfig
    from xrlenv.types import Step as _Step

    rid = "r-running"
    store = SqliteStateStore(state_db)
    _seed_rollout(
        store, rollout_id=rid, status=RolloutStatus.RUNNING, node_id="local",
    )
    # Append two steps; sink writes them into trajectory.jsonl.
    for idx in range(2):
        store.append_step(rid, _Step(
            index=idx, action={"a": idx}, obs={"o": idx},
            reward=0.0, done=False, truncated=False,
            info={}, ts=float(idx),
        ))
    store.close()

    # Write the rollout's run dir with the first 2 steps via the
    # platform sink, mirroring the live phase-0 path.
    sink = PlatformJsonlSink(runs_root)
    sink.open(rollout_id=rid, manifest=_manifest(), init={}, node_id="local")
    for idx in range(2):
        sink.record_step(rid, _Step(
            index=idx, action={"a": idx}, obs={"o": idx},
            reward=0.0, done=False, truncated=False,
            info={}, ts=float(idx),
        ))

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        trajectory_cache_config=TrajectoryCacheConfig(
            cache_root=tmp_path / "admin-cache",
        ),
    )
    client = TestClient(build_admin_app(cfg))

    # First render — should pick up step indices 0 and 1.
    r1 = client.get(f"/rollouts/{rid}")
    assert r1.status_code == 200
    assert 'data-index="0"' in r1.text
    assert 'data-index="1"' in r1.text
    assert 'data-index="2"' not in r1.text

    # Append a third step (rollout is still running).
    store2 = SqliteStateStore(state_db)
    store2.append_step(rid, _Step(
        index=2, action={"a": 2}, obs={"o": 2},
        reward=0.0, done=False, truncated=False, info={}, ts=2.0,
    ))
    store2.close()
    sink.record_step(rid, _Step(
        index=2, action={"a": 2}, obs={"o": 2},
        reward=0.0, done=False, truncated=False, info={}, ts=2.0,
    ))

    # Second render — should pick up the new step (cache bypass).
    r2 = client.get(f"/rollouts/{rid}")
    assert r2.status_code == 200
    assert 'data-index="2"' in r2.text, (
        "non-terminal rollout's body must be re-read on each render; "
        "the spec-17 cache should not have served a stale snapshot"
    )

    # The cache directory must NOT contain a serialized body for
    # this running rollout.
    cache_dir = tmp_path / "admin-cache"
    if cache_dir.exists():
        cached_files = list(cache_dir.rglob(f"*{rid}*"))
        assert cached_files == [], (
            f"non-terminal rollout was cached to {cached_files}; "
            f"the bypass should have skipped cache.get / write"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Existing dispatcher test
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# /images — input validation (422)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/images", {"view": "blobs"}),
        ("/images", {"page_size": "999"}),
        ("/images", {"tier": "hot"}),
        ("/images", {"pressure": "high"}),
        ("/images", {"pinned": "maybe"}),
        ("/images/cache", {"sort": "badkey"}),
        ("/images/catalog", {"sort": "badkey"}),
    ],
    ids=[
        "view-invalid",
        "page_size-out-of-range",
        "tier-invalid",
        "pressure-invalid",
        "pinned-invalid",
        "sort-key-invalid-nodes-view",
        "sort-key-invalid-catalog-view",
    ],
)
def test_images_invalid_query_params_return_422(
    state_db: Path,
    runs_root: Path,
    path: str,
    params: dict[str, str],
) -> None:
    """Invalid query params on the /images family return 422 (not a silent
    no-op). One node is wired with an empty image report so the route gets
    far enough to reach query-param validation."""
    from xrlenv.node.image_cache import NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=0, pinned=())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get(path, params=params)
    assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# /images/nodes/{node_id} — detail route edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_image_node_detail_unwired_renders_hint(
    state_db: Path, runs_root: Path,
) -> None:
    """When node_lookup is not wired, the detail page renders an explanatory
    hint (not a 500 or empty page)."""
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/nodes/some-node")
    assert r.status_code == 200
    assert "xrlenv up" in r.text


def test_image_node_detail_returns_404_for_unknown_node(
    state_db: Path, runs_root: Path,
) -> None:
    """When node_lookup is wired but the node ID is not in the connected list,
    the detail route returns 404 rather than an unreachable row."""
    from xrlenv.node.image_cache import NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "real-node", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=0, pinned=())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda nid: _T() if nid == "real-node" else None,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/nodes/ghost-node")
    assert r.status_code == 404
    assert "ghost-node" in r.json()["detail"]


def test_image_node_detail_unreachable_renders_error_row(
    state_db: Path, runs_root: Path,
) -> None:
    """When the node is connected but report_images raises, the detail page
    renders the unreachable marker rather than 500-ing."""
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "flaky", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _Flaky:
        async def report_images(self) -> Any:
            raise RuntimeError("rpc unavailable")

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _: _Flaky(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/nodes/flaky")
    assert r.status_code == 200
    assert "unreachable" in r.text


def test_image_node_detail_invalid_sort_returns_422(
    state_db: Path, runs_root: Path,
) -> None:
    """Invalid ?sort= key on the node-detail route is 422."""
    from xrlenv.node.image_cache import NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=0, pinned=())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/nodes/n", params={"sort": "badkey"})
    assert r.status_code == 422


def test_image_node_detail_paginates_to_page2(
    state_db: Path, runs_root: Path,
) -> None:
    """Page 2 of a node-detail view exposes the next batch of images and
    has_prev=True, providing correct pagination navigation."""
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(
                images=[
                    ImageStateRecord(
                        name=f"bench/task-{i:03d}:1", tier="cold",
                        size_bytes=1024**3, in_use_count=0,
                        last_used_at=float(i), pinned=False,
                        owner="xrlenv_final",
                    )
                    for i in range(60)
                ],
                free_disk_bytes=100 * 1024**3, pinned=(),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))

    r = client.get("/images/nodes/n", params={"page_size": "25", "page": "2"})
    assert r.status_code == 200
    body = r.text
    assert "Showing 26-50 of 60 images" in body
    # Page 1 items absent; page 2 items present (sorted by operational default).
    assert 'aria-label="previous page"' in body


# ──────────────────────────────────────────────────────────────────────────────
# /images/nodes/<id> — ownership filter (B7.6 admin-filter follow-on)
# ──────────────────────────────────────────────────────────────────────────────


def test_image_node_detail_default_filter_drops_intermediate_and_external(
    state_db: Path, runs_root: Path,
) -> None:
    """Explicit ``?include=default`` hides ``xrlenv_intermediate`` +
    ``external`` rows so operators who pick the "xrlenv only" slice
    from the dropdown see just the sandbox-runnable final images.

    (No-flag default is ``all`` post-2026-05-11; first-time operators
    see every cached image without having to fiddle with the
    dropdown. The "xrlenv only" filter is still available — this
    test exercises it via the explicit query param.)"""
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(
                images=[
                    ImageStateRecord(
                        name="bench/task-final:1", tier="cold",
                        size_bytes=1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="xrlenv_final",
                    ),
                    ImageStateRecord(
                        name="bench-base/task-final:1", tier="cold",
                        size_bytes=8 * 1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="xrlenv_intermediate",
                    ),
                    ImageStateRecord(
                        name="python:3.12-slim", tier="cold",
                        size_bytes=200 * 1024**2, in_use_count=0,
                        last_used_at=0.0, pinned=False,
                        owner="external",
                    ),
                ],
                free_disk_bytes=100 * 1024**3, pinned=(),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))

    # Explicit ``?include=default``: only the final image visible.
    # The "Showing N of M" hint announces the suppressed rows so
    # operators know the filter is active.
    r = client.get("/images/nodes/n?include=default")
    assert r.status_code == 200
    body = r.text
    assert "bench/task-final:1" in body
    assert "bench-base/task-final:1" not in body
    assert "python:3.12-slim" not in body
    assert "Showing 1 of 3 images" in body


def test_image_node_detail_include_dropdown_reveals_suppressed_rows(
    state_db: Path, runs_root: Path,
) -> None:
    """The include dropdown maps the two suppressed ownership classes onto
    the four visibility states without rendering a pair of loose checkboxes."""
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(
                images=[
                    ImageStateRecord(
                        name="bench/task-final:1", tier="cold",
                        size_bytes=1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="xrlenv_final",
                    ),
                    ImageStateRecord(
                        name="bench-base/task-final:1", tier="cold",
                        size_bytes=8 * 1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="xrlenv_intermediate",
                    ),
                    ImageStateRecord(
                        name="python:3.12-slim", tier="cold",
                        size_bytes=200 * 1024**2, in_use_count=0,
                        last_used_at=0.0, pinned=False,
                        owner="external",
                    ),
                ],
                free_disk_bytes=100 * 1024**3, pinned=(),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))

    r = client.get("/images/nodes/n", params={"include": "intermediates"})
    assert r.status_code == 200
    body = r.text
    assert "bench/task-final:1" in body
    assert "bench-base/task-final:1" in body
    assert "python:3.12-slim" not in body
    assert "Showing 2 of 3 images" in body

    r = client.get("/images/nodes/n", params={"include": "foreign"})
    assert r.status_code == 200
    body = r.text
    assert "bench/task-final:1" in body
    assert "bench-base/task-final:1" not in body
    assert "python:3.12-slim" in body
    assert "Showing 2 of 3 images" in body

    # Both filters off → all three rows visible. The hint disappears
    # since neither default-on filter is active.
    r = client.get("/images/nodes/n", params={"include": "all"})
    body = r.text
    assert "bench/task-final:1" in body
    assert "bench-base/task-final:1" in body
    assert "python:3.12-slim" in body
    assert "Showing 2 of 3 images" not in body
    assert "Showing 3 of 3 images" not in body


# ──────────────────────────────────────────────────────────────────────────────
# /images/image?ref=... — detail route edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_image_detail_unwired_renders_hint(
    state_db: Path, runs_root: Path,
) -> None:
    """When node_lookup is not wired, the image detail page renders a hint."""
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/image", params={"ref": "bench/task:1"})
    assert r.status_code == 200
    assert "xrlenv up" in r.text


def test_image_detail_returns_404_for_unknown_image(
    state_db: Path, runs_root: Path,
) -> None:
    """When the image ref is not found on any connected node, 404."""
    from xrlenv.node.image_cache import NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=0, pinned=())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/image", params={"ref": "no-such/image:99"})
    assert r.status_code == 404
    assert "no-such/image:99" in r.json()["detail"]


# ──────────────────────────────────────────────────────────────────────────────
# /images — filter tests (q, tier, pressure, pinned)
# ──────────────────────────────────────────────────────────────────────────────


def _two_node_lookup(
    state_db: Path,
) -> tuple[dict[str, Any], Any]:
    """Seed two nodes with distinct image types and return (reports, lookup)."""
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    reports = {
        "alpha": NodeImageReport(
            images=[
                ImageStateRecord(
                    name="bench/alpha-task:1", tier="in_use",
                    size_bytes=2 * 1024**3, in_use_count=1,
                    last_used_at=10.0, pinned=True,
                    owner="xrlenv_final",
                ),
            ],
            free_disk_bytes=60 * 1024**3, pinned=("bench/alpha-task:1",),
        ),
        "beta": NodeImageReport(
            images=[
                ImageStateRecord(
                    name="bench/beta-task:1", tier="cold",
                    size_bytes=1 * 1024**3, in_use_count=0,
                    last_used_at=5.0, pinned=False,
                    owner="xrlenv_final",
                ),
            ],
            free_disk_bytes=8 * 1024**3,  # watch-level (< 50 GiB)
            pinned=(),
        ),
    }

    class _T:
        def __init__(self, nid: str) -> None:
            self._nid = nid

        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return reports[self._nid]

    def lookup(node_id: str) -> object | None:
        return _T(node_id) if node_id in reports else None

    return reports, lookup


def test_images_q_filter_narrows_nodes_view(
    state_db: Path, runs_root: Path,
) -> None:
    """?q= substring filter on the nodes view hides non-matching node IDs."""
    store = SqliteStateStore(state_db)
    for nid in ("alpha", "beta"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="i1",
        )
    store.close()
    _, lookup = _two_node_lookup(state_db)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images", params={"q": "alpha"})
    assert r.status_code == 200
    assert "alpha" in r.text
    # The q filter hides beta from the table rows (which render the node_id
    # inside <code> tags); "beta" may still appear in summary cards (e.g.
    # lowest-free-disk) — those are computed pre-filter and are correct.
    assert "<code>beta</code>" not in r.text


def test_images_pressure_filter_narrows_nodes_view(
    state_db: Path, runs_root: Path,
) -> None:
    """?pressure=watch shows only watch-pressure nodes; critical/ok nodes hidden."""
    store = SqliteStateStore(state_db)
    for nid in ("alpha", "beta"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="i1",
        )
    store.close()
    _, lookup = _two_node_lookup(state_db)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    # beta has 8 GiB free → "watch"; alpha has 60 GiB → "ok"
    r = client.get("/images", params={"pressure": "watch"})
    assert r.status_code == 200
    assert "beta" in r.text
    # alpha should be absent from the filtered table (not from the summary cards)
    assert "No nodes match the current filters" not in r.text or "alpha" not in r.text.split("No nodes")[0]


def test_images_pinned_filter_yes_narrows_nodes_view(
    state_db: Path, runs_root: Path,
) -> None:
    """?pinned=yes shows only nodes with at least one operator-pinned image."""
    store = SqliteStateStore(state_db)
    for nid in ("alpha", "beta"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="i1",
        )
    store.close()
    _, lookup = _two_node_lookup(state_db)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images", params={"pinned": "yes"})
    assert r.status_code == 200
    # alpha has a pinned image; beta does not.
    assert "alpha" in r.text


def test_images_q_filter_narrows_images_catalog_view(
    state_db: Path, runs_root: Path,
) -> None:
    """?q= on the images catalog view hides non-matching image names."""
    store = SqliteStateStore(state_db)
    for nid in ("alpha", "beta"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="i1",
        )
    store.close()
    _, lookup = _two_node_lookup(state_db)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/catalog", params={"q": "alpha"})
    assert r.status_code == 200
    assert "bench/alpha-task:1" in r.text
    assert "bench/beta-task:1" not in r.text


def test_images_catalog_default_include_filter_drops_intermediate_and_external(
    state_db: Path, runs_root: Path,
) -> None:
    """``/images/catalog`` mirrors the per-node detail page's
    xrlenv-only / +intermediates / +foreign include filter. The
    explicit ``?include=default`` slice hides ``xrlenv_intermediate``
    + ``external`` rows so operators picking "xrlenv only" from the
    dropdown see just the sandbox-runnable final images — the
    ``Showing N of M`` hint announces the suppressed rows.

    (No-flag default is ``all`` post-2026-05-11; this test exercises
    the explicit-filter path.)"""
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(
                images=[
                    ImageStateRecord(
                        name="bench/task-final:1", tier="cold",
                        size_bytes=1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="xrlenv_final",
                    ),
                    ImageStateRecord(
                        name="bench-base/task-final:1", tier="cold",
                        size_bytes=8 * 1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="xrlenv_intermediate",
                    ),
                    ImageStateRecord(
                        name="python:3.12-slim", tier="cold",
                        size_bytes=200 * 1024**2, in_use_count=0,
                        last_used_at=0.0, pinned=False,
                        owner="external",
                    ),
                ],
                free_disk_bytes=100 * 1024**3, pinned=(),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))

    # Explicit ``?include=default``: only the xrlenv_final image is
    # visible cluster-wide.
    r = client.get("/images/catalog", params={"include": "default"})
    assert r.status_code == 200
    body = r.text
    assert "bench/task-final:1" in body
    assert "bench-base/task-final:1" not in body
    assert "python:3.12-slim" not in body
    assert "Showing 1 of 3 images cluster-wide" in body

    # +intermediates reveals the build byproduct but still hides foreign.
    r = client.get("/images/catalog", params={"include": "intermediates"})
    body = r.text
    assert "bench/task-final:1" in body
    assert "bench-base/task-final:1" in body
    assert "python:3.12-slim" not in body

    # all reveals every owner.
    r = client.get("/images/catalog", params={"include": "all"})
    body = r.text
    assert "bench/task-final:1" in body
    assert "bench-base/task-final:1" in body
    assert "python:3.12-slim" in body
    # No filter active -> the muted hint is suppressed.
    assert "Showing 3 of 3 images cluster-wide" not in body
    # Ownership breakdown chip is always present so operators can
    # diagnose label-missing scenarios.
    assert "Cluster ownership breakdown" in body
    assert "1</strong> xrlenv final" in body
    assert "1</strong> intermediate" in body
    assert "1</strong> external" in body


def test_images_catalog_warns_when_no_xrlenv_labels_present(
    state_db: Path, runs_root: Path,
) -> None:
    """Regression for the 2026-05-05 user report: include filter
    "always shows all images". Root cause was operator-side — every
    image had no ``org.xrlenv.owned`` label (built before the labeling
    commit landed), so they all classified as ``external``. The
    catalog page now renders an actionable hint when the breakdown
    shows zero xrlenv-labeled images so the operator gets a clear
    "rebuild to label" message instead of silently empty results."""
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "n", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            # Two images, BOTH external — the operator-rebuild-needed
            # state.
            return NodeImageReport(
                images=[
                    ImageStateRecord(
                        name="bench/task-pre-label:1", tier="cold",
                        size_bytes=1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="external",
                    ),
                    ImageStateRecord(
                        name="bench/task-pre-label:2", tier="cold",
                        size_bytes=1024**3, in_use_count=0,
                        last_used_at=10.0, pinned=False,
                        owner="external",
                    ),
                ],
                free_disk_bytes=100 * 1024**3, pinned=(),
            )

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))

    body = client.get("/images/catalog").text
    assert "No xrlenv-owned images detected." in body
    assert "Rebuild" in body or "rebuild" in body
    # The breakdown chip surfaces the all-external state explicitly.
    assert "0</strong> xrlenv final" in body
    assert "2</strong> external" in body


def test_images_tier_filter_narrows_images_catalog_view(
    state_db: Path, runs_root: Path,
) -> None:
    """?tier=in_use on the images catalog view hides cold images."""
    store = SqliteStateStore(state_db)
    for nid in ("alpha", "beta"):
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="i1",
        )
    store.close()
    _, lookup = _two_node_lookup(state_db)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0, node_lookup=lookup,
    )
    client = TestClient(build_admin_app(cfg))
    r = client.get("/images/catalog", params={"tier": "in_use"})
    assert r.status_code == 200
    # Only alpha-task is in_use; beta-task is cold.
    assert "bench/alpha-task:1" in r.text
    assert "bench/beta-task:1" not in r.text


# ──────────────────────────────────────────────────────────────────────────────
# /images — fan-out timeout produces unreachable marker
# ──────────────────────────────────────────────────────────────────────────────


def test_images_timeout_produces_unreachable_marker(
    state_db: Path, runs_root: Path,
) -> None:
    """A node whose report_images call exceeds the 5-second deadline is
    marked unreachable in the cluster view rather than blocking the render
    or raising an exception. This test uses a coroutine that sleeps longer
    than the timeout budget to trigger the TimeoutError path."""
    store = SqliteStateStore(state_db)
    store.record_node_connected(
        "slow-node", backends=["docker"], stream_epoch="ep-1", instance_id="i1",
    )
    store.close()

    class _SlowTransport:
        async def report_images(self) -> Any:
            # Sleep longer than the 5-second fan-out budget.
            await asyncio.sleep(60)
            raise AssertionError("should not reach here")

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _: _SlowTransport(),
    )
    # Override the wait_for timeout to 0.05 s so the test completes quickly.
    from xrlenv.admin import server as admin_server

    async def _fast_timeout(cfg_: Any, node_ids: list[str]) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(32)

        async def _one(node_id: str) -> dict[str, Any]:
            async with sem:
                try:
                    return await asyncio.wait_for(
                        admin_server._fetch_image_row_for(cfg_, node_id),
                        timeout=0.05,
                    )
                except TimeoutError:
                    return admin_server._unreachable_image_row(
                        node_id, "report_images timed out",
                    )

        return await asyncio.gather(*(_one(nid) for nid in node_ids))

    import unittest.mock as _mock

    with _mock.patch.object(admin_server, "_fetch_image_rows_for_nodes", _fast_timeout):
        client = TestClient(build_admin_app(cfg))
        r = client.get("/images")

    assert r.status_code == 200
    assert "slow-node" in r.text
    assert "unreachable" in r.text


# ──────────────────────────────────────────────────────────────────────────────
# /images — risk detection (_image_risks unit tests)
# ──────────────────────────────────────────────────────────────────────────────


def test_image_risks_flags_unreachable_nodes() -> None:
    """When one or more rows are unreachable, risks gets a 'bad' entry."""
    from xrlenv.admin.server import _image_risks, _image_summary

    reachable_row = {
        "node_id": "n1", "reachable": True, "error": None,
        "free_disk_bytes": 100 * 1024**3,
        "histogram": {t: {"count": 0, "bytes": 0} for t in ("in_use", "pinned", "recently_used", "cold")},
        "pinned": [], "images": [], "total_count": 0, "total_bytes": 0,
        "cold_bytes": 0, "pinned_tier_bytes": 0,
        "pressure": "ok", "pressure_label": "ok", "pressure_rank": 1,
        "pinned_count": 0, "detail_url": "/images/nodes/n1",
    }
    unreachable_row = {
        **reachable_row,
        "node_id": "n2", "reachable": False, "error": "timeout",
        "pressure": "unreachable", "pressure_label": "unreachable", "pressure_rank": 4,
        "detail_url": "/images/nodes/n2",
    }
    summary = _image_summary(
        [reachable_row, unreachable_row],
        [reachable_row, unreachable_row],
        [],
    )
    risks = _image_risks(summary, [reachable_row, unreachable_row])
    assert any(r["level"] == "bad" and "did not report images" in r["text"] for r in risks)


def test_image_risks_flags_low_free_disk() -> None:
    """A reachable node below the critical threshold raises a 'bad' risk."""
    from xrlenv.admin.server import _image_risks, _image_summary

    critical_row = {
        "node_id": "n1", "reachable": True, "error": None,
        "free_disk_bytes": 5 * 1024**3,  # < 10 GiB critical threshold
        "histogram": {t: {"count": 0, "bytes": 0} for t in ("in_use", "pinned", "recently_used", "cold")},
        "pinned": [], "images": [], "total_count": 0, "total_bytes": 0,
        "cold_bytes": 0, "pinned_tier_bytes": 0,
        "pressure": "critical", "pressure_label": "critical free disk", "pressure_rank": 3,
        "pinned_count": 0, "detail_url": "/images/nodes/n1",
    }
    summary = _image_summary([critical_row], [critical_row], [])
    risks = _image_risks(summary, [critical_row])
    assert any(r["level"] == "bad" and "low on free disk" in r["text"] for r in risks)


def test_image_risks_cold_bytes_ratio_warns() -> None:
    """When cold bytes constitute ≥ 50% of total bytes, a 'warn' risk fires."""
    from xrlenv.admin.server import _image_risks

    summary = {
        "unreachable_nodes": 0,
        "total_bytes": 100 * 1024**3,
        "cold_bytes": 60 * 1024**3,  # 60% cold
        "pinned_bytes": 0,
        "pinned_instances": 0,
    }
    risks = _image_risks(summary, [])
    assert any("Cold images" in r["text"] and r["level"] == "warn" for r in risks)


def test_image_risks_pinned_bytes_ratio_warns() -> None:
    """When pinned bytes constitute ≥ 50% of total bytes, a 'warn' risk fires."""
    from xrlenv.admin.server import _image_risks

    summary = {
        "unreachable_nodes": 0,
        "total_bytes": 100 * 1024**3,
        "cold_bytes": 0,
        "pinned_bytes": 55 * 1024**3,  # 55% pinned
        "pinned_instances": 10,
    }
    risks = _image_risks(summary, [])
    assert any("Pinned images" in r["text"] and r["level"] == "warn" for r in risks)


def test_image_risks_no_warnings_when_healthy() -> None:
    """A fully reachable cluster with ample free disk and low cold ratio is risk-free."""
    from xrlenv.admin.server import _image_risks

    summary = {
        "unreachable_nodes": 0,
        "total_bytes": 100 * 1024**3,
        "cold_bytes": 10 * 1024**3,  # 10% cold — fine
        "pinned_bytes": 5 * 1024**3,  # 5% pinned — fine
        "pinned_instances": 2,
    }
    risks = _image_risks(summary, [])
    assert risks == []


# ──────────────────────────────────────────────────────────────────────────────
# _page_items — pure-function unit tests
# ──────────────────────────────────────────────────────────────────────────────


def test_page_items_first_page() -> None:
    from xrlenv.admin.server import _page_items

    rows = [{"id": i} for i in range(10)]
    result = _page_items(rows, page=1, page_size=5)
    assert result["rows"] == rows[:5]
    assert result["total"] == 10
    assert result["first_index"] == 1
    assert result["last_index"] == 5
    assert result["has_prev"] is False
    assert result["has_next"] is True
    assert result["page_count"] == 2


def test_page_items_last_page() -> None:
    from xrlenv.admin.server import _page_items

    rows = [{"id": i} for i in range(10)]
    result = _page_items(rows, page=2, page_size=5)
    assert result["rows"] == rows[5:]
    assert result["has_prev"] is True
    assert result["has_next"] is False
    assert result["first_index"] == 6
    assert result["last_index"] == 10


def test_page_items_empty_list() -> None:
    from xrlenv.admin.server import _page_items

    result = _page_items([], page=1, page_size=25)
    assert result["rows"] == []
    assert result["total"] == 0
    assert result["first_index"] == 0
    assert result["last_index"] == 0
    assert result["has_prev"] is False
    assert result["has_next"] is False
    assert result["page_count"] == 1


def test_page_items_beyond_last_page_returns_empty() -> None:
    from xrlenv.admin.server import _page_items

    rows = [{"id": i} for i in range(5)]
    result = _page_items(rows, page=99, page_size=25)
    assert result["rows"] == []
    assert result["first_index"] == 0
    assert result["last_index"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# /images — page 2 navigation for the cluster nodes view
# ──────────────────────────────────────────────────────────────────────────────


def test_images_cluster_page2_shows_second_batch_of_nodes(
    state_db: Path, runs_root: Path,
) -> None:
    """Navigating to page=2 on the cluster nodes view shows the second batch
    and exposes the previous-page arrow; nodes from page 1 are absent."""
    from xrlenv.node.image_cache import NodeImageReport

    store = SqliteStateStore(state_db)
    node_ids = [f"node-{i:03d}" for i in range(60)]
    for nid in node_ids:
        store.record_node_connected(
            nid, backends=["docker"], stream_epoch="ep-1", instance_id="i1",
        )
    store.close()

    class _T:
        async def report_images(self, *, include_shared_size=False) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=100 * 1024**3, pinned=())

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _: _T(),
    )
    client = TestClient(build_admin_app(cfg))

    r = client.get("/images", params={"page_size": "25", "page": "2"})
    assert r.status_code == 200
    body = r.text
    assert "Showing 26-50 of 60 nodes" in body
    # Page 2 nodes are present (nodes sorted by pressure then node_id).
    assert 'aria-label="previous page"' in body
    # The prev-page arrow must be an <a>, not the disabled <span>.
    assert 'aria-label="previous page">&larr;</a>' in body


def test_dispatcher_wires_admin_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-test ``xrlenv up`` plumbs --admin-port through to the runtime."""
    import xrlenv.cli.__main__ as cli_module

    captured: dict[str, object] = {}

    async def _fake_serve(args: Any) -> int:
        captured["admin_port"] = args.admin_port
        captured["admin_host"] = args.admin_host
        captured["admin_allow_public"] = args.admin_allow_public
        captured["admin_rollout_page_size"] = args.admin_rollout_page_size
        return 0

    monkeypatch.setattr(cli_module, "_serve_control_plane", _fake_serve)
    rc = cli_module.main(
        [
            "--state-db", str(tmp_path / "state.db"),
            "--runs-root", str(tmp_path / "runs"),
            "up", "--admin-port", "1234", "--admin-host", "127.0.0.1",
            "--admin-rollout-page-size", "64",
        ]
    )
    assert rc == 0
    assert captured == {
        "admin_port": 1234, "admin_host": "127.0.0.1",
        "admin_allow_public": False, "admin_rollout_page_size": 64,
    }


# ──────────────────────────────────────────────────────────────────────────────
# _ImageRowsSnapshot — stale-while-revalidate image-rows cache (admin /images
# responsiveness under heavy builds)
# ──────────────────────────────────────────────────────────────────────────────


async def test_image_rows_snapshot_caches_and_revalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrlenv.admin import server as srv

    calls = {"n": 0}

    async def _fake_fetch(cfg: Any, node_ids: list[str]) -> list[dict[str, Any]]:
        calls["n"] += 1
        return [{"node_id": nid, "fetch": calls["n"]} for nid in node_ids]

    monkeypatch.setattr(srv, "_fetch_image_rows_for_nodes", _fake_fetch)

    snap = srv._ImageRowsSnapshot(ttl_s=10.0)
    # Cold cache → blocks on one fan-out.
    rows, age, refreshing = await snap.rows_for(None, ["a", "b"])
    assert calls["n"] == 1
    assert age == 0.0 and refreshing is False
    # Fresh within TTL → served from memory, no refetch.
    rows2, _age2, _r2 = await snap.rows_for(None, ["a", "b"])
    assert calls["n"] == 1
    assert rows2 == rows
    # Node set changed → key miss → refetch.
    await snap.rows_for(None, ["a", "b", "c"])
    assert calls["n"] == 2


async def test_image_rows_snapshot_stale_serves_stale_then_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrlenv.admin import server as srv

    calls = {"n": 0}

    async def _fake_fetch(cfg: Any, node_ids: list[str]) -> list[dict[str, Any]]:
        calls["n"] += 1
        return [{"fetch": calls["n"]}]

    monkeypatch.setattr(srv, "_fetch_image_rows_for_nodes", _fake_fetch)

    snap = srv._ImageRowsSnapshot(ttl_s=10.0)
    await snap.rows_for(None, ["a"])  # cold fetch → count=1
    snap._at = 0.0  # force age past the TTL

    # Stale read serves the OLD rows immediately and kicks off a refresh.
    rows, age, refreshing = await snap.rows_for(None, ["a"])
    assert rows == [{"fetch": 1}]
    assert age > 10.0
    assert refreshing is True

    # Let the background refresh complete; the snapshot is now fresh.
    assert snap._refresh_task is not None
    await snap._refresh_task
    assert calls["n"] == 2
    rows3, age3, _r3 = await snap.rows_for(None, ["a"])
    assert rows3 == [{"fetch": 2}]
    assert age3 < 10.0


# ──────────────────────────────────────────────────────────────────────────────
# /images: disk used / capacity columns sourced from disk_state (so the page's
# numbers reconcile and 'images size' is clearly not disk-used)
# ──────────────────────────────────────────────────────────────────────────────


async def test_image_row_surfaces_disk_used_and_total(
    state_db: Path, runs_root: Path,
) -> None:
    from xrlenv.admin.server import _fetch_image_row_for, _shape_image_node_rows
    from xrlenv.node.image_cache import ImageStateRecord, NodeImageReport

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(
                images=[ImageStateRecord(
                    name="x:1", tier="cold", size_bytes=10 * 1024**3,
                    in_use_count=0, last_used_at=None, pinned=False,
                )],
                free_disk_bytes=40 * 1024**3,
            )

        def disk_state(self) -> tuple[int, int]:
            return (40 * 1024**3, 500 * 1024**3)  # (free, total)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _n: _Transport(),
    )
    row = await _fetch_image_row_for(cfg, "n1")
    assert row["disk_total_bytes"] == 500 * 1024**3
    assert row["disk_free_bytes"] == 40 * 1024**3
    # "images size" is the per-image sum — deliberately distinct from disk used.
    assert row["total_bytes"] == 10 * 1024**3
    shaped = _shape_image_node_rows([row])[0]
    assert shaped["disk_used_bytes"] == 460 * 1024**3


async def test_image_row_disk_columns_default_zero_without_disk_state(
    state_db: Path, runs_root: Path,
) -> None:
    # A transport with no disk_state (older node) → capacity unknown (0), and
    # disk_free falls back to the report's statvfs free so the page still works.
    from xrlenv.admin.server import _fetch_image_row_for
    from xrlenv.node.image_cache import NodeImageReport

    class _Transport:
        async def report_images(
            self, *, include_shared_size: bool = False,
        ) -> NodeImageReport:
            return NodeImageReport(images=[], free_disk_bytes=7 * 1024**3)

    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        node_lookup=lambda _n: _Transport(),
    )
    row = await _fetch_image_row_for(cfg, "n1")
    assert row["disk_total_bytes"] == 0
    assert row["disk_free_bytes"] == 7 * 1024**3


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: pure-function unit tests (_safe_next / _wants_html)
# ──────────────────────────────────────────────────────────────────────────────


def test_safe_next_accepts_plain_path() -> None:
    """A simple absolute path starting with a single '/' is returned as-is."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("/nodes") == "/nodes"


def test_safe_next_accepts_path_with_query() -> None:
    """Paths with query strings are still same-site and must be honoured."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("/nodes?x=1") == "/nodes?x=1"


def test_safe_next_accepts_nested_path() -> None:
    """Multi-segment paths are valid same-site paths."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("/rollouts/raw?status=failed") == "/rollouts/raw?status=failed"


def test_safe_next_rejects_empty_string() -> None:
    """An empty string is not a valid redirect target; falls back to '/'."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("") == "/"


def test_safe_next_rejects_none() -> None:
    """None falls back to '/'."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next(None) == "/"


def test_safe_next_rejects_protocol_relative_double_slash() -> None:
    """'//evil.example' is resolved by browsers as another host; must fall back."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("//evil.example/x") == "/"


def test_safe_next_rejects_absolute_https_url() -> None:
    """An absolute https URL is an open redirect; must fall back to '/'."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("https://evil.example") == "/"


def test_safe_next_rejects_absolute_http_url() -> None:
    """An absolute http URL is an open redirect; must fall back to '/'."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("http://evil.example/path") == "/"


def test_safe_next_rejects_javascript_scheme() -> None:
    """javascript: URIs are a classic XSS vector; must fall back to '/'."""
    from xrlenv.admin.server import _safe_next
    assert _safe_next("javascript:alert(1)") == "/"


def test_safe_next_rejects_backslash_path() -> None:
    """A backslash-prefixed path (Windows-style) is not a valid URL path;
    some browsers normalise '\\evil.example' to '//evil.example'."""
    from xrlenv.admin.server import _safe_next
    # Does not start with '/', so falls through to the default.
    assert _safe_next("\\evil.example") == "/"


def test_wants_html_true_for_get_with_text_html() -> None:
    """A GET with 'text/html' in Accept is a browser navigation."""
    from unittest.mock import MagicMock

    from xrlenv.admin.server import _wants_html

    request = MagicMock()
    request.method = "GET"
    request.headers = {"accept": "text/html,application/xhtml+xml"}
    assert _wants_html(request) is True


def test_wants_html_false_for_get_with_wildcard_accept() -> None:
    """A CLI / curl caller sends Accept: */*; must NOT trigger HTML redirect."""
    from unittest.mock import MagicMock

    from xrlenv.admin.server import _wants_html

    request = MagicMock()
    request.method = "GET"
    request.headers = {"accept": "*/*"}
    assert _wants_html(request) is False


def test_wants_html_false_for_post_even_with_text_html_accept() -> None:
    """POST requests are never browser navigations for auth-redirect purposes."""
    from unittest.mock import MagicMock

    from xrlenv.admin.server import _wants_html

    request = MagicMock()
    request.method = "POST"
    request.headers = {"accept": "text/html"}
    assert _wants_html(request) is False


def test_wants_html_false_for_get_with_no_accept_header() -> None:
    """A GET with no Accept header does not qualify as a browser navigation."""
    from unittest.mock import MagicMock

    from xrlenv.admin.server import _wants_html

    request = MagicMock()
    request.method = "GET"
    request.headers = {}
    assert _wants_html(request) is False


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: POST /login edge cases
# ──────────────────────────────────────────────────────────────────────────────

def _public_store() -> Any:
    """Return a pre-populated TokenStore for public-bind tests."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    store.add("viewer", "read_view-tok")
    store.add("consumer", "read_con-tok")
    return store


def _public_cfg(state_db: Path, runs_root: Path) -> AdminServerConfig:
    return AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=_public_store(),
    )


def test_login_post_empty_body_returns_401_no_cookie(
    state_db: Path, runs_root: Path,
) -> None:
    """POST /login with a completely empty body re-renders the form with 401.
    No session cookie must be set — we should never grant a session on a
    missing token."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.post(
        "/login", content="",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 401
    assert _SESSION_COOKIE not in r.cookies


def test_login_post_missing_token_field_returns_401_no_cookie(
    state_db: Path, runs_root: Path,
) -> None:
    """POST /login with a body that omits the 'token' field entirely
    re-renders 401 and sets no session cookie."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.post(
        "/login", content="next=%2Fnodes",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 401
    assert _SESSION_COOKIE not in r.cookies


def test_login_post_blank_token_returns_401_no_cookie(
    state_db: Path, runs_root: Path,
) -> None:
    """POST /login with token= (present but blank after strip) → 401, no cookie.
    Guards against a user who submits the form without pasting a token."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.post(
        "/login", content="token=   ",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 401
    assert _SESSION_COOKIE not in r.cookies


def test_login_post_malformed_body_returns_401_not_500(
    state_db: Path, runs_root: Path,
) -> None:
    """Garbage in the POST body must not crash the server — parse_qs is lenient
    and should treat it as an empty form (no 'token') → 401, never 500."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.post(
        "/login", content="\x00\xff\xfe garbage %%% ==",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 401
    assert r.status_code != 500
    assert _SESSION_COOKIE not in r.cookies


def test_login_post_with_whitespace_around_token_succeeds(
    state_db: Path, runs_root: Path,
) -> None:
    """The token field is .strip()-ped, so leading/trailing whitespace does
    not cause a false rejection for a real token."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.post(
        "/login", content="token=+read_view-tok+",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    # URL-encoded space is '+'; strip() on "read_view-tok" with spaces → match.
    # '+' decodes to ' ', so the stripped token is "read_view-tok".
    assert r.status_code == 303
    assert _SESSION_COOKIE in r.cookies


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: cookie attributes
# ──────────────────────────────────────────────────────────────────────────────


def test_login_post_cookie_attributes(
    state_db: Path, runs_root: Path,
) -> None:
    """The session cookie set by POST /login must be HttpOnly, SameSite=Lax,
    Path=/, and have Max-Age close to 7 days (604800 s)."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.post(
        "/login", content="token=read_view-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 303
    # Starlette's TestClient exposes cookies, but the raw Set-Cookie header
    # carries the attributes we care about.
    set_cookie = r.headers.get("set-cookie", "")
    assert _SESSION_COOKIE in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie.lower().replace("samesite=", "SameSite=")
    assert "Path=/" in set_cookie
    # Max-Age must be present and ≈ 604800 (7 days).
    import re
    m = re.search(r"Max-Age=(\d+)", set_cookie, re.IGNORECASE)
    assert m is not None, f"Max-Age missing from Set-Cookie: {set_cookie!r}"
    max_age = int(m.group(1))
    assert max_age == 7 * 24 * 3600, f"Expected 604800, got {max_age}"


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: GET /login edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_login_get_already_signed_in_redirects_to_next(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /login when already signed in (valid cookie) should 303 to the
    requested ?next= target rather than rendering the form again."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    # Sign in to obtain the session cookie.
    client.post(
        "/login", content="token=read_view-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    # GET /login with a valid next param and an active session → redirect, no form.
    r = client.get("/login", params={"next": "/nodes"},
                   headers={"accept": "text/html"})
    assert r.status_code == 303
    assert r.headers["location"] == "/nodes"


def test_login_get_already_signed_in_does_not_clear_cookie(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /login when already signed in must not delete the cookie — that
    would log the operator out just by visiting the login URL."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    client.post(
        "/login", content="token=read_view-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    r = client.get("/login", headers={"accept": "text/html"})
    assert r.status_code == 303
    # The response must not delete the cookie (no max-age=0 or expires=past).
    set_cookie = r.headers.get("set-cookie", "")
    # A deletion sets Max-Age=0; an absence of set-cookie header is fine too.
    if set_cookie and _SESSION_COOKIE in set_cookie:
        assert "Max-Age=0" not in set_cookie


def test_login_get_loopback_bind_redirects_immediately(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /login on a loopback bind (no auth at all) must 303-redirect
    immediately — there's nothing to sign into, so the form is never shown."""
    # Default cfg uses host="127.0.0.1" → loopback.
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="127.0.0.1",
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.get("/login", headers={"accept": "text/html"})
    assert r.status_code == 303


def test_login_get_loopback_redirects_to_safe_next(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /login?next=/nodes on loopback redirects to /nodes (not to the form).
    The ?next= must still pass through _safe_next, so an evil next → /."""
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="127.0.0.1",
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.get("/login", params={"next": "/nodes"},
                   headers={"accept": "text/html"})
    assert r.status_code == 303
    assert r.headers["location"] == "/nodes"

    r2 = client.get("/login", params={"next": "//evil.example"},
                    headers={"accept": "text/html"})
    assert r2.status_code == 303
    assert r2.headers["location"] == "/"


def test_login_get_stale_cookie_is_cleared_on_form_render(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /login with a stale/revoked cookie in the request renders the form
    (not a redirect) and clears the bad cookie so the browser starts fresh."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("viewer", "read_view-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    # Send a stale cookie that the store doesn't know.
    r = client.get(
        "/login",
        headers={
            **_session_cookie("stale-token-not-in-store"),
            "accept": "text/html",
        },
    )
    # Form is rendered (not a redirect, because the cookie is invalid).
    assert r.status_code == 200
    # The response must delete the stale cookie.
    set_cookie = r.headers.get("set-cookie", "")
    assert _SESSION_COOKIE in set_cookie
    # Deletion is signalled by Max-Age=0 or an expires in the past.
    assert "Max-Age=0" in set_cookie or "expires" in set_cookie.lower()


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: POST /logout edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_logout_when_not_signed_in_is_idempotent(
    state_db: Path, runs_root: Path,
) -> None:
    """POST /logout with no session cookie still 303-redirects to /login
    and does not raise a 400/500 — logout is idempotent."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.post("/logout")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_logout_deletes_cookie(
    state_db: Path, runs_root: Path,
) -> None:
    """POST /logout must emit a Set-Cookie header that deletes the session
    cookie (Max-Age=0 or expiry in the past)."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    # Sign in first so there is a cookie to delete.
    client.post(
        "/login", content="token=read_view-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    r = client.post("/logout")
    assert r.status_code == 303
    set_cookie = r.headers.get("set-cookie", "")
    # Must carry a Set-Cookie that clears the session.
    assert _SESSION_COOKIE in set_cookie
    assert "Max-Age=0" in set_cookie or "expires" in set_cookie.lower()


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: POST to gated route → JSON 401 (not redirect)
# ──────────────────────────────────────────────────────────────────────────────


def test_post_without_creds_returns_json_401_not_redirect(
    state_db: Path, runs_root: Path,
) -> None:
    """A POST (non-GET) to a gated route with no credentials must return a
    JSON 401 — not a 303 redirect — because _wants_html is False for POSTs.
    This holds even when the Accept header says text/html."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.post(
        "/api/build/apply",
        json={"plan": {"version": 1, "benchmarks": []}},
        headers={"accept": "text/html"},
    )
    # Must be a JSON 401, not a redirect.
    assert r.status_code == 401
    assert r.status_code != 303


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: consumer-role login and read access
# ──────────────────────────────────────────────────────────────────────────────


def test_consumer_role_can_sign_in_and_read(
    state_db: Path, runs_root: Path,
) -> None:
    """A consumer-role token is in _ADMIN_READ_ROLES and must be able to sign
    in via POST /login and subsequently read the admin panel (GET /nodes)."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("consumer", "read_con-tok")
    store.add("viewer", "read_view-tok")  # needed for bind guard (admin-capable check)
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.post(
        "/login", content="token=read_con-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    # Successful login → 303 redirect + cookie set.
    assert r.status_code == 303
    assert _SESSION_COOKIE in r.cookies
    # Cookie session grants access to a read route.
    r2 = client.get("/nodes", headers={"accept": "text/html"})
    assert r2.status_code == 200


def test_consumer_login_succeeds_consumer_is_a_read_role(
    state_db: Path, runs_root: Path,
) -> None:
    """A consumer CAN sign in (consumer IS in _ADMIN_READ_ROLES). This is
    distinct from the bind guard, which needs at least one viewer/operator
    token — that check happens at startup, not login. We verify the login
    itself succeeds (the bind guard is bypassed in-process via
    build_admin_app)."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    store.add("consumer", "read_con-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    r = client.post(
        "/login", content="token=read_con-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 303, "consumer token must be accepted at /login"
    assert _SESSION_COOKIE in r.cookies


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: login.html template content
# ──────────────────────────────────────────────────────────────────────────────


def test_login_html_has_token_field_and_hidden_next(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /login renders the standalone login.html which must contain:
    - a password input named 'token'
    - a hidden input named 'next' carrying the sanitized next value
    It must NOT include the nav bar (no /logout button) since login.html
    is standalone (does not extend base.html)."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    r = client.get("/login", params={"next": "/nodes"},
                   headers={"accept": "text/html"})
    assert r.status_code == 200
    body = r.text
    # Token password field present.
    assert 'name="token"' in body
    assert 'type="password"' in body
    # Hidden next field with the sanitized value.
    assert 'name="next"' in body
    assert 'value="/nodes"' in body
    # No nav bar / logout form (standalone page, not base.html-derived).
    assert "/logout" not in body
    assert "log out" not in body.lower()


def test_login_html_evil_next_sanitized_to_root_in_form(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /login with a hostile ?next= renders the form with next='/';
    the evil URL must not appear anywhere in the rendered HTML."""
    client = TestClient(build_admin_app(_public_cfg(state_db, runs_root)),
                        follow_redirects=False)
    evil = "//evil.example/steal"
    r = client.get("/login", params={"next": evil},
                   headers={"accept": "text/html"})
    assert r.status_code == 200
    assert evil not in r.text
    assert 'value="/"' in r.text


# ──────────────────────────────────────────────────────────────────────────────
# B7.4 cookie-session: WWW-Authenticate challenge header
# ──────────────────────────────────────────────────────────────────────────────


def test_auth_challenge_header_is_bearer_not_basic(
    state_db: Path, runs_root: Path,
) -> None:
    """The WWW-Authenticate challenge emitted on 401 must advertise Bearer,
    never Basic. A Basic challenge pops the browser's native credential dialog
    which caches creds and makes logout impossible — exactly the bug B7.4 fixes.
    Verified for both the no-cookie 401 and the revoked-cookie 401 paths."""
    from xrlenv.control.security import TokenStore
    store = TokenStore()
    store.add("operator", "write_op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True,
        token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)
    # No credentials at all → 401 with Bearer challenge.
    r = client.get("/nodes", headers={"accept": "*/*"})
    assert r.status_code == 401
    challenge = r.headers.get("www-authenticate", "")
    assert "Bearer" in challenge
    assert "Basic" not in challenge
    # Revoked / unknown cookie → 401 with Bearer challenge.
    r2 = client.get(
        "/nodes",
        headers={**_session_cookie("revoked-token"), "accept": "*/*"},
    )
    assert r2.status_code == 401
    challenge2 = r2.headers.get("www-authenticate", "")
    assert "Bearer" in challenge2
    assert "Basic" not in challenge2


# ──────────────────────────────────────────────────────────────────────────────
# /users — per-tenant raw-rollout scoreboard (operator-only)
# ──────────────────────────────────────────────────────────────────────────────


def _raw(rollout_id: str, owner: str, status: str, node: str | None) -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id, status=status, image="img:1",
        owner_id=owner, node_id=node,
    )


def _seed_users_mix(store: SqliteStateStore) -> None:
    for rec in (
        _raw("a1", "alice", "released", "node-A"),
        _raw("a2", "alice", "released", "node-A"),
        _raw("a3", "alice", "failed", "node-B"),
        _raw("a4", "alice", "running", "node-A"),
        _raw("b1", "bob", "released", "node-B"),
        _raw("b2", "bob", "reaped", "node-B"),
        _raw("b3", "bob", "cancelled", None),
    ):
        store.record_raw_rollout(rec)


def test_users_page_aggregates_per_owner(
    client: TestClient, state_db: Path,
) -> None:
    store = SqliteStateStore(state_db)
    _seed_users_mix(store)
    store.close()

    r = client.get("/users")
    assert r.status_code == 200
    body = r.text
    assert "alice" in body and "bob" in body
    # alice: 4 total, 2 released → 50.0% success; bob: 3 total, 1 released → 33.3%.
    assert "50.0%" in body
    assert "33.3%" in body
    # A totals footer reconciles the columns (7 total across both owners).
    assert "all" in body


def test_users_paced_excluded_from_success_denominator(
    state_db: Path, runs_root: Path,
) -> None:
    """A ``capacity_rejected`` (paced) row is a scheduler decline, not an
    outcome: it shows in its own ``paced`` column but is EXCLUDED from
    ``total`` and the success denominator, so a paced-then-retried run isn't
    scored as a partial failure."""
    from xrlenv.admin.server import _users_blocking

    store = SqliteStateStore(state_db)
    for rec in (
        _raw("a1", "alice", "released", "node-A"),
        _raw("a2", "alice", "released", "node-A"),
        _raw("a3", "alice", "capacity_rejected", "node-A"),
        _raw("a4", "alice", "capacity_rejected", "node-A"),
    ):
        store.record_raw_rollout(rec)
    store.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root, port=0)
    data = _users_blocking(cfg)
    (alice,) = data["rows"]
    assert alice["owner"] == "alice"
    assert alice["paced"] == 2
    assert alice["released"] == 2
    # total EXCLUDES the two paced declines → 2, so success is 2/2 = 100%,
    # not 2/4 = 50%.
    assert alice["total"] == 2
    assert alice["success_pct"] == 100.0
    assert data["totals"]["paced"] == 2
    assert data["totals"]["total"] == 2

    # The rendered page carries the paced column + value.
    client = TestClient(build_admin_app(cfg))
    body = client.get("/users").text
    assert "paced" in body
    assert "100.0%" in body


def test_users_page_empty(client: TestClient) -> None:
    r = client.get("/users")
    assert r.status_code == 200
    assert "No raw rollouts recorded yet" in r.text


def test_users_page_operator_only_on_public_bind(
    state_db: Path, runs_root: Path,
) -> None:
    """GET /users is gated to the operator role even though it's a read route —
    a viewer/consumer token must not see every tenant's activity."""
    from xrlenv.control.security import TokenStore

    store = TokenStore()
    store.add("viewer", "view-tok")
    store.add("operator", "op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True, token_store=store,
    )
    client = TestClient(build_admin_app(cfg), follow_redirects=False)

    # Viewer is authenticated but lacks the role → 403 (not a login redirect).
    viewer = client.get(
        "/users", headers={"authorization": "Bearer view-tok", "accept": "*/*"}
    )
    assert viewer.status_code == 403

    # Operator → 200.
    operator = client.get(
        "/users", headers={"authorization": "Bearer op-tok", "accept": "*/*"}
    )
    assert operator.status_code == 200

    # A credential-less browser GET is redirected to the sign-in form.
    anon = client.get("/users", headers={"accept": "text/html"})
    assert anon.status_code == 303
    assert anon.headers["location"] == "/login?next=%2Fusers"


def test_users_nav_link_hidden_for_viewer_shown_for_operator(
    state_db: Path, runs_root: Path,
) -> None:
    from xrlenv.control.security import TokenStore

    store = TokenStore()
    store.add("viewer", "view-tok")
    store.add("operator", "op-tok")
    cfg = AdminServerConfig(
        state_db=state_db, runs_root=runs_root, port=0,
        host="0.0.0.0", allow_public=True, token_store=store,
    )

    # Operator session sees the nav link.
    op_client = TestClient(build_admin_app(cfg))
    op_client.post(
        "/login", content="token=op-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert 'href="/users"' in op_client.get("/").text

    # Viewer session does not.
    vw_client = TestClient(build_admin_app(cfg))
    vw_client.post(
        "/login", content="token=view-tok",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert 'href="/users"' not in vw_client.get("/nodes").text


# ──────────────────────────────────────────────────────────────────────────────
# /nodes — rollout distribution figure
# ──────────────────────────────────────────────────────────────────────────────


def test_nodes_page_shows_rollout_distribution(
    client: TestClient, state_db: Path,
) -> None:
    store = SqliteStateStore(state_db)
    for nid in ("node-A", "node-B", "node-C"):  # node-C stays idle
        store.record_node_connected(nid, backends=["docker"])
    for rec in (
        _raw("a1", "alice", "released", "node-A"),
        _raw("a2", "alice", "released", "node-A"),
        _raw("a3", "alice", "failed", "node-A"),
        _raw("b1", "bob", "released", "node-B"),
        _raw("u1", "alice", "acquiring", None),       # unassigned
        _raw("g1", "carol", "released", "node-GONE"), # off-roster
    ):
        store.record_raw_rollout(rec)
    store.close()

    body = client.get("/nodes").text
    assert "Rollout distribution across nodes" in body
    assert "spread (CV)" in body
    # Idle rostered node appears as a 0-bar (so imbalance is visible). The
    # off-roster node is NOT drawn as a per-node bar — after a reboot its
    # all-time history would dominate the figure and drown out the live fleet —
    # so it's folded into the "no longer in the roster" note instead, and the
    # not-yet-assigned rollout is reconciled in its own note.
    assert "node-C" in body
    assert "node-GONE" not in body, "off-roster node must not be drawn as a bar"
    assert "(gone)" not in body, "off-roster nodes are folded into a note, not bars"
    assert "not yet assigned" in body
    assert "no longer in the roster" in body
    # The bars must carry inline geometry so they render even if a browser
    # serves a stale cached stylesheet (regression guard). The busiest node
    # (node-A, 3 rollouts) is the full-width bar.
    fills = re.findall(r'distbar-fill"\s+style="width:([\d.]+)%', body)
    assert fills, "distribution bars are not inline-styled"
    assert any(float(w) >= 99.0 for w in fills), "busiest node bar full width"


def test_node_distribution_folds_gone_nodes_into_note(
    cfg: AdminServerConfig, state_db: Path,
) -> None:
    """The distribution figure draws bars ONLY for the current roster; rollouts
    on nodes no longer in the roster (reboot-orphaned IP-derived ids) are folded
    into the ``off_roster`` count, not per-node bars — and the pct/total are
    computed over the live fleet so a busy dead node can't pin the live nodes at
    ~0%."""
    from xrlenv.admin.server import _node_distribution_blocking

    store = SqliteStateStore(state_db)
    for i in range(20):
        store.record_raw_rollout(_raw(f"ga{i}", "alice", "released", "aws-ip-gone-a"))
    for i in range(10):
        store.record_raw_rollout(_raw(f"gb{i}", "bob", "released", "aws-ip-gone-b"))
    for i in range(3):
        store.record_raw_rollout(_raw(f"l{i}", "alice", "released", "aws-ip-live-1"))
    for i in range(2):
        store.record_raw_rollout(_raw(f"u{i}", "alice", "acquiring", None))
    store.close()

    # Roster = the current fleet only (live-2 is idle — no rollouts).
    node_rows = [{"id": "aws-ip-live-1"}, {"id": "aws-ip-live-2"}]
    dist = _node_distribution_blocking(cfg, node_rows)

    by_id = {e["node"]: e for e in dist["entries"]}
    assert set(by_id) == {"aws-ip-live-1", "aws-ip-live-2"}   # gone nodes excluded
    assert all(e["rostered"] for e in dist["entries"])
    assert by_id["aws-ip-live-2"]["count"] == 0               # idle node = 0-bar
    assert by_id["aws-ip-live-1"]["count"] == 3
    assert by_id["aws-ip-live-1"]["pct"] == 100.0             # over the live fleet
    assert dist["total"] == 3
    assert dist["node_count"] == 2
    assert dist["off_roster"] == 30                           # 20 + 10, folded
    assert dist["unassigned"] == 2


def test_node_distribution_no_roster_shows_every_node(
    cfg: AdminServerConfig, state_db: Path,
) -> None:
    """With no roster yet (bare control plane) the figure falls back to showing
    every node with history so it isn't empty; nothing is flagged off-roster."""
    from xrlenv.admin.server import _node_distribution_blocking

    store = SqliteStateStore(state_db)
    store.record_raw_rollout(_raw("r1", "alice", "released", "node-x"))
    store.record_raw_rollout(_raw("r2", "bob", "released", "node-y"))
    store.close()

    dist = _node_distribution_blocking(cfg, [])
    assert {e["node"] for e in dist["entries"]} == {"node-x", "node-y"}
    assert all(e["rostered"] for e in dist["entries"])
    assert dist["off_roster"] == 0


def test_scratch_active_digests_endpoint(
    client: TestClient, state_db: Path,
) -> None:
    """GET /api/scratch/active-digests returns the scratch repos referenced by
    running sandboxes (the GC's active-run exemption set); non-scratch images
    are excluded."""
    from xrlenv.control.state import SqliteStateStore

    store = SqliteStateStore(state_db)
    _seed_sandbox(store, sandbox_id="s1", node_id="n1", image="cp:5012/scratch/aaa:latest")
    _seed_sandbox(store, sandbox_id="s2", node_id="n1", image="docker.io/library/busybox:latest")
    store.close()

    resp = client.get("/api/scratch/active-digests")
    assert resp.status_code == 200
    assert resp.json() == {"repos": ["scratch/aaa"]}


def test_scratch_active_digests_empty_when_no_scratch_sandboxes(
    client: TestClient, state_db: Path,
) -> None:
    from xrlenv.control.state import SqliteStateStore

    store = SqliteStateStore(state_db)
    _seed_sandbox(store, sandbox_id="s1", node_id="n1", image="busybox:latest")
    store.close()
    resp = client.get("/api/scratch/active-digests")
    assert resp.status_code == 200
    assert resp.json() == {"repos": []}
