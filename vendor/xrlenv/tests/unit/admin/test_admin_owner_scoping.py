"""Multi-user (Slice B) — admin-panel per-owner scoping.

The admin panel scopes reads by the caller's tenant: a per-user ``viewer``
token sees only its own owner's rollouts / sandboxes / artifacts; an
``operator`` token (and the loopback no-auth dev flow) sees everything.

Two layers are covered:

1. Pure helpers ``_caller_owner_id`` + ``_owner_forbidden`` (no app needed).
2. Route scoping through a FastAPI ``TestClient`` (ASGI, no real port bind),
   mirroring the harness in ``test_admin_raw_rollouts.py`` but with a
   ``token_store`` wired so the auth middleware stashes an identity.
"""

from __future__ import annotations

import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xrlenv.admin.server import (
    _SESSION_COOKIE,
    AdminServerConfig,
    _caller_owner_id,
    _owner_forbidden,
    build_admin_app,
)
from xrlenv.control.security import TokenStore, write_user_record
from xrlenv.control.state import (
    RawRolloutRecord,
    RolloutRecord,
    SandboxRecord,
    SqliteStateStore,
)
from xrlenv.types import RolloutStatus

# ── Pure helpers ──────────────────────────────────────────────────────────────


def _fake_request(identity: object | None):
    """A stand-in Request whose only read surface is ``request.state.identity``."""
    return types.SimpleNamespace(state=types.SimpleNamespace(identity=identity))


def test_caller_owner_id_none_for_no_identity() -> None:
    # Loopback / no-auth dev flow: no identity stashed → admin (sees all).
    assert _caller_owner_id(_fake_request(None)) is None


def test_caller_owner_id_none_for_operator() -> None:
    operator = types.SimpleNamespace(role="operator", owner_id="x")
    assert _caller_owner_id(_fake_request(operator)) is None


def test_caller_owner_id_returns_owner_for_viewer() -> None:
    viewer = types.SimpleNamespace(role="viewer", owner_id="alice")
    assert _caller_owner_id(_fake_request(viewer)) == "alice"


def test_owner_forbidden_admin_never_blocked() -> None:
    # caller_owner None → admin → never forbidden, regardless of record owner.
    assert _owner_forbidden(None, "alice") is False
    assert _owner_forbidden(None, None) is False


def test_owner_forbidden_same_owner_allowed() -> None:
    assert _owner_forbidden("alice", "alice") is False


def test_owner_forbidden_other_owner_blocked() -> None:
    assert _owner_forbidden("alice", "bob") is True


def test_owner_forbidden_unknown_record_owner_blocked() -> None:
    # A record whose owner can't be resolved is treated as not-the-caller's.
    assert _owner_forbidden("alice", None) is True


# ── Route scoping harness ─────────────────────────────────────────────────────


VIEWER_ALICE_TOKEN = "alice-viewer-secret"
CONSUMER_ALICE_TOKEN = "alice-consumer-secret"
OPERATOR_TOKEN = "op-secret"


@pytest.fixture
def scoped_cfg(tmp_path: Path) -> AdminServerConfig:
    """Admin config with a token store holding a viewer-for-alice token and an
    operator token, plus a seeded state db (alice / bob / default rollouts)."""
    state_db = tmp_path / "state.db"
    store = SqliteStateStore(state_db)
    now = time.time()
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-alice", status="released", image="busybox:1",
        displayed_name="instance-ALICE", owner_id="alice", created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-bob", status="released", image="busybox:1",
        displayed_name="instance-BOB", owner_id="bob", created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-default", status="released", image="busybox:1",
        displayed_name="instance-DEFAULT", created_at=now,
    ))
    store.close()

    secrets = tmp_path / "secrets"
    users = secrets / "users.json"
    write_user_record(
        users, token=VIEWER_ALICE_TOKEN, role="viewer", owner_id="alice",
    )
    # alice's per-user CONSUMER token (the one she puts in .env to submit jobs)
    # now also opens the admin, read-only + owner-scoped — one token per user.
    write_user_record(
        users, token=CONSUMER_ALICE_TOKEN, role="consumer", owner_id="alice",
    )
    token_store = TokenStore.load(secrets_root=secrets, env={})
    token_store.add("operator", OPERATOR_TOKEN)

    # Public bind so the auth middleware engages and stashes
    # request.state.identity (loopback binds bypass auth entirely, which would
    # leave every caller as admin/see-all). build_admin_app does not run the
    # bind guard, and TestClient never binds a real socket, so a public host
    # string is safe in-process.
    return AdminServerConfig(
        state_db=state_db,
        runs_root=tmp_path / "runs",
        host="0.0.0.0",
        allow_public=True,
        token_store=token_store,
    )


def _session(token: str) -> dict[str, str]:
    """Browser auth header: the B7.4 cookie session. A signed-in browser
    carries the token in the ``xrlenv_admin_session`` cookie (set by
    ``POST /login``); passing it as a ``Cookie`` header drives the exact
    ``_require_role`` cookie-verification path a real session would."""
    return {"Cookie": f"{_SESSION_COOKIE}={token}"}


def test_raw_list_scoped_to_viewer_owner(scoped_cfg: AdminServerConfig) -> None:
    app = build_admin_app(scoped_cfg)
    with TestClient(app) as client:
        body = client.get(
            "/rollouts/raw", headers=_session(VIEWER_ALICE_TOKEN),
        ).text
    assert "instance-ALICE" in body
    assert "instance-BOB" not in body
    assert "instance-DEFAULT" not in body


def test_raw_list_operator_sees_all(scoped_cfg: AdminServerConfig) -> None:
    app = build_admin_app(scoped_cfg)
    with TestClient(app) as client:
        # Operator uses bearer (the CLI path); basic-auth would also work.
        body = client.get(
            "/rollouts/raw",
            headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"},
        ).text
    assert "instance-ALICE" in body
    assert "instance-BOB" in body
    assert "instance-DEFAULT" in body


def test_raw_list_no_auth_flow_sees_all(tmp_path: Path) -> None:
    """The harness's default no-token config (empty/absent TokenStore) is the
    loopback dev flow: scoping is inert, every owner is visible. Pins that
    Slice B doesn't change single-tenant behaviour."""
    state_db = tmp_path / "state.db"
    store = SqliteStateStore(state_db)
    now = time.time()
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-alice", status="released", image="busybox:1",
        displayed_name="instance-ALICE", owner_id="alice", created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-bob", status="released", image="busybox:1",
        displayed_name="instance-BOB", owner_id="bob", created_at=now,
    ))
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=tmp_path / "runs")
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        body = client.get("/rollouts/raw").text
    assert "instance-ALICE" in body
    assert "instance-BOB" in body


def test_cross_owner_detail_404(scoped_cfg: AdminServerConfig) -> None:
    app = build_admin_app(scoped_cfg)
    with TestClient(app) as client:
        hdrs = _session(VIEWER_ALICE_TOKEN)
        # Alice may not read bob's rollout detail.
        assert client.get("/raw-rollouts/r-bob", headers=hdrs).status_code == 404
        # But may read her own.
        assert client.get("/raw-rollouts/r-alice", headers=hdrs).status_code == 200


def test_cross_owner_detail_visible_to_operator(
    scoped_cfg: AdminServerConfig,
) -> None:
    app = build_admin_app(scoped_cfg)
    with TestClient(app) as client:
        hdrs = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}
        assert client.get("/raw-rollouts/r-bob", headers=hdrs).status_code == 200
        assert client.get("/raw-rollouts/r-alice", headers=hdrs).status_code == 200


# ── Sandbox scoping ───────────────────────────────────────────────────────────


@pytest.fixture
def sandbox_cfg(tmp_path: Path) -> AdminServerConfig:
    """Seed a gym rollout owned by bob with a sandbox joined to it, plus an
    alice-owned rollout + sandbox, so /sandboxes scoping can be checked."""
    state_db = tmp_path / "state.db"
    store = SqliteStateStore(state_db)
    now = time.time()

    def _seed(rollout_id: str, owner: str, sandbox_id: str) -> None:
        store.insert_rollout(RolloutRecord(
            rollout_id=rollout_id, template="hello-shell",
            status=RolloutStatus.RUNNING, owner_id=owner,
        ))
        store.insert_sandbox(SandboxRecord(
            sandbox_id=sandbox_id, backend="docker", backend_ref="ref",
            stub_endpoint="tcp://x", template="hello-shell",
            node_id="node-A", rollout_id=rollout_id,
            created_at=now,
        ))

    _seed("g-alice", "alice", "sb-alice")
    _seed("g-bob", "bob", "sb-bob")
    store.close()

    secrets = tmp_path / "secrets"
    write_user_record(
        secrets / "users.json", token=VIEWER_ALICE_TOKEN,
        role="viewer", owner_id="alice",
    )
    token_store = TokenStore.load(secrets_root=secrets, env={})
    token_store.add("operator", OPERATOR_TOKEN)

    return AdminServerConfig(
        state_db=state_db, runs_root=tmp_path / "runs",
        host="0.0.0.0", allow_public=True,
        token_store=token_store,
    )


def test_sandboxes_scoped_to_viewer_owner(sandbox_cfg: AdminServerConfig) -> None:
    app = build_admin_app(sandbox_cfg)
    with TestClient(app) as client:
        body = client.get(
            "/sandboxes", headers=_session(VIEWER_ALICE_TOKEN),
        ).text
    assert "sb-alice" in body
    assert "sb-bob" not in body


def test_sandboxes_operator_sees_all(sandbox_cfg: AdminServerConfig) -> None:
    app = build_admin_app(sandbox_cfg)
    with TestClient(app) as client:
        body = client.get(
            "/sandboxes",
            headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"},
        ).text
    assert "sb-alice" in body
    assert "sb-bob" in body


# ── Consumer token opens a read-only, owner-scoped dashboard ──────────────────


def test_consumer_token_logs_in_scoped_to_own_jobs(
    scoped_cfg: AdminServerConfig,
) -> None:
    """A per-user consumer token opens the admin read-only, scoped to its owner
    — so a user needs just the one token they already keep in .env."""
    app = build_admin_app(scoped_cfg)
    client = TestClient(app)
    r = client.get("/rollouts/raw", headers=_session(CONSUMER_ALICE_TOKEN))
    assert r.status_code == 200
    assert "instance-ALICE" in r.text
    assert "instance-BOB" not in r.text
    assert "instance-DEFAULT" not in r.text


def test_consumer_login_flow_sets_scoped_session(
    scoped_cfg: AdminServerConfig,
) -> None:
    """The real browser flow: POST /login with a consumer token mints a cookie
    session that is owner-scoped, so a user signs in with the one token they
    already keep in .env."""
    app = build_admin_app(scoped_cfg)
    # follow_redirects so the post-login 303 lands on the scoped page.
    client = TestClient(app, follow_redirects=False)
    r = client.post(
        "/login",
        data={"token": CONSUMER_ALICE_TOKEN, "next": "/rollouts/raw"},
        headers={"accept": "text/html"},
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/rollouts/raw"
    assert _SESSION_COOKIE in r.cookies  # the jar now carries the session
    body = client.get("/rollouts/raw", headers={"accept": "text/html"}).text
    assert "instance-ALICE" in body
    assert "instance-BOB" not in body


def test_consumer_sees_infra_tabs_but_not_writes_or_fairshare(
    scoped_cfg: AdminServerConfig,
) -> None:
    app = build_admin_app(scoped_cfg)
    client = TestClient(app)
    hdr = _session(CONSUMER_ALICE_TOKEN)
    # Read-only cluster-infra tabs are visible (global, no per-user data).
    assert client.get("/capacity", headers=hdr).status_code == 200
    assert client.get("/nodes", headers=hdr).status_code == 200
    # Writes stay operator-only.
    assert client.post("/api/build/apply", headers=hdr, json={}).status_code in (401, 403)
    # Fair-share tab is operator-only — a scoped consumer is 404'd.
    assert client.get("/fairshare", headers=hdr).status_code == 404


def test_consumer_cross_owner_detail_404(scoped_cfg: AdminServerConfig) -> None:
    """A consumer can't open another owner's rollout by id."""
    app = build_admin_app(scoped_cfg)
    client = TestClient(app)
    hdr = _session(CONSUMER_ALICE_TOKEN)
    assert client.get("/raw-rollouts/r-bob", headers=hdr).status_code == 404
    assert client.get("/raw-rollouts/r-alice", headers=hdr).status_code == 200
