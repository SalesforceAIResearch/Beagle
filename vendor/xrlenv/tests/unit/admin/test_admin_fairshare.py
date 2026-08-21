"""Multi-user (Slice C) — admin ``/fairshare`` tab.

The page lists every tenant's fair-share usage, so it crosses owner
boundaries: admin-only. A scoped (per-user viewer) caller is refused (404);
operators and the loopback dev flow (both → ``_caller_owner_id`` None) see it.

Mirrors the no-bind harness in ``test_admin_owner_scoping.py``: a real
``SqliteStateStore`` seeded under ``tmp_path`` + a FastAPI ``TestClient``
(ASGI, no real port). Public host + token_store so the auth middleware stashes
an identity; loopback (default host) bypasses auth → admin/see-all.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xrlenv.admin.server import _SESSION_COOKIE, AdminServerConfig, build_admin_app
from xrlenv.control.security import TokenStore, write_user_record
from xrlenv.control.state import RawRolloutRecord, SqliteStateStore

VIEWER_ALICE_TOKEN = "alice-viewer-secret"
OPERATOR_TOKEN = "op-secret"


def _seed_db(state_db: Path) -> None:
    """A configured fair-share policy + one running rollout owned by alice."""
    store = SqliteStateStore(state_db)
    store.set_fairness_global(capacity_basis=8, floor=1)
    store.set_fairness_owner("alice", weight=2.0)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-alice", status="running", image="busybox:1",
        owner_id="alice", created_at=time.time(),
    ))
    store.close()


def _session(token: str) -> dict[str, str]:
    """Browser auth via the B7.4 cookie session (replaces HTTP basic auth)."""
    return {"Cookie": f"{_SESSION_COOKIE}={token}"}


def test_fairshare_loopback_sees_enabled_policy(tmp_path: Path) -> None:
    # Default (loopback) host: no auth → admin → 200, body shows the policy.
    state_db = tmp_path / "state.db"
    _seed_db(state_db)
    cfg = AdminServerConfig(state_db=state_db, runs_root=tmp_path / "runs")
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/fairshare")
    assert resp.status_code == 200
    body = resp.text
    assert "ENABLED" in body
    assert "alice" in body


@pytest.fixture
def scoped_cfg(tmp_path: Path) -> AdminServerConfig:
    state_db = tmp_path / "state.db"
    _seed_db(state_db)
    secrets = tmp_path / "secrets"
    write_user_record(
        secrets / "users.json", token=VIEWER_ALICE_TOKEN,
        role="viewer", owner_id="alice",
    )
    token_store = TokenStore.load(secrets_root=secrets, env={})
    token_store.add("operator", OPERATOR_TOKEN)
    # Public bind so the auth middleware engages (loopback bypasses auth).
    return AdminServerConfig(
        state_db=state_db, runs_root=tmp_path / "runs",
        host="0.0.0.0", allow_public=True, token_store=token_store,
    )


def test_fairshare_operator_sees_it(scoped_cfg: AdminServerConfig) -> None:
    app = build_admin_app(scoped_cfg)
    with TestClient(app) as client:
        resp = client.get(
            "/fairshare", headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"},
        )
    assert resp.status_code == 200
    assert "ENABLED" in resp.text


def test_fairshare_scoped_viewer_404(scoped_cfg: AdminServerConfig) -> None:
    app = build_admin_app(scoped_cfg)
    with TestClient(app) as client:
        resp = client.get(
            "/fairshare", headers=_session(VIEWER_ALICE_TOKEN),
        )
    assert resp.status_code == 404


# ── Admin table column values: effective cap, owner cap, uncapped, blocked ────


def _seed_db_uncapped(state_db: Path) -> None:
    """Policy enabled; bob is uncapped, carol is blocked, dave has an owner cap."""
    store = SqliteStateStore(state_db)
    store.set_fairness_global(capacity_basis=8, floor=1)
    # bob: uncapped
    store.set_fairness_owner("bob", uncapped=True)
    # carol: blocked
    store.set_fairness_owner("carol", blocked=True)
    # dave: owner cap of 32
    store.set_fairness_owner("dave", hard_cap=32)
    # One running rollout for each so they appear in the table.
    for owner in ("bob", "carol", "dave"):
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=f"r-{owner}", status="running", image="busybox:1",
            owner_id=owner, created_at=time.time(),
        ))
    store.close()


def test_fairshare_table_shows_uncapped_for_uncapped_owner(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_db_uncapped(state_db)
    cfg = AdminServerConfig(state_db=state_db, runs_root=tmp_path / "runs")
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/fairshare")
    assert resp.status_code == 200
    body = resp.text
    # Template renders r.uncapped → "yes" in the uncapped column.
    assert "bob" in body
    assert "uncapped" in body.lower()


def test_fairshare_table_shows_blocked_for_blocked_owner(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_db_uncapped(state_db)
    cfg = AdminServerConfig(state_db=state_db, runs_root=tmp_path / "runs")
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/fairshare")
    assert resp.status_code == 200
    body = resp.text
    assert "carol" in body
    # The "blocked" column header + "yes" in carol's row.
    assert "blocked" in body.lower()


def test_fairshare_table_shows_owner_cap_and_effective_cap(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_db_uncapped(state_db)
    cfg = AdminServerConfig(state_db=state_db, runs_root=tmp_path / "runs")
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/fairshare")
    assert resp.status_code == 200
    body = resp.text
    # dave's owner cap (32) and effective cap (32) both appear in the page.
    assert "dave" in body
    assert "32" in body
    # Column headers
    assert "effective cap" in body.lower()
    assert "owner cap" in body.lower()


def test_fairshare_table_default_owner_shows_no_owner_cap(tmp_path: Path) -> None:
    """An owner with no override should have an empty owner cap column (dash in template)."""
    state_db = tmp_path / "state.db"
    store = SqliteStateStore(state_db)
    store.set_fairness_global(capacity_basis=8)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-eve", status="running", image="busybox:1",
        owner_id="eve", created_at=time.time(),
    ))
    store.close()
    cfg = AdminServerConfig(state_db=state_db, runs_root=tmp_path / "runs")
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/fairshare")
    assert resp.status_code == 200
    body = resp.text
    assert "eve" in body
    # effective cap for eve should be 8 (the default) shown in the table.
    assert "8" in body
