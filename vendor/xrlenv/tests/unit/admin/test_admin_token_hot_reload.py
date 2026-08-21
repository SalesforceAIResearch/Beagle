"""Regression tests for two hot-reload / thread-safety fixes on TokenStore + admin.

Fix #1 — Admin HTTP path now calls ``store.maybe_reload()`` inside
``_require_role``, BEFORE the ``is_empty`` early-return, so per-user tokens
issued after the admin server starts are picked up without a restart or a
gRPC call.

Fix #2 — ``TokenStore`` is now protected by ``threading.Lock``
(``_reload_lock``).  ``maybe_reload()`` rebuilds the maps via a non-atomic
clear-then-repopulate; without the lock a ``verify()`` on the admin thread
could observe a half-cleared store mid-rebuild on the gRPC thread and
spuriously return ``None`` for a valid token.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from xrlenv.admin.server import _SESSION_COOKIE, AdminServerConfig, build_admin_app
from xrlenv.control.security import TokenStore, write_user_record
from xrlenv.control.state import RawRolloutRecord, SqliteStateStore

# ─── helpers ──────────────────────────────────────────────────────────────────


def _session(token: str) -> dict[str, str]:
    """Browser auth via the B7.4 cookie session. The hot-reload these tests
    pin happens in ``_require_role`` (``store.maybe_reload()``) regardless of
    transport, so the cookie path exercises the same fix the old basic-auth
    header did."""
    return {"Cookie": f"{_SESSION_COOKIE}={token}"}


def _seed_raw_rollout(
    state_db: Path,
    *,
    rollout_id: str,
    displayed_name: str,
    owner_id: str,
) -> None:
    """Insert a single RawRolloutRecord into state_db and close the store."""
    store = SqliteStateStore(state_db)
    store.record_raw_rollout(
        RawRolloutRecord(
            rollout_id=rollout_id,
            status="released",
            image="busybox:1",
            displayed_name=displayed_name,
            owner_id=owner_id,
            created_at=time.time(),
        )
    )
    store.close()


def _force_mtime_forward(path: Path, *, delta: float = 10.0) -> None:
    """Advance ``path``'s mtime by ``delta`` seconds without writing content.

    ``maybe_reload()`` only rebuilds when the mtime *strictly advances*
    relative to the snapshot taken during load.  On coarse-mtime (1-second
    granularity) filesystems the write that happens inside ``write_user_record``
    may land in the same mtime tick as ``TokenStore.load``, causing a spurious
    cache hit.  Bumping the mtime explicitly makes the test deterministic
    without sleeping.
    """
    m = path.stat().st_mtime
    os.utime(path, (m + delta, m + delta))


# ─── Test 1: admin HTTP path hot-reloads a token issued after app build ───────


def test_admin_http_picks_up_token_issued_after_app_built(
    tmp_path: Path,
) -> None:
    """Primary regression for fix #1.

    Timeline:
    1.  App is built with a non-empty TokenStore (one initial ``operator``
        token) so auth is engaged.
    2.  Carol's ``consumer`` token does NOT exist yet → first request returns
        401 (sanity: unknown token is rejected).
    3.  Carol's token is written to users.json at runtime (no restart, no
        gRPC call).
    4.  mtime is bumped forward so ``maybe_reload()`` sees the change.
    5.  The next ``GET /rollouts/raw`` with carol's token must return 200 and
        be owner-scoped to carol.

    Before fix #1, step 5 returned 401 because ``_require_role`` never called
    ``maybe_reload()``, so the admin's snapshot of the TokenStore was frozen at
    app-build time.
    """
    state_db = tmp_path / "state.db"
    _seed_raw_rollout(
        state_db, rollout_id="r-carol", displayed_name="instance-CAROL",
        owner_id="carol",
    )
    _seed_raw_rollout(
        state_db, rollout_id="r-dave", displayed_name="instance-DAVE",
        owner_id="dave",
    )

    secrets = tmp_path / "secrets"
    users_path = secrets / "users.json"

    # Seed ONE initial token so the store is non-empty at load time —
    # an empty store no-ops auth entirely (the loopback dev flow), which
    # would make the test vacuous.
    INITIAL_OP_TOKEN = "initial-operator-tok"
    write_user_record(
        users_path, token=INITIAL_OP_TOKEN, role="operator",
        owner_id="__bootstrap__",
    )

    token_store = TokenStore.load(secrets_root=secrets, env={})
    cfg = AdminServerConfig(
        state_db=state_db,
        runs_root=tmp_path / "runs",
        host="0.0.0.0",
        allow_public=True,
        token_store=token_store,
    )
    app = build_admin_app(cfg)

    CAROL_TOKEN = "carol-runtime-consumer-secret"

    with TestClient(app) as client:
        # Sanity: carol's not-yet-issued token is rejected.
        r_before = client.get(
            "/rollouts/raw", headers=_session(CAROL_TOKEN),
        )
        assert r_before.status_code == 401, (
            f"expected 401 for unknown token, got {r_before.status_code}"
        )

        # Issue carol's token at runtime — no restart, no gRPC call.
        write_user_record(
            users_path, token=CAROL_TOKEN, role="consumer", owner_id="carol",
        )
        # Force the mtime strictly forward so maybe_reload() detects the change.
        # Never use time.sleep — os.utime is deterministic even on coarse-mtime fs.
        _force_mtime_forward(users_path)

        # The admin HTTP request itself must trigger the reload.
        # Do NOT call token_store.maybe_reload() here — that would bypass
        # the fix being tested.
        r_after = client.get(
            "/rollouts/raw", headers=_session(CAROL_TOKEN),
        )
        assert r_after.status_code == 200, (
            f"expected 200 after runtime token issue, got {r_after.status_code}; "
            f"body={r_after.text[:300]}"
        )
        # Owner-scoping still applies: carol sees her own rollout, not dave's.
        assert "instance-CAROL" in r_after.text, (
            "carol's rollout not found in response"
        )
        assert "instance-DAVE" not in r_after.text, (
            "dave's rollout leaked into carol's scoped view"
        )


# ─── Test 2: reload before is_empty — first token at runtime engages auth ─────


def test_empty_store_at_boot_engages_auth_after_first_token_issued(
    tmp_path: Path,
) -> None:
    """Pins fix #1 ordering: ``maybe_reload()`` runs BEFORE the ``is_empty``
    check inside ``_require_role``.

    When the secrets directory is empty at app-build time the store is empty
    and ``is_empty`` is True, which normally triggers the no-auth dev-flow
    (every caller is admin, no scoping).  After the FIRST token is written at
    runtime and the mtime is bumped, the next HTTP request must:
    - call ``maybe_reload()`` (which sees the new users.json and rebuilds),
    - then evaluate ``is_empty`` → False (because the rebuild populated
      ``_by_token_sha``),
    - and therefore engage auth + owner-scoping.

    Without the ``maybe_reload()``-before-``is_empty`` ordering the store
    would stay empty-as-of-boot and every caller would be treated as admin.
    """
    state_db = tmp_path / "state.db"
    _seed_raw_rollout(
        state_db, rollout_id="r-eve", displayed_name="instance-EVE",
        owner_id="eve",
    )
    _seed_raw_rollout(
        state_db, rollout_id="r-other", displayed_name="instance-OTHER",
        owner_id="other-user",
    )

    # Empty secrets dir at load time → is_empty is True at boot.
    secrets = tmp_path / "secrets"
    secrets.mkdir(parents=True, exist_ok=True)
    users_path = secrets / "users.json"

    token_store = TokenStore.load(secrets_root=secrets, env={})
    assert token_store.is_empty, "precondition: store must be empty at boot"

    cfg = AdminServerConfig(
        state_db=state_db,
        runs_root=tmp_path / "runs",
        host="0.0.0.0",
        allow_public=True,
        token_store=token_store,
    )
    app = build_admin_app(cfg)

    EVE_TOKEN = "eve-runtime-consumer-token"

    with TestClient(app) as client:
        # Issue the FIRST token at runtime.
        write_user_record(
            users_path, token=EVE_TOKEN, role="consumer", owner_id="eve",
        )
        _force_mtime_forward(users_path)

        r = client.get("/rollouts/raw", headers=_session(EVE_TOKEN))
        assert r.status_code == 200, (
            f"expected 200 after first runtime token, got {r.status_code}; "
            f"body={r.text[:300]}"
        )
        # Owner-scoping engages: eve sees her rollout but not the other user's.
        assert "instance-EVE" in r.text, "eve's rollout not found"
        assert "instance-OTHER" not in r.text, (
            "other-user's rollout leaked into eve's scoped view; "
            "auth did not engage (store still treated as empty)"
        )


# ─── Test 3: TokenStore thread-safety under concurrent reload + verify ─────────


def test_token_store_verify_never_returns_none_under_concurrent_reload(
    tmp_path: Path,
) -> None:
    """Regression for fix #2: ``_reload_lock`` prevents torn reads.

    Spawns a background thread that loops, bumping the mtime of users.json
    and calling ``maybe_reload()`` to force the clear-then-repopulate cycle.
    The main thread concurrently calls ``store.verify(known_token)`` and
    asserts it NEVER returns ``None`` and always yields the expected identity.

    Before fix #2, the non-atomic clear-then-repopulate inside
    ``maybe_reload()`` could leave ``_by_token_sha`` momentarily empty
    between the ``.clear()`` and the subsequent ``register_user()`` calls,
    causing a concurrent ``verify()`` to spuriously return ``None``.
    """
    secrets = tmp_path / "secrets"
    users_path = secrets / "users.json"

    KNOWN_TOKEN = "known-consumer-token-for-thread-test"
    KNOWN_OWNER = "thread-test-user"

    write_user_record(
        users_path, token=KNOWN_TOKEN, role="consumer", owner_id=KNOWN_OWNER,
    )

    store = TokenStore.load(secrets_root=secrets, env={})

    # Confirm the token resolves correctly before the concurrent stress.
    initial_identity = store.verify(KNOWN_TOKEN)
    assert initial_identity is not None, "precondition: token must verify before stress"
    assert initial_identity.owner_id == KNOWN_OWNER
    assert initial_identity.role == "consumer"

    N_ITERATIONS = 300
    background_errors: list[Exception] = []
    ready = threading.Event()

    def _reload_loop() -> None:
        """Repeatedly force mtime forward and call maybe_reload()."""
        ready.wait()
        try:
            m = users_path.stat().st_mtime
            for _i in range(N_ITERATIONS):
                # Each iteration bumps the mtime by one second, guaranteeing
                # a change is detected on every maybe_reload() call.
                m += 1.0
                os.utime(users_path, (m, m))
                store.maybe_reload()
        except Exception as exc:
            background_errors.append(exc)

    thread = threading.Thread(target=_reload_loop, daemon=True)
    thread.start()

    # Signal both loops to overlap.
    ready.set()

    verify_failures: list[int] = []
    wrong_identity: list[int] = []
    for i in range(N_ITERATIONS):
        identity = store.verify(KNOWN_TOKEN)
        if identity is None:
            # This is the torn-read failure: the store was caught mid-clear.
            # Before fix #2 this would intermittently occur.
            verify_failures.append(i)
        elif identity.owner_id != KNOWN_OWNER or identity.role != "consumer":
            wrong_identity.append(i)

    thread.join(timeout=10.0)
    assert not thread.is_alive(), "reload thread did not finish in time"

    assert not background_errors, (
        f"background reload thread raised: {background_errors}"
    )
    assert not verify_failures, (
        f"store.verify() returned None on {len(verify_failures)} of {N_ITERATIONS} "
        "iterations — torn read detected (lock missing or not covering the "
        f"right section). First failure at iteration {verify_failures[0]}."
    )
    assert not wrong_identity, (
        f"store.verify() returned wrong identity on {len(wrong_identity)} iterations; "
        f"first at iteration {wrong_identity[0]}"
    )
