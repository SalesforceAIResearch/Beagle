"""Tests for the Slice 8 security surface (spec 19).

Covers:
- ``TokenStore`` loading from disk (mode 0600 enforced) + env vars
- ``TokenStore.add`` / ``verify`` / ``digest_hint``
- ``required_scope_for_method`` + ``scope_satisfies``
- ``write_secret_file`` mode + atomic semantics
- Audit table parity (InMemoryStateStore + SqliteStateStore)
- ``cmd_tokens_issue`` CLI: writes mode-0600 file + refuses overwrite
- gRPC interceptor accept / reject paths (unit-level, no real server)
- ``TemplateCatalog`` mount-allowlist denial + image digest pin
"""

from __future__ import annotations

import io
import stat
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import MountSpec, ResourceSpec
from xrlenv.cli.commands import (
    cmd_tokens_issue,
    cmd_tokens_list,
    cmd_tokens_revoke,
    cmd_tokens_rotate,
)
from xrlenv.control import security as _security
from xrlenv.control.auth_interceptor import (
    BearerScopeInterceptor,
    _bearer_from_metadata,
)
from xrlenv.control.security import (
    ROLE_DEFAULT_SCOPE,
    ROLE_TOKEN_PREFIX,
    TokenStore,
    append_revocation,
    generate_token,
    required_scope_for_method,
    scope_satisfies,
    token_digest_hint,
    token_full_id,
    token_sha256,
    write_grace_record,
    write_secret_file,
    write_user_record,
)
from xrlenv.control.state import (
    AuditRecord,
    InMemoryStateStore,
    SqliteStateStore,
)
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    MountDenied,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)

# ──────────────────────────────────────────────────────────────────────────────
# token_digest_hint + scope helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_token_digest_hint_is_first_6_hex() -> None:
    h = token_digest_hint("hello-world")
    assert len(h) == 6
    assert all(c in "0123456789abcdef" for c in h)
    assert h == token_digest_hint("hello-world")  # stable


def test_required_scope_for_method_known_and_unknown() -> None:
    # Bidi stream is the one phase-0 gRPC method.
    assert required_scope_for_method(
        "/xrlenv.node_control.v1.NodeControl/NodeControlStream"
    ) == "node.report"
    # Unknown methods (future consumer/operator RPCs) return None until added.
    assert required_scope_for_method("/some.other.Service/Foo") is None


# ──────────────────────────────────────────────────────────────────────────────
# audit M4 — raw-container + plan RPCs carry the right required scope.
#
# Before M4 these methods were absent from the scope map, so
# ``required_scope_for_method`` returned None and the interceptor let *any*
# valid role (viewer / node / operator) through — a viewer token could acquire
# and drive a raw container. The map now pins each raw session RPC to
# ``consumer.rollout`` and the cluster-planning RPC to ``operator.admin``.
# ──────────────────────────────────────────────────────────────────────────────

_RC = "/xrlenv.rollout_control.v1.RolloutControl"


@pytest.mark.parametrize(
    "method",
    [
        "AcquireContainer",
        "ContainerExec",
        "ContainerExecStream",
        "DestroyContainer",
        "ContainerPutArchive",
        "ContainerGetArchive",
        "QueueStatus",
    ],
)
def test_raw_container_rpcs_require_consumer_rollout(method: str) -> None:
    assert required_scope_for_method(f"{_RC}/{method}") == "consumer.rollout"


def test_plan_image_distribution_requires_operator_admin() -> None:
    assert (
        required_scope_for_method(f"{_RC}/PlanImageDistribution")
        == "operator.admin"
    )


def test_start_rollout_stays_consumer_rollout() -> None:
    # Sanity: the M4 additions didn't perturb the canonical consumer RPC.
    assert required_scope_for_method(f"{_RC}/StartRollout") == "consumer.rollout"


def test_scope_satisfies_is_flat() -> None:
    """Phase 0: roles do not imply each other. operator.admin cannot
    impersonate node or consumer (spec 19 threat model)."""
    assert scope_satisfies("node.report", "node.report")
    assert not scope_satisfies("operator.admin", "node.report")
    assert not scope_satisfies("consumer.rollout", "operator.admin")


# ──────────────────────────────────────────────────────────────────────────────
# TokenStore
# ──────────────────────────────────────────────────────────────────────────────


def test_token_store_add_and_verify_roundtrip() -> None:
    s = TokenStore()
    identity = s.add("node", "node-secret")
    assert identity.role == "node"
    assert identity.scope == "node.report"
    assert s.verify("node-secret") is identity
    assert s.verify("wrong") is None
    assert s.verify(None) is None


def test_token_store_add_rejects_empty_token() -> None:
    s = TokenStore()
    with pytest.raises(ValueError, match="empty token"):
        s.add("node", "")
    with pytest.raises(ValueError, match="empty token"):
        s.add("node", "   ")


def test_token_store_replaces_existing_role_token() -> None:
    s = TokenStore()
    s.add("consumer", "old")
    s.add("consumer", "new")
    assert s.verify("old") is None
    assert s.verify("new") is not None


def test_token_store_load_from_env_var(tmp_path: Path) -> None:
    s = TokenStore.load(
        secrets_root=tmp_path,
        env={"XRLENV_NODE_TOKEN": "from-env", "XRLENV_CONSUMER_TOKEN": "trn"},
    )
    assert s.verify("from-env").role == "node"
    assert s.verify("trn").role == "consumer"
    assert s.verify("missing") is None


def test_token_store_load_from_secret_file_with_mode_0600(tmp_path: Path) -> None:
    secret = tmp_path / "operator.token"
    secret.write_text("op-secret")
    secret.chmod(0o600)
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("op-secret").role == "operator"


def test_token_store_skips_loose_permission_secret_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    secret = tmp_path / "node.token"
    secret.write_text("oops-public")
    secret.chmod(0o644)  # loose perms
    with caplog.at_level("WARNING"):
        s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.is_empty
    assert any("0600" in rec.message for rec in caplog.records)


def test_token_store_env_var_overrides_secret_file(tmp_path: Path) -> None:
    secret = tmp_path / "node.token"
    secret.write_text("from-disk")
    secret.chmod(0o600)
    s = TokenStore.load(
        secrets_root=tmp_path, env={"XRLENV_NODE_TOKEN": "from-env"},
    )
    assert s.verify("from-env") is not None
    assert s.verify("from-disk") is None


def test_known_roles_default_scopes() -> None:
    # Sanity: every role we expose has a default scope.
    assert set(ROLE_DEFAULT_SCOPE) == {"node", "consumer", "operator", "viewer"}


# ──────────────────────────────────────────────────────────────────────────────
# write_secret_file
# ──────────────────────────────────────────────────────────────────────────────


def test_write_secret_file_creates_mode_0600(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "ops.token"
    write_secret_file(target, "the-token-bytes")
    assert target.read_text() == "the-token-bytes"
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600
    # Parent created with 0700.
    parent_mode = stat.S_IMODE(target.parent.stat().st_mode)
    assert parent_mode == 0o700


def test_write_secret_file_replaces_atomically(tmp_path: Path) -> None:
    target = tmp_path / "ops.token"
    write_secret_file(target, "first")
    write_secret_file(target, "second")
    assert target.read_text() == "second"


# ──────────────────────────────────────────────────────────────────────────────
# Audit table (StateStore parity)
# ──────────────────────────────────────────────────────────────────────────────


def _audit_parity(store: Any) -> None:
    a = store.append_audit(
        "auth.token_used", role="node", digest_hint="abc123",
        method="/svc/Method", source="127.0.0.1", result="ok",
        payload={"k": "v"},
    )
    b = store.append_audit("admin.action", role="operator", payload={"action": "drain"})
    rows = list(store.audit_since(0))
    assert [r.kind for r in rows] == ["auth.token_used", "admin.action"]
    assert isinstance(rows[0], AuditRecord)
    assert rows[0].role == "node"
    assert rows[0].digest_hint == "abc123"
    assert rows[0].payload == {"k": "v"}
    assert b.seq == a.seq + 1
    # Filter by seq.
    later = list(store.audit_since(a.seq))
    assert [r.kind for r in later] == ["admin.action"]


def test_audit_in_memory_store_parity() -> None:
    _audit_parity(InMemoryStateStore())


def test_audit_sqlite_store_parity(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "state.db")
    try:
        _audit_parity(store)
    finally:
        store.close()


# ──────────────────────────────────────────────────────────────────────────────
# cmd_tokens_issue
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_tokens_issue_writes_mode_0600_and_prints_token(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = cmd_tokens_issue("node", secrets_root=tmp_path, out=out)
    assert rc == 0
    target = tmp_path / "node.token"
    assert target.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    body = out.getvalue()
    assert "issued node token" in body
    assert "scope:     node.report" in body
    assert "raw token:" in body
    # The printed token matches what was stored.
    on_disk = target.read_text()
    assert on_disk in body


def test_cmd_tokens_issue_refuses_overwrite(tmp_path: Path) -> None:
    out = io.StringIO()
    cmd_tokens_issue("consumer", secrets_root=tmp_path, out=out)
    out2 = io.StringIO()
    rc = cmd_tokens_issue("consumer", secrets_root=tmp_path, out=out2)
    assert rc == 1
    assert "already exists" in out2.getvalue()


def test_cmd_tokens_issue_rejects_unknown_role(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = cmd_tokens_issue("badrole", secrets_root=tmp_path, out=out)
    assert rc == 2
    assert "unknown role" in out.getvalue()


def test_cmd_tokens_issue_shared_consumer_warns_recommend_owner(
    tmp_path: Path,
) -> None:
    """A no-``--owner`` consumer token still issues (warn-only), but the output
    recommends ``--owner`` and explains it is not an admin credential."""
    out = io.StringIO()
    rc = cmd_tokens_issue("consumer", secrets_root=tmp_path, out=out)
    assert rc == 0
    assert (tmp_path / "consumer.token").exists()
    body = out.getvalue()
    # The token still prints (non-breaking) ...
    assert "issued consumer token" in body
    # ... plus the nudge toward per-user tokens, with the reason it bit us:
    # a shared token is owner_id="default", not full admin.
    assert "consumer --owner" in body
    assert 'owner_id="default"' in body
    assert "operator token" in body


def test_cmd_tokens_issue_per_owner_consumer_has_no_shared_warning(
    tmp_path: Path,
) -> None:
    """The recommend-``--owner`` nudge is only for the shared path — a token
    minted *with* ``--owner`` must not carry it."""
    out = io.StringIO()
    rc = cmd_tokens_issue(
        "consumer", owner="alice", secrets_root=tmp_path, out=out,
    )
    assert rc == 0
    assert "consumer --owner <id>" not in out.getvalue()


def test_cmd_tokens_issue_shared_operator_has_no_consumer_warning(
    tmp_path: Path,
) -> None:
    """Only ``consumer`` gets the nudge; operator/node/viewer are unchanged."""
    out = io.StringIO()
    rc = cmd_tokens_issue("operator", secrets_root=tmp_path, out=out)
    assert rc == 0
    assert "consumer --owner <id>" not in out.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# gRPC interceptor (unit-level)
# ──────────────────────────────────────────────────────────────────────────────


def _details(method: str, metadata: list[tuple[str, str]]) -> Any:
    """Build a minimal HandlerCallDetails-like object."""
    obj = MagicMock()
    obj.method = method
    obj.invocation_metadata = metadata
    return obj


async def test_interceptor_no_op_when_store_empty() -> None:
    store = TokenStore()
    intercept = BearerScopeInterceptor(store=store)
    handler = MagicMock()

    async def _continuation(d: Any) -> Any:
        return handler

    result = await intercept.intercept_service(
        _continuation,
        _details("/xrlenv.node_control.v1.NodeControl/NodeControlStream", []),
    )
    assert result is handler


async def test_interceptor_accepts_valid_node_token() -> None:
    store = TokenStore()
    store.add("node", "good-token")
    state = InMemoryStateStore()
    # audit_success=True: explicitly opt into per-RPC success auditing so this
    # test keeps exercising the auth.token_used write path. The default changed
    # to False (XRLENV_AUDIT_AUTH_SUCCESS=off) to prevent WAL runaway on prod —
    # see BearerScopeInterceptor.__init__ for rationale.
    intercept = BearerScopeInterceptor(store=store, state=state, audit_success=True)
    handler = MagicMock()

    async def _continuation(d: Any) -> Any:
        return handler

    result = await intercept.intercept_service(
        _continuation,
        _details(
            "/xrlenv.node_control.v1.NodeControl/NodeControlStream",
            [("authorization", "Bearer good-token")],
        ),
    )
    assert result is handler
    rows = list(state.audit_since(0))
    assert [r.kind for r in rows] == ["auth.token_used"]
    assert rows[0].role == "node"
    assert rows[0].result == "ok"


async def test_interceptor_rejects_unknown_token() -> None:
    store = TokenStore()
    store.add("node", "good-token")
    state = InMemoryStateStore()
    intercept = BearerScopeInterceptor(store=store, state=state)

    async def _continuation(d: Any) -> Any:
        return MagicMock()

    handler = await intercept.intercept_service(
        _continuation,
        _details(
            "/xrlenv.node_control.v1.NodeControl/NodeControlStream",
            [("authorization", "Bearer bad")],
        ),
    )
    # Returned an abort handler, not the real one. Calling its slot raises.
    assert handler is not None
    audit = list(state.audit_since(0))
    assert [r.kind for r in audit] == ["auth.denied"]
    assert audit[0].result == "bad_token"


async def test_interceptor_rejects_wrong_scope() -> None:
    """An operator token cannot satisfy the node-only scope on the bidi stream."""
    store = TokenStore()
    store.add("operator", "op-tok")
    state = InMemoryStateStore()
    intercept = BearerScopeInterceptor(store=store, state=state)

    async def _continuation(d: Any) -> Any:
        return MagicMock()

    await intercept.intercept_service(
        _continuation,
        _details(
            "/xrlenv.node_control.v1.NodeControl/NodeControlStream",
            [("authorization", "Bearer op-tok")],
        ),
    )
    audit = list(state.audit_since(0))
    assert [r.kind for r in audit] == ["auth.denied"]
    assert audit[0].result == "wrong_scope"
    assert audit[0].payload["required"] == "node.report"
    assert audit[0].payload["have"] == "operator.admin"


def test_bearer_from_metadata_parses_bearer_prefix_case_insensitive() -> None:
    assert _bearer_from_metadata([("Authorization", "Bearer abc")]) == "abc"
    assert _bearer_from_metadata([("authorization", "Bearer xyz")]) == "xyz"
    # Tolerates bare token.
    assert _bearer_from_metadata([("authorization", "raw-token")]) == "raw-token"
    assert _bearer_from_metadata([]) is None
    assert _bearer_from_metadata(None) is None


# ──────────────────────────────────────────────────────────────────────────────
# TemplateCatalog: mount allowlist + image digest pin
# ──────────────────────────────────────────────────────────────────────────────


def _manifest(
    *,
    image: str = "im/t:1",
    mounts: tuple[MountSpec, ...] = (),
) -> TemplateManifest:
    return TemplateManifest(
        name="t", version="0.1", digest="sha256:t", image=image,
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000, mounts=mounts,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


@pytest.mark.parametrize(
    "denied_path",
    ["/etc", "/etc/passwd", "/root", "/home/user", "/proc/1",
     "/sys/kernel", "/var/run/docker.sock", "/dev", "/dev/sda"],
)
def test_register_rejects_denied_mount_paths(denied_path: str) -> None:
    catalog = TemplateCatalog()
    m = _manifest(mounts=(MountSpec(host_path=denied_path, sandbox_path="/x"),))
    with pytest.raises(MountDenied, match=r"denied prefix|controlled set"):
        catalog.register(m)


@pytest.mark.parametrize("allowed_path", ["/dev/null", "/dev/zero", "/dev/random"])
def test_register_allows_controlled_dev_paths(allowed_path: str) -> None:
    catalog = TemplateCatalog()
    m = _manifest(
        mounts=(MountSpec(host_path=allowed_path, sandbox_path="/x"),),
    )
    catalog.register(m)
    assert "t" in {m.name for m in catalog.list()}


def test_register_emits_mount_denied_audit_event() -> None:
    audit_calls: list[tuple[str, dict[str, Any]]] = []
    catalog = TemplateCatalog(
        audit_callback=lambda kind, payload: audit_calls.append((kind, payload)),
    )
    m = _manifest(mounts=(MountSpec(host_path="/etc", sandbox_path="/x"),))
    with pytest.raises(MountDenied):
        catalog.register(m)
    assert audit_calls and audit_calls[0][0] == "mount.denied"
    assert audit_calls[0][1]["host_path"] == "/etc"


def test_register_pins_image_via_digest_resolver() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []

    def resolver(image_ref: str) -> str | None:
        return "sha256:abcd" + "1" * 60 if image_ref == "im/t:1" else None

    catalog = TemplateCatalog(
        digest_resolver=resolver,
        audit_callback=lambda k, p: captured.append((k, p)),
    )
    catalog.register(_manifest(image="im/t:1"))
    pinned = catalog.get("t")
    assert pinned.image.startswith("im/t@sha256:")
    # Audit shows the pinned form on template.registered.
    reg_event = next(p for k, p in captured if k == "template.registered")
    assert reg_event["pinned_by_digest"] is True


def test_register_warns_when_resolver_returns_none() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []
    catalog = TemplateCatalog(
        digest_resolver=lambda _ref: None,
        audit_callback=lambda k, p: captured.append((k, p)),
    )
    catalog.register(_manifest(image="im/t:latest"))
    # Image stays unpinned; audit records the warning.
    assert catalog.get("t").image == "im/t:latest"
    assert any(k == "template.image_unpinned" for k, _ in captured)


def test_register_skips_pinning_when_already_pinned() -> None:
    """A digest-pinned image is left as-is, no resolver call."""
    resolver = MagicMock(return_value="sha256:" + "0" * 64)
    catalog = TemplateCatalog(digest_resolver=resolver)
    pinned_img = "im/t@sha256:" + "1" * 64
    catalog.register(_manifest(image=pinned_img))
    assert catalog.get("t").image == pinned_img
    resolver.assert_not_called()


def test_register_handles_resolver_exception_gracefully() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []

    def boom(_ref: str) -> str:
        raise RuntimeError("docker daemon unreachable")

    catalog = TemplateCatalog(
        digest_resolver=boom,
        audit_callback=lambda k, p: captured.append((k, p)),
    )
    catalog.register(_manifest(image="im/t:1"))
    # Falls back to unpinned + audit row.
    assert catalog.get("t").image == "im/t:1"
    assert any(p.get("reason") == "resolver_error" for _, p in captured)


# ──────────────────────────────────────────────────────────────────────────────
# Hot-reload — `xrlenv tokens issue ...` while the control plane is already
# running must take effect on the next RPC, not require a restart.
# ──────────────────────────────────────────────────────────────────────────────


def test_maybe_reload_picks_up_newly_written_token(tmp_path: Path) -> None:
    """Issue a consumer token *after* TokenStore.load() and prove the
    next maybe_reload() call surfaces it. Regression for the live
    smoke-test failure where Connect mode rejected a freshly issued
    token because the store hadn't reloaded.
    """
    # Empty secrets dir at load time.
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.is_empty
    assert not s.maybe_reload()  # no files → no-op

    # Now write a consumer token (mode 0o600, the on-disk format).
    secret = tmp_path / "consumer.token"
    secret.write_text("freshly-issued")
    secret.chmod(0o600)

    assert s.maybe_reload() is True
    identity = s.verify("freshly-issued")
    assert identity is not None
    assert identity.role == "consumer"


def test_maybe_reload_picks_up_token_rotation(tmp_path: Path) -> None:
    """Replacing an existing token file (operator rotation) is picked
    up by the next maybe_reload(); the old token stops verifying."""
    secret = tmp_path / "consumer.token"
    secret.write_text("old-token")
    secret.chmod(0o600)
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("old-token") is not None

    # Bump mtime to ensure detection on systems with coarse mtime.
    import os
    import time
    new_ts = secret.stat().st_mtime + 5.0
    secret.write_text("rotated-token")
    secret.chmod(0o600)
    os.utime(secret, (new_ts, new_ts))

    assert s.maybe_reload() is True
    assert s.verify("old-token") is None
    assert s.verify("rotated-token").role == "consumer"
    _ = time  # silence "imported but unused" if the OS skips utime


def test_maybe_reload_is_noop_when_nothing_changed(tmp_path: Path) -> None:
    secret = tmp_path / "consumer.token"
    secret.write_text("stable")
    secret.chmod(0o600)
    s = TokenStore.load(secrets_root=tmp_path, env={})

    # Two reloads in a row, no fs changes.
    assert s.maybe_reload() is False
    assert s.maybe_reload() is False
    assert s.verify("stable").role == "consumer"


def test_maybe_reload_silent_on_test_fakes(tmp_path: Path) -> None:
    """A store built without ``load(...)`` (test fakes that just call
    ``add()``) has no secrets dir to watch and ``maybe_reload()`` is a
    no-op — never raises."""
    s = TokenStore()
    s.add("consumer", "fake")
    assert s.maybe_reload() is False
    assert s.verify("fake") is not None


# ──────────────────────────────────────────────────────────────────────────────
# B5.2 — token_full_id + TokenStore.rotate / revoke
# ──────────────────────────────────────────────────────────────────────────────


def test_token_full_id_is_12_hex_and_extends_digest_hint() -> None:
    """``token_id`` is the 12-char extension of the 6-char digest_hint,
    so an operator quoting a 6-char hint from a log line still resolves
    via revoke-by-prefix."""
    tid = token_full_id("hello-world")
    assert len(tid) == 12
    assert all(c in "0123456789abcdef" for c in tid)
    assert tid.startswith(token_digest_hint("hello-world"))


def test_rotate_immediate_cutover_drops_prior_token() -> None:
    """``grace_s=0`` (default): the old token is invalidated synchronously."""
    s = TokenStore()
    s.add("operator", "old-op")
    s.rotate("operator", "new-op")
    assert s.verify("old-op") is None
    new_identity = s.verify("new-op")
    assert new_identity is not None
    assert new_identity.role == "operator"
    assert new_identity.token_id == token_full_id("new-op")


def test_rotate_with_grace_keeps_old_alive_until_window_elapses() -> None:
    """Both old and new verify during the grace window; the old one
    flips to ``None`` once wall-clock crosses the expiry."""
    import time

    s = TokenStore()
    s.add("consumer", "old-c")
    s.rotate("consumer", "new-c", grace_s=60.0)
    assert s.verify("old-c") is not None
    assert s.verify("new-c") is not None

    # Force the grace expiry into the past — verify() must reject the
    # old token and evict it from the in-memory map.
    s._grace_expires["old-c"] = time.time() - 1
    assert s.verify("old-c") is None
    assert s.verify("new-c") is not None  # new token still good.


def test_rotate_rejects_empty_token() -> None:
    s = TokenStore()
    s.add("node", "n1")
    with pytest.raises(ValueError, match="empty token"):
        s.rotate("node", "")


def test_revoke_by_full_id_blocks_subsequent_verify() -> None:
    s = TokenStore()
    s.add("operator", "op-tok")
    matched = s.revoke(token_full_id("op-tok"))
    assert matched == token_full_id("op-tok")
    assert s.verify("op-tok") is None


def test_revoke_by_short_prefix_resolves_when_unique() -> None:
    s = TokenStore()
    s.add("node", "n-tok")
    s.add("consumer", "c-tok")
    # Use the 6-char digest_hint as the prefix.
    matched = s.revoke(token_digest_hint("n-tok"))
    assert matched == token_full_id("n-tok")
    assert s.verify("n-tok") is None
    assert s.verify("c-tok") is not None


def test_revoke_refuses_prefix_under_six_chars() -> None:
    s = TokenStore()
    s.add("node", "n-tok")
    with pytest.raises(ValueError, match="at least 6 hex chars"):
        s.revoke(token_full_id("n-tok")[:5])


def test_revoke_unknown_id_raises_lookup_error() -> None:
    s = TokenStore()
    s.add("node", "n-tok")
    with pytest.raises(LookupError, match="no known token matches"):
        s.revoke("ffffffffffff")


# ──────────────────────────────────────────────────────────────────────────────
# B5.2 — on-disk integration: write_grace_record + append_revocation + reload
# ──────────────────────────────────────────────────────────────────────────────


def test_load_picks_up_grace_token_within_window(tmp_path: Path) -> None:
    """``rotate --grace`` writes a ``<role>.token.previous.json``
    sidecar; ``TokenStore.load`` accepts both old and new tokens
    until the grace expiry."""
    import datetime as _dt
    active = tmp_path / "consumer.token"
    write_secret_file(active, "new-c")
    previous = tmp_path / "consumer.token.previous.json"
    write_grace_record(
        previous, "old-c",
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=120),
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("new-c").role == "consumer"
    assert s.verify("old-c").role == "consumer"


def test_load_ignores_expired_grace_sidecar(tmp_path: Path) -> None:
    """A previous-token sidecar whose ``grace_until`` is in the past
    is ignored — ``verify`` rejects the token."""
    import datetime as _dt
    active = tmp_path / "consumer.token"
    write_secret_file(active, "new-c")
    previous = tmp_path / "consumer.token.previous.json"
    write_grace_record(
        previous, "old-c",
        _dt.datetime.now(_dt.UTC) - _dt.timedelta(seconds=10),
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("old-c") is None
    assert s.verify("new-c").role == "consumer"


def test_load_honors_revoked_json(tmp_path: Path) -> None:
    """Tokens listed in ``revoked.json`` are refused even if they
    appear in the active token file."""
    active = tmp_path / "node.token"
    write_secret_file(active, "n-tok")
    append_revocation(tmp_path / "revoked.json", token_full_id("n-tok"))
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("n-tok") is None


def test_append_revocation_is_idempotent(tmp_path: Path) -> None:
    """Re-revoking the same token_id leaves the file unchanged
    (one entry, not duplicated)."""
    import json
    revoked = tmp_path / "revoked.json"
    append_revocation(revoked, "abc123abc123")
    append_revocation(revoked, "abc123abc123")
    entries = json.loads(revoked.read_text())
    assert len(entries) == 1
    assert entries[0]["token_id"] == "abc123abc123"


def test_maybe_reload_picks_up_revoked_json(tmp_path: Path) -> None:
    """Append to ``revoked.json`` while the store is already loaded;
    the next ``maybe_reload`` rejects the matching token."""
    import os
    import time
    active = tmp_path / "node.token"
    write_secret_file(active, "n-tok")
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("n-tok") is not None

    revoked = tmp_path / "revoked.json"
    append_revocation(revoked, token_full_id("n-tok"))
    # Bump mtime on coarse-grained filesystems so the watcher detects the change.
    new_ts = revoked.stat().st_mtime + 5.0
    os.utime(revoked, (new_ts, new_ts))
    assert s.maybe_reload() is True
    assert s.verify("n-tok") is None
    _ = time  # silence unused-import if utime is a no-op on the host fs.


# ──────────────────────────────────────────────────────────────────────────────
# B5.2 — CLI: tokens rotate / revoke / list
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_tokens_rotate_immediate_cutover(tmp_path: Path) -> None:
    out = io.StringIO()
    cmd_tokens_issue("operator", secrets_root=tmp_path, out=out)
    original = (tmp_path / "operator.token").read_text().strip()

    out2 = io.StringIO()
    rc = cmd_tokens_rotate("operator", secrets_root=tmp_path, out=out2)
    assert rc == 0
    new = (tmp_path / "operator.token").read_text().strip()
    assert new != original
    body = out2.getvalue()
    assert "rotated operator token" in body
    assert "invalidated immediately" in body
    # No grace sidecar should be present after an immediate-cutover rotate.
    assert not (tmp_path / "operator.token.previous.json").exists()


def test_cmd_tokens_rotate_with_grace_writes_sidecar(tmp_path: Path) -> None:
    import json
    out = io.StringIO()
    cmd_tokens_issue("consumer", secrets_root=tmp_path, out=out)
    original = (tmp_path / "consumer.token").read_text().strip()

    out2 = io.StringIO()
    rc = cmd_tokens_rotate(
        "consumer", grace="24h", secrets_root=tmp_path, out=out2,
    )
    assert rc == 0
    sidecar = tmp_path / "consumer.token.previous.json"
    assert sidecar.exists()
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    payload = json.loads(sidecar.read_text())
    assert payload["token"] == original
    assert "grace_until" in payload
    assert "prior token kept valid" in out2.getvalue()


def test_cmd_tokens_rotate_refuses_if_no_existing_token(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = cmd_tokens_rotate("node", secrets_root=tmp_path, out=out)
    assert rc == 1
    assert "does not exist" in out.getvalue()


def test_cmd_tokens_rotate_rejects_negative_grace(tmp_path: Path) -> None:
    out = io.StringIO()
    cmd_tokens_issue("node", secrets_root=tmp_path, out=out)
    out2 = io.StringIO()
    rc = cmd_tokens_rotate(
        "node", grace="-1", secrets_root=tmp_path, out=out2,
    )
    assert rc == 2
    assert "non-negative" in out2.getvalue()


def test_cmd_tokens_rotate_shared_consumer_warns_recommend_owner(
    tmp_path: Path,
) -> None:
    """Rotate rewrites the shared ``consumer.token`` (owner_id="default"), so it
    carries the same recommend-``--owner`` nudge as ``tokens issue``."""
    out = io.StringIO()
    cmd_tokens_issue("consumer", secrets_root=tmp_path, out=out)
    out2 = io.StringIO()
    rc = cmd_tokens_rotate("consumer", secrets_root=tmp_path, out=out2)
    assert rc == 0
    body = out2.getvalue()
    assert "rotated consumer token" in body
    assert "consumer --owner <id>" in body
    assert 'owner_id="default"' in body


def test_cmd_tokens_rotate_operator_has_no_consumer_warning(
    tmp_path: Path,
) -> None:
    """Only the consumer role gets the nudge — rotating operator must not."""
    out = io.StringIO()
    cmd_tokens_issue("operator", secrets_root=tmp_path, out=out)
    out2 = io.StringIO()
    rc = cmd_tokens_rotate("operator", secrets_root=tmp_path, out=out2)
    assert rc == 0
    assert "consumer --owner <id>" not in out2.getvalue()


def test_cmd_tokens_revoke_appends_and_blocks_verify(tmp_path: Path) -> None:
    out = io.StringIO()
    cmd_tokens_issue("operator", secrets_root=tmp_path, out=out)
    token = (tmp_path / "operator.token").read_text().strip()
    token_id = token_full_id(token)

    out2 = io.StringIO()
    rc = cmd_tokens_revoke(token_id, secrets_root=tmp_path, out=out2)
    assert rc == 0
    assert f"revoked token_id={token_id}" in out2.getvalue()
    # Re-loading the store now refuses the still-on-disk active token.
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify(token) is None


def test_cmd_tokens_revoke_rejects_unknown_prefix(tmp_path: Path) -> None:
    out = io.StringIO()
    cmd_tokens_issue("node", secrets_root=tmp_path, out=out)
    out2 = io.StringIO()
    rc = cmd_tokens_revoke("ffffff", secrets_root=tmp_path, out=out2)
    assert rc == 1
    assert "no known token matches" in out2.getvalue()


def test_cmd_tokens_revoke_rejects_short_prefix(tmp_path: Path) -> None:
    out = io.StringIO()
    cmd_tokens_issue("node", secrets_root=tmp_path, out=out)
    out2 = io.StringIO()
    rc = cmd_tokens_revoke("abc", secrets_root=tmp_path, out=out2)
    assert rc == 2
    assert "at least 6 hex chars" in out2.getvalue()


def test_cmd_tokens_list_shows_active_grace_and_revoked(tmp_path: Path) -> None:
    out = io.StringIO()
    cmd_tokens_issue("consumer", secrets_root=tmp_path, out=out)
    cmd_tokens_rotate(
        "consumer", grace="1h", secrets_root=tmp_path, out=io.StringIO(),
    )
    # Issue another role so the operator can see the per-role breakout.
    cmd_tokens_issue("node", secrets_root=tmp_path, out=io.StringIO())
    node_token = (tmp_path / "node.token").read_text().strip()
    cmd_tokens_revoke(
        token_full_id(node_token), secrets_root=tmp_path, out=io.StringIO(),
    )

    out2 = io.StringIO()
    rc = cmd_tokens_list(secrets_root=tmp_path, out=out2)
    assert rc == 0
    body = out2.getvalue()
    assert "consumer  active" in body
    assert "consumer  grace" in body
    assert "node      active" in body  # role left-padded to 9 chars.
    assert "revoked token_ids" in body
    # Token bytes are never printed.
    assert (tmp_path / "consumer.token").read_text().strip() not in body
    assert node_token not in body


def test_cmd_tokens_list_empty_dir_returns_zero(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = cmd_tokens_list(secrets_root=tmp_path, out=out)
    assert rc == 0
    assert "no tokens loaded" in out.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# B5.2 — interceptor-level end-to-end: CLI rotate → next RPC with the OLD
# token is denied. This is the unit form of the slice's smoke ("re-issue an
# operator token, confirm the old one's API calls 401") — no real gRPC server,
# but the full TokenStore → maybe_reload → BearerScopeInterceptor path.
# ──────────────────────────────────────────────────────────────────────────────


async def test_interceptor_denies_old_token_after_cli_rotate_no_grace(
    tmp_path: Path,
) -> None:
    """The operator's rotate-then-401 expectation in concrete form."""
    import os

    # Issue a consumer token then bring up the store the way the control
    # plane does (load + hot-reload from disk).
    cmd_tokens_issue("consumer", secrets_root=tmp_path, out=io.StringIO())
    original = (tmp_path / "consumer.token").read_text().strip()
    store = TokenStore.load(secrets_root=tmp_path, env={})
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state)
    handler = MagicMock()

    async def _continuation(d: Any) -> Any:
        return handler

    # Old token works pre-rotation.
    result = await interceptor.intercept_service(
        _continuation,
        _details(
            "/xrlenv.rollout_control.v1.RolloutControl/Heartbeat",
            [("authorization", f"Bearer {original}")],
        ),
    )
    assert result is handler

    # Rotate with immediate cutover; bump mtime so the watcher detects
    # the change on coarse-mtime filesystems.
    cmd_tokens_rotate(
        "consumer", secrets_root=tmp_path, out=io.StringIO(),
    )
    secret = tmp_path / "consumer.token"
    new_ts = secret.stat().st_mtime + 5.0
    os.utime(secret, (new_ts, new_ts))
    new_token = secret.read_text().strip()
    assert new_token != original

    # Old token now denied; the interceptor reloads the store and
    # returns the abort handler (not the real one).
    denied = await interceptor.intercept_service(
        _continuation,
        _details(
            "/xrlenv.rollout_control.v1.RolloutControl/Heartbeat",
            [("authorization", f"Bearer {original}")],
        ),
    )
    assert denied is not handler  # abort handler, not the wrapped handler.
    audit_rows = [r.kind for r in state.audit_since(0)]
    assert "auth.denied" in audit_rows

    # New token authorizes successfully.
    ok = await interceptor.intercept_service(
        _continuation,
        _details(
            "/xrlenv.rollout_control.v1.RolloutControl/Heartbeat",
            [("authorization", f"Bearer {new_token}")],
        ),
    )
    assert ok is handler


# ──────────────────────────────────────────────────────────────────────────────
# B7.3 — generate_token + viewer role
# ──────────────────────────────────────────────────────────────────────────────


def test_generate_token_admin_tier_carries_prefix() -> None:
    """``generate_token`` stamps a privilege-signaling prefix on the
    admin-tier roles so the token's purpose is visible at a glance
    when an operator pastes it into chat / a runbook."""
    viewer_tok = generate_token("viewer")
    operator_tok = generate_token("operator")
    assert viewer_tok.startswith("read_")
    assert operator_tok.startswith("write_")


def test_generate_token_rpc_tier_stays_raw() -> None:
    """``node`` and ``consumer`` tokens are systemd / env-installed,
    rarely hand-shared — they don't need the privilege prefix."""
    node_tok = generate_token("node")
    consumer_tok = generate_token("consumer")
    assert not node_tok.startswith(("read_", "write_"))
    assert not consumer_tok.startswith(("read_", "write_"))


def test_generate_token_round_trips_through_token_store() -> None:
    """Generated tokens verify cleanly and report the right role."""
    store = TokenStore()
    viewer = generate_token("viewer")
    store.add("viewer", viewer)
    identity = store.verify(viewer)
    assert identity is not None
    assert identity.role == "viewer"
    assert identity.scope == "admin.read"


def test_role_token_prefix_table_in_sync() -> None:
    """Sanity: every role known to ``ROLE_DEFAULT_SCOPE`` has a prefix
    entry (even if it's empty). Keeps the two tables aligned so a
    future role addition fails loudly if someone forgets the prefix."""
    assert set(ROLE_TOKEN_PREFIX) == set(ROLE_DEFAULT_SCOPE)


def test_viewer_token_unprefixed_legacy_still_verifies() -> None:
    """A token registered without the prefix (e.g. a legacy token
    issued before B7.3 shipped) still verifies normally. The prefix
    is generator-side; ``verify`` matches by raw string."""
    store = TokenStore()
    store.add("viewer", "legacy-unprefixed")
    identity = store.verify("legacy-unprefixed")
    assert identity is not None
    assert identity.role == "viewer"


# ──────────────────────────────────────────────────────────────────────────────
# Multi-user / fair-share Slice A — per-user tokens (users.json, owner_id).
#
# Per-user tokens carry a distinct ``owner_id`` and are persisted *hashed*
# (full SHA-256) in ``users.json``; the plaintext is printed once at issue
# and never written to disk. ``TokenStore.verify`` falls back to the SHA map
# after the legacy raw-token miss. These tests pin: round-trip, coexistence
# with legacy shared tokens, hash-at-rest, idempotency, validation, revocation,
# hot-add, malformed-row tolerance, and the two CLI surfaces.
# ──────────────────────────────────────────────────────────────────────────────


# 1 — full round-trip through disk: write_user_record → load → verify.
@pytest.mark.parametrize(
    ("role", "expected_scope"),
    [("consumer", "consumer.rollout"), ("viewer", "admin.read")],
)
def test_write_user_record_roundtrips_through_load_and_verify(
    tmp_path: Path, role: str, expected_scope: str,
) -> None:
    users = tmp_path / "users.json"
    raw = "raw-bearer-for-alice"
    token_id = write_user_record(
        users, token=raw, role=role,  # type: ignore[arg-type]
        owner_id="alice", display_name="Alice",
    )
    assert token_id == token_sha256(raw)[:12]

    s = TokenStore.load(secrets_root=tmp_path, env={})
    identity = s.verify(raw)
    assert identity is not None
    assert identity.owner_id == "alice"
    assert identity.role == role
    assert identity.display_name == "Alice"
    assert identity.scope == expected_scope
    # token_id / digest_hint derive from the full SHA, and the 6-char hint is
    # the prefix of the 12-char id (revoke-by-6-prefix invariant).
    assert identity.token_id == token_sha256(raw)[:12]
    assert identity.digest_hint == token_sha256(raw)[:6]
    assert identity.token_id.startswith(identity.digest_hint)


# 2 — legacy shared role-token still verifies and carries owner_id="default",
#     coexisting with a per-user token of the same role.
def test_legacy_shared_token_coexists_with_per_user_tokens(tmp_path: Path) -> None:
    shared = tmp_path / "consumer.token"
    write_secret_file(shared, "shared-consumer")
    write_user_record(
        tmp_path / "users.json", token="alice-bearer",
        role="consumer", owner_id="alice",
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})

    shared_identity = s.verify("shared-consumer")
    assert shared_identity is not None
    assert shared_identity.owner_id == "default"

    user_identity = s.verify("alice-bearer")
    assert user_identity is not None
    assert user_identity.owner_id == "alice"


# 2b — collision footgun: a per-user token whose value ALSO leaked into the
#      control plane as the shared role-token (env var or <role>.token). The
#      more-specific per-user identity must win, and a warning must fire.
def test_colliding_env_and_per_user_token_resolves_to_per_user(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """yutong's exact case: the same secret is both XRLENV_CONSUMER_TOKEN (which
    would otherwise register as the shared owner_id='default' role-token) and a
    per-user token for owner='yutong'. Per-user wins → owner='yutong', and the
    operator is warned about the leaked client bearer."""
    write_user_record(
        tmp_path / "users.json", token="dual-purpose-tok",
        role="consumer", owner_id="yutong",
    )
    with caplog.at_level("WARNING"):
        s = TokenStore.load(
            secrets_root=tmp_path,
            env={"XRLENV_CONSUMER_TOKEN": "dual-purpose-tok"},
        )
    identity = s.verify("dual-purpose-tok")
    assert identity is not None
    assert identity.owner_id == "yutong"  # per-user wins, NOT "default"
    assert identity.role == "consumer"
    warned = [r.message for r in caplog.records if "per-user identity" in r.message]
    assert warned, "expected a collision warning"
    assert "yutong" in warned[0]
    assert "XRLENV_CONSUMER_TOKEN" in warned[0]


def test_colliding_file_and_per_user_token_resolves_to_per_user(
    tmp_path: Path,
) -> None:
    """Same collision via the on-disk <role>.token file rather than the env
    var — per-user still wins."""
    write_secret_file(tmp_path / "consumer.token", "dual-purpose-tok")
    write_user_record(
        tmp_path / "users.json", token="dual-purpose-tok",
        role="consumer", owner_id="bob",
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})
    identity = s.verify("dual-purpose-tok")
    assert identity is not None
    assert identity.owner_id == "bob"


def test_no_collision_leaves_shared_token_intact(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Sanity: when the shared role-token's value is DISTINCT from every
    per-user token, the reconciliation must not fire — the shared
    owner_id='default' token keeps working and no warning is logged."""
    write_secret_file(tmp_path / "consumer.token", "shared-distinct")
    write_user_record(
        tmp_path / "users.json", token="alice-bearer",
        role="consumer", owner_id="alice",
    )
    with caplog.at_level("WARNING"):
        s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("shared-distinct").owner_id == "default"
    assert s.verify("alice-bearer").owner_id == "alice"
    assert not [r for r in caplog.records if "per-user identity" in r.message]


# 2d — regression (yutong's prod crash): `tokens list` must not KeyError when a
#      role's active token was reconciled away by the collision above. Step 5
#      drops the colliding token from ``_by_token`` but leaves ``_by_role`` still
#      pointing at it; the CLI has to resolve the shadowing per-user identity
#      instead of blindly indexing ``_by_token[_by_role[role]]``.
def test_cmd_tokens_list_survives_shared_per_user_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ambient XRLENV_*_TOKEN (e.g. a dev box .env auto-load) would override the
    # on-disk consumer.token and defeat the collision — scrub them so the test
    # is deterministic regardless of the runner's environment.
    for var in (
        "XRLENV_NODE_TOKEN", "XRLENV_CONSUMER_TOKEN",
        "XRLENV_OPERATOR_TOKEN", "XRLENV_VIEWER_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    # The shared consumer role-token value collides with yutong's per-user token.
    write_secret_file(tmp_path / "consumer.token", "dual-purpose-tok")
    write_user_record(
        tmp_path / "users.json", token="dual-purpose-tok",
        role="consumer", owner_id="yutong", display_name="Yutong",
    )
    out = io.StringIO()
    rc = cmd_tokens_list(secrets_root=tmp_path, out=out)  # must NOT KeyError
    assert rc == 0
    body = out.getvalue()
    # The consumer role row is shown as shadowed by the per-user identity — not
    # crashed on, and not silently mislabelled as the shared owner=default.
    assert "consumer  shadowed" in body
    assert "owner=yutong" in body
    assert "consumer  active" not in body
    # The raw bearer is never printed.
    assert "dual-purpose-tok" not in body


# 3 — unknown / garbage / empty bearer returns None on the SHA fallback path.
def test_verify_rejects_unknown_and_empty_per_user_bearer(tmp_path: Path) -> None:
    write_user_record(
        tmp_path / "users.json", token="alice-bearer",
        role="consumer", owner_id="alice",
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("not-a-real-token") is None
    assert s.verify("") is None
    assert s.verify(None) is None


# 4 — users.json holds ONLY hashes; the raw bearer never lands on disk.
def test_users_json_stores_only_hashes_never_plaintext(tmp_path: Path) -> None:
    users = tmp_path / "users.json"
    raw = "super-secret-bearer-value-xyz"
    write_user_record(users, token=raw, role="consumer", owner_id="alice")
    blob = users.read_bytes()
    assert raw.encode("utf-8") not in blob
    # The hash *is* present (sanity that we wrote something usable).
    assert token_sha256(raw).encode("utf-8") in blob


# 5 — idempotent on the same bearer: same token_id, no duplicate record.
def test_write_user_record_idempotent_on_same_bearer(tmp_path: Path) -> None:
    import json

    users = tmp_path / "users.json"
    raw = "alice-bearer"
    first = write_user_record(users, token=raw, role="consumer", owner_id="alice")
    second = write_user_record(users, token=raw, role="consumer", owner_id="alice")
    assert first == second
    records = json.loads(users.read_text())
    assert len(records) == 1


# 6 — register_user validates owner_id and token_sha length.
def test_register_user_rejects_empty_owner_id() -> None:
    s = TokenStore()
    with pytest.raises(ValueError, match="empty owner_id"):
        s.register_user(token_sha="a" * 64, role="consumer", owner_id="")
    with pytest.raises(ValueError, match="empty owner_id"):
        s.register_user(token_sha="a" * 64, role="consumer", owner_id="   ")


def test_register_user_rejects_non_64_char_sha() -> None:
    s = TokenStore()
    with pytest.raises(ValueError, match="64-hex"):
        s.register_user(token_sha="abc", role="consumer", owner_id="alice")
    with pytest.raises(ValueError, match="64-hex"):
        s.register_user(token_sha="a" * 65, role="consumer", owner_id="alice")


# 7 — revoking one per-user token leaves the others valid.
def test_revoking_one_per_user_token_leaves_others_valid(tmp_path: Path) -> None:
    users = tmp_path / "users.json"
    write_user_record(users, token="alice-bearer", role="consumer", owner_id="alice")
    write_user_record(users, token="bob-bearer", role="consumer", owner_id="bob")

    append_revocation(
        tmp_path / "revoked.json", token_sha256("alice-bearer")[:12],
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("alice-bearer") is None
    bob = s.verify("bob-bearer")
    assert bob is not None
    assert bob.owner_id == "bob"


# 7b — revocation appended after load is honored on maybe_reload().
def test_revoked_per_user_token_picked_up_by_maybe_reload(tmp_path: Path) -> None:
    import os

    users = tmp_path / "users.json"
    write_user_record(users, token="alice-bearer", role="consumer", owner_id="alice")
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("alice-bearer") is not None

    revoked = tmp_path / "revoked.json"
    append_revocation(revoked, token_sha256("alice-bearer")[:12])
    new_ts = revoked.stat().st_mtime + 5.0
    os.utime(revoked, (new_ts, new_ts))

    assert s.maybe_reload() is True
    assert s.verify("alice-bearer") is None


# 7c — audit M1: re-revoking a per-user token_id is an idempotent success.
def test_cmd_tokens_revoke_per_user_is_idempotent(tmp_path: Path) -> None:
    """A second revoke of the same per-user token_id exits 0, not 1.

    Regression for audit M1: _load_from() used to skip user records already
    listed in revoked.json, so the second revoke couldn't resolve the
    token_id and returned "no known token matches" (exit 1). The fix loads
    revoked identities (verify() still refuses them) so revoke stays a no-op
    success — matching the role-token path and the documented CLI contract.
    """
    write_user_record(
        tmp_path / "users.json", token="alice-bearer", role="consumer",
        owner_id="alice",
    )
    token_id = token_sha256("alice-bearer")[:12]
    out1 = io.StringIO()
    assert cmd_tokens_revoke(token_id, secrets_root=tmp_path, out=out1) == 0
    # Second revoke of the same id is idempotent.
    out2 = io.StringIO()
    assert cmd_tokens_revoke(token_id, secrets_root=tmp_path, out=out2) == 0
    assert f"revoked token_id={token_id}" in out2.getvalue()


# 7d — a revoked per-user identity stays resolvable but never verifies.
def test_revoked_per_user_identity_loaded_but_unverifiable(tmp_path: Path) -> None:
    write_user_record(
        tmp_path / "users.json", token="alice-bearer", role="consumer",
        owner_id="alice",
    )
    token_id = token_sha256("alice-bearer")[:12]
    append_revocation(tmp_path / "revoked.json", token_id)
    s = TokenStore.load(secrets_root=tmp_path, env={})
    # verify() refuses the revoked bearer...
    assert s.verify("alice-bearer") is None
    # ...but the identity is still loaded, so revoke() can resolve the id
    # (this is what keeps re-revoke idempotent).
    assert any(i.token_id == token_id for i in s.users())
    assert s.revoke(token_id) == token_id


# 7e — tokens list marks a revoked per-user row instead of showing it as live.
def test_cmd_tokens_list_marks_revoked_per_user_row(tmp_path: Path) -> None:
    write_user_record(
        tmp_path / "users.json", token="alice-bearer", role="consumer",
        owner_id="alice",
    )
    token_id = token_sha256("alice-bearer")[:12]
    cmd_tokens_revoke(token_id, secrets_root=tmp_path, out=io.StringIO())
    out = io.StringIO()
    cmd_tokens_list(secrets_root=tmp_path, out=out)
    text = out.getvalue()
    assert "revoked" in text
    assert "owner=alice" in text
    # The raw bearer is never printed.
    assert "alice-bearer" not in text


# 8 — hot-add: write a new user after load, maybe_reload picks it up.
def test_hot_add_new_user_via_maybe_reload(tmp_path: Path) -> None:
    import os

    users = tmp_path / "users.json"
    write_user_record(users, token="alice-bearer", role="consumer", owner_id="alice")
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert s.verify("bob-bearer") is None

    write_user_record(users, token="bob-bearer", role="consumer", owner_id="bob")
    new_ts = users.stat().st_mtime + 5.0
    os.utime(users, (new_ts, new_ts))

    assert s.maybe_reload() is True
    bob = s.verify("bob-bearer")
    assert bob is not None
    assert bob.owner_id == "bob"
    # The pre-existing user survives the rebuild.
    assert s.verify("alice-bearer") is not None


# 9 — _load_user_records skips malformed rows without dropping the good one.
def test_load_user_records_skips_malformed_rows(tmp_path: Path) -> None:
    import json

    good_sha = token_sha256("good-bearer")
    users = tmp_path / "users.json"
    users.write_text(json.dumps([
        {"token_sha": good_sha, "token_id": good_sha[:12],
         "role": "consumer", "owner_id": "alice"},
        "not-a-dict",
        {"role": "consumer", "owner_id": "bob"},               # missing token_sha
        {"token_sha": "tooshort", "role": "consumer", "owner_id": "x"},  # bad len
        {"token_sha": "f" * 64, "role": "wizard", "owner_id": "y"},      # bad role
        {"token_sha": "e" * 64, "role": "consumer"},           # missing owner_id
        {"token_sha": "d" * 64, "role": "consumer", "owner_id": ""},     # empty owner
    ]))

    records = _security._load_user_records(users)
    assert len(records) == 1
    assert records[0]["token_sha"] == good_sha

    # End-to-end: the one good row still verifies through a full load.
    s = TokenStore.load(secrets_root=tmp_path, env={})
    identity = s.verify("good-bearer")
    assert identity is not None
    assert identity.owner_id == "alice"


def test_load_user_records_tolerates_malformed_json(tmp_path: Path) -> None:
    users = tmp_path / "users.json"
    users.write_text("{ this is not json")
    assert _security._load_user_records(users) == []
    # And a non-list top-level shape.
    users.write_text('{"token_sha": "x"}')
    assert _security._load_user_records(users) == []


# 10 — users() returns only per-user identities; is_empty reflects the SHA map.
def test_users_returns_only_per_user_identities(tmp_path: Path) -> None:
    write_secret_file(tmp_path / "consumer.token", "shared-consumer")
    write_user_record(
        tmp_path / "users.json", token="alice-bearer",
        role="consumer", owner_id="alice",
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})
    owners = [i.owner_id for i in s.users()]
    assert owners == ["alice"]  # the shared role token is excluded.


def test_is_empty_false_when_only_per_user_tokens_present(tmp_path: Path) -> None:
    write_user_record(
        tmp_path / "users.json", token="alice-bearer",
        role="consumer", owner_id="alice",
    )
    s = TokenStore.load(secrets_root=tmp_path, env={})
    assert not s.is_empty
    assert s.known_roles == []  # no shared role tokens loaded.


# ──────────────────────────────────────────────────────────────────────────────
# Multi-user Slice A — CLI: tokens issue --owner / tokens list per-user section
# ──────────────────────────────────────────────────────────────────────────────


# 11 — issue --owner: rc 0, writes users.json, prints owner + token_id, and the
#      raw token appears in stdout exactly once.
def test_cmd_tokens_issue_owner_writes_users_json_and_prints_once(
    tmp_path: Path,
) -> None:
    out = io.StringIO()
    rc = cmd_tokens_issue(
        "consumer", owner="alice", display_name="Alice",
        secrets_root=tmp_path, out=out,
    )
    assert rc == 0
    users = tmp_path / "users.json"
    assert users.exists()
    body = out.getvalue()
    assert "owner=alice" in body
    assert "(Alice)" in body
    # Extract the printed raw token and confirm it (a) verifies and (b) is
    # printed exactly once and (c) is never persisted to disk.
    raw_line = next(ln for ln in body.splitlines() if "raw token:" in ln)
    raw = raw_line.split("raw token:", 1)[1].strip()
    assert body.count(raw) == 1
    assert raw.encode("utf-8") not in users.read_bytes()

    s = TokenStore.load(secrets_root=tmp_path, env={})
    identity = s.verify(raw)
    assert identity is not None
    assert identity.owner_id == "alice"
    # The printed token_id matches the stored identity.
    assert f"token_id={identity.token_id}" in body


# 12 — role=node with --owner is rejected (exit 2); nothing written.
def test_cmd_tokens_issue_owner_rejects_node_role(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = cmd_tokens_issue("node", owner="x", secrets_root=tmp_path, out=out)
    assert rc == 2
    assert "node" in out.getvalue()
    assert not (tmp_path / "users.json").exists()


# 13 — empty / whitespace owner is rejected (exit 2).
def test_cmd_tokens_issue_owner_rejects_empty_owner(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = cmd_tokens_issue("consumer", owner="  ", secrets_root=tmp_path, out=out)
    assert rc == 2
    assert "non-empty tenant id" in out.getvalue()
    assert not (tmp_path / "users.json").exists()


# 14 — tokens list renders the per-user section without leaking raw bytes.
def test_cmd_tokens_list_shows_per_user_section(tmp_path: Path) -> None:
    issue_out = io.StringIO()
    cmd_tokens_issue(
        "consumer", owner="alice", display_name="Alice",
        secrets_root=tmp_path, out=issue_out,
    )
    raw_line = next(
        ln for ln in issue_out.getvalue().splitlines() if "raw token:" in ln
    )
    raw = raw_line.split("raw token:", 1)[1].strip()
    token_id = token_sha256(raw)[:12]

    out = io.StringIO()
    rc = cmd_tokens_list(secrets_root=tmp_path, out=out)
    assert rc == 0
    body = out.getvalue()
    assert "per-user tokens" in body
    assert "owner=alice" in body
    assert f"token_id={token_id}" in body
    assert "(Alice)" in body
    # The raw bearer is never printed by list.
    assert raw not in body
