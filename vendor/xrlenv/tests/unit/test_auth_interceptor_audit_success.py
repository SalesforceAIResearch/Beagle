"""Tests for the BearerScopeInterceptor ``audit_success`` flag (Change 1).

Verifies:

1. Default (no env, ``audit_success=None``) — a successful authenticated call
   does NOT produce an ``auth.token_used`` audit row.
2. ``audit_success=True`` — it DOES produce the row.
3. ``auth.denied`` is written on bad/unknown tokens regardless of
   ``audit_success`` value.
4. ``auth.denied`` is written on wrong-scope requests regardless of
   ``audit_success`` value.
5. Env-var parsing: ``XRLENV_AUDIT_AUTH_SUCCESS`` truthy values
   (``"1"``, ``"true"``, ``"on"``, ``"yes"``) → ``_audit_success=True``;
   falsy / absent (``""``, ``"0"``, ``"no"``) → ``False``.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.control.auth_interceptor import BearerScopeInterceptor
from xrlenv.control.security import TokenStore
from xrlenv.control.state import InMemoryStateStore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _details(method: str, metadata: list[tuple[str, str]]) -> Any:
    """Build a minimal HandlerCallDetails-like object (mirrors test_security.py)."""
    obj = MagicMock()
    obj.method = method
    obj.invocation_metadata = metadata
    return obj


_NODE_METHOD = "/xrlenv.node_control.v1.NodeControl/NodeControlStream"


async def _noop_continuation(details: Any) -> Any:
    return MagicMock()


def _store_with_node_token(token: str = "good-token") -> TokenStore:
    store = TokenStore()
    store.add("node", token)
    return store


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 & 2 — successful auth row suppressed / emitted based on audit_success
# ──────────────────────────────────────────────────────────────────────────────


async def test_default_audit_success_none_suppresses_token_used_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no env var and audit_success=None, a valid call does NOT write
    auth.token_used."""
    monkeypatch.delenv("XRLENV_AUDIT_AUTH_SUCCESS", raising=False)

    store = _store_with_node_token()
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer good-token")]),
    )

    audit_rows = list(state.audit_since(0))
    assert audit_rows == [], (
        "No audit rows expected for a successful call when audit_success is off "
        f"(default); got: {[r.kind for r in audit_rows]}"
    )


async def test_audit_success_true_writes_token_used_row() -> None:
    """With audit_success=True, a valid call DOES write auth.token_used."""
    store = _store_with_node_token()
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state, audit_success=True)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer good-token")]),
    )

    audit_rows = list(state.audit_since(0))
    assert len(audit_rows) == 1
    assert audit_rows[0].kind == "auth.token_used"
    assert audit_rows[0].result == "ok"
    assert audit_rows[0].role == "node"


async def test_audit_success_false_suppresses_token_used_row() -> None:
    """With audit_success=False (explicit), a valid call does NOT write the row."""
    store = _store_with_node_token()
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state, audit_success=False)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer good-token")]),
    )

    audit_rows = list(state.audit_since(0))
    assert audit_rows == []


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — auth.denied always fires on bad token regardless of audit_success
# ──────────────────────────────────────────────────────────────────────────────


async def test_denied_bad_token_fires_when_audit_success_false() -> None:
    """auth.denied for an unknown/bad token must appear even with audit_success=False."""
    store = _store_with_node_token()
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state, audit_success=False)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer wrong-token")]),
    )

    audit_rows = list(state.audit_since(0))
    assert len(audit_rows) == 1
    assert audit_rows[0].kind == "auth.denied"
    assert audit_rows[0].result == "bad_token"


async def test_denied_bad_token_fires_when_audit_success_true() -> None:
    """auth.denied for an unknown token must appear even with audit_success=True."""
    store = _store_with_node_token()
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state, audit_success=True)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer wrong-token")]),
    )

    audit_rows = list(state.audit_since(0))
    assert len(audit_rows) == 1
    assert audit_rows[0].kind == "auth.denied"
    assert audit_rows[0].result == "bad_token"


async def test_denied_bad_token_fires_when_audit_success_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auth.denied for an unknown token must appear even with default (None) setting."""
    monkeypatch.delenv("XRLENV_AUDIT_AUTH_SUCCESS", raising=False)

    store = _store_with_node_token()
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer wrong-token")]),
    )

    audit_rows = list(state.audit_since(0))
    assert len(audit_rows) == 1
    assert audit_rows[0].kind == "auth.denied"
    assert audit_rows[0].result == "bad_token"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — auth.denied always fires on wrong scope regardless of audit_success
# ──────────────────────────────────────────────────────────────────────────────


async def test_denied_wrong_scope_fires_when_audit_success_false() -> None:
    """auth.denied for wrong-scope must fire even with audit_success=False.

    An operator token cannot satisfy the node-only scope on the bidi stream.
    """
    store = TokenStore()
    store.add("operator", "op-tok")
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state, audit_success=False)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer op-tok")]),
    )

    audit_rows = list(state.audit_since(0))
    assert len(audit_rows) == 1
    assert audit_rows[0].kind == "auth.denied"
    assert audit_rows[0].result == "wrong_scope"


async def test_denied_wrong_scope_fires_when_audit_success_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auth.denied for wrong-scope must fire even with the default off setting."""
    monkeypatch.delenv("XRLENV_AUDIT_AUTH_SUCCESS", raising=False)

    store = TokenStore()
    store.add("operator", "op-tok")
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state)

    await interceptor.intercept_service(
        _noop_continuation,
        _details(_NODE_METHOD, [("authorization", "Bearer op-tok")]),
    )

    audit_rows = list(state.audit_since(0))
    assert len(audit_rows) == 1
    assert audit_rows[0].kind == "auth.denied"
    assert audit_rows[0].result == "wrong_scope"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — env-var parsing for XRLENV_AUDIT_AUTH_SUCCESS
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("env_val", ["1", "true", "True", "TRUE", "on", "ON", "yes", "YES", " 1 ", " true "])
def test_env_truthy_values_enable_audit_success(
    env_val: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any case-insensitive truthy value (1, true, on, yes + whitespace) → True."""
    monkeypatch.setenv("XRLENV_AUDIT_AUTH_SUCCESS", env_val)
    store = _store_with_node_token()
    interceptor = BearerScopeInterceptor(store=store, audit_success=None)
    assert interceptor._audit_success is True, (
        f"Expected _audit_success=True for env value {env_val!r}"
    )


@pytest.mark.parametrize("env_val", ["", "0", "no", "No", "NO", "false", "False", "FALSE", "off", "OFF", "nope", "2"])
def test_env_falsy_values_disable_audit_success(
    env_val: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any value not in {1, true, on, yes} (case-insensitive) → False."""
    monkeypatch.setenv("XRLENV_AUDIT_AUTH_SUCCESS", env_val)
    store = _store_with_node_token()
    interceptor = BearerScopeInterceptor(store=store, audit_success=None)
    assert interceptor._audit_success is False, (
        f"Expected _audit_success=False for env value {env_val!r}"
    )


def test_env_absent_disables_audit_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When XRLENV_AUDIT_AUTH_SUCCESS is unset, _audit_success must be False."""
    monkeypatch.delenv("XRLENV_AUDIT_AUTH_SUCCESS", raising=False)
    store = _store_with_node_token()
    interceptor = BearerScopeInterceptor(store=store, audit_success=None)
    assert interceptor._audit_success is False


def test_explicit_audit_success_true_overrides_env_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit audit_success=True bypasses env lookup entirely."""
    monkeypatch.delenv("XRLENV_AUDIT_AUTH_SUCCESS", raising=False)
    store = _store_with_node_token()
    interceptor = BearerScopeInterceptor(store=store, audit_success=True)
    assert interceptor._audit_success is True


def test_explicit_audit_success_false_overrides_truthy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit audit_success=False bypasses env lookup even when env is truthy."""
    monkeypatch.setenv("XRLENV_AUDIT_AUTH_SUCCESS", "1")
    store = _store_with_node_token()
    interceptor = BearerScopeInterceptor(store=store, audit_success=False)
    assert interceptor._audit_success is False


# ──────────────────────────────────────────────────────────────────────────────
# Ensure store-is-empty bypass is unaffected by audit_success
# ──────────────────────────────────────────────────────────────────────────────


async def test_no_op_when_store_empty_regardless_of_audit_success() -> None:
    """Empty TokenStore → pass-through (phase-0 bypass); no audit writes."""
    store = TokenStore()  # empty
    state = InMemoryStateStore()
    interceptor = BearerScopeInterceptor(store=store, state=state, audit_success=True)

    handler = MagicMock()

    async def _continuation(d: Any) -> Any:
        return handler

    result = await interceptor.intercept_service(
        _continuation,
        _details(_NODE_METHOD, []),
    )
    assert result is handler
    assert list(state.audit_since(0)) == []
