"""gRPC server-side bearer-token + scope interceptor (spec 19, Slice 8).

Wired into :func:`build_distributed_runtime` so every NodeControl gRPC
call (currently just the ``NodeControlStream`` bidi RPC) goes through
the same auth check:

1. Read ``authorization: Bearer <token>`` from the incoming metadata.
2. Verify against the loaded :class:`TokenStore`.
3. Look up the method-to-scope map; reject if the token's scope
   doesn't satisfy the required scope.
4. Write an audit row (``auth.token_used`` / ``auth.denied``).

When the :class:`TokenStore` is **empty** (no tokens issued yet,
phase-0 single-host smoke), the interceptor is a no-op so the
``tests/smoke/single_rollout.py`` and ``tests/smoke/cluster_smoke.py``
embedded-mode paths keep working without an explicit
``xrlenv tokens issue``.

Aborting an in-flight gRPC bidi stream from an interceptor is
spec-21-aware: the spec says the node connects out + sends
``NodeHello`` first; we reject the call with a method handler that
raises immediately, which becomes a stream-open error on the node
side and triggers the existing reconnect-with-backoff loop.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import grpc
from grpc.aio import ServerInterceptor

from xrlenv.control.security import (
    TokenIdentity,
    TokenStore,
    required_scope_for_method,
    scope_satisfies,
)
from xrlenv.control.state import StateStore

LOGGER = logging.getLogger(__name__)

_AUTH_HEADER = "authorization"
_BEARER_PREFIX = "Bearer "


class BearerScopeInterceptor(ServerInterceptor):
    """Async-server interceptor that gates each method on
    :class:`TokenStore` + :func:`required_scope_for_method`.

    A no-op when the :class:`TokenStore` is empty (spec-19 phase-0 lets
    operators run the slice-1 smoke without issuing tokens). Once any
    token is registered the interceptor enforces every request.
    """

    def __init__(
        self,
        *,
        store: TokenStore,
        state: StateStore | None = None,
        audit_success: bool | None = None,
    ) -> None:
        self._store = store
        self._state = state
        # ``auth.token_used`` (successful-auth) rows are written per-RPC, so at
        # scale they dominate the audit table AND churn the SQLite WAL on every
        # call — on prod (2026-07-30) this was 13.2M rows / 2.6 GB, ~99.9% of the
        # table, and the write-path WAL/-shm churn on the Lustre-backed state.db
        # was implicated in repeated control-plane SIGBUS crashes. Default OFF:
        # keep the high-value ``auth.denied`` rows, drop the high-volume /
        # low-value success rows. Set ``XRLENV_AUDIT_AUTH_SUCCESS=1`` to restore
        # per-RPC success auditing (spec 19 full audit trail).
        if audit_success is None:
            audit_success = os.environ.get(
                "XRLENV_AUDIT_AUTH_SUCCESS", "",
            ).strip().lower() in ("1", "true", "yes", "on")
        self._audit_success = audit_success

    async def intercept_service(  # type: ignore[override]
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails],
            Awaitable[grpc.RpcMethodHandler[Any, Any] | None],
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler[Any, Any] | None:
        method = handler_call_details.method
        # Cheap stat() of the secret files; rebuilds the store iff one
        # of them changed. Closes the order-of-operations gotcha where
        # `xrlenv tokens issue ...` after `xrlenv up` started would be
        # ignored until restart.
        self._store.maybe_reload()
        if self._store.is_empty:
            # Phase-0 escape hatch: no tokens issued yet → keep working.
            return await continuation(handler_call_details)

        identity = self._verify(handler_call_details)
        required = required_scope_for_method(method)
        if identity is None:
            self._audit(
                "auth.denied", method=method, role=None, digest_hint=None,
                result="bad_token",
            )
            return _abort_handler(
                grpc.StatusCode.UNAUTHENTICATED,
                "missing or unknown bearer token",
            )
        if required is not None and not scope_satisfies(identity.scope, required):
            self._audit(
                "auth.denied", method=method, role=identity.role,
                digest_hint=identity.digest_hint, result="wrong_scope",
                payload={"required": required, "have": identity.scope},
            )
            return _abort_handler(
                grpc.StatusCode.PERMISSION_DENIED,
                f"scope {identity.scope!r} cannot call {method!r} "
                f"(needs {required!r})",
            )
        if self._audit_success:
            self._audit(
                "auth.token_used", method=method,
                role=identity.role, digest_hint=identity.digest_hint,
                result="ok",
            )
        return await continuation(handler_call_details)

    # ── Internals ────────────────────────────────────────────────────────────

    def _verify(
        self, details: grpc.HandlerCallDetails
    ) -> TokenIdentity | None:
        token = _bearer_from_metadata(details.invocation_metadata)
        return self._store.verify(token)

    def _audit(
        self,
        kind: str,
        *,
        method: str,
        role: str | None,
        digest_hint: str | None,
        result: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._state is None:
            return
        try:
            self._state.append_audit(
                kind, method=method, role=role,
                digest_hint=digest_hint, result=result,
                payload=payload,
            )
        except Exception:
            # Audit is best-effort; an audit-write failure must never block
            # an authenticated request from running.
            LOGGER.exception(
                "auth interceptor: failed to write audit row kind=%s", kind,
            )


def _bearer_from_metadata(metadata: Any) -> str | None:
    """Extract ``Bearer <token>`` from gRPC metadata (case-insensitive key)."""
    if metadata is None:
        return None
    for key, value in metadata:
        if key.lower() != _AUTH_HEADER:
            continue
        decoded = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        decoded = decoded.strip()
        if decoded.startswith(_BEARER_PREFIX):
            return decoded[len(_BEARER_PREFIX):].strip()
        # Tolerate operators who set the bare token without the prefix.
        return decoded
    return None


def _abort_handler(
    code: grpc.StatusCode, detail: str,
) -> grpc.RpcMethodHandler[Any, Any]:
    """Return a stream-stream handler that aborts every call with ``code``.

    The bidi RPC NodeControl.NodeControlStream is a stream-stream method;
    a unary handler would dispatch into the wrong slot and the framework
    would fall through to "method not implemented" instead of returning
    our auth error. Aborting via ``context.abort`` raises an aio
    ``AbortError`` that gRPC translates into the requested status code on
    the wire — exactly what the spec-21 reconnect loop expects to see.
    """

    async def _abort(
        request_iterator: Any, context: grpc.aio.ServicerContext[Any, Any],
    ) -> Any:
        await context.abort(code, detail)

    return grpc.stream_stream_rpc_method_handler(_abort)


__all__ = ["BearerScopeInterceptor"]
