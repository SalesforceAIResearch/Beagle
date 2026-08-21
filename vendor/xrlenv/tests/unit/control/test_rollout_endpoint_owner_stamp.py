"""Multi-user (Slice B) — server-authoritative ``owner_id`` stamping on the
RolloutControl gRPC servicer.

The control plane never trusts a client-supplied owner: it resolves the tenant
from the *verified bearer token* on the request and stamps it onto every
rollout / raw-container acquire. These tests pin
``RolloutControlServicer._owner_from_context`` and the ``StartRollout`` stamp
path.

We avoid binding a real gRPC port (the sandbox blocks ``127.0.0.1:0``): each
test calls the servicer method directly with a hand-rolled fake context that
exposes ``invocation_metadata()`` (and an async ``abort`` for the error path).
"""

from __future__ import annotations

from typing import Any

import grpc
import pytest
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.control.rollout_endpoint import RolloutControlServicer
from xrlenv.control.security import TokenStore, write_user_record
from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse

# ── Fakes ─────────────────────────────────────────────────────────────────────


class _FakeContext:
    """Minimal gRPC ServicerContext: only ``invocation_metadata`` is read by
    ``_owner_from_context``. ``abort`` is async + raises, matching grpc-aio."""

    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self._metadata = metadata
        self._trailing: list[tuple[str, Any]] = []

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    def set_trailing_metadata(self, md: list[tuple[str, Any]]) -> None:
        self._trailing = md

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise AssertionError(f"unexpected abort: {code} {details}")


class _RecordingService:
    """Captures the StartRolloutRequest the servicer forwards."""

    def __init__(self, response: StartRolloutResponse) -> None:
        self._response = response
        self.captured: list[StartRolloutRequest] = []

    async def start_rollout(
        self, req: StartRolloutRequest,
    ) -> StartRolloutResponse:
        self.captured.append(req)
        return self._response


def _user_store(
    tmp_path: Any, *, token: str, owner_id: str, role: str = "consumer",
) -> TokenStore:
    """A TokenStore loaded from a users.json holding one per-user token."""
    users = tmp_path / "users.json"
    write_user_record(
        users, token=token, role=role, owner_id=owner_id,  # type: ignore[arg-type]
    )
    return TokenStore.load(secrets_root=tmp_path, env={})


def _bearer(token: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token}"),)


# ── _owner_from_context ───────────────────────────────────────────────────────


def test_owner_from_context_reads_owner_off_token(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    servicer = RolloutControlServicer(service=object(), token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))
    assert servicer._owner_from_context(ctx) == "alice"


def test_owner_from_context_default_when_no_store() -> None:
    servicer = RolloutControlServicer(service=object(), token_store=None)
    ctx = _FakeContext(_bearer("anything"))
    assert servicer._owner_from_context(ctx) == "default"


def test_owner_from_context_default_when_store_empty(tmp_path: Any) -> None:
    # An empty store (no users.json, no role tokens) is the single-tenant /
    # no-auth path: every caller resolves to "default".
    empty = TokenStore.load(secrets_root=tmp_path, env={})
    assert empty.is_empty
    servicer = RolloutControlServicer(service=object(), token_store=empty)
    ctx = _FakeContext(_bearer("alice-secret"))
    assert servicer._owner_from_context(ctx) == "default"


def test_owner_from_context_default_for_unknown_token(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    servicer = RolloutControlServicer(service=object(), token_store=store)
    ctx = _FakeContext(_bearer("not-a-real-token"))
    # Store is non-empty but the presented bearer doesn't resolve → default.
    assert servicer._owner_from_context(ctx) == "default"


def test_owner_from_context_handles_bare_token_without_bearer_prefix(
    tmp_path: Any,
) -> None:
    """Some clients send the raw token without the ``Bearer `` prefix; the
    resolver strips/accepts both forms."""
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    servicer = RolloutControlServicer(service=object(), token_store=store)
    ctx = _FakeContext((("authorization", "alice-secret"),))
    assert servicer._owner_from_context(ctx) == "alice"


def test_owner_from_context_default_when_no_auth_metadata(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    servicer = RolloutControlServicer(service=object(), token_store=store)
    ctx = _FakeContext(())  # no authorization metadata at all
    assert servicer._owner_from_context(ctx) == "default"


# ── StartRollout stamping ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_rollout_stamps_owner_from_token(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="bob-secret", owner_id="bob")
    service = _RecordingService(
        StartRolloutResponse(
            rollout_id="rid-1", init_obs=None, reward_mode="env_step",
        )
    )
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("bob-secret"))

    await servicer.StartRollout(rpb.StartRolloutRequest(template="t"), ctx)

    assert len(service.captured) == 1
    assert service.captured[0].owner_id == "bob"


@pytest.mark.asyncio
async def test_start_rollout_stamps_default_without_store() -> None:
    service = _RecordingService(
        StartRolloutResponse(
            rollout_id="rid-2", init_obs=None, reward_mode="env_step",
        )
    )
    servicer = RolloutControlServicer(service=service, token_store=None)
    ctx = _FakeContext(_bearer("ignored"))

    await servicer.StartRollout(rpb.StartRolloutRequest(template="t"), ctx)

    assert service.captured[0].owner_id == "default"


@pytest.mark.asyncio
async def test_start_rollout_overwrites_any_inbound_owner(tmp_path: Any) -> None:
    """Server-authoritative: even if a client could smuggle an owner into the
    request, the servicer overwrites it with the token-derived owner. We can't
    set owner via the proto (it has no such field), so we assert the captured
    value is the token's owner regardless — pinning that StartRollout always
    stamps rather than passing a client value through."""
    store = _user_store(tmp_path, token="carol-secret", owner_id="carol")
    service = _RecordingService(
        StartRolloutResponse(
            rollout_id="rid-3", init_obs=None, reward_mode="env_step",
        )
    )
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("carol-secret"))

    await servicer.StartRollout(
        rpb.StartRolloutRequest(template="t", task_key="tk"), ctx,
    )

    assert service.captured[0].owner_id == "carol"
    # And the rest of the request still flowed through intact.
    assert service.captured[0].template == "t"
    assert service.captured[0].task_key == "tk"
