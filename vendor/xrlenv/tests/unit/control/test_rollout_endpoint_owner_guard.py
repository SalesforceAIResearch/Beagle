"""Multi-user (audit M2) — cross-owner enforcement on follow-up RolloutControl
RPCs.

A rollout / raw-container session is stamped server-side with the owner_id of
the bearer that created it (pinned in ``test_rollout_endpoint_owner_stamp.py``).
The *follow-up* RPCs (Step / Finish / Cancel / Replay / SetFinalReward and the
raw ContainerExec / DestroyContainer / Put / Get / ExecStream) must then refuse
a caller whose verified owner differs from the stored owner — otherwise tenant
A could drive / tear down tenant B's session by id.

These tests drive ``RolloutControlServicer`` methods directly with a hand-rolled
fake context (no real gRPC port — the sandbox blocks ``127.0.0.1:0``) and a fake
service that records call-throughs + answers ``rollout_owner`` / ``raw_session_owner``.
The guard either:
  * aborts ``PERMISSION_DENIED`` (cross-owner) — the fake ``abort`` raises
    ``_Aborted`` and the underlying service method is NOT called, or
  * passes through — the fake service records the call (and raises a distinct
    ``_CalledThrough`` sentinel so we never exercise the wire-conversion layer
    on stub return values).
"""

from __future__ import annotations

from typing import Any

import grpc
import pytest
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.control.rollout_endpoint import RolloutControlServicer
from xrlenv.control.security import TokenStore, write_user_record

# ── Sentinels ─────────────────────────────────────────────────────────────────


class _Aborted(Exception):
    """Raised by the fake context's ``abort`` (mirrors grpc-aio's abort which
    raises so control never returns to the method body)."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


class _CalledThrough(Exception):
    """Raised by a fake service method after it records the call, so the guard
    "allow" path is observable without driving the proto-conversion layer on a
    stub return value."""


# ── Fakes ─────────────────────────────────────────────────────────────────────


class _FakeContext:
    """Minimal grpc ServicerContext. ``abort`` is async + raises ``_Aborted``."""

    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self._metadata = metadata
        self.aborted: tuple[grpc.StatusCode, str] | None = None

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    def set_trailing_metadata(self, md: Any) -> None:
        # _abort_with_xrlenv_error sets structured error metadata before abort.
        self.trailing_metadata = md

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.aborted = (code, details)
        raise _Aborted(code, details)


class _FakeService:
    """Records call-throughs and answers the owner lookups the guard reads.

    ``owners`` maps rollout_id -> owner_id for *both* gym/step rollouts
    (``rollout_owner``) and raw sessions (``raw_session_owner``); the guard
    calls one or the other depending on the RPC, and a real id is only ever one
    kind, so a single map is faithful.
    """

    def __init__(self, owners: dict[str, str]) -> None:
        self._owners = owners
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    # Owner resolvers the servicer reaches via getattr.
    def rollout_owner(self, rollout_id: str) -> str | None:
        return self._owners.get(rollout_id)

    def raw_session_owner(self, rollout_id: str) -> str | None:
        return self._owners.get(rollout_id)

    # Recorded follow-up methods. Each records then raises _CalledThrough so the
    # servicer never reaches its wire-conversion step on a stub value.
    async def step(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("step", args, kw))
        raise _CalledThrough

    async def finish(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("finish", args, kw))
        raise _CalledThrough

    async def cancel(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("cancel", args, kw))
        raise _CalledThrough

    async def replay(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("replay", args, kw))
        raise _CalledThrough

    async def set_final_reward(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("set_final_reward", args, kw))
        raise _CalledThrough

    async def container_exec(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("container_exec", args, kw))
        raise _CalledThrough

    async def apply_egress(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("apply_egress", args, kw))
        raise _CalledThrough

    async def destroy_container(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("destroy_container", args, kw))
        raise _CalledThrough

    async def cancel_group(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("cancel_group", args, kw))
        raise _CalledThrough

    async def terminate_raw_group(self, *args: Any, **kw: Any) -> Any:
        self.calls.append(("terminate_raw_group", args, kw))
        raise _CalledThrough

    async def heartbeat_many(self, ids: list[str]) -> None:
        # Heartbeat returns HeartbeatResponse() without converting this value,
        # so a plain return (no sentinel) keeps the allow path clean.
        self.calls.append(("heartbeat_many", (ids,), {}))

    def called(self, name: str) -> bool:
        return any(c[0] == name for c in self.calls)

    def last(self, name: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for kind, args, kw in reversed(self.calls):
            if kind == name:
                return args, kw
        raise AssertionError(f"{name} was never called")


def _user_store(
    tmp_path: Any, *, token: str, owner_id: str,
) -> TokenStore:
    write_user_record(
        tmp_path / "users.json", token=token, role="consumer",  # type: ignore[arg-type]
        owner_id=owner_id,
    )
    return TokenStore.load(secrets_root=tmp_path, env={})


def _bearer(token: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token}"),)


# A standard owners map used across tests: one id per tenant.
_R_ALICE = "rid-alice"
_R_BOB = "rid-bob"
_OWNERS = {_R_ALICE: "alice", _R_BOB: "bob"}


# ── Cross-owner abort vs. same-owner pass-through across the guarded RPCs ──────


@pytest.mark.asyncio
async def test_step_aborts_on_cross_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_Aborted) as exc:
        await servicer.Step(rpb.StepRequest(rollout_id=_R_BOB, action_json=b"0"), ctx)

    assert exc.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert not service.called("step")


@pytest.mark.asyncio
async def test_step_passes_through_for_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.Step(rpb.StepRequest(rollout_id=_R_ALICE, action_json=b"0"), ctx)

    assert ctx.aborted is None
    assert service.called("step")


@pytest.mark.asyncio
async def test_finish_aborts_on_cross_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_Aborted) as exc:
        await servicer.Finish(rpb.FinishRequest(rollout_id=_R_BOB), ctx)
    assert exc.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert not service.called("finish")


@pytest.mark.asyncio
async def test_finish_passes_through_for_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.Finish(rpb.FinishRequest(rollout_id=_R_ALICE), ctx)
    assert ctx.aborted is None
    assert service.called("finish")


@pytest.mark.asyncio
async def test_cancel_aborts_on_cross_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_Aborted) as exc:
        await servicer.Cancel(rpb.CancelRequest(rollout_id=_R_BOB, reason="x"), ctx)
    assert exc.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert not service.called("cancel")


@pytest.mark.asyncio
async def test_cancel_passes_through_for_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.Cancel(rpb.CancelRequest(rollout_id=_R_ALICE, reason="x"), ctx)
    assert ctx.aborted is None
    assert service.called("cancel")


@pytest.mark.asyncio
async def test_container_exec_aborts_on_cross_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_Aborted) as exc:
        await servicer.ContainerExec(
            rpb.ContainerExecRequest(
                rollout_id=_R_BOB, container_id="c", cmd=["ls"],
            ),
            ctx,
        )
    assert exc.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert not service.called("container_exec")


@pytest.mark.asyncio
async def test_container_exec_passes_through_for_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.ContainerExec(
            rpb.ContainerExecRequest(
                rollout_id=_R_ALICE, container_id="c", cmd=["ls"],
            ),
            ctx,
        )
    assert ctx.aborted is None
    assert service.called("container_exec")


# ── ApplyEgress (spec 07) — owner guard + decode at the consumer RPC (T1) ─────


@pytest.mark.asyncio
async def test_apply_egress_aborts_on_cross_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_Aborted) as exc:
        await servicer.ApplyEgress(
            rpb.ApplyEgressRequest(
                rollout_id=_R_BOB, container_id="c",
                allow=[rpb.EgressAllowEntry(cidr="10.0.0.0/8")],
            ),
            ctx,
        )
    assert exc.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert not service.called("apply_egress")


@pytest.mark.asyncio
async def test_apply_egress_passes_through_with_allowlist(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.ApplyEgress(
            rpb.ApplyEgressRequest(
                rollout_id=_R_ALICE, container_id="c",
                allow=[
                    rpb.EgressAllowEntry(cidr="3.149.157.52/32", ports=[443]),
                    rpb.EgressAllowEntry(cidr="18.225.81.238/32"),
                ],
                dns_resolver="10.0.0.2/32",
            ),
            ctx,
        )
    assert ctx.aborted is None
    _args, kw = service.last("apply_egress")
    al = kw["allowlist"]
    assert [(r.cidr, r.ports) for r in al.rules] == [
        ("3.149.157.52/32", (443,)), ("18.225.81.238/32", None),
    ]
    assert kw["dns_resolver"] == "10.0.0.2/32"


@pytest.mark.asyncio
async def test_apply_egress_empty_allowlist_passes(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.ApplyEgress(
            rpb.ApplyEgressRequest(rollout_id=_R_ALICE, container_id="c"),
            ctx,
        )
    _args, kw = service.last("apply_egress")
    assert kw["allowlist"].rules == ()  # empty = block-all, not an error
    assert kw["dns_resolver"] is None


@pytest.mark.asyncio
async def test_apply_egress_invalid_allowlist_aborts(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    # Owner guard passes (alice owns _R_ALICE); the bad cidr fails decode.
    with pytest.raises(_Aborted):
        await servicer.ApplyEgress(
            rpb.ApplyEgressRequest(
                rollout_id=_R_ALICE, container_id="c",
                allow=[rpb.EgressAllowEntry(cidr="not-a-cidr")],
            ),
            ctx,
        )
    assert ctx.aborted is not None
    assert not service.called("apply_egress")


@pytest.mark.asyncio
async def test_destroy_container_aborts_on_cross_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_Aborted) as exc:
        await servicer.DestroyContainer(
            rpb.DestroyContainerRequest(rollout_id=_R_BOB, container_id="c"),
            ctx,
        )
    assert exc.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert not service.called("destroy_container")


@pytest.mark.asyncio
async def test_destroy_container_passes_through_for_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.DestroyContainer(
            rpb.DestroyContainerRequest(rollout_id=_R_ALICE, container_id="c"),
            ctx,
        )
    assert ctx.aborted is None
    assert service.called("destroy_container")


# ── Auth-off path: empty / no store → guard is a no-op (single-tenant) ─────────


@pytest.mark.asyncio
async def test_step_auth_off_none_store_passes_bob_id() -> None:
    """No TokenStore → single-tenant: the guard never fires even for a
    bob-owned id (there are no tenants to cross)."""
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=None)
    ctx = _FakeContext(_bearer("ignored"))

    with pytest.raises(_CalledThrough):
        await servicer.Step(rpb.StepRequest(rollout_id=_R_BOB, action_json=b"0"), ctx)
    assert ctx.aborted is None
    assert service.called("step")


@pytest.mark.asyncio
async def test_step_auth_off_empty_store_passes_bob_id(tmp_path: Any) -> None:
    empty = TokenStore.load(secrets_root=tmp_path, env={})
    assert empty.is_empty
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=empty)
    ctx = _FakeContext(_bearer("anything"))

    with pytest.raises(_CalledThrough):
        await servicer.Step(rpb.StepRequest(rollout_id=_R_BOB, action_json=b"0"), ctx)
    assert ctx.aborted is None
    assert service.called("step")


# ── Unknown id: stored owner None → no abort, call through ─────────────────────


@pytest.mark.asyncio
async def test_step_unknown_id_passes_through(tmp_path: Any) -> None:
    """An id no one owns (rollout_owner → None) is not a cross-owner case; the
    guard is a no-op and the underlying method (which yields its own NotFound)
    runs."""
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.Step(
            rpb.StepRequest(rollout_id="rid-unknown", action_json=b"0"), ctx,
        )
    assert ctx.aborted is None
    assert service.called("step")


# ── CancelGroup: owner_id threaded into service.cancel_group ───────────────────


@pytest.mark.asyncio
async def test_cancel_group_threads_scoped_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.CancelGroup(
            rpb.CancelGroupRequest(group_id="g1", reason="x"), ctx,
        )
    _args, kw = service.last("cancel_group")
    assert kw["owner_id"] == "alice"


@pytest.mark.asyncio
async def test_cancel_group_owner_none_without_store() -> None:
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=None)
    ctx = _FakeContext(_bearer("ignored"))

    with pytest.raises(_CalledThrough):
        await servicer.CancelGroup(
            rpb.CancelGroupRequest(group_id="g1", reason="x"), ctx,
        )
    _args, kw = service.last("cancel_group")
    assert kw["owner_id"] is None


# ── TerminateRawGroup: owner_id threaded into service.terminate_raw_group ───────


@pytest.mark.asyncio
async def test_terminate_raw_group_threads_scoped_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    with pytest.raises(_CalledThrough):
        await servicer.TerminateRawGroup(
            rpb.TerminateRawGroupRequest(group_id="g1", reason="x"), ctx,
        )
    _args, kw = service.last("terminate_raw_group")
    assert kw["owner_id"] == "alice"


@pytest.mark.asyncio
async def test_terminate_raw_group_owner_none_without_store() -> None:
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=None)
    ctx = _FakeContext(_bearer("ignored"))

    with pytest.raises(_CalledThrough):
        await servicer.TerminateRawGroup(
            rpb.TerminateRawGroupRequest(group_id="g1", reason="x"), ctx,
        )
    _args, kw = service.last("terminate_raw_group")
    assert kw["owner_id"] is None


# ── Heartbeat: drop ids owned by another tenant ────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_drops_foreign_ids(tmp_path: Any) -> None:
    """alice heartbeats [alice, bob, unknown]; only [alice, unknown] reach
    heartbeat_many — bob's id is silently dropped so alice can't keep a
    stranger's session alive."""
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _FakeContext(_bearer("alice-secret"))

    await servicer.Heartbeat(
        rpb.HeartbeatRequest(rollout_ids=[_R_ALICE, _R_BOB, "rid-unknown"]), ctx,
    )
    (ids,), _kw = service.last("heartbeat_many")
    assert ids == [_R_ALICE, "rid-unknown"]
    assert ctx.aborted is None


@pytest.mark.asyncio
async def test_heartbeat_passes_all_ids_without_store() -> None:
    """Single-tenant (no store): every id passes through, including the
    bob-owned one — there is no tenant boundary to enforce."""
    service = _FakeService(_OWNERS)
    servicer = RolloutControlServicer(service=service, token_store=None)
    ctx = _FakeContext(_bearer("ignored"))

    await servicer.Heartbeat(
        rpb.HeartbeatRequest(rollout_ids=[_R_ALICE, _R_BOB, "rid-unknown"]), ctx,
    )
    (ids,), _kw = service.last("heartbeat_many")
    assert ids == [_R_ALICE, _R_BOB, "rid-unknown"]
