"""RolloutControl servicer compose handlers (multi-service step 3a-3c).

Verifies the AcquireComposeProject / DestroyComposeProject servicer handlers unpack
the request, delegate to the service, and pack the response — the last hop wiring
the consumer RPC to the coordinator. No auth (token_store=None → owner "default").
"""
from __future__ import annotations

from typing import Any

import grpc
import pytest
from xrlenv.api._pb2 import rollout_control_pb2 as rpb
from xrlenv.control.raw_container_service import RawComposeAcquireResult
from xrlenv.control.rollout_endpoint import RolloutControlServicer
from xrlenv.control.security import TokenStore, write_user_record


class _Aborted(Exception):
    def __init__(self, code: Any, details: str) -> None:
        self.code = code
        self.details = details
        super().__init__(details)


class _Ctx:
    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self._metadata = metadata
        self.aborted: tuple[Any, str] | None = None

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    def set_trailing_metadata(self, md: Any) -> None:
        self.trailing_metadata = md

    async def abort(self, code: Any, details: str) -> None:
        self.aborted = (code, details)
        raise _Aborted(code, details)


def _user_store(tmp_path: Any, *, token: str, owner_id: str) -> TokenStore:
    write_user_record(
        tmp_path / "users.json", token=token, role="consumer",  # type: ignore[arg-type]
        owner_id=owner_id,
    )
    return TokenStore.load(secrets_root=tmp_path, env={})


def _bearer(token: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token}"),)


class _Service:
    def __init__(self, *, owner: str | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._owner = owner

    def raw_session_owner(self, rollout_id: str) -> str | None:
        return self._owner

    async def acquire_compose_project(self, **kwargs: Any) -> RawComposeAcquireResult:
        self.calls.append(("acquire", kwargs))
        return RawComposeAcquireResult(
            rollout_id="r1", node_id="node-A",
            main_container_id="cidmainfull", main_container_name="proj-main",
            project_name="proj",
            service_container_ids={"main": "cidmainfull", "pg": "cidpgfull"},
            queue_wait_s=1.5,
        )

    async def destroy_compose_project(self, **kwargs: Any) -> None:
        self.calls.append(("destroy", kwargs))


async def test_acquire_compose_project_delegates_and_packs() -> None:
    service = _Service()
    servicer = RolloutControlServicer(service=service, token_store=None)
    req = rpb.AcquireComposeProjectRequest(
        compose_yaml="services:\n  main: {}\n",
        main_service="main",
        up_timeout_s=300.0,
        project_name="proj",
    )
    req.footprint.cpu_request = 6.0
    req.footprint.mem_request_bytes = 6 * 1024**3
    req.images.extend(["reg/ns/tw@sha256:abc", "postgres:14"])

    resp = await servicer.AcquireComposeProject(req, _Ctx())

    # response packed from the service result
    assert resp.rollout_id == "r1"
    assert resp.node_id == "node-A"
    assert resp.main_container_id == "cidmainfull"
    assert resp.project_name == "proj"
    assert dict(resp.service_container_ids) == {
        "main": "cidmainfull", "pg": "cidpgfull",
    }
    assert resp.queue_wait_s == 1.5

    # request unpacked + delegated
    kind, kw = service.calls[0]
    assert kind == "acquire"
    assert kw["compose_yaml"].startswith("services")
    assert kw["images"] == ["reg/ns/tw@sha256:abc", "postgres:14"]
    assert kw["main_service"] == "main"
    assert kw["up_timeout_s"] == 300.0
    # footprint built from the proto ResourceSpec
    assert kw["footprint"].cpu_request == 6.0
    assert kw["footprint"].mem_request_bytes == 6 * 1024**3
    assert kw["owner_id"] == "default"  # no token store


async def test_destroy_compose_project_delegates() -> None:
    service = _Service()
    servicer = RolloutControlServicer(service=service, token_store=None)
    req = rpb.DestroyComposeProjectRequest(rollout_id="r1", project_name="proj")
    await servicer.DestroyComposeProject(req, _Ctx())
    kind, kw = service.calls[0]
    assert kind == "destroy"
    assert kw == {"rollout_id": "r1", "project_name": "proj"}


async def test_destroy_compose_project_aborts_on_cross_owner(tmp_path: Any) -> None:
    # P1 audit: multi-tenant owner scope. The project belongs to "bob"; "alice"
    # must not be able to tear it down.
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _Service(owner="bob")
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _Ctx(_bearer("alice-secret"))
    with pytest.raises(_Aborted) as exc:
        await servicer.DestroyComposeProject(
            rpb.DestroyComposeProjectRequest(rollout_id="rid-bob"), ctx,
        )
    assert exc.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert not service.calls  # never delegated


async def test_destroy_compose_project_passes_for_owner(tmp_path: Any) -> None:
    store = _user_store(tmp_path, token="alice-secret", owner_id="alice")
    service = _Service(owner="alice")
    servicer = RolloutControlServicer(service=service, token_store=store)
    ctx = _Ctx(_bearer("alice-secret"))
    await servicer.DestroyComposeProject(
        rpb.DestroyComposeProjectRequest(rollout_id="rid-alice"), ctx,
    )
    assert ctx.aborted is None
    assert service.calls and service.calls[0][0] == "destroy"
