"""ListRawContainers correlation labels (multi-service step 3c-2).

The node returns per-container ``(container_id, rollout_id, compose_project)`` so
the raw-GC reconciler can recognise a compose PROJECT's main container (route its
rebuild / whole-project teardown). The filter stays ``session_kind=raw`` — sidecars
(``session_kind=compose``) never appear, so the existing node-truth diff is
unchanged.
"""
from __future__ import annotations

from typing import Any

from xrlenv.node.raw_container import RawContainerManager


class _FakeApi:
    def __init__(self, containers: list[dict]) -> None:
        self._containers = containers

    def containers(
        self, *, filters: dict[str, str] | None = None, all: bool = False,
    ) -> list[dict[str, Any]]:
        # mirror docker's label filter: "key=value" matches that label value;
        # a bare "key" matches any container carrying that label (key-exists).
        label = (filters or {}).get("label", "xrlenv.session_kind=raw")
        if "=" in label:
            key, _, value = label.partition("=")
            return [c for c in self._containers if c["Labels"].get(key) == value]
        return [c for c in self._containers if label in c["Labels"]]


class _FakeClient:
    def __init__(self, containers: list[dict]) -> None:
        self.api = _FakeApi(containers)


def _mgr(containers: list[dict]) -> RawContainerManager:
    return RawContainerManager(docker_client=_FakeClient(containers))


async def test_list_on_docker_info_returns_correlation_labels() -> None:
    containers = [
        {"Id": "cid_main", "Labels": {
            "xrlenv.session_kind": "raw",
            "xrlenv.rollout_id": "r1",
            "xrlenv.compose_project": "proj",
        }},
        {"Id": "cid_plain", "Labels": {
            "xrlenv.session_kind": "raw", "xrlenv.rollout_id": "r2",
        }},
        {"Id": "cid_sidecar", "Labels": {
            "xrlenv.session_kind": "compose",
            "xrlenv.rollout_id": "r1", "xrlenv.compose_project": "proj",
        }},
    ]
    info = await _mgr(containers).list_on_docker_info()
    # compose-main carries its project; a plain raw container has empty project;
    # the session_kind=compose sidecar is filtered out (stays off the raw diff).
    assert ("cid_main", "r1", "proj") in info
    assert ("cid_plain", "r2", "") in info
    assert not any(cid == "cid_sidecar" for cid, _r, _p in info)


async def test_list_all_managed_on_docker_info_includes_sidecars() -> None:
    # audit H11 — the managed listing filters on the presence of xrlenv.rollout_id (key-exists),
    # so it catches compose SIDECARS (session_kind=compose) the raw-only listing omits, each
    # tagged with its session_kind. readopt-on-connect uses it to spot sidecar-only survivors.
    containers = [
        {"Id": "cid_main", "Labels": {
            "xrlenv.session_kind": "raw",
            "xrlenv.rollout_id": "r1", "xrlenv.compose_project": "proj",
        }},
        {"Id": "cid_sidecar", "Labels": {
            "xrlenv.session_kind": "compose",
            "xrlenv.rollout_id": "r1", "xrlenv.compose_project": "proj",
        }},
        {"Id": "cid_unmanaged", "Labels": {"some.other": "x"}},  # no rollout_id → excluded
    ]
    info = await _mgr(containers).list_all_managed_on_docker_info()
    assert ("cid_main", "r1", "proj", "raw") in info
    assert ("cid_sidecar", "r1", "proj", "compose") in info   # the sidecar IS included
    assert not any(cid == "cid_unmanaged" for cid, *_ in info)  # unmanaged excluded


class _FakeAgent:
    """Minimal NodeAgent surface the ListRawContainers handler touches."""

    def supported_backends(self) -> list[str]:
        return ["docker"]

    async def list_raw_containers_info(
        self, *, backend: str = "docker",
    ) -> list[tuple[str, str, str]]:
        return [("c-main", "r1", "proj")]                 # raw-only (main)

    async def list_managed_container_info(
        self, *, backend: str = "docker",
    ) -> list[tuple[str, str, str, str]]:
        return [                                          # raw + sidecar
            ("c-main", "r1", "proj", "raw"),
            ("c-side", "r1", "proj", "compose"),
        ]

    def raw_container_manager(self, backend: str) -> None:
        return None


async def test_grpc_handler_include_all_managed_returns_sidecars() -> None:
    # audit H11: the ListRawContainers handler returns the BROADER managed set (incl sidecars)
    # only when include_all_managed is set; container_ids stays raw-only in BOTH modes.
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    link = NodeGrpcLink(_FakeAgent(), control_addr="127.0.0.1:0")  # type: ignore[arg-type]

    raw_reply = await link._exec_list_raw_containers(
        pb.ListRawContainersCommand(backend="docker"),
    )
    kinds = {c.container_id: c.session_kind for c in raw_reply.list_raw_containers.containers}
    assert set(kinds) == {"c-main"}                       # raw-only: no sidecar
    assert list(raw_reply.list_raw_containers.container_ids) == ["c-main"]
    # a raw-only sweep does NOT ACK the broad capability.
    assert raw_reply.list_raw_containers.all_managed_supported is False

    managed_reply = await link._exec_list_raw_containers(
        pb.ListRawContainersCommand(backend="docker", include_all_managed=True),
    )
    kinds = {c.container_id: c.session_kind for c in managed_reply.list_raw_containers.containers}
    assert kinds == {"c-main": "raw", "c-side": "compose"}   # sidecar included, tagged
    # container_ids MUST stay raw-only even in the broader mode (raw-GC diff depends on it).
    assert list(managed_reply.list_raw_containers.container_ids) == ["c-main"]
    # a node that honoured include_all_managed ACKs the capability (audit H11).
    assert managed_reply.list_raw_containers.all_managed_supported is True
