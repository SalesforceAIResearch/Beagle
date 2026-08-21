"""RawContainerManager.apply_egress — node-side egress restriction (spec 07).

Recording fake enforcer, no nsenter/iptables, no daemon. Pins the safety
contract: ownership, private-netns-only, fail-closed, block-all on empty.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from xrlenv.backends.egress import (
    EgressAllowlist,
    EgressApplyError,
    EgressEnforcer,
    EgressProgram,
    EgressRule,
    compile_egress_rules,
)
from xrlenv.errors import XRLEnvError
from xrlenv.node.raw_container import RawContainerManager, RawContainerRecord


class _RecordingEnforcer(EgressEnforcer):
    def __init__(self) -> None:
        self.calls: list[tuple[int, EgressProgram]] = []

    async def apply(self, *, container_pid: int, program: EgressProgram) -> None:
        self.calls.append((container_pid, program))


class _FailingEnforcer(EgressEnforcer):
    async def apply(self, *, container_pid: int, program: EgressProgram) -> None:
        raise EgressApplyError("iptables -A OUTPUT failed (rc=1)")


class _FakeContainer:
    def __init__(
        self,
        cid: str,
        pid: int,
        network_mode: str = "bridge",
        *,
        cap_add: list[str] | None = None,
        privileged: bool = False,
    ) -> None:
        self.id = cid
        self.name = f"c-{cid}"
        self.attrs: dict[str, Any] = {
            "State": {"Pid": pid},
            "HostConfig": {
                "NetworkMode": network_mode,
                "CapAdd": cap_add,
                "Privileged": privileged,
            },
        }

    def reload(self) -> None: ...

    def remove(self, *, force: bool = False) -> None: ...


class _FakeClient:
    def __init__(self, c: _FakeContainer) -> None:
        self._c = c
        self.containers = self

    def get(self, container_id: str) -> _FakeContainer:
        if container_id != self._c.id:
            raise XRLEnvError(f"no such container: {container_id}")
        return self._c


def _mgr(
    *, pid: int = 4242, network_mode: str = "bridge",
    cap_add: list[str] | None = None, privileged: bool = False,
    enforcer: EgressEnforcer | None = None,
) -> tuple[RawContainerManager, EgressEnforcer]:
    enf = enforcer or _RecordingEnforcer()
    m = RawContainerManager(
        docker_client=_FakeClient(
            _FakeContainer(
                "cid-1", pid, network_mode,
                cap_add=cap_add, privileged=privileged,
            ),
        ),
        egress_enforcer=enf,
    )
    m._records["cid-1"] = RawContainerRecord(
        rollout_id="r1", container_id="cid-1", container_name="c-cid-1",
        image="img:1", created_at=datetime(2026, 1, 1),
    )
    return m, enf


_GW = EgressAllowlist(rules=(EgressRule(cidr="3.149.157.52/32", ports=(443,)),))


@pytest.mark.asyncio
async def test_applies_compiled_program() -> None:
    m, enf = _mgr()
    await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=_GW)
    assert len(enf.calls) == 1  # type: ignore[attr-defined]
    pid, program = enf.calls[0]  # type: ignore[attr-defined]
    assert pid == 4242
    assert program == compile_egress_rules(_GW)


@pytest.mark.asyncio
async def test_empty_allowlist_block_all() -> None:
    m, enf = _mgr()
    await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=EgressAllowlist())
    _, program = enf.calls[0]  # type: ignore[attr-defined]
    assert program == compile_egress_rules(EgressAllowlist())


@pytest.mark.asyncio
async def test_rejects_wrong_owner() -> None:
    m, enf = _mgr()
    with pytest.raises(XRLEnvError, match=r"not registered|does not"):
        await m.apply_egress(rollout_id="WRONG", container_id="cid-1", allowlist=_GW)
    assert enf.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refuses_host_netns() -> None:
    m, enf = _mgr(network_mode="host")
    with pytest.raises(XRLEnvError, match="host"):
        await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=_GW)
    assert enf.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refuses_shared_container_netns() -> None:
    m, enf = _mgr(network_mode="container:abc")
    with pytest.raises(XRLEnvError, match=r"netns|container:"):
        await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=_GW)
    assert enf.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refuses_privileged() -> None:
    m, enf = _mgr(privileged=True)
    with pytest.raises(XRLEnvError, match="privileged or holds CAP_NET_ADMIN"):
        await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=_GW)
    assert enf.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", ["NET_ADMIN", "CAP_NET_ADMIN", "net_admin"])
async def test_refuses_net_admin(cap: str) -> None:
    m, enf = _mgr(cap_add=["SYS_PTRACE", cap])
    with pytest.raises(XRLEnvError, match="privileged or holds CAP_NET_ADMIN"):
        await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=_GW)
    assert enf.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fails_when_no_live_pid() -> None:
    m, enf = _mgr(pid=0)
    with pytest.raises(XRLEnvError, match="no live"):
        await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=_GW)
    assert enf.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fail_closed_destroys_on_enforcer_failure() -> None:
    m, _enf = _mgr(enforcer=_FailingEnforcer())
    destroyed: list[tuple[str, str, bool]] = []

    async def _spy_destroy(
        *, rollout_id: str, container_id: str, force: bool = True,
        backend: str = "docker",
    ) -> None:
        destroyed.append((rollout_id, container_id, force))

    m.destroy = _spy_destroy  # type: ignore[method-assign]
    with pytest.raises(EgressApplyError, match="iptables"):
        await m.apply_egress(rollout_id="r1", container_id="cid-1", allowlist=_GW)
    assert destroyed == [("r1", "cid-1", True)]
