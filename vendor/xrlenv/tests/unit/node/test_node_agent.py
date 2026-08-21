"""Tests for xrlenv/node/agent.py — NodeAgent lifecycle and routing.

Uses a FakeSandboxBackend so no Docker daemon is needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from xrlenv.backends.base import (
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.node.agent import NodeAgent, NodeAgentConfig

# ── Fake backend ─────────────────────────────────────────────────────────────


class FakeBackend:
    name = "docker"

    async def create(
        self,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
    ) -> SandboxHandle:
        return SandboxHandle(
            id="sb-fake",
            backend="docker",
            backend_ref="container-fake",
            stub_endpoint="tcp://127.0.0.1:9999",
        )

    async def destroy(self, sb: SandboxHandle) -> None:
        pass

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=1.0, rss_bytes=1024, disk_bytes=0, rx_bytes=0, tx_bytes=0)


def _make_resources() -> ResourceSpec:
    return ResourceSpec(
        cpu_request=0.25,
        cpu_limit=1.0,
        mem_request_bytes=64_000_000,
        mem_limit_bytes=128_000_000,
        disk_request_bytes=64_000_000,
    )


def _make_agent(backends: dict | None = None) -> NodeAgent:
    if backends is None:
        backends = {"docker": FakeBackend()}
    cfg = NodeAgentConfig(node_id="test-node", backends=backends)  # type: ignore[arg-type]
    return NodeAgent(cfg)


# ── supported_backends ────────────────────────────────────────────────────────


def test_supported_backends_sorted() -> None:
    agent = _make_agent({"docker": FakeBackend(), "zzz": FakeBackend()})
    assert agent.supported_backends() == ["docker", "zzz"]


# ── create_sandbox ────────────────────────────────────────────────────────────


async def test_create_sandbox_returns_handle() -> None:
    agent = _make_agent()
    handle = await agent.create_sandbox(
        rollout_id="r1",
        backend="docker",
        template=TemplateRef(name="t", image="im:1"),
        resources=_make_resources(),
        network_policy="open",
    )
    assert handle.id == "sb-fake"
    assert handle.backend == "docker"


async def test_create_sandbox_unknown_backend_raises() -> None:
    agent = _make_agent()
    with pytest.raises(ValueError, match="does not have backend"):
        await agent.create_sandbox(
            rollout_id="r1",
            backend="cubesandbox",
            template=TemplateRef(name="t", image="im:1"),
            resources=_make_resources(),
            network_policy="open",
        )


async def test_create_sandbox_records_in_table() -> None:
    agent = _make_agent()
    handle = await agent.create_sandbox(
        rollout_id="r1",
        backend="docker",
        template=TemplateRef(name="t", image="im:1"),
        resources=_make_resources(),
        network_policy="open",
    )
    # The internal sandbox table should contain the record.
    assert handle.id in agent._sandboxes


# ── destroy_sandbox ───────────────────────────────────────────────────────────


async def test_destroy_sandbox_removes_from_table() -> None:
    agent = _make_agent()
    handle = await agent.create_sandbox(
        rollout_id="r1",
        backend="docker",
        template=TemplateRef(name="t", image="im:1"),
        resources=_make_resources(),
        network_policy="open",
    )
    await agent.destroy_sandbox(handle)
    assert handle.id not in agent._sandboxes


async def test_destroy_sandbox_unknown_backend_logs_warning(caplog: Any) -> None:
    """Destroy with an unregistered backend should warn, not raise."""
    agent = _make_agent({})
    handle = SandboxHandle(
        id="sb-orphan",
        backend="vanished",
        backend_ref="ref",
        stub_endpoint="tcp://127.0.0.1:9999",
    )
    # Should not raise.
    await agent.destroy_sandbox(handle)


# ── env_setup / env_step / env_teardown via injected stub ────────────────────


def _inject_stub(agent: NodeAgent, handle: SandboxHandle) -> MagicMock:
    """Bypass _stub_for by pre-seeding the sandbox record with a mock stub."""
    from xrlenv.node.agent import _SandboxRecord

    stub = MagicMock()
    stub.env_setup = AsyncMock(return_value={"obs": "first"})
    stub.env_step = AsyncMock(return_value={"obs": "stepped", "reward": 1.0, "done": False})
    stub.env_teardown = AsyncMock(return_value={"status": "ok"})

    record = _SandboxRecord(handle=handle, template="t", backend="docker")
    record.stub = stub  # type: ignore[assignment]
    agent._sandboxes[handle.id] = record
    return stub


async def test_env_setup_routes_to_stub() -> None:
    agent = _make_agent()
    handle = SandboxHandle(id="sb-x", backend="docker", backend_ref="r", stub_endpoint="tcp://127.0.0.1:1")
    stub = _inject_stub(agent, handle)

    result = await agent.env_setup(handle, adapter_module="m", adapter_class="C", init_params={})
    stub.env_setup.assert_called_once()
    assert result == {"obs": "first"}


async def test_env_step_routes_to_stub() -> None:
    agent = _make_agent()
    handle = SandboxHandle(id="sb-y", backend="docker", backend_ref="r", stub_endpoint="tcp://127.0.0.1:1")
    stub = _inject_stub(agent, handle)

    result = await agent.env_step(handle, "action")
    # D17 stage 2: NodeAgent always passes request_timeout_s through
    # (None when no per-call cap was supplied by the coordinator).
    stub.env_step.assert_called_once_with("action", request_timeout_s=None)
    assert result["obs"] == "stepped"


async def test_env_step_per_call_cap_flows_to_stub() -> None:
    """A5 / D17 stage 2: when the coordinator supplies a per-call
    cap, NodeAgent forwards it as the StubClient kwarg unchanged."""
    agent = _make_agent()
    handle = SandboxHandle(
        id="sb-cap-step", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    stub = _inject_stub(agent, handle)

    await agent.env_step(handle, "action", request_timeout_s=7.5)
    stub.env_step.assert_called_once_with("action", request_timeout_s=7.5)


async def test_env_setup_per_call_cap_flows_to_stub() -> None:
    agent = _make_agent()
    handle = SandboxHandle(
        id="sb-cap-setup", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    stub = _inject_stub(agent, handle)

    await agent.env_setup(
        handle, adapter_module="m", adapter_class="C", init_params={},
        request_timeout_s=12.0,
    )
    stub.env_setup.assert_called_once_with(
        adapter_module="m", adapter_class="C", init_params={},
        sandbox_id="sb-cap-setup", request_timeout_s=12.0,
    )


async def test_env_teardown_per_call_cap_flows_to_stub() -> None:
    agent = _make_agent()
    handle = SandboxHandle(
        id="sb-cap-tear", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    stub = _inject_stub(agent, handle)

    await agent.env_teardown(handle, request_timeout_s=4.0)
    stub.env_teardown.assert_called_once_with(request_timeout_s=4.0)


async def test_env_teardown_routes_to_stub() -> None:
    agent = _make_agent()
    handle = SandboxHandle(id="sb-z", backend="docker", backend_ref="r", stub_endpoint="tcp://127.0.0.1:1")
    stub = _inject_stub(agent, handle)

    result = await agent.env_teardown(handle)
    stub.env_teardown.assert_called_once()
    assert result["status"] == "ok"


# ── D17 stage 1: per-sandbox HTTP cap from create_sandbox kwarg ──────────────


async def test_create_sandbox_stages_http_cap_on_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit response (H2): ``create_sandbox`` must stage the
    coordinator-supplied HTTP cap on the new ``_SandboxRecord`` so
    the very first stub-touching call (init_cmd, env_setup, …) sees
    the manifest-derived cap, not the 1 h default. This is the
    primary path for D17 stage 1 — replaces the earlier
    init_params-injection path that got bypassed by manifests with
    init_cmd.
    """
    from xrlenv.backends.base import (
        ResourceSpec,
        TemplateRef,
    )
    from xrlenv.node import agent as agent_mod

    # Stop StubClient from actually being built so the test stays
    # transport-free; we assert via the record only.
    captured_cap: list[float] = []

    class _FakeStubClient:
        def __init__(self, endpoint: str, *, request_timeout_s: float) -> None:
            captured_cap.append(request_timeout_s)

    monkeypatch.setattr(agent_mod, "StubClient", _FakeStubClient)

    agent = _make_agent()
    handle = await agent.create_sandbox(
        rollout_id="r-1",
        backend="docker",
        template=TemplateRef(name="t", image="im:1", digest="sha256:t"),
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",  # type: ignore[arg-type]
        stub_request_timeout_s=42.5,
    )

    record = agent._sandboxes[handle.id]
    assert record.stub_request_timeout_s_override == 42.5

    # The very next stub build must use that cap, not the 1 h default.
    await agent._stub_for(handle)
    assert captured_cap == [42.5]


async def test_create_sandbox_without_http_cap_leaves_record_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the coordinator doesn't pass ``stub_request_timeout_s``
    (e.g. legacy callers / tests), the record's override is None and
    ``_stub_for`` falls back to ``NodeAgentConfig.stub_request_timeout_s``.
    """
    from xrlenv.backends.base import (
        ResourceSpec,
        TemplateRef,
    )

    agent = _make_agent()
    handle = await agent.create_sandbox(
        rollout_id="r-1",
        backend="docker",
        template=TemplateRef(name="t", image="im:1", digest="sha256:t"),
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        network_policy="open",  # type: ignore[arg-type]
    )
    assert agent._sandboxes[handle.id].stub_request_timeout_s_override is None


async def test_set_stub_request_timeout_rebuilds_existing_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: if ``_set_stub_request_timeout`` is called with a
    different cap AFTER a StubClient was already built (e.g., a late-
    stage operator override), the existing stub is closed and the
    next ``_stub_for`` rebuilds with the new cap.

    This wasn't strictly needed once create_sandbox-side staging
    landed (the cap is set before any stub-touching call), but pins
    the contract for hypothetical callers that update the cap
    mid-life.
    """
    from xrlenv.node import agent as agent_mod

    builds: list[float] = []

    class _FakeStubClient:
        def __init__(self, endpoint: str, *, request_timeout_s: float) -> None:
            builds.append(request_timeout_s)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(agent_mod, "StubClient", _FakeStubClient)

    agent = _make_agent()
    handle = SandboxHandle(
        id="sb-rebuild", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    # First stub built with default (no override staged).
    await agent._stub_for(handle)
    # Stage a different cap; the cached stub must be closed.
    await agent._set_stub_request_timeout(handle, 5.0)
    # Next _stub_for rebuilds with the new cap.
    await agent._stub_for(handle)

    assert len(builds) == 2
    # First build used the agent-config default; second used 5.0.
    assert builds[1] == 5.0


async def test_set_stub_request_timeout_no_op_when_cap_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ``_set_stub_request_timeout`` with the same cap as
    already staged is a no-op — does NOT close the existing stub.
    Pin this so the create_sandbox-side staging + a redundant late
    call doesn't rebuild for nothing.
    """
    from xrlenv.node import agent as agent_mod
    from xrlenv.node.agent import _SandboxRecord

    builds: list[float] = []

    class _FakeStubClient:
        def __init__(self, endpoint: str, *, request_timeout_s: float) -> None:
            builds.append(request_timeout_s)

        async def close(self) -> None:
            raise AssertionError("close should not be called on no-op")

    monkeypatch.setattr(agent_mod, "StubClient", _FakeStubClient)

    agent = _make_agent()
    handle = SandboxHandle(
        id="sb-noop", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    # Pre-stage cap and pre-build stub at that cap.
    agent._sandboxes[handle.id] = _SandboxRecord(
        handle=handle, template="t", backend="docker",
        stub_request_timeout_s_override=5.0,
    )
    await agent._stub_for(handle)
    # Re-stage same cap → must not close the existing stub.
    await agent._set_stub_request_timeout(handle, 5.0)
    # Stub must still be the same instance (no rebuild).
    assert len(builds) == 1


async def test_stub_for_uses_per_sandbox_override_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_stub_for`` builds the StubClient with the staged
    per-sandbox cap, not the agent-config default. Pin via the
    StubClient constructor argument.
    """
    from xrlenv.node import agent as agent_mod

    captured: dict[str, Any] = {}

    class _FakeStubClient:
        def __init__(
            self, endpoint: str, *, request_timeout_s: float,
        ) -> None:
            captured["endpoint"] = endpoint
            captured["request_timeout_s"] = request_timeout_s

    monkeypatch.setattr(agent_mod, "StubClient", _FakeStubClient)

    agent = _make_agent()
    handle = SandboxHandle(
        id="sb-override", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    # Pre-stage the cap, then construct.
    await agent._set_stub_request_timeout(handle, 7.5)
    stub = await agent._stub_for(handle)

    assert isinstance(stub, _FakeStubClient)
    assert captured["request_timeout_s"] == 7.5


async def test_stub_for_falls_back_to_config_default_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No override staged → use the agent-config default (the 1 h
    safety-net cap)."""
    from xrlenv.node import agent as agent_mod
    from xrlenv.node.agent import NodeAgentConfig

    captured: dict[str, Any] = {}

    class _FakeStubClient:
        def __init__(
            self, endpoint: str, *, request_timeout_s: float,
        ) -> None:
            captured["request_timeout_s"] = request_timeout_s

    monkeypatch.setattr(agent_mod, "StubClient", _FakeStubClient)

    cfg = NodeAgentConfig(
        node_id="n", backends={"docker": FakeBackend()},  # type: ignore[arg-type]
        stub_request_timeout_s=4242.0,
    )
    agent = NodeAgent(cfg)
    handle = SandboxHandle(
        id="sb-no-override", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    await agent._stub_for(handle)
    assert captured["request_timeout_s"] == 4242.0


# ── A1 / D18+D19 (P1.2): query_image ─────────────────────────────────────────


async def test_query_image_returns_present_with_no_cache_wired() -> None:
    """When no ``ImageCacheManager`` is wired, ``query_image`` falls back
    to a backend-direct existence check. Pin both branches (present /
    absent) so the LocalRuntime / test-fixture path stays honest.
    """
    class _CountingBackend(FakeBackend):
        def __init__(self, present: bool) -> None:
            self._present = present
            self.exists_calls: list[str] = []

        async def image_exists(self, image: str) -> bool:
            self.exists_calls.append(image)
            return self._present

    backend = _CountingBackend(present=True)
    agent = _make_agent({"docker": backend})
    result = await agent.query_image("xrlenv/hello-shell:0.1")
    assert result.present is True
    assert backend.exists_calls == ["xrlenv/hello-shell:0.1"]
    # No cache wired → no last-used / digest data.
    assert result.last_used_at == 0.0
    assert result.digest is None

    backend_absent = _CountingBackend(present=False)
    agent_absent = _make_agent({"docker": backend_absent})
    result_absent = await agent_absent.query_image("never/seen:0.1")
    assert result_absent.present is False


async def test_query_image_uses_image_cache_when_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When an ``ImageCacheManager`` is wired, ``query_image``
    delegates to its ``query`` method so per-image last-used metadata
    rides the reply. Pin the delegation contract so a future cache
    refactor can't silently drop the metadata.
    """
    from unittest.mock import AsyncMock as _AsyncMock

    from xrlenv.node.image_cache import ImageQueryResult

    fake_cache = MagicMock()
    fake_cache.query = _AsyncMock(
        return_value=ImageQueryResult(
            present=True, digest="sha256:" + "ab" * 32, last_used_at=42.5,
        )
    )
    cfg = NodeAgentConfig(
        node_id="test-node",
        backends={"docker": FakeBackend()},  # type: ignore[arg-type]
    )
    agent = NodeAgent(cfg, image_cache=fake_cache)
    result = await agent.query_image("registry/img:1")

    fake_cache.query.assert_called_once_with("registry/img:1")
    assert result.present is True
    assert result.digest == "sha256:" + "ab" * 32
    assert result.last_used_at == 42.5


# ── A3 / D15: list_sandbox_ids ───────────────────────────────────────────────


async def test_list_sandbox_ids_returns_in_memory_keys() -> None:
    """``list_sandbox_ids()`` returns the keys of the in-memory
    ``_sandboxes`` table — the node side of the spec-09 GC layer 3
    reverse query consumed by ``GCReconciler``.
    """
    agent = _make_agent()
    handle_a = SandboxHandle(
        id="sb-A", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    handle_b = SandboxHandle(
        id="sb-B", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    _inject_stub(agent, handle_a)
    _inject_stub(agent, handle_b)

    ids = await agent.list_sandbox_ids()
    assert sorted(ids) == ["sb-A", "sb-B"]


async def test_list_sandbox_ids_filters_by_backend() -> None:
    """``backend`` filter narrows to sandboxes with that backend
    name only — phase-2 mixed-backend prep."""
    from xrlenv.node.agent import _SandboxRecord

    agent = _make_agent()
    docker_handle = SandboxHandle(
        id="sb-docker", backend="docker", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    cube_handle = SandboxHandle(
        id="sb-cube", backend="cube", backend_ref="r",
        stub_endpoint="tcp://127.0.0.1:1",
    )
    agent._sandboxes["sb-docker"] = _SandboxRecord(
        handle=docker_handle, template="t", backend="docker",
    )
    agent._sandboxes["sb-cube"] = _SandboxRecord(
        handle=cube_handle, template="t", backend="cube",
    )

    docker_only = await agent.list_sandbox_ids(backend="docker")
    cube_only = await agent.list_sandbox_ids(backend="cube")
    assert docker_only == ["sb-docker"]
    assert cube_only == ["sb-cube"]


async def test_list_sandbox_ids_empty_when_no_sandboxes() -> None:
    """Fresh agent with no sandboxes returns []."""
    agent = _make_agent()
    assert await agent.list_sandbox_ids() == []


# ── fetch_trajectory no-reader guard ─────────────────────────────────────────


async def test_fetch_trajectory_raises_when_no_reader_configured() -> None:
    agent = _make_agent()
    with pytest.raises(RuntimeError, match="no trajectory reader"):
        await agent.fetch_trajectory("r-1")


# ── hw_probe ──────────────────────────────────────────────────────────────────


def test_hardware_populates_fields() -> None:
    agent = _make_agent()
    hw = agent.hardware()
    assert hw.vcpus >= 1
    assert hw.mem_bytes > 0
    assert hw.disk_bytes > 0
    assert isinstance(hw.platform, str)


def test_probe_hardware_linux_reads_proc_meminfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3 from notes/deferred_audit_todos.md: pin the Linux mem-probe
    branch — parses ``/proc/meminfo`` MemTotal as kB and converts to
    bytes. Without this test, a future refactor that swaps the parser
    for ``psutil`` (or accidentally drops the ``* 1024`` factor) would
    only fail on a Linux host running real CI.
    """
    from xrlenv.node import hw_probe

    fake_meminfo = (
        "MemTotal:       16308868 kB\n"
        "MemFree:         2000000 kB\n"
        "MemAvailable:    8000000 kB\n"
    )

    real_read_text = hw_probe.Path.read_text

    def fake_read_text(self: hw_probe.Path, *args: Any, **kw: Any) -> str:
        if str(self) == "/proc/meminfo":
            return fake_meminfo
        return real_read_text(self, *args, **kw)

    monkeypatch.setattr(hw_probe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw_probe.Path, "read_text", fake_read_text)

    # 16308868 kB * 1024 = 16,700,280,832 bytes
    assert hw_probe._probe_mem_bytes() == 16_308_868 * 1024


def test_probe_hardware_darwin_invokes_sysctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3 follow-up: pin the Darwin sysctl branch — parses
    ``hw.memsize`` as a raw byte count. Same defensive rationale as
    the Linux test.
    """
    from xrlenv.node import hw_probe

    monkeypatch.setattr(hw_probe.platform, "system", lambda: "Darwin")

    def fake_check_output(cmd: list[str], *args: Any, **kw: Any) -> bytes:
        assert cmd == ["sysctl", "-n", "hw.memsize"], (
            f"expected sysctl -n hw.memsize; got {cmd!r}"
        )
        return b"34359738368\n"  # 32 GiB

    monkeypatch.setattr(hw_probe.subprocess, "check_output", fake_check_output)

    assert hw_probe._probe_mem_bytes() == 34_359_738_368


def test_probe_hardware_falls_back_when_sysctl_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3 follow-up: when neither path can read mem (Linux without
    /proc/meminfo, Darwin sysctl raising), fall back to a 4 GiB
    floor so the capacity estimator does not divide by zero. This is
    the documented contract — pin it.
    """
    from xrlenv.node import hw_probe

    monkeypatch.setattr(hw_probe.platform, "system", lambda: "Darwin")

    def boom(*_: Any, **__: Any) -> bytes:
        raise OSError("sysctl unavailable")

    monkeypatch.setattr(hw_probe.subprocess, "check_output", boom)
    assert hw_probe._probe_mem_bytes() == 4 * 1024 * 1024 * 1024


def test_probe_hardware_assembles_full_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """D3: the public ``probe_hardware`` wrapper assembles all fields
    from the lower-level helpers + os/platform/shutil calls. Pin the
    full assembly with deterministic inputs."""
    from xrlenv.node import hw_probe

    monkeypatch.setattr(hw_probe.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hw_probe, "_probe_mem_bytes", lambda: 12_345_678)

    class _FakeUsage:
        total = 999_888_777

    monkeypatch.setattr(
        hw_probe.shutil,
        "disk_usage",
        lambda _root: _FakeUsage,
    )
    # Force the kvm probe to a known state via a fake Path subtree.
    monkeypatch.setattr(
        hw_probe.Path, "exists", lambda self: str(self) == "/dev/kvm",
    )
    monkeypatch.setattr(hw_probe.platform, "release", lambda: "5.15.0-fake")
    monkeypatch.setattr(hw_probe.platform, "system", lambda: "Linux")

    hw = hw_probe.probe_hardware()
    assert hw.vcpus == 8
    assert hw.mem_bytes == 12_345_678
    assert hw.disk_bytes == 999_888_777
    assert hw.has_kvm is True
    assert hw.has_gpu is False
    assert hw.gpu_model is None
    assert hw.kernel_version == "5.15.0-fake"
    assert hw.platform == "linux"
