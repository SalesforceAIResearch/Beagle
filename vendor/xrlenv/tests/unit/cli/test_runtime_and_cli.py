"""Tests for build_local_runtime (runtime.py) and the CLI (cli/__main__.py).

build_local_runtime requires a Docker connection to instantiate DockerBackend;
we avoid that by passing a tmp runs_root and catching the connection error,
OR we mock the DockerBackend constructor.  The test asserts on the runtime
shape (components wired correctly, hello-shell template registered) without
actually connecting to Docker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from xrlenv.backends.base import ExecResult, ResourceSpec, ResourceUsage, SandboxHandle
from xrlenv.cli.__main__ import main
from xrlenv.client.client import Client
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.runtime import LocalRuntime, build_local_runtime
from xrlenv.control.scheduler import Placement
from xrlenv.control.service import CoordinatorRolloutService
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.control.trajectory_sink import PlatformJsonlSink
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.types import RolloutStatus

# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    # argparse's `--version` exits via SystemExit(0).
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip()


def test_cli_version_short_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["-V"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip()


def test_cli_help_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "xrlenv" in captured.out.lower()


def test_cli_no_args_errors_without_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """The Slice 5b CLI requires a subcommand; argparse exits SystemExit(2)."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    # argparse writes the "required" error to stderr.
    assert "required" in captured.err.lower() or "subcommand" in captured.err.lower()


# ── build_local_runtime ────────────────────────────────────────────────────────


def _fake_docker_backend_cls(config: object) -> MagicMock:
    """Returns a MagicMock that quacks like DockerBackend."""
    mock = MagicMock()
    mock.name = "docker"
    return mock


@pytest.fixture
def tmp_runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def test_build_local_runtime_wires_components(tmp_runs_root: Path) -> None:
    """Verify LocalRuntime has all expected components after build_local_runtime."""
    with patch("xrlenv.control.runtime.DockerBackend") as MockDockerBackend:
        MockDockerBackend.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(runs_root=tmp_runs_root)

    assert isinstance(runtime, LocalRuntime)
    assert runtime.state is not None
    assert runtime.catalog is not None
    assert runtime.node is not None
    assert runtime.scheduler is not None
    assert runtime.coordinator is not None
    assert runtime.service is not None


def test_build_local_runtime_registers_hello_shell(tmp_runs_root: Path) -> None:
    with patch("xrlenv.control.runtime.DockerBackend") as MockDockerBackend:
        MockDockerBackend.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(runs_root=tmp_runs_root)

    names = [m.name for m in runtime.catalog.list()]
    assert "hello-shell" in names


def test_build_local_runtime_custom_template_dir(
    tmp_runs_root: Path, tmp_path: Path, hello_shell_manifest_path: Path
) -> None:
    """Passing an explicit template_dirs list registers only those templates."""
    import shutil

    custom_dir = tmp_path / "my-templates" / "hello-shell"
    custom_dir.mkdir(parents=True)
    shutil.copy(hello_shell_manifest_path, custom_dir / "manifest.yaml")

    with patch("xrlenv.control.runtime.DockerBackend") as MockDockerBackend:
        MockDockerBackend.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(
            runs_root=tmp_runs_root,
            template_dirs=[tmp_path / "my-templates"],
        )

    names = [m.name for m in runtime.catalog.list()]
    assert "hello-shell" in names


async def test_runtime_shutdown_is_noop(tmp_runs_root: Path) -> None:
    with patch("xrlenv.control.runtime.DockerBackend") as MockDockerBackend:
        MockDockerBackend.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(runs_root=tmp_runs_root)

    # Should not raise.
    await runtime.shutdown()


async def test_local_runtime_starts_image_cache_sweep(tmp_runs_root: Path) -> None:
    """Issue #13: ``LocalRuntime.start()`` must launch the image-cache
    eviction sweep so an in-process consumer gets the same disk-pressure
    protection as the daemon-attached node-agent path. ``shutdown()``
    must cancel it cleanly.
    """
    with patch("xrlenv.control.runtime.DockerBackend") as MockDockerBackend:
        MockDockerBackend.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(runs_root=tmp_runs_root)

    assert runtime._image_sweep_task is None
    try:
        await runtime.start()
        task = runtime._image_sweep_task
        assert task is not None, "expected sweep task created on start()"
        assert not task.done(), "sweep task should still be running"
        # Yield once so the loop hits its first await sleep, otherwise
        # cancellation in shutdown() races the create_task scheduling.
        await asyncio.sleep(0)
    finally:
        await runtime.shutdown()

    assert runtime._image_sweep_task is None


async def test_build_local_runtime_metrics_port_lifecycle(
    tmp_runs_root: Path,
) -> None:
    """D2 from notes/deferred_audit_todos.md: when ``metrics_port`` is
    supplied, ``LocalRuntime`` must instantiate a ``MetricsServer``,
    bind on ``start()`` (kernel-assigned port when ``port=0`` so the
    test isn't flaky on a busy host), and release it on ``shutdown()``.

    Pin the contract so a refactor that splits the metrics wiring out
    of ``build_local_runtime`` can't silently regress. The default-no-
    metrics case is already covered by the existing
    ``test_build_local_runtime_wires_components`` test (which leaves
    ``metrics_port=None`` and asserts no surprise wiring).
    """
    import urllib.request

    with patch("xrlenv.control.runtime.DockerBackend") as MockDockerBackend:
        MockDockerBackend.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(
            runs_root=tmp_runs_root,
            metrics_port=0,  # kernel-assigned to keep the test parallel-safe
        )

    assert runtime.metrics_server is not None
    try:
        await runtime.start()
        bound_port = runtime.metrics_server.port
        assert bound_port > 0, (
            f"metrics_server.port should reflect the kernel-assigned port "
            f"after start(); got {bound_port}"
        )
        # Smoke-check the server actually serves /metrics. We don't
        # assert payload contents — just that the bind succeeded and
        # the response is HTTP 200.
        with urllib.request.urlopen(
            f"http://127.0.0.1:{bound_port}/metrics", timeout=5.0,
        ) as resp:
            assert resp.status == 200
    finally:
        await runtime.shutdown()

    # After shutdown the underlying server handle is released; a
    # subsequent stop() must be a clean no-op (idempotent).
    runtime.metrics_server.stop()


# ── template_catalog.register_dir corner cases ────────────────────────────────


def test_register_dir_no_manifests_returns_empty(tmp_path: Path) -> None:
    from xrlenv.control.template_catalog import TemplateCatalog

    cat = TemplateCatalog()
    (tmp_path / "subdir").mkdir()
    # No manifest.yaml anywhere — should return an empty list.
    result = cat.register_dir(tmp_path)
    assert result == []


def test_register_dir_nested_dirs(tmp_path: Path, hello_shell_manifest_path: Path) -> None:
    """register_dir must recurse into subdirectories (rglob behaviour)."""
    import shutil

    from xrlenv.control.template_catalog import TemplateCatalog

    nested = tmp_path / "a" / "b" / "hello-shell"
    nested.mkdir(parents=True)
    shutil.copy(hello_shell_manifest_path, nested / "manifest.yaml")

    cat = TemplateCatalog()
    registered = cat.register_dir(tmp_path)
    names = [m.name for m in registered]
    assert "hello-shell" in names


# ── Client.in_process round-trip ─────────────────────────────────────────────


async def test_client_in_process_round_trip() -> None:
    """Client.in_process wires to an InProcessTransport; verify a full
    start/step/finish cycle against a fake RolloutService."""
    from xrlenv.client.client import Client
    from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse
    from xrlenv.types import RolloutStatus, StepResult, Trajectory

    class FakeService:
        async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
            return StartRolloutResponse(rollout_id="fake-r", init_obs={"hi": True})

        async def step(self, rollout_id: str, action: object) -> StepResult:
            return StepResult(obs={"done": True}, reward=1.0, done=True)

        async def finish(self, rollout_id: str) -> Trajectory:
            return Trajectory(
                rollout_id=rollout_id,
                template="t",
                steps=[],
                status=RolloutStatus.FINISHED,
                reason=None,
                final_reward=1.0,
            )

        async def cancel(self, rollout_id: str, reason: str) -> Trajectory:
            return Trajectory(
                rollout_id=rollout_id,
                template="t",
                steps=[],
                status=RolloutStatus.CANCELLED,
                reason=reason,
                final_reward=0.0,
            )

    svc = FakeService()
    client = Client.in_process(svc)

    session = await client.rollout("t")
    assert session.observation == {"hi": True}

    result = await session.step("action")
    assert result.done is True

    async with session:
        pass  # session already done; triggers finish() on __aexit__

    assert session.trajectory.status == RolloutStatus.FINISHED


# ── Shared fake-node helpers for drain + replay tests ─────────────────────────


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _manifest(name: str = "t") -> TemplateManifest:
    return TemplateManifest(
        name=name, version="0.1", digest=f"sha256:{name}", image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
        hard_s_default=10.0,
    )


class _FakeNode:
    def __init__(self, node_id: str = "fake") -> None:
        self.node_id = node_id
        self._created = 0
        self._steps: dict[str, int] = {}
        self._max: dict[str, int] = {}

    def supported_backends(self) -> list[str]:
        return ["docker"]

    def hardware(self) -> HardwareInfo:
        return _hw()

    async def create_sandbox(self, **_: Any) -> SandboxHandle:
        self._created += 1
        sid = f"sb-{self._created}"
        return SandboxHandle(id=sid, backend="docker", backend_ref=f"c-{self._created}", stub_endpoint="tcp://127.0.0.1:0")

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        self._steps.pop(sb.id, None)
        self._max.pop(sb.id, None)

    async def env_setup(self, sb: SandboxHandle, *, adapter_module: str, adapter_class: str, init_params: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        self._max[sb.id] = int(init_params.get("max_steps") or 1)
        self._steps[sb.id] = 0
        return {"obs": {"kind": "first"}}

    async def env_step(self, sb: SandboxHandle, action: Any, **_kw: Any) -> dict[str, Any]:
        self._steps[sb.id] = self._steps.get(sb.id, 0) + 1
        done = self._steps[sb.id] >= self._max.get(sb.id, 1)
        return {"obs": {"step": self._steps[sb.id]}, "reward": 1.0, "done": done, "truncated": False, "info": {}}

    async def env_teardown(self, sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
        return {"status": "ok"}

    async def run_in_sandbox(self, sb: SandboxHandle, cmd: list[str], **_: Any) -> ExecResult:
        return ExecResult(exit_code=0)

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=0.0, rss_bytes=0, disk_bytes=0, rx_bytes=0, tx_bytes=0)

    async def query_image(self, _image: str) -> Any:
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _build_runtime_with_sink(
    runs_root: Path,
    *,
    default_idle_ttl_s: float = 120.0,
) -> tuple[Client, RolloutCoordinator, InMemoryStateStore, _FakeNode, AdmissionQueue]:
    node = _FakeNode()
    catalog = TemplateCatalog()
    catalog.register(_manifest())

    sched = MagicMock()
    sched.place.return_value = Placement(node=node, backend="docker", score=1)
    sched.nodes = [node]

    state = InMemoryStateStore()
    sink = PlatformJsonlSink(runs_root)
    admission = AdmissionQueue(scheduler=sched, state=state)
    coordinator = RolloutCoordinator(
        catalog=catalog,
        scheduler=sched,
        state=state,
        trajectory_sink=sink,
        admission=admission,
        default_idle_ttl_s=default_idle_ttl_s,
    )
    service = CoordinatorRolloutService(coordinator)
    client = Client.in_process(service)
    return client, coordinator, state, node, admission


# ── Gap 1: LocalRuntime.shutdown() drain-with-grace ───────────────────────────


async def test_shutdown_drains_when_rollout_finishes_before_grace(
    tmp_path: Path,
) -> None:
    """If the in-flight rollout terminates before drain_timeout_s, shutdown
    returns without waiting the full grace period.
    """
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    client, coord, state, _node, admission = _build_runtime_with_sink(runs_root)

    session = await client.rollout(template="t", init={"max_steps": 1})

    async def _drive() -> None:
        await session.step({})
        async with session:
            pass

    driver = asyncio.create_task(_drive())

    with patch("xrlenv.control.runtime.DockerBackend") as _mock:
        _mock.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(
            runs_root=runs_root,
            state=state,
        )
    runtime.coordinator = coord
    runtime.admission = admission
    runtime.drain_timeout_s = 5.0

    await driver

    assert all(r.status.is_terminal for r in state.list_rollouts())

    await runtime.shutdown()

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_shutdown_cancels_pending_and_tears_down_watchers_when_grace_expires(
    tmp_path: Path,
) -> None:
    """When a rollout is still running at shutdown, the drain loop times out and
    cancel_pending + watcher teardown still run — shutdown must not wedge.

    D7 strengthened (2026-04-29): also pin the in-flight rollout's
    final state. The phase-0 contract is that ``LocalRuntime.shutdown``
    does NOT seal still-running rollouts as ``cancelled`` /
    ``failed`` / ``truncated``; they stay in ``RUNNING`` so the
    next-process recovery loop can reclaim them. A future refactor
    that changed that without warning would silently turn surviving
    rollouts into orphans on every process bounce — we want the test
    to fail loudly if that contract changes.
    """
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    client, coord, state, _node, admission = _build_runtime_with_sink(runs_root)

    session = await client.rollout(template="t", init={"max_steps": 100})
    in_flight_id = session.rollout_id
    assert any(r.status == RolloutStatus.RUNNING for r in state.list_rollouts())

    with patch("xrlenv.control.runtime.DockerBackend") as _mock:
        _mock.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(runs_root=runs_root, state=state)
    runtime.coordinator = coord
    runtime.admission = admission
    runtime.drain_timeout_s = 0.1

    await asyncio.wait_for(runtime.shutdown(), timeout=3.0)

    # Pin the surviving-rollout contract: the in-flight rollout
    # must NOT have been auto-sealed by shutdown. It stays in
    # RUNNING for next-process recovery (spec 03 + spec 20).
    rec = state.get_rollout(in_flight_id)
    assert rec.status == RolloutStatus.RUNNING, (
        f"shutdown must not seal in-flight rollouts; got "
        f"status={rec.status!r}"
    )

    await client.close()


# ── Gap 2: Client.replay() end-to-end ─────────────────────────────────────────


async def test_client_replay_returns_sealed_trajectory(tmp_path: Path) -> None:
    """Client.replay() → coordinator → sink.read() returns the full trajectory."""
    runs_root = tmp_path / "runs"
    client, coord, _state, _node, _admission = _build_runtime_with_sink(runs_root)

    session = await client.rollout(template="t", init={"max_steps": 2})
    rollout_id = session.rollout_id

    await session.step({})
    await session.step({})
    async with session:
        pass

    traj = await client.replay(rollout_id)
    assert traj.rollout_id == rollout_id
    assert traj.status == RolloutStatus.FINISHED
    assert len(traj.steps) == 2
    assert traj.final_reward == pytest.approx(2.0)

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


async def test_client_replay_unknown_rollout_raises(tmp_path: Path) -> None:
    """Replaying an id that was never started raises KeyError (no state row)."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    client, coord, _state, _node, _admission = _build_runtime_with_sink(runs_root)

    with pytest.raises((KeyError, FileNotFoundError)):
        await client.replay("rollout-that-never-existed")

    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()
