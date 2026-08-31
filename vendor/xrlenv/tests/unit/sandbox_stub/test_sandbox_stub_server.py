"""Tests for xrlenv/sandbox_stub/server.py.

Uses aiohttp's TestServer/TestClient to run the full aiohttp app in-process.
No Docker required — the adapter is injected via direct state manipulation or
a real in-process EnvAdapter subclass.
"""

from __future__ import annotations

from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from xrlenv.sandbox_stub.server import StubState, build_app
from xrlenv.types import StepResult

# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
async def client() -> TestClient:  # type: ignore[return]
    app = build_app()
    async with TestClient(TestServer(app)) as c:
        yield c


# ── /healthz ─────────────────────────────────────────────────────────────────


async def test_healthz_returns_ok(client: TestClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"


# ── /commands ────────────────────────────────────────────────────────────────


async def test_commands_echo(client: TestClient) -> None:
    resp = await client.post("/commands", json={"cmd": ["echo", "hi"], "cwd": "/tmp"})
    assert resp.status == 200
    body = await resp.json()
    assert body["exit_code"] == 0
    assert "hi" in body["stdout"]
    assert body["timed_out"] is False


async def test_commands_invalid_cmd_type_rejected(client: TestClient) -> None:
    resp = await client.post("/commands", json={"cmd": "not-a-list"})
    assert resp.status == 400


async def test_commands_non_zero_exit_code(client: TestClient) -> None:
    resp = await client.post("/commands", json={"cmd": ["false"], "cwd": "/tmp"})
    body = await resp.json()
    assert body["exit_code"] != 0
    assert body["timed_out"] is False


async def test_commands_timeout(client: TestClient) -> None:
    resp = await client.post(
        "/commands",
        json={"cmd": ["sleep", "10"], "timeout_s": 0.1, "cwd": "/tmp"},
    )
    body = await resp.json()
    assert body["timed_out"] is True
    assert body["exit_code"] == 124


# ── /files ────────────────────────────────────────────────────────────────────


async def test_write_and_read_file(client: TestClient, tmp_path: Any) -> None:
    # Write via stub's /files endpoint (writes to host filesystem)
    path = str(tmp_path / "hello.txt")
    write_resp = await client.post(f"/files{path}", data=b"hello world")
    assert write_resp.status == 200
    meta = await write_resp.json()
    assert meta["size"] == 11

    # Read it back
    read_resp = await client.get(f"/files{path}")
    assert read_resp.status == 200
    content = await read_resp.read()
    assert content == b"hello world"


async def test_read_nonexistent_file_returns_404(client: TestClient) -> None:
    resp = await client.get("/files/does/not/exist/at/all.txt")
    assert resp.status == 404


# ── /env/setup → /env/step → /env/teardown ───────────────────────────────────


class _EchoAdapter:
    """A minimal EnvAdapter that echoes the action back as obs."""

    supported_reward_modes: frozenset = frozenset({"env_step"})

    async def setup(self, init_params: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "echo.greeting", "params": init_params}

    async def step(self, action: Any) -> StepResult:
        return StepResult(obs={"echoed": action}, reward=0.0, done=False)

    async def teardown(self) -> None:
        pass

    @classmethod
    def capabilities(cls):  # type: ignore[no-untyped-def]
        from xrlenv.envs.base import AdapterCapabilities
        return AdapterCapabilities(
            xrlenv_api_version_supported=("0.0",),
            supported_reward_modes=cls.supported_reward_modes,
        )


@pytest.fixture
async def client_with_adapter() -> TestClient:  # type: ignore[return]
    """Client whose StubState already has an adapter pre-loaded."""
    app = build_app()
    state: StubState = app["state"]
    state.adapter = _EchoAdapter()  # type: ignore[assignment]
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_env_setup_missing_fields_rejected(client: TestClient) -> None:
    resp = await client.post("/env/setup", json={"adapter_module": "xrlenv.templates.hello_shell.adapter"})
    assert resp.status == 400


async def test_env_setup_bad_module_returns_500(client: TestClient) -> None:
    resp = await client.post(
        "/env/setup",
        json={"adapter_module": "nonexistent.module.xyz", "adapter_class": "Foo"},
    )
    assert resp.status == 500
    body = await resp.json()
    assert body["error"] == "adapter_import_failed"


async def test_env_step_before_setup_rejected(client: TestClient) -> None:
    resp = await client.post("/env/step", json={"action": "x"})
    assert resp.status == 400


async def test_env_step_returns_result(client_with_adapter: TestClient) -> None:
    resp = await client_with_adapter.post("/env/step", json={"action": {"cmd": "ls"}})
    assert resp.status == 200
    body = await resp.json()
    # _env_step wraps result.obs into {"obs": ...}; EchoAdapter.step returns
    # StepResult(obs={"echoed": action}, ...) so body["obs"] is {"echoed": action}.
    assert body["obs"] == {"echoed": {"cmd": "ls"}}


async def test_env_teardown_without_setup_is_noop(client: TestClient) -> None:
    resp = await client.post("/env/teardown", json={})
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "noop"


async def test_env_teardown_with_adapter_clears_state(
    client_with_adapter: TestClient,
) -> None:
    state: StubState = client_with_adapter.app["state"]
    assert state.adapter is not None

    resp = await client_with_adapter.post("/env/teardown", json={})
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"
    assert state.adapter is None


async def test_env_full_lifecycle_with_shell_adapter(client: TestClient) -> None:
    """Happy-path with the real ShellEnvAdapter loaded by the stub."""
    setup_resp = await client.post(
        "/env/setup",
        json={
            "adapter_module": "xrlenv.templates.hello_shell.adapter",
            "adapter_class": "ShellEnvAdapter",
            "init_params": {"cwd": "/tmp", "max_steps": 3},
            "sandbox_id": "test-sb",
        },
    )
    assert setup_resp.status == 200
    setup_body = await setup_resp.json()
    assert setup_body["obs"]["kind"] == "shell.greeting"

    step_resp = await client.post("/env/step", json={"action": {"cmd": "echo xrlenv"}})
    assert step_resp.status == 200
    step_body = await step_resp.json()
    assert "xrlenv" in step_body["obs"]["stdout"]

    teardown_resp = await client.post("/env/teardown", json={})
    assert teardown_resp.status == 200


# ──────────────────────────────────────────────────────────────────────────────
# UDS socket file permissions
# ──────────────────────────────────────────────────────────────────────────────


async def test_serve_forever_chmods_uds_world_rw() -> None:
    """The stub runs inside the container as ``USER sandbox`` (uid 1000) but
    the host node-agent that connects to the bind-mounted socket runs as
    the host's xrlenv system user (different uid). With default umask the
    bound socket is 0o644 → host hits ``Permission denied`` on
    ``connect()`` because connect requires write on the socket file.
    The stub must chmod 0o666 right after ``site.start()`` so any uid can
    connect.

    Regression for the failure surfaced during the Scenario-1 acceptance
    smoke (stub bound, but ``UnixClientConnectorError: ... [Permission
    denied]`` on the next env_setup call).
    """
    import asyncio
    import os
    import shutil
    import stat
    import tempfile
    from pathlib import Path

    from xrlenv.sandbox_stub.server import StubServer

    # macOS ``sun_path`` is capped at 104 chars; pytest's tmp_path can
    # blow past that under /private/var/folders. Use a short /tmp dir.
    short_dir = Path(tempfile.mkdtemp(prefix="xstub-", dir="/tmp"))
    uds_path = short_dir / "s.sock"
    try:
        server = StubServer(uds_path=str(uds_path))
        serve_task = asyncio.create_task(server.serve_forever())
        # Wait briefly for ``site.start()`` + ``os.chmod`` to complete.
        for _ in range(100):
            if uds_path.exists():
                break
            await asyncio.sleep(0.02)
        try:
            assert uds_path.exists(), "stub never created the uds path"
            mode = stat.S_IMODE(os.stat(uds_path).st_mode)
            assert mode == 0o666, (
                f"socket mode is {oct(mode)}; expected 0o666 so cross-uid "
                "connect() works. Without this, the host node-agent gets "
                "'Permission denied' connecting to the bind-mounted socket."
            )
        finally:
            serve_task.cancel()
            from contextlib import suppress

            with suppress(asyncio.CancelledError, Exception):
                await serve_task
    finally:
        shutil.rmtree(short_dir, ignore_errors=True)
