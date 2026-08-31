"""Tests for ShellEnvAdapter (runs without Docker — exercises the adapter
directly inside the test process).
"""

from __future__ import annotations

import pytest
from xrlenv.templates.hello_shell.adapter import ShellEnvAdapter


@pytest.fixture
async def adapter(tmp_path):  # type: ignore[no-untyped-def]
    a = ShellEnvAdapter()
    await a.setup({"cwd": str(tmp_path), "max_steps": 2})
    yield a
    await a.teardown()


async def test_setup_returns_greeting(tmp_path) -> None:  # type: ignore[no-untyped-def]
    a = ShellEnvAdapter()
    obs = await a.setup({"cwd": str(tmp_path)})
    assert obs["kind"] == "shell.greeting"
    assert obs["cwd"] == str(tmp_path)


async def test_step_runs_command(adapter: ShellEnvAdapter) -> None:
    result = await adapter.step({"cmd": "echo hello"})
    assert result.obs["exit_code"] == 0
    assert "hello" in result.obs["stdout"]
    assert result.done is False  # only 1 of max_steps=2


async def test_step_done_after_max_steps(adapter: ShellEnvAdapter) -> None:
    await adapter.step({"cmd": "echo step1"})
    result = await adapter.step({"cmd": "echo step2"})
    assert result.done is True


async def test_string_action_is_accepted(adapter: ShellEnvAdapter) -> None:
    result = await adapter.step("echo plain-string")
    assert result.obs["exit_code"] == 0


async def test_exit_sentinel_terminates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    a = ShellEnvAdapter()
    await a.setup({"cwd": str(tmp_path), "max_steps": 100})
    result = await a.step({"__exit__": True})
    assert result.done is True


async def test_invalid_action_raises(adapter: ShellEnvAdapter) -> None:
    with pytest.raises(TypeError):
        await adapter.step(42)


async def test_step_timeout_marks_truncated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    a = ShellEnvAdapter()
    await a.setup({"cwd": str(tmp_path), "step_timeout_s": 0.1})
    result = await a.step({"cmd": "sleep 5"})
    assert result.truncated is True
    assert result.info["timed_out"] is True
