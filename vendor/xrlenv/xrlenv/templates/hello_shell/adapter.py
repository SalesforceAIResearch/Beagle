"""ShellEnvAdapter — generic shell-command env (spec 14).

Action: a shell command string (or a ``{cmd: str}`` dict for forwards-compat).
Observation: stdout/stderr text plus exit code.
Reward: delegated to the template's ``RewardContract``; the adapter itself
returns ``reward=0.0`` per step. ``done`` is signaled when a configurable
``max_steps`` is reached *or* when the agent issues the sentinel action
``{"__exit__": True}``.

Used for one-off custom envs and as the smoke-test adapter in Slice 1.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar

from xrlenv.envs.base import EnvAdapter
from xrlenv.types import Action, Observation, StepResult


class ShellEnvAdapter(EnvAdapter):
    """Run shell commands inside the sandbox; observe stdout/stderr/exit_code."""

    supported_reward_modes: ClassVar[frozenset[str]] = frozenset(
        {"in_sandbox_final", "consumer_final", "external_final", "env_step"}
    )

    def __init__(self) -> None:
        self._cwd: str = "/sandbox"
        self._env: dict[str, str] = {}
        self._steps_taken: int = 0
        self._max_steps: int | None = None
        self._timeout_s: float = 30.0

    async def setup(self, init_params: dict[str, Any]) -> Observation:
        self._cwd = str(init_params.get("cwd") or "/sandbox")
        self._env = {**os.environ, **(init_params.get("env") or {})}
        self._max_steps = init_params.get("max_steps")
        self._timeout_s = float(init_params.get("step_timeout_s") or 30.0)

        os.makedirs(self._cwd, exist_ok=True)
        return {
            "kind": "shell.greeting",
            "cwd": self._cwd,
            "message": "shell adapter ready; send {'cmd': 'echo hi'} to act",
        }

    async def step(self, action: Action) -> StepResult:
        if isinstance(action, dict) and action.get("__exit__"):
            return StepResult(
                obs={"kind": "shell.exit"},
                reward=0.0,
                done=True,
                info={"steps": self._steps_taken},
            )

        cmd = self._normalize_action(action)
        self._steps_taken += 1

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout_s
            )
            exit_code = proc.returncode if proc.returncode is not None else -1
            timed_out = False
        except TimeoutError:
            proc.kill()
            await proc.wait()
            stdout, stderr = b"", b""
            exit_code = 124
            timed_out = True

        done = (
            self._max_steps is not None and self._steps_taken >= int(self._max_steps)
        )
        obs = {
            "kind": "shell.exec_result",
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "exit_code": exit_code,
            "cmd": cmd,
        }
        return StepResult(
            obs=obs,
            reward=0.0,
            done=done,
            info={
                "steps": self._steps_taken,
                "timed_out": timed_out,
            },
            truncated=timed_out,
        )

    async def teardown(self) -> None:
        return None

    @staticmethod
    def _normalize_action(action: Action) -> str:
        if isinstance(action, str):
            return action
        if isinstance(action, dict) and "cmd" in action:
            cmd = action["cmd"]
            if not isinstance(cmd, str):
                raise TypeError("ShellEnvAdapter: action['cmd'] must be a string")
            return cmd
        raise TypeError(
            "ShellEnvAdapter: action must be a str or {'cmd': str} dict; "
            f"got {type(action).__name__}"
        )
