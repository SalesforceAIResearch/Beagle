"""``BeagleInstalledAgent`` — the ONE harbor shim that runs any beagle agent.

Harbor drives an *installed agent* through its own ``BaseEnvironment`` (async exec,
no host bind-mount). Rather than write a harbor-native class per agent
(``MonetHarborAgent``, ``MiniSweHarborAgent``, …) — an M×N trap — we write this
shim once. It reconstructs the beagle agent from a small serializable *identity*,
wraps harbor's environment as a :class:`HarborEnvRuntime`, and calls the agent's
ordinary ``run(task, task_ctx, *, runtime)``. Every agent then works on harbor with
**zero** harbor-specific code; adding an agent is one ``run()``, adding a harness is
one runtime adapter.

Harbor's ``AgentFactory`` constructs this by ``import_path`` and spreads
``AgentConfig.kwargs`` flat into the constructor, so ``identity`` arrives as a
keyword. This module is only ever imported *by harbor* (via that import path), so
importing harbor at module top is fine — beagle core never imports it, keeping
harbor an optional dependency.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from beagle.benchmarks.harness._common import _GIT_BOOTSTRAP, _rebuild_agent
from beagle.rollout.runtime.harbor_env import HarborEnvRuntime
from beagle.types import Task, TaskContext


class BeagleInstalledAgent(BaseInstalledAgent):
    """Adapts any beagle ``Runnable`` to harbor's installed-agent interface."""

    SUPPORTS_ATIF: bool = False
    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "beagle"

    def __init__(self, logs_dir: Path, *, identity: dict[str, Any], **kwargs: Any) -> None:
        # ``model_name`` / ``extra_env`` / ``logger`` etc. arrive in kwargs from the
        # factory and flow up to BaseInstalledAgent/BaseAgent unchanged.
        super().__init__(logs_dir, **kwargs)
        self._identity = identity
        self._agent = _rebuild_agent(identity)

    async def install(self, environment: BaseEnvironment) -> None:
        """Ensure git is present (root); the agent clones + builds itself in run()."""
        await self.exec_as_root(environment, command=_GIT_BOOTSTRAP, timeout_sec=300)

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Bridge harbor's async trial to the agent's sync ``run``.

        The agent runs in a worker thread; its ``runtime.exec`` calls hop back to
        harbor's event loop via :class:`HarborEnvRuntime`. Harbor supplies the task
        instruction (``instruction.md``); the task's working directory is resolved
        with ``pwd`` (harbor doesn't pass it to the agent).

        Cancel semantics: if harbor cancels this coroutine at the agent deadline
        (``asyncio.wait_for``), the ``to_thread`` result is unavailable and the
        ``context`` write-back below is skipped — but monet's raw stream is already
        on disk under ``/logs/agent`` (a native artifact harbor syncs on cancel) and
        the reward comes from the verifier, so nothing load-bearing is lost. Keep the
        agent's own timeout (``config.timeout``) below harbor's agent deadline so the
        agent returns normally and populates ``context`` first.
        """
        loop = asyncio.get_running_loop()
        runtime = HarborEnvRuntime(environment, loop)

        pwd = await environment.exec("pwd")
        repo_path = (pwd.stdout or "/").strip()

        task = Task(task_id="trial", problem_statement=instruction, benchmark="")
        task_ctx = TaskContext(image=None, repo_path=repo_path)

        result = await asyncio.to_thread(
            self._agent.run, task, task_ctx, runtime=runtime
        )

        # Feed harbor-side observability; the reward itself comes from harbor's
        # verifier (native artifacts), not from anything returned here.
        tokens = result.tokens or {}
        # harbor's n_input_tokens is total input INCLUDING cache (== beagle ``prompt``); n_cache_tokens
        # is the cached subset. Set BOTH so the cache split survives into result.json / run.json.
        context.n_input_tokens = tokens.get("prompt")
        context.n_output_tokens = tokens.get("completion")
        context.n_cache_tokens = (tokens.get("cache_read") or 0) + (tokens.get("cache_write") or 0)
        context.metadata = {
            "agent": self._identity.get("agent"),
            "error": result.error,
            "patch": result.patch,
        }

        # NB: agent/trajectory.json (ATIF) is emitted by the harness POST-JOB, not here.
        # On the xrlenv cluster the agent's native stream (monet.stream.jsonl) is synced
        # from the container to the host trial dir only AFTER this step returns, so a
        # conversion here would find no stream. The harness converts once artifacts land
        # (see HarborHarness._emit_trajectories). The raw stream is a native artifact.


__all__ = ["BeagleInstalledAgent"]
