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
import logging
import time
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from beagle.benchmarks.harness._common import (
    _GIT_BOOTSTRAP,
    WorkspaceSetupError,
    _rebuild_agent,
    effective_agent_budget_s,
    graceful_agent_timeout_s,
    interpret_workspace_check,
    workspace_check_command,
    workspace_probe_command,
)
from beagle.rollout.runtime.harbor_env import HarborEnvRuntime
from beagle.types import Task, TaskContext

LOGGER = logging.getLogger(__name__)

#: Re-exported so callers that catch it need not know where it is defined; the class itself lives
#: in the framework-free module, so it is importable (and testable) without harbor installed.
__all_errors__ = (WorkspaceSetupError,)


class BeagleInstalledAgent(BaseInstalledAgent):
    """Adapts any beagle ``Runnable`` to harbor's installed-agent interface."""

    SUPPORTS_ATIF: bool = False
    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "beagle"

    def __init__(
        self, logs_dir: Path, *, identity: dict[str, Any],
        task_env: dict[str, str] | None = None, **kwargs: Any,
    ) -> None:
        # ``model_name`` / ``extra_env`` / ``logger`` etc. arrive in kwargs from the
        # factory and flow up to BaseInstalledAgent/BaseAgent unchanged.
        super().__init__(logs_dir, **kwargs)
        self._identity = identity
        # Benchmark-supplied agent-phase container facts (HarborBenchmark.task_env). Empty for a
        # benchmark whose images already declare WORKDIR + PATH (terminal-bench) — then this shim
        # behaves exactly as before.
        self._task_env = dict(task_env or {})
        # harbor's own ``<trial>/agent`` dir — the handle to the trial's config.json, hence to the
        # task's declared agent budget (see _task_budget_s).
        self._logs_dir = Path(logs_dir)
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
        in-container (``pwd``, or the benchmark's own ``repo_path_cmd`` — harbor
        doesn't pass it to the agent).

        Cancel semantics: if harbor cancels this coroutine at the agent deadline
        (``asyncio.wait_for``), the ``to_thread`` result is unavailable and the
        ``context`` write-back below is skipped — but monet's raw stream is already
        on disk under ``/logs/agent`` (a native artifact harbor syncs on cancel) and
        the reward comes from the verifier, so nothing load-bearing is lost. The agent's
        own clock is set from the task's budget less a capture reserve (:meth:`_task_budget_s`),
        so it returns normally and populates ``context`` before harbor's deadline.
        """
        # harbor's agent deadline starts the moment IT calls us, so the phase clock starts here —
        # the workspace resolution below is already spending it.
        phase_started = time.monotonic()
        loop = asyncio.get_running_loop()
        runtime = HarborEnvRuntime(environment, loop)

        repo_path, setup = await self._resolve_workspace(environment)

        task = Task(task_id="trial", problem_statement=instruction, benchmark="")
        task_ctx = TaskContext(image=None, repo_path=repo_path,
                               shell_preamble=self._task_env.get("shell_preamble", ""),
                               agent_timeout_s=self._task_budget_s(
                                   spent_s=time.monotonic() - phase_started))

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
            # Workspace setup outcome (empty for benchmarks that declare no task_env). Recorded
            # rather than raised for the env half — see _resolve_workspace.
            **setup,
        }

        # NB: agent/trajectory.json (ATIF) is emitted by the harness POST-JOB, not here.
        # On the xrlenv cluster the agent's native stream (monet.stream.jsonl) is synced
        # from the container to the host trial dir only AFTER this step returns, so a
        # conversion here would find no stream. The harness converts once artifacts land
        # (see HarborHarness._emit_trajectories). The raw stream is a native artifact.

    def _task_budget_s(self, *, spent_s: float = 0.0) -> float | None:
        """The clock this trial's agent gets: the TASK's budget minus a margin, or ``None``.

        The benchmark declares how long its tasks get (``task.toml`` ``[agent] timeout_sec``) and
        the framework enforces it; the agent needs its own slightly-shorter clock so it RETURNS
        before the cancel and still captures its patch. Resolving it per trial is what lets one
        corpus carry heterogeneous budgets (SWE-rebench: 3000 s, four tasks at 6000 s) instead of
        being flattened to a single config number.

        ``spent_s`` is what this phase has ALREADY consumed before the agent is handed control
        (the workspace probe + check). Each layer deducts only its own segment — the shim deducts
        the resolution, ``Runnable.run`` then deducts acquire+install — so nothing is double-counted
        and nothing is missed. A resolution that hangs drives the remainder to the 1 s floor, which
        fails fast and visibly instead of being cancelled by harbor mid-capture.

        ``None`` when the task declares nothing (the framework imposes no deadline either) or the
        budget can't be read — the run config's ``agent.timeout`` is then the bound.
        """
        budget = graceful_agent_timeout_s(
            effective_agent_budget_s(self._logs_dir.parent / "config.json"), None)
        return None if budget is None else max(1.0, float(round(budget - spent_s)))

    async def _resolve_workspace(
        self, environment: BaseEnvironment
    ) -> tuple[str, dict[str, Any]]:
        """Resolve the agent's working directory and verify the benchmark's contract.

        Harbor-shaped I/O only: two ``environment.exec`` round trips. WHAT to run and what the
        output means are :mod:`_common`'s (``workspace_probe_command`` / ``workspace_check_command``
        / ``interpret_workspace_check``) — that decision logic is the seam's real behaviour, and
        keeping it here would leave it untested wherever the optional harbor extra isn't installed.

        Returns ``(repo_path, setup_metadata)``; with no ``task_env`` that is the historical bare
        ``pwd`` and an empty dict, unverified because no contract was declared.
        """
        res = await environment.exec(workspace_probe_command(self._task_env))
        repo_path = (res.stdout or "").strip() or "/"
        if not self._task_env:
            return repo_path, {}

        check = await environment.exec(
            workspace_check_command(repo_path, self._task_env.get("shell_preamble", "")))
        return repo_path, interpret_workspace_check(repo_path, check.stdout or "")



__all__ = ["BeagleInstalledAgent"]
