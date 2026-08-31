"""Copy this file to onboard your own agent.

Steps
-----
1. Copy this file to ``beagle/agents/<your_agent>/__init__.py`` (its own package,
   alongside the other concrete agents; add helper modules beside it as needed).
2. Rename the class, pick a registry name, and compose the **capability mixins**
   your agent supports — implement their methods. Do NOT pick a role; role is the
   user's choice at run time and is validated against your capabilities.
3. Done — auto-discovered on ``import beagle``. No other file changes.

Capabilities (mix in what applies):

* ``Runnable``   → implement ``run``: attempt a benchmark task. One harness-agnostic
  method that runs on every benchmark harness.
* ``Evolvable``  → implement ``_default_source``: name the agent's git repo + default
  ref + entrypoint. A run pins a ref (baseline = un-evolved, candidate = evolved);
  the runner materializes it. White-box only.
* ``Editor``     → implement ``edit``: run one coding instruction against a workspace.
  This is the thin primitive an evolution algorithm drives (it authors the prompts
  and owns the analyze/implement/review recipe — your agent stays dumb).

A white-box coding agent typically composes all three (usable as either role); a
closed-source CLI is an ``Editor`` only. This file lives under ``core`` (skipped by
auto-discovery); the copy you make under ``beagle/agents/<your_agent>/`` is what
gets registered.
"""

from __future__ import annotations

from pathlib import Path

from beagle.agents.core import (
    Agent,
    AgentSource,
    EditResult,
    Editor,
    Evolvable,
    Runnable,
    register,
)
from beagle.rollout.runtime import ContainerRuntime
from beagle.types import Task, TaskContext, TaskResult, Transparency


@register("my-agent")
class MyAgent(Agent, Runnable, Evolvable, Editor):
    """A white-box agent usable as evolvee or evolver. Drop capabilities you lack.

    (For a closed-source CLI, use ``class MyAgent(Agent, Editor)`` and set
    ``transparency = Transparency.BLACK_BOX`` — it can only be an evolver.)
    """

    transparency = Transparency.WHITE_BOX
    REPO = "https://github.com/you/your-agent"  # a fork you control (for evolved branches)

    # Evolvable: name the versioned source (repo @ ref). The runner materializes it.
    def _default_source(self) -> AgentSource:
        return self.spec.source or AgentSource(repo=self.REPO, entrypoint="")

    # Runnable: attempt a task, running the source at self.source() (a specific ref).
    def run(self, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> TaskResult:
        src = self.source()
        handle = runtime.acquire(image=task_ctx.image or "", command=["sleep", "infinity"])
        try:
            runtime.exec(handle, ["bash", "-lc",
                                  f"git clone {src.repo} /agent && cd /agent && git checkout {src.ref or 'main'}"])
            ...  # build + invoke your agent; collect a patch/trajectory
            return TaskResult(task_id=task.task_id)
        finally:
            runtime.destroy(handle)

    # Editor: run one coding instruction in a workspace (the algorithm calls this).
    def edit(self, instruction: str, workspace: Path, *, plan_mode: bool = False,
             model: str | None = None, timeout_s: int | None = None,
             extra_args: list[str] | None = None,
             log_path: str | Path | None = None) -> EditResult:
        ...  # run your coding agent against `workspace`; leave edits in its git state
        return EditResult(text="")
