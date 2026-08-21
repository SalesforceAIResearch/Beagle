"""Terminus — reference adapter for an open-source terminal agent.

White-box, usable as evolvee or evolver. Terminus is natively a harbor agent, so a
real implementation may override :meth:`rollout_binding` to return a native harbor
binding instead of the default ``run`` wrapper. Shown here with the same in-container
installed path as the others for consistency.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from beagle.agents.core.base import (
    Agent,
    AgentSource,
    EditResult,
    Editor,
    Evolvable,
    Runnable,
    Topology,
)
from beagle.agents.core.registry import register
from beagle.rollout.runtime import ContainerRuntime
from beagle.types import Task, TaskContext, TaskResult, Transparency, TrajectoryRef


@register("terminus")
class TerminusAgent(Agent, Runnable, Evolvable, Editor):
    """The Terminus terminal agent — white-box, usable as evolvee or evolver."""

    transparency = Transparency.WHITE_BOX
    topology = Topology.IN_CONTAINER
    #: Point at your fork so the evolver can push branches.
    REPO = ""

    def _default_source(self) -> AgentSource:
        return self.spec.source or AgentSource(repo=self.REPO, entrypoint="")

    def run(self, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> TaskResult:
        src = self.source()
        handle = runtime.acquire(image=task_ctx.image or "", command=["sleep", "infinity"])
        try:
            runtime.exec(
                handle,
                ["bash", "-lc",
                 f"git clone {shlex.quote(src.repo)} /agent && cd /agent && "
                 f"git checkout {shlex.quote(src.ref or 'main')}"],
                timeout=600,
            )
            # TODO: install + invoke terminus against task_ctx; collect patch/trajectory.
            diff = runtime.exec(
                handle,
                ["bash", "-lc", f"cd {shlex.quote(task_ctx.repo_path)} && git add -A && git diff --cached"],
            ).stdout
            return TaskResult(
                task_id=task.task_id,
                patch=diff or None,
                trajectory=TrajectoryRef(path=Path("terminus.trajectory.json"), format="terminus"),
            )
        finally:
            runtime.destroy(handle)

    def edit(
        self,
        instruction: str,
        workspace: Path,
        *,
        plan_mode: bool = False,
        model: str | None = None,
        timeout_s: int | None = None,
        extra_args: list[str] | None = None,
        log_path: str | Path | None = None,
    ) -> EditResult:
        raise NotImplementedError("terminus edit() not yet implemented")


__all__ = ["TerminusAgent"]
