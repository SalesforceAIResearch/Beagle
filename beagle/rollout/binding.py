"""Rollout bindings — the seam between an agent and a benchmark's native harness.

A :class:`RolloutBinding` describes *how* a benchmark harness should run a given
agent on a given task. It is what lets beagle honor the "respect the original
harness" contract: instead of a bespoke run loop, the benchmark harness drives
the rollout and emits artifacts in its own native format, while the agent plugs in
through one of these bindings.

Two shapes:

* :class:`GenericBinding` — wraps an agent's harness-agnostic ``run(task, task_ctx,
  *, runtime)``. The harness supplies a runtime (a real container substrate, or an
  adapter over the harness's own environment) and calls back into the agent. This
  is the default, and the only thing most agents ever need — implement ``run`` once
  and it works on every harness.

* :class:`HarborBinding` — for agents that are already native to a harbor-style
  harness. The harness imports ``import_path`` and constructs the agent itself; no
  callback. Use this only when bringing a pre-existing native agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from beagle.rollout.runtime import ContainerRuntime
    from beagle.types import Task, TaskContext, TaskResult


@runtime_checkable
class RunFn(Protocol):
    """The harness-agnostic run callable an agent exposes."""

    def __call__(
        self, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime
    ) -> TaskResult: ...


@runtime_checkable
class RolloutBinding(Protocol):
    """Marker protocol for a harness-specific agent descriptor."""

    #: Short tag identifying the binding kind, e.g. ``"generic"`` / ``"harbor"``.
    kind: str


@dataclass
class GenericBinding:
    """Run an agent via its own ``run`` callable, driven by a harness-supplied runtime.

    Works on any harness: a docker-based harness passes a container runtime; a
    harbor-style harness passes a runtime backed by its trial environment. Either
    way the agent's ``run`` is the single integration point.
    """

    run: RunFn
    kind: str = "generic"


@dataclass
class HarborBinding:
    """Run an agent that is already a native harbor installed-agent.

    The harness imports ``import_path`` (``"module.path:AgentClass"``) and builds
    the agent with ``kwargs``; harbor owns the container + trial lifecycle and the
    trajectory format. Only needed to bring a pre-existing native agent — new
    agents should implement ``run`` and use :class:`GenericBinding`.
    """

    import_path: str
    model_name: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: str = "harbor"


__all__ = ["RunFn", "RolloutBinding", "GenericBinding", "HarborBinding"]
