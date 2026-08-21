"""The Agent Factory.

Public surface::

    import beagle as bgl
    bgl.agents.build("monet")       # a white-box coding agent (either role)
    bgl.agents.build("cursor")      # a black-box evolver (editor only)
    bgl.agents.available()          # -> ["claude-code", "codex", "cursor", ...]

An agent's **role** (evolvee vs evolver) is chosen when you use it, not fixed by
its class. Instead an agent declares the **capabilities** it has by mixing them in
— :class:`Runnable`, :class:`Editor`, :class:`Evolvable` — and the trainer assigns
a role its capabilities support. See :mod:`beagle.agents.core.base`.

Layout: framework primitives live in ``beagle/agents/core/`` (the :class:`Agent`
base, capability mixins, specs, registry); each concrete agent is its own package
``beagle/agents/<name>/``.

Adding your own agent is a one-package drop — no edits here:

    # beagle/agents/my_agent/__init__.py
    from beagle.agents.core import Agent, Runnable, Evolvable, register

    @register("my-agent")
    class MyAgent(Agent, Runnable, Evolvable):   # a white-box evolvee
        def run(self, task, task_ctx, *, runtime): ...
        def _default_source(self): ...

Every package under ``beagle/agents/`` (except ``core`` and names starting with
``_``) is imported at startup, so a ``@register``-ed class becomes available the
moment its package exists. Copy ``core/_template.py`` to start.
"""

from __future__ import annotations

import importlib
import pkgutil

from beagle.agents.core import (
    AGENTS,
    Agent,
    AgentSource,
    AgentSpec,
    Capability,
    EditResult,
    Editor,
    Evolvable,
    ModelSpec,
    Runnable,
    Topology,
    available,
    build,
    register,
)

# ``core`` holds the framework primitives (not an agent) — skip during discovery.
_FRAMEWORK_MODULES = {"core"}


def _autodiscover() -> None:
    """Import every agent module/subpackage so its ``@register`` runs."""
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name in _FRAMEWORK_MODULES:
            continue
        importlib.import_module(f"{__name__}.{info.name}")


_autodiscover()

__all__ = [
    # factory
    "AGENTS",
    "build",
    "available",
    "register",
    # specs
    "AgentSpec",
    "ModelSpec",
    "AgentSource",
    # base + capabilities
    "Agent",
    "Capability",
    "Topology",
    "Runnable",
    "Editor",
    "Evolvable",
    "EditResult",
]
