"""Agent framework primitives — the atomic pieces every concrete agent builds on.

Kept separate from the concrete agents (each in ``beagle/agents/<name>/``) so the
factory surface and the capability contracts live in one place:

* :mod:`~beagle.agents.core.base` — the :class:`Agent` base + the capability mixins
  (:class:`Runnable`, :class:`Editor`, :class:`Evolvable`) and :class:`Capability`.
* :mod:`~beagle.agents.core.spec` — :class:`AgentSpec` / :class:`AgentSource` /
  :class:`ModelSpec` (the serializable description a factory builds an agent from).
* :mod:`~beagle.agents.core.registry` — :func:`register` / :func:`build` /
  :func:`available`.

Import primitives from here (``from beagle.agents.core import Agent, register``);
this package never triggers agent auto-discovery, so it's safe to import early.
"""

from __future__ import annotations

from beagle.agents.core.base import (
    Agent,
    AgentSource,
    Capability,
    EditResult,
    Editor,
    Evolvable,
    Runnable,
    Topology,
)
from beagle.agents.core.registry import AGENTS, available, build, register
from beagle.agents.core.spec import AgentSpec, ModelSpec

__all__ = [
    "Agent",
    "AgentSource",
    "Capability",
    "EditResult",
    "Editor",
    "Evolvable",
    "Runnable",
    "Topology",
    "AGENTS",
    "available",
    "build",
    "register",
    "AgentSpec",
    "ModelSpec",
]
