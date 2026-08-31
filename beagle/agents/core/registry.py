"""The agent registry and the ``build`` factory.

Onboarding: compose the capability mixins your agent supports
(:class:`~beagle.agents.core.base.Runnable` / :class:`~beagle.agents.core.base.Evolvable` /
:class:`~beagle.agents.core.base.Editor`), decorate with :func:`register`, and drop the
file under ``beagle/agents/``. The package auto-discovers it — no other edits.

    from beagle.agents import Agent, Editor, register

    @register("my-agent")
    class MyAgent(Agent, Editor):
        def edit(self, instruction, workspace, **kw): ...
"""

from __future__ import annotations

from collections.abc import Callable

from beagle.agents.core.base import Agent
from beagle.agents.core.spec import AgentSpec
from beagle.registry import Registry

#: name -> Agent subclass. Populated by the ``@register`` decorator at import.
AGENTS: Registry[type[Agent]] = Registry("agent")


def register(name: str) -> Callable[[type[Agent]], type[Agent]]:
    """Class decorator: register an agent under ``name`` and stamp it on the class."""

    def _decorate(cls: type[Agent]) -> type[Agent]:
        cls.NAME = name
        AGENTS.register(name, cls)
        return cls

    return _decorate


def build(spec: AgentSpec | str, /, **overrides) -> Agent:
    """Build an agent from a registered name, a declarative :class:`~beagle.config.AgentConfig`,
    or a runtime :class:`AgentSpec`.

    ``build("cursor")`` / ``build("cursor", role=AgentRole.EVOLVER)`` /
    ``build(AgentConfig(name="monet", model=…, source=…))`` / ``build(AgentSpec(name="monet", …))``.
    The config is the declarative surface (what a YAML holds); it's converted to its spec here.
    """
    if isinstance(spec, str):
        spec = AgentSpec(name=spec, **overrides)
    elif hasattr(spec, "to_spec"):     # an AgentConfig (declarative) → its AgentSpec
        if overrides:
            raise TypeError("pass overrides only when building from a name, not an AgentConfig")
        spec = spec.to_spec()
    elif overrides:
        raise TypeError("pass overrides only when building from a name, not an AgentSpec")
    return AGENTS.get(spec.name)(spec)


def available() -> list[str]:
    """Names of all registered agents."""
    return AGENTS.names()


__all__ = ["AGENTS", "register", "build", "available"]
