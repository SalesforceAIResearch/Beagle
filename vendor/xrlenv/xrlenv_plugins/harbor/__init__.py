"""xrlenv plug-in for the harbor RL framework.

harbor (https://github.com/harbor-framework/harbor) ships its own
``BaseEnvironment`` Protocol with built-in implementations for
docker, modal, runloop, e2b, gke, and others. This plug-in adds an
xrlenv-aware variant — :class:`XrlenvHarborEnvironment` — that
satisfies the same Protocol while routing container ops through
xrlenv's primitives when the consumer wants cluster placement /
capacity-aware scheduling / cancellation primitives without giving
up harbor's task-format and grading conventions.

Naming convention
-----------------

We use the ``Xrlenv<Framework><BaseRole>`` pattern so plug-ins for
multiple frameworks coexist without collision:

- ``XrlenvHarborEnvironment(harbor.BaseEnvironment)`` — this plug-in
- (future) ``XrlenvFooAgent(foo.BaseAgent)`` — same shape, different
  framework
- (future) ``XrlenvBarProvider(bar.Provider)`` — same shape, role
  spelled correctly per the host framework

See ``xrlenv_plugins/harbor/README.md`` for the canonical "how to
build an xrlenv plug-in for a new RL framework" doc — this package
is the worked reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xrlenv_plugins.harbor.environment import (
        XrlenvHarborEnvironment,
        XrlenvHarborEnvironmentCluster,
    )

__all__ = [
    "XrlenvHarborEnvironment",
    "XrlenvHarborEnvironmentCluster",
]


# Lazy re-export (PEP 562). ``environment`` imports the ``harbor`` runtime lib;
# importing it eagerly here would force every ``import xrlenv_plugins.harbor.*``
# to pull harbor in. The pure ``xrlenv_plugins.harbor.compose`` helper is shared
# with the deliberately harbor-free build-plan generator, so the two classes are
# resolved on first attribute access instead. ``from xrlenv_plugins.harbor import
# XrlenvHarborEnvironmentCluster`` keeps working unchanged.
def __getattr__(name: str) -> Any:
    if name in __all__:
        from xrlenv_plugins.harbor import environment

        return getattr(environment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
