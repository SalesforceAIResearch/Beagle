"""xrlenv plug-in for the pier RL framework.

pier (https://github.com/datacurve-ai/pier) is a harbor fork that reimplements the
trial/verifier harness in-tree (it does NOT import harbor at runtime). It ships its
own ``BaseEnvironment`` with built-in implementations for docker, modal, and
daytona, and — first-class — an ``EnvironmentConfig.import_path`` escape hatch to
select a custom environment class. This plug-in adds an xrlenv-aware variant —
:class:`XrlenvPierEnvironment` — that satisfies pier's environment contract while
routing container ops through xrlenv's primitives when the consumer wants cluster
placement / capacity-aware scheduling / cancellation primitives without giving up
pier's task-format and grading conventions.

This is the direct analog of :mod:`xrlenv_plugins.harbor`, retargeted at pier's
classes; see ``xrlenv_plugins/harbor/README.md`` for the canonical
"``Xrlenv<Framework><BaseRole>``" plug-in pattern this follows.

Select it exactly the way pier's built-in environments are selected — via
``EnvironmentConfig.import_path`` (or ``pier run --environment-import-path``)::

    environment:
      import_path: xrlenv_plugins.pier:XrlenvPierEnvironmentCluster
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xrlenv_plugins.pier.environment import (
        XrlenvPierEnvironment,
        XrlenvPierEnvironmentCluster,
    )

__all__ = [
    "XrlenvPierEnvironment",
    "XrlenvPierEnvironmentCluster",
]


# Lazy re-export (PEP 562). ``environment`` imports the ``pier`` runtime lib;
# importing it eagerly here would force every ``import xrlenv_plugins.pier.*`` to
# pull pier in. The pure ``xrlenv_plugins.pier.compose`` helper is shared with the
# deliberately pier-free build-plan generator, so the two classes are resolved on
# first attribute access instead. ``from xrlenv_plugins.pier import
# XrlenvPierEnvironmentCluster`` keeps working unchanged.
def __getattr__(name: str) -> Any:
    if name in __all__:
        from xrlenv_plugins.pier import environment

        return getattr(environment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
