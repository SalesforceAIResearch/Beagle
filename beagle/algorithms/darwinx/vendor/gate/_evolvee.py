"""Which subdirectory holds the agent under evolution.

The evolving agent lives in a submodule of the eval worktree. Its directory name was
historically the literal ``monet_code`` written at every site that needed it, which
quietly made this package single-agent: evolving anything else meant editing the
literal in a dozen files, and missing one produced a silent misclassification rather
than an error.

The name is read from the environment, with the historical value as the default, so
existing campaigns behave identically and a host can point the driver at a different
agent without patching source.

WHY THIS IS READ HERE RATHER THAN IMPORTED FROM ``self_evolve.worktree`` (which defines
the same knob): ``scope_filter`` classifies every path of every diff, and this package
is otherwise importable without the driver package. Taking an import dependency for one
string would couple the two packages on a hot path for no benefit. The env var is the
shared contract; keep the two defaults identical.
"""
from __future__ import annotations

import os

DEFAULT_AGENT_SUBMODULE = "monet_code"


def agent_submodule() -> str:
    """Directory name of the agent under evolution (e.g. ``monet_code``).

    Read per call rather than cached at import: a host may set it after this module is
    first imported, and the cost is a dict lookup.
    """
    return os.environ.get("DARWINX_EVOLVE_AGENT_SUBMODULE", "").strip() or DEFAULT_AGENT_SUBMODULE


__all__ = ["agent_submodule", "DEFAULT_AGENT_SUBMODULE"]
