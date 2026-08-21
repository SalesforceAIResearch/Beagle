"""Meta-agent (proposer) dispatch — REPLACED for beagle integration (seam B).

The original env-gated dispatcher (``META_AGENT`` → cursor/monet/claude backends) is replaced
by a thin delegator to beagle's Editor. Every DarwinX call site does ``from . import
meta_agent; meta_agent.run(...)`` — those now route to the **injected beagle Editor** via
``beagle.algorithms.darwinx.meta_agent`` (seam B of the drop-in contract), so the evolver is
any beagle Editor (cursor / claude_code / monet / codex), swappable by config, with the rest
of DarwinX unchanged. The result object carries ``text``/``exit_code``/``error``/``usage``/
``tool_calls`` — the fields the callers read.

This file is replaced (not `sys.modules`-shadowed) because DarwinX's workers are subprocesses
that re-import ``self_evolve.meta_agent`` fresh — they must load the routed version.
The original dispatcher is in git history / the DarwinX reference.
"""

from __future__ import annotations

from typing import Any


def active_backend() -> str:
    """The active proposer backend — here, the injected beagle Editor's name (else ``beagle``)."""
    from beagle.algorithms.darwinx import meta_agent as shim

    ed = shim.current_editor()
    return ed.name if ed is not None else "beagle"


def run(*args: Any, reasoning_effort: str | None = None, **kwargs: Any):
    """Dispatch one proposer call → the injected beagle ``Editor.edit`` (via the shim).

    The call sites pass ``(prompt, workspace)`` + keywords (``plan_mode``/``model``/``timeout_s``/
    ``extra_args``/``log_path``/…); the shim's ``run`` forwards the ones the Editor uses and
    ignores the rest. ``reasoning_effort`` is dropped (the Editor's own config owns effort).
    """
    from beagle.algorithms.darwinx import meta_agent as shim

    return shim.run(*args, **kwargs)


__all__ = ["run", "active_backend"]
