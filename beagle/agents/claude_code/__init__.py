"""Claude Code — a black-box evolver.

The ``claude`` CLI used as a coding agent, analogous to :mod:`cursor`. Black-box
internals, so an :class:`Editor` only — it cannot be evolved.
"""

from __future__ import annotations

from pathlib import Path

from beagle.agents.core.base import Agent, EditResult, Editor
from beagle.agents.core.registry import register
from beagle.types import Transparency


@register("claude-code")
class ClaudeCodeAgent(Agent, Editor):
    """External ``claude`` CLI — a black-box editor (evolver)."""

    transparency = Transparency.BLACK_BOX

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
        raise NotImplementedError("claude-code edit() not yet implemented")


__all__ = ["ClaudeCodeAgent"]
