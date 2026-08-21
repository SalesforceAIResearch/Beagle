"""``meta_agent`` shim — route DarwinX's evolver call to a swappable beagle :class:`Editor`.

DarwinX's evolver is a ``meta_agent`` module it imports; ``meta_agent.run(prompt, workspace,
plan_mode=…)`` dispatches to a coding CLI (its native pick is hard-wired via an env var). This
is a drop-in with the SAME shape whose ``run()`` calls the **injected** ``Editor.edit(...)`` —
so the evolver becomes any beagle Editor (cursor / claude_code / monet / codex), **swappable
by config**, with zero change to DarwinX. This is seam B of the drop-in contract
(``notes/darwinx-dropin-contract.md`` §2).

Injection is explicit (:func:`set_editor`) — the adapter/Trainer sets the run's evolver before
DarwinX runs; swapping it swaps the evolver. The returned :class:`EditResult` carries
``text``/``exit_code``/``error``/``usage``/``tool_calls`` — the fields DarwinX's callers read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beagle.agents.core.base import EditResult, Editor

_EDITOR: Editor | None = None


def set_editor(editor: Editor | None) -> None:
    """Inject the Editor this shim routes to (the run's evolver), or ``None`` to clear it."""
    global _EDITOR
    _EDITOR = editor


def set_editor_from_spec(spec: Any) -> None:
    """Build the evolver Editor from an ``AgentConfig``-shaped spec (``{name, config, model?}``,
    or an :class:`~beagle.config.AgentConfig`) and inject it.

    This is the **config-based injection** (no env var, no flag): the evolver rides in the
    config as an ``AgentConfig`` (seam C), and DarwinX's config-load hook calls this once
    per-process — so every worker subprocess reconstructs the same Editor from the same config.
    """
    import beagle as bgl
    from beagle.config import AgentConfig

    ac = spec if isinstance(spec, AgentConfig) else AgentConfig.model_validate(spec)
    editor = bgl.agents.build(ac.to_spec())
    if not editor.can_be_evolver():
        raise TypeError(f"evolver {ac.name!r} is not an Editor")
    set_editor(editor)


def current_editor() -> Editor | None:
    """The currently-injected evolver Editor (or ``None``)."""
    return _EDITOR


def run(
    prompt: str,
    workspace: str | Path,
    plan_mode: bool = False,
    model: str | None = None,
    timeout_s: int | None = None,
    extra_args: list[str] | None = None,
    log_path: str | Path | None = None,
    **_ignored: Any,
) -> EditResult:
    """``meta_agent.run``-shaped entry → the injected ``Editor.edit``.

    ``plan_mode`` selects analysis (read-only) vs edit. Extra kwargs the algorithm may pass
    (e.g. ``reasoning_effort``) are accepted and ignored — the Editor's own config owns those.
    """
    if _EDITOR is None:
        raise RuntimeError(
            "darwinx meta_agent shim: no Editor injected — call set_editor(...) before running "
            "(the adapter wires the run's evolver)."
        )
    return _EDITOR.edit(
        str(prompt), Path(workspace),
        plan_mode=plan_mode, model=model, timeout_s=timeout_s, extra_args=extra_args,
        log_path=log_path,
    )


__all__ = ["set_editor", "set_editor_from_spec", "current_editor", "run"]
