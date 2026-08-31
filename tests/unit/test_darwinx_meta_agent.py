"""seam B — the meta_agent shim routes DarwinX's evolver call to a swappable Editor.

Hermetic: a fake Editor stands in for cursor/claude/monet; we assert the shim forwards the
right args to Editor.edit, fails loud with no editor, and lets the evolver be swapped."""

from __future__ import annotations

import pytest

from beagle.agents.core.base import EditResult
from beagle.algorithms.darwinx import meta_agent


class _FakeEditor:
    def __init__(self, tag: str = "ok") -> None:
        self.tag = tag
        self.seen: dict = {}

    def edit(self, instruction, workspace, *, plan_mode, model, timeout_s, extra_args, log_path=None):  # noqa: ANN001
        self.seen = dict(instruction=instruction, workspace=str(workspace), plan_mode=plan_mode,
                         model=model, timeout_s=timeout_s, extra_args=extra_args)
        return EditResult(text=self.tag, usage={"inputTokens": 3})


def test_meta_agent_routes_to_injected_editor(tmp_path) -> None:
    ed = _FakeEditor()
    meta_agent.set_editor(ed)
    try:
        res = meta_agent.run("analyze this", tmp_path / "ws", plan_mode=True, model="m",
                             reasoning_effort="high")   # extra kwarg accepted + ignored
        assert isinstance(res, EditResult) and res.text == "ok" and res.usage == {"inputTokens": 3}
        assert ed.seen == {"instruction": "analyze this", "workspace": str(tmp_path / "ws"),
                           "plan_mode": True, "model": "m", "timeout_s": None, "extra_args": None}
    finally:
        meta_agent.set_editor(None)


def test_set_editor_from_spec_builds_the_evolver(tmp_path) -> None:
    """Config-based injection: build the Editor from an AgentConfig-shaped evolver spec (no env
    var). A worker's config-load hook calls this, so it reconstructs the same Editor per-process."""
    meta_agent.set_editor_from_spec({"name": "cursor", "config": {"model": "auto"}})
    try:
        ed = meta_agent.current_editor()
        assert ed is not None and ed.name == "cursor" and ed.can_be_evolver()
    finally:
        meta_agent.set_editor(None)


def test_meta_agent_without_editor_fails_loud(tmp_path) -> None:
    meta_agent.set_editor(None)
    with pytest.raises(RuntimeError, match="no Editor injected"):
        meta_agent.run("x", tmp_path / "ws")


def test_meta_agent_evolver_is_swappable(tmp_path) -> None:
    """Swapping the injected Editor swaps the evolver — DarwinX's call is unchanged."""
    meta_agent.set_editor(_FakeEditor("cursor"))
    try:
        assert meta_agent.run("x", tmp_path / "ws").text == "cursor"
        meta_agent.set_editor(_FakeEditor("claude"))
        assert meta_agent.run("x", tmp_path / "ws").text == "claude"
        assert meta_agent.current_editor() is not None
    finally:
        meta_agent.set_editor(None)
