"""Per-model-family reasoning effort: typed valid sets on GptModelConfig / ClaudeModelConfig,
carried into ModelSpec; cursor is the special case (effort in the slug)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import beagle as bgl
from beagle.config import AgentConfig, ClaudeModelConfig, GptModelConfig, ModelConfig


def test_gpt_effort_set() -> None:
    assert GptModelConfig(name="gpt-5.5", reasoning_effort="xhigh").reasoning_effort == "xhigh"
    with pytest.raises(ValidationError):
        GptModelConfig(name="gpt-5.5", reasoning_effort="max")   # claude-only level


def test_claude_effort_set() -> None:
    assert ClaudeModelConfig(name="claude-opus-4-8", reasoning_effort="max").reasoning_effort == "max"
    with pytest.raises(ValidationError):
        ClaudeModelConfig(name="claude-opus-4-8", reasoning_effort="minimal")  # gpt-only level


def test_base_model_config_is_permissive_on_effort() -> None:
    # the base takes any string (unknown/cursor); families are where it's validated
    assert ModelConfig(name="m", reasoning_effort="whatever").reasoning_effort == "whatever"


def test_to_spec_carries_reasoning_effort() -> None:
    spec = GptModelConfig(name="gpt-5.5", reasoning_effort="high").to_spec()
    assert spec.reasoning_effort == "high" and spec.name == "gpt-5.5"


def test_cursor_rejects_reasoning_effort_use_the_slug(tmp_path) -> None:
    # cursor encodes effort in the model slug — a separate reasoning_effort is a mis-spec.
    cur = bgl.agents.build(AgentConfig(
        name="cursor", model=GptModelConfig(name="gpt-5.5", reasoning_effort="high")))
    res = cur.edit("do it", tmp_path / "ws")   # returns before any subprocess
    assert not res.ok and "slug" in (res.error or "")


def test_cursor_slug_model_is_fine() -> None:
    # the right way for cursor: effort in the name, no reasoning_effort set
    cur = bgl.agents.build(AgentConfig(name="cursor", model=ModelConfig(name="gpt-5.5-high")))
    assert cur.spec.model is not None and cur.spec.model.reasoning_effort is None
