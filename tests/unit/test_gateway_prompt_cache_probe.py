from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_PATH = Path(__file__).parents[1] / "integration" / "gateway_prompt_cache_by_model.py"
_SPEC = importlib.util.spec_from_file_location("gateway_prompt_cache_by_model", _PATH)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def _turn(*, prompt=1_000, cached=400, write=100, served="gpt-5.6-sol"):
    return probe.Turn(prompt=prompt, cached=cached, write=write, served=served)


def _count_key(value, key: str) -> int:
    if isinstance(value, dict):
        return (key in value) + sum(_count_key(item, key) for item in value.values())
    if isinstance(value, list):
        return sum(_count_key(item, key) for item in value)
    return 0


def test_explicit_shape_uses_key_options_and_retained_tool_breakpoints() -> None:
    messages, extra = probe.build_messages("session", 3, 6, "explicit")

    assert extra == {
        "prompt_cache_key": "cache-probe-session",
        "prompt_cache_options": {"mode": "explicit"},
    }
    # The gateway-validated shape marks retained tool results only.
    assert _count_key(messages, "prompt_cache_breakpoint") == 3
    assert messages[-1]["role"] == "tool"
    assert "<token_budget" in messages[-1]["content"][-1]["text"]


def test_automatic_shape_has_no_hints_and_budget_inside_latest_tool() -> None:
    messages, extra = probe.build_messages("session", 3, 6, "automatic")

    assert extra is None
    assert _count_key(messages, "prompt_cache_breakpoint") == 0
    assert _count_key(messages, "cache_control") == 0
    assert messages[-1]["role"] == "tool"
    assert "<token_budget" in messages[-1]["content"][-1]["text"]


def test_gpt56_implicit_shape_uses_key_without_explicit_markers() -> None:
    messages, extra = probe.build_messages("session", 3, 6, "implicit56")

    assert extra == {
        "prompt_cache_key": "cache-probe-session",
        "prompt_cache_options": {"mode": "implicit"},
    }
    assert _count_key(messages, "prompt_cache_breakpoint") == 0
    assert _count_key(messages, "cache_control") == 0
    assert messages[-1]["role"] == "tool"
    assert "<token_budget" in messages[-1]["content"][-1]["text"]


def test_anthropic_shape_uses_three_cache_controls_and_no_openai_hints() -> None:
    messages, extra = probe.build_messages("session", 3, 6, "anthropic")

    assert extra is None
    # stable system + last two user/tool messages
    assert _count_key(messages, "cache_control") == 3
    assert _count_key(messages, "prompt_cache_breakpoint") == 0
    assert messages[-1]["role"] == "user"
    assert "<token_budget" in messages[-1]["content"]


def test_missing_or_inconsistent_usage_is_inconclusive() -> None:
    assert probe.validate_turn(_turn(cached=None), "gpt-5.6-sol")
    assert probe.validate_turn(_turn(prompt=None), "gpt-5.6-sol")
    assert probe.validate_turn(_turn(prompt=100, cached=80, write=30), "gpt-5.6-sol")
    assert probe.validate_turn(_turn(served=None), "gpt-5.6-sol")
    assert probe.validate_turn(_turn(), "gpt-5.6-sol") is None


def test_marker_role_control_changes_only_message_role_structure() -> None:
    user_messages = probe.build_marker_role_messages("session", "user", "suffix")
    tool_messages = probe.build_marker_role_messages("session", "tool", "suffix")

    assert _count_key(user_messages, "prompt_cache_breakpoint") == 1
    assert _count_key(tool_messages, "prompt_cache_breakpoint") == 1
    assert user_messages[-1]["role"] == "user"
    assert tool_messages[-1]["role"] == "tool"
    assert user_messages[-1]["content"][0] == tool_messages[-1]["content"][0]
