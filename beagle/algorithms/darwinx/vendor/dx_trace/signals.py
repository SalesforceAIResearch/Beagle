"""Shared, agent-agnostic signal extraction used by rule proposers/filters.

Tool *names* differ per agent, so classification is by substring/regex (an edit
is any tool whose name contains ``edit``/``write``/…; a test run is any shell
command matching a test-runner pattern), which generalizes across agents.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from .model import ToolCall

_EDIT_HINTS = ("edit", "write", "replace", "str_replace", "apply_patch", "create_file")
_PLAN_HINTS = ("todo", "plan")
_BASH_HINTS = ("bash", "shell", "run_shell", "execute", "command", "terminal")

_TEST_CMD = re.compile(
    r"\b(pytest|unittest|nosetests|tox|py\.test|"
    r"npm (?:run )?test|yarn test|jest|vitest|mocha|"
    r"go test|cargo test|ctest|gradle test|mvn test|"
    r"rspec|phpunit|make (?:test|check)|\./run_tests|run_tests\.sh|test\.sh)\b",
    re.IGNORECASE,
)

# Output that looks like a *failing test* (not a tool malfunction). The blog
# filters these out of tool-error reports — failing tests are normal debugging.
TEST_FAILURE = re.compile(
    r"\bFAILED\b|\b[1-9]\d* (?:failed|errors?)\b|Traceback \(most recent call last\)|"
    r"\bAssertionError\b|tests? failed|did not pass|FAIL(?:\s*\(|:)",
)

SUCCESS_LANG = re.compile(
    r"\b(done|completed?|success(?:fully)?|all (?:the )?tests? pass(?:ed|ing)?|"
    r"fixed|resolved|implemented|works? (?:now|correctly|as expected)|"
    r"the (?:issue|bug|problem) (?:is|has been) (?:fixed|resolved))\b",
    re.IGNORECASE,
)
GIVEUP_LANG = re.compile(
    r"\b(give up|giving up|unable to|cannot (?:fix|solve|resolve|figure)|"
    r"could not (?:fix|solve|resolve)|i (?:am|'m) stuck|not sure how|"
    r"ran out of|need(?:s)? (?:more|human) help|beyond (?:my|the) (?:scope|ability))\b",
    re.IGNORECASE,
)

TRUNCATION_EVIDENCE = re.compile(
    r"content_block_stop|stream ended|truncat|incomplete (?:response|stream)|unexpected end of",
    re.IGNORECASE,
)


def _has(name: str, hints: Iterable[str]) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


def is_edit(call: ToolCall) -> bool:
    return _has(call.name, _EDIT_HINTS) and not is_plan(call)


def is_plan(call: ToolCall) -> bool:
    return _has(call.name, _PLAN_HINTS)


def bash_command(call: ToolCall) -> str:
    if not _has(call.name, _BASH_HINTS):
        return ""
    for key in ("command", "cmd", "script", "code", "input"):
        val = call.arg(key)
        if isinstance(val, str) and val:
            return val
    return ""


def is_test_run(call: ToolCall) -> bool:
    return bool(_TEST_CMD.search(bash_command(call)))


def call_signature(call: ToolCall) -> str:
    """Stable key for loop detection: tool name + normalized argument fingerprint."""
    cmd = bash_command(call)
    if cmd:
        return f"{call.name}:{cmd.strip()}"
    if call.arguments is not None:
        return f"{call.name}:{json.dumps(call.arguments, sort_keys=True)[:400]}"
    return f"{call.name}:{(call.arguments_raw or '').strip()[:400]}"


def excerpt(text: str, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
