"""Additional coverage for the cursor editor — retry loop, plan_body_started, and
log_path forwarding.

All tests are hermetic: the retry loop uses a monkeypatched run_cli so no real
subprocess is spawned and time.sleep is stubbed to keep the suite fast."""

from __future__ import annotations

import json
import stat
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from beagle.agents.core import edit_driver
from beagle.agents.core.spec import AgentSpec
from beagle.agents.cursor import (
    CursorAgent,
    _is_trivial_plan_body,
    _parse_stream,
)


# ---------------------------------------------------------------------------
# _is_trivial_plan_body edge cases
# ---------------------------------------------------------------------------

def test_is_trivial_plan_body_none() -> None:
    assert _is_trivial_plan_body(None) is True


def test_is_trivial_plan_body_empty() -> None:
    assert _is_trivial_plan_body("") is True


def test_is_trivial_plan_body_too_short() -> None:
    assert _is_trivial_plan_body("# short") is True


def test_is_trivial_plan_body_real_plan() -> None:
    assert _is_trivial_plan_body("# Real plan\n" + "step 1. do it. " * 20) is False


@pytest.mark.parametrize("marker", [
    "see assistant message",
    "see the assistant message",
    "see assistant text",
    "final plan is rendered",
    "in the final assistant message",
])
def test_is_trivial_plan_body_stub_markers(marker: str) -> None:
    body = marker.upper() + " " + "x" * 300  # long enough but stub marker present
    assert _is_trivial_plan_body(body) is True


# ---------------------------------------------------------------------------
# plan_body_started fallback in stream parser
# ---------------------------------------------------------------------------

def test_parse_stream_plan_body_started_falls_back_when_no_completed(
) -> None:
    """When a CreatePlanToolCall body appears in a 'started' event (but no
    'completed' event), the started body is the fallback plan channel."""
    plan = "# Plan\n" + "do the thing. " * 30  # > 200 chars, real plan
    st = _parse_stream("\n".join([
        json.dumps({
            "type": "tool_call", "subtype": "started",
            "tool_call": {"createPlanToolCall": {"args": {"plan": plan}}},
        }),
        json.dumps({"type": "result", "result": "result text"}),
    ]))
    assert st.plan_body_started == plan
    assert st.plan_body_completed is None
    assert st.final_text() == plan  # started > result when completed absent


def test_parse_stream_completed_beats_started() -> None:
    """A completed plan body takes priority over a started one."""
    plan_started = "# Started plan\n" + "a. " * 80
    plan_completed = "# Completed plan\n" + "b. " * 80
    st = _parse_stream("\n".join([
        json.dumps({
            "type": "tool_call", "subtype": "started",
            "tool_call": {"createPlanToolCall": {"args": {"plan": plan_started}}},
        }),
        json.dumps({
            "type": "tool_call", "subtype": "completed",
            "tool_call": {"createPlanToolCall": {"args": {"plan": plan_completed}}},
        }),
    ]))
    assert st.plan_body_started == plan_started
    assert st.plan_body_completed == plan_completed
    assert st.final_text() == plan_completed  # completed wins


def test_parse_stream_result_is_error_flag() -> None:
    st = _parse_stream(json.dumps({"type": "result", "is_error": True, "result": ""}))
    assert st.is_error is True


def test_parse_stream_assistant_last_non_empty_wins() -> None:
    """Of multiple assistant text parts, the last non-empty one is kept."""
    st = _parse_stream("\n".join([
        json.dumps({"type": "assistant",
                    "message": {"content": [{"type": "text", "text": "first"}]}}),
        json.dumps({"type": "assistant",
                    "message": {"content": [{"type": "text", "text": "second"}]}}),
    ]))
    assert st.assistant_text == "second"


def test_parse_stream_usage_non_int_values_ignored() -> None:
    """Non-int usage values are dropped; valid ints are kept."""
    st = _parse_stream(json.dumps({
        "type": "result",
        "usage": {"inputTokens": 10, "outputTokens": "bad", "cacheRead": 5},
    }))
    assert st.usage == {"inputTokens": 10, "cacheRead": 5}


# ---------------------------------------------------------------------------
# retry loop — monkeypatched run_cli, stubbed time.sleep
# ---------------------------------------------------------------------------

def _make_cursor(bin: str = "cursor-agent", **cfg) -> CursorAgent:
    cfg.setdefault("model", "auto")
    cfg["bin"] = bin
    return CursorAgent(AgentSpec(name="cursor", config=cfg))


def _outcome(exit_code: int, stdout: str = "") -> edit_driver.CliOutcome:
    return edit_driver.CliOutcome(exit_code=exit_code, stdout=stdout, duration_ms=10)


def test_retry_loop_retries_on_retryable_error_and_succeeds(monkeypatch, tmp_path) -> None:
    """If the first attempt returns a retryable error, the loop retries and
    returns the second (successful) attempt. time.sleep is stubbed so the
    test is fast."""
    calls: list[int] = []
    ok_stdout = json.dumps({"type": "result", "result": "done"})

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        attempt = len(calls) + 1
        calls.append(attempt)
        if attempt == 1:
            return _outcome(1, stdout=json.dumps({"type": "result",
                                                   "result": "rate limit exceeded"}))
        return _outcome(0, stdout=ok_stdout)

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)
    monkeypatch.setattr("beagle.agents.cursor.time.sleep", lambda s: None)

    res = _make_cursor(max_attempts=3).edit("do it", tmp_path / "ws")

    assert res.ok
    assert res.text == "done"
    assert len(calls) == 2  # retried once


def test_retry_loop_stops_after_max_attempts(monkeypatch, tmp_path) -> None:
    """After max_attempts retryable failures, the last result is returned and
    no further attempts are made."""
    calls: list[int] = []
    retryable_stdout = json.dumps({"type": "result", "result": "rate limit exceeded"})

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        calls.append(len(calls) + 1)
        return _outcome(1, stdout=retryable_stdout)

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)
    monkeypatch.setattr("beagle.agents.cursor.time.sleep", lambda s: None)

    res = _make_cursor(max_attempts=3).edit("do it", tmp_path / "ws")

    assert not res.ok
    assert len(calls) == 3  # exactly max_attempts attempts made


def test_retry_loop_does_not_retry_non_retryable_error(monkeypatch, tmp_path) -> None:
    """A non-retryable failure (missing binary, bad output) does not trigger a
    retry — the first attempt's result is returned immediately."""
    calls: list[int] = []

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        calls.append(1)
        return _outcome(1, stdout=json.dumps({"type": "result", "result": "syntax error"}))

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)
    monkeypatch.setattr("beagle.agents.cursor.time.sleep", lambda s: None)

    res = _make_cursor(max_attempts=4).edit("do it", tmp_path / "ws")

    assert not res.ok
    assert len(calls) == 1  # no retry on non-retryable error


def test_retry_loop_does_not_retry_timeout_exit_124(monkeypatch, tmp_path) -> None:
    """Exit 124 (hard timeout) is not retryable — 'timeout after Xs' doesn't
    match the retryable regex."""
    calls: list[int] = []

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        calls.append(1)
        return _outcome(124)  # from edit_driver on TimeoutExpired

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)
    # give the outcome an error that matches what run_cli sets
    import beagle.agents.cursor as cursor_mod
    _real_run_cli = cursor_mod.run_cli

    def _timeout_outcome(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        calls.append(1)
        return edit_driver.CliOutcome(exit_code=124, duration_ms=5,
                                      error=f"timeout after {timeout_s}s")

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _timeout_outcome)
    monkeypatch.setattr("beagle.agents.cursor.time.sleep", lambda s: None)

    res = _make_cursor(max_attempts=3).edit("x", tmp_path / "ws", timeout_s=1)

    assert res.exit_code == 124
    assert len(calls) == 1  # not retried


def test_retry_loop_max_attempts_zero_runs_once(monkeypatch, tmp_path) -> None:
    """max_attempts=0 is treated as 1 via max(1, …), so exactly one attempt is made."""
    calls: list[int] = []

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        calls.append(1)
        return _outcome(0, stdout=json.dumps({"type": "result", "result": "ok"}))

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)

    res = _make_cursor(max_attempts=0).edit("x", tmp_path / "ws")

    assert res.ok
    assert len(calls) == 1


def test_retry_backoff_sleep_is_proportional_to_attempt(monkeypatch, tmp_path) -> None:
    """The sleep between attempts is ``backoff_base_s * attempt``, so the second
    attempt sleeps 2× the base, not just 1×."""
    sleep_calls: list[float] = []
    retryable_stdout = json.dumps({"type": "result", "result": "quota exceeded"})

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        return _outcome(1, stdout=retryable_stdout)

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)
    monkeypatch.setattr("beagle.agents.cursor.time.sleep",
                        lambda s: sleep_calls.append(s))

    _make_cursor(max_attempts=3, backoff_base_s=5.0).edit("x", tmp_path / "ws")

    # 3 attempts → 2 sleeps (after attempt 1 and after attempt 2)
    assert sleep_calls == [5.0 * 1, 5.0 * 2]


# ---------------------------------------------------------------------------
# log_path is wired end-to-end: Editor.edit() → run_cli → EditResult
# ---------------------------------------------------------------------------

def test_log_path_is_wired_through_edit(monkeypatch, tmp_path) -> None:
    """log_path flows caller → CursorAgent.edit() → run_cli (per-iteration logs),
    and the resulting log file path is echoed back on the EditResult. The vendored
    pipeline.py passes log_path= to meta_agent.run(); the shim forwards it here."""
    seen: dict = {}
    log = tmp_path / "stage.log"

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, log_path=None, **kw):  # noqa: ANN001
        seen["log_path"] = log_path
        out = _outcome(0, stdout=json.dumps({"type": "result", "result": "ok"}))
        return edit_driver.CliOutcome(exit_code=0, stdout=out.stdout, duration_ms=10,
                                      log_path=log_path)

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)

    res = _make_cursor().edit("x", tmp_path / "ws", log_path=log)

    assert res.ok
    assert seen["log_path"] == log       # forwarded into run_cli
    assert res.log_path == log           # and echoed back on the result


# ---------------------------------------------------------------------------
# a stream-level is_error must surface as failure (EditResult.ok would else lie)
# ---------------------------------------------------------------------------

def test_stream_is_error_makes_result_not_ok(monkeypatch, tmp_path) -> None:
    """cursor-agent can exit 0 yet flag the turn failed via ``is_error`` in the
    result event. That must map to a failed EditResult (ok=False, exit_code=1,
    error mentions is_error) — otherwise a refusal reads as success."""
    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, log_path=None, **kw):  # noqa: ANN001
        return _outcome(0, stdout=json.dumps(
            {"type": "result", "result": "refused", "is_error": True}))

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)

    res = _make_cursor(max_attempts=1).edit("x", tmp_path / "ws")

    assert not res.ok
    assert res.exit_code == 1
    assert "is_error" in (res.error or "")
