"""The cursor editor — stream-json parsing + the edit() driver/retry loop.

Hermetic: the pure parser is unit-tested; edit() is exercised against a *fake* cursor-agent
script (a real subprocess through run_cli) for success / missing-binary / timeout, and via a
monkeypatched run_cli for argv construction. No real cursor-agent, no network."""

from __future__ import annotations

import json
import stat
import textwrap

from beagle.agents.core import edit_driver
from beagle.agents.core.spec import AgentSpec
from beagle.agents.cursor import (
    CursorAgent,
    _looks_retryable,
    _parse_stream,
)


# --- pure stream-json parsing -------------------------------------------------

def test_parse_stream_assistant_toolcall_result() -> None:
    st = _parse_stream("\n".join([
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}),
        json.dumps({"type": "tool_call", "subtype": "completed", "tool_call": {"name": "edit_file"}}),
        json.dumps({"type": "result", "result": "final", "duration_ms": 42,
                    "usage": {"inputTokens": 100, "outputTokens": 20}, "is_error": False}),
    ]))
    assert st.assistant_text == "hi"
    assert st.tool_calls == [{"name": "edit_file"}]
    assert st.usage == {"inputTokens": 100, "outputTokens": 20}
    assert st.duration_ms == 42
    assert st.final_text() == "hi"          # assistant text beats result when there's no plan body


def test_final_text_prefers_nontrivial_plan_body() -> None:
    plan = "# Plan\n" + "do the thing. " * 30   # > 200 chars, real markdown
    st = _parse_stream("\n".join([
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "assist"}]}}),
        json.dumps({"type": "tool_call", "subtype": "completed",
                    "tool_call": {"createPlanToolCall": {"args": {"plan": plan}}}}),
        json.dumps({"type": "result", "result": "res"}),
    ]))
    assert st.final_text() == plan


def test_final_text_skips_trivial_plan_body() -> None:
    st = _parse_stream("\n".join([
        json.dumps({"type": "tool_call", "subtype": "completed",
                    "tool_call": {"createPlanToolCall": {"args": {"plan": "see assistant message " + "x" * 200}}}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "assist"}]}}),
    ]))
    assert st.final_text() == "assist"      # stub-marker plan is skipped → assistant text


def test_parse_stream_skips_malformed_lines() -> None:
    st = _parse_stream("not json at all\n\n" + json.dumps({"type": "result", "result": "ok"}))
    assert st.result_text == "ok"


def test_looks_retryable() -> None:
    assert _looks_retryable("", "exit 1: 429 Too Many Requests", "")
    assert _looks_retryable(None, None, "…\nrate limit exceeded\n…")
    assert not _looks_retryable("all good", None, "clean output")


# --- edit() over a fake cursor-agent (real subprocess) ------------------------

def _fake_cli(tmp_path, body: str):
    p = tmp_path / "fake-cursor"
    p.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _cursor(bin_path, **cfg) -> CursorAgent:
    cfg = {"bin": str(bin_path), "model": cfg.pop("model", "auto"), **cfg}
    return CursorAgent(AgentSpec(name="cursor", config=cfg))


def test_edit_success_over_fake_cli(tmp_path) -> None:
    fake = _fake_cli(tmp_path, '''
        import sys, json
        prompt = sys.stdin.read().strip()
        print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "did: " + prompt}]}}))
        print(json.dumps({"type": "tool_call", "subtype": "completed", "tool_call": {"name": "edit"}}))
        print(json.dumps({"type": "result", "result": "ok", "duration_ms": 7, "usage": {"inputTokens": 5}}))
    ''')
    ws = tmp_path / "ws"
    res = _cursor(fake).edit("fix the bug", ws)

    assert res.ok and res.exit_code == 0
    assert res.text == "did: fix the bug"           # prompt reached the CLI via stdin
    assert res.tool_calls == [{"name": "edit"}] and res.usage == {"inputTokens": 5}
    assert ws.is_dir()                               # workspace created


def test_edit_missing_binary_is_exit_127(tmp_path) -> None:
    res = _cursor(tmp_path / "does-not-exist").edit("x", tmp_path / "ws")
    assert res.exit_code == 127 and not res.ok and "not found" in (res.error or "")


def test_edit_timeout_is_exit_124(tmp_path) -> None:
    fake = _fake_cli(tmp_path, "import time; time.sleep(5)")
    res = _cursor(fake).edit("x", tmp_path / "ws", timeout_s=1)
    assert res.exit_code == 124 and not res.ok and "timeout" in (res.error or "")


def test_edit_no_model_is_exit_2(tmp_path) -> None:
    agent = CursorAgent(AgentSpec(name="cursor", config={"bin": "cursor-agent"}))  # no model anywhere
    res = agent.edit("x", tmp_path / "ws")
    assert res.exit_code == 2 and "no model" in (res.error or "")


def test_edit_builds_argv_plan_mode_model_and_stdin(monkeypatch, tmp_path) -> None:
    seen: dict = {}

    def _fake_run_cli(argv, *, prompt, cwd, timeout_s, **kw):  # noqa: ANN001
        seen.update(argv=argv, prompt=prompt, cwd=str(cwd))
        return edit_driver.CliOutcome(exit_code=0, stdout=json.dumps({"type": "result", "result": "ok"}))

    monkeypatch.setattr("beagle.agents.cursor.run_cli", _fake_run_cli)
    ws = tmp_path / "ws"
    _cursor("cursor-agent").edit("do it", ws, plan_mode=True, model="gpt-5.5", extra_args=["--foo"])

    argv = seen["argv"]
    assert argv[:6] == ["cursor-agent", "-p", "--force", "--output-format", "stream-json", "--workspace"]
    assert argv[argv.index("--model") + 1] == "gpt-5.5"
    assert "--mode" in argv and argv[argv.index("--mode") + 1] == "plan"
    assert "--foo" in argv                            # extra_args appended
    assert seen["prompt"] == "do it" and seen["cwd"] == str(ws)   # prompt via stdin, cwd = workspace


def test_installed_version_probes_cursor_agent(monkeypatch) -> None:
    # installed_version() runs the SAME bin edit() uses (cursor-agent, not cursor) and returns the
    # first --version line — used by the config version gate. Hermetic: fake subprocess.run.
    import subprocess
    from types import SimpleNamespace

    seen = {}

    def _fake_run(argv, **kw):  # noqa: ANN001
        seen["argv"] = argv
        return SimpleNamespace(stdout="2026.08.04-aaa8809\ndeadbeef\nx64\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    agent = CursorAgent(AgentSpec(name="cursor"))
    assert agent.installed_version() == "2026.08.04-aaa8809"   # first line only
    assert seen["argv"][:2] == ["cursor-agent", "--version"]   # the agent CLI, not `cursor`

    # config bin override is honored; a missing binary → None (nothing to verify against)
    def _boom(argv, **kw):  # noqa: ANN001
        raise FileNotFoundError(argv[0])
    monkeypatch.setattr(subprocess, "run", _boom)
    assert CursorAgent(AgentSpec(name="cursor", config={"bin": "nope"})).installed_version() is None


def test_agent_base_installed_version_is_none_by_default() -> None:
    # Source-versioned agents (monet) inherit the base None → exempt from the version gate.
    from beagle.agents.monet import MonetAgent
    assert MonetAgent(AgentSpec(name="monet")).installed_version() is None
