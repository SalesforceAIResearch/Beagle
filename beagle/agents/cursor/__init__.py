"""Cursor — a black-box evolver (the ``cursor-agent`` CLI driven as an :class:`Editor`).

The Cursor CLI is an external coding agent driven as a subprocess. Its internals are opaque,
so it *cannot be evolved* — it is an ``Editor`` only. The evolution algorithm calls
:meth:`edit` with its own prompts (analyze / implement / review …), in plan or edit mode,
against the target agent's source worktree. Edits are left in the worktree (git state); a
separate step snapshots the diff. Model routing is via Cursor's own account backend (its auth
token is a credential from the environment) — no gateway. Config knobs (``model``, ``bin``,
``timeout``, ``max_attempts``, ``backoff_base_s``, ``extra_args``) come from ``agent.config``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beagle.agents.core.base import Agent, EditResult, Editor
from beagle.agents.core.edit_driver import run_cli
from beagle.agents.core.registry import register
from beagle.types import Transparency

# --- stream-json parsing ------------------------------------------------------
# cursor-agent emits `--output-format stream-json`: one JSON event per line. We look for four
# shapes — `assistant` (text), `tool_call` (completed → tool_calls + CreatePlan body), and
# `result` (final text + usage + duration + is_error). Unknown events are ignored.

_MIN_PLAN_BODY_CHARS = 200
_PLAN_STUB_MARKERS = (
    "see assistant message", "see the assistant message", "see assistant text",
    "final plan is rendered", "in the final assistant message",
)
#: transient failures (rate-limit / model-flap / 5xx) worth a backoff+retry.
_RETRYABLE_RE = re.compile(
    r"Model name is not valid|AI Model Not Found|rate.?limit|ratelimit|\b429\b|too many requests|"
    r"quota|overloaded|temporarily unavailable|timed out|ECONNRESET|ETIMEDOUT|service unavailable|\b50[234]\b",
    re.IGNORECASE,
)


def _is_trivial_plan_body(body: str | None) -> bool:
    """A CreatePlan body counts as the plan only if it's a real markdown plan: long enough
    and without a "see other channel" stub marker."""
    if not body:
        return True
    s = body.strip()
    if len(s) < _MIN_PLAN_BODY_CHARS:
        return True
    low = s.lower()
    return any(marker in low for marker in _PLAN_STUB_MARKERS)


def _extract_plan_body(evt: dict[str, Any]) -> str | None:
    """The ``args.plan`` string of a ``createPlanToolCall`` tool_call event, else None."""
    if evt.get("type") != "tool_call":
        return None
    tc = evt.get("tool_call") or {}
    cp = tc.get("createPlanToolCall") if isinstance(tc, dict) else None
    if not isinstance(cp, dict):
        return None
    args = cp.get("args") or {}
    plan = args.get("plan") if isinstance(args, dict) else None
    return plan if isinstance(plan, str) else None


def _select_final_text(*, plan_body_completed: str | None, plan_body_started: str | None,
                       assistant_text: str, result_text: str) -> str:
    """Pick the most plan-like text across Cursor's channels: a non-trivial CreatePlan body
    (completed > started) > the last assistant text > the result text."""
    for body in (plan_body_completed, plan_body_started):
        if not _is_trivial_plan_body(body):
            assert body is not None  # narrowed by the trivial check
            return body
    return assistant_text or result_text


@dataclass
class _StreamState:
    assistant_text: str = ""
    result_text: str = ""
    plan_body_completed: str | None = None
    plan_body_started: str | None = None
    duration_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False

    def final_text(self) -> str:
        return _select_final_text(
            plan_body_completed=self.plan_body_completed, plan_body_started=self.plan_body_started,
            assistant_text=self.assistant_text, result_text=self.result_text)


def _consume_event(state: _StreamState, evt: dict[str, Any]) -> None:
    etype = evt.get("type")
    if etype == "tool_call":
        if evt.get("subtype") == "completed":
            state.tool_calls.append(evt.get("tool_call") or {})
        body = _extract_plan_body(evt)
        if body is not None:
            if evt.get("subtype") == "completed":
                state.plan_body_completed = body
            elif evt.get("subtype") == "started":
                state.plan_body_started = body
    elif etype == "assistant":
        for part in (evt.get("message") or {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                state.assistant_text = part.get("text") or state.assistant_text  # last non-empty wins
    elif etype == "result":
        if evt.get("result") is not None:
            state.result_text = evt.get("result") or state.result_text
        if evt.get("duration_ms") is not None:
            state.duration_ms = int(evt["duration_ms"])
        u = evt.get("usage")
        if isinstance(u, dict):
            state.usage = {k: int(v) for k, v in u.items() if isinstance(v, int)}
        state.is_error = bool(evt.get("is_error")) or state.is_error


def _parse_stream(stdout: str) -> _StreamState:
    state = _StreamState()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(evt, dict):
            _consume_event(state, evt)
    return state


def _looks_retryable(text: str | None, error: str | None, stdout: str) -> bool:
    return bool(_RETRYABLE_RE.search(f"{error or ''}\n{text or ''}\n{stdout[-8000:]}"))


@register("cursor")
class CursorAgent(Agent, Editor):
    """External ``cursor-agent`` CLI — a black-box editor (evolver)."""

    transparency = Transparency.BLACK_BOX

    def installed_version(self) -> str | None:
        """The version of the agent binary this adapter actually runs — ``cursor-agent --version``
        (first line, e.g. ``2026.08.04-aaa8809``). This is ``cursor-agent`` (the CLI coding agent),
        NOT ``cursor`` (the IDE launcher, a different tool/version). Probes the SAME ``bin`` that
        :meth:`edit` invokes (``agent.config['bin']``, default ``cursor-agent``). Returns ``None``
        if the binary isn't found (nothing to verify against)."""
        import subprocess

        bin_ = str((self.config or {}).get("bin", "cursor-agent"))
        try:
            out = subprocess.run([bin_, "--version"], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        lines = (out.stdout or "").strip().splitlines()
        return lines[0].strip() if lines else None

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
        """Run ``instruction`` in ``workspace`` via ``cursor-agent`` → an :class:`EditResult`.

        Edits are left in ``workspace`` (git state); ``plan_mode`` → ``--mode plan`` (native
        read-only). Retries transient failures (rate-limit / model-flap / 5xx) with backoff.
        """
        cfg = self.config or {}
        model = model or cfg.get("model") or (self.spec.model.name if self.spec.model else None)
        if not model:
            return EditResult(exit_code=2, error="cursor edit(): no model (set model= or agent.config.model)")
        if self.spec.model is not None and self.spec.model.reasoning_effort:
            # cursor is the special case: reasoning effort rides in the model slug, not a flag.
            return EditResult(exit_code=2, error=(
                "cursor edit(): cursor encodes reasoning effort in the model slug "
                "(e.g. 'gpt-5.5-high'), not a separate reasoning_effort — set it in the model name."))

        argv = [str(cfg.get("bin", "cursor-agent")), "-p", "--force",
                "--output-format", "stream-json", "--workspace", str(workspace), "--model", model]
        if plan_mode:
            argv += ["--mode", "plan"]
        argv += list(extra_args or []) + list(cfg.get("extra_args") or [])

        timeout = int(timeout_s or cfg.get("timeout", 1800))
        max_attempts = max(1, int(cfg.get("max_attempts", 4)))
        backoff = float(cfg.get("backoff_base_s", 8.0))
        Path(workspace).mkdir(parents=True, exist_ok=True)

        result: EditResult | None = None
        for attempt in range(1, max_attempts + 1):
            outcome = run_cli(argv, prompt=instruction, cwd=workspace, timeout_s=timeout,
                              log_path=log_path)
            st = _parse_stream(outcome.stdout)
            # A stream-level ``is_error`` (the CLI exits 0 but flags the turn failed) must
            # surface as a failure — otherwise EditResult.ok would lie and the retry/caller
            # would treat a refusal as success.
            error = outcome.error or ("cursor-agent stream reported is_error" if st.is_error else None)
            exit_code = outcome.exit_code or (1 if st.is_error else 0)
            result = EditResult(
                text=st.final_text(), exit_code=exit_code, usage=st.usage,
                tool_calls=st.tool_calls, duration_ms=st.duration_ms or outcome.duration_ms,
                error=error, log_path=outcome.log_path,
            )
            if result.ok or attempt == max_attempts \
                    or not _looks_retryable(result.text, result.error, outcome.stdout):
                return result
            time.sleep(backoff * attempt)
        return result  # type: ignore[return-value]  # loop runs ≥1×, so never None


__all__ = ["CursorAgent"]
