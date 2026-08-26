"""Generic normalizer for flat ``role``/``content`` chat traces.

This is the agent-agnostic fallback and the parity path with the reference
``adb`` tool's default ``openai_messages`` format. It accepts either:

* a JSON object ``{"trace_id"?: ..., "messages": [{"role", "content", ...}]}``
  (the reference's shape), or
* a ``*.messages.jsonl`` file with one message object per line (the shape
  ``agents.monet.trajectory`` and our :meth:`CanonicalTrajectory.messages`
  emit), so a normalized export round-trips back in.

Turns are reconstructed by grouping each ``assistant`` message with the
``tool``/``function`` result messages that follow it. OpenAI-style
``tool_calls`` on an assistant message and the matching ``tool`` results are
threaded together by ``tool_call_id`` when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model import CanonicalTrajectory, ToolCall, Turn
from ..normalizer import NormalizeError, TrajectoryNormalizer, register, trace_id_from_path


def _load_messages(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") and '"messages"' in stripped[:4096]:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise NormalizeError(f"{path}: not valid JSON: {exc}") from exc
        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            raise NormalizeError(f"{path}: 'messages' must be a non-empty array")
        return messages, obj.get("trace_id")
    # Otherwise treat as JSONL (one message per line).
    messages = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            messages.append(obj)
    if not messages:
        raise NormalizeError(f"{path}: no message objects found")
    return messages, None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Anthropic-style content blocks
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return "" if content is None else str(content)


class OpenAiMessagesNormalizer(TrajectoryNormalizer):
    name = "openai_messages"

    def normalize(self, path: Path) -> CanonicalTrajectory:
        messages, trace_id = _load_messages(path)
        turns: list[Turn] = []
        cur: Turn | None = None
        pending: dict[str, ToolCall] = {}  # tool_call_id -> call awaiting result

        def open_turn() -> Turn:
            nonlocal cur
            idx = turns[-1].index + 1 if turns else 0
            cur = Turn(index=idx)
            turns.append(cur)
            return cur

        for msg in messages:
            role = msg.get("role")
            if role in ("system", "user"):
                # A user/system message starts a fresh exchange; skip system
                # preamble but let user turns bound the conversation.
                cur = None
                continue
            if role == "assistant":
                turn = open_turn()
                turn.text = _content_to_text(msg.get("content"))
                turn.stop_reason = msg.get("stop_reason")
                turn.error = msg.get("error")
                if isinstance(msg.get("usage"), dict):
                    turn.usage = {
                        k: int(v) for k, v in msg["usage"].items() if isinstance(v, int)
                    }
                for raw in msg.get("tool_calls") or []:
                    call = _parse_tool_call(raw)
                    turn.tool_calls.append(call)
                    pending[call.id] = call
            elif role in ("tool", "function"):
                target = pending.get(str(msg.get("tool_call_id") or msg.get("id") or ""))
                if target is None and cur is not None and cur.tool_calls:
                    target = cur.tool_calls[-1]  # positional fallback
                if target is not None:
                    content = msg.get("content")
                    target.result = content if isinstance(content, str) else json.dumps(content)
                    target.is_error = bool(msg.get("is_error"))
        return CanonicalTrajectory(
            trace_id=str(trace_id) if trace_id else trace_id_from_path(path),
            source=self.name,
            turns=turns,
            terminal="end" if turns else None,
            final_usage=None,
            metadata={},
        )

    @classmethod
    def sniff(cls, path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                head = f.read(4096)
        except OSError:
            return False
        stripped = head.lstrip()
        if stripped.startswith("{") and '"messages"' in head:
            return True
        # JSONL of message rows: first non-empty line is an object with a role.
        for line in head.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return False
            return isinstance(obj, dict) and "role" in obj
        return False


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    # Accept both our flat shape and OpenAI's nested {"function": {...}}.
    nested = raw.get("function")
    fn = nested if isinstance(nested, dict) else raw
    name = str(fn.get("name") or raw.get("name") or "")
    args = fn.get("arguments", raw.get("arguments"))
    arguments: dict[str, Any] | None = None
    arguments_raw: str | None = None
    if isinstance(args, dict):
        arguments = args
    elif isinstance(args, str):
        arguments_raw = args
        try:
            arguments = json.loads(args) if args else None
        except json.JSONDecodeError:
            pass
    return ToolCall(id=str(raw.get("id") or name), name=name, arguments=arguments,
                    arguments_raw=arguments_raw)


register(OpenAiMessagesNormalizer())
