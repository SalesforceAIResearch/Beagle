"""Trace view + chunking — the "trace as numbered messages" environment.

The blog presents a trace to proposers as a list of numbered messages and lets
each proposer pick *what* goes into each chunk (`context`) and *how big* the
chunk is (`chunk_size`, `chunk_format`). Tool-error checks want small chunks of
just tool I/O; semantic checks want larger chunks with the assistant's text.

:class:`TraceView` wraps a :class:`CanonicalTrajectory` with its numbered
messages and the index maps rule proposers need (turn → assistant row,
tool-call-id → tool row), and :meth:`TraceView.chunks` renders chunks for LLM
proposers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import CanonicalTrajectory

_MAX_FIELD = 4000  # cap per rendered field so one huge tool output can't blow context


@dataclass
class ContextSpec:
    """What content a proposer's chunks should include (blog ``context``)."""

    iterate_by: str = "message"
    lm_response: bool = True
    tool_call: bool = True
    user_input: bool = True
    system_prompt: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ContextSpec":
        d = d or {}
        return cls(
            iterate_by=d.get("iterate_by", "message"),
            lm_response=d.get("lm_response", True),
            tool_call=d.get("tool_call", True),
            user_input=d.get("user_input", True),
            system_prompt=d.get("system_prompt", False),
        )


@dataclass
class Chunk:
    """A rendered batch of messages plus the message indices it contains."""

    indices: list[int]
    text: str


@dataclass
class TraceView:
    traj: CanonicalTrajectory
    messages: list[dict[str, Any]]
    assistant_row: dict[int, int] = field(default_factory=dict)  # turn -> msg index
    tool_row: dict[str, int] = field(default_factory=dict)  # tool_call_id -> msg index

    @classmethod
    def build(cls, traj: CanonicalTrajectory) -> "TraceView":
        messages = traj.messages()
        assistant_row: dict[int, int] = {}
        tool_row: dict[str, int] = {}
        for row in messages:
            if row["role"] == "assistant":
                assistant_row.setdefault(row["turn"], row["index"])
            elif row["role"] == "tool":
                tool_row[row["tool_call_id"]] = row["index"]
        return cls(traj, messages, assistant_row, tool_row)

    def msg_index_for_turn(self, turn_index: int) -> int | None:
        return self.assistant_row.get(turn_index)

    def msg_index_for_tool(self, tool_call_id: str) -> int | None:
        return self.tool_row.get(tool_call_id)

    def chunks(
        self,
        *,
        size: int = 8,
        fmt: str = "xml",
        context: ContextSpec | None = None,
    ) -> list[Chunk]:
        context = context or ContextSpec()
        rendered: list[tuple[int, str]] = []
        for row in self.messages:
            piece = _render_row(row, context, fmt)
            if piece:
                rendered.append((row["index"], piece))
        chunks: list[Chunk] = []
        for i in range(0, len(rendered), max(1, size)):
            batch = rendered[i : i + size]
            chunks.append(Chunk(indices=[idx for idx, _ in batch],
                                text="\n".join(text for _, text in batch)))
        return chunks


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= _MAX_FIELD else text[:_MAX_FIELD] + " …[clipped]"


def _render_row(row: dict[str, Any], ctx: ContextSpec, fmt: str) -> str:
    role = row["role"]
    if role == "system" and not ctx.system_prompt:
        return ""
    if role == "user" and not ctx.user_input:
        return ""
    if role == "tool" and not ctx.tool_call:
        return ""

    idx, turn = row["index"], row.get("turn")
    if fmt == "markdown":
        return _render_markdown(row, ctx, idx, turn)
    return _render_xml(row, ctx, idx, turn)


def _render_xml(row: dict[str, Any], ctx: ContextSpec, idx: int, turn: Any) -> str:
    role = row["role"]
    if role == "assistant":
        parts = []
        if ctx.lm_response and row.get("content", "").strip():
            parts.append(f"  <text>{_clip(row['content'])}</text>")
        if ctx.tool_call:
            for c in row.get("tool_calls", []):
                parts.append(f"  <tool_call name=\"{c['name']}\">{_clip(_args(c))}</tool_call>")
        if not parts:
            return ""
        body = "\n".join(parts)
        return f'<message index="{idx}" role="assistant" turn="{turn}">\n{body}\n</message>'
    if role == "tool":
        err = "true" if row.get("is_error") else "false"
        return (
            f'<message index="{idx}" role="tool" name="{row.get("name", "")}" '
            f'is_error="{err}">{_clip(row.get("content", ""))}</message>'
        )
    return f'<message index="{idx}" role="{role}">{_clip(row.get("content", ""))}</message>'


def _render_markdown(row: dict[str, Any], ctx: ContextSpec, idx: int, turn: Any) -> str:
    role = row["role"]
    if role == "assistant":
        lines = [f"### [{idx}] assistant (turn {turn})"]
        if ctx.lm_response and row.get("content", "").strip():
            lines.append(_clip(row["content"]))
        if ctx.tool_call:
            for c in row.get("tool_calls", []):
                lines.append(f"→ call {c['name']}({_clip(_args(c))})")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)
    if role == "tool":
        tag = "ERROR" if row.get("is_error") else "ok"
        return f"### [{idx}] tool:{row.get('name', '')} ({tag})\n{_clip(row.get('content', ''))}"
    return f"### [{idx}] {role}\n{_clip(row.get('content', ''))}"


def _args(call: dict[str, Any]) -> str:
    import json

    args = call.get("arguments")
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)
