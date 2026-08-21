"""Monet normalizer — tailored to monet's ``stream-json`` trajectory format.

Monet's ``--output-format stream-json`` writes the raw provider SSE event
stream as NDJSON (one event per line): every text fragment, tool-arg JSON
chunk, and block boundary. This normalizer reduces that event stream to the
agent-agnostic :class:`CanonicalTrajectory`.

Why not reuse ``agents.monet.trajectory.reduce_stream_to_turns``?  That reducer
keys assistant text on an integer ``index`` carried by ``content_block_start`` /
``text_delta`` events. Current monet builds emit ``text_delta`` events with **no
``index``** (and no ``content_block_start`` at all) — verified across the
``swe-bench-verified`` and ``terminal-bench-v2.1`` runs in ``results/`` — so
that reducer drops every assistant free-text fragment (tool calls survive only
because ``tool_use_delta`` still carries an ``index``). The detectors and the
LLM ``ask`` view depend on that text, so this normalizer owns a reduction that
captures text whether or not an ``index`` is present, while keeping the same
turn/tool-call model and robustness (malformed lines skipped, truncated files
yield whatever completed). The event vocabulary mirrors that module's
documented schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..model import CanonicalTrajectory, ToolCall, Turn
from ..normalizer import TrajectoryNormalizer, register, trace_id_from_path

# Events whose presence on the first lines fingerprints a monet stream-json
# file versus any other JSONL trace.
_MONET_MARKERS = (
    '"type":"session_meta"',
    '"type": "session_meta"',
    '"type":"tool_use_delta"',
    '"type":"turn_complete"',
    '"type":"text_delta"',
)

# Source terminal markers that mean "the run ended" (vs. a truncated file).
_TERMINAL_EVENTS = ("max_turns_reached", "max_tool_errors", "tool_denied")


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


class MonetNormalizer(TrajectoryNormalizer):
    name = "monet"

    def normalize(self, path: Path) -> CanonicalTrajectory:
        session_id: str | None = None
        protocol_version: int | None = None
        final_usage: dict[str, int] | None = None
        terminal: str | None = None

        turns: list[Turn] = []
        cur = Turn(index=0)
        text_parts: list[str] = []
        block_to_call: dict[int, ToolCall] = {}  # tool_use block index -> call
        arg_chunks: dict[str, list[str]] = {}  # tool id -> partial-json fragments
        call_by_id: dict[str, ToolCall] = {}

        def flush(stop_reason: str | None) -> None:
            nonlocal cur, text_parts, block_to_call
            cur.text = "".join(text_parts)
            for call in cur.tool_calls:
                if call.arguments is None and call.id in arg_chunks:
                    raw = "".join(arg_chunks[call.id])
                    call.arguments_raw = raw
                    try:
                        call.arguments = json.loads(raw) if raw else None
                    except json.JSONDecodeError:
                        pass
            cur.stop_reason = stop_reason
            turns.append(cur)
            cur = Turn(index=cur.index + 1)
            text_parts = []
            block_to_call = {}

        for ev in _iter_events(path):
            t = ev.get("type")

            if t == "session_meta":
                session_id = ev.get("session_id") or session_id
                pv = ev.get("protocol_version")
                if isinstance(pv, int):
                    protocol_version = pv

            elif t == "text_delta":
                # Robust to both schemas: index-less (current monet) and indexed.
                txt = ev.get("text", "")
                if isinstance(txt, str):
                    text_parts.append(txt)

            elif t == "tool_use_start":
                idx = ev.get("index")
                tid = ev.get("toolId") or ""
                if isinstance(idx, int) and tid:
                    call = ToolCall(id=tid, name=ev.get("toolName") or "")
                    block_to_call[idx] = call
                    call_by_id[tid] = call
                    cur.tool_calls.append(call)
                    arg_chunks[tid] = []

            elif t == "tool_use_delta":
                idx = ev.get("index")
                chunk = ev.get("partialJson", "")
                if isinstance(idx, int) and isinstance(chunk, str):
                    target = block_to_call.get(idx)
                    if target is not None:
                        arg_chunks.setdefault(target.id, []).append(chunk)

            elif t == "tool_start":
                target = call_by_id.get(ev.get("id") or "")
                if target is not None and isinstance(ev.get("input"), dict):
                    target.arguments = ev["input"]

            elif t == "tool_output":
                target = call_by_id.get(ev.get("id") or "")
                if target is not None:
                    out = ev.get("output")
                    target.result = out if isinstance(out, str) else json.dumps(out)
                    target.is_error = bool(ev.get("isError"))

            elif t == "tool_end":
                target = call_by_id.get(ev.get("id") or "")
                if target is not None and not target.is_error:
                    target.is_error = bool(ev.get("isError"))

            elif t == "message_delta":
                usage = ev.get("usage")
                if isinstance(usage, dict):
                    cur.usage = {k: int(v) for k, v in usage.items() if isinstance(v, int)}

            elif t == "stream_error":
                err = ev.get("error")
                if isinstance(err, str):
                    cur.error = err

            elif t == "empty_assistant_turn":
                if isinstance(ev.get("turn"), int):
                    cur.index = ev["turn"]
                flush(stop_reason=None)

            elif t == "turn_complete":
                if isinstance(ev.get("turn"), int):
                    cur.index = ev["turn"]
                flush(stop_reason=ev.get("stopReason"))

            elif t == "usage":
                usage = ev.get("usage")
                if isinstance(usage, dict):
                    final_usage = {k: int(v) for k, v in usage.items() if isinstance(v, int)}

            elif t in _TERMINAL_EVENTS:
                terminal = t

            elif t == "turn_done":
                terminal = terminal or "turn_done"

        # An unflushed turn with content means the file was truncated mid-turn.
        if text_parts or cur.tool_calls:
            flush(stop_reason=None)

        return CanonicalTrajectory(
            trace_id=trace_id_from_path(path),
            source=self.name,
            turns=turns,
            terminal=terminal,
            final_usage=final_usage,
            metadata={"session_id": session_id, "protocol_version": protocol_version},
        )

    @classmethod
    def sniff(cls, path: Path) -> bool:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                if any(marker in line for marker in _MONET_MARKERS):
                    return True
        return False


register(MonetNormalizer())
