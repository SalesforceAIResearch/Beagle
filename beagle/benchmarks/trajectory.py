"""Agent trajectory → harbor's ATIF (``agent/trajectory.json``).

**ATIF is beagle's canonical trajectory format** — it is harbor's on-disk contract
(Agent Trajectory Interchange Format), and every agent/benchmark aligns to it. Each
agent's *native* trajectory (monet's stream-json, mini-swe's, …) is converted to ATIF by
a **per-format converter keyed on ``TrajectoryRef.format``** — one converter per format,
not per agent×harness (M + N). The harbor shim (:mod:`beagle.benchmarks.harness._harbor_agent`)
calls :func:`write_trajectory_json` so a trial carries ``agent/trajectory.json`` exactly
like a native-harbor trial — the artifact tree stays byte-compatible (honor the harness).

Built on harbor's own ATIF pydantic models (imported lazily — harbor is optional), so the
output is valid by construction and round-trips through harbor's ``validate_trajectory``.

The monet reducer converts monet's stream-json events to ATIF steps; it targets ATIF directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from beagle.agents.core.usage import Usage

#: format string -> converter(logs_dir, *, instruction, agent_name, agent_version, model_name) -> Trajectory|None
_CONVERTERS: dict[str, Callable[..., Any]] = {}


def register_converter(fmt: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _CONVERTERS[fmt] = fn
        return fn

    return _deco


def write_trajectory_json(
    logs_dir: str | Path,
    *,
    trajectory_format: str,
    instruction: str,
    agent_name: str,
    agent_version: str = "",
    model_name: str | None = None,
) -> Path | None:
    """Convert the agent's native trajectory in ``logs_dir`` to ATIF and write
    ``logs_dir/trajectory.json``. Returns the path, or ``None`` if no converter is
    registered for ``trajectory_format`` (or the native trajectory is absent)."""
    conv = _CONVERTERS.get(trajectory_format)
    if conv is None:
        return None
    traj = conv(Path(logs_dir), instruction=instruction, agent_name=agent_name,
                agent_version=agent_version, model_name=model_name)
    if traj is None:
        return None
    from harbor.utils.trajectory_utils import format_trajectory_json

    path = Path(logs_dir) / "trajectory.json"
    path.write_text(format_trajectory_json(traj.to_json_dict()))
    return path


def write_trajectory_json_auto(
    logs_dir: str | Path,
    *,
    instruction: str,
    agent_name: str,
    agent_version: str = "",
    model_name: str | None = None,
) -> Path | None:
    """Try every registered converter; write ATIF from whichever finds its native
    trajectory in ``logs_dir`` (each converter returns ``None`` when its stream file is
    absent). This is the M+N seam for post-hoc emission: the caller need not know which
    agent produced the trial — the converter that recognizes the on-disk stream wins."""
    for fmt in list(_CONVERTERS):
        p = write_trajectory_json(
            logs_dir, trajectory_format=fmt, instruction=instruction,
            agent_name=agent_name, agent_version=agent_version, model_name=model_name)
        if p is not None:
            return p
    return None


# --- monet stream-json → ATIF -------------------------------------------------

_MONET_STREAM_FILENAME = "monet.stream.jsonl"
_TERMINAL_EVENTS = ("max_turns_reached", "max_tool_errors", "tool_denied")


@dataclass
class _Call:
    id: str
    name: str
    arguments: dict[str, Any] | None = None
    result: str | None = None
    is_error: bool = False


@dataclass
class _Turn:
    text: str = ""
    calls: list[_Call] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


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


def _reduce_monet_stream(path: Path) -> tuple[list[_Turn], "Usage", str | None, str | None]:
    """monet stream-json events → turns (text + tool calls + usage). Robust to missing
    ``index`` on text deltas and to truncated files.

    The second return value is the **session token total** — accumulated by mirroring
    :func:`~beagle.agents.monet._helpers.parse_monet_usage` *exactly* (sum
    ``_monet_event_usage`` over every event carrying a ``usage`` dict, regardless of its
    ``type`` or where it sits relative to ``turn_complete``). This makes the ATIF
    ``final_metrics`` equal to the canonical usage / ``run.json`` *by construction*, rather
    than depending on which turn a trailing standalone ``usage`` event flushes into.
    Returns (turns, total_usage, terminal, session_id)."""
    from beagle.agents.core.usage import Usage
    from beagle.agents.core.usage import add as usage_add
    from beagle.agents.monet._helpers import _monet_event_usage

    session_id: str | None = None
    terminal: str | None = None
    total_usage = Usage()
    turns: list[_Turn] = []
    cur = _Turn()
    text_parts: list[str] = []
    block_to_call: dict[int, _Call] = {}   # tool_use block index -> call
    arg_chunks: dict[str, list[str]] = {}  # tool id -> partial-json fragments
    call_by_id: dict[str, _Call] = {}

    def flush() -> None:
        nonlocal cur, text_parts, block_to_call
        cur.text = "".join(text_parts)
        for c in cur.calls:
            if c.arguments is None and c.id in arg_chunks:
                raw = "".join(arg_chunks[c.id])
                try:
                    c.arguments = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    c.arguments = {}
        turns.append(cur)
        cur = _Turn()
        text_parts = []
        block_to_call = {}

    for ev in _iter_events(path):
        t = ev.get("type")
        # Session total: mirror parse_monet_usage — any event with a `usage` dict contributes,
        # once, regardless of `type`. (Also sets per-turn `cur.usage` in the branches below for
        # display metrics; the total here is authoritative and placement-independent.)
        if isinstance(ev.get("usage"), dict):
            total_usage = usage_add(total_usage, _monet_event_usage(ev["usage"]))
        if t == "session_meta":
            session_id = ev.get("session_id") or session_id
        elif t == "text_delta":
            txt = ev.get("text", "")
            if isinstance(txt, str):
                text_parts.append(txt)
        elif t == "tool_use_start":
            idx, tid = ev.get("index"), ev.get("toolId") or ""
            if isinstance(idx, int) and tid:
                c = _Call(id=tid, name=ev.get("toolName") or "")
                block_to_call[idx] = c
                call_by_id[tid] = c
                cur.calls.append(c)
                arg_chunks[tid] = []
        elif t == "tool_use_delta":
            idx, chunk = ev.get("index"), ev.get("partialJson", "")
            if isinstance(idx, int) and isinstance(chunk, str) and idx in block_to_call:
                arg_chunks.setdefault(block_to_call[idx].id, []).append(chunk)
        elif t == "tool_start":
            c = call_by_id.get(ev.get("id") or "")
            if c is not None and isinstance(ev.get("input"), dict):
                c.arguments = ev["input"]
        elif t == "tool_output":
            c = call_by_id.get(ev.get("id") or "")
            if c is not None:
                out = ev.get("output")
                c.result = out if isinstance(out, str) else json.dumps(out)
                c.is_error = bool(ev.get("isError"))
        elif t == "message_delta":
            u = ev.get("usage")
            if isinstance(u, dict):
                cur.usage = {k: int(v) for k, v in u.items() if isinstance(v, int)}
        elif t == "turn_complete":
            flush()
        elif t == "usage":
            u = ev.get("usage")
            if isinstance(u, dict):
                # Attach to the CURRENT turn for its per-step display metrics (monet emits the
                # turn's usage as a standalone `usage` event just before `turn_complete`). The
                # authoritative session total is accumulated above, independent of this.
                cur.usage = {k: int(v) for k, v in u.items() if isinstance(v, int)}
        elif t in _TERMINAL_EVENTS:
            terminal = t
        elif t == "turn_done":
            terminal = terminal or "turn_done"

    if text_parts or cur.calls:  # truncated mid-turn
        flush()
    return turns, total_usage, terminal, session_id


def _metrics(usage: dict[str, int]):
    """monet usage keys → ATIF Metrics, via the canonical provider-robust split
    (:func:`~beagle.agents.monet._helpers._monet_event_usage`) — so an OpenAI-style ``cacheTokens``
    (a SUBSET of ``inputTokens``) isn't double-counted into prompt."""
    from harbor.models.trajectories import Metrics

    from beagle.agents.monet._helpers import _monet_event_usage

    if not usage:
        return None
    tc = _monet_event_usage(usage).to_token_counts()
    return Metrics(prompt_tokens=tc["prompt"] or None,
                   completion_tokens=tc["completion"] or None,
                   cached_tokens=tc["cache_read"] or None)


@register_converter("monet-stream-json")
def _monet_to_atif(logs_dir: Path, *, instruction: str, agent_name: str,
                   agent_version: str, model_name: str | None):
    stream = logs_dir / _MONET_STREAM_FILENAME
    if not stream.exists():
        return None
    from harbor.models.trajectories import (
        Agent, FinalMetrics, Observation, ObservationResult, Step, ToolCall, Trajectory,
    )

    turns, total_usage, terminal, session_id = _reduce_monet_stream(stream)
    steps: list[Any] = []
    sid = 1
    if instruction:  # step 1: the task prompt (a user step carries only its message)
        steps.append(Step(step_id=sid, source="user", message=instruction))
        sid += 1
    for turn in turns:
        tool_calls = [ToolCall(tool_call_id=c.id, function_name=c.name,
                               arguments=c.arguments or {}) for c in turn.calls]
        obs = [ObservationResult(source_call_id=c.id, content=c.result)
               for c in turn.calls if c.result is not None]
        steps.append(Step(
            step_id=sid, source="agent", model_name=model_name, llm_call_count=1,
            message=turn.text or "",
            tool_calls=tool_calls or None,
            observation=Observation(results=obs) if obs else None,
            metrics=_metrics(turn.usage),
        ))
        sid += 1

    # Total is the canonical provider-robust sum accumulated by the reducer (mirrors
    # parse_monet_usage exactly) — the same numbers run.json / the dashboard report.
    tc = total_usage.to_token_counts()
    fm = (FinalMetrics(
        total_prompt_tokens=tc["prompt"] or None,
        total_completion_tokens=tc["completion"] or None,
        total_cached_tokens=tc["cache_read"] or None,
        total_steps=len(steps),
    ) if (tc["prompt"] or tc["completion"]) else None)
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(name=agent_name or "monet", version=agent_version or "unknown",
                    model_name=model_name),
        steps=steps,
        final_metrics=fm,
        notes=f"terminal: {terminal}" if terminal else None,
    )


# --- mini-swe trajectory → ATIF ----------------------------------------------

_MINI_TRAJ_FILENAME = "mini.traj.json"


def _mini_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _mini_is_agent_turn(m: Any) -> bool:
    """A model turn in either mini format: Chat Completions (``role == "assistant"``) or the
    Responses API (a stored ``object == "response"`` item carrying ``output`` + ``extra.actions``)."""
    return isinstance(m, dict) and (m.get("role") == "assistant" or m.get("object") == "response")


def _mini_is_observation(m: Any) -> bool:
    """A command-output item in either format: chat ``role`` user/tool, or a Responses API
    ``type == "function_call_output"`` item."""
    if not isinstance(m, dict):
        return False
    return m.get("role") in ("user", "tool") or m.get("type") == "function_call_output"


def _mini_obs_content(m: dict) -> str:
    """The observation text — the Responses item keys it under ``output``, chat under ``content``."""
    return _mini_text(m.get("output") if m.get("type") == "function_call_output" else m.get("content"))


def _mini_obs_after(messages: list, idx: int) -> str | None:
    """The observation (command output) that follows a model turn: the next observation item before
    the next model turn. Format-agnostic. ``None`` if the turn produced no observation."""
    for j in range(idx + 1, len(messages)):
        m = messages[j]
        if _mini_is_agent_turn(m):
            return None
        if _mini_is_observation(m):
            return _mini_obs_content(m)
    return None


def _mini_turn_usage(extra: dict, m: dict) -> tuple[int, int]:
    """(prompt, completion) tokens for a turn — chat stashes usage under ``extra.response.usage``
    (``prompt_tokens``/``completion_tokens``); the Responses API puts it at the item's top-level
    ``usage`` (``input_tokens``/``output_tokens``). Try both."""
    u = (extra.get("response") or {}).get("usage") or {}
    p, c = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    if not (p or c):
        u = m.get("usage") or {}
        p = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
        c = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
    return p, c


def _mini_metrics(prompt: int, completion: int):
    from harbor.models.trajectories import Metrics

    if not (prompt or completion):
        return None
    return Metrics(prompt_tokens=prompt or None, completion_tokens=completion or None)


@register_converter("mini-swe")
def _mini_swe_to_atif(logs_dir: Path, *, instruction: str, agent_name: str,
                      agent_version: str, model_name: str | None):
    """mini-swe's ``mini.traj.json`` (``{"messages": [{role, content, extra}, …]}``) → ATIF. The
    ``system`` message becomes a system step, the first ``user`` message the task step, and each
    model turn an agent step (message + its bash ``actions`` as tool calls + usage), with the
    following command output attached as its observation. Handles BOTH mini model formats: Chat
    Completions (``role=assistant`` turns, ``role=user/tool`` observations) and the Responses API
    (``object=response`` turns, ``type=function_call_output`` observations) — reasoning models use
    the latter (see :func:`beagle.agents.mini_swe._mini_vocab_c_args`).
    Best-effort: returns ``None`` if absent/unreadable (the raw stream is kept as the fallback)."""
    path = Path(logs_dir) / _MINI_TRAJ_FILENAME
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (ValueError, OSError):
        return None
    messages = doc.get("messages") if isinstance(doc, dict) else doc
    if not isinstance(messages, list):
        return None
    from harbor.models.trajectories import (
        Agent, FinalMetrics, Observation, ObservationResult, Step, ToolCall, Trajectory,
    )

    steps: list[Any] = []
    sid = 0
    prompt_tot = comp_tot = 0
    seen_assistant = instruction_done = False
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = _mini_text(m.get("content"))
        extra = m.get("extra") or {}
        if role == "system":
            sid += 1
            steps.append(Step(step_id=sid, source="system", message=content))
            continue
        if _mini_is_agent_turn(m):
            seen_assistant = True
            sid += 1
            calls, call_ids = [], []
            for i, a in enumerate(extra.get("actions") or []):
                cid = f"mini-{sid}-{i}"
                cmd = a.get("command") if isinstance(a, dict) else str(a)
                calls.append(ToolCall(tool_call_id=cid, function_name="bash",
                                      arguments={"command": cmd} if cmd is not None else {}))
                call_ids.append(cid)
            p, c = _mini_turn_usage(extra, m)
            prompt_tot += p
            comp_tot += c
            obs_text = _mini_obs_after(messages, idx)
            obs = None
            if obs_text is not None:
                cid = call_ids[0] if call_ids else f"mini-{sid}-obs"
                obs = Observation(results=[ObservationResult(source_call_id=cid, content=obs_text)])
            steps.append(Step(
                step_id=sid, source="agent", model_name=model_name, llm_call_count=1,
                message=content, tool_calls=calls or None, observation=obs,
                metrics=_mini_metrics(p, c)))
        elif not seen_assistant and not instruction_done and role in ("user", "tool"):
            instruction_done = True
            sid += 1
            steps.append(Step(step_id=sid, source="user", message=content or instruction))
        # observations (chat user/tool after a turn, or Responses function_call_output) are consumed
        # by the look-ahead above

    if not steps:
        return None
    fm = FinalMetrics(total_prompt_tokens=prompt_tot or None,
                      total_completion_tokens=comp_tot or None, total_steps=len(steps))
    return Trajectory(
        schema_version="ATIF-v1.7",
        agent=Agent(name=agent_name or "mini-swe", version=agent_version or "unknown",
                    model_name=model_name),
        steps=steps, final_metrics=fm)


# --- opencode --format json → ATIF -------------------------------------------

_OPENCODE_STREAM_FILENAME = "opencode.stream.jsonl"


def _opencode_part_text(part: Any) -> str:
    """The text of a ``text`` / ``reasoning`` part (opencode streams it, growing per update)."""
    return part.get("text", "") if isinstance(part, dict) else ""


def _reduce_opencode_stream(path: Path) -> tuple[list[_Turn], dict[str, int], str | None, str | None]:
    """opencode ``--format json`` events → turns (text + tool calls + usage).

    Each line is ``{type, sessionID, part?|error?}``. A turn is delimited by ``step_start`` …
    ``step_finish``; ``text``/``reasoning`` parts stream (same ``part.id``, last wins) and tool
    parts (keyed by ``callID``: ``tool`` name, ``state.input`` args, ``state.output`` result,
    ``state.status`` error) resolve as their state settles. Robust to truncation.
    Returns (turns, final_usage, error, session_id)."""
    session_id: str | None = None
    error: str | None = None
    final_usage: dict[str, int] = {}
    turns: list[_Turn] = []
    cur = _Turn()
    text_by_id: dict[str, str] = {}
    text_order: list[str] = []
    call_by_id: dict[str, _Call] = {}
    started = False

    def flush() -> None:
        nonlocal cur, text_by_id, text_order, call_by_id
        cur.text = "\n".join(text_by_id[i] for i in text_order if text_by_id.get(i))
        turns.append(cur)
        cur = _Turn()
        text_by_id, text_order, call_by_id = {}, [], {}

    for ev in _iter_events(path):
        sess = ev.get("sessionID")
        session_id = session_id or (sess if isinstance(sess, str) else None)
        t = ev.get("type")
        raw_part = ev.get("part")
        part = raw_part if isinstance(raw_part, dict) else {}
        if t == "step_start":
            if started and (text_order or cur.calls):
                flush()
            started = True
        elif t in ("text", "reasoning"):
            pid = part.get("id") or f"{t}-{len(text_order)}"
            if pid not in text_by_id:
                text_order.append(pid)
            text_by_id[pid] = _opencode_part_text(part)
        elif t == "tool_use":
            cid = part.get("callID") or part.get("id") or ""
            if cid:
                c = call_by_id.get(cid)
                if c is None:
                    c = _Call(id=cid, name=part.get("tool") or "")
                    call_by_id[cid] = c
                    cur.calls.append(c)
                if part.get("tool"):
                    c.name = part["tool"]
                raw_state = part.get("state")
                state = raw_state if isinstance(raw_state, dict) else {}
                if isinstance(state.get("input"), dict):
                    c.arguments = state["input"]
                out = state.get("output")
                if out is not None:
                    c.result = out if isinstance(out, str) else json.dumps(out)
                if state.get("status") in ("error", "output-error"):
                    c.is_error = True
        elif t == "step_finish":
            raw_tok = part.get("tokens")
            tok = raw_tok if isinstance(raw_tok, dict) else {}
            raw_cache = tok.get("cache")
            cache = raw_cache if isinstance(raw_cache, dict) else {}
            usage = {
                "input": int(tok.get("input") or 0),
                "output": int(tok.get("output") or 0),
                "reasoning": int(tok.get("reasoning") or 0),
                "cache_read": int(cache.get("read") or 0),
                "cache_write": int(cache.get("write") or 0),
            }
            cur.usage = usage
            for k, v in usage.items():
                final_usage[k] = final_usage.get(k, 0) + v
            flush()
        elif t == "error":
            err = ev.get("error")
            if isinstance(err, str):
                error = err
            elif isinstance(err, dict):
                error = err.get("message") or err.get("name") or json.dumps(err)

    if text_order or cur.calls:  # truncated mid-turn / no explicit step_finish
        flush()
    return turns, final_usage, error, session_id


def _opencode_metrics(usage: dict[str, int]):
    """opencode step tokens → ATIF Metrics (cache folds into prompt, reasoning into completion)."""
    from harbor.models.trajectories import Metrics

    if not usage:
        return None
    prompt = usage.get("input", 0) + usage.get("cache_read", 0) + usage.get("cache_write", 0)
    completion = usage.get("output", 0) + usage.get("reasoning", 0)
    if not (prompt or completion):
        return None
    return Metrics(prompt_tokens=prompt or None, completion_tokens=completion or None,
                   cached_tokens=usage.get("cache_read") or None)


@register_converter("opencode-json")
def _opencode_to_atif(logs_dir: Path, *, instruction: str, agent_name: str,
                      agent_version: str, model_name: str | None):
    """opencode's ``opencode.stream.jsonl`` (``--format json`` events) → ATIF. The task prompt
    becomes a user step; each LLM step (``step_start`` … ``step_finish``) an agent step (its text
    message + tool parts as tool calls with their outputs as observations + token usage).
    Best-effort: returns ``None`` if absent/unreadable (the raw stream is kept as the fallback)."""
    stream = logs_dir / _OPENCODE_STREAM_FILENAME
    if not stream.exists():
        return None
    from harbor.models.trajectories import (
        Agent, FinalMetrics, Observation, ObservationResult, Step, ToolCall, Trajectory,
    )

    turns, final_usage, error, session_id = _reduce_opencode_stream(stream)
    steps: list[Any] = []
    sid = 1
    if instruction:  # step 1: the task prompt (a user step carries only its message)
        steps.append(Step(step_id=sid, source="user", message=instruction))
        sid += 1
    for turn in turns:
        tool_calls = [ToolCall(tool_call_id=c.id, function_name=c.name,
                               arguments=c.arguments or {}) for c in turn.calls]
        obs = [ObservationResult(source_call_id=c.id, content=c.result)
               for c in turn.calls if c.result is not None]
        steps.append(Step(
            step_id=sid, source="agent", model_name=model_name, llm_call_count=1,
            message=turn.text or "",
            tool_calls=tool_calls or None,
            observation=Observation(results=obs) if obs else None,
            metrics=_opencode_metrics(turn.usage),
        ))
        sid += 1

    fm = None
    if final_usage:
        fm = FinalMetrics(
            total_prompt_tokens=(final_usage.get("input", 0) + final_usage.get("cache_read", 0)
                                 + final_usage.get("cache_write", 0)) or None,
            total_completion_tokens=(final_usage.get("output", 0)
                                     + final_usage.get("reasoning", 0)) or None,
            total_cached_tokens=final_usage.get("cache_read") or None,
            total_steps=len(steps),
        )
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(name=agent_name or "opencode", version=agent_version or "unknown",
                    model_name=model_name),
        steps=steps,
        final_metrics=fm,
        notes=f"error: {error}" if error else None,
    )


__all__ = ["register_converter", "write_trajectory_json", "write_trajectory_json_auto"]
