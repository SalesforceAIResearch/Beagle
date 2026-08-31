"""Proposers — each inspects (part of) a trace and proposes issues.

This is the blog's QC core: proposers run independently (rule proposers are
deterministic and local; LLM proposers chunk the trace and apply a rubric per
chunk, in parallel), then the pipeline post-processes their output with filters
and dedup. A proposer type is just an implementation of :class:`Proposer`.

* **Rule proposers** — deterministic, offline, high precision, for *local*
  pathologies that are structurally visible (tool errors, truncation, loops,
  and the claim-vs-evidence slice of premature termination).
* **LLM proposers** — a rubric prompt over chunks, for *semantic* categories
  that can't be computed (instruction-not-followed, fabricated facts, no-progress
  /should-replan). They run only when an LLM client is supplied.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from . import signals as S
from .chunks import ContextSpec, TraceView
from .model import Issue, IssueCategory, Severity

log = logging.getLogger("trace_analyzer.proposers")


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class Proposer(abc.ABC):
    name: str = ""
    requires_llm: bool = False

    @abc.abstractmethod
    def propose(self, view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        raise NotImplementedError


# ── rule proposers ──────────────────────────────────────────────────────────
class ToolErrorProposer(Proposer):
    """Tool calls that returned an error. (Failing tests are filtered later.)"""

    name = "tool_error"

    def propose(self, view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        issues = []
        for turn_idx, call in view.traj.tool_calls():
            if not call.is_error:
                continue
            issues.append(
                Issue(
                    category=IssueCategory.TOOL_ERROR,
                    summary=f"{call.name} call returned an error",
                    evidence=S.excerpt(call.result or ""),
                    message_index=view.msg_index_for_tool(call.id),
                    turn_index=turn_idx,
                    proposer=self.name,
                    severity=Severity.MEDIUM,
                    data={"tool": call.name},
                )
            )
        return issues


class IncompleteRunProposer(Proposer):
    """The whole run never finished: no terminal marker (crash / timeout / the
    log was cut off). Distinct from a mid-turn truncated reply."""

    name = "incomplete_run"

    def propose(self, view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        traj = view.traj
        if traj.terminal is not None:
            return []
        last = view.messages[-1]["index"] if view.messages else None
        return [
            Issue(
                category=IssueCategory.INCOMPLETE_RUN,
                summary="run has no terminal marker — it never finished (crash / timeout / cut-off)",
                evidence=S.excerpt(traj.final_text()) or "(no final assistant text)",
                message_index=last,
                turn_index=traj.turns[-1].index if traj.turns else None,
                proposer=self.name,
                severity=Severity.HIGH,
            )
        ]


class EarlyTruncationProposer(Proposer):
    """The model emitted incomplete output *mid-turn*: a stream error during a
    turn, or a tool call whose arguments never parsed."""

    name = "early_truncation"

    def propose(self, view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        issues = []
        for turn in view.traj.turns:
            if turn.error:
                issues.append(
                    Issue(
                        category=IssueCategory.EARLY_TRUNCATION,
                        summary="stream/transport error during a turn",
                        evidence=S.excerpt(turn.error),
                        message_index=view.msg_index_for_turn(turn.index),
                        turn_index=turn.index,
                        proposer=self.name,
                        severity=Severity.MEDIUM,
                    )
                )
            for call in turn.tool_calls:
                if call.arguments is None and (call.arguments_raw or "").strip():
                    issues.append(
                        Issue(
                            category=IssueCategory.EARLY_TRUNCATION,
                            summary=f"{call.name} tool call emitted incomplete (unparseable args)",
                            evidence=S.excerpt(call.arguments_raw or ""),
                            message_index=view.msg_index_for_turn(turn.index),
                            turn_index=turn.index,
                            proposer=self.name,
                            severity=Severity.MEDIUM,
                        )
                    )
        return issues


class BehavioralLoopProposer(Proposer):
    """The agent repeats the same tool call — the structurally-visible slice of
    "repeats an action without progress"."""

    name = "behavioral_loop"

    def __init__(self, *, run_length: int = 3, recur_threshold: int = 5) -> None:
        self.run_length = run_length
        self.recur_threshold = recur_threshold

    def propose(self, view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        calls = view.traj.tool_calls()
        issues: list[Issue] = []
        prev: str | None = None
        streak = 0
        streak_turn = 0
        for turn_idx, call in calls:
            sig = S.call_signature(call)
            if sig == prev:
                streak += 1
            else:
                if streak >= self.run_length:
                    issues.append(self._issue(view, streak, streak_turn, prev, "consecutive"))
                prev, streak, streak_turn = sig, 1, turn_idx
        if streak >= self.run_length:
            issues.append(self._issue(view, streak, streak_turn, prev, "consecutive"))
        for sig, n in Counter(S.call_signature(c) for _, c in calls).items():
            if n >= self.recur_threshold:
                first_turn = next(t for t, c in calls if S.call_signature(c) == sig)
                issues.append(self._issue(view, n, first_turn, sig, "recurring"))
        return issues

    def _issue(self, view, n, turn_idx, sig, kind) -> Issue:
        return Issue(
            category=IssueCategory.BEHAVIORAL_LOOP,
            summary=f"{kind} repetition of the same tool call ×{n} (no apparent progress)",
            evidence=S.excerpt(sig or "", 220),
            message_index=view.msg_index_for_turn(turn_idx),
            turn_index=turn_idx,
            proposer=self.name,
            severity=Severity.MEDIUM if n >= self.recur_threshold else Severity.LOW,
            data={"count": n, "kind": kind},
        )


class PrematureCompletionProposer(Proposer):
    """Deterministic, high-precision slice of premature completion: the agent
    declares success while its own last test run is red — it thinks it's done,
    yet isn't. The softer "bare Done. with no verification" case is left to the
    LLM proposer, to keep this rule layer high-precision."""

    name = "premature_completion"

    def propose(self, view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        final = view.traj.final_text()
        if not final.strip() or S.GIVEUP_LANG.search(final) or not S.SUCCESS_LANG.search(final):
            return []
        last_test = self._last_test_result(view)
        if last_test is None or not S.TEST_FAILURE.search(last_test):
            return []
        turn_idx = view.traj.turns[-1].index if view.traj.turns else None
        return [
            Issue(
                category=IssueCategory.PREMATURE_COMPLETION,
                summary="declares success but the last test run shows failures",
                evidence=S.excerpt(last_test, 220),
                message_index=view.msg_index_for_turn(turn_idx) if turn_idx is not None else None,
                turn_index=turn_idx,
                proposer=self.name,
                severity=Severity.HIGH,
                data={"signoff": S.excerpt(final, 120)},
            )
        ]

    @staticmethod
    def _last_test_result(view: TraceView) -> str | None:
        for _, call in reversed(view.traj.tool_calls()):
            if S.is_test_run(call) and call.result is not None:
                return call.result
        return None


RULE_PROPOSERS: dict[str, type[Proposer]] = {
    "tool_error": ToolErrorProposer,
    "incomplete_run": IncompleteRunProposer,
    "early_truncation": EarlyTruncationProposer,
    "behavioral_loop": BehavioralLoopProposer,
    "premature_completion": PrematureCompletionProposer,
}


# ── LLM proposer ────────────────────────────────────────────────────────────
_LLM_SYSTEM = """\
You are an agent-trajectory QC analyst. You are given a CHUNK of an agent run as
numbered messages. Find ONLY issues of this category:

{category}: {rubric}

Rules:
- Report an issue only if the chunk shows clear evidence; prefer precision over recall.
- Cite the exact message_index from the chunk for each issue.
- Quote a short (<200 char) piece of evidence verbatim from that message.
- If there are no such issues in this chunk, return an empty list.
Respond with ONLY JSON: {{"issues": [{{"summary": str, "message_index": int, "evidence": str}}]}}"""


class LLMProposer(Proposer):
    """A configurable proposer that applies a rubric prompt to each chunk."""

    requires_llm = True

    def __init__(
        self,
        name: str,
        category: IssueCategory,
        rubric: str,
        *,
        chunk_size: int = 8,
        chunk_format: str = "xml",
        context: ContextSpec | None = None,
        severity: Severity = Severity.MEDIUM,
        max_workers: int = 4,
    ) -> None:
        self.name = name
        self.category = category
        self.rubric = rubric
        self.chunk_size = chunk_size
        self.chunk_format = chunk_format
        self.context = context or ContextSpec()
        self.severity = severity
        self.max_workers = max_workers

    def propose(self, view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        if llm is None:
            raise RuntimeError(f"proposer {self.name!r} requires an LLM client")
        chunks = view.chunks(size=self.chunk_size, fmt=self.chunk_format, context=self.context)
        system = _LLM_SYSTEM.format(category=self.category.value, rubric=self.rubric)

        def run(chunk):
            allowed = set(chunk.indices)
            try:
                raw = llm.complete(system, chunk.text)
            except Exception as exc:  # one chunk's LLM hiccup must not kill the whole digest
                log.warning("LLM proposer %s: chunk failed (%s); skipping it", self.name, exc)
                return []
            return [self._to_issue(d, allowed) for d in _parse_issues(raw)]

        issues: list[Issue] = []
        if not chunks:
            return issues
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as pool:
            for got in pool.map(run, chunks):
                issues.extend(i for i in got if i is not None)
        return issues

    def _to_issue(self, d: dict, allowed: set[int]) -> Issue | None:
        idx = d.get("message_index")
        if not isinstance(idx, int) or idx not in allowed:
            idx = min(allowed) if allowed else None  # clamp hallucinated indices into the chunk
        summary = str(d.get("summary") or "").strip()
        if not summary:
            return None
        return Issue(
            category=self.category,
            summary=summary,
            evidence=S.excerpt(str(d.get("evidence") or "")),
            message_index=idx,
            proposer=self.name,
            severity=self.severity,
        )


def _parse_issues(raw: str) -> list[dict]:
    """Defensively pull an ``{"issues": [...]}`` payload out of an LLM reply."""
    if not raw:
        return []
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    issues = obj.get("issues") if isinstance(obj, dict) else None
    return [d for d in issues if isinstance(d, dict)] if isinstance(issues, list) else []
