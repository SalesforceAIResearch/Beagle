"""Canonical trajectory model + the QC issue taxonomy.

A :class:`~trace_analyzer.normalizer.TrajectoryNormalizer` converts each agent's
raw run into the agent-agnostic :class:`CanonicalTrajectory` defined here; every
analyzer downstream reads only this model. :meth:`CanonicalTrajectory.messages`
flattens it to the numbered ``role``/``content`` view the blog's design operates
on — each row's list position is its ``message_index`` (the citation handle).

The :class:`IssueCategory` taxonomy is the *general* trace-pathology taxonomy
from the agent-debugger blog (tool error / instruction-not-followed / early
truncation / fabricated facts / behavioral loop), not a benchmark-specific one.
Per the blog the taxonomy is meant to be user-configurable; this is the default.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool invocation extracted from a turn."""

    id: str
    name: str
    arguments: dict[str, Any] | None = None
    arguments_raw: str | None = None
    result: str | None = None
    is_error: bool = False

    def arg(self, key: str, default: Any = None) -> Any:
        return self.arguments.get(key, default) if isinstance(self.arguments, dict) else default


@dataclass
class Turn:
    """One assistant turn plus the tool results that landed in it."""

    index: int
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    error: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class CanonicalTrajectory:
    """A whole agent run, normalized to one agent-agnostic shape."""

    trace_id: str
    source: str
    turns: list[Turn]
    terminal: str | None = None
    final_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def tool_calls(self) -> list[tuple[int, ToolCall]]:
        """All tool calls across the run, paired with their turn index."""
        return [(t.index, c) for t in self.turns for c in t.tool_calls]

    def final_text(self) -> str:
        """Free-text of the last turn that produced any (the agent's sign-off)."""
        for turn in reversed(self.turns):
            if turn.text.strip():
                return turn.text
        return ""

    def messages(self) -> list[dict[str, Any]]:
        """Flatten to numbered ``role``/``content`` rows.

        Each turn → one ``assistant`` row (with ``tool_calls``) followed by one
        ``tool`` row per result. ``index`` is the row's stable citation handle
        (the blog's ``message_index``); ``turn`` back-references the turn.
        """
        out: list[dict[str, Any]] = []

        def push(row: dict[str, Any]) -> None:
            row["index"] = len(out)
            out.append(row)

        for turn in self.turns:
            assistant: dict[str, Any] = {
                "role": "assistant",
                "turn": turn.index,
                "content": turn.text,
            }
            if turn.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": c.id,
                        "name": c.name,
                        "arguments": c.arguments if c.arguments is not None else c.arguments_raw,
                    }
                    for c in turn.tool_calls
                ]
            if turn.stop_reason is not None:
                assistant["stop_reason"] = turn.stop_reason
            if turn.error is not None:
                assistant["error"] = turn.error
            push(assistant)
            for c in turn.tool_calls:
                if c.result is None:
                    continue
                push(
                    {
                        "role": "tool",
                        "turn": turn.index,
                        "name": c.name,
                        "tool_call_id": c.id,
                        "content": c.result,
                        "is_error": c.is_error,
                    }
                )
        return out


class Severity(enum.IntEnum):
    """Ordered so ``--min-severity`` filtering is a simple comparison.

    Not part of the blog's issue shape (which is type/summary/index/evidence);
    an additive convenience for ranking and filtering.
    """

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: str) -> "Severity":
        return cls[value.strip().upper()]


class IssueCategory(str, enum.Enum):
    """Trace-pathology taxonomy: the agent-debugger blog's five, refined.

    Two refinements over the blog's set, because they are different failures
    with different fixes:

    * ``instruction_not_followed`` (blog) means the agent *abandons or ignores*
      the task. Distinct from ``premature_completion`` — doing the task but
      *wrongly believing it is finished* (stopping / declaring success without
      adequate self-verification: "thinks it's done, yet isn't"), the dominant
      observed failure mode.
    * the blog's ``early_truncation`` is split: ``early_truncation`` = the model
      emitted an incomplete reply/tool call *mid-turn*; ``incomplete_run`` = the
      whole run never finished (crashed / timed out / log cut off).
    """

    TOOL_ERROR = "tool_error"  # the tool itself fails or is poorly designed
    INSTRUCTION_NOT_FOLLOWED = "instruction_not_followed"  # drops/ignores the task; does something else
    PREMATURE_COMPLETION = "premature_completion"  # declares done / stops without verifying it's done
    EARLY_TRUNCATION = "early_truncation"  # the model emits an incomplete reply / tool call mid-turn
    INCOMPLETE_RUN = "incomplete_run"  # the whole run never finished (crash / timeout / cut-off log)
    FABRICATED_FACTS = "fabricated_facts"  # ungrounded claims; mock data instead of real
    BEHAVIORAL_LOOP = "behavioral_loop"  # repeats an action without progress

    @classmethod
    def parse(cls, value: str) -> "IssueCategory":
        return cls(value.strip().lower())


@dataclass
class Issue:
    """One proposed issue. Mirrors the blog's four fields, plus provenance.

    The blog's issue is ``{issue_type, summary, message_index, evidence}``; here
    ``category`` is the issue type, and ``proposer``/``severity``/``data`` are
    additive provenance for filtering and dedup.
    """

    category: IssueCategory
    summary: str
    evidence: str
    message_index: int | None = None
    turn_index: int | None = None
    proposer: str = ""
    severity: Severity = Severity.MEDIUM
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.category.value,
            "summary": self.summary,
            "message_index": self.message_index,
            "evidence": self.evidence,
            "turn_index": self.turn_index,
            "proposer": self.proposer,
            "severity": str(self.severity),
            "data": self.data,
        }
