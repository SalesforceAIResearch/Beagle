"""Per-run statistics — orthogonal to QC, useful context for triage.

These are stats, not issues: turn/tool counts, token usage, sign-off language.
Long runs and large context correlate with failure (the failure analysis), but
that's a *prior over the population*, not a per-trace pathology — so it lives
here as a number to bucket on, not as a hardcoded "thrash" issue.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from . import signals as S
from .model import CanonicalTrajectory


@dataclass
class Summary:
    trace_id: str
    source: str
    num_turns: int
    num_tool_calls: int
    tool_histogram: dict[str, int]
    num_tool_errors: int
    num_edits: int
    num_test_runs: int
    num_plan_calls: int
    terminal: str | None
    final_usage: dict[str, int] | None
    peak_prompt_tokens: int | None
    ends_with_success_language: bool
    final_text_excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def peak_prompt_tokens(traj: CanonicalTrajectory) -> int | None:
    peak: int | None = None
    sources = [t.usage for t in traj.turns if t.usage]
    if traj.final_usage:
        sources.append(traj.final_usage)
    for usage in sources:
        for key in ("inputTokens", "input_tokens", "promptTokens", "prompt_tokens"):
            val = usage.get(key)
            if isinstance(val, int):
                peak = val if peak is None else max(peak, val)
    return peak


def summarize(traj: CanonicalTrajectory) -> Summary:
    calls = [c for _, c in traj.tool_calls()]
    final = traj.final_text()
    return Summary(
        trace_id=traj.trace_id,
        source=traj.source,
        num_turns=len(traj.turns),
        num_tool_calls=len(calls),
        tool_histogram=dict(Counter(c.name for c in calls).most_common()),
        num_tool_errors=sum(1 for c in calls if c.is_error),
        num_edits=sum(1 for c in calls if S.is_edit(c)),
        num_test_runs=sum(1 for c in calls if S.is_test_run(c)),
        num_plan_calls=sum(1 for c in calls if S.is_plan(c)),
        terminal=traj.terminal,
        final_usage=traj.final_usage,
        peak_prompt_tokens=peak_prompt_tokens(traj),
        ends_with_success_language=bool(S.SUCCESS_LANG.search(final))
        and not S.GIVEUP_LANG.search(final),
        final_text_excerpt=S.excerpt(final, 240),
    )
