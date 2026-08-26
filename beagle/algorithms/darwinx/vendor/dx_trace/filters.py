"""Filters + mergers — the QC post-processing stage.

Proposers favor recall; this stage removes the recurring false positives the
blog calls out (they couldn't be fixed by prompt-tuning the proposers) and
collapses duplicates:

* **Filters** drop issues that don't meet a criterion (the blog's ``keep`` idea).
  The three rule filters encode the blog's named false positives: failing unit
  tests are *normal* (not tool errors), models stuck in 2024 wrongly call
  2025/2026 dates "fake", and search engines returning irrelevant results is
  expected.
* **Mergers** collapse duplicate issues (the blog's ``issue_mergers``).
"""

from __future__ import annotations

import abc
import re

from . import signals as S
from .chunks import TraceView
from .model import Issue, IssueCategory
from .proposers import LLMClient, _parse_issues


class Filter(abc.ABC):
    name: str = ""
    requires_llm: bool = False

    @abc.abstractmethod
    def apply(self, issues: list[Issue], view: TraceView, llm: LLMClient | None = None) -> list[Issue]:
        raise NotImplementedError


class Merger(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def apply(self, issues: list[Issue]) -> list[Issue]:
        raise NotImplementedError


# ── rule filters (the blog's named false positives) ─────────────────────────
class FailingTestsAreNormal(Filter):
    """Drop tool-error issues that are really just a failing unit test.

    A red test is a normal part of debugging, not a tool malfunction. Scoped to
    TOOL_ERROR so it never touches premature-termination (which *intends* to use
    a red test as contradiction evidence)."""

    name = "failing_tests_are_normal"

    def apply(self, issues, view, llm=None):
        return [
            i
            for i in issues
            if not (i.category is IssueCategory.TOOL_ERROR and S.TEST_FAILURE.search(i.evidence))
        ]


class FakeDates(Filter):
    """Drop fabricated-fact issues that call a real 2025–2027 date "fake".

    Many models are "stuck in 2024" and mislabel current dates as future/invalid."""

    name = "fake_dates"
    _pat = re.compile(
        r"(202[4-9]).{0,40}(fake|future|invalid|wrong|hallucinat|isn't real|not real)"
        r"|(fake|future|invalid|wrong|hallucinat).{0,40}(202[4-9])",
        re.IGNORECASE | re.DOTALL,
    )

    def apply(self, issues, view, llm=None):
        return [
            i
            for i in issues
            if not (
                i.category is IssueCategory.FABRICATED_FACTS
                and self._pat.search(i.summary + " " + i.evidence)
            )
        ]


class IrrelevantSearchResults(Filter):
    """Drop tool-error issues that just complain a search returned irrelevant hits.

    Search/grep returning irrelevant or no results is expected, not a tool error."""

    name = "irrelevant_search_results"
    _pat = re.compile(
        r"(search|grep|query|result).{0,40}(irrelevant|not relevant|no relevant|unrelated|"
        r"no results|nothing relevant)",
        re.IGNORECASE | re.DOTALL,
    )

    def apply(self, issues, view, llm=None):
        return [
            i
            for i in issues
            if not (
                i.category is IssueCategory.TOOL_ERROR
                and self._pat.search(i.summary + " " + i.evidence)
            )
        ]


class LLMFilter(Filter):
    """Optional: ask an LLM whether to keep each issue (the blog's filter prompt)."""

    requires_llm = True

    def __init__(self, name: str, prompt: str) -> None:
        self.name = name
        self.prompt = prompt

    def apply(self, issues, view, llm=None):
        if llm is None:
            raise RuntimeError(f"filter {self.name!r} requires an LLM client")
        kept = []
        for issue in issues:
            user = (
                f"Issue: {issue.category.value} — {issue.summary}\n"
                f"Evidence: {issue.evidence}\n\n"
                'Respond ONLY JSON: {"keep": true|false}.'
            )
            verdict = _parse_keep(llm.complete(self.prompt, user))
            if verdict:
                kept.append(issue)
        return kept


# ── mergers ─────────────────────────────────────────────────────────────────
class DedupMerger(Merger):
    """Collapse issues with the same (category, message_index), keep highest severity."""

    name = "dedup"

    def apply(self, issues: list[Issue]) -> list[Issue]:
        best: dict[tuple, Issue] = {}
        order: list[tuple] = []
        for issue in issues:
            key = (issue.category, issue.message_index, _norm(issue.evidence))
            if key not in best:
                best[key] = issue
                order.append(key)
            elif issue.severity > best[key].severity:
                best[key] = issue
        return [best[k] for k in order]


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())[:120]


def _parse_keep(raw: str) -> bool:
    issues = _parse_issues(raw)  # tolerant JSON extraction; reuse object parsing
    if issues:  # unexpected shape; default to keep
        return True
    import json

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return True
    try:
        return bool(json.loads(raw[start : end + 1]).get("keep", True))
    except (json.JSONDecodeError, AttributeError):
        return True


RULE_FILTERS: dict[str, type[Filter]] = {
    "failing_tests_are_normal": FailingTestsAreNormal,
    "fake_dates": FakeDates,
    "irrelevant_search_results": IrrelevantSearchResults,
}

MERGERS: dict[str, type[Merger]] = {
    "dedup": DedupMerger,
}
