"""Aggregate QC over a *set* of trajectories → an evidence block for analysis.

This is the reusable surface the evolving systems (self_evolve, atelier) call to
turn a batch of run trajectories into a compact, evidence-cited markdown block —
so a meta-agent's analyze step starts from structured symptoms instead of
cold-reading raw transcripts. It is deterministic and offline by default (rule
proposers only); pass an ``llm`` client to add the semantic proposers.

Design contract: this never raises on a bad trajectory — an unreadable file is
recorded as an error and counted, so one corrupt run can't sink a campaign's
analyze step.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .model import Issue, Severity
from .normalizer import NormalizeError, load
from .pipeline import run_qc
from .proposers import LLMClient


@dataclass
class TrajectoryDigest:
    """QC result for one trajectory (or an error if it couldn't be read)."""

    label: str
    trace_id: str | None = None
    issues: list[Issue] = field(default_factory=list)
    error: str | None = None

    @property
    def readable(self) -> bool:
        return self.error is None

    def categories(self) -> list[str]:
        return sorted({i.category.value for i in self.issues})


@dataclass
class Digest:
    """QC over a batch of trajectories, with a renderable evidence block."""

    items: list[TrajectoryDigest]
    config: str = "default"

    @property
    def readable(self) -> list[TrajectoryDigest]:
        return [d for d in self.items if d.readable]

    @property
    def unreadable(self) -> list[TrajectoryDigest]:
        return [d for d in self.items if not d.readable]

    def category_trace_counts(self) -> Counter[str]:
        """How many trajectories exhibit each category (not issue count)."""
        c: Counter[str] = Counter()
        for d in self.readable:
            for cat in d.categories():
                c[cat] += 1
        return c

    def examples_by_category(self, limit: int = 2) -> dict[str, list[tuple[str, Issue]]]:
        out: dict[str, list[tuple[str, Issue]]] = defaultdict(list)
        # HIGH-severity examples first, so the table shows the strongest evidence
        ranked = sorted(
            ((d.label, i) for d in self.readable for i in d.issues),
            key=lambda li: -int(li[1].severity),
        )
        for label, issue in ranked:
            bucket = out[issue.category.value]
            if len(bucket) < limit:
                bucket.append((label, issue))
        return out

    def render_markdown(self, *, max_examples: int = 2) -> str:
        n, m = len(self.items), len(self.readable)
        flagged = [d for d in self.readable if d.issues]
        lines = [
            "## Trajectory QC digest (trace_analyzer)",
            "",
            f"Automated QC over {n} trajectories ({m} readable, {len(flagged)} with "
            f"findings), config `{self.config}`. These are **candidate symptoms with "
            "evidence, not verdicts** — and an empty result is **not** proof of a clean "
            "run (semantic / wrong-but-clean failures need a closer look). Use this to "
            "prioritise; confirm against the original trajectories.",
            "",
        ]
        counts = self.category_trace_counts()
        if counts:
            examples = self.examples_by_category(max_examples)
            lines.append("Dominant issue categories across the set:")
            lines.append("")
            lines.append("| category | # traces | example evidence |")
            lines.append("|---|---|---|")
            for cat, ct in counts.most_common():
                ex = examples.get(cat, [])
                ex_txt = "; ".join(
                    f"`{label}` #{i.message_index}: {i.summary}" for label, i in ex
                ) or "—"
                lines.append(f"| `{cat}` | {ct} | {ex_txt} |")
            lines.append("")
        else:
            lines.append("_No deterministic issues found (rule proposers). This does "
                         "not mean the runs are clean — inspect for semantic failures._")
            lines.append("")

        lines.append("Per-trajectory:")
        for d in self.items:
            if not d.readable:
                lines.append(f"- `{d.label}`: ⚠ unreadable ({d.error})")
            elif not d.issues:
                lines.append(f"- `{d.label}`: — none —")
            else:
                # collapse repeats: "tool_error(medium)×4"
                per_cat: Counter[str] = Counter(i.category.value for i in d.issues)
                worst = {
                    cat: max(i.severity for i in d.issues if i.category.value == cat)
                    for cat in per_cat
                }
                tags = ", ".join(
                    f"{cat}({worst[cat]})" + (f"×{per_cat[cat]}" if per_cat[cat] > 1 else "")
                    for cat in sorted(per_cat, key=lambda c: -int(worst[c]))
                )
                lines.append(f"- `{d.label}`: {tags}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "total": len(self.items),
            "readable": len(self.readable),
            "category_trace_counts": dict(self.category_trace_counts()),
            "trajectories": [
                {
                    "label": d.label,
                    "trace_id": d.trace_id,
                    "error": d.error,
                    "issues": [i.to_dict() for i in d.issues],
                }
                for d in self.items
            ],
        }


def digest_paths(
    items: Iterable[tuple[str, str | Path]],
    *,
    config: str = "default",
    source: str = "auto",
    llm: LLMClient | None = None,
    min_severity: Severity = Severity.INFO,
) -> Digest:
    """QC each ``(label, path)`` and aggregate. Never raises on a bad file."""
    out: list[TrajectoryDigest] = []
    for label, path in items:
        try:
            traj = load(path, source=source)
        except (NormalizeError, OSError) as exc:
            out.append(TrajectoryDigest(label=label, error=str(exc)))
            continue
        try:
            result = run_qc(traj, config, llm=llm, min_severity=min_severity)
        except Exception as exc:  # a detector bug shouldn't sink the batch
            out.append(TrajectoryDigest(label=label, trace_id=traj.trace_id, error=f"qc failed: {exc}"))
            continue
        out.append(TrajectoryDigest(label=label, trace_id=traj.trace_id, issues=result.issues))
    return Digest(items=out, config=config)
