"""QC pipeline: proposers → filters → mergers (the blog's QC flow).

Proposers that need an LLM are skipped (and recorded) when no client is given,
so the deterministic rule proposers always run offline; supply a client to add
the semantic ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunks import TraceView
from .config import QCConfig, load_config
from .model import CanonicalTrajectory, Issue, Severity
from .proposers import LLMClient


@dataclass
class QCResult:
    trace_id: str
    config: str
    issues: list[Issue]
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "config": self.config,
            "issues_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
            "skipped": self.skipped,
        }


def run_qc(
    traj: CanonicalTrajectory,
    config: QCConfig | str = "default",
    *,
    llm: LLMClient | None = None,
    min_severity: Severity = Severity.INFO,
) -> QCResult:
    if isinstance(config, str):
        config = load_config(config)
    view = TraceView.build(traj)
    issues: list[Issue] = []
    skipped: list[str] = []

    for proposer in config.proposers:
        if proposer.requires_llm and llm is None:
            skipped.append(f"{proposer.name} (needs LLM)")
            continue
        try:
            issues.extend(proposer.propose(view, llm))
        except Exception as exc:  # one bad proposer shouldn't sink the run
            skipped.append(f"{proposer.name} (error: {exc})")

    for filt in config.filters:
        if filt.requires_llm and llm is None:
            skipped.append(f"{filt.name} (needs LLM)")
            continue
        issues = filt.apply(issues, view, llm)

    for merger in config.mergers:
        issues = merger.apply(issues)

    issues = [i for i in issues if i.severity >= min_severity]
    issues.sort(
        key=lambda i: (-int(i.severity), i.message_index if i.message_index is not None else 1 << 30)
    )
    return QCResult(traj.trace_id, config.name, issues, skipped)
