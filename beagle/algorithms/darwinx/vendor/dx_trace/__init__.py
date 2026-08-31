"""Generic agent-trajectory analyzer (proposer/filter QC + QA).

Modeled on the agent-debugger design: a source-specific
:class:`~trace_analyzer.normalizer.TrajectoryNormalizer` converts a raw agent
run into one :class:`~trace_analyzer.model.CanonicalTrajectory`; then QC runs as
**proposers → filters → mergers** (:func:`~trace_analyzer.pipeline.run_qc`) over
a general, configurable issue taxonomy, and QA is :func:`~trace_analyzer.llm.ask`.

Proposers are deterministic *rule* proposers (offline) for locally-visible
pathologies plus *LLM* proposers (a rubric per chunk) for semantic ones; filters
remove known false positives. The taxonomy/proposers are config-driven
(``configs/*.yaml``), so bringing your own is the intended extension point.
"""

from __future__ import annotations

from . import normalizers  # noqa: F401  (registers built-in normalizers)
from .config import builtin_configs, load_config
from .model import (
    CanonicalTrajectory,
    Issue,
    IssueCategory,
    Severity,
    ToolCall,
    Turn,
)
from .normalizer import TrajectoryNormalizer, available, load, register
from .pipeline import QCResult, run_qc
from .summarize import Summary, summarize

__all__ = [
    "CanonicalTrajectory",
    "Turn",
    "ToolCall",
    "Issue",
    "IssueCategory",
    "Severity",
    "TrajectoryNormalizer",
    "register",
    "available",
    "load",
    "run_qc",
    "QCResult",
    "summarize",
    "Summary",
    "load_config",
    "builtin_configs",
]
