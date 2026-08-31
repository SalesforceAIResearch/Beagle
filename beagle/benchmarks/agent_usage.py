"""Agent-robust token usage from a trial's native stream — the CANONICAL token source.

Harbor's ``agent_result`` token counters are second-hand and lossy for some agents (opencode on the
harbor path reports 0 even though the agent ran), so the authoritative usage is the agent's OWN
stream parsed by its canonical parser into :class:`~beagle.agents.core.usage.Usage`. This mirrors the
trajectory M+N seam (:mod:`beagle.benchmarks.trajectory`): a per-format ``(stream filename, parser)``
registry, auto-detected from what's on disk. An agent with no registered parser — or a trial with no
native stream (the agent never ran) — yields ``None``, and callers fall back to harbor's counters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from beagle.agents.core.usage import Usage


def _registry() -> tuple[tuple[str, Callable[[str], Usage]], ...]:
    """``(stream filename under <trial>/agent/, canonical parser: stream text -> Usage)``, one per
    agent format. Lazy-imported so the agent helpers aren't pulled in at module-import time. A new
    agent adds one line here (M + N, not M × N)."""
    from beagle.agents.monet._helpers import parse_monet_usage
    from beagle.agents.opencode._helpers import parse_opencode_usage

    return (
        ("opencode.stream.jsonl", parse_opencode_usage),
        ("monet.stream.jsonl", parse_monet_usage),
    )


def usage_from_agent_dir(agent_dir: Path) -> Usage | None:
    """Canonical :class:`Usage` parsed from the native stream in ``agent_dir`` (a trial's ``agent/``
    dir), or ``None`` when no recognized **non-empty** stream is present — which is itself the robust
    "the agent never ran" signal (an empty/absent stream), independent of harbor's token accounting.
    A malformed stream falls through to the next parser, then ``None`` (never raises)."""
    for fname, parser in _registry():
        f = agent_dir / fname
        try:
            if f.is_file() and f.stat().st_size > 0:
                return parser(f.read_text())
        except Exception:  # noqa: BLE001 — malformed stream: try the next parser, else None
            continue
    return None


__all__ = ["usage_from_agent_dir"]
