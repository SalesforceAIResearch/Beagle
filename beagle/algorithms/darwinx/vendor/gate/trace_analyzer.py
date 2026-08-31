"""Per-task trace digest (Agent-Debugger-lite).

Adopts NexAU-AHE's Agent-Debugger concept: rather than feeding the
proposer raw multi-MB transcripts, distill each trial into a focused
markdown digest that names the failure-mode pattern + key tool
calls + final state. The proposer then spends its analyze-step tokens
on root-cause reasoning instead of transcript parsing.

This module is read-only — it reads existing Harbor trial dirs and
emits markdown digests to ``reports/<campaign>/atelier/digests/``.
It does NOT modify the pipeline. The current ``analyze.md`` prompt
already references raw transcript paths; a later integration step
will swap those references for digest paths once we've validated the
digest content is useful.

Harbor trial dir layout:

    <eval_root>/
        <task>__<hash>/
            agent/transcript.md       ← Monet's full transcript
            verifier/test-stdout.txt  ← grader output (failure cues!)
            verifier/reward.txt       ← 0 / 1 / partial
            result.json               ← structured metadata
            trial.log                 ← Harbor's per-trial events

The digest extracts:
- agent metadata header (events, tokens, tool calls, errors)
- final assistant text (last "answer" the agent gave)
- last 3 turns (to spot stuck-loops or premature termination)
- reward
- verifier stdout excerpt (1st 2 KB — usually contains the failure)
- one-line failure pattern classification

Pattern classification (heuristic, not ML):
- "incomplete deliverables"  — agent declared done, verifier says missing
- "wrong output format"      — agent produced output, verifier says wrong format
- "tool errors / retries"    — high tool_error_count
- "max turns hit"            — max_tool_errors_hit or circuit_breaker_hit
- "premature termination"    — agent said done with very few turns
- "no apparent failure mode" — fallback when nothing above fires

Used by:
- ``atelier trace-analyze`` CLI subcommand (this module's main()),
  for generating digests after a campaign completes.
- Future: pipeline.py's ``_render_analyze_prompt`` to point cursor-
  agent at the digest instead of the raw transcript.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


__all__ = [
    "TrialDigest",
    "FailurePattern",
    "load_trial",
    "render_digest",
    "classify_failure_pattern",
    "scan_eval_dir",
]


logger = logging.getLogger("atelier.trace_analyzer")


# ─── Schema ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FailurePattern:
    """Heuristic classification of a trial's failure mode.

    Multiple patterns can be present; ``primary`` is the most likely
    cause for the proposer's attention.
    """

    primary: str
    """One of: incomplete_deliverables, wrong_output_format,
    tool_errors, max_turns, premature_termination,
    no_apparent_failure, passing."""

    indicators: tuple[str, ...] = ()
    """Concrete evidence strings (e.g., 'tool_error_count=5',
    'final_text mentions \"Done\" but reward=0')."""


@dataclass(frozen=True)
class TrialDigest:
    """One trial's distilled state."""

    task: str
    trial_id: str
    trial_dir: str

    # From result.json:
    reward: float
    n_turns: int
    n_events: int
    tool_call_counts: dict[str, int]
    tool_error_count: int
    max_tool_errors_hit: bool
    circuit_breaker_hit: bool
    final_text: str
    final_text_chars: int

    # From transcript.md:
    transcript_header: str
    """First ~20 lines of transcript.md (the structured agent
    metadata header). Useful when result.json is missing."""

    last_turn_excerpt: str
    """Tail of the transcript — last ~50 lines covering the final
    turn(s). This shows what the agent was DOING at the end."""

    # From verifier/:
    verifier_excerpt: str
    """First 2 KB of test-stdout.txt — usually contains the failure
    message."""

    # Classification:
    pattern: FailurePattern

    @property
    def passed(self) -> bool:
        return self.reward > 0.5

    def to_markdown(self) -> str:
        """Render the digest as markdown for the proposer to read."""
        verdict = "PASSED" if self.passed else "FAILED"
        out = []
        out.append(f"# Trial digest: `{self.task}` — {verdict} (reward={self.reward:.2f})")
        out.append("")
        out.append(f"- trial_id: `{self.trial_id}`")
        out.append(f"- trial_dir: `{self.trial_dir}`")
        out.append("")
        out.append("## Agent stats")
        out.append(f"- events: {self.n_events}, turns: {self.n_turns}")
        out.append(f"- tool calls: `{self.tool_call_counts}`")
        out.append(
            f"- tool errors: {self.tool_error_count} "
            f"(max_tool_errors_hit={self.max_tool_errors_hit}, "
            f"circuit_breaker={self.circuit_breaker_hit})"
        )
        out.append(f"- final_text_chars: {self.final_text_chars}")
        out.append("")
        out.append("## Failure pattern (heuristic)")
        out.append(f"- **primary**: `{self.pattern.primary}`")
        if self.pattern.indicators:
            for ind in self.pattern.indicators:
                out.append(f"- evidence: _{ind}_")
        out.append("")
        out.append("## Final assistant text")
        out.append("```")
        out.append(self.final_text or "(empty)")
        out.append("```")
        out.append("")
        if not self.passed:
            out.append("## Verifier output (first 2 KB)")
            out.append("```")
            out.append(self.verifier_excerpt or "(no verifier output)")
            out.append("```")
            out.append("")
        out.append("## Last turn(s) of transcript")
        out.append("```")
        out.append(self.last_turn_excerpt or "(transcript unavailable)")
        out.append("```")
        qc = _shared_qc_section(Path(self.trial_dir))
        if qc:
            out.append("")
            out.append(qc)
        return "\n".join(out)


# ─── Loaders ──────────────────────────────────────────────────────────────


def _shared_qc_section(trial_dir: Path) -> str:
    """Structured QC for this trial via the shared ``trace_analyzer`` engine.

    Augments the heuristic digest above with the principled, taxonomy-based
    findings (evidence + message_index) when the monet stream-json sidecar is
    retained for the trial. Best-effort: returns ``""`` if the sidecar is
    absent or anything goes wrong, so the digest never depends on it.
    """
    try:
        from dx_trace.digest import digest_paths
    except Exception:
        return ""
    candidates = [trial_dir / "agent" / "trajectory.jsonl", trial_dir / "trajectory.jsonl"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        try:
            hits = list(trial_dir.rglob("*.trajectory.jsonl")) or list(
                trial_dir.rglob("trajectory.jsonl")
            )
        except OSError:
            hits = []
        path = hits[0] if hits else None
    if path is None:
        return ""
    try:
        item = digest_paths([(trial_dir.name, path)]).items[0]
    except Exception:
        return ""
    if not item.readable or not item.issues:
        return ""
    lines = ["## trace_analyzer QC (taxonomy-based)", ""]
    for issue in sorted(item.issues, key=lambda i: -int(i.severity)):
        lines.append(
            f"- **{issue.category.value}** ({issue.severity}, msg {issue.message_index}): "
            f"{issue.summary}"
        )
        if issue.evidence:
            lines.append(f"  - _{issue.evidence[:160]}_")
    return "\n".join(lines)


def _parse_trial_name(trial_dir_name: str) -> tuple[str, str]:
    """`pypi-server__DtTpzGK` → ('pypi-server', 'DtTpzGK')."""
    if "__" in trial_dir_name:
        task, _, trial_id = trial_dir_name.rpartition("__")
        return task, trial_id
    return trial_dir_name, ""


def _safe_read(path: Path, max_bytes: int = 4096) -> str:
    """Read at most ``max_bytes`` from a file. Returns '' on error."""
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tail(path: Path, n_bytes: int = 4096) -> str:
    """Read the last ``n_bytes`` of a file."""
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def classify_failure_pattern(
    *,
    reward: float,
    n_turns: int,
    tool_error_count: int,
    max_tool_errors_hit: bool,
    circuit_breaker_hit: bool,
    final_text: str,
    final_text_chars: int,
    verifier_excerpt: str,
) -> FailurePattern:
    """Heuristic failure-mode classifier.

    Returns the most likely root cause + concrete evidence strings.
    Heuristics tuned for TB-2 failure modes but generally applicable.
    """
    indicators: list[str] = []

    if reward > 0.5:
        return FailurePattern(primary="passing", indicators=("reward > 0.5",))

    if max_tool_errors_hit or circuit_breaker_hit:
        indicators.append(
            f"max_tool_errors_hit={max_tool_errors_hit} circuit_breaker_hit={circuit_breaker_hit}"
        )
        return FailurePattern(primary="max_turns", indicators=tuple(indicators))

    if tool_error_count >= 5:
        indicators.append(f"tool_error_count={tool_error_count}")
        return FailurePattern(primary="tool_errors", indicators=tuple(indicators))

    # Final-text heuristics: did the agent claim done?
    ft_lower = (final_text or "").lower()
    claims_done = any(
        s in ft_lower
        for s in ("done.", "completed", "successfully", "final", "finished")
    )

    # Verifier message heuristics.
    ve = (verifier_excerpt or "").lower()
    mentions_missing = any(
        s in ve for s in ("not found", "missing", "does not exist", "no such")
    )
    mentions_format = any(
        s in ve for s in ("format", "expected", "got:", "doesn't match", "incorrect")
    )

    if claims_done and mentions_missing:
        indicators.append("agent claimed done but verifier reports missing artifacts")
        return FailurePattern(
            primary="incomplete_deliverables", indicators=tuple(indicators)
        )

    if claims_done and mentions_format:
        indicators.append("agent claimed done but verifier reports format mismatch")
        return FailurePattern(
            primary="wrong_output_format", indicators=tuple(indicators)
        )

    if final_text_chars < 50 and n_turns < 5:
        indicators.append(
            f"final_text_chars={final_text_chars} n_turns={n_turns} — agent gave up early"
        )
        return FailurePattern(
            primary="premature_termination", indicators=tuple(indicators)
        )

    return FailurePattern(primary="no_apparent_failure", indicators=tuple(indicators))


def load_trial(trial_dir: Path) -> TrialDigest | None:
    """Build a digest from one trial directory.

    Returns ``None`` if essential files (result.json) are missing.
    """
    if not trial_dir.is_dir():
        return None

    task, trial_id = _parse_trial_name(trial_dir.name)

    # result.json — the structured source of truth.
    result_path = trial_dir / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Extract from result.json (defensive — schemas vary).
    reward = 0.0
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if "reward" in rewards:
        try:
            reward = float(rewards["reward"])
        except (TypeError, ValueError):
            pass

    agent_stats = (result.get("agent_results") or {})
    # Some result.json files put stats under a different shape:
    if not agent_stats:
        agent_stats = result.get("execution_results") or {}
    n_events = int(agent_stats.get("n_events", 0) or 0)
    tool_call_counts = dict(agent_stats.get("tool_call_counts") or {})
    tool_error_count = int(agent_stats.get("tool_error_count", 0) or 0)
    max_tool_errors_hit = bool(agent_stats.get("max_tool_errors_hit", False))
    circuit_breaker_hit = bool(agent_stats.get("circuit_breaker_hit", False))
    final_text = str(agent_stats.get("final_text") or "")
    final_text_chars = int(
        agent_stats.get("final_text_chars", len(final_text)) or 0
    )

    # Count turns from event_counts if present.
    event_counts = agent_stats.get("event_counts") or {}
    n_turns = int(event_counts.get("TurnDone", 0) or 0)

    # transcript.md sample.
    transcript = trial_dir / "agent" / "transcript.md"
    transcript_header = _safe_read(transcript, max_bytes=2048)
    last_turn = _tail(transcript, n_bytes=4096)

    # verifier excerpt.
    verifier_stdout = trial_dir / "verifier" / "test-stdout.txt"
    verifier_excerpt = _safe_read(verifier_stdout, max_bytes=2048)

    pattern = classify_failure_pattern(
        reward=reward,
        n_turns=n_turns,
        tool_error_count=tool_error_count,
        max_tool_errors_hit=max_tool_errors_hit,
        circuit_breaker_hit=circuit_breaker_hit,
        final_text=final_text,
        final_text_chars=final_text_chars,
        verifier_excerpt=verifier_excerpt,
    )

    return TrialDigest(
        task=task,
        trial_id=trial_id,
        trial_dir=str(trial_dir),
        reward=reward,
        n_turns=n_turns,
        n_events=n_events,
        tool_call_counts=tool_call_counts,
        tool_error_count=tool_error_count,
        max_tool_errors_hit=max_tool_errors_hit,
        circuit_breaker_hit=circuit_breaker_hit,
        final_text=final_text,
        final_text_chars=final_text_chars,
        transcript_header=transcript_header,
        last_turn_excerpt=last_turn,
        verifier_excerpt=verifier_excerpt,
        pattern=pattern,
    )


def scan_eval_dir(eval_dir: Path) -> list[TrialDigest]:
    """Scan one eval directory (e.g., ``nodes/<node>/evals/<final_X>/``)
    and return one digest per trial subdir."""
    if not eval_dir.is_dir():
        return []
    digests: list[TrialDigest] = []
    for child in sorted(eval_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "result.json").is_file():
            continue
        d = load_trial(child)
        if d is not None:
            digests.append(d)
    return digests


def render_digest(digest: TrialDigest) -> str:
    """Convenience wrapper around ``TrialDigest.to_markdown``."""
    return digest.to_markdown()


def render_cross_task_overview(digests: list[TrialDigest]) -> str:
    """Render a cross-task overview of all digests.

    Highlights cluster patterns: how many trials failed with each
    pattern, top tool-error tasks, etc. Useful for the proposer to
    spot common failure modes vs idiosyncratic ones.
    """
    if not digests:
        return "_No digests._"

    n_passed = sum(1 for d in digests if d.passed)
    n_failed = len(digests) - n_passed

    out = []
    out.append(f"# Cross-task overview ({len(digests)} trials)")
    out.append("")
    out.append(f"- passed: **{n_passed}** / {len(digests)} ({n_passed / len(digests):.0%})")
    out.append(f"- failed: **{n_failed}**")
    out.append("")

    # Group failures by pattern.
    by_pattern: dict[str, list[TrialDigest]] = {}
    for d in digests:
        if not d.passed:
            by_pattern.setdefault(d.pattern.primary, []).append(d)
    if by_pattern:
        out.append("## Failures grouped by pattern")
        out.append("")
        for pat in sorted(by_pattern, key=lambda k: -len(by_pattern[k])):
            tasks = ", ".join(f"`{d.task}`" for d in by_pattern[pat])
            out.append(f"- **{pat}** ({len(by_pattern[pat])}): {tasks}")
        out.append("")

    # Tool-error leaderboard.
    high_errors = sorted(
        (d for d in digests if d.tool_error_count > 0),
        key=lambda d: -d.tool_error_count,
    )[:5]
    if high_errors:
        out.append("## Top tool-error trials")
        out.append("")
        for d in high_errors:
            out.append(
                f"- `{d.task}`: tool_errors={d.tool_error_count}, "
                f"max_tool_errors_hit={d.max_tool_errors_hit}"
            )
        out.append("")

    return "\n".join(out)


# ─── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="atelier-trace-analyze",
        description=(
            "Generate per-trial markdown digests + cross-task overview "
            "from a Harbor eval directory."
        ),
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help=(
            "Path to the eval dir (e.g. nodes/<node>/evals/<final_X>/) "
            "containing per-trial subdirs."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write digests. Default: <eval_dir>/../digests/. "
            "Overview goes to <output-dir>/overview.md."
        ),
    )
    args = p.parse_args(argv)

    digests = scan_eval_dir(args.eval_dir)
    if not digests:
        print(f"No trials found under {args.eval_dir}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (args.eval_dir.parent / "digests")
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in digests:
        (out_dir / f"{d.task}.md").write_text(d.to_markdown(), encoding="utf-8")
    (out_dir / "overview.md").write_text(
        render_cross_task_overview(digests), encoding="utf-8"
    )

    print(
        f"Wrote {len(digests)} per-trial digests + overview to {out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
