"""LongTermMemory persistence for the Atelier-X proposer brain.

Adopts NexAU-AHE's LongTermMemory.md mechanism: persistent
engineering wisdom that accumulates across iterations. Where
``predicted_impact`` is one-shot per candidate, LongTermMemory carries
*general lessons* (e.g. "agents fail when /app isn't pre-chmod'd",
"don't trust the proposer's own self-checks") across the whole
campaign so each iteration's proposer sees the cumulative learnings.

This module owns:

1. **Storage**: where the file lives (per-campaign under
   ``reports/<campaign>/atelier/LongTermMemory.md``).
2. **Read**: load the current contents, optionally trimmed to the
   most recent N entries (to avoid prompt bloat).
3. **Append**: parse ``learnings_to_persist:`` YAML block from a
   proposer's review text and append new entries.
4. **Schema**: one entry per lesson with timestamp + source node +
   one-paragraph lesson body.

Why per-campaign rather than cross-campaign? Two reasons:
- Each campaign uses one search-set subset; lessons from a different
  subset may not transfer. We don't want subset-A lessons polluting
  the subset-B proposer.
- Cross-campaign sharing is a future enhancement. For E2 we want a
  simple, observable mechanism we can ablation-test against.

How the proposer sees this: the analyze.md prompt template inlines
the LTM contents under a "Persistent engineering wisdom" section. If
the file is missing or empty, the section is omitted entirely (no
empty-string artifacts in the prompt).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


__all__ = [
    "LongTermMemoryEntry",
    "long_term_memory_path",
    "rejected_sources_path",
    "load_memory",
    "load_rejected_sources",
    "mark_source_rejected",
    "render_memory_for_prompt",
    "parse_learnings_to_persist",
    "append_learnings",
    "DEFAULT_MAX_ENTRIES",
]


logger = logging.getLogger("atelier.long_term_memory")


DEFAULT_MAX_ENTRIES = 30
"""Default cap on entries kept in the file. Older entries fall off the
top when this is exceeded. 30 is a soft default — enough to carry
~20 iterations of wisdom without blowing up the analyze prompt."""


@dataclass(frozen=True)
class LongTermMemoryEntry:
    """One persistent lesson.

    ``source_node`` is the child_node_id whose review emitted the
    lesson. ``timestamp`` is set when the entry is appended.

    ``body`` is the lesson itself — should be 1-3 sentences, written
    as actionable general wisdom (not task-specific debugging notes).
    """

    timestamp: str
    source_node: str
    body: str

    def to_markdown(self) -> str:
        """Render one entry as markdown for the file (and the prompt)."""
        return (
            f"### {self.timestamp} (from node `{self.source_node}`)\n\n"
            f"{self.body}\n"
        )


# ─── File location ────────────────────────────────────────────────────────


def long_term_memory_path(
    *, reports_root: Path | str, campaign: str
) -> Path:
    """Canonical path for the LongTermMemory file (per-campaign)."""
    return Path(reports_root) / campaign / "atelier" / "LongTermMemory.md"


def rejected_sources_path(
    *, reports_root: Path | str, campaign: str
) -> Path:
    """Path to the per-campaign list of node ids whose LTM
    contributions should be filtered out at render time.

    Populated by ``mark_source_rejected`` when the equivalence gate
    (or any other downstream consumer) rejects a candidate after the
    candidate has already persisted learnings. Filtering at render
    time (rather than at append time) keeps the LTM file itself
    immutable + auditable: a retrospective can still see what every
    proposer wrote, and what subset was actually shown to future
    proposers.
    """
    return Path(reports_root) / campaign / "atelier" / "LongTermMemory.rejected.txt"


def load_rejected_sources(
    *, reports_root: Path | str, campaign: str
) -> set[str]:
    """Return the set of node ids whose LTM contributions are
    filtered out. Empty set when no rejections recorded.
    Never raises (file is human-editable, OK if missing/malformed)."""
    path = rejected_sources_path(reports_root=reports_root, campaign=campaign)
    if not path.is_file():
        return set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    out: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Line format: "<node_id>\t<iso_ts>\t<reason>" (tab-separated)
        # — split on tab so we recover just the node_id.
        node_id = s.split("\t", 1)[0].strip()
        if node_id:
            out.add(node_id)
    return out


def mark_source_rejected(
    *,
    reports_root: Path | str,
    campaign: str,
    node_id: str,
    reason: str = "",
) -> bool:
    """Append ``node_id`` to the rejected-sources file. Returns True
    iff the node id was newly added (False if already present).

    Called by the equivalence gate's pipeline integration when a
    candidate's verdict comes back MODIFIED/INCONCLUSIVE → the
    learnings that candidate persisted during its iterations should
    not be shown to future proposers (they're likely anti-patterns
    that drove the regression).
    """
    existing = load_rejected_sources(reports_root=reports_root, campaign=campaign)
    if node_id in existing:
        return False

    path = rejected_sources_path(reports_root=reports_root, campaign=campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Line format: "<node_id>\t<iso_timestamp>\t<reason>"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reason_str = (reason or "").replace("\n", " ").replace("\t", " ")[:200]
    line = f"{node_id}\t{ts}\t{reason_str}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    logger.info(
        "atelier-LTM[%s] marked source as rejected (reason=%r) → %s",
        node_id, reason_str[:80], path,
    )
    return True


# ─── Read ─────────────────────────────────────────────────────────────────


# Header that the file always starts with (useful for sanity checks +
# clean re-init).
_FILE_HEADER = (
    "# Atelier LongTermMemory\n"
    "\n"
    "Persistent engineering wisdom accumulated across iterations. Each\n"
    "entry is a general lesson surfaced by a proposer's review step\n"
    "(see `learnings_to_persist:` in review.md). New entries are\n"
    "appended at the bottom; older entries fall off the top when the\n"
    "file exceeds the configured cap.\n"
    "\n"
    "---\n"
    "\n"
)

# Markdown heading that begins one entry: `### <iso8601> (from node \`xxx\`)`
_ENTRY_HEADING_RE = re.compile(
    r"^### (?P<ts>\S+) \(from node `(?P<node>[^`]+)`\)\s*$",
    re.MULTILINE,
)


def load_memory(
    *, reports_root: Path | str, campaign: str
) -> list[LongTermMemoryEntry]:
    """Parse the LongTermMemory file into a list of entries.

    Returns ``[]`` if the file doesn't exist or is empty. Malformed
    entries are silently dropped — the file is human-editable, so we
    expect occasional minor corruption.
    """
    path = long_term_memory_path(reports_root=reports_root, campaign=campaign)
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("LongTermMemory read failed (%s): %s", path, e)
        return []
    if not raw.strip():
        return []

    entries: list[LongTermMemoryEntry] = []
    # Split the file on entry-heading lines. Body is everything between
    # one heading and the next (or end of file).
    matches = list(_ENTRY_HEADING_RE.finditer(raw))
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[body_start:body_end].strip()
        if not body:
            continue
        entries.append(
            LongTermMemoryEntry(
                timestamp=m.group("ts"),
                source_node=m.group("node"),
                body=body,
            )
        )
    return entries


def render_memory_for_prompt(
    *,
    reports_root: Path | str,
    campaign: str,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> str:
    """Return a markdown block suitable for inlining into a proposer
    prompt. Empty string if no memory file exists.

    Entries whose ``source_node`` is in the rejected-sources file are
    filtered out — those lessons came from candidates the equivalence
    gate rejected, so they're likely anti-patterns. The file itself
    is not modified (preserves the audit trail); we just hide
    rejected entries from the proposer's view.

    The trimmed view shows the *most recent* ``max_entries`` after
    filtering; the file may grow longer but the prompt-side view is
    capped to keep analyze-step token spend bounded.
    """
    entries = load_memory(reports_root=reports_root, campaign=campaign)
    if not entries:
        return ""
    rejected = load_rejected_sources(reports_root=reports_root, campaign=campaign)
    if rejected:
        entries = [e for e in entries if e.source_node not in rejected]
        if not entries:
            return ""
    tail = entries[-max_entries:]
    blocks = [e.to_markdown() for e in tail]
    return (
        "## Persistent engineering wisdom (LongTermMemory)\n\n"
        "These are lessons the campaign has accumulated across prior\n"
        "iterations. Use them to inform your analysis; they're general\n"
        "patterns, not task-specific notes.\n\n"
        + "\n".join(blocks)
    )


# ─── Append ───────────────────────────────────────────────────────────────


# Match a `learnings_to_persist:` YAML block in the proposer's review
# text. Same shape as predictions parser — a fenced YAML mapping with
# a list of strings under `learnings`.
_LEARNINGS_BLOCK_RE = re.compile(
    r"(?:```(?:ya?ml)?\s*\n)?"
    r"^learnings_to_persist\s*:\s*\n"
    r"(?P<body>(?:[ \t]+.*\n|\s*\n)+)"
    r"(?:```\s*\n?)?",
    re.MULTILINE,
)


def parse_learnings_to_persist(text: str) -> list[str]:
    """Extract a ``learnings_to_persist:`` YAML block from review text.

    Returns a list of lesson strings (each will become one entry).
    Returns ``[]`` if the block is missing, malformed, or yields no
    strings.
    """
    if not text:
        return []
    match = _LEARNINGS_BLOCK_RE.search(text)
    if not match:
        return []

    body = match.group("body")
    lines = body.splitlines()
    indents = [
        len(line) - len(line.lstrip(" "))
        for line in lines if line.strip()
    ]
    min_indent = min(indents) if indents else 0
    stripped = "\n".join(
        line[min_indent:] if line.strip() else "" for line in lines
    )

    try:
        import yaml
    except ImportError:
        logger.warning("yaml unavailable; cannot parse learnings_to_persist")
        return []

    try:
        parsed = yaml.safe_load(stripped)
    except yaml.YAMLError as e:
        logger.warning("learnings_to_persist YAML parse failed: %s", e)
        return []

    if not isinstance(parsed, dict):
        return []
    learnings = parsed.get("learnings")
    if not isinstance(learnings, list):
        return []
    out: list[str] = []
    for item in learnings:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def append_learnings(
    *,
    reports_root: Path | str,
    campaign: str,
    source_node: str,
    learnings: Iterable[str],
    max_entries: int = DEFAULT_MAX_ENTRIES,
    now: datetime | None = None,
) -> int:
    """Append new learnings to the LongTermMemory file.

    Returns the number of NEW entries actually appended (may be 0
    if all learnings duplicate existing entries — duplicate detection
    uses exact-string match on the body).

    File layout:
        - header (auto-inserted on first write)
        - entries oldest → newest
        - when the total exceeds ``max_entries``, oldest entries are
          dropped from the top during the next write

    The write is atomic: writes to ``.tmp`` then renames.
    """
    learnings_list = [s.strip() for s in learnings if s and s.strip()]
    if not learnings_list:
        return 0

    existing = load_memory(reports_root=reports_root, campaign=campaign)
    existing_bodies = {e.body for e in existing}

    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    new_entries: list[LongTermMemoryEntry] = []
    for lesson in learnings_list:
        if lesson in existing_bodies:
            continue
        new_entries.append(
            LongTermMemoryEntry(
                timestamp=ts, source_node=source_node, body=lesson,
            )
        )
        existing_bodies.add(lesson)

    if not new_entries:
        return 0

    combined = existing + new_entries
    # Trim oldest if over cap.
    if len(combined) > max_entries:
        combined = combined[-max_entries:]

    path = long_term_memory_path(reports_root=reports_root, campaign=campaign)
    path.parent.mkdir(parents=True, exist_ok=True)

    body = _FILE_HEADER + "\n\n".join(e.to_markdown() for e in combined)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)

    logger.info(
        "atelier-LTM[%s] appended %d new lesson(s); total entries=%d → %s",
        source_node, len(new_entries), len(combined), path,
    )
    return len(new_entries)
