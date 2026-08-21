"""Load trajectories from a Harbor job dir into ``verifier.TrajectoryInput``.

Each Harbor trial under ``<job_dir>/trials/<trial_name>/`` produces:

- ``agent/transcript.md`` — human-readable Markdown timeline rendered by
  ``monet_eval.core.transcript`` after the run.
- ``agent/trajectory.json`` — ATIF-v1.6 JSON (for ``harbor view``).
- ``agent/trajectory.jsonl`` — raw stream-json NDJSON (when retained;
  deleted post-run by ``harbor.agent`` in the canonical pipeline).
- ``task/instruction.md`` (or ``task.json[instruction]``) — the task
  statement Harbor handed to the agent.

The verifier wants ``(transcript, instruction)`` per task to assess the
trajectory. This module exposes:

- ``load_trial(trial_dir)`` — one trial → ``LoadedTrajectory`` (or None
  if the trial is incomplete, e.g. agent crash before transcript wrote).
- ``load_job(job_dir, task_ids=...)`` — walk all trials, optionally
  filtered to a task_id subset (e.g. the search subset).
- ``to_verifier_input(loaded)`` — adapter to
  ``atelier.verifier.TrajectoryInput``.

The loader is defensive: missing files produce a ``LoadedTrajectory``
with empty fields rather than raising. The verifier degrades gracefully
on empty transcripts (it returns a low-confidence assessment), so a
missing transcript shouldn't crash the campaign.

Trial-name convention: Harbor names trials ``<task_id>__<6charhash>``.
Anything that doesn't match returns ``task_id=trial_name`` so the loader
remains useful for non-standard layouts (custom datasets, ad-hoc runs).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .verifier import TrajectoryInput


logger = logging.getLogger("atelier.trajectory_loader")


# Trial names like "1012__abc123" or "qemu-startup__a1b2c3".
_TRIAL_NAME_RE = re.compile(r"^(?P<task_id>.+?)__(?P<hash>[A-Za-z0-9]+)$")

# Workspace summary trimming — keeps verifier prompts within budget while
# preserving signal on whether the agent left useful artifacts.
_TRANSCRIPT_CHAR_LIMIT = 60_000
"""Hard cap on transcript length. Larger than this is almost always tool
output spam (npm install logs, search dumps). The verifier reads heading
+ tail; we don't need the middle."""

_INSTRUCTION_CHAR_LIMIT = 8_000
"""Instructions in TB-2 are typically < 2 KB. The cap protects against
unusual long-instruction tasks consuming the prompt budget."""


# ─── Loaded result dataclass ─────────────────────────────────────────────


@dataclass(frozen=True)
class LoadedTrajectory:
    """One trial's loaded trajectory data.

    All string fields default to ``""`` when the source file is missing;
    callers can detect that via ``has_transcript`` / ``has_instruction``.
    """

    task_id: str
    trial_name: str
    trial_dir: Path
    transcript: str = ""
    instruction: str = ""
    workspace_summary: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript.strip())

    @property
    def has_instruction(self) -> bool:
        return bool(self.instruction.strip())


# ─── Trial-name parsing ──────────────────────────────────────────────────


def parse_trial_name(trial_name: str) -> str:
    """Extract the task_id from a Harbor trial name.

    Harbor's convention is ``<task_id>__<6charhash>`` (the hash protects
    against name collisions in re-runs). If the name doesn't match,
    return it verbatim — the loader still works for ad-hoc datasets that
    don't use the convention.
    """
    m = _TRIAL_NAME_RE.match(trial_name)
    if m:
        return m.group("task_id")
    return trial_name


# ─── Loaders ─────────────────────────────────────────────────────────────


def _read_text_capped(path: Path, *, cap: int) -> str:
    """Read a file's text, truncating at ``cap`` chars with a marker.

    Returns ``""`` if the file is missing or unreadable.
    """
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("failed to read %s: %s", path, e)
        return ""
    if len(text) <= cap:
        return text
    head = text[: cap // 2]
    tail = text[-(cap // 2):]
    omitted = len(text) - len(head) - len(tail)
    return (
        head
        + f"\n\n[... {omitted} chars omitted ...]\n\n"
        + tail
    )


def _load_instruction(trial_dir: Path) -> str:
    """Load the task instruction from a trial dir, trying common layouts."""
    candidates = [
        trial_dir / "task" / "instruction.md",
        trial_dir / "task.md",
    ]
    for c in candidates:
        if c.is_file():
            return _read_text_capped(c, cap=_INSTRUCTION_CHAR_LIMIT)

    task_json = trial_dir / "task.json"
    if task_json.is_file():
        try:
            data = json.loads(task_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("failed to parse %s: %s", task_json, e)
            return ""
        if isinstance(data, dict):
            instr = data.get("instruction") or data.get("task") or ""
            if isinstance(instr, str) and instr:
                return instr[:_INSTRUCTION_CHAR_LIMIT]
    return ""


def _load_transcript(trial_dir: Path) -> str:
    """Load a rendered transcript, preferring ``agent/transcript.md``."""
    candidates = [
        trial_dir / "agent" / "transcript.md",
        trial_dir / "transcript.md",
    ]
    for c in candidates:
        if c.is_file():
            return _read_text_capped(c, cap=_TRANSCRIPT_CHAR_LIMIT)
    return ""


def load_trial(trial_dir: Path) -> LoadedTrajectory | None:
    """Load one Harbor trial's transcript + instruction.

    Returns ``None`` if ``trial_dir`` doesn't look like a trial dir (no
    ``agent/`` subdir AND no top-level ``transcript.md``). Otherwise
    returns a ``LoadedTrajectory`` with whatever was found (empty fields
    for missing pieces).
    """
    trial_dir = Path(trial_dir)
    if not trial_dir.is_dir():
        return None
    has_agent = (trial_dir / "agent").is_dir()
    has_transcript_top = (trial_dir / "transcript.md").is_file()
    if not (has_agent or has_transcript_top):
        return None

    transcript = _load_transcript(trial_dir)
    instruction = _load_instruction(trial_dir)
    return LoadedTrajectory(
        task_id=parse_trial_name(trial_dir.name),
        trial_name=trial_dir.name,
        trial_dir=trial_dir,
        transcript=transcript,
        instruction=instruction,
    )


def load_job(
    job_dir: Path,
    *,
    task_ids: set[str] | None = None,
) -> list[LoadedTrajectory]:
    """Load all trial trajectories from a Harbor job dir.

    Tries two layouts (in order):
    1. ``<job_dir>/trials/<trial>/`` — the layout Harbor writes for
       its top-level ``jobs/<job_name>/`` directory.
    2. ``<job_dir>/<trial>/``       — the layout self_evolve's
       ``_archive_final_eval_job_if_small`` produces when it copies
       trials into ``reports/<campaign>/nodes/<id>/evals/<job>/``.
       This is what ``atelier_hook`` passes in via
       ``HookContext.final_eval_job_dir``.

    Original layout assumption (1) was the bug found in the
    `ab_overnight_2_atelier` campaign: the hook pointed at the
    archived eval dir (layout 2), the loader found no trials/ subdir,
    and verifier-fitness silently no-op'd for every node. Without this
    flexibility the L2 path doesn't pick up real trial transcripts.

    ``task_ids`` (optional) filters to a subset of task IDs (after
    parsing via ``parse_trial_name``). Missing trial subdirs are
    silently skipped — Harbor occasionally elides trials that errored
    before any output, and we don't want a partial job dir to crash
    the loader.

    Returns a list (in deterministic ``sorted(trial_name)`` order).
    """
    job_dir = Path(job_dir)
    # Try layout 1 first (Harbor canonical), then layout 2 (archived).
    trials_root = job_dir / "trials"
    if not trials_root.is_dir():
        trials_root = job_dir  # archived layout — trials live directly here
        if not trials_root.is_dir():
            logger.warning(
                "job dir %s has neither trials/ subdir nor top-level "
                "trial subdirs; returning empty trajectory list",
                job_dir,
            )
            return []

    loaded: list[LoadedTrajectory] = []
    for entry in sorted(trials_root.iterdir()):
        if not entry.is_dir():
            continue
        # Defend against picking up non-trial dirs at the archived
        # layout's root (e.g., job-level config.json's parent).
        if not (entry / "agent").is_dir() and not (entry / "transcript.md").is_file():
            continue
        tr = load_trial(entry)
        if tr is None:
            continue
        if task_ids is not None and tr.task_id not in task_ids:
            continue
        loaded.append(tr)
    return loaded


def by_task_id(
    loaded: list[LoadedTrajectory],
) -> dict[str, LoadedTrajectory]:
    """Index a list by task_id.

    When multiple trials share a task_id (re-runs), the later one wins
    (sorted-order's last entry). Typically Harbor only runs one trial
    per task per job, so this is unambiguous.
    """
    return {tr.task_id: tr for tr in loaded}


# ─── Adapter to verifier.TrajectoryInput ─────────────────────────────────


def to_verifier_input(loaded: LoadedTrajectory) -> TrajectoryInput:
    """Convert a ``LoadedTrajectory`` into a verifier-ready input.

    The ``trajectory_id`` is set to the trial_name (which encodes both
    task_id and run hash), so verifier assessments are addressable per
    trial when the same task gets multiple runs.
    """
    return TrajectoryInput(
        trajectory_id=loaded.trial_name,
        task_id=loaded.task_id,
        task_instruction=loaded.instruction,
        transcript=loaded.transcript,
        workspace_summary=loaded.workspace_summary,
        extra={"trial_dir": str(loaded.trial_dir), **loaded.extra},
    )


def to_verifier_inputs_by_task(
    loaded: list[LoadedTrajectory],
) -> dict[str, TrajectoryInput]:
    """Bulk-convert + index by task_id (the shape ``FitnessRunner`` wants)."""
    return {tr.task_id: to_verifier_input(tr) for tr in loaded}


__all__ = [
    "LoadedTrajectory",
    "parse_trial_name",
    "load_trial",
    "load_job",
    "by_task_id",
    "to_verifier_input",
    "to_verifier_inputs_by_task",
]
