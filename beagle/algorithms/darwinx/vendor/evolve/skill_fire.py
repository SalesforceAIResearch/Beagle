"""Did this extension actually fire on tasks we did not claim?

WHY THIS EXISTS
───────────────
The analyze prompt already tells the proposer that a cue-gated skill "tends to
help ONLY the task it was written for and does NOT generalize". The gate then
says the opposite: an extension edit gets a bias toward PROMOTE because its
downside is bounded. Both statements are true and they are about different
things -- bounded *risk* is not evidence of *value* -- and the result is that a
skill whose trigger matches one benchmark's issue template is cheap to accept
and never removed, because the prune that would remove it needs lineage depth
that a wide tree never reaches.

The static overfitting scan cannot see this class of overfit. It looks for task
ids, `/app/` paths and copied verifier strings. A cue written as English prose
over SWE-bench issue language contains none of those and passes cleanly.

So this module asks a different question, cheaply: take the cue strings the
diff added, and count how many tasks *outside the claim pool* they match in
trajectories that are already on disk. Below a floor, the extension is
unproven and the verdict is downgraded PROMOTE -> ARCHIVE -- kept as evidence,
not promoted to the tip.

DEFAULT OFF
───────────
`DARWINX_GATE_SKILL_FIRE_GATE` is unset by default, so an existing arm is
byte-identical. It is opt-in for two reasons: it makes the gate strictly
stricter, and on this evolvee the binding problem is currently that candidates
do not land at all. Enabling it is an experiment arm, not a default.

WHAT COUNTS AS A CUE, PER EVOLVEE
─────────────────────────────────
monet declares skills in a JS array (`name:`/`description:`/`whenToUse:`),
opencode as filesystem `SKILL.md` files with YAML frontmatter under the
evolve-owned extension root. Both shapes are parsed, because the cue is the
thing being tested and it is written differently in each.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# JS/TS object form: name: 'x', description: "y", whenToUse: `z`
_QUOTED = re.compile(
    r"""(?:name|description|whenToUse|when_to_use|cue)\s*[:=]\s*['"`]([^'"`]{8,})['"`]"""
)
# SKILL.md / YAML frontmatter form: description: y
_FRONTMATTER = re.compile(r"^(name|description|whenToUse|when_to_use|cue)\s*:\s*(.+)$", re.I)

# Cue fragments too generic to be evidence of anything.
_GENERIC = {
    "use this skill", "when you want", "the user", "this skill", "follow the",
    "review the", "fix any", "you should", "make sure", "if needed",
}


def enabled() -> bool:
    return os.environ.get("DARWINX_GATE_SKILL_FIRE_GATE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def min_nonclaimed() -> int:
    """Non-claimed tasks a cue must match to count as proven (default 2).

    One match is an anecdote and could be the claim pool's sibling; two is the
    smallest number that says the cue describes something recurring.
    """
    try:
        return max(1, int(os.environ.get("DARWINX_GATE_SKILL_FIRE_MIN", "2")))
    except ValueError:
        return 2


def cues_from_diff(diff_text: str) -> list[str]:
    """Distinctive cue strings this diff ADDED. Deduped, generics dropped."""
    cues: list[str] = []
    for raw in (diff_text or "").splitlines():
        if not raw.startswith("+") or raw.startswith("+++"):
            continue
        line = raw[1:].strip()
        for m in _QUOTED.finditer(line):
            cues.append(m.group(1))
        fm = _FRONTMATTER.match(line)
        if fm:
            cues.append(fm.group(2).strip().strip("'\""))
    out, seen = [], set()
    for c in cues:
        c = re.sub(r"\s+", " ", c).strip()
        if len(c) < 8:
            continue
        key = c.lower()
        if key in seen or any(g in key for g in _GENERIC):
            continue
        seen.add(key)
        out.append(c)
    return out


def _task_text(path: Path) -> str:
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return ""
    if path.suffix != ".jsonl":
        return raw[:80_000]
    bits: list[str] = []
    for line in raw.splitlines()[:400]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bits.append(line[:2000])
            continue
        if isinstance(obj, dict):
            for k in ("content", "text", "message", "instruction"):
                v = obj.get(k)
                if isinstance(v, str):
                    bits.append(v[:4000])
                elif isinstance(v, dict) and isinstance(v.get("content"), str):
                    bits.append(v["content"][:4000])
        else:
            bits.append(str(obj)[:2000])
    return "\n".join(bits)


def _task_id(name: str) -> str:
    for suffix in (".messages.jsonl", ".trajectory.jsonl", ".jsonl", ".log", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def fire_tasks(
    cues: list[str],
    panel_dir: str | Path,
    *,
    claimed: list[str] | None = None,
    limit: int = 80,
) -> list[str]:
    """Task ids in ``panel_dir`` whose trajectory text matches any cue.

    Claimed tasks are excluded -- matching the task the cue was written for is
    the null hypothesis, not evidence. The file walk is capped because this runs
    inside a verdict on a Lustre filesystem where listing a full 500-task run
    costs real time.
    """
    root = Path(panel_dir)
    claimed_set = set(claimed or [])
    if not cues or not root.is_dir():
        return []
    files: list[Path] = []
    for sub in (root / "raw", root):
        if not sub.is_dir():
            continue
        try:
            for p in sub.iterdir():
                if p.is_file() and p.name.endswith((".messages.jsonl", ".trajectory.jsonl")):
                    files.append(p)
                if len(files) >= limit * 3:
                    break
        except OSError:
            continue
        if files:
            break
    lowered = [c.lower() for c in cues]
    hits, seen = [], set()
    for path in files[: limit * 3]:
        tid = _task_id(path.name)
        if tid in claimed_set or tid in seen:
            continue
        text = _task_text(path).lower()
        if text and any(c in text for c in lowered):
            hits.append(tid)
            seen.add(tid)
            if len(hits) >= limit:
                break
    return hits


def gate(
    diff_text: str,
    *,
    claimed: list[str],
    panel_dir: str | Path | None,
) -> tuple[bool, list[str], list[str]]:
    """(proven, cues, nonclaimed_hits).

    Fail-CLOSED on purpose: no cues found, or no panel to measure against,
    counts as unproven. An extension that cannot be shown to fire anywhere is
    exactly the thing being screened out, and "we could not check" must not be
    the cheapest way to pass a gate.
    """
    cues = cues_from_diff(diff_text)
    if not cues or not panel_dir:
        return False, cues, []
    hits = fire_tasks(cues, panel_dir, claimed=claimed)
    return len(hits) >= min_nonclaimed(), cues, hits


__all__ = ["enabled", "min_nonclaimed", "cues_from_diff", "fire_tasks", "gate"]
