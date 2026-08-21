"""teacher_supervision.py -- OPTIONAL contrastive teacher-trace supervision (PROTOTYPE).

WHAT
----
The default analyze-step gradient (``trace_qc.analyze_digest_block``) only sees
monet's OWN failed trajectories, so the proposer gets a *failure diagnosis* but
no *demonstration of success*. This module adds a TEACHER signal: run a
reference solver (cursor-agent) on the SAME task, and if it succeeds, distill its
successful trajectory into a contrastive "reference approach" block appended to
the analyze prompt. The proposer then distills the teacher's APPROACH into
monet's harness instead of guessing from failures alone.

STATUS
------
Prototype. DISABLED by default. Enable with ``ATELIER_TEACHER_SUPERVISION=1``.
NOT wired into any live campaign. Adding this file does not change behaviour:
nothing imports it unless the integration hook (below) is added AND the flag set.

WHY GATED / OFF FOR NOW
-----------------------
1. Untested -- a misleading teacher signal could steer the proposer wrong across
   a whole campaign.
2. A teacher rollout per claimed task ~doubles rollout cost and gateway load.
3. The contrastive block changes proposer input; it must be validated against the
   failure-only baseline (does it raise the genuine-new-win rate?) before trust.

TRANSFER CAVEAT
---------------
Helps PROCEDURAL gaps (monet didn't know the approach; the base model CAN execute
once shown). Does NOT help raw-capability gaps (the base model can't execute even
with the recipe). Distill the teacher's APPROACH (tools, commands, checks), not
verbatim actions, so monet adapts it to its own capability and we avoid skill
bloat the equivalence GATE penalizes.

INTEGRATION (apply only when testing; do NOT apply to a live campaign)
---------------------------------------------------------------------
In ``self_evolve/trace_qc.analyze_digest_block``, AFTER the existing digest
string is built and BEFORE returning it::

    from .teacher_supervision import maybe_teacher_block
    block += maybe_teacher_block(task, trials, eval_dir)

``maybe_teacher_block`` is a strict no-op (returns "") unless
``ATELIER_TEACHER_SUPERVISION=1``; it is also wrapped in try/except so any failure
degrades to the failure-only baseline.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("self_evolve.teacher_supervision")

ENV_FLAG = "ATELIER_TEACHER_SUPERVISION"
# runner.run YAML with agent=cursor-agent (teacher SOLVER) + terminal-bench-v2.1.
# Created/validated during prototype testing; absent => module degrades to no-op.
TEACHER_CONFIG = os.environ.get(
    "ATELIER_TEACHER_CONFIG",
    "configs/self_evolve/teacher_cursor_solver_tb21.yaml",
)
TEACHER_RESULTS_ROOT = os.environ.get(
    "ATELIER_TEACHER_RESULTS_ROOT", "/opt/sagemaker/tmp/teacher_rollouts"
)
TEACHER_TIMEOUT_S = int(os.environ.get("ATELIER_TEACHER_TIMEOUT_S", "5400"))


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "") == "1"


# --------------------------------------------------------------------------- #
# teacher rollout                                                             #
# --------------------------------------------------------------------------- #
def teacher_rollout(task: str) -> tuple[float | None, Path | None]:
    """Run the teacher solver on ``task``. Return (reward, trajectory_jsonl) or (None, None).

    Idempotent-ish: reuses an existing successful rollout for the task if present
    under TEACHER_RESULTS_ROOT so we never pay for the same teacher trace twice.
    """
    cached = _find_successful_rollout(task)
    if cached:
        return 1.0, cached
    if not Path(TEACHER_CONFIG).exists():
        log.warning("teacher rollout skipped: config %s not found", TEACHER_CONFIG)
        return None, None
    group = f"teacher_{task}"
    cmd = [
        sys.executable, "-m", "runner.run", TEACHER_CONFIG,
        "--campaign-id", group, "--results-root", TEACHER_RESULTS_ROOT,
        "--task-ids", task,  # prototype assumes runner.run honours --task-ids override
    ]
    env = dict(os.environ, XRLENV_GROUP_ID=group)
    try:
        subprocess.run(cmd, env=env, timeout=TEACHER_TIMEOUT_S, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        log.warning("teacher rollout failed for %s: %s", task, exc)
        return None, None
    traj = _find_successful_rollout(task)
    return (1.0, traj) if traj else (0.0, None)


def _find_successful_rollout(task: str) -> Path | None:
    """Return a trajectory file of a teacher trial that solved ``task`` (reward>=1).
    Handles both cursor-agent naming ``<hash>__<task>`` (num_samples=1, no __s) and
    monet naming ``<hash>__<task>__s<n>``; finds the cursor ``agent/trajectory.json``
    or the monet ``.trajectory.jsonl`` / ``coding_bench_monet.json``.
    """
    pats = [f"*__{task}", f"*__{task}__s*"]
    for pat in pats:
        for rj in glob.glob(os.path.join(TEACHER_RESULTS_ROOT, "runs", "*", "raw", "trials", pat, "result.json")):
            try:
                t = json.load(open(rj))
            except Exception:
                continue
            rew = ((t.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if rew is None or float(rew) < 1.0:
                continue
            trial_dir = Path(rj).parent
            base = trial_dir.parent.parent  # .../raw
            name = trial_dir.name
            for cand in (trial_dir / "agent" / "trajectory.json",
                         base / f"{name}.trajectory.jsonl",
                         trial_dir / "agent" / "coding_bench_monet.json",
                         base / f"{task}.log"):
                if cand.exists():
                    return cand
    return None


# --------------------------------------------------------------------------- #
# contrastive block                                                          #
# --------------------------------------------------------------------------- #
def _summarize_teacher_trajectory(traj_path: Path, max_chars: int = 2500) -> str:
    """Extract the teacher's APPROACH: the sequence of shell commands / tool calls
    and any final summary. We deliberately surface *what it did* (commands, tools,
    checks) rather than full text, so the proposer distills a reusable recipe.
    """
    cmds: list[str] = []
    final = ""
    try:
        raw = open(traj_path, "r", errors="ignore").read()
        for m in re.finditer(r'"(?:command|cmd|input|bash|args|tool_input)"\s*:\s*"([^"]{3,200})"', raw):
            c = m.group(1).strip()
            if c and c not in cmds:
                cmds.append(c)
        texts = re.findall(r'"(?:text|content|message|summary)"\s*:\s*"([^"]{40,})"', raw)
        if texts:
            final = texts[-1]
    except Exception as exc:
        log.debug("teacher trajectory parse failed: %s", exc)
    lines = []
    if cmds:
        lines.append("Key actions the reference solver took (distill the APPROACH, adapt to monet):")
        for c in cmds[:25]:
            lines.append(f"  $ {c}")
    if final:
        lines.append("\nReference solver's closing rationale:")
        lines.append("  " + final[:600].replace("\n", " "))
    out = "\n".join(lines)
    return out[:max_chars]


def contrastive_block(task: str, teacher_traj_path: Path) -> str:
    """Render the markdown block injected after the failure digest."""
    approach = _summarize_teacher_trajectory(teacher_traj_path)
    if not approach:
        return ""
    return (
        f"\n\n### Reference solution signal (teacher: cursor-agent solved `{task}`)\n"
        "A reference agent SUCCEEDED on this exact task. Use it as supervision: "
        "diagnose *why monet's trajectory diverged* from this working approach, and "
        "propose the smallest generic harness change (skill/guidance/tool wiring) that "
        "would make monet reliably follow an equivalent approach. Distill the APPROACH, "
        "not verbatim commands; do not hardcode this task.\n\n"
        f"{approach}\n"
    )


# --------------------------------------------------------------------------- #
# public entry (the integration hook)                                        #
# --------------------------------------------------------------------------- #
def maybe_teacher_block(task: str, trials: Any = None, eval_dir: Any = None) -> str:
    """Strict no-op unless ATELIER_TEACHER_SUPERVISION=1. Returns the contrastive
    block for the analyze prompt, or "" on any failure / when disabled."""
    if not is_enabled():
        return ""
    try:
        # CACHE-ONLY + NON-BLOCKING: read a pre-generated successful teacher trace.
        # Never run an inline solve here (it would stall the analyze step / campaign).
        # A separate batch pre-gen (cursor-agent solver on the failing/flaky set, on
        # Cursor's backend) populates TEACHER_RESULTS_ROOT. Cache miss => no-op.
        traj = _find_successful_rollout(task)
        if not traj:
            return ""
        blk = contrastive_block(task, traj)
        if blk:
            log.info("teacher supervision: INJECTED reference block for %s (%d chars)", task, len(blk))
        return blk
    except Exception as exc:
        log.debug("teacher supervision skipped for %s: %s", task, exc)
        return ""
