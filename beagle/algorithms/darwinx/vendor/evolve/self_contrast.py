"""self_contrast.py -- OPTIONAL GRPO-style self-rollout contrastive supervision (PROTOTYPE).

WHAT
----
Where ``teacher_supervision`` injects an EXTERNAL solver's (cursor-agent) successful
trajectory, this module injects monet's OWN group-relative signal: among monet's k
self-rollouts on a task, contrast a PASSING rollout against a FAILING one and distill
"what the win did that the loss didn't" into the analyze prompt. This is the GRPO
*advantage* (pass - fail) applied at the SCAFFOLD level (we cannot train gpt-5.5's
weights; we improve the harness instead).

WHERE IT HAS SIGNAL (and where it does not)
-------------------------------------------
- VARIANCE band (task sometimes passes, sometimes fails): both a positive and a
  negative rollout exist -> a clean group-relative contrast -> THIS is the target.
  This is exactly parentA's headroom (21 flaky tasks at 1-4/5).
- WALLS (0/k): no positive rollout in the group -> degenerate advantage (the classic
  GRPO-on-hard-tasks failure). Returns "" -> fall back to teacher_supervision there.

So self-contrast (variance) and teacher (walls) are COMPLEMENTARY, not redundant.

STATUS
------
Prototype. DISABLED by default. Enable with ``DARWINX_GATE_SELF_CONTRAST=1``. Cache-only +
non-blocking: it only READS already-produced monet trials (the k-sample evals the
campaign/baseline already generate); it never launches a rollout inline. No import or
flag => zero behaviour change.

INTEGRATION (apply only when piloting)
--------------------------------------
In ``self_evolve/trace_qc.analyze_digest_block`` (same site as the teacher hook)::

    from .self_contrast import maybe_self_contrast_block
    block += maybe_self_contrast_block(task, trials, eval_dir)

Strict no-op unless ``DARWINX_GATE_SELF_CONTRAST=1``; wrapped so any failure degrades to
the failure-only baseline.

CONFIG
------
- ``DARWINX_GATE_SELF_CONTRAST``         : "1" to enable.
- ``DARWINX_GATE_SELF_CONTRAST_SOURCES`` : comma-separated run dirs (or globs) to mine for
  monet pass/fail trials. Default: the baseline avg@5 run + any pa_clean* reruns.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("self_evolve.self_contrast")

ENV_FLAG = "DARWINX_GATE_SELF_CONTRAST"
#: Comma-separated globs naming the historical run dirs to contrast against. There is no
#: default: which runs are worth contrasting is specific to a deployment and a campaign, so
#: a run that turns self-contrast on has to say where its own history lives.
SOURCES_ENV = "DARWINX_GATE_SELF_CONTRAST_SOURCES"


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "") == "1"


def _sources() -> list[str]:
    raw = os.environ.get(SOURCES_ENV, "")
    if not raw.strip():
        if is_enabled():
            log.warning(
                "%s is on but %s is unset, so there is no history to contrast against",
                ENV_FLAG, SOURCES_ENV,
            )
        return []
    out: list[str] = []
    for pat in (s.strip() for s in raw.split(",") if s.strip()):
        out.extend(sorted(glob.glob(pat)) or [pat])
    return out


def _reward(t: dict) -> float | None:
    r = ((t.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return None if r is None else float(r)


def _traj_for_trial(trial_dir: Path, task: str) -> Path | None:
    base = trial_dir.parent.parent  # .../raw
    name = trial_dir.name
    for cand in (
        base / f"{name}.trajectory.jsonl",
        trial_dir / "agent" / "coding_bench_monet.json",
        trial_dir / "agent" / "trajectory.json",
        base / f"{task}.log",
    ):
        if cand.exists():
            return cand
    return None


def _find_pass_fail_rollouts(task: str) -> tuple[Path | None, Path | None]:
    """Return (passing_traj, failing_traj) among monet's OWN trials for ``task``.
    Group-relative: we need BOTH a win and a loss (variance) for a contrast.
    """
    win: Path | None = None
    loss: Path | None = None
    for run in _sources():
        run = run if run.endswith("/") else run + "/"
        for pat in (f"*__{task}__s*", f"*__{task}"):
            for rj in glob.glob(os.path.join(run, "raw", "trials", pat, "result.json")):
                try:
                    t = json.load(open(rj))
                except Exception:
                    continue
                rew = _reward(t)
                if rew is None:
                    continue
                tdir = Path(rj).parent
                tj = _traj_for_trial(tdir, task)
                if tj is None:
                    continue
                if rew >= 1.0 and win is None:
                    win = tj
                elif rew < 1.0 and loss is None:
                    loss = tj
                if win and loss:
                    return win, loss
    return win, loss


def _summarize_trajectory(traj_path: Path, max_chars: int = 1800) -> list[str]:
    """Extract the APPROACH (shell commands + file ops) from a monet stream-json
    trajectory. The trajectory is JSONL: bash tool calls carry a complete
    ``command`` (top-level or under ``input``); file ops carry a ``path``.
    """
    actions: list[str] = []
    try:
        for line in open(traj_path, "r", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            inp = ev.get("input") if isinstance(ev.get("input"), dict) else {}
            cmd = ev.get("command")
            if not isinstance(cmd, str):
                cmd = inp.get("command") if isinstance(inp.get("command"), str) else None
            if isinstance(cmd, str) and cmd.strip():
                c = " ".join(cmd.split())[:200]
                if c not in actions:
                    actions.append(c)
                continue
            tool = ev.get("toolName") or ev.get("name")
            path = inp.get("path") or ev.get("path")
            if tool in ("file_write", "file_edit") and isinstance(path, str):
                a = f"[{tool}] {path}"
                if a not in actions:
                    actions.append(a)
    except Exception as exc:
        log.debug("self-contrast trajectory parse failed: %s", exc)
    return actions[:30]


def contrastive_block(task: str, win: Path, loss: Path) -> str:
    win_cmds = _summarize_trajectory(win)
    loss_cmds = _summarize_trajectory(loss)
    if not win_cmds:
        return ""
    win_set = set(win_cmds)
    loss_set = set(loss_cmds)
    only_win = [c for c in win_cmds if c not in loss_set]   # the advantage delta
    lines = [
        f"\n\n### Self-rollout contrast (monet PASSED `{task}` on one sample, FAILED on another)",
        "This is YOUR OWN behaviour: the task is *recoverable* (you solve it sometimes). "
        "Treat the passing rollout as the positive and the failing one as the negative "
        "(group-relative advantage). Diagnose what the WIN did that the LOSS didn't, then "
        "propose the smallest generic harness change (skill/guidance/ordering) that makes "
        "monet reliably take the winning approach. Do not hardcode this task.",
    ]
    if only_win:
        lines.append("\nActions present in the WIN but absent from the LOSS (the likely advantage):")
        lines += [f"  + $ {c}" for c in only_win[:18]]
    lines.append("\nFull winning approach (distill, adapt):")
    lines += [f"  $ {c}" for c in win_cmds[:18]]
    return "\n".join(lines)[:3000]


def maybe_self_contrast_block(task: str, trials: Any = None, eval_dir: Any = None) -> str:
    """Strict no-op unless DARWINX_GATE_SELF_CONTRAST=1. Returns the self-contrast block, or
    "" when disabled / no win+loss pair exists (i.e. solved-5/5 or 0/k walls)."""
    if not is_enabled():
        return ""
    try:
        win, loss = _find_pass_fail_rollouts(task)
        if not (win and loss):
            return ""  # no variance -> no group-relative signal (walls: use teacher)
        blk = contrastive_block(task, win, loss)
        if blk:
            log.info("self-contrast: INJECTED group-relative block for %s (%d chars)", task, len(blk))
        return blk
    except Exception as exc:
        log.debug("self-contrast skipped for %s: %s", task, exc)
        return ""
