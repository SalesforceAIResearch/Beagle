"""How much the model spends per task under a candidate harness.

WHY THIS EXISTS
───────────────
`harness_complexity` measures the harness as a *text*: lines and branches. That
is what an author changed, not what the change did. A rewrite can shed lines and
still leave the model flailing, and a good consolidation can hold line count
while making every step easier to decide.

The measured case for grow-then-consolidate (docs/RESULTS_0802.md §2) came out
of the second axis, not the first. Over 180 SWE-V tasks the consolidated harness
resolved the same tasks as its parent with 7.2% fewer API calls (p=0.0032) and
21.8% fewer reasoning tokens (p=0.0002) -- and per call, reasoning fell 14.8%
(p=0.0011) while cost, prompt and completion length did not move. Capability
over the same 180 tasks was 6-2 in its favour at p=0.29: not significant.

So the only axis on which a real harness change produced a resolvable effect is
one the loop cannot currently see. The picker reads score and line count, both
of which are flat for a good consolidation, which is precisely why it kept
abandoning them.

WHICH NUMBER, AND WHY A VOTE
────────────────────────────
Running the *same* harness twice on the same 60 SWE-V tasks (progress
20260803/06_PRO_PROBE.md) disqualified most of the list above. Per-task means
moved this much with nothing changed at all:

* cost              −2.3%   against a −6.1% harness effect
* api_calls         −1.3%   against −7.3%
* reasoning tokens **+10.2%** against −21.4%

Cost and API calls are the two obvious things to optimise and both are inside
their own noise: re-running moved cost down on 57% of tasks, changing the
harness on 58%. A picker selecting on either is reading the dice.

Reasoning tokens do separate, but the *mean* is a bad way to say so -- it is
heavy-tailed, a few deep-thinking tasks dominate any average, and that is where
the +10.2% came from. The statistic that separated cleanly is the **per-task
vote**: on what share of tasks did the metric drop at all? The harness moved 66%
of tasks down (sign test p=0.0001); a re-run moved 40% (p=0.24). So ``delta``
reports the vote alongside the means, and the vote is what the picker should read.

WHAT IS MEASURED
────────────────
Per task, from the mini-eval's own trajectory logs (already archived under the
node's ``evals/`` dir, so this costs no compute and no extra run):

* ``reasoning``      — reasoning tokens spent on the task. The signal.
* ``api_calls``      — model round-trips to finish the task.
* ``reasoning_per_call`` — reasoning tokens per round-trip, which survives
  dividing out the call reduction and so speaks to clarity rather than length.
* ``cost``           — dollars, when the trajectory recorded it. Reported for
  bookkeeping and explicitly *not* for selection: see above.

DELIBERATE NON-GOALS
────────────────────
Same rule as complexity: this is one half of a two-sided test and must never
justify a candidate on its own. A harness that gives up in one call is maximally
"efficient". Efficiency is only meaningful across tasks the parent *also*
solved, so callers must pass the comparison set.

A caveat the vote does not remove: a −20% effect is diffuse across tasks, so it
needs ~60-70 shared tasks to resolve, while an iteration's mini-eval is 8-13.
Over a mini-eval this signal is directional at best. Reading it as a verdict
requires a fixed shared panel, which is a change to the evaluation loop rather
than to this file.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
from dataclasses import dataclass
from pathlib import Path

# Guard against walking a huge archive on Lustre: trajectories are one file per
# task and a mini-eval is tens of tasks, so anything past this is a wrong dir.
_MAX_FILES = 400

# Below this many shared tasks the vote is reported but flagged as thin, because
# the effect it is looking for needs ~60 to resolve.
_THIN_VOTE = 20


@dataclass(frozen=True)
class Spend:
    tasks: int = 0
    api_calls: float = 0.0
    reasoning_per_call: float = 0.0
    cost: float = 0.0

    def as_dict(self) -> dict:
        return {
            "tasks": self.tasks,
            "api_calls": round(self.api_calls, 2),
            "reasoning_per_call": round(self.reasoning_per_call, 1),
            "cost": round(self.cost, 4),
        }


def _per_task(path: Path) -> tuple[str, int, int, float] | None:
    """(task_id, api_calls, reasoning_tokens, cost) from one trajectory file."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    stats = (d.get("info") or {}).get("model_stats") or {}
    calls = int(stats.get("api_calls") or 0)
    if not calls:
        return None
    reasoning = 0
    for m in d.get("messages") or []:
        usage = (((m or {}).get("extra") or {}).get("response") or {}).get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning += int(details.get("reasoning_tokens") or 0)
    task_id = path.name
    for suffix in (".log.json", ".json"):
        if task_id.endswith(suffix):
            task_id = task_id[: -len(suffix)]
            break
    return task_id, calls, reasoning, float(stats.get("instance_cost") or 0.0)


def measure(job_dir: str | Path, tasks: set[str] | None = None) -> tuple[Spend, dict[str, tuple[int, int, float]]]:
    """Mean spend over a mini-eval's trajectories, optionally restricted to ``tasks``."""
    root = Path(job_dir)
    per: dict[str, tuple[int, int, float]] = {}
    if not root.is_dir():
        return Spend(), per
    seen = 0
    for path in root.rglob("*.json"):
        seen += 1
        if seen > _MAX_FILES:
            break
        row = _per_task(path)
        if row is None:
            continue
        task_id, calls, reasoning, cost = row
        if tasks is not None and task_id not in tasks:
            continue
        # A task can be retried; keep the costliest attempt rather than a random
        # one, so a candidate cannot look cheap by having its retry counted.
        prev = per.get(task_id)
        if prev is None or calls > prev[0]:
            per[task_id] = (calls, reasoning, cost)
    if not per:
        return Spend(), per
    calls = [v[0] for v in per.values()]
    rpc = [v[1] / v[0] for v in per.values() if v[0]]
    return Spend(
        tasks=len(per),
        api_calls=st.mean(calls),
        reasoning_per_call=st.mean(rpc) if rpc else 0.0,
        cost=st.mean(v[2] for v in per.values()),
    ), per


def sign_test_p(down: int, total: int) -> float:
    """Two-sided exact binomial p for ``down`` of ``total`` under p=0.5.

    Exact rather than normal-approximated because a mini-eval's shared set is
    small enough that the approximation is wrong where it matters.
    """
    if total <= 0:
        return 1.0
    counts = [math.comb(total, i) for i in range(total + 1)]
    tail = sum(c for c in counts if c <= counts[down] + 1e-9)
    return min(1.0, tail / float(sum(counts)))


def _vote(parent_per: dict, child_per: dict, shared: list[str]) -> dict:
    """Share of tasks on which the child spent fewer reasoning tokens.

    Magnitude-free on purpose: this is the statistic that survived the noise
    floor, where the ratio of means did not.
    """
    down = sum(1 for t in shared if child_per[t][1] < parent_per[t][1])
    tied = sum(1 for t in shared if child_per[t][1] == parent_per[t][1])
    return {
        "vote_tasks": len(shared),
        "vote_down": down,
        "vote_tied": tied,
        "vote_frac": round(down / float(len(shared)), 3) if shared else 0.0,
        "vote_p": round(sign_test_p(down, len(shared)), 4),
        "vote_thin": len(shared) < _THIN_VOTE,
    }


def delta(parent_job_dir: str | Path | None, child_job_dir: str | Path | None,
          tasks: set[str] | None = None) -> dict | None:
    """Paired spend comparison over the tasks both runs completed.

    Paired rather than mean-vs-mean because mini-eval task sets differ between
    iterations, and an unpaired mean would mostly measure which tasks were drawn.

    ``tasks`` should be the set both arms solved. Without it a candidate that
    fails a task early is credited with the cheap run.
    """
    if not parent_job_dir or not child_job_dir:
        return None
    _, parent_per = measure(parent_job_dir, tasks)
    _, child_per = measure(child_job_dir, tasks)
    shared = sorted(set(parent_per) & set(child_per))
    if not shared:
        return None
    p_calls = st.mean(parent_per[t][0] for t in shared)
    c_calls = st.mean(child_per[t][0] for t in shared)
    p_rpc = st.mean(parent_per[t][1] / parent_per[t][0] for t in shared)
    c_rpc = st.mean(child_per[t][1] / child_per[t][0] for t in shared)
    p_reason = st.mean(parent_per[t][1] for t in shared)
    c_reason = st.mean(child_per[t][1] for t in shared)
    out = {
        "shared_tasks": len(shared),
        "d_api_calls": round(c_calls - p_calls, 2),
        "d_api_calls_pct": round((c_calls - p_calls) / p_calls * 100, 1) if p_calls else 0.0,
        "d_reasoning_per_call": round(c_rpc - p_rpc, 1),
        "d_reasoning_per_call_pct": round((c_rpc - p_rpc) / p_rpc * 100, 1) if p_rpc else 0.0,
        "d_reasoning_pct": round((c_reason - p_reason) / p_reason * 100, 1) if p_reason else 0.0,
        "before": {"api_calls": round(p_calls, 2), "reasoning_per_call": round(p_rpc, 1),
                   "reasoning": round(p_reason, 1)},
        "after": {"api_calls": round(c_calls, 2), "reasoning_per_call": round(c_rpc, 1),
                  "reasoning": round(c_reason, 1)},
    }
    out.update(_vote(parent_per, child_per, shared))
    return out


def summary_line(parent_job_dir: str | Path | None, child_job_dir: str | Path | None,
                 tasks: set[str] | None = None) -> str:
    """One line for the picker prompt, or "" when there is nothing to say.

    Leads with the vote because that is the part that survived a noise floor;
    the percentages follow so a reader can see the size, and API calls are
    included for context only -- they did not survive.
    """
    try:
        d = delta(parent_job_dir, child_job_dir, tasks)
    except Exception:
        return ""
    if not d:
        return ""
    thin = " — thin, directional only" if d.get("vote_thin") else ""
    return (f"reasoning tokens down on {d['vote_down']}/{d['vote_tasks']} tasks "
            f"(sign p={d['vote_p']:.3f}), mean {d['d_reasoning_pct']:+.1f}%; "
            f"API calls {d['d_api_calls_pct']:+.1f}% (context only, inside noise)"
            f"{thin}")


def enabled() -> bool:
    """Off by default: it changes what the picker sees, so opt in per campaign."""
    return os.environ.get("DARWINX_GATE_EFFICIENCY_SIGNAL", "").strip().lower() in {"1", "true", "yes", "on"}
