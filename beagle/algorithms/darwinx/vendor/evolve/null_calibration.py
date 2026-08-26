"""Tell the judge which task deltas are results and which are coin flips.

WHY THIS EXISTS
───────────────
``reasoned_verdict.render_evidence`` hands the judge a list of
``{task, before_rate, after_rate}`` and asks it to decide whether a child is
better than its parent. The judge is good at building a causal story from a
delta. It has no way to know that on both panels we evolve against, roughly
40% of tasks return a different answer when *nothing* changed:

    TB2.1 panel45   20 always solved,  10 always failed,  15 flaky (33%)
    SWE-V panel50   20 always solved,   9 always failed,  21 flaky (42%)

(measured, 3 and 6 replicates of an unmodified harness; see
progress/20260810/WHY_THE_CAMPAIGN_FOUND_NOTHING.md)

So the judge is regularly shown a task that "went from 0.0 to 1.0 because the
edit fixed it", when the same task goes 0.0 -> 1.0 on a rerun with no edit at
all. It writes a plausible story every time, and those stories became
promotions: the one child promoted in mini_tb21_0809 scored 28/45, which is
exactly what the untouched seed scores on a rerun.

The existing ``k_samples`` caveat says "rates over k samples are noisy". That is
true and far too weak — it is the same sentence for every task on every panel.
This module makes it specific: *this* task flipped on 4 of 6 identical runs, and
*this* panel cannot resolve anything smaller than 8 tasks.

WHAT IT DOES NOT DO
───────────────────
It does not veto, score, or override anything. It annotates evidence. The
decision stays with the judge, which is the DarwinX gate-as-enabler position:
give the decision better information rather than a harder threshold. A task
absent from the null is reported as unknown rather than assumed stable —
silence should not read as a clean bill of health.

RELATION TO ``dx.fitness.noise``
────────────────────────────────
``dx_pipeline`` already has a fuller treatment (``NullModel``, exact
convolution of the null delta distribution, ``resolvable_delta``). Nothing in
``pipeline/`` imports ``dx`` — the two packages have never been connected, which
is why the analysis existed for weeks while campaigns kept promoting noise.
This module is deliberately standalone and dependency-free so that wiring it in
is not blocked on that integration; the arithmetic here is the subset needed to
annotate evidence, and ``dx.fitness.noise`` remains the reference for gating.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Two-sided alpha=0.05 plus power 0.80, the usual pairing. Only discordant tasks
# carry information about a difference, so the panel's flip rate sets the scale
# rather than p(1-p).
_Z_ALPHA_PLUS_BETA = 1.959964 + 0.8416212

STABLE_SOLVED = "stable-solved"
STABLE_FAILED = "stable-failed"
FLAKY = "flaky"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class PanelNull:
    """Per-task behaviour of a panel under a fixed harness.

    ``outcomes`` maps task -> the outcome it produced in each replicate. Built
    from >=2 runs of the *same* harness on the *same* panel; anything else is
    measuring the harness, not the panel.
    """

    outcomes: dict[str, tuple[float, ...]]
    label: str = ""

    @property
    def n_replicates(self) -> int:
        return max((len(v) for v in self.outcomes.values()), default=0)

    @property
    def panel_size(self) -> int:
        return len(self.outcomes)

    def status(self, task: str) -> str:
        vals = self.outcomes.get(task)
        if not vals:
            return UNKNOWN
        if all(v == vals[0] for v in vals):
            return STABLE_SOLVED if vals[0] >= 1.0 else STABLE_FAILED
        return FLAKY

    def flip_count(self, task: str) -> tuple[int, int]:
        """(times solved, replicates) for a task; (0, 0) when unmeasured."""
        vals = self.outcomes.get(task)
        if not vals:
            return (0, 0)
        return (sum(1 for v in vals if v >= 1.0), len(vals))

    @property
    def flaky_tasks(self) -> list[str]:
        return sorted(t for t in self.outcomes if self.status(t) == FLAKY)

    @property
    def flaky_frac(self) -> float:
        return (len(self.flaky_tasks) / self.panel_size) if self.panel_size else 0.0

    @property
    def psi(self) -> float:
        """Mean pairwise discordance: how often two identical runs disagree.

        Averaged over replicate pairs rather than derived from ``flaky_frac``,
        because a task that flips once in six runs and one that flips three
        times contribute very differently to how much a score moves.
        """
        tasks = list(self.outcomes)
        if not tasks or self.n_replicates < 2:
            return 0.0
        n = self.n_replicates
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        total = 0.0
        for i, j in pairs:
            disc = sum(1 for t in tasks
                       if len(self.outcomes[t]) > max(i, j)
                       and self.outcomes[t][i] != self.outcomes[t][j])
            total += disc / len(tasks)
        return total / len(pairs) if pairs else 0.0

    def mde(self, k: int = 1) -> float:
        """Smallest paired difference this panel can resolve, as a rate."""
        if self.panel_size <= 0 or self.psi <= 0 or k <= 0:
            return 0.0
        return _Z_ALPHA_PLUS_BETA * math.sqrt(self.psi / (k * self.panel_size))

    @property
    def thin(self) -> bool:
        """Too few replicates for the stable/flaky split to mean much.

        With r replicates a task that truly flips 10% of the time still looks
        stable with probability 0.9**r + 0.1**r — 73% at r=3, 53% at r=6. So
        the flaky set is always a lower bound, and badly so when r is small.
        """
        return self.n_replicates < 5

    @classmethod
    def from_replicates(cls, replicates: Sequence[dict[str, float]],
                        label: str = "") -> PanelNull:
        if len(replicates) < 2:
            raise ValueError(
                "a panel null needs >=2 runs of the same harness on the same "
                f"panel; got {len(replicates)}. With one run there is nothing to "
                "compare and every delta looks real."
            )
        common = set(replicates[0])
        for r in replicates[1:]:
            common &= set(r)
        return cls(
            outcomes={t: tuple(float(r[t]) for r in replicates) for t in sorted(common)},
            label=label,
        )

    @classmethod
    def from_json(cls, path: str) -> PanelNull:
        """Load from ``{"label": ..., "replicates": [{task: reward}, ...]}``."""
        with open(path) as fh:
            blob = json.load(fh)
        reps = blob["replicates"] if isinstance(blob, dict) else blob
        label = blob.get("label", os.path.basename(path)) if isinstance(blob, dict) else path
        return cls.from_replicates(reps, label=label)


def load_from_env() -> PanelNull | None:
    """``DARWINX_GATE_PANEL_NULL=/path/to/null.json``; absent or broken => None.

    Defensive on purpose: a malformed null file must degrade to today's
    behaviour, never take down a campaign that would otherwise have run.
    """
    path = os.environ.get("DARWINX_GATE_PANEL_NULL", "").strip()
    if not path or not os.path.exists(path):
        return None
    try:
        return PanelNull.from_json(path)
    except Exception:  # noqa: BLE001 — evidence is optional, the loop is not
        return None


def annotate_delta(task: str, before: Any, after: Any,
                   null: PanelNull | None) -> str:
    """One evidence line for a task, carrying its measured stability."""
    tag = ""
    try:
        if after is not None and before is not None:
            tag = " [up]" if after > before else (
                " [down]" if after < before else " [unchanged]")
    except TypeError:
        pass
    if null is None:
        return f"- {task}: {before} -> {after}{tag}"
    status = null.status(task)
    if status == FLAKY:
        solved, reps = null.flip_count(task)
        note = (f" ** FLAKY: solved {solved}/{reps} times with NO harness change "
                f"— this delta is not evidence **")
    elif status == STABLE_SOLVED:
        note = f" [stable: solved in all {null.n_replicates} null runs]"
    elif status == STABLE_FAILED:
        note = f" [stable: failed in all {null.n_replicates} null runs]"
    else:
        note = " [not in the measured null — stability unknown]"
    return f"- {task}: {before} -> {after}{tag}{note}"


def evidence_header(null: PanelNull | None, k: int | None = None) -> str | None:
    """Panel-level statement of what these numbers can and cannot show."""
    if null is None or null.panel_size == 0:
        return None
    k = max(1, k or 1)
    mde_rate = null.mde(k)
    mde_tasks = mde_rate * null.panel_size
    parts = [
        f"## Panel null (MEASURED over {null.n_replicates} runs of an UNCHANGED harness"
        + (f", {null.label}" if null.label else "") + ")",
        f"Of {null.panel_size} tasks: {sum(1 for t in null.outcomes if null.status(t) == STABLE_SOLVED)} "
        f"always solved, {sum(1 for t in null.outcomes if null.status(t) == STABLE_FAILED)} always failed, "
        f"**{len(null.flaky_tasks)} flaky ({null.flaky_frac:.0%})**. Two identical runs "
        f"disagree on {null.psi:.0%} of tasks.",
        f"At k={k} this panel cannot resolve a difference smaller than "
        f"**{mde_tasks:.1f} tasks ({mde_rate:.1%})**. A delta below that is "
        f"consistent with re-running the parent unchanged, whatever story the "
        f"diff suggests.",
    ]
    if null.thin:
        parts.append(
            f"CAUTION: only {null.n_replicates} null runs, so the flaky set is a "
            f"lower bound — a task flipping 10% of the time looks stable "
            f"{0.9 ** null.n_replicates + 0.1 ** null.n_replicates:.0%} of the time here."
        )
    return "\n".join(parts)


def net_stable_delta(task_deltas: Iterable[dict[str, Any]],
                     null: PanelNull | None) -> tuple[int, int, int]:
    """(improved, regressed, ignored) counting only tasks the null calls stable.

    The stable subset is where a change is attributable. Movement on the flaky
    subset is reported to the judge but excluded here, because including it is
    exactly how a re-roll of the coins gets read as a gain.
    """
    up = down = ignored = 0
    for d in task_deltas:
        t = d.get("task", "?")
        b, a = d.get("before_rate"), d.get("after_rate")
        if b is None or a is None:
            continue
        if null is not None and null.status(t) in (FLAKY, UNKNOWN):
            ignored += 1
            continue
        try:
            if a > b:
                up += 1
            elif a < b:
                down += 1
        except TypeError:
            continue
    return up, down, ignored
