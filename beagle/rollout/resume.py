"""Resume/retry selection — the single source of truth for *which tasks re-run*.

Both the live runner (:mod:`beagle.rollout.runner`) and the ``beagle evaluate --dry-run`` preview
call :func:`plan_resume`, so the previewed plan is byte-for-byte the plan that runs.

Every task lands in one intrinsic category (all signals normalized by the harness's ``completed()``
into :class:`TaskResult`); the error-vs-no-error split is deliberately the ONLY benchmark-invariant
signal — *was an error ever seen?* — not a guess about whether an error is "deterministic" (that
shifts run-to-run and benchmark-to-benchmark, so it's the user's call, not beagle's):

* **missing** — no prior result on disk (interrupted).
* **error** — the trial recorded ANY error (a 500, an agent/verifier timeout, a transient clone, a
  no-attempt, …). Whether re-running helps is benchmark-dependent, so it's the user's discretion.
* **genuine-fail** — unresolved with NO error: the agent produced a real attempt that didn't pass. A
  true capability failure.
* **resolved** — passed.

The three flags are **independent** and each re-runs exactly its own category (they compose by union):

    --resume            → missing
    --retry-errors      → error
    --retry-unresolved  → error + genuine-fail        (the whole non-pass class with a result)

A **resolved** task is never re-run. With NO flag it's a fresh run (everything runs). ``--retry-``
``unresolved`` needs a real resolved signal, so a trial with neither a reward nor an error (ungraded)
fails loud instead of being re-run blind."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beagle.benchmarks.base import Task, TaskContext, TaskResult

#: Intrinsic failure categories (what KIND of outcome the prior trial had), independent of flags.
CATEGORIES = ("missing", "resolved", "error", "genuine-fail")


@dataclass
class TaskDecision:
    """One task's resume verdict, for the ``--dry-run`` table. ``category`` is intrinsic (what the
    prior trial was); ``retry`` is the flag-derived decision to re-run it."""

    task_id: str
    retry: bool
    category: str      # one of CATEGORIES
    signal: str        # short human reason (error head, ``reward=…``, ``no result``)


@dataclass
class ResumePlan:
    """The re-run decision for one benchmark group. ``keep`` is the runner's ``done`` map;
    ``rerun_ids`` (in ``items`` order) is what re-runs; ``decisions`` is the per-task detail."""

    keep: dict[str, TaskResult] = field(default_factory=dict)
    rerun_ids: list[str] = field(default_factory=list)
    decisions: list[TaskDecision] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for d in self.decisions:
            c[d.category] = c.get(d.category, 0) + 1
        return c


def _short(error: str) -> str:
    """A compact one-line signal from a full error string: the ``<Type>`` head plus a trimmed tail."""
    head, _, tail = error.partition(":")
    head, tail = head.strip(), tail.strip()
    return f"{head}: {tail[:48]}" if tail else head[:60]


def _fmt_reward(reward: float | None) -> str:
    return f"reward={reward}" if reward is not None else "reward=None"


def plan_resume(
    items: list[tuple[Task, TaskContext]],
    prior: list[TaskResult],
    *,
    resume: bool = False,
    retry_errors: bool = False,
    retry_unresolved: bool = False,
    only_task_ids: set[str] | None = None,
    label: str = "",
) -> ResumePlan:
    """Decide, per task in ``items``, whether it re-runs given the prior ``completed()`` results.

    The three flags are independent and each re-runs exactly its own category (they compose by
    union): ``resume`` → **missing**, ``retry_errors`` → **error**, ``retry_unresolved`` → **error +
    genuine-fail**. A **resolved** task is never re-run. With NO flag it's a fresh run — every task
    (all ``missing``, since ``prior`` is empty) runs.

    ``only_task_ids`` (from ``--task-ids``) is a re-run **restriction**, not a dataset filter: a task
    NOT in the set is never re-run, but if it has a prior result it's still **kept** (so it stays in
    ``run.json`` — the aggregate covers the whole benchmark, not just the scoped subset).

    Raises ``RuntimeError`` if ``retry_unresolved`` is set but a prior trial carries no gradeable
    signal (no reward, no error)."""
    prior_by_id = {r.task_id: r for r in prior}
    fresh = not (resume or retry_errors or retry_unresolved)   # no flag → run everything

    if retry_unresolved:
        blind = [r.task_id for r in prior if r.reward is None and r.error is None]
        if blind:
            raise RuntimeError(
                f"--retry-unresolved needs a per-task resolved signal, but {len(blind)} "
                f"completed {label or 'trial'!r} trial(s) have NO resolved signal (no reward, no "
                f"error; first: {blind[0]!r}) — 'unresolved' can't be distinguished from 'ungraded'. "
                f"Grade the run first, or use --retry-errors (which keys off the error signal).")

    plan = ResumePlan()
    for task, _ctx in items:
        r = prior_by_id.get(task.task_id)
        # intrinsic category (WHAT the prior trial was) → retry decision (which FLAG covers it).
        if r is None:
            cat, sig = "missing", "no result on disk"
            retry = resume or fresh                       # --resume re-runs missing; fresh runs all
        elif r.resolved:
            cat, sig, retry = "resolved", _fmt_reward(r.reward), False   # a pass is never re-run
        elif r.error:
            cat, sig = "error", _short(r.error)
            retry = retry_errors or retry_unresolved      # --retry-errors ⊆ --retry-unresolved
        else:
            cat, sig = "genuine-fail", _fmt_reward(r.reward)
            retry = retry_unresolved                      # a clean unresolved attempt: only --retry-unresolved
        if only_task_ids is not None and task.task_id not in only_task_ids:
            retry = False                                 # --task-ids restricts the re-run set (kept if it has a result)

        plan.decisions.append(TaskDecision(task.task_id, retry, cat, sig))
        if retry:
            plan.rerun_ids.append(task.task_id)
        elif r is not None:
            plan.keep[task.task_id] = r
        # a missing task that isn't re-run (e.g. --retry-errors alone) is neither re-run nor kept —
        # it's simply left absent, exactly as asked.
    return plan
