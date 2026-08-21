"""``beagle.rollout.resume.plan_resume`` — the shared resume/retry decision (runner + --dry-run).

Each task lands in one intrinsic category; the retry decision is derived from the flags. A resolved
task is ALWAYS kept; ``--retry-errors`` re-runs only retryable errors (not deterministic ones);
``--retry-unresolved`` re-runs every non-pass and fails loud on an ungraded (blind) trial."""

from __future__ import annotations

import pytest

from beagle.rollout.resume import plan_resume
from beagle.types import Task, TaskContext, TaskResult


def _items(*ids: str):
    return [(Task(task_id=i, benchmark="b"), TaskContext(image=None)) for i in ids]


def _r(task_id: str, *, resolved=False, reward=0.0, error=None) -> TaskResult:
    return TaskResult(task_id=task_id, resolved=resolved, reward=reward, error=error)


def _by_id(plan):
    return {d.task_id: d for d in plan.decisions}


def test_categories_are_intrinsic() -> None:
    # The split is by the ONLY benchmark-invariant signal: an error was seen (E) vs a clean
    # unresolved attempt (F). A clone error and a timeout are BOTH just 'error' — beagle doesn't
    # guess which is "deterministic" (that's benchmark/run-dependent, the user's call).
    items = _items("pass", "clone", "timeout", "genuine", "gone")
    prior = [
        _r("pass", resolved=True, reward=1.0),
        _r("clone", error="git clone failed (rc=128): HTTP 401"),
        _r("timeout", error="AgentTimeoutError: timed out after 900s"),
        _r("genuine", reward=0.0),                       # unresolved, no error
        # 'gone' has no prior result → missing
    ]
    d = _by_id(plan_resume(items, prior))
    assert d["pass"].category == "resolved"
    assert d["clone"].category == "error"                # E — any error
    assert d["timeout"].category == "error"              # E — timeout is NOT a special class
    assert d["genuine"].category == "genuine-fail"       # F — no error, not solved
    assert d["gone"].category == "missing"


def test_resume_only_reruns_missing() -> None:
    items = _items("pass", "clone", "genuine", "gone")
    prior = [_r("pass", resolved=True, reward=1.0),
             _r("clone", error="git clone failed: HTTP 401"),
             _r("genuine", reward=0.0)]
    plan = plan_resume(items, prior, resume=True)   # --resume: ONLY missing (leaves error/genuine)
    assert plan.rerun_ids == ["gone"]
    assert set(plan.keep) == {"pass", "clone", "genuine"}


def test_retry_errors_alone_does_not_touch_missing() -> None:
    # Independent flags: --retry-errors re-runs errored ONLY. A missing task is left alone (neither
    # re-run nor kept) — add --resume to also finish it. They compose by union.
    items = _items("clone", "gone")
    prior = [_r("clone", error="git clone failed: HTTP 401")]   # 'gone' has no result → missing
    plan = plan_resume(items, prior, retry_errors=True)
    assert plan.rerun_ids == ["clone"]          # error re-run; missing NOT (no --resume)
    assert set(plan.keep) == set()              # 'gone' is skipped, not kept
    union = plan_resume(items, prior, resume=True, retry_errors=True)
    assert set(union.rerun_ids) == {"clone", "gone"}


def test_retry_errors_reruns_all_errors_but_not_genuine_fail() -> None:
    # --retry-errors re-runs the WHOLE error class E (clone AND timeout — user's discretion), but
    # never a genuine capability failure F (no error) or a resolved task.
    items = _items("clone", "timeout", "genuine", "pass")
    prior = [_r("clone", error="git clone failed: HTTP 401"),
             _r("timeout", error="VerifierTimeoutError: timed out after 900s"),
             _r("genuine", reward=0.0),
             _r("pass", resolved=True, reward=1.0)]
    plan = plan_resume(items, prior, retry_errors=True)
    assert set(plan.rerun_ids) == {"clone", "timeout"}    # all E
    assert set(plan.keep) == {"genuine", "pass"}          # F + resolved kept


def test_resolved_is_always_kept_even_with_error() -> None:
    # An agent that solved the task then timed out on teardown lands resolved=True WITH an error.
    items = _items("solved_then_timed_out")
    prior = [_r("solved_then_timed_out", resolved=True, reward=1.0,
                error="AgentTimeoutError: timed out after 1800s")]
    for flags in ({}, {"retry_errors": True}, {"retry_unresolved": True}):
        plan = plan_resume(items, prior, **flags)
        assert plan.rerun_ids == [], flags
        assert set(plan.keep) == {"solved_then_timed_out"}


def test_retry_unresolved_reruns_every_nonpass() -> None:
    items = _items("clone", "timeout", "genuine", "pass")
    prior = [_r("clone", error="git clone failed: HTTP 401"),
             _r("timeout", error="AgentTimeoutError: timed out after 900s"),
             _r("genuine", reward=0.0),
             _r("pass", resolved=True, reward=1.0)]
    plan = plan_resume(items, prior, retry_unresolved=True)
    assert set(plan.rerun_ids) == {"clone", "timeout", "genuine"}   # superset of --retry-errors
    assert set(plan.keep) == {"pass"}


def test_only_task_ids_restricts_rerun_but_keeps_the_rest() -> None:
    # --task-ids scopes which tasks re-run; an out-of-scope errored task is KEPT (stays in run.json),
    # not re-run — so the aggregate still covers the whole benchmark.
    items = _items("e1", "e2", "pass", "g1")
    prior = [_r("e1", error="git clone failed"), _r("e2", error="git clone failed"),
             _r("pass", resolved=True, reward=1.0), _r("g1", reward=0.0)]
    plan = plan_resume(items, prior, retry_errors=True, only_task_ids={"e1"})
    assert plan.rerun_ids == ["e1"]                 # only the in-scope errored task re-runs
    assert set(plan.keep) == {"e2", "pass", "g1"}   # e2 (errored, out of scope) is KEPT, not re-run
    assert len(plan.keep) + len(plan.rerun_ids) == 4  # every task accounted for → run.json stays whole


def test_retry_unresolved_fails_loud_on_ungraded() -> None:
    # A trial with neither reward nor error is UNGRADED — 'unresolved' can't be distinguished from it.
    items = _items("blind")
    prior = [_r("blind", reward=None, error=None)]
    with pytest.raises(RuntimeError, match="resolved signal"):
        plan_resume(items, prior, retry_unresolved=True)
    # but --retry-errors (keys off the error signal) does NOT trip the guard
    assert plan_resume(items, prior, retry_errors=True).rerun_ids == []


def test_no_prior_reruns_everything() -> None:
    items = _items("a", "b", "c")
    plan = plan_resume(items, [])   # not resuming → every task is 'missing'
    assert plan.rerun_ids == ["a", "b", "c"]
    assert [d.category for d in plan.decisions] == ["missing", "missing", "missing"]
