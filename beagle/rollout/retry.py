"""Task-level retry primitives, shared by every harness shape.

Two independent layers, mirroring the xrlenv benchmark sweeps
(``xrlenv_plugins/benchmarks/*/run_oracle_sweep.py``):

* **infra retry** — re-run a trial ONLY on an infra-transient error (a cluster
  acquire / capacity / node blip), matched by ``type(exc).__name__`` against
  :data:`INFRA_RETRY_EXCEPTIONS`. A *content* outcome (agent timeout, verifier
  failure, ``resolved=False``, an API usage limit) is **never** infra-retried, so
  eval signal is never re-rolled into a fluke pass. On the harbor path this rides
  harbor's own ``RetryConfig`` (the trial queue retries in a fresh container); on the
  per-task (docker) path :func:`run_with_infra_retry` wraps the call.
* **content retry** — re-run UNRESOLVED tasks up to N times; a task counts solved if
  ANY attempt passes. This absorbs rate-limit / transient content flakes and lives in
  the Runner (harness-agnostic), not here.

The exception set matches the harbor / pier / vanilla oracle sweeps' ``RetryConfig``
``include_exceptions`` whitelist — keep it in sync with those.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from beagle.types import TaskResult

#: Infra-transient exception type names the trial queue may retry — and ONLY these.
#: (An acquire that times out waiting for a slot, a control-plane restart, a node that
#: drops its stream, a node RPC deadline.) Content failures — ``AgentTimeoutError``,
#: verifier errors, ``ApiUsageLimitError`` — are deliberately OUT: a rate-limited or
#: budget-exhausted agent is a content outcome, handled by the content-retry layer.
INFRA_RETRY_EXCEPTIONS = frozenset({
    "CapacityExhausted",   # admission queue timed out waiting for a slot
    "ControlPlaneLost",    # CP restarted under the run
    "NodeLost",            # node dropped its stream mid-acquire
    "NodeCommandTimeout",  # a node RPC deadline (teardown / exec) tripped
})

_T = TypeVar("_T")


def is_infra_error(exc: BaseException | str) -> bool:
    """True if ``exc`` (an exception or its type name) is an infra-transient we may retry.

    For an exception, walks the ``__cause__`` / ``__context__`` chain and matches if ANY link's
    type name is in the whitelist. This is load-bearing on the docker path: ``XrlenvDockerRuntime``
    wraps xrlenv errors in a generic ``RuntimeError`` (``raise RuntimeError(...) from e``), so the
    real ``CapacityExhausted`` / ``NodeLost`` sits on ``.__cause__``. Matching only the outermost
    type would miss every wrapped infra error, silently defeating ``run_with_infra_retry`` (the
    harbor path is unaffected — harbor matches the raw exception inside its own trial queue)."""
    if isinstance(exc, str):
        return exc in INFRA_RETRY_EXCEPTIONS
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:  # id() guard: __context__ chains can cycle
        seen.add(id(cur))
        if type(cur).__name__ in INFRA_RETRY_EXCEPTIONS:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def run_with_infra_retry(fn: Callable[[], _T], *, attempts: int) -> _T:  # noqa: UP047 — TypeVar keeps py3.10 compat
    """Call ``fn`` up to ``attempts`` times, retrying ONLY on an infra-transient error
    (:func:`is_infra_error`); any other exception propagates immediately, and the last
    infra error is re-raised once attempts are exhausted. ``attempts <= 1`` runs once.

    The per-task (docker) analogue of harbor's ``RetryConfig``: each retry re-runs the
    whole task in a fresh container, so a post-acquire infra blip re-executes cleanly.
    """
    n = max(1, attempts)
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            if not is_infra_error(e) or i >= n - 1:
                raise
    raise RuntimeError("unreachable: run_with_infra_retry loop always returns or raises")


def better_attempt(current: TaskResult | None, candidate: TaskResult) -> bool:
    """Content-retry pick rule: keep ``candidate`` over ``current`` if there is no current
    result, or ``current`` is unresolved and ``candidate`` resolves, or ``candidate`` has a
    strictly higher reward. A task counts solved if ANY attempt resolves."""
    if current is None:
        return True
    if not current.resolved and candidate.resolved:
        return True
    if current.resolved != candidate.resolved:
        return False
    cur_r = current.reward if current.reward is not None else float("-inf")
    cand_r = candidate.reward if candidate.reward is not None else float("-inf")
    return cand_r > cur_r


__all__ = ["INFRA_RETRY_EXCEPTIONS", "better_attempt", "is_infra_error", "run_with_infra_retry"]
