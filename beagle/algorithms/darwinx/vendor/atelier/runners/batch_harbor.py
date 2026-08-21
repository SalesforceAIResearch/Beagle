"""Batched Harbor invocation shared by honeypot + transfer adapters.

Atelier's runner protocols (``HoneypotRunner``, ``TransferEvaluator``)
are per-task by design — the corresponding scoring helpers (e.g.
``honeypot.score_candidate``) walk task IDs one at a time so callers
can interleave streaming progress, retries, etc.

Harbor, however, pays a non-trivial startup cost per invocation
(docker-compose spin-up, harness import, agent init). Running it 30
times for a 30-task honeypot would dwarf the actual model time. To
keep the protocol-per-task ergonomics but the cost amortized, we
batch:

1. ``BatchHarborRunner.prime(task_ids)`` runs Harbor once for the
   entire batch and caches the EvalResult.
2. The per-task protocol callable looks the task up in the cache.

Atelier callers always know the task ID set up front (it comes from
``honeypot.HoneypotCorpus.default_task_ids`` or a hardcoded transfer
list), so prime + cached-lookup is the natural fit.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from self_evolve import eval_runner


logger = logging.getLogger("atelier.runners.batch_harbor")


@dataclass
class HarborInvocation:
    """The set of parameters needed to call ``eval_runner.run_full``.

    Kept as a dataclass so callers can construct it once at gate setup
    and pass it around (it carries the per-substrate config / env that
    distinguishes honeypot vs cross-model vs cross-benchmark).
    """

    config_path: Path
    """Path to the Harbor YAML config (TW honeypot config, TB-2 cross-model
    config with model env override, SWE-bench config, etc)."""

    cwd: Path
    """Per-pipeline worktree to run Harbor inside. Typically the same cwd
    that ``self_evolve.pipeline`` uses for its main final eval."""

    subset_label: str = "atelier"
    """Label recorded on ``EvalResult.subset`` for downstream filtering /
    visualization. Does **not** filter tasks — that's task_ids."""

    extra_env: dict[str, str] = field(default_factory=dict)
    """Env overrides forwarded to the Harbor subprocess. Used for
    cross-model transfer (override MONET_EVAL_* model variables) and
    any other substrate-specific config that's not in the YAML."""


class BatchHarborRunner:
    """Run Harbor once for a batch of task_ids; expose per-task results.

    Thread-safe init (the cache is built once under a lock). After
    priming, per-task lookups are read-only and cheap.

    Typical use:

    ```python
    runner = BatchHarborRunner(invocation=HarborInvocation(...))
    runner.prime(task_ids=("1012", "1018", "1025"))
    result_for_1012 = runner.get_per_task("1012")
    ```

    Errors during priming propagate to the caller (the gate will record
    the layer as ``NOT_IMPLEMENTED`` / ``ACCEPT_BY_DEFAULT`` depending on
    its own policy — that's not this module's concern).
    """

    def __init__(self, *, invocation: HarborInvocation) -> None:
        self.invocation = invocation
        self._lock = threading.Lock()
        self._primed_for: tuple[str, ...] | None = None
        self._eval_result: eval_runner.EvalResult | None = None

    @property
    def is_primed(self) -> bool:
        return self._eval_result is not None

    def prime(self, *, task_ids: tuple[str, ...]) -> None:
        """Run Harbor with the given task list. Subsequent get_per_task()
        calls return cached values.

        Re-priming with a *different* task list is allowed (re-runs
        Harbor); re-priming with the same task list is a no-op.
        """
        normalized = tuple(sorted(set(task_ids)))
        with self._lock:
            if self._primed_for == normalized and self._eval_result is not None:
                return
            logger.info(
                "harbor batch prime: cfg=%s cwd=%s n_tasks=%d label=%s",
                self.invocation.config_path,
                self.invocation.cwd,
                len(normalized),
                self.invocation.subset_label,
            )
            self._eval_result = eval_runner.run_full(
                config_path=self.invocation.config_path,
                cwd=self.invocation.cwd,
                subset=self.invocation.subset_label,
                task_names=list(normalized),
                extra_env=self.invocation.extra_env or None,
            )
            self._primed_for = normalized

    # ─── Per-task lookup helpers ──────────────────────────────────────

    def get_per_task(self, task_id: str) -> "PerTaskOutcome":
        """Return the cached (reward, passed) for one task.

        Raises ``LookupError`` if the runner hasn't been primed.
        Returns a "missing" outcome (passed=False, reward=0.0, found=False)
        if the batch ran but Harbor didn't produce a result for that task
        (typically because the task name is not in the configured dataset
        — the caller should treat this as an eval error, not a fail).
        """
        if self._eval_result is None:
            raise LookupError(
                "BatchHarborRunner.get_per_task called before prime()"
            )
        rewards = self._eval_result.rewards_per_task or {}
        passing = set(self._eval_result.passing_tasks or [])
        # Harbor records trial names like "<task>__<6charhash>"; the
        # rewards_per_task and passing_tasks lists also use trial names.
        # Match by prefix to recover the underlying task_id.
        matched_reward = None
        matched_passed = False
        for k, v in rewards.items():
            if k == task_id or k.startswith(f"{task_id}__"):
                matched_reward = float(v)
                break
        for k in passing:
            if k == task_id or k.startswith(f"{task_id}__"):
                matched_passed = True
                break
        if matched_reward is None:
            return PerTaskOutcome(
                task_id=task_id, passed=False, reward=0.0, found=False
            )
        return PerTaskOutcome(
            task_id=task_id,
            passed=matched_passed,
            reward=matched_reward,
            found=True,
        )

    @property
    def job_dir(self) -> Path | None:
        return self._eval_result.job_dir if self._eval_result else None


@dataclass(frozen=True)
class PerTaskOutcome:
    """A per-task slice of a batched ``EvalResult``."""

    task_id: str
    passed: bool
    reward: float
    found: bool
    """False iff Harbor's result.json didn't include this task at all.
    Helpful for detecting config / task-name mismatches separately from
    a real fail (reward=0)."""

    def __post_init__(self) -> None:
        if not isinstance(self.reward, float):
            object.__setattr__(self, "reward", float(self.reward))
