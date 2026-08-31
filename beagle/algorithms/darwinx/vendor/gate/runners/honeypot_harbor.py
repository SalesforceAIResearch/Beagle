"""Harbor-backed implementation of ``atelier.honeypot.HoneypotRunner``.

Bridges the per-task ``HoneypotRunner`` protocol to a batched Harbor
invocation. Use one ``HarborHoneypotRunner`` per (candidate, eval) — it
batches Harbor once for all task_ids on first call and caches.

Typical use inside ``AtelierGate``:

```python
from gate.runners import (
    BatchHarborRunner, HarborHoneypotRunner, HarborInvocation,
)
from gate import honeypot

corpus = honeypot.HoneypotCorpus.from_repo("external/terminal-wrench")
task_ids = corpus.default_task_ids(max_count=20)

batch = BatchHarborRunner(invocation=HarborInvocation(
    config_path=Path("configs/terminal_wrench_honeypot.yaml"),
    cwd=worktree_path,
    subset_label="atelier-honeypot",
))
runner = HarborHoneypotRunner(batch=batch, task_ids=task_ids)
result = honeypot.score_candidate(
    candidate_id=node.id, task_ids=task_ids, runner=runner,
)
```

The TW honeypot Harbor config (``configs/terminal_wrench_honeypot.yaml``)
points at the locally-cloned TW corpus at ``external/terminal-wrench/``.
That config is added separately (see ``atelier/README.md``); this module
is config-agnostic so a future TW-2 corpus can swap in by changing only
the config_path.
"""

from __future__ import annotations

from gate import honeypot

from .batch_harbor import BatchHarborRunner


class HarborHoneypotRunner:
    """Per-task runner that pulls from a batched Harbor invocation.

    Implements ``atelier.honeypot.HoneypotRunner`` (a Protocol, so no
    explicit inheritance is needed; we still get type-check coverage).

    On first ``__call__`` we prime the batch with all task_ids the caller
    committed to up front. Subsequent calls hit the cache.
    """

    def __init__(
        self,
        *,
        batch: BatchHarborRunner,
        task_ids: tuple[str, ...],
    ) -> None:
        if not task_ids:
            raise ValueError("HarborHoneypotRunner requires a non-empty task_ids tuple")
        self._batch = batch
        self._task_ids = task_ids

    def __call__(
        self, *, candidate_id: str, task_id: str
    ) -> honeypot.TaskResult:
        if task_id not in self._task_ids:
            raise ValueError(
                f"task_id {task_id!r} not in pre-committed task_ids; "
                "HarborHoneypotRunner needs all task_ids upfront for batching"
            )
        if not self._batch.is_primed:
            self._batch.prime(task_ids=self._task_ids)

        outcome = self._batch.get_per_task(task_id)
        trial_dir = (
            str(self._batch.job_dir / "trials" / task_id)
            if self._batch.job_dir
            else None
        )
        return honeypot.TaskResult(
            task_id=task_id,
            passed=outcome.passed,
            reward=outcome.reward,
            trial_dir=trial_dir if outcome.found else None,
        )
