"""Harbor-backed implementation of ``atelier.transfer.TransferEvaluator``.

Same batched-Harbor architecture as ``honeypot_harbor.HarborHoneypotRunner``.
The two substrates differ only in:

- **cross-model**: same Harbor config as the main TB-2 eval, but with
  the model overridden via ``extra_env`` (typically
  ``MONET_EVAL_MONET_MODEL`` or by pointing at a sibling config).
- **cross-benchmark**: different Harbor config pointing at a different
  dataset (e.g. a SWE-bench-lite slice config).

This module knows neither — both are encoded in the ``HarborInvocation``
the caller constructs.

The mapping back to ``TransferTaskResult`` (rather than
``honeypot.TaskResult``) is the only meaningful divergence from the
honeypot adapter, so we keep them as separate small modules.
"""

from __future__ import annotations

from atelier import transfer

from .batch_harbor import BatchHarborRunner


class HarborTransferEvaluator:
    """Per-task evaluator that pulls from a batched Harbor invocation.

    Implements ``atelier.transfer.TransferEvaluator`` (a Protocol).
    Like the honeypot runner, we require ``task_ids`` upfront so the
    batch invocation knows what to include.
    """

    def __init__(
        self,
        *,
        batch: BatchHarborRunner,
        task_ids: tuple[str, ...],
    ) -> None:
        if not task_ids:
            raise ValueError(
                "HarborTransferEvaluator requires a non-empty task_ids tuple"
            )
        self._batch = batch
        self._task_ids = task_ids

    def __call__(
        self, *, candidate_id: str, task_id: str
    ) -> transfer.TransferTaskResult:
        if task_id not in self._task_ids:
            raise ValueError(
                f"task_id {task_id!r} not in pre-committed task_ids; "
                "HarborTransferEvaluator needs all task_ids upfront for batching"
            )
        if not self._batch.is_primed:
            self._batch.prime(task_ids=self._task_ids)
        outcome = self._batch.get_per_task(task_id)
        return transfer.TransferTaskResult(
            task_id=task_id,
            passed=outcome.passed,
            reward=outcome.reward,
        )
