"""Concrete Harbor-backed adapters for Atelier's runner protocols.

``atelier`` defines protocols (``HoneypotRunner``, ``TransferEvaluator``,
``FitnessRunner``) without committing to a particular eval substrate.
This subpackage provides the production wiring: adapters that run their
evaluations via ``self_evolve.eval_runner.run_full`` (i.e. Harbor) so
the rest of the campaign uses the same trial infrastructure.

Modules:

- ``batch_harbor`` — internal helper that runs Harbor once for a batch
  of task names and caches per-task results. Both honeypot and transfer
  use this so we don't pay Harbor's startup cost N times per layer.
- ``honeypot_harbor`` — ``HarborHoneypotRunner`` implementing
  ``HoneypotRunner`` against a configurable Harbor config (the TW
  honeypot config).
- ``transfer_harbor`` — ``HarborTransferEvaluator`` implementing
  ``TransferEvaluator`` against a configurable Harbor config (different
  model for cross-model, different dataset for cross-benchmark).

All adapters take the ``cwd`` (a per-pipeline worktree) so they can be
invoked from inside ``self_evolve``'s pipeline at the same point the
main eval is run.
"""

from __future__ import annotations

from .batch_harbor import BatchHarborRunner, HarborInvocation
from .honeypot_harbor import HarborHoneypotRunner
from .transfer_harbor import HarborTransferEvaluator

__all__ = [
    "HarborInvocation",
    "BatchHarborRunner",
    "HarborHoneypotRunner",
    "HarborTransferEvaluator",
]
