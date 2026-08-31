"""``beagle.algorithms.darwinx`` — the DarwinX self-evolution algorithm integration.

Everything DarwinX-*specific* lives here, grouped by algorithm (maintenance/readability): the
**eval adapter** that backs its candidate evaluation with beagle's general
:func:`beagle.eval.evaluate` (:mod:`beagle.algorithms.darwinx.eval`), and — as the drop-in
lands — the vendored algorithm, the ``meta_agent``→``Editor`` evolver shim, and the Trainer
wiring (see ``notes/darwinx-dropin-contract.md``).

General, algorithm-agnostic seams (``evaluate``, editors, worktree) do NOT live here — they're
framework surface (:mod:`beagle.eval`, :mod:`beagle.agents`, …) that any algorithm reuses.
"""

from __future__ import annotations

from beagle.algorithms.darwinx.algorithm import DarwinX
from beagle.algorithms.darwinx.config import DarwinXConfig
from beagle.algorithms.darwinx.eval import (
    run_eval,
    to_darwinx_run_json,
    translate_config,
    write_darwinx_run_json,
)

__all__ = [
    "DarwinX",
    "DarwinXConfig",
    "run_eval",
    "to_darwinx_run_json",
    "translate_config",
    "write_darwinx_run_json",
]
