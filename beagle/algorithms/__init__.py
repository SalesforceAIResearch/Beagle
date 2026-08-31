"""The Evolve Algorithm Factory.

Public surface (mirrors ``bgl.agents.build`` — build algorithms by name)::

    import beagle as bgl
    algo = bgl.algorithms.build("darwinx", max_loop_iters=8)
    bgl.algorithms.available()   # -> ["darwinx"]

Every algorithm implements :meth:`~beagle.algorithms.base.EvolveAlgorithm.evolve` and declares
its typed knobs via an :class:`~beagle.algorithms.base.AlgorithmConfig` subclass on ``Config``.
Adding one is a one-file drop: subclass it, decorate with :func:`register`, place the file (or
package) under ``beagle/algorithms/`` — it is auto-discovered on import.
"""

from __future__ import annotations

import importlib
import pkgutil

from beagle.algorithms.base import AlgorithmConfig, Candidate, CandidateStatus, EvolveAlgorithm
from beagle.algorithms.registry import ALGORITHMS, available, build, register

_FRAMEWORK_MODULES = {"base", "registry"}


def _autodiscover() -> None:
    """Import every algorithm module/subpackage so its ``@register`` runs."""
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name in _FRAMEWORK_MODULES:
            continue
        importlib.import_module(f"{__name__}.{info.name}")


_autodiscover()

# Concrete algorithms are exposed by name via build(); import the common one for
# direct reference too.
from beagle.algorithms.darwinx import DarwinX, DarwinXConfig  # noqa: E402

__all__ = [
    "ALGORITHMS",
    "register",
    "build",
    "available",
    "EvolveAlgorithm",
    "AlgorithmConfig",
    "Candidate",
    "CandidateStatus",
    "DarwinX",
    "DarwinXConfig",
]
