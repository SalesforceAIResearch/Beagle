"""Framework harness drivers + their installed-agent shims.

Groups what used to sit flat in ``benchmarks/``: the Job-driver harnesses (harbor's + its ``pier``
fork's), the raw-container/native harnesses (docker drop-in, native-runner), the ``HarborBenchmark``
zero-code base, and the M+N installed-agent **shims** — the shims (``_harbor_agent`` / ``_pier_agent``)
are loaded by harbor/pier via their ``import_path`` strings and are deliberately **not** imported
here, since importing one pulls in its (optional) framework.
"""

from __future__ import annotations

from beagle.benchmarks.harness.benchmark import HarborBenchmark
from beagle.benchmarks.harness.drivers import (
    DockerHarness,
    HarborHarness,
    NativeRunnerHarness,
    PierHarness,
)

__all__ = [
    "HarborHarness",
    "PierHarness",
    "DockerHarness",
    "NativeRunnerHarness",
    "HarborBenchmark",
]
