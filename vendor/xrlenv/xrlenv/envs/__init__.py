"""EnvAdapter Protocol — the case-1 RL training wire contract.

This module is the **platform** half of the adapter system: the
:class:`EnvAdapter` Protocol (and its sync helper :class:`SyncEnvAdapter`)
plus :class:`AdapterCapabilities` / :class:`StepTimeout`.

Under the slim pivot, EnvAdapter is the **case-1 mechanism only** —
for RL training where the trainer drives an ``act → obs`` step
loop. Case-2/3 evaluation harnesses (SWE-bench, harbor) plug in
via the docker-py drop-in (:mod:`xrlenv.compat.docker_client`) or
per-framework adapters (:mod:`xrlenv_plugins.harbor`) instead.

Concrete case-1 adapters live elsewhere:

- Spine smoke: :mod:`xrlenv.templates.hello_shell.adapter`.
- Case-1 plug-ins: ``xrlenv_plugins/benchmarks/<name>/adapter.py``.
  Worked external pip-package example at
  ``examples/pip_new_datasets_or_benchmark/echo_bench/``.
"""

from xrlenv.envs.base import (
    AdapterCapabilities,
    EnvAdapter,
    StepTimeout,
    SyncEnvAdapter,
)

__all__ = [
    "AdapterCapabilities",
    "EnvAdapter",
    "StepTimeout",
    "SyncEnvAdapter",
]
