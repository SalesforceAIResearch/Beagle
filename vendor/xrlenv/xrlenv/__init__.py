"""XRLEnv — agentic RL sandbox infrastructure.

The package is split along three planes (spec 00):

- ``xrlenv.client`` — consumer-facing SDK (`Client`, `RolloutSession`).
- ``xrlenv.control`` — orchestrator (state, scheduler, coordinator).
- ``xrlenv.node`` and ``xrlenv.backends`` — data plane (node agent + backends).
- ``xrlenv.sandbox_stub`` — process running *inside* each sandbox.
- ``xrlenv.envs`` — built-in EnvAdapters (spec 14).

The top-level package keeps imports cheap so the in-sandbox stub (which only
needs :mod:`xrlenv.sandbox_stub`, :mod:`xrlenv.envs`, and :mod:`xrlenv.types`)
does not transitively pull in :mod:`docker` or other host-only deps. Consumer-
facing classes are exposed lazily through ``__getattr__`` for readable imports
like ``from xrlenv import Client``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Operator-UX (2026-05-11): when an importer's CWD (or any parent
# dir) holds a ``.env`` file, populate ``os.environ`` from it before
# any consumer-facing code reads env vars. Existing shell-exported
# values win — ``.env`` is the fallback layer, not an override. Set
# ``XRLENV_DOTENV=off`` in the shell to opt out. Safe + silent on
# failure; never raises from the import path.
#
# We import from the private top-level module — NOT from
# ``xrlenv.client.dotenv`` — to avoid triggering
# ``xrlenv.client.__init__`` (which imports Client, which fans out
# to docker / gRPC / prometheus). The in-sandbox stub's slim image
# explicitly avoids those deps (test_import_cycles.py).
from xrlenv._dotenv_autoload import _maybe_auto_load_dotenv
from xrlenv._version import __version__

_maybe_auto_load_dotenv()

from xrlenv.types import (  # noqa: E402 — must run AFTER the .env auto-load above
    BatchRolloutResult,
    CancelGroupReport,
    Deadline,
    FailedRollout,
    RolloutStatus,
    Step,
    StepResult,
    Trajectory,
)

if TYPE_CHECKING:
    from xrlenv.client import Client, Template
    from xrlenv.compat.docker_client import from_env
    from xrlenv.compat.metadata import rollout_metadata

_LAZY_EXPORTS: dict[str, str] = {
    "Client": "xrlenv.client",
    "Template": "xrlenv.client",
    # docker-py drop-in entry point. Lazy so importing xrlenv stays
    # cheap even when the consumer doesn't use the compat shim, and
    # so the in-sandbox stub doesn't transitively load docker-py.
    "from_env": "xrlenv.compat.docker_client",
    # P1.7.B.3: per-rollout metadata contextvar (smoke drivers wrap
    # their per-instance run_instance call in
    # ``with xrlenv.rollout_metadata(...):`` to set artifact_path /
    # displayed_name on the cluster's RawRolloutRecord).
    "rollout_metadata": "xrlenv.compat.metadata",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'xrlenv' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)


__all__ = [
    "BatchRolloutResult",
    "CancelGroupReport",
    "Client",
    "Deadline",
    "FailedRollout",
    "RolloutStatus",
    "Step",
    "StepResult",
    "Template",
    "Trajectory",
    "__version__",
    "from_env",
    "rollout_metadata",
]
