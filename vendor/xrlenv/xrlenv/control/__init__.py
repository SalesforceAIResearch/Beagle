"""Control-plane components (spec 03).

Phase-0 control plane is a single Python process containing the StateStore,
TemplateCatalog, Scheduler, NodeRegistry, and RolloutCoordinator. Phase 1
will optionally run an additional warm-standby instance behind a leader lease.

Imports are lazy at this package level so a code path that only needs a
single submodule (``from xrlenv.control.instance_resolver import
VerifierUpload``) doesn't transitively load the coordinator → admission
→ observability → ``prometheus_client`` chain. The in-sandbox stub
imports the per-plug-in adapter, which imports
``xrlenv.control.instance_resolver`` to type its return value, but the
sandbox image doesn't have ``prometheus_client``. Eager re-exports here
were forcing the chain regardless. The ``__getattr__`` pattern below
mirrors the top-level ``xrlenv/__init__.py`` lazy-export shape.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xrlenv.control.coordinator import RolloutCoordinator
    from xrlenv.control.distributed_runtime import (
        DistributedRuntime,
        build_distributed_runtime,
    )
    from xrlenv.control.runtime import LocalRuntime, build_local_runtime
    from xrlenv.control.scheduler import Scheduler
    from xrlenv.control.service import (
        RolloutService,
        StartRolloutRequest,
        StartRolloutResponse,
    )
    from xrlenv.control.state import (
        InMemoryStateStore,
        NodeRecord,
        PendingRolloutRecord,
        RolloutRecord,
        SandboxRecord,
        SqliteStateStore,
        StateStore,
    )
    from xrlenv.control.template_catalog import TemplateCatalog, TemplateManifest

_LAZY_EXPORTS: dict[str, str] = {
    "DistributedRuntime": "xrlenv.control.distributed_runtime",
    "InMemoryStateStore": "xrlenv.control.state",
    "LocalRuntime": "xrlenv.control.runtime",
    "NodeRecord": "xrlenv.control.state",
    "PendingRolloutRecord": "xrlenv.control.state",
    "RolloutCoordinator": "xrlenv.control.coordinator",
    "RolloutRecord": "xrlenv.control.state",
    "RolloutService": "xrlenv.control.service",
    "SandboxRecord": "xrlenv.control.state",
    "Scheduler": "xrlenv.control.scheduler",
    "SqliteStateStore": "xrlenv.control.state",
    "StartRolloutRequest": "xrlenv.control.service",
    "StartRolloutResponse": "xrlenv.control.service",
    "StateStore": "xrlenv.control.state",
    "TemplateCatalog": "xrlenv.control.template_catalog",
    "TemplateManifest": "xrlenv.control.template_catalog",
    "build_distributed_runtime": "xrlenv.control.distributed_runtime",
    "build_local_runtime": "xrlenv.control.runtime",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'xrlenv.control' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)


__all__ = [
    "DistributedRuntime",
    "InMemoryStateStore",
    "LocalRuntime",
    "NodeRecord",
    "PendingRolloutRecord",
    "RolloutCoordinator",
    "RolloutRecord",
    "RolloutService",
    "SandboxRecord",
    "Scheduler",
    "SqliteStateStore",
    "StartRolloutRequest",
    "StartRolloutResponse",
    "StateStore",
    "TemplateCatalog",
    "TemplateManifest",
    "build_distributed_runtime",
    "build_local_runtime",
]
