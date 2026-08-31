"""Container runtime: thin sync API for acquiring, exec-ing, and tearing
down per-task containers.

The :class:`ContainerRuntime` Protocol pins the contract every adapter
talks to (``acquire`` / ``exec`` / ``destroy``, opaque handle, timeout→124,
idempotent destroy, thread-safe, silent acceptance of unused kwargs). Two
implementations satisfy it:

* :class:`LocalDockerRuntime` — shells out to the local ``docker`` CLI.
* :class:`XrlenvDockerRuntime` — routes to the xrlenv cluster via
  ``xrlenv.from_env()`` (xrlenv imported lazily).

:func:`build_runtime` picks one from a :class:`RuntimeConfig`.
"""
from __future__ import annotations

from beagle.rollout.runtime.config import RuntimeConfig, build_runtime
from beagle.rollout.runtime.protocol import ContainerRuntime, Handle
from beagle.rollout.runtime.runtime import (
    ContainerHandle,
    ContainerResources,
    ExecResult,
    LocalDockerRuntime,
)
from beagle.rollout.runtime.transport import (
    BindMount,
    GitClone,
    Transport,
    clone_with_retry,
    git_clone_argv,
)
from beagle.rollout.runtime.xrlenv_runtime import XrlenvDockerRuntime, acquire_labels

__all__ = [
    # protocol
    "ContainerRuntime",
    "Handle",
    # local + cluster impls
    "LocalDockerRuntime",
    "XrlenvDockerRuntime",
    "acquire_labels",
    # data types
    "ContainerHandle",
    "ContainerResources",
    "ExecResult",
    # transport
    "BindMount",
    "GitClone",
    "Transport",
    "git_clone_argv",
    "clone_with_retry",
    # selection glue
    "RuntimeConfig",
    "build_runtime",
]
