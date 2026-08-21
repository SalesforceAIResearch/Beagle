"""Rollout infra (design-plot "rollout infra" box).

The unified runner plus the xrlenv-backed container runtime and the agent<->harness
binding types. This is the layer that turns "respect the original harness" from a
principle into plumbing: rollouts flow through each benchmark's native harness via
xrlenv's drop-in substrate.

Public surface::

    from beagle.rollout import Runner, build_runtime, RuntimeConfig
    runtime = build_runtime(RuntimeConfig(kind="xrlenv-cluster"))
    result  = Runner(runtime, parallelism=8).run(agent, dataset)
"""

from __future__ import annotations

from beagle.rollout.binding import GenericBinding, HarborBinding, RolloutBinding
from beagle.rollout.runner import Runner, RunResult
from beagle.rollout.runtime import (
    BindMount,
    ContainerHandle,
    ContainerResources,
    ContainerRuntime,
    ExecResult,
    GitClone,
    Handle,
    LocalDockerRuntime,
    RuntimeConfig,
    XrlenvDockerRuntime,
    acquire_labels,
    build_runtime,
    clone_with_retry,
    git_clone_argv,
)

__all__ = [
    # runner
    "Runner",
    "RunResult",
    # runtime
    "ContainerRuntime",
    "LocalDockerRuntime",
    "XrlenvDockerRuntime",
    "acquire_labels",
    "RuntimeConfig",
    "build_runtime",
    "Handle",
    "ContainerHandle",
    "ContainerResources",
    "ExecResult",
    # transport
    "BindMount",
    "GitClone",
    "git_clone_argv",
    "clone_with_retry",
    # bindings
    "RolloutBinding",
    "GenericBinding",
    "HarborBinding",
]
