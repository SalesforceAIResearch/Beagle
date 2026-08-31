"""Runtime selection: config → a concrete :class:`ContainerRuntime`.

The one bit of new glue over the ported runtimes — picks ``local`` vs
``xrlenv-cluster`` from a run config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beagle.rollout.runtime.protocol import ContainerRuntime
from beagle.rollout.runtime.runtime import LocalDockerRuntime
from beagle.rollout.runtime.xrlenv_runtime import XrlenvDockerRuntime


@dataclass
class RuntimeConfig:
    """Selects and parameterizes the runtime."""

    kind: str = "local"  # "local" | "xrlenv-cluster"
    grpc_host: str | None = None
    grpc_port: int | None = None
    token: str | None = None
    run_id: str | None = None
    artifact_root: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)


def build_runtime(config: RuntimeConfig) -> ContainerRuntime:
    """Instantiate the runtime named by ``config.kind``."""
    if config.kind == "local":
        return LocalDockerRuntime()
    if config.kind == "xrlenv-cluster":
        return XrlenvDockerRuntime(
            grpc_host=config.grpc_host,
            grpc_port=config.grpc_port,
            consumer_token=config.token,
            run_id=config.run_id,
        )
    raise ValueError(f"unknown runtime kind {config.kind!r}")


__all__ = ["RuntimeConfig", "build_runtime"]
