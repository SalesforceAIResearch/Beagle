"""Sandbox runtime adapters (spec 01).

Phase 0 ships ``docker`` and ``local-process-debug``. Phase 1 adds
``cubesandbox``. Each backend implements :class:`SandboxBackend` and advertises
its :class:`SandboxCapabilities` so the scheduler can refuse placements that
would violate template requirements.
"""

from xrlenv.backends.base import (
    ExecChunk,
    MountSpec,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxBackend,
    SandboxCapabilities,
    SandboxHandle,
    ServiceSpec,
    SnapshotID,
    TemplateRef,
)

__all__ = [
    "ExecChunk",
    "MountSpec",
    "NetworkPolicy",
    "ResourceSpec",
    "ResourceUsage",
    "SandboxBackend",
    "SandboxCapabilities",
    "SandboxHandle",
    "ServiceSpec",
    "SnapshotID",
    "TemplateRef",
]
