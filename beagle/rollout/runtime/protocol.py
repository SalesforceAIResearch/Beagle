"""ContainerRuntime Protocol — the contract every runtime impl honors.

``Handle`` is intentionally ``Any``: adapters treat it as an opaque token,
so the local impl's :class:`ContainerHandle` and a future cloud impl's
session-id-or-whatever both fit without a common base class.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from beagle.rollout.runtime.runtime import ContainerResources, ExecResult
from beagle.rollout.runtime.transport import BindMount

# Opaque token returned by acquire() and passed back to exec()/destroy().
# Adapters MUST NOT read fields off the handle; cloud impls may use a
# session id, a node id, or any other shape. The Protocol can't express
# "opaque" in the type system, so we lean on ``Any``.
Handle = Any


@runtime_checkable
class ContainerRuntime(Protocol):
    """Sync container lifecycle API. Three calls, opaque handles.

    Contract every implementation must honor: a timeout is reported as
    ``ExecResult(returncode=124, ...)`` (never raised); ``destroy`` is
    idempotent and swallows errors; the handle is opaque; instances are
    thread-safe; and any kwarg an impl doesn't use is silently accepted.
    """

    def acquire(
        self,
        *,
        image: str,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        mounts: list[BindMount] | None = None,
        workspace_dir: str | None = None,
        platform: str | None = None,
        run_args: list[str] | None = None,
        resources: ContainerResources | None = None,
        acquire_timeout: float = ...,
    ) -> Handle:
        """Start a container/sandbox. Return an opaque handle.

        Cloud impls MUST reject ``mounts`` (BindMount is local-only) at
        ``acquire`` time with a clear error pointing operators to a
        ``GitClone`` transport. ``resources`` is the benchmark's declared
        per-task cap; every impl honors it as a hard cap when set, ``None``
        means no explicit cap.
        """
        ...

    def exec(
        self,
        handle: Handle,
        command: list[str],
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecResult:
        """Run ``command`` inside the acquired container.

        On timeout, return ``ExecResult(returncode=124, ...)`` rather than
        raise. Adapters branch on rc==124 to produce ``timeout after Ns``.
        """
        ...

    def destroy(self, handle: Handle) -> None:
        """Tear down. Idempotent. Errors are swallowed."""
        ...


__all__ = ["ContainerRuntime", "Handle"]
