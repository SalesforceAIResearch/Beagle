"""P1.7.A.1 — Consumer-side ergonomic wrapper for raw container sessions.

The internal ``RawContainerCoordinator`` (control-plane side) and
the proto-level RPCs (``AcquireContainer`` / ``ContainerExec`` /
``DestroyContainer``) work fine on their own, but case-2/3 harnesses
calling them through ``Client`` need a more ergonomic surface:

- The session ties its own ``rollout_id`` + ``container_id`` so
  callers don't have to keep them in scope.
- ``__aenter__`` / ``__aexit__`` guarantee destroy on context exit
  even if the harness raises mid-evaluation.
- ``exec`` returns the raw bytes (stdout/stderr) so the harness can
  pipe them into its own logs / artifact archives without forcing
  a UTF-8 decode.

Use as an async context manager::

    async with await client.acquire_container(image="busybox:1") as session:
        result = await session.exec(["echo", "hi"], timeout_s=5.0)
        assert result.stdout == b"hi\\n"
    # destroy fires automatically on context exit

Or imperatively::

    session = await client.acquire_container(image="busybox:1")
    try:
        await session.exec(...)
    finally:
        await session.destroy()
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import TYPE_CHECKING

from xrlenv.backends.egress import EgressAllowlist
from xrlenv.control.raw_container_service import RawComposeAcquireResult
from xrlenv.control.service import RawAcquireResult, RawExecChunk, RawExecResult

if TYPE_CHECKING:
    from xrlenv.client.transport import ClientTransport

LOGGER = logging.getLogger(__name__)


class ClusterContainerSession:
    """Per-rollout handle around a remote raw container.

    Construction is via :meth:`Client.acquire_container` — direct
    instantiation is internal-only and not part of the public
    surface (the constructor takes the raw transport + the
    AcquireContainer response, which the SDK fills in).
    """

    def __init__(
        self,
        transport: ClientTransport,
        acquire_result: RawAcquireResult,
        *,
        on_destroy: Callable[[str], None] | None = None,
    ) -> None:
        self._transport = transport
        self._rollout_id = acquire_result.rollout_id
        self._container_id = acquire_result.container_id
        self._container_name = acquire_result.container_name
        self._node_id = acquire_result.node_id
        self._queue_wait_s = acquire_result.queue_wait_s
        self._destroyed = False
        # Called with the rollout_id when the session is destroyed, so the
        # Client's keepalive stops beating this (now-gone) session.
        self._on_destroy = on_destroy

    # ── Read-only attributes ──────────────────────────────────────────────

    @property
    def rollout_id(self) -> str:
        return self._rollout_id

    @property
    def container_id(self) -> str:
        return self._container_id

    @property
    def container_name(self) -> str:
        return self._container_name

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def queue_wait_s(self) -> float:
        """Issue #18 (Ask #1) — seconds this acquire spent waiting in
        the control-plane admission queue before a node was assigned.
        ``0.0`` on the fast path (capacity was available
        immediately). A consistently non-zero value across a batch
        means the requested concurrency exceeds cluster capacity."""
        return self._queue_wait_s

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    # ── Operations ────────────────────────────────────────────────────────

    async def exec(
        self,
        cmd: list[str],
        *,
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> RawExecResult:
        """Run ``cmd`` inside the remote container.

        Batched: returns full stdout/stderr after the command exits
        or the timeout fires (``timed_out=True``). Streaming exec
        (for swebench's 30+ min test runs) is queued for P1.7.A.2.

        Raises ``XRLEnvError`` if the session has been destroyed.
        """
        if self._destroyed:
            from xrlenv.errors import XRLEnvError
            raise XRLEnvError(
                f"ClusterContainerSession rollout={self._rollout_id!r} "
                f"already destroyed; cannot exec.",
            )
        return await self._transport.container_exec(
            rollout_id=self._rollout_id,
            container_id=self._container_id,
            cmd=cmd,
            timeout_s=timeout_s,
            cwd=cwd,
            env=env,
            user=user,
        )

    async def apply_egress(
        self,
        allowlist: EgressAllowlist,
        *,
        dns_resolver: str | None = None,
    ) -> None:
        """Restrict this container's egress to ``allowlist`` (spec 07).

        Generic mechanism: the caller decides what to allow. An empty
        ``allowlist`` blocks all external egress (loopback stays up). Rules
        are installed host-side via nsenter in the container's netns; the
        workload holds no CAP_NET_ADMIN so it can't remove them. Idempotent;
        node-side enforcement is fail-closed (a partial apply destroys the
        container). Raises ``XRLEnvError`` if the session was destroyed.
        """
        if self._destroyed:
            from xrlenv.errors import XRLEnvError
            raise XRLEnvError(
                f"ClusterContainerSession rollout={self._rollout_id!r} "
                f"already destroyed; cannot apply_egress.",
            )
        await self._transport.apply_egress(
            rollout_id=self._rollout_id,
            container_id=self._container_id,
            allowlist=allowlist,
            dns_resolver=dns_resolver,
        )

    def exec_stream(
        self,
        cmd: list[str],
        *,
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> AsyncIterator[RawExecChunk]:
        """Streaming-exec variant of :meth:`exec`. Returns an
        async iterator yielding ``RawExecChunk`` instances as the
        node produces output. The terminator chunk has
        ``done=True`` + the final ``exit_code``; consumers
        iterate with ``async for`` until they see it.

        Required for long-running runs (swebench's 30+ min eval
        scripts; tb2's 1-2 hour tasks) where batched exec would
        risk idle TCP drops along the consumer↔control-plane↔node
        path. The streaming wire keeps the path provably alive
        because bytes are flowing.

        Raises ``XRLEnvError`` immediately if the session has
        been destroyed.
        """
        if self._destroyed:
            from xrlenv.errors import XRLEnvError
            raise XRLEnvError(
                f"ClusterContainerSession rollout={self._rollout_id!r} "
                f"already destroyed; cannot exec_stream.",
            )
        # Type-cast: the ClientTransport Protocol declares the
        # return as ``Any`` to keep the lightweight typing the
        # other transport methods carry; here we narrow to the
        # SDK's typed iterator.
        result: AsyncIterator[RawExecChunk] = (
            self._transport.container_exec_stream(
                rollout_id=self._rollout_id,
                container_id=self._container_id,
                cmd=cmd,
                timeout_s=timeout_s,
                cwd=cwd,
                env=env,
                user=user,
            )
        )
        return result

    async def put_archive(
        self, target_dir: str, tarball: bytes,
    ) -> None:
        """Extract ``tarball`` into ``target_dir`` inside the
        container — symmetric with ``docker.container.put_archive``.

        ``tarball`` may be plain or gzipped tar. Caller builds it
        (e.g. via ``tarfile`` over an in-memory ``BytesIO``);
        size is bounded by the gRPC max-message-bytes setting
        (default 100 MiB; consumers needing larger transfers
        should chunk on their side, out of scope for P1.7.A.2).

        Raises ``XRLEnvError`` if the session has been destroyed.
        """
        if self._destroyed:
            from xrlenv.errors import XRLEnvError
            raise XRLEnvError(
                f"ClusterContainerSession rollout={self._rollout_id!r} "
                f"already destroyed; cannot put_archive.",
            )
        await self._transport.container_put_archive(
            rollout_id=self._rollout_id,
            container_id=self._container_id,
            target_dir=target_dir,
            tarball=tarball,
        )

    async def get_archive(self, source_path: str) -> bytes:
        """Tar up ``source_path`` inside the container and return
        the bytes — symmetric with ``docker.container.get_archive``.

        Returns the full tarball as a single bytes object. Same
        size bounds as :meth:`put_archive`.

        Raises ``XRLEnvError`` if the session has been destroyed.
        """
        if self._destroyed:
            from xrlenv.errors import XRLEnvError
            raise XRLEnvError(
                f"ClusterContainerSession rollout={self._rollout_id!r} "
                f"already destroyed; cannot get_archive.",
            )
        return await self._transport.container_get_archive(
            rollout_id=self._rollout_id,
            container_id=self._container_id,
            source_path=source_path,
        )

    async def destroy(self, *, force: bool = True) -> None:
        """Tear down the remote container + drop the control-plane
        session.

        Idempotent: a second ``destroy()`` call is a no-op so the
        ``__aexit__`` cleanup doesn't double-fire when the harness
        already destroyed manually.
        """
        if self._destroyed:
            return
        # Mark this CLIENT handle destroyed *before* the wire call so a transient network error
        # during destroy doesn't leave the handle in a "half-destroyed, half-usable" state.
        # NOTE (audit H8): a wire-level destroy FAILURE does NOT free capacity — the control
        # plane RETAINS the session + its capacity charge (invariant 2: released only on
        # node-confirmed destroy) and defers teardown to the raw-GC reconciler. This handle is
        # still marked done: the consumer's work is over and it does not retry; reconciliation
        # (deadline / liveness / confirmed-absence) completes the actual teardown.
        self._destroyed = True
        try:
            await self._transport.destroy_container(
                rollout_id=self._rollout_id,
                container_id=self._container_id,
                force=force,
            )
        finally:
            # Stop the local keepalive from beating this handle even if the wire destroy errored
            # (capacity stays charged control-plane-side until the reconciler confirms teardown).
            if self._on_destroy is not None:
                self._on_destroy(self._rollout_id)

    # ── Async context manager ─────────────────────────────────────────────

    async def __aenter__(self) -> ClusterContainerSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Best-effort destroy on exit — never swallows the original
        # exception, but logs if the destroy itself fails so a leaked
        # node-side container is visible to the operator.
        try:
            await self.destroy()
        except Exception as cleanup_exc:
            LOGGER.warning(
                "ClusterContainerSession.__aexit__ destroy failed "
                "rollout=%s container=%s err=%r",
                self._rollout_id, self._container_id[:12], cleanup_exc,
            )


class ClusterComposeSession(ClusterContainerSession):
    """Per-rollout handle around a remote multi-service compose PROJECT
    (P1.7.C.2).

    The compose analog of :class:`ClusterContainerSession`, constructed via
    :meth:`Client.acquire_compose_project`. It exposes the **same**
    ``exec`` / ``exec_stream`` / ``put_archive`` / ``get_archive`` /
    ``apply_egress`` surface — all bound to the project's ``main`` service
    container, so a harness that already drives a container session works
    unchanged against a compose project. Two things differ:

    * construction is from a :class:`RawComposeAcquireResult` (``main`` +
      the whole-project ``service_container_ids`` map), and
    * :meth:`destroy` tears down the **whole project** (``docker compose
      down``) via ``destroy_compose_project`` rather than removing a single
      container — so sidecars never leak.
    """

    def __init__(
        self,
        transport: ClientTransport,
        acquire_result: RawComposeAcquireResult,
        *,
        on_destroy: Callable[[str], None] | None = None,
    ) -> None:
        # The base class stores from a RawAcquireResult; the compose result has a
        # different shape (main_* + the project map), so set the fields directly
        # rather than routing through super().__init__.
        self._transport = transport
        self._rollout_id = acquire_result.rollout_id
        # exec/archive target the ``main`` service container (full docker id).
        self._container_id = acquire_result.main_container_id
        self._container_name = acquire_result.main_container_name
        self._node_id = acquire_result.node_id
        self._queue_wait_s = acquire_result.queue_wait_s
        self._project_name = acquire_result.project_name
        self._service_container_ids = dict(acquire_result.service_container_ids)
        self._destroyed = False
        self._on_destroy = on_destroy

    # ── Compose-specific read-only attributes ─────────────────────────────

    @property
    def project_name(self) -> str:
        """The docker-compose project name (the teardown handle)."""
        return self._project_name

    @property
    def service_container_ids(self) -> dict[str, str]:
        """``service -> full container id`` for every service in the project
        (including ``main``). A copy, so callers can't mutate internal state."""
        return dict(self._service_container_ids)

    async def apply_egress(
        self,
        allowlist: EgressAllowlist,
        *,
        dns_resolver: str | None = None,
    ) -> None:
        """Not supported for compose projects (yet) — raises.

        The inherited container-session ``apply_egress`` would install the
        allowlist on **only** ``main``'s netns, silently leaving every sidecar
        unrestricted — a caller could believe the whole project's egress was
        locked down when it wasn't. Per the design
        (``notes/multi-service-compose-plan.md`` §4.3), compose egress must be
        enforced at the **project-network** level (all services), which needs a
        compose-aware transport/RPC path that doesn't exist yet. Until it lands
        this fails loud rather than under-enforcing.
        """
        del allowlist, dns_resolver
        raise NotImplementedError(
            "apply_egress is not supported for a compose project "
            f"(rollout={self._rollout_id!r} project={self._project_name!r}): "
            "egress must be enforced at the project-network level across all "
            "services, but the container-session path would restrict only the "
            "`main` container and silently leave sidecars unrestricted. "
            "Project-network egress is a follow-on slice "
            "(notes/multi-service-compose-plan.md §4.3).",
        )

    async def destroy(self, *, force: bool = True) -> None:
        """Tear down the **whole compose project** (``docker compose down``)
        and drop the control-plane session.

        Unlike :meth:`ClusterContainerSession.destroy` — whose coordinator
        drops the session even when the wire call fails, so it marks destroyed
        *before* the call — compose teardown is **strict** server-side: a failed
        ``docker compose down`` RETAINS the project session + its capacity
        reservation (invariant 2) so the caller can retry. So we mark the
        session destroyed and stop the keepalive **only after a confirmed
        down**. On failure the exception propagates and the session stays usable
        for a retry *and keeps heartbeating*, so the control plane doesn't
        liveness-reap a project the caller is still tearing down.

        Idempotent after a successful down (a second call no-ops). ``force`` is
        accepted for signature-compatibility with the base session; compose
        ``down`` is always a full stop+remove.
        """
        if self._destroyed:
            return
        del force  # compose down is always a full stop+remove
        # Wire call FIRST — a raise here leaves _destroyed False + keepalive
        # registered (retryable), matching the server's strict-retain semantics.
        await self._transport.destroy_compose_project(
            rollout_id=self._rollout_id, project_name=self._project_name,
        )
        self._destroyed = True
        if self._on_destroy is not None:
            self._on_destroy(self._rollout_id)

    async def __aenter__(self) -> ClusterComposeSession:
        return self


__all__ = ["ClusterComposeSession", "ClusterContainerSession"]
