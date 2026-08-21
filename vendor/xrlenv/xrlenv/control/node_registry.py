"""NodeRegistry + heartbeat watchdog (spec 03 / spec 04).

Tracks the set of currently-connected nodes (their :class:`NodeTransport`
plus the per-node ``last_heartbeat_at`` timestamp the gRPC servicer maintains
on the :class:`RemoteNodeTransport`). A background watchdog task wakes every
``check_interval_s`` and marks any node whose heartbeat is older than
``disconnect_grace_s`` as lost — calling out to a registered handler so the
coordinator can seal that node's rollouts as ``failed`` /
``reason=node_lost`` (spec 02 reason table).

Slice 3.5 keeps state in-memory; spec 20's ``nodes`` table will get the
persistence path in Slice 4 alongside the admin panel's "last seen" column.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from xrlenv.control.state import StateStore

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Heartbeat-bearing transport surface
# ──────────────────────────────────────────────────────────────────────────────


class _Heartbeating(Protocol):
    """Subset the watchdog needs from a transport.

    :class:`RemoteNodeTransport` exposes both fields. The in-process
    :class:`NodeAgent` does not (it cannot drop off the network), so the
    registry skips it — hardcoded ``last_heartbeat_at = +inf`` in
    :py:meth:`NodeRegistry.register`.
    """

    @property
    def node_id(self) -> str: ...
    @property
    def last_heartbeat_at(self) -> float: ...


class _PersistableTransport(_Heartbeating, Protocol):
    """Optional fields the registry mirrors to state when present.

    Real :class:`RemoteNodeTransport` exposes ``supported_backends()``
    (a method, not an attribute) plus ``stream_epoch`` /
    ``control_instance_id``. Test fakes that only implement the
    heartbeat surface are still accepted; the ``_extract_backends``
    helper + ``getattr`` fallbacks below tolerate them.

    Operator-reported regression (2026-05-04): an earlier version
    of this file declared ``backends`` as a property on this
    Protocol and the registry called ``getattr(transport, "backends",
    [])``. RemoteNodeTransport doesn't have that attribute (it
    stores ``_backends`` privately + exposes a
    ``supported_backends()`` method), so every gRPC-attached node
    landed in state with ``backends=[]`` and the SDK's
    ``wait_for_nodes(backend="docker")`` filter rejected them all
    with the misleading "saw N total registry rows" timeout.
    """

    def supported_backends(self) -> list[str]: ...
    @property
    def stream_epoch(self) -> str: ...
    @property
    def control_instance_id(self) -> str: ...


def _extract_backends(transport: object) -> list[str]:
    """Pull the supported-backends list off any transport shape.

    Tries ``supported_backends()`` (the canonical interface
    :class:`xrlenv.node.agent.NodeAgent` and
    :class:`xrlenv.control.grpc_endpoint.RemoteNodeTransport` both
    implement), then falls back to a ``backends`` attribute (any
    future / test transport that exposes a list directly), then
    returns ``[]``. The fallback chain matters because the watchdog
    surface only requires ``node_id`` + ``last_heartbeat_at``;
    test fakes that minimally implement the watchdog Protocol
    won't carry backends at all.
    """
    method = getattr(transport, "supported_backends", None)
    if callable(method):
        try:
            return list(method() or [])
        except Exception:
            pass
    attr = getattr(transport, "backends", None)
    if isinstance(attr, list):
        return [str(b) for b in attr]
    return []


def _extract_isolation_capable(transport: object) -> bool:
    """P6 step-2c — the node's advertised CPU-isolation capability, off any
    transport shape. Defensive: a transport predating ``isolation_capable()``
    (a minimal watchdog test fake) → ``False``."""
    method = getattr(transport, "isolation_capable", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass
    return False


def _extract_pinned_cpu_state(transport: object) -> tuple[int, int]:
    """P6 step-2c — ``(free, total)`` pinnable-CPU counts off any transport
    shape; ``(0, 0)`` ("unknown") for a transport predating
    ``pinned_cpu_state()``."""
    method = getattr(transport, "pinned_cpu_state", None)
    if callable(method):
        try:
            result = method()
            if isinstance(result, tuple) and len(result) == 2:
                return (int(result[0]), int(result[1]))
        except Exception:
            pass
    return (0, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────


def _accepts_transport_kwarg(fn: Callable[..., Awaitable[None]]) -> bool:
    """True iff ``fn`` accepts a ``transport=`` keyword — so the watchdog can hand a
    transport-aware loss handler the specific lost stream (H11) while still supporting a legacy
    1-arg handler. Detected precisely (a ``transport`` param, or ``**kwargs``) and always PASSED
    BY KEYWORD, so it works whether the param is positional-or-keyword or keyword-only. Robust to
    bound methods."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == "transport" or p.kind == p.VAR_KEYWORD
        for p in params
    )


NodeLostCallback = Callable[..., Awaitable[None]]
"""``async def on_node_lost(node_id: str, transport=None) -> None`` — called from the
watchdog task when a node has missed too many heartbeats. The handler typically seals that
node's in-flight rollouts as ``failed`` / ``reason=node_lost``. A handler that also accepts a
second ``transport`` arg gets the specific lost stream, so its teardown can be
stream-generation-scoped (audit H11); a legacy 1-arg handler is called with the id only.
"""


class NodeRegistry:
    """In-memory registry of connected nodes + heartbeat watchdog.

    Lifecycle:
        registry = NodeRegistry(on_node_lost=callback)
        registry.register(transport)        # called from servicer's on_connected
        await registry.start()              # spins the watchdog task
        ...
        registry.deregister(node_id)        # called from servicer's on_disconnected
        await registry.shutdown()           # cancels the watchdog
    """

    def __init__(
        self,
        *,
        on_node_lost: NodeLostCallback,
        disconnect_grace_s: float = 60.0,
        check_interval_s: float = 5.0,
        state: StateStore | None = None,
    ) -> None:
        self._on_node_lost = on_node_lost
        # Does the loss handler accept a 2nd (transport) arg? (H11) — inspected once so the
        # watchdog can pass the lost stream to a transport-aware handler while still supporting
        # a legacy 1-arg handler. Robust to bound methods, *args, and **kwargs.
        self._loss_handler_wants_transport = _accepts_transport_kwarg(on_node_lost)
        self._grace = disconnect_grace_s
        self._interval = check_interval_s
        self._transports: dict[str, _Heartbeating] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Optional state-store mirror so out-of-process callers
        # (`xrlenv nodes`, future admin RPC clients) can see who's
        # currently attached without going through gRPC.
        self._state = state

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="node-registry-watchdog")

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ── Membership ───────────────────────────────────────────────────────────

    def register(self, transport: _Heartbeating) -> None:
        self._transports[transport.node_id] = transport
        LOGGER.info("node-registry: %s registered", transport.node_id)
        if self._state is not None:
            try:
                self._state.record_node_connected(
                    transport.node_id,
                    backends=_extract_backends(transport),
                    stream_epoch=getattr(transport, "stream_epoch", None),
                    instance_id=getattr(transport, "control_instance_id", None),
                    # P6 step-2c — persist the advertised isolation capability
                    # (static, from NodeHello) for operator observability.
                    isolation_capable=_extract_isolation_capable(transport),
                )
            except Exception:
                LOGGER.exception(
                    "node-registry: state mirror write failed for register node=%s",
                    transport.node_id,
                )
            # Install the per-heartbeat hook so the state-store's
            # ``nodes.last_seen_at`` tracks the live timestamp instead
            # of staying frozen at register-time. Out-of-process readers
            # (``xrlenv nodes``, the admin ``/nodes`` view) only see the
            # SQLite mirror, so without this their "last seen Xm ago"
            # column drifts unboundedly even on healthy nodes.
            setter = getattr(transport, "set_on_heartbeat", None)
            if setter is not None:
                setter(self._note_heartbeat)

    def deregister(self, node_id: str, expected: _Heartbeating | None = None) -> bool:
        """Deregister ``node_id``. Returns True iff a transport was removed.

        STREAM-GENERATION-SAFE (audit H11): when ``expected`` is given, only deregister if the
        CURRENTLY-registered transport IS that object — so a stale stream close / a watchdog
        sweep processing a snapshotted-then-reconnected id can't evict the REPLACEMENT
        generation. ``expected=None`` keeps the unconditional legacy behavior."""
        if expected is not None and self._transports.get(node_id) is not expected:
            return False
        transport = self._transports.pop(node_id, None)
        if transport is not None:
            LOGGER.info("node-registry: %s deregistered", node_id)
            # Drop the heartbeat callback so a stray late heartbeat
            # (rare; can happen during teardown) doesn't write to the
            # state store after the disconnect was recorded.
            setter = getattr(transport, "set_on_heartbeat", None)
            if setter is not None:
                setter(None)
            if self._state is not None:
                # Shutdown race: ``deregister`` runs from the gRPC
                # ``NodeControlStream`` ``finally`` block, which can
                # fire *after* the runtime's ``shutdown()`` already
                # closed the state store (``grpc_server.stop()``
                # returns before every stream coroutine has finished
                # unwinding). A write against the closed store would
                # raise ``sqlite3.ProgrammingError`` — expected, not
                # an error. Skip the mirror write quietly when the
                # store reports itself closed.
                if getattr(self._state, "is_closed", False):
                    LOGGER.debug(
                        "node-registry: state store closed; skipping "
                        "disconnect mirror for node=%s", node_id,
                    )
                else:
                    try:
                        self._state.record_node_disconnected(node_id)
                    except Exception:
                        LOGGER.exception(
                            "node-registry: state mirror write failed for "
                            "deregister node=%s", node_id,
                        )
            return True
        return False

    def _note_heartbeat(self, node_id: str) -> None:
        """Mirror a heartbeat into ``nodes.last_seen_at``. Called from
        each :class:`RemoteNodeTransport.touch` via the callback the
        registry installs at ``register`` time. Bounded SQLite write
        rate (one per heartbeat per node, typically every few seconds);
        WAL mode handles the load comfortably.
        """
        if self._state is None:
            return
        # Same shutdown race as ``deregister`` — a heartbeat can land
        # while the store is being torn down. Skip quietly if closed.
        if getattr(self._state, "is_closed", False):
            return
        try:
            self._state.update_node_seen(node_id, time.time())
        except Exception:
            LOGGER.exception(
                "node-registry: update_node_seen failed for node=%s", node_id,
            )
        # Stage-1 admission/capacity — mirror the per-node health stats
        # the transport stashed from the last heartbeat
        # (notes/admission-stage-1-observability.md). Best-effort: a
        # pre-Stage-1 node-agent reports no health (``health_json`` None).
        transport = self._transports.get(node_id)
        health_json = getattr(transport, "health_json", None)
        if health_json is not None:
            try:
                self._state.update_node_health(node_id, health_json)
            except Exception:
                LOGGER.exception(
                    "node-registry: update_node_health failed for node=%s",
                    node_id,
                )
        # P6 step-2c — piggyback the last-known pinnable-CPU counts on the same
        # per-heartbeat write (observability only; nothing schedules on them).
        if transport is not None:
            free, total = _extract_pinned_cpu_state(transport)
            try:
                self._state.update_node_pinned_cpus(node_id, free=free, total=total)
            except Exception:
                LOGGER.exception(
                    "node-registry: update_node_pinned_cpus failed for node=%s",
                    node_id,
                )

    def get(self, node_id: str) -> _Heartbeating | None:
        return self._transports.get(node_id)

    @property
    def node_ids(self) -> list[str]:
        return list(self._transports)

    # ── Watchdog ─────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                await self._sweep()
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
        except asyncio.CancelledError:
            return

    async def _sweep(self) -> None:
        now = time.monotonic()
        # Snapshot (node_id, transport) so a node that reconnects under the same id BETWEEN the
        # snapshot and processing isn't torn down by the stale generation (audit H11).
        lost: list[tuple[str, _Heartbeating]] = []
        for node_id, transport in list(self._transports.items()):
            if now - transport.last_heartbeat_at > self._grace:
                lost.append((node_id, transport))
        for node_id, transport in lost:
            # Only act if THIS transport is still the registered one — a replacement that
            # reconnected in the meantime must be left alone.
            if not self.deregister(node_id, expected=transport):
                LOGGER.info(
                    "node-registry: %s heartbeat-lost stream superseded by a reconnect; "
                    "skipping stale teardown (H11)", node_id,
                )
                continue
            LOGGER.warning(
                "node-registry: %s exceeded heartbeat grace (%.1fs); marking lost",
                node_id,
                self._grace,
            )
            try:
                if self._loss_handler_wants_transport:
                    await self._on_node_lost(node_id, transport=transport)
                else:
                    await self._on_node_lost(node_id)
            except Exception:
                LOGGER.exception("node-loss handler raised for node=%s", node_id)


__all__ = ["NodeLostCallback", "NodeRegistry"]
