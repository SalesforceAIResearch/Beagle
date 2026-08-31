"""Control-plane → node keepalive loop (2026-08-21).

An idle control plane sends a node nothing after ControlHello, so a node has no
application-level way to distinguish a healthy-but-idle CP from one that has
silently dropped it — deregistered it (the heartbeat watchdog), or a half-open
stream whose HTTP/2 keepalive the CP still answers. This loop periodically
pushes an empty-body ``ControlMsg`` to every *registered* transport. A node
that stops receiving these beats — because its transport left the registry, or
its stream went half-open — redials and re-registers on its own (see the
node-side server-liveness watchdog in ``xrlenv/node/grpc_link.py``).

This is defense-in-depth behind the watchdog's active stream-abort
(:meth:`RemoteNodeTransport.request_terminate`): the abort recovers a lost node
in seconds; this keepalive-silence path recovers it within the node's silence
deadline even if the abort never reached it (e.g. the abort itself was blocked).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xrlenv.control.node_registry import NodeRegistry

LOGGER = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 5.0


class ControlKeepaliveLoop:
    """Periodically beats every registered node. Same start/shutdown shape as
    the other control-plane background loops."""

    def __init__(
        self,
        registry: NodeRegistry,
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ) -> None:
        self._registry = registry
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Schedule the keepalive task. Idempotent."""
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name="control-keepalive")

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._interval_s,
                    )
                if self._stop.is_set():
                    return
                self._beat_once()
        except asyncio.CancelledError:
            return

    def _beat_once(self) -> None:
        # Snapshot the ids so a concurrent (de)register during iteration is
        # tolerated. ``get`` returns the current transport or None (already gone).
        for node_id in self._registry.node_ids:
            transport = self._registry.get(node_id)
            # Duck-typed: the watchdog Protocol doesn't require ``send_keepalive``
            # (the in-process NodeAgent has no stream to beat), so tolerate its
            # absence rather than widen the Protocol.
            send = getattr(transport, "send_keepalive", None)
            if send is None:
                continue
            try:
                send()
            except Exception:
                LOGGER.debug(
                    "control-keepalive: send failed for node=%s", node_id,
                    exc_info=True,
                )


__all__ = ["ControlKeepaliveLoop"]
