"""GC layer 3 — periodic control-plane reconcile via bidi (A3 / D15, P1.1).

Spec 09 lists five GC layers; layers 1, 2, 4 (deadline-watcher TTL,
node startup orphan sweep, run-dir rotation) shipped in slice 5b.
Layer 3 is the missing **control-plane-driven** reconcile: every
``interval_s`` (default 60 s), the coordinator asks each connected
node-agent for its current sandbox-ID set, diffs against
``state.list_sandboxes()`` filtered by node, and acts on each side's
orphans:

- **Node-only orphan**: the node still owns a sandbox the state store
  no longer tracks. Almost certainly a control-plane crash leftover
  that survived the node's startup sweep, or an out-of-band ``docker
  run`` by an operator. Issue ``destroy_sandbox`` and emit
  ``gc.reconcile.orphan_sandbox``.
- **State-only orphan**: the state store has a ``running`` sandbox
  attached to a rollout but the node's reply does not include it. The
  container died on the node (Docker crash, OOM-killed, manual
  ``docker rm``). Seal the owning rollout ``failed/sandbox_lost`` via
  the coordinator's :py:meth:`handle_sandbox_lost` and emit
  ``gc.reconcile.lost_sandbox``.

Disconnected nodes are skipped — :py:meth:`NodeRegistry.handle_node_lost`
already seals their in-flight rollouts as ``failed/node_lost``; the
reconciler runs only against the live set.

Lifecycle mirrors :class:`~xrlenv.control.idle_ttl.IdleTtlWatcher`:
``await reconciler.start()`` from runtime startup, ``await
reconciler.shutdown()`` from runtime shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from xrlenv.control.coordinator import RolloutCoordinator
    from xrlenv.control.node_registry import NodeRegistry
    from xrlenv.control.state import StateStore

LOGGER = logging.getLogger(__name__)


class _NodeLister(Protocol):
    """Subset of :class:`~xrlenv.control.node_transport.NodeTransport`
    the reconciler needs. Defined locally so tests can pass a minimal
    fake without dragging in the full Protocol.
    """

    @property
    def node_id(self) -> str: ...

    async def list_sandbox_ids(
        self, *, backend: str | None = None,
    ) -> list[str]: ...

    async def destroy_sandbox(self, sb: object) -> None: ...  # SandboxHandle


class GCReconciler:
    """Spec 09 GC layer 3 driver.

    Runs one background task that fans out :py:meth:`list_sandbox_ids`
    to every node in the registry on a fixed interval, diffs the
    union, and dispatches per-orphan handlers. Failures on one node
    do not stop the sweep on others — each node is tried independently
    inside a ``try/except`` so a single hung RPC can't pin the loop.
    """

    def __init__(
        self,
        *,
        registry: NodeRegistry,
        coordinator: RolloutCoordinator,
        state: StateStore,
        interval_s: float = 60.0,
        per_node_timeout_s: float = 30.0,
    ) -> None:
        self._registry = registry
        self._coordinator = coordinator
        self._state = state
        self._interval_s = interval_s
        self._per_node_timeout_s = per_node_timeout_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._loop(), name="gc-reconciler",
            )

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ── Internals ────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._interval_s,
                    )
                    return  # stop fired during sleep
                except TimeoutError:
                    pass
                try:
                    await self.reconcile_once()
                except Exception:
                    LOGGER.exception(
                        "gc-reconciler: sweep raised; will retry next interval",
                    )
        except asyncio.CancelledError:
            raise

    async def reconcile_once(self) -> dict[str, dict[str, int]]:
        """One sweep across all connected nodes. Returns a per-node
        report ``{node_id: {"node_only": N, "state_only": M,
        "destroy_pending_cleanup": K}}`` for observability; primarily
        exposed so tests can drive a single sweep without spinning the
        background task.

        Audit response (H1): ``destroy_pending`` rows are tracked
        separately from ``running`` rows. The coordinator's
        ``_terminate`` path leaves a row at ``status='destroy_pending'``
        when ``destroy_sandbox`` raises or times out; without dedicated
        handling here, the row would leak forever (the scheduler counts
        anything not ``destroyed`` against capacity). Two new actions
        cover the gap:

          - **node-still-has + state destroy_pending**: retry the
            destroy, drop the state row on success.
          - **node-no-longer-has + state destroy_pending**: destroy
            actually completed (timed out our way, succeeded the
            node's), drop the state row.
        """
        report: dict[str, dict[str, int]] = {}
        node_ids = list(self._registry.node_ids)
        for node_id in node_ids:
            transport = self._registry.get(node_id)
            if transport is None:
                continue  # raced disconnect
            try:
                node_ids_returned = await asyncio.wait_for(
                    transport.list_sandbox_ids(),  # type: ignore[attr-defined]
                    timeout=self._per_node_timeout_s,
                )
            except Exception:
                LOGGER.exception(
                    "gc-reconciler: list_sandbox_ids failed for node=%s; "
                    "skipping this node for this sweep",
                    node_id,
                )
                continue
            node_set = set(node_ids_returned)
            running_set: set[str] = set()
            destroy_pending_set: set[str] = set()
            for sb in self._state.list_sandboxes():
                if sb.node_id != node_id:
                    continue
                if sb.status == "running":
                    running_set.add(sb.sandbox_id)
                elif sb.status == "destroy_pending":
                    destroy_pending_set.add(sb.sandbox_id)

            # Genuine orphan: node has a sandbox the state doesn't
            # know about in either bucket. Likely a CP-crash leftover
            # the node-side GC layer 2 startup sweep didn't catch, or
            # an out-of-band ``docker run``.
            node_only = node_set - running_set - destroy_pending_set
            # Sandbox died on a still-healthy node — seal the rollout.
            state_only_running = running_set - node_set
            # Audit H1: destroy_pending row whose node still has the
            # sandbox — the prior _terminate destroy timed out; retry.
            destroy_pending_node_present = node_set & destroy_pending_set
            # Audit H1: destroy_pending row whose node no longer has
            # the sandbox — the prior destroy actually succeeded after
            # the CP timed out; drop the leaked row so the scheduler
            # stops counting it against capacity.
            destroy_pending_node_absent = destroy_pending_set - node_set

            report[node_id] = {
                "node_only": len(node_only),
                "state_only": len(state_only_running),
                "destroy_pending_cleanup": (
                    len(destroy_pending_node_present)
                    + len(destroy_pending_node_absent)
                ),
            }

            for sandbox_id in node_only:
                await self._handle_node_only(node_id, sandbox_id, transport)
            for sandbox_id in destroy_pending_node_present:
                await self._handle_destroy_pending_node_present(
                    node_id, sandbox_id, transport,
                )
            for sandbox_id in state_only_running:
                await self._handle_state_only(node_id, sandbox_id)
            for sandbox_id in destroy_pending_node_absent:
                await self._handle_destroy_pending_node_absent(
                    node_id, sandbox_id,
                )
        return report

    async def _handle_node_only(
        self, node_id: str, sandbox_id: str, transport: Any,
    ) -> None:
        """Node knows about a sandbox the state store doesn't — issue
        a ``destroy_sandbox`` on the node. Failures are logged but
        do not stop the sweep.

        We don't have a full ``SandboxHandle`` to pass — only the ID.
        Construct a minimal handle from what we know; the backend
        will look up the rest by ID. Backend ``backend_ref`` and
        ``stub_endpoint`` are placeholders since destroy only
        consults the backend's container/sandbox table by ID.
        """
        from xrlenv.backends.base import SandboxHandle

        self._state.append_event(
            "gc.reconcile.orphan_sandbox",
            payload={"node_id": node_id, "sandbox_id": sandbox_id},
        )
        LOGGER.warning(
            "gc-reconciler: node-only orphan node=%s sandbox=%s — destroying",
            node_id, sandbox_id,
        )
        # Phase-0 nodes have one backend; the destroy lookup keys on
        # sandbox_id alone. Pass an empty stub_endpoint — destroy
        # doesn't open the stub session.
        synthetic_handle = SandboxHandle(
            id=sandbox_id, backend="docker", backend_ref="",
            stub_endpoint="tcp://127.0.0.1:0",
        )
        with suppress(Exception):
            await transport.destroy_sandbox(synthetic_handle)

    async def _handle_destroy_pending_node_present(
        self, node_id: str, sandbox_id: str, transport: Any,
    ) -> None:
        """Audit H1: state has the sandbox at ``destroy_pending`` and
        the node still has it — the prior ``_terminate`` destroy
        timed out / raised. Retry the destroy; drop the state row on
        success so the scheduler stops counting it against capacity.
        Keep the row if destroy fails again — next sweep retries.
        """
        from xrlenv.backends.base import SandboxHandle

        self._state.append_event(
            "gc.reconcile.destroy_pending_retry",
            payload={"node_id": node_id, "sandbox_id": sandbox_id},
        )
        LOGGER.warning(
            "gc-reconciler: destroy_pending node=%s sandbox=%s — retrying destroy",
            node_id, sandbox_id,
        )
        synthetic_handle = SandboxHandle(
            id=sandbox_id, backend="docker", backend_ref="",
            stub_endpoint="tcp://127.0.0.1:0",
        )
        destroy_succeeded = False
        try:
            await transport.destroy_sandbox(synthetic_handle)
            destroy_succeeded = True
        except Exception:
            LOGGER.exception(
                "gc-reconciler: retry destroy_sandbox failed for "
                "node=%s sandbox=%s; row stays destroy_pending for next sweep",
                node_id, sandbox_id,
            )
        if destroy_succeeded:
            with suppress(KeyError):
                self._state.update_sandbox(sandbox_id, status="destroyed")
                self._state.remove_sandbox(sandbox_id)

    async def _handle_destroy_pending_node_absent(
        self, node_id: str, sandbox_id: str,
    ) -> None:
        """Audit H1: state has the sandbox at ``destroy_pending`` but
        the node's reply doesn't include the ID — the destroy
        actually completed even though the CP timed out / errored.
        Drop the state row so the scheduler stops counting it against
        capacity.
        """
        self._state.append_event(
            "gc.reconcile.destroy_pending_cleared",
            payload={"node_id": node_id, "sandbox_id": sandbox_id},
        )
        LOGGER.info(
            "gc-reconciler: destroy_pending cleared node=%s sandbox=%s "
            "(node-side destroy completed after CP timeout)",
            node_id, sandbox_id,
        )
        with suppress(KeyError):
            self._state.update_sandbox(sandbox_id, status="destroyed")
            self._state.remove_sandbox(sandbox_id)

    async def _handle_state_only(
        self, node_id: str, sandbox_id: str,
    ) -> None:
        """State store has a ``running`` sandbox the node no longer
        knows about — seal the owning rollout ``failed/sandbox_lost``.
        """
        self._state.append_event(
            "gc.reconcile.lost_sandbox",
            payload={"node_id": node_id, "sandbox_id": sandbox_id},
        )
        LOGGER.warning(
            "gc-reconciler: state-only orphan node=%s sandbox=%s — sealing rollout",
            node_id, sandbox_id,
        )
        await self._coordinator.handle_sandbox_lost(
            node_id, sandbox_id, reason="sandbox_lost",
        )


__all__ = ["GCReconciler"]
