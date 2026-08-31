"""Distributed bring-up: control plane runs a gRPC server, nodes connect out.

Slice 3 distributed runtime. Single-node-aware (the runtime exposes the first
node that connects); multi-node `NodeRegistry` lands in Slice 3.5+.

Usage::

    runtime = await build_distributed_runtime(grpc_port=50051)
    # ... separately, run `xrlenv-node serve --control-plane localhost:50051 --node-id foo`
    await runtime.wait_for_node()
    client = Client.in_process(runtime.service)
    async with await client.rollout("hello-shell", init={"max_steps": 3}) as session:
        ...
    await runtime.shutdown()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import grpc
from pydantic import BaseModel, ConfigDict, SkipValidation

import xrlenv
from xrlenv import paths
from xrlenv.api._pb2 import node_control_pb2_grpc as pb_grpc
from xrlenv.api.constants import GRPC_SERVER_OPTIONS
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.auth_interceptor import BearerScopeInterceptor
from xrlenv.control.build_coordinator import BuildCoordinator
from xrlenv.control.capacity import AimdConfig, HealthAimdController
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.grpc_endpoint import NodeControlServicer, RemoteNodeTransport
from xrlenv.control.image_planner import NodeBudget
from xrlenv.control.node_builder import GrpcNodeBuilder
from xrlenv.control.node_registry import NodeRegistry
from xrlenv.control.node_transport import NodeTransport
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.registry_resolver import resolver_from_env
from xrlenv.control.run_dir_janitor import RunDirJanitor
from xrlenv.control.scheduler import Scheduler
from xrlenv.control.security import TokenStore
from xrlenv.control.service import CoordinatorRolloutService, RolloutService
from xrlenv.control.state import SqliteStateStore, StateStore
from xrlenv.control.state_retention_janitor import StateRetentionJanitor
from xrlenv.control.template_catalog import TemplateCatalog
from xrlenv.control.trajectory_sink import PlatformJsonlSink, TrajectorySink
from xrlenv.observability.metrics import MetricsRegistry
from xrlenv.observability.server import MetricsServer

if TYPE_CHECKING:
    # IDE / mypy only. Runtime resolution would close a cycle:
    # xrlenv.admin.server -> xrlenv.control.state -> xrlenv.control.__init__
    # -> xrlenv.control.distributed_runtime (us). The model field below uses
    # ``SkipValidation[Any]`` so Pydantic never needs to resolve the
    # AdminServer name at runtime.
    from xrlenv.admin.server import AdminServer

LOGGER = logging.getLogger(__name__)
AdminRolloutPageSize = Literal[32, 64, 128, 256]


def _is_wire_timeout(exc: BaseException) -> bool:
    """True if ``exc`` is a control-plane reply timeout (the node may still be
    working), as opposed to a real failure.

    The bidi-stream ``_send_and_wait`` re-raises a reply timeout as
    ``XRLEnvError`` chained ``from`` the underlying ``TimeoutError`` — so check
    the cause chain, with a message fallback. A wire timeout means "we stopped
    waiting", NOT "the build failed": the node's pull may complete afterward, so
    the coordinator records it ``registered`` (lazy) instead of ``failed``.
    """
    cause: BaseException | None = exc
    seen = 0
    while cause is not None and seen < 5:
        if isinstance(cause, TimeoutError):
            return True
        cause = cause.__cause__
        seen += 1
    return "timed out after" in str(exc)

# ``$XRLENV_HOME/runs`` (default ``~/.xrlenv/runs``); see :mod:`xrlenv.paths`.
DEFAULT_RUNS_ROOT = paths.runs_root()


class DistributedRuntime(BaseModel):
    """Control-plane half of a multi-process runtime.

    Owns the gRPC server, the (single, for now) connected node's
    :class:`RemoteNodeTransport`, plus the standard control-plane components
    (state, catalog, scheduler, admission, coordinator, service, sink).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    state: SkipValidation[StateStore]
    catalog: TemplateCatalog
    scheduler: Scheduler
    coordinator: RolloutCoordinator
    service: SkipValidation[RolloutService]
    sink: SkipValidation[TrajectorySink]
    admission: AdmissionQueue
    registry: NodeRegistry
    grpc_server: SkipValidation[grpc.aio.Server]
    grpc_port: int
    metrics: MetricsRegistry
    build_coordinator: BuildCoordinator
    """P1.6.c — drives ``xrlenv build apply`` against the connected
    node fleet via the spec-21 ``BuildImagesCommand`` RPC."""
    metrics_server: SkipValidation[MetricsServer | None] = None
    run_dir_janitor: RunDirJanitor | None = None
    state_retention_janitor: StateRetentionJanitor | None = None
    gc_reconciler: SkipValidation[Any] = None  # GCReconciler | None
    raw_gc_reconciler: SkipValidation[Any] = None  # RawGCReconciler | None — P1.7.A.2
    aimd_loop: SkipValidation[Any] = None  # AimdControlLoop | None — Stage-3 P1
    wal_checkpointer: SkipValidation[Any] = None  # WalCheckpointer | None
    loop_monitor: SkipValidation[Any] = None  # LoopLagMonitor | None — 2026-08-21
    control_keepalive: SkipValidation[Any] = None  # ControlKeepaliveLoop | None
    # Type is ``Any`` so Pydantic v2 doesn't need to resolve
    # ``AdminServer`` at model-class-creation time (see TYPE_CHECKING note
    # above). Validation is already skipped via ``SkipValidation``;
    # callers who want IDE help should annotate locally:
    # ``runtime.admin_server  # type: AdminServer | None``.
    admin_server: SkipValidation[Any] = None
    drain_timeout_s: float = 30.0
    _node_connected: SkipValidation[asyncio.Event] = None  # type: ignore[assignment]
    # Tasks spawned by ``_on_disconnected`` (one per gRPC node-loss seal).
    # ``shutdown()`` drains this set before closing the state store so
    # the seal-on-disconnect path doesn't hit a closed SQLite handle.
    # Operator-reported regression (2026-05-04): with the previous
    # ordering, ``state.close()`` ran before the in-flight
    # ``handle_node_lost`` coroutines finished their ``list_rollouts``
    # query, raising ``ProgrammingError: Cannot operate on a closed
    # database``.
    _disconnect_tasks: SkipValidation[set[asyncio.Task[None]]] = None  # type: ignore[assignment]

    async def start(self) -> None:
        """Boot background tasks (admission worker, node-registry watchdog)."""
        await self.admission.start()
        await self.registry.start()
        if self.metrics_server is not None:
            self.metrics_server.start()
        if self.run_dir_janitor is not None:
            await self.run_dir_janitor.start()
        if self.state_retention_janitor is not None:
            await self.state_retention_janitor.start()
        if self.gc_reconciler is not None:
            await self.gc_reconciler.start()
        if self.raw_gc_reconciler is not None:
            await self.raw_gc_reconciler.start()
        if self.aimd_loop is not None:
            await self.aimd_loop.start()
        if self.wal_checkpointer is not None:
            await self.wal_checkpointer.start()
        if self.loop_monitor is not None:
            await self.loop_monitor.start()
        if self.control_keepalive is not None:
            await self.control_keepalive.start()
        if self.admin_server is not None:
            self.admin_server.start()

    async def wait_for_node(self, timeout_s: float = 30.0) -> NodeTransport:
        """Block until a node connects out to us. Returns the first transport.

        Slice 3 is single-node; the multi-node `NodeRegistry` lands in 3.5.
        """
        if self._node_connected is None:
            raise RuntimeError("DistributedRuntime not initialized properly")
        try:
            await asyncio.wait_for(self._node_connected.wait(), timeout=timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"no node connected to gRPC server within {timeout_s}s"
            ) from exc
        if not self.scheduler.nodes:
            raise RuntimeError("node connected event fired but scheduler has no nodes")
        return self.scheduler.nodes[0]

    async def shutdown(self) -> None:
        """Drain in-flight rollouts, cancel pending, stop background tasks,
        gracefully stop the gRPC server.
        """
        self.admission.stop_accepting()

        deadline = time.monotonic() + self.drain_timeout_s
        while time.monotonic() < deadline:
            running = [
                r for r in self.state.list_rollouts() if not r.status.is_terminal
            ]
            if not running:
                break
            await asyncio.sleep(0.05)

        self.admission.cancel_pending()
        await self.coordinator.deadline_watcher.shutdown()
        await self.coordinator.idle_ttl_watcher.shutdown()
        await self.admission.stop()
        if self.gc_reconciler is not None:
            await self.gc_reconciler.shutdown()
        if self.raw_gc_reconciler is not None:
            await self.raw_gc_reconciler.shutdown()
        if self.aimd_loop is not None:
            await self.aimd_loop.shutdown()
        if self.wal_checkpointer is not None:
            await self.wal_checkpointer.shutdown()
        if self.loop_monitor is not None:
            await self.loop_monitor.shutdown()
        if self.control_keepalive is not None:
            await self.control_keepalive.shutdown()
        await self.registry.shutdown()
        # Node↔control streams are long-lived bidi RPCs that never close
        # on their own, so when this grace window expires grpc-aio
        # force-cancels them and its C layer logs, per still-open stream:
        #   grpc._cython.cygrpc: Exception not handled by _handle_exceptions
        #   in servicer method [.../NodeControlStream] ... CancelledError
        #   (from _schedule_rpc_coro in server.pyx.pxi)
        # This is HARMLESS shutdown noise, NOT an xrlenv error: our
        # servicer already catches its own CancelledError and seals/
        # persists state before this stop() (NodeControlStream's except/
        # finally in grpc_endpoint.py). The trace originates in grpc's
        # extension, outside any Python try/except we control. We do NOT
        # filter the grpc logger here — that risks hiding a real error;
        # the operator-facing note in docs/deploy/multi_node_deployment/
        # runbook.md documents it as expected on Ctrl-C / SIGTERM.
        await self.grpc_server.stop(grace=2.0)

        # gRPC stream cancellation runs ``_on_disconnected`` for each
        # attached node, which spawns a ``coordinator.handle_node_lost``
        # task that reads + updates the state store. Wait for those to
        # finish *before* closing the state store; otherwise the seal
        # path raises ``sqlite3.ProgrammingError: Cannot operate on a
        # closed database``. Bounded wait (a couple of seconds) so a
        # stuck seal doesn't hold up shutdown indefinitely; a leaked
        # task on a closed store is no worse than the pre-fix race.
        if self._disconnect_tasks:
            pending = [t for t in self._disconnect_tasks if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=5.0)

        if self.run_dir_janitor is not None:
            await self.run_dir_janitor.shutdown()
        if self.state_retention_janitor is not None:
            await self.state_retention_janitor.shutdown()
        if self.admin_server is not None:
            self.admin_server.stop()
        if self.metrics_server is not None:
            self.metrics_server.stop()

        if hasattr(self.state, "close"):
            self.state.close()


def _try_local_docker_digest_resolver() -> Any:
    """Return a Docker-backed digest resolver when a local daemon is
    reachable, or ``None`` when it isn't.

    The control plane in some topologies (laptop operator running both
    ``xrlenv up`` and the data-plane node, or node-on-control-plane
    Scenario-1) HAS a local Docker daemon and can resolve manifest
    image tags into ``sha256:...`` digests at register time. Other
    topologies (control plane on a dedicated VM, no Docker) don't —
    those see ``self._client`` raise during construction. We try
    once and fall back to ``None`` (the unpinned-warning path) on
    any failure.
    """
    try:
        from xrlenv.backends.docker import DockerBackend, DockerBackendConfig

        # Construct a minimal backend just for the resolver method.
        # We don't need runs_root / xrlenv_pkg_path for digest lookup
        # since ``resolve_image_digest`` only touches ``self._client``;
        # supply harmless placeholders so the BaseModel passes
        # validation.
        backend = DockerBackend(
            DockerBackendConfig(
                runs_root=Path("/tmp"),
                xrlenv_pkg_path=Path(xrlenv.__file__).resolve().parent,
                stub_transport="tcp",
            ),
        )
        # Tickle the client; if Docker is unreachable this raises.
        backend._client.ping()
        return backend.resolve_image_digest
    except Exception as exc:
        LOGGER.info(
            "no local Docker daemon reachable for digest resolution: %s; "
            "templates with tag-only image refs will register unpinned "
            "(spec 19 warning).",
            exc,
        )
        return None


def _mark_stale_connected_nodes_lost(state: StateStore) -> int:
    """Sweep stale ``connected`` rows in the ``nodes`` table at
    control-plane startup. Returns the count of swept rows.

    The watchdog only mutates rows for transports it owns in-memory;
    a previous-process row marked ``connected`` that never received
    a clean ``deregister`` (process kill, host reboot) is invisible
    to the new registry. ``Client.list_nodes()`` would otherwise
    return those stale rows and ``Client.wait_for_nodes()`` would
    satisfy readiness immediately. Marking them ``lost`` here is
    safe — when a real node reattaches, ``record_node_connected``
    flips the row back to ``connected`` (its UPSERT semantics
    overwrite both status and connected_at).

    Audit H1 (2026-05-01).
    """
    swept = 0
    for record in state.list_nodes(status="connected"):
        try:
            state.record_node_disconnected(record.node_id)
            swept += 1
        except Exception:
            LOGGER.exception(
                "startup-sweep: failed to mark stale node=%s as lost",
                record.node_id,
            )
    return swept


def _prune_unrostered_lost_nodes(state: StateStore, roster: set[str]) -> list[str]:
    """Startup reconciliation — reap ``lost`` node rows whose node_id is absent
    from the current ``nodes.yaml`` roster (a decommissioned host; commonly an
    IP-derived node_id orphaned by a cluster reboot). Returns the pruned ids.

    ``nodes.yaml`` is generated from ``clusters.yaml`` (the single source of
    truth) on every redeploy, and a redeploy always bounces ``xrlenv up`` — so
    this fires automatically whenever the topology changes, keeping the registry
    from accumulating dead rows reboot-over-reboot. Skips entirely when the
    roster is empty (a missing / failed-to-load ``nodes.yaml`` must never nuke
    the whole registry), and swallows its own errors (reconciliation is
    best-effort — it must never block control-plane startup).
    """
    if not roster:
        return []
    try:
        pruned = state.prune_lost_nodes(keep=roster)
    except Exception:
        LOGGER.exception("startup-prune: failed to reap unrostered lost nodes")
        return []
    if pruned:
        LOGGER.info(
            "startup-prune: reaped %d unrostered lost node(s) from the registry: %s",
            len(pruned), ", ".join(sorted(pruned)),
        )
    return pruned


class _DistributedBudgetProvider:
    """P1.6.c — derive per-node disk budgets from the connected fleet.

    For each currently-registered node, reads the heartbeat-cached free
    disk (``transport.disk_state()``) and computes ``available_bytes =
    free_disk - reservations`` — falling back to a direct
    ``report_images`` probe only when no heartbeat sample exists yet.
    Disconnected nodes are skipped (the planner sees only the live
    fleet — the operator must wait for nodes to attach before
    ``xrlenv build apply`` produces a useful placement).
    """

    def __init__(self, *, registry: NodeRegistry) -> None:
        self._registry = registry

    async def get_budgets(
        self,
        *,
        reserved_runtime_gb: int,
        buffer_gb: int,
        cap_per_node_gb: int | None,
    ) -> list[NodeBudget]:
        reserved_bytes = (reserved_runtime_gb + buffer_gb) * 1024**3
        budgets: list[NodeBudget] = []
        for node_id in self._registry.node_ids:
            transport = self._registry.get(node_id)
            if transport is None:
                continue
            # Budgets only need free disk, which heartbeats already cache
            # (issue #14). Use that instead of a full ``report_images``
            # (``docker system df``) per apply — df takes the containerd
            # metadata lock and, under a pull storm, times out (see
            # backends/docker.py ``_gc_containerd_content``). Fall back to
            # a direct probe only for a just-connected node with no
            # heartbeat sample yet.
            free_disk = 0
            disk_state = getattr(transport, "disk_state", None)
            if callable(disk_state):
                free_disk = int(disk_state()[0])
            if free_disk <= 0:
                try:
                    report = await transport.report_images()  # type: ignore[attr-defined]
                    free_disk = int(report.free_disk_bytes)
                except Exception:
                    LOGGER.exception(
                        "build-plan budget probe: disk sample unavailable "
                        "for %s", node_id,
                    )
                    continue
            available = max(0, free_disk - reserved_bytes)
            if cap_per_node_gb is not None:
                available = min(available, cap_per_node_gb * 1024**3)
            budgets.append(NodeBudget(
                node_id=node_id,  # type: ignore[arg-type]
                available_bytes=available,
            ))
        # Helper for tests + callers that prefer "no nodes connected"
        # to surface as InsufficientCapacity rather than an empty
        # placement.
        return budgets

    async def get_inventory(self) -> dict[str, set[str]]:
        """ClusterInventoryProvider — return
        ``{image_ref: {node_id_where_present, ...}}`` for the connected
        fleet. Used by ``--fill-missing`` apply mode to decide which
        plan entries already exist on at least one node + don't need
        re-dispatch. Disconnected nodes are skipped (same shape as
        ``get_budgets`` — operator must wait for connectivity).
        """
        inventory: dict[str, set[str]] = {}
        for node_id in self._registry.node_ids:
            transport = self._registry.get(node_id)
            if transport is None:
                continue
            try:
                report = await transport.report_images()  # type: ignore[attr-defined]
            except Exception:
                LOGGER.exception(
                    "fill-missing inventory probe: report_images "
                    "failed for %s", node_id,
                )
                continue
            for img in getattr(report, "images", ()):
                inventory.setdefault(img.name, set()).add(node_id)
        return inventory

    @staticmethod
    def _placeholder() -> int:
        # Reserved for a future ``probe_disk_capacity`` helper that
        # asks the node for its full disk capacity (today's
        # ``report_images`` only returns ``free_disk_bytes``).
        return 0


def _primary_outbound_ip() -> str | None:
    """Best-effort primary IPv4 of this host, for display only.

    Uses the standard UDP-connect trick: ``connect`` on a datagram socket
    sends no packet, it just makes the OS pick the source address its routing
    table would use to reach the target — so this works offline as long as a
    default route exists. Returns ``None`` if it can't be determined.
    """
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        return None


def _resolve_advertise_endpoint(bind_host: str, port: int) -> str:
    """The ``host:port`` to *show* node operators for the control stream.

    A wildcard / loopback bind (``0.0.0.0`` / ``::`` / ``127.0.0.1``) isn't a
    dialable address from another box, so we substitute the host's detected
    primary IP when we can find one; otherwise we fall back to the configured
    host verbatim (always truthful, never invents an address).
    """
    if bind_host in ("0.0.0.0", "::", "", "127.0.0.1", "localhost", "::1"):
        detected = _primary_outbound_ip()
        if detected is not None:
            return f"{detected}:{port}"
    return f"{bind_host}:{port}"


async def build_distributed_runtime(
    *,
    grpc_port: int = 50051,
    grpc_host: str = "127.0.0.1",
    runs_root: Path | None = None,
    template_dirs: list[Path] | None = None,
    state: StateStore | None = None,
    state_db_path: Path | None = None,
    metrics: MetricsRegistry | None = None,
    metrics_host: str | None = None,
    metrics_port: int | None = None,
    run_dir_retention_days: int | None = 14,
    audit_retention_days: int | None = 30,
    events_retention_days: int | None = 14,
    raw_rollout_retention_days: int | None = 14,
    admin_host: str | None = None,
    admin_port: int | None = None,
    admin_allow_public: bool = False,
    admin_nodes_yaml: Path | None = None,
    admin_rollout_page_size: AdminRolloutPageSize = 32,
    token_store: TokenStore | None = None,
    scheduler_max_runs_per_task: int = 4,
    gc_reconcile_interval_s: float | None = 60.0,
    adaptive_admission: bool = False,
    aimd_config: AimdConfig | None = None,
) -> DistributedRuntime:
    """Spin up a control-plane process: state + catalog + scheduler + coordinator
    + admission, plus a gRPC server on ``grpc_host:grpc_port`` for nodes to
    connect out to.

    The returned runtime has no nodes yet — start one or more
    ``xrlenv-node serve --control-plane <host>:<port>`` processes, then call
    :py:meth:`DistributedRuntime.wait_for_node`.
    """
    runs_root = runs_root or DEFAULT_RUNS_ROOT
    runs_root.mkdir(parents=True, exist_ok=True)

    xrlenv_pkg_path = Path(xrlenv.__file__).resolve().parent

    if state is None:
        db_path = state_db_path or (runs_root.parent / "state.db")
        state = SqliteStateStore(db_path)
    # Crash-recovery sweep for stale ``connected`` node rows. The
    # ``nodes`` table is a persistent shadow of the in-memory
    # NodeRegistry; ``record_node_connected`` / ``record_node_disconnected``
    # mutate it on stream open/close. A control-plane crash, kill -9,
    # or host reboot bypasses ``record_node_disconnected``, leaving
    # rows pinned at ``status='connected'`` even though no transport
    # is attached anywhere. Without this sweep, a fresh ``xrlenv up``
    # boots an empty ``NodeRegistry`` but ``Client.list_nodes()``
    # would still return the stale rows, and the connect-mode
    # smoke driver's ``Client.wait_for_nodes(min_nodes=N)`` would
    # pass instantly before any node had reattached. Audit H1 from
    # the 2026-05-01 review.
    _swept_stale_nodes = _mark_stale_connected_nodes_lost(state)
    if _swept_stale_nodes:
        LOGGER.warning(
            "startup-sweep: marked %d previously-``connected`` node row(s) "
            "as ``lost`` (these are stale from a prior control-plane "
            "instance and will return to ``connected`` only when a fresh "
            "stream reattaches)", _swept_stale_nodes,
        )
    sink = PlatformJsonlSink(runs_root)
    # Spec 19 §"Audit logging": route template.registered + mount.denied
    # through the audit table. Distributed mode normally runs the
    # control plane without a local Docker daemon, but the laptop
    # operator topology (Scenario 1) co-locates Docker; opportunistically
    # wire a digest resolver when a daemon is reachable so the
    # template.image_unpinned warning doesn't fire spuriously for
    # locally-built tags. Falls back to None (warning-on-register)
    # when no daemon is available — phase-1 swaps to a control-plane-
    # side registry resolver that hits the cluster mirror.
    def _audit(kind: str, payload: dict[str, Any]) -> None:
        state.append_audit(kind, payload=payload)

    catalog = TemplateCatalog(
        digest_resolver=_try_local_docker_digest_resolver(),
        audit_callback=_audit,
    )

    if template_dirs is None:
        # Built-in templates ship under ``xrlenv/templates/`` (just
        # hello-shell today). Benchmark plug-ins each live at a
        # canonical depth ``xrlenv_plugins/<cat>/<name>/manifest.yaml``
        # discovered separately below.
        template_dirs = [xrlenv_pkg_path / "templates"]
    from xrlenv.control.template_discovery import (
        find_entry_point_manifest_files,
        find_external_template_dir_manifests,
        find_plugin_manifest_files,
    )

    for d in template_dirs:
        if d.exists():
            registered = catalog.register_dir(d)
            LOGGER.info(
                "registered %d template(s) from %s: %s",
                len(registered),
                d,
                [m.name for m in registered],
            )

    # Plug-in manifests — in-tree (flat layout under
    # ``xrlenv_plugins/<cat>/<name>/manifest.yaml``) plus B11 external
    # discovery via XRLENV_TEMPLATE_DIRS + entry-points. The
    # control-plane catalog only needs the manifest paths; the actual
    # DockerBackend lives on the data-plane node, where
    # xrlenv/node/cli.py wires extra_plugin_roots into
    # DockerBackendConfig.
    plugin_manifests = find_plugin_manifest_files(xrlenv_pkg_path)
    plugin_manifests.extend(
        m.manifest_path for m in (
            find_external_template_dir_manifests()
            + find_entry_point_manifest_files()
        )
    )
    if plugin_manifests:
        registered = catalog.register_paths(plugin_manifests)
        LOGGER.info(
            "registered %d plug-in template(s): %s",
            len(registered),
            [m.name for m in registered],
        )

    # Scheduler starts empty; nodes are added as they connect (allow_empty
    # bypasses the at-least-one-node guard the in-process runtime relies on).
    # Stage-3 (P1) — health-derived adaptive admission. Off by default
    # so a run can A/B it against the static estimator. When on, the
    # scheduler gates placement on each node's AIMD limit and the
    # AimdControlLoop (created below, once the registry exists) ticks
    # the controller from live per-node health.
    aimd_controller = (
        HealthAimdController(aimd_config) if adaptive_admission else None
    )

    scheduler = Scheduler(
        [],
        catalog=catalog,
        state=state,
        allow_empty=True,
        max_runs_per_task=scheduler_max_runs_per_task,
        aimd_controller=aimd_controller,
    )

    metrics_registry = metrics or MetricsRegistry()
    admission = AdmissionQueue(scheduler=scheduler, state=state, metrics=metrics_registry)
    from xrlenv.control.scratch_build import scratch_registry_host_from_env
    coordinator = RolloutCoordinator(
        catalog=catalog,
        scheduler=scheduler,
        state=state,
        trajectory_sink=sink,
        admission=admission,
        metrics=metrics_registry,
        scratch_registry_host=scratch_registry_host_from_env(),
    )
    # Crash-recovery sweep: see runtime.py for the rationale. Runs
    # before the gRPC server accepts the first node connection so
    # the registry mirror + scheduler don't see stale sandbox rows
    # tied to long-gone rollouts.
    coordinator.sweep_stuck_transients()
    # Issue #6: load the cluster-wide docker-kwarg policy from
    # nodes.yaml. Missing file / missing ``policy:`` section both
    # fall back to DEFAULT_POLICY so existing single-node / laptop
    # deployments behave exactly as before. The same path also
    # drives the admin server's inventory display.
    _kwargs_policy = None
    if admin_nodes_yaml is not None:
        try:
            from xrlenv.control.nodes_yaml import load_nodes_yaml
            _inventory = load_nodes_yaml(admin_nodes_yaml)
            _kwargs_policy = _inventory.policy
            # Startup reconciliation against the roster: reap `lost` registry
            # rows for nodes no longer in nodes.yaml (decommissioned hosts /
            # reboot-orphaned IP-derived node_ids). Best-effort + guarded on a
            # non-empty roster inside the helper.
            _prune_unrostered_lost_nodes(state, {n.id for n in _inventory.nodes})
            LOGGER.info(
                "loaded cluster docker-kwarg policy from %s "
                "(allowed_devices=%s, denied_caps=%s, "
                "allow_host_network=%s, allow_privileged=%s, "
                "allowed_host_paths=%s)",
                admin_nodes_yaml,
                list(_kwargs_policy.allowed_devices),
                list(_kwargs_policy.denied_caps),
                _kwargs_policy.allow_host_network,
                _kwargs_policy.allow_privileged,
                list(_kwargs_policy.allowed_host_paths),
            )
            # Per-node per-runtime concurrency caps (sysbox-fs wedge prevention
            # — notes/design-per-node-runtime-concurrency-cap.md). Only nodes
            # with a non-empty map contribute; absent ⇒ unlimited ⇒ the
            # scheduler gate is a no-op.
            _runtime_caps = {
                n.id: dict(n.max_concurrent_by_runtime)
                for n in _inventory.nodes
                if n.max_concurrent_by_runtime
            }
            if _runtime_caps:
                scheduler.set_runtime_caps(_runtime_caps)
                LOGGER.info(
                    "loaded per-node runtime concurrency caps from %s: %s",
                    admin_nodes_yaml, _runtime_caps,
                )
        except Exception:
            LOGGER.warning(
                "failed to load kwargs policy from %s; falling back "
                "to DEFAULT_POLICY", admin_nodes_yaml, exc_info=True,
            )
    raw_container_coordinator = RawContainerCoordinator(
        scheduler=scheduler, state=state, kwargs_policy=_kwargs_policy,
        metrics=metrics_registry,
        # Issue #18 fix #1: route raw acquires through the admission
        # queue so ``CapacityExhausted`` blocks (up to the queue
        # timeout) instead of cascading ``RESOURCE_EXHAUSTED`` errors
        # to consumers under load.
        admission=admission,
        # Freshness model: resolve a registry tag -> content digest per
        # acquire so a rebuilt-and-re-pushed image reaches the nodes
        # without consumers threading digests. Env-tunable; kill-switch
        # is XRLENV_REGISTRY_DIGEST_RESOLVE=0.
        digest_resolver=resolver_from_env(),
        # audit H11 — a deployment that opts OUT of the periodic raw-GC reconciler
        # (``gc_reconcile_interval_s=None``) has no way to inventory a reconnecting node's
        # surviving raw/compose containers, so it is EXPLICITLY not capable of restart-safe
        # raw/compose sessions (gym/step only): raw + compose acquires fail loud rather than
        # silently accrue un-reconcilable load.
        raw_reconnect_capable=(
            gc_reconcile_interval_s is not None and gc_reconcile_interval_s > 0
        ),
    )
    # See ``runtime.py`` for the rationale: the scheduler's capacity gate
    # needs a second load source for raw-container sessions, which don't
    # appear in ``state.list_sandboxes()``.
    scheduler.set_raw_session_provider(
        raw_container_coordinator.iter_load_entries,
    )
    service = CoordinatorRolloutService(
        coordinator,
        raw_container_coordinator=raw_container_coordinator,
    )

    metrics_server: MetricsServer | None = None
    if metrics_port is not None:
        metrics_server = MetricsServer(
            registry=metrics_registry,
            host=metrics_host or "127.0.0.1",
            port=metrics_port,
            # Let the metrics dashboard's role-clarifier banner link to the
            # admin panel for per-entity drill-down (None when admin is off).
            admin_port=admin_port,
        )

    node_connected = asyncio.Event()

    async def _handle_node_lost_all(
        node_id: str, transport: RemoteNodeTransport | None = None,
    ) -> None:
        # Seal BOTH rollout planes on the lost node. The gym/step
        # coordinator seals its ``rollouts`` table; the raw-container
        # coordinator seals its ``raw_rollouts`` sessions — pre-fix only
        # the former ran, so a lost node's raw sessions lingered as
        # ``running`` forever (inflating the admin "active containers"
        # count). Both are idempotent.
        #
        # H11: on the stream-close path ``transport`` is the specific closed stream, so the raw
        # coordinator seals only sessions whose ``session.node IS`` it — a stale old-stream close
        # that fires after a reconnected replacement re-adopted the node can't seal the
        # replacement's live sessions. The watchdog path passes no transport (whole node lost).
        await coordinator.handle_node_lost(node_id)
        with contextlib.suppress(Exception):
            await raw_container_coordinator.handle_node_lost(
                node_id, transport=transport,
            )

    async def _on_node_lost(
        node_id: str, transport: RemoteNodeTransport | None = None,
    ) -> None:
        # Issue #18: the heartbeat watchdog's node-loss path must run
        # the SAME teardown as the stream-disconnect path
        # (``_on_disconnected`` below) — drop the node from the
        # scheduler too, not just the registry + state store. Without
        # the scheduler removal the two paths disagreed: a
        # watchdog-lost node kept receiving placements while every
        # operator view (``xrlenv nodes``, admin ``/nodes``) showed it
        # ``lost``. ``remove_node`` and ``handle_node_lost`` are both
        # idempotent, so a node lost via watchdog *and* stream close
        # gets a harmless second teardown.
        #
        # H11: the registry already confirmed ``transport`` was still the current stream before
        # calling us (identity-conditional deregister), and passes it here so the raw seal is
        # scoped to that exact generation — a reconnected replacement's sessions are untouched.
        scheduler.remove_node(node_id)
        # Self-heal (2026-08-21): actively CLOSE the lost node's control
        # stream so it reconnects and re-registers. Deregistering from the
        # in-memory registry (done by the watchdog before it calls us) does
        # NOT break the live bidi stream — the node keeps a half-open
        # connection whose HTTP/2 keepalive is still answered, so it never
        # sees an error and never redials, staying ``lost`` indefinitely.
        # ``request_terminate`` makes the servicer end the stream; the
        # node's reconnect loop then dials a fresh one. Guarded on
        # ``transport`` (the watchdog always passes the exact stale stream;
        # a legacy 1-arg loss handler would pass None).
        if transport is not None:
            transport.request_terminate("heartbeat-grace exceeded")
        await _handle_node_lost_all(node_id, transport)

    def _on_mass_loss(lost: int, registered: int) -> None:
        # The watchdog deferred a would-be fleet-wide eviction (suspected
        # control-plane-side stall). Bump a metric + emit an operator-facing
        # ALERT line so this surfaces in seconds rather than a 13-h outage.
        metrics_registry.nodes_mass_loss_deferred_total.inc()
        LOGGER.critical(
            "ALERT: node watchdog deferred a mass eviction (%d/%d nodes stale "
            "in one sweep) — suspected control-plane stall. Check "
            "xrlenv_control_loop_lag_seconds and the overview 'nodes lost' "
            "count; do NOT restart the fleet before confirming a CP-side cause.",
            lost, registered,
        )

    registry = NodeRegistry(
        on_node_lost=_on_node_lost,
        state=state,
        on_mass_loss=_on_mass_loss,
    )
    # Strong refs to background "seal on disconnect" tasks so the GC does not
    # reap them mid-flight (RUF006). Cleared as each task completes.
    disconnect_tasks: set[asyncio.Task[None]] = set()

    def _on_connected(t: RemoteNodeTransport) -> None:
        # Register the transport FIRST (RPCs — incl. the readoption inventory below — need it),
        # but do NOT make the node schedulable yet.
        registry.register(t)
        # H11 — EVICT any prior generation's scheduler entry the instant this transport registers,
        # BEFORE the (async, possibly-failing) readoption runs. ``registry.register`` already
        # replaced the registry identity, so the OLD stream's later disconnect will see itself as
        # stale and skip teardown (it must, so it can't evict this replacement) — which means if we
        # deferred the scheduler eviction to a SUCCESSFUL readopt, a FAILED replacement readoption
        # would leave the old (dead) scheduler entry live and placement could keep selecting it
        # indefinitely. Evicting here makes the node UNSCHEDULABLE for the whole readopt window; a
        # trustworthy pass re-adds it below. Idempotent no-op for a first-time connect.
        scheduler.remove_node(t.node_id)
        # H11 — after a control-plane restart ``_sessions`` is empty, so admitting the node for
        # placement before its surviving raw/compose load is re-adopted lets admission OVER-place
        # it (``iter_load_entries`` reports nothing). Gate ``scheduler.add_node`` (and the
        # ``node_connected`` signal, so it still implies "schedulable") behind a synchronous
        # per-node readoption.
        #
        # FAIL-CLOSED (audit H11): only admit after a COMPLETE, trustworthy readoption pass;
        # retry with backoff on failure rather than admitting with surviving load unaccounted. A
        # node that never readopts cleanly stays connected but UNSCHEDULABLE (safe) until it
        # reconnects. STREAM-GENERATION-SAFE: abandon if this transport is no longer the current
        # registered one (a disconnect/reconnect replaced it), so a stale task can't re-admit a
        # dead transport or fire ``node_connected`` for it.
        _READOPT_MAX_ATTEMPTS = 5

        async def _rollback_stale() -> None:
            # Roll back any sessions THIS transport's readopt inserted, so a stale/superseded
            # generation never leaves a session routed through a closed stream (H11). The seal is
            # transport-scoped, so it can't touch a replacement's sessions.
            with contextlib.suppress(Exception):
                await raw_container_coordinator.handle_node_lost(t.node_id, transport=t)

        async def _readopt_then_admit() -> None:
            for attempt in range(_READOPT_MAX_ATTEMPTS):
                if registry.get(t.node_id) is not t:
                    LOGGER.info(
                        "node=%s readopt-on-connect abandoned — transport no longer current "
                        "(disconnected/replaced, H11)", t.node_id,
                    )
                    await _rollback_stale()   # a prior partial pass may have inserted sessions
                    return
                ok = True
                if raw_gc_reconciler is not None:
                    try:
                        ok = await raw_gc_reconciler.readopt_node_on_connect(t)
                    except Exception:
                        LOGGER.exception(
                            "node=%s readopt-on-connect raised (H11)", t.node_id,
                        )
                        ok = False
                elif raw_container_coordinator.iter_load_entries():
                    # H11: no raw-GC reconciler ⇒ no reconnect inventory/readoption is possible.
                    # This config (``gc_reconcile_interval_s=None``) is only valid for gym/step —
                    # it is NOT capable of distributed raw/compose sessions across restart. If the
                    # raw coordinator nonetheless already carries load, fail closed rather than
                    # admit uninventoried; otherwise (the gym/step-only case) admit normally.
                    LOGGER.critical(
                        "node=%s: raw sessions exist but there is no raw-GC reconciler to "
                        "inventory reconnect survivors — NOT admitting (fail closed, H11). This "
                        "deployment (gc_reconcile_interval_s=None) is not raw-capable.", t.node_id,
                    )
                    return
                if ok:
                    # Stream-generation guard: only admit if STILL the current transport.
                    if registry.get(t.node_id) is t:
                        # Evict any STALE prior generation's scheduler entry first (H11): a
                        # replacement must supersede the old stream atomically here, not rely on
                        # the old stream's later (skipped) disconnect — else two entries share the
                        # node_id and placement can pick the dead one.
                        scheduler.remove_node(t.node_id)
                        scheduler.add_node(t)
                        node_connected.set()
                        LOGGER.info(
                            "node=%s registered with scheduler + registry (post-readoption)",
                            t.node_id,
                        )
                    else:
                        LOGGER.info(
                            "node=%s became stale during readopt — rolling back, not admitted "
                            "(H11)", t.node_id,
                        )
                        await _rollback_stale()
                    return
                await asyncio.sleep(min(2 ** attempt, 10))
            # Exhausted without a clean pass — stays UNSCHEDULABLE (fail closed). If we went
            # stale, roll back any partial mutations rather than leaving them.
            if registry.get(t.node_id) is not t:
                await _rollback_stale()
            LOGGER.critical(
                "node=%s FAILED readoption after %d attempts — left connected but "
                "UNSCHEDULABLE (fail closed, H11); placement is withheld until it reconnects. "
                "Investigate node health / durable state.", t.node_id, _READOPT_MAX_ATTEMPTS,
            )
        task = asyncio.create_task(
            _readopt_then_admit(), name=f"readopt-on-connect-{t.node_id}",
        )
        disconnect_tasks.add(task)
        task.add_done_callback(disconnect_tasks.discard)

    def _on_disconnected(t: RemoteNodeTransport) -> None:
        # H11: a STALE stream close — an OLD stream that closes AFTER a reconnected replacement
        # already registered under the same node_id — must NOT evict the replacement. Only tear
        # down if this transport is STILL the current registered one; otherwise the replacement
        # owns the node_id and this close is a no-op.
        if registry.get(t.node_id) is not t:
            LOGGER.info(
                "node=%s stale stream close — a replacement is active; skipping teardown (H11)",
                t.node_id,
            )
            return
        scheduler.remove_node(t.node_id)
        registry.deregister(t.node_id, expected=t)   # identity-conditional (H11)
        LOGGER.info("node=%s removed from scheduler + registry", t.node_id)
        # Stream disconnect (process crash, TCP RST) is the primary node-loss
        # mode and must seal in-flight rollouts the same way the heartbeat
        # watchdog does. ``handle_node_lost`` is idempotent (skips terminal
        # rollouts), so a node that timed out via heartbeat *and* then closed
        # its stream gets a no-op second invocation rather than corrupted
        # state. Pass THIS transport so the raw seal is generation-scoped (H11).
        task = asyncio.create_task(
            _handle_node_lost_all(t.node_id, transport=t),
            name=f"node-lost-disconnect-{t.node_id}",
        )
        disconnect_tasks.add(task)
        task.add_done_callback(disconnect_tasks.discard)

    # Spec 19 §"API authz scopes": load the issued bearer tokens (or
    # use the empty store, which short-circuits the interceptor for
    # phase-0 unauth smoke runs). Wired into the gRPC server *before*
    # the servicer so unauthorized calls fail before the bidi handler
    # touches application state.
    store = token_store if token_store is not None else TokenStore.load()
    interceptors = (BearerScopeInterceptor(store=store, state=state),)
    # Audit M1 (2026-04-29): pin max_*_message_length on every bidi
    # channel/server so PutArchiveCommand's verifier-asset tarballs
    # don't hit gRPC's 4 MB default ceiling at remote rollout time.
    grpc_server = grpc.aio.server(
        interceptors=interceptors, options=GRPC_SERVER_OPTIONS,
    )
    pb_grpc.add_NodeControlServicer_to_server(
        NodeControlServicer(
            on_connected=_on_connected,
            on_disconnected=_on_disconnected,
            control_instance_id=str(uuid.uuid4()),
        ),
        grpc_server,
    )
    # Spec 05 — consumer-facing surface so a separate trainer / smoke
    # process can dial this control plane via Client.grpc(host, port,
    # token=...). The same BearerScopeInterceptor (consumer role)
    # gates these RPCs.
    from xrlenv.api._pb2 import rollout_control_pb2_grpc as rpb_grpc
    from xrlenv.control.rollout_endpoint import RolloutControlServicer

    rpb_grpc.add_RolloutControlServicer_to_server(
        # Stage-2: hand the servicer the AdmissionQueue so the
        # QueueStatus RPC can report a request's queue position.
        RolloutControlServicer(
            service=service, admission=admission, token_store=store,
        ),
        grpc_server,
    )
    grpc_server.add_insecure_port(f"{grpc_host}:{grpc_port}")
    await grpc_server.start()
    LOGGER.info("control-plane gRPC server listening on %s:%d", grpc_host, grpc_port)

    janitor: RunDirJanitor | None = None
    if run_dir_retention_days is not None:
        janitor = RunDirJanitor(
            runs_root=runs_root,
            retention_days=run_dir_retention_days,
        )

    # spec 20 Retention/GC matrix — bound state.db growth (audit trail dominates).
    # Skip only when every window is disabled.
    retention_janitor: StateRetentionJanitor | None = None
    if any(
        d is not None
        for d in (
            audit_retention_days,
            events_retention_days,
            raw_rollout_retention_days,
        )
    ):
        retention_janitor = StateRetentionJanitor(
            state,
            audit_retention_days=audit_retention_days,
            events_retention_days=events_retention_days,
            raw_rollout_retention_days=raw_rollout_retention_days,
        )

    # P1.6.f cluster-RPC — build coordinator constructed here so the
    # admin server (when bound) gets a live reference to it for the
    # ``POST /api/build/apply`` operator path. DistributedRuntime
    # exposes the same coordinator for in-process callers.
    async def _distributed_ensure_present(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        # P1.7.C.2 dispatch hook for per-image-ref plans. Looks up the
        # node's live transport via the registry, then invokes the
        # spec-21 ``EnsurePresentCommand`` path. Returns the same
        # ``(status, error)`` shape that ``RemoteNodeTransport.ensure_present``
        # already uses for the eager-prefetch flow.
        transport = registry.get(node_id)
        if transport is None:
            return ("failed", f"node {node_id!r} has no live transport")
        ensure = getattr(transport, "ensure_present", None)
        if ensure is None:
            return (
                "failed",
                f"node {node_id!r} transport missing ensure_present",
            )
        try:
            result = await ensure(image_ref, timeout_s=timeout_s)
        except Exception as exc:
            kind = "timeout" if _is_wire_timeout(exc) else "failed"
            return (kind, f"{type(exc).__name__}: {exc}")
        if isinstance(result, tuple) and len(result) == 2:
            status, error = result
            if status == "ok":
                return ("ok", None)
            return ("failed", error or status)
        return ("ok", None)

    async def _distributed_build_image(
        node_id: str, image_ref: str, source: Any,
        timeout_s: float, labels: dict[str, str],
        skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        # Source-build dispatch hook: resolve the node's transport
        # via the registry and ship a ``BuildImageCommand`` over the
        # spec-21 stream. ``apply_plan_remote`` (smoke) and the
        # admin /api/build/apply path both reach this through the
        # coordinator's ``_apply_per_image_ref`` branching.
        transport = registry.get(node_id)
        if transport is None:
            return ("failed", f"node {node_id!r} has no live transport")
        build_image = getattr(transport, "build_image", None)
        if build_image is None:
            return (
                "failed",
                f"node {node_id!r} transport missing build_image; "
                "this control plane is older than its remote nodes",
            )
        try:
            result = await build_image(
                image_ref=image_ref, source=source,
                timeout_s=timeout_s, labels=labels,
                skip_if_present=skip_if_present,
            )
        except Exception as exc:
            kind = "timeout" if _is_wire_timeout(exc) else "failed"
            return (kind, f"{type(exc).__name__}: {exc}")
        if isinstance(result, tuple) and len(result) == 2:
            status, error = result
            if status == "ok":
                return ("ok", None)
            return ("failed", error or status)
        return ("ok", None)

    async def _distributed_build_push(
        node_id: str, image_ref: str, source: Any,
        timeout_s: float, labels: dict[str, str],
    ) -> tuple[str, str | None, str | None]:
        # Source-build-AND-push dispatch hook (``xrlenv build push``): ship a
        # ``BuildImageCommand`` with push=true so the node builds image_ref and
        # pushes it to the registry the ref encodes, returning the pushed digest
        # for the coordinator's pin plan. Registry-HEAD skip is implied node-side
        # (build-once fleet-wide, resumable).
        transport = registry.get(node_id)
        if transport is None:
            return ("failed", f"node {node_id!r} has no live transport", None)
        build_and_push = getattr(transport, "build_and_push_image", None)
        if build_and_push is None:
            return (
                "failed",
                f"node {node_id!r} transport missing build_and_push_image; "
                "this control plane is older than its remote nodes",
                None,
            )
        try:
            result = await build_and_push(
                image_ref=image_ref, source=source,
                timeout_s=timeout_s, labels=labels,
            )
        except Exception as exc:
            kind = "timeout" if _is_wire_timeout(exc) else "failed"
            return (kind, f"{type(exc).__name__}: {exc}", None)
        if isinstance(result, tuple) and len(result) == 3:
            status, error, repo_digest = result
            if status == "ok":
                return ("ok", None, repo_digest)
            return ("failed", error or status, None)
        return ("failed", "build_and_push_image returned unexpected shape", None)

    async def _distributed_free_disk(node_id: str) -> tuple[int, int] | None:
        # Disk-aware build pacing: return the node's heartbeat-cached
        # ``(free, total)`` so the coordinator throttles dispatch before a
        # heavy plan overruns the node's eviction reserve. ``None`` when the
        # node has no live transport / no heartbeat yet (gate skips it).
        transport = registry.get(node_id)
        if transport is None:
            return None
        disk_state = getattr(transport, "disk_state", None)
        if not callable(disk_state):
            return None
        try:
            free, total = disk_state()
            return (int(free), int(total))
        except Exception:
            return None

    _budget_provider = _DistributedBudgetProvider(registry=registry)
    build_coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=GrpcNodeBuilder(node_lookup=registry.get),
        budget_provider=_budget_provider,
        inventory_provider=_budget_provider,
        ensure_present_fn=_distributed_ensure_present,
        build_image_fn=_distributed_build_image,
        build_push_fn=_distributed_build_push,
        free_disk_fn=_distributed_free_disk,
    )

    admin: AdminServer | None = None
    if admin_port is not None:
        # Local import: avoid the circular xrlenv.admin → xrlenv.cli.commands
        # → xrlenv.control.distributed_runtime cycle that triggers when
        # tests import xrlenv.admin.server directly.
        from xrlenv.admin.server import AdminServer, AdminServerConfig

        if state_db_path is None and isinstance(state, SqliteStateStore):
            state_db_path = state.path
        if state_db_path is None:
            state_db_path = runs_root.parent / "state.db"

        # node_lookup bridges the admin server's spec-17 cache to the
        # live NodeRegistry: on cache miss the admin dispatches a
        # FetchTrajectoryCommand to the rollout's owning node over the
        # same bidi stream the coordinator uses. Single-host setups (no
        # remote nodes registered) silently fall back to the local
        # runs/ disk read inside the cache fetch_fn.
        def _node_lookup(node_id: str) -> NodeTransport | None:
            for n in scheduler.nodes:
                if n.node_id == node_id:
                    return n
            return None

        admin = AdminServer(
            config=AdminServerConfig(
                state_db=state_db_path,
                runs_root=runs_root,
                nodes_yaml=admin_nodes_yaml,
                host=admin_host or "127.0.0.1",
                port=admin_port,
                allow_public=admin_allow_public,
                node_lookup=_node_lookup,
                rollout_page_size=admin_rollout_page_size,
                build_coordinator=build_coordinator,
                # Pass the *resolved* store (the one already loaded
                # from disk if the caller didn't pass one explicitly),
                # NOT the raw input parameter. Otherwise ``xrlenv up``
                # — which doesn't pass token_store= — leaves the
                # admin server with token_store=None, the auth
                # middleware no-ops, and every browser request lands
                # at every page even though gRPC-side bearer tokens
                # are actively enforcing.
                token_store=store,
                # Cosmetic overview banner (optional). The control-plane
                # endpoint is the gRPC bind resolved to a dialable IP; the
                # registries mirror the bootstrap env vars (unset → hidden).
                control_plane_endpoint=_resolve_advertise_endpoint(
                    grpc_host, grpc_port,
                ),
                registry_mirror=(
                    os.environ.get("XRLENV_REGISTRY_MIRROR") or None
                ),
                private_registry=(
                    os.environ.get("XRLENV_PRIVATE_REGISTRY") or None
                ),
            ),
        )

    # Spec-09 GC layer 3 reconciler (A3 / D15). Built after the
    # registry exists so it can iterate `registry.node_ids` on each
    # tick, but only enabled when the operator hasn't explicitly
    # opted out via ``gc_reconcile_interval_s=None`` (tests use that
    # to keep deterministic).
    from xrlenv.control.gc_reconciler import GCReconciler
    # Stage-3 P1 — the AIMD control loop (only when adaptive admission
    # is on). Created here, after registry + scheduler exist.
    aimd_loop = None
    if aimd_controller is not None:
        from xrlenv.control.aimd_loop import AimdControlLoop
        aimd_loop = AimdControlLoop(
            controller=aimd_controller,
            registry=registry,
            scheduler=scheduler,
            state=state,
            metrics=metrics_registry,
        )

    from xrlenv.control.raw_gc_reconciler import RawGCReconciler

    gc_reconciler: GCReconciler | None = None
    raw_gc_reconciler: RawGCReconciler | None = None
    if gc_reconcile_interval_s is not None and gc_reconcile_interval_s > 0:
        gc_reconciler = GCReconciler(
            registry=registry,
            coordinator=coordinator,
            state=state,
            interval_s=gc_reconcile_interval_s,
        )
        # P1.7.A.2 — parallel reconciler for raw containers. Same
        # interval; failures don't stop the case-1 reconciler and
        # vice versa.
        raw_gc_reconciler = RawGCReconciler(
            registry=registry,
            coordinator=raw_container_coordinator,
            interval_s=gc_reconcile_interval_s,
            metrics=metrics_registry,
            # Issue #18 fix #3: pass the StateStore so the reconciler
            # can sweep ghost ``raw_rollouts`` rows (rows in
            # ``acquiring`` / ``running`` whose in-memory session
            # disappeared — typically a control-plane restart, or any
            # path that bypassed the destroy ``finally``).
            state=state,
        )

    # Keep the SQLite WAL bounded (see wal_checkpointer.py). Only a sqlite-backed
    # store IN WAL MODE has a -wal to check-point: the in-memory store (tests) has
    # none, and a rollback-journal mode (TRUNCATE on a network FS — spec 20) has
    # none either, so don't even schedule the task there (it would just wake and
    # to-thread each interval to no-op). A checkpoint interval is a cadence, not a
    # capacity threshold, so a fixed default is fine — it reuses the reconcile
    # cadence knob, falling back to 60s when reconcilers are opted out
    # (``gc_reconcile_interval_s=None``).
    wal_checkpointer = None
    if (
        isinstance(state, SqliteStateStore)
        and getattr(state, "_journal_mode", "WAL") == "WAL"
    ):
        from xrlenv.control.wal_checkpointer import WalCheckpointer
        _ckpt_interval_s = (
            gc_reconcile_interval_s
            if gc_reconcile_interval_s and gc_reconcile_interval_s > 0
            else 60.0
        )
        wal_checkpointer = WalCheckpointer(
            state=state, interval_s=_ckpt_interval_s,
        )

    # Event-loop stall detector (2026-08-21). Surfaces a loop freeze — the
    # thing that lets a synchronous-I/O hiccup false-mark the fleet lost —
    # within seconds via a loud log + metric, so an operator learns of it long
    # before a 13-h outage. Detection only; the watchdog itself refuses to
    # mass-evict after a stall.
    from xrlenv.control.loop_monitor import LoopLagMonitor

    _max_loop_lag = [0.0]

    def _on_loop_stall(lag_s: float) -> None:
        metrics_registry.control_loop_stalls_total.inc()
        _max_loop_lag[0] = max(_max_loop_lag[0], lag_s)
        metrics_registry.control_loop_lag_seconds.set(_max_loop_lag[0])

    loop_monitor = LoopLagMonitor(on_stall=_on_loop_stall)

    # CP→node keepalive so an idle-but-healthy control plane is distinguishable
    # from one that has silently dropped a node — the node redials on keepalive
    # silence (defense-in-depth behind ``request_terminate``).
    from xrlenv.control.keepalive import ControlKeepaliveLoop

    control_keepalive = ControlKeepaliveLoop(registry=registry)

    runtime = DistributedRuntime(
        state=state,
        catalog=catalog,
        scheduler=scheduler,
        coordinator=coordinator,
        service=service,
        sink=sink,
        admission=admission,
        registry=registry,
        grpc_server=grpc_server,
        grpc_port=grpc_port,
        metrics=metrics_registry,
        build_coordinator=build_coordinator,
        metrics_server=metrics_server,
        run_dir_janitor=janitor,
        state_retention_janitor=retention_janitor,
        gc_reconciler=gc_reconciler,
        raw_gc_reconciler=raw_gc_reconciler,
        aimd_loop=aimd_loop,
        wal_checkpointer=wal_checkpointer,
        loop_monitor=loop_monitor,
        control_keepalive=control_keepalive,
        admin_server=admin,
    )
    runtime._node_connected = node_connected
    runtime._disconnect_tasks = disconnect_tasks
    return runtime


