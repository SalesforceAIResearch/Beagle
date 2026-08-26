"""Local single-process runtime bring-up (Slice 1).

For the laptop smoke test: instantiate state + catalog + Docker driver + node
agent + scheduler + coordinator + service in one process and hand back a
:class:`LocalRuntime` the example script can drive directly. In Slice 3 the
node agent and control plane will run as separate processes wired together by
the spec-21 bidi gRPC stream; the same :class:`LocalRuntime` shape will then
expose just the control-plane half.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, SkipValidation

import xrlenv
from xrlenv import paths
from xrlenv.backends.base import SandboxBackend
from xrlenv.backends.docker import DockerBackend, DockerBackendConfig, _default_stub_transport
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.build_coordinator import BuildCoordinator, NodeBudgetProvider
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.image_planner import NodeBudget
from xrlenv.control.node_builder import InProcessNodeBuilder
from xrlenv.control.raw_container_service import RawContainerCoordinator
from xrlenv.control.registry_resolver import resolver_from_env
from xrlenv.control.run_dir_janitor import RunDirJanitor
from xrlenv.control.scheduler import Scheduler
from xrlenv.control.service import CoordinatorRolloutService, RolloutService
from xrlenv.control.state import SqliteStateStore, StateStore
from xrlenv.control.template_catalog import TemplateCatalog
from xrlenv.control.trajectory_sink import PlatformJsonlSink, TrajectorySink
from xrlenv.node.agent import NodeAgent, NodeAgentConfig
from xrlenv.node.image_cache import ImageCacheConfig, ImageCacheManager
from xrlenv.node.image_pins import DEFAULT_PIN_FILE, load_image_pins
from xrlenv.node.trajectory_reader import JsonlTrajectoryReader
from xrlenv.observability.metrics import MetricsRegistry
from xrlenv.observability.server import MetricsServer

LOGGER = logging.getLogger(__name__)

# ``$XRLENV_HOME/runs`` (default ``~/.xrlenv/runs``); see :mod:`xrlenv.paths`.
DEFAULT_RUNS_ROOT = paths.runs_root()


class LocalRuntime(BaseModel):
    """Bag of singletons the example script and tests interact with.

    Several fields are :class:`typing.Protocol` types (``StateStore``,
    ``RolloutService``, ``TrajectorySink``); pydantic can't isinstance-check
    those at construction time, so they're wrapped in
    :class:`pydantic.SkipValidation` — the static-checker contract is
    preserved while runtime validation is bypassed for those slots.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    state: SkipValidation[StateStore]
    catalog: TemplateCatalog
    node: NodeAgent
    scheduler: Scheduler
    coordinator: RolloutCoordinator
    service: SkipValidation[RolloutService]
    sink: SkipValidation[TrajectorySink]
    admission: AdmissionQueue
    metrics: MetricsRegistry
    build_coordinator: BuildCoordinator
    """P1.6.b — drives ``xrlenv build apply`` against this in-process
    runtime. Always wired (the field is present even on test runtimes
    that don't exercise the build flow); cheap to construct."""
    metrics_server: SkipValidation[MetricsServer | None] = None
    run_dir_janitor: RunDirJanitor | None = None

    drain_timeout_s: float = 30.0

    _image_aimd_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    """Node-local adaptive pull-concurrency (AIMD) loop. Mirrors the task
    ``NodeGrpcLink`` starts on the daemon path so an in-process
    LocalRuntime gets the same load-aware pull behavior."""

    _image_sweep_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    """Issue #13 — periodic image-cache eviction sweep. Mirrors the
    background task ``NodeGrpcLink`` starts on the daemon path so an
    in-process LocalRuntime gets the same disk-pressure protection
    against steady-state overlay growth on cached images. ``None``
    until ``start()`` runs (or when the node has no ``_image_cache``
    wired, which only happens in stripped test fixtures)."""

    async def start(self) -> None:
        """Spin up background tasks. Idempotent."""
        await self.admission.start()
        if self.metrics_server is not None:
            self.metrics_server.start()
        if self.run_dir_janitor is not None:
            await self.run_dir_janitor.start()
        cache = getattr(self.node, "_image_cache", None)
        if cache is not None and self._image_sweep_task is None:
            self._image_sweep_task = asyncio.create_task(
                cache.run_sweep_loop(),
                name=f"image-sweep-{self.node.node_id}",
            )
        if cache is not None and self._image_aimd_task is None:
            self._image_aimd_task = asyncio.create_task(
                cache.run_pull_aimd_loop(),
                name=f"image-pull-aimd-{self.node.node_id}",
            )

    async def shutdown(self) -> None:
        """Drain in-flight rollouts (with grace timeout), cancel pending,
        then tear down background tasks and close the state store.

        Shutdown order:

        1. Stop accepting new admission requests.
        2. Wait up to ``drain_timeout_s`` for running rollouts to terminate.
        3. Cancel any still-pending waiters (their rows stay in sqlite for
           next-process recovery).
        4. Cancel deadline watchers and the admission worker.
        5. Close the sqlite connection.
        """
        import asyncio
        import time

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

        if self._image_aimd_task is not None and not self._image_aimd_task.done():
            self._image_aimd_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._image_aimd_task
        self._image_aimd_task = None
        if self._image_sweep_task is not None and not self._image_sweep_task.done():
            self._image_sweep_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._image_sweep_task
        self._image_sweep_task = None

        if self.run_dir_janitor is not None:
            await self.run_dir_janitor.shutdown()
        if self.metrics_server is not None:
            self.metrics_server.stop()

        if hasattr(self.state, "close"):
            self.state.close()


def build_local_runtime(
    *,
    node_id: str = "local-laptop",
    runs_root: Path | None = None,
    template_dirs: list[Path] | None = None,
    extra_backends: dict[str, SandboxBackend] | None = None,
    state: StateStore | None = None,
    state_db_path: Path | None = None,
    metrics: MetricsRegistry | None = None,
    metrics_host: str | None = None,
    metrics_port: int | None = None,
    run_dir_retention_days: int | None = 14,
    image_pin_file: Path | None = None,
    image_cache_config: ImageCacheConfig | None = None,
    skip_stale_node_sweep: bool = False,
) -> LocalRuntime:
    """Wire up the in-process control plane + node agent.

    Defaults:

    - Docker backend bound to the local Docker daemon (``DOCKER_HOST``).
    - Templates auto-loaded from ``xrlenv/templates`` inside the package.
    - ``SqliteStateStore`` at ``$XRLENV_HOME/state.db`` (slice 2 default).
      Pass ``state=InMemoryStateStore()`` for tests that want to skip disk.
    """
    runs_root = runs_root or DEFAULT_RUNS_ROOT
    runs_root.mkdir(parents=True, exist_ok=True)

    xrlenv_pkg_path = Path(xrlenv.__file__).resolve().parent

    from xrlenv.control.template_discovery import (
        find_entry_point_manifest_files,
        find_external_template_dir_manifests,
        find_plugin_manifest_files,
        find_plugin_root,
    )

    # D22 — discover external manifests once; their plug-in roots become
    # extra_plugin_roots on the docker backend, their manifest paths feed
    # the catalog. Single source of truth: env-var + entry-points.
    in_tree_root = find_plugin_root(xrlenv_pkg_path)
    external_manifests = (
        find_external_template_dir_manifests()
        + find_entry_point_manifest_files()
    )
    extra_plugin_roots = tuple(sorted({
        m.plugin_root for m in external_manifests
        if m.plugin_root is not None and m.plugin_root != in_tree_root
    }))
    if extra_plugin_roots:
        # D22 startup audit (see xrlenv/node/cli.py for the full
        # rationale). LocalRuntime mirrors the node-agent's signal
        # so an in-process consumer sees the same inventory.
        for idx, root in enumerate(extra_plugin_roots):
            LOGGER.info(
                "plugin_root.mounted host_path=%s container_target=/opt/xrlenv-extras/%d ro=true",
                root, idx,
            )
    docker_backend = DockerBackend(
        DockerBackendConfig(
            runs_root=runs_root,
            xrlenv_pkg_path=xrlenv_pkg_path,
            xrlenv_plugins_path=in_tree_root,
            extra_plugin_roots=extra_plugin_roots,
            stub_transport=_default_stub_transport(),
        ),
    )
    backends: dict[str, SandboxBackend] = {"docker": docker_backend}
    if extra_backends:
        backends.update(extra_backends)

    if state is None:
        db_path = state_db_path or (runs_root.parent / "state.db")
        state = SqliteStateStore(db_path)
    # Audit H1 (2026-05-01): sweep stale ``connected`` rows in the
    # nodes table left behind by a prior crashed control-plane
    # instance. See ``distributed_runtime._mark_stale_connected_nodes_lost``
    # for the rationale; LocalRuntime needs the same sweep so
    # in-process consumers calling ``Client.wait_for_nodes()``
    # don't get fooled by a previous process's stale rows.
    #
    # Audit P1.6-H1 (2026-05-05) follow-up: callers running short
    # one-shot CLI invocations (``xrlenv build apply --benchmark X``)
    # against a state.db shared with a live distributed control plane
    # would otherwise mark genuinely-connected nodes as ``lost``,
    # corrupting the live registry's persisted shadow until fresh
    # heartbeats repair it. The CLI sets ``skip_stale_node_sweep=True``
    # to opt out of the sweep; ``xrlenv up`` keeps the default so
    # cold-start recovery still works.
    if not skip_stale_node_sweep:
        from xrlenv.control.distributed_runtime import (
            _mark_stale_connected_nodes_lost,
        )

        _swept_stale_nodes = _mark_stale_connected_nodes_lost(state)
        if _swept_stale_nodes:
            LOGGER.warning(
                "startup-sweep: marked %d previously-``connected`` node row(s) "
                "as ``lost``", _swept_stale_nodes,
            )
    sink = PlatformJsonlSink(runs_root)
    # Spec 19 §"Image and asset supply chain": wire the catalog to the
    # local Docker daemon so tag-only image refs get rewritten to
    # ``image@sha256:...`` at register. Audit hook routes through the
    # state store so ``template.registered`` + ``mount.denied`` rows
    # land in the spec-19 audit table.
    def _audit(kind: str, payload: dict[str, Any]) -> None:
        state.append_audit(kind, payload=payload)

    catalog = TemplateCatalog(
        digest_resolver=docker_backend.resolve_image_digest,
        audit_callback=_audit,
    )

    if template_dirs is None:
        template_dirs = [xrlenv_pkg_path / "templates"]
    for d in template_dirs:
        if d.exists():
            registered = catalog.register_dir(d)
            LOGGER.info(
                "registered %d template(s) from %s: %s",
                len(registered),
                d,
                [m.name for m in registered],
            )

    # In-tree plug-ins under xrlenv_plugins/ + the external manifests
    # discovered above (XRLENV_TEMPLATE_DIRS + xrlenv.benchmarks
    # entry-points). All register via the same register_paths code
    # path; their plug-in roots already wired into extra_plugin_roots.
    plugin_manifests = find_plugin_manifest_files(xrlenv_pkg_path)
    plugin_manifests.extend(m.manifest_path for m in external_manifests)
    if plugin_manifests:
        registered = catalog.register_paths(plugin_manifests)
        LOGGER.info(
            "registered %d plug-in template(s): %s",
            len(registered),
            [m.name for m in registered],
        )

    pin_set = load_image_pins(image_pin_file or DEFAULT_PIN_FILE)
    image_cache = ImageCacheManager(
        backend=docker_backend,
        pins=pin_set,
        config=image_cache_config,
    )
    trajectory_reader = JsonlTrajectoryReader(runs_root)
    node = NodeAgent(
        NodeAgentConfig(node_id=node_id, backends=backends),
        image_cache=image_cache,
        trajectory_reader=trajectory_reader,
    )
    scheduler = Scheduler([node], catalog=catalog, state=state)
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
    # Crash-recovery sweep: any rollout left in a transient state by
    # a previous process (or by ``_terminate`` hitting one of the
    # bidi timeouts) gets force-sealed before the new coordinator
    # starts accepting RPCs. Without this, the dashboard carries
    # stale ``cancelling`` / ``finishing`` rows across restarts and
    # only direct SQL fixed them.
    coordinator.sweep_stuck_transients()
    raw_container_coordinator = RawContainerCoordinator(
        scheduler=scheduler, state=state,
        metrics=metrics_registry,
        # Freshness model: resolve registry tag -> content digest per
        # acquire (kill-switch XRLENV_REGISTRY_DIGEST_RESOLVE=0).
        digest_resolver=resolver_from_env(),
    )
    # Wire the raw-container coordinator's session list into the scheduler's
    # capacity gate. Without this, raw containers (which don't pass through
    # ``state.list_sandboxes()``) would be invisible to ``_gather_cluster_load``
    # and the scheduler would over-place — silent over-subscription that
    # breaks the operator's parallelism contract. See P1.7.A raw-container
    # ``_pending`` leak fix + ``Scheduler.set_raw_session_provider`` docstring.
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
        )

    janitor: RunDirJanitor | None = None
    if run_dir_retention_days is not None:
        janitor = RunDirJanitor(
            runs_root=runs_root,
            retention_days=run_dir_retention_days,
        )

    async def _local_ensure_present(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        # LocalRuntime has a single in-process node agent; node_id is
        # ignored (no multi-node lookup). Returns the same shape the
        # remote-node transport's ensure_present uses.
        cache = node.image_cache if node is not None else None
        if cache is None:
            return ("failed", "local node has no image_cache")
        try:
            # Build-apply / prefetch dispatch → load-aware prefetch lane.
            await cache.ensure_present(
                image_ref, deadline_s=timeout_s, prefetch=True,
            )
        except Exception as exc:
            return ("failed", f"{type(exc).__name__}: {exc}")
        return ("ok", None)

    # Source-build dispatch hook: a per-runtime GitSourceBuilder
    # instance shared across applies. LocalRuntime always has the
    # local docker daemon; the builder lazy-constructs its
    # docker-py client on first build.
    from xrlenv.node.source_builder import GitSourceBuilder
    _local_source_builder = GitSourceBuilder()

    async def _local_build_image(
        node_id: str, image_ref: str, source: Any,
        timeout_s: float, labels: dict[str, str],
        skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        return await _local_source_builder.build(
            image_ref=image_ref, source=source,
            timeout_s=timeout_s, labels=labels,
            skip_if_present=skip_if_present,
        )

    async def _local_build_push(
        node_id: str, image_ref: str, source: Any,
        timeout_s: float, labels: dict[str, str],
    ) -> tuple[str, str | None, str | None]:
        # ``xrlenv build push`` on the local node: build AND push image_ref to
        # the registry it encodes, returning the pushed digest.
        result = await _local_source_builder.build_and_push(
            image_ref=image_ref, source=source,
            timeout_s=timeout_s, labels=labels,
            check_registry_first=True,
        )
        if result.status == "ok":
            return ("ok", None, result.repo_digest)
        return ("failed", result.error, None)

    build_coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(node_agent=node),
        budget_provider=_LocalNodeBudgetProvider(node=node),
        ensure_present_fn=_local_ensure_present,
        build_image_fn=_local_build_image,
        build_push_fn=_local_build_push,
    )

    return LocalRuntime(
        state=state,
        catalog=catalog,
        node=node,
        scheduler=scheduler,
        coordinator=coordinator,
        service=service,
        sink=sink,
        admission=admission,
        metrics=metrics_registry,
        build_coordinator=build_coordinator,
        metrics_server=metrics_server,
        run_dir_janitor=janitor,
    )


class _LocalNodeBudgetProvider(NodeBudgetProvider):
    """Compute per-node budgets from the in-process node-agent's
    ``report_images()`` snapshot.

    LocalRuntime has exactly one node; its budget is
    ``free_disk_bytes - reservations``. Distributed runtimes (P1.6.c)
    will replace this with a heartbeat-derived view.
    """

    def __init__(self, *, node: NodeAgent) -> None:
        self._node = node

    async def get_budgets(
        self,
        *,
        reserved_runtime_gb: int,
        buffer_gb: int,
        cap_per_node_gb: int | None,
    ) -> list[NodeBudget]:
        report = await self._node.report_images()
        reserved_bytes = (reserved_runtime_gb + buffer_gb) * 1024**3
        available = max(0, report.free_disk_bytes - reserved_bytes)
        if cap_per_node_gb is not None:
            available = min(available, cap_per_node_gb * 1024**3)
        return [NodeBudget(
            node_id=self._node._config.node_id,  # type: ignore[arg-type]
            available_bytes=available,
        )]
