"""``NodeTransport`` — the abstraction the coordinator + scheduler call into.

Two implementations:

- **In-process** (``InProcessNodeTransport``, Slices 1+2): wraps a local
  :class:`xrlenv.node.NodeAgent` and forwards method calls directly. The
  coordinator and the node-agent run in the same Python process; ideal for
  laptop iteration and unit tests.

- **gRPC** (``GrpcNodeTransport``, Slice 3): the control plane sees a remote
  node-agent through a per-connection client that ships commands down the
  spec-21 bidi stream and awaits matching ``CommandReply`` responses.

Both implementations satisfy the same Protocol so the coordinator never
branches on which is in play. New transports (e.g. an in-tree fake for tests)
slot in by implementing the same surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from xrlenv.backends.base import (
    ExecResult,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    RuntimeLimits,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.backends.egress import EgressAllowlist
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.types import Trajectory

if TYPE_CHECKING:
    # Runtime-only import would close a cycle: trajectory_reader pulls in
    # xrlenv.control.trajectory_sink → xrlenv.control.__init__ → coordinator
    # → admission → scheduler → node_transport (us). Since
    # `FetchRangeKind` is used only as a string annotation under
    # `from __future__ import annotations`, we don't need it at runtime.
    from xrlenv.node.image_cache import (
        EvictOutcome,
        ImageQueryResult,
        NodeImageReport,
    )
    from xrlenv.node.trajectory_reader import FetchRangeKind


class NodeTransport(Protocol):
    """Surface the coordinator + scheduler use to drive a node-agent.

    The methods are intentionally a 1:1 mirror of :class:`NodeAgent` so the
    in-process transport is a thin pass-through; only the gRPC transport adds
    real serialization + reconnect handling on top.

    ``node_id`` is exposed as a read-only ``@property`` so both NodeAgent
    (which delegates to ``self._config.node_id``) and RemoteNodeTransport
    (plain attribute) satisfy the Protocol without variance issues.
    """

    @property
    def node_id(self) -> str: ...

    def supported_backends(self) -> list[str]: ...

    def supported_runtimes(self) -> list[str]:
        """OCI runtimes this node's docker daemon has registered (§5.3).

        Parallel to :meth:`supported_backends`. The scheduler filters
        candidate nodes by this when an acquire requests a non-default
        ``container_runtime``. Implementations that predate this method
        (or non-docker stand-ins) should be treated by callers as
        advertising only ``["runc"]``.
        """
        return ["runc"]

    def default_runtime(self) -> str:
        """The node's docker daemon default runtime (§5.3, §9). ``"runc"``
        for implementations that don't advertise it."""
        return "runc"

    def hardware(self) -> HardwareInfo: ...

    def disk_state(self) -> tuple[int, int]:
        """Issue #14 — most recent ``(free_bytes, total_bytes)`` known
        for the node's disk pool. The remote (gRPC) transport returns
        the last heartbeat sample; the in-process transport probes the
        backend on demand. ``(0, 0)`` is the documented "unknown"
        sentinel — callers (scheduler placement gate, admin pressure
        indicator) treat it as healthy until a real value arrives.
        """
        ...

    def seconds_since_last_command_timeout(self) -> float | None:
        """Issue #18 fix (Ask #2) — elapsed seconds since the last
        command reply-timeout against this node, or ``None`` if the
        node has never timed out.

        The remote (gRPC) transport records a timeout whenever
        ``_send_and_wait`` hits its ceiling waiting for a
        ``CommandReply`` — a signal the node-agent is wedged or
        overloaded even though it may still be heartbeating. The
        in-process transport always returns ``None`` (same-process
        calls have no wire timeout). The scheduler's placement gate
        excludes a node whose last timeout is within a recent
        cooldown window so a degraded node stops attracting traffic.
        """
        ...

    async def create_sandbox(
        self,
        *,
        rollout_id: str,
        backend: str,
        template: TemplateRef,
        resources: ResourceSpec,
        network_policy: NetworkPolicy,
        stub_request_timeout_s: float | None = None,
    ) -> SandboxHandle:
        """A5 / D17 stage 1: ``stub_request_timeout_s`` is the
        coordinator-derived HTTP cap that the node stages on the
        new per-sandbox record BEFORE any stub-touching call.
        ``None`` falls back to the node's
        :attr:`NodeAgentConfig.stub_request_timeout_s` default (1 h
        safety net). Audit response: this kwarg replaces the earlier
        ``_xrlenv_http_timeout_s`` injection through ``env_setup``'s
        ``init_params``, which got bypassed by manifests with
        ``init_cmd`` (``run_in_sandbox`` built the ``StubClient``
        before ``env_setup`` could stage the cap).
        """
        ...

    async def destroy_sandbox(self, sb: SandboxHandle) -> None: ...

    async def env_setup(
        self,
        sb: SandboxHandle,
        *,
        adapter_module: str,
        adapter_class: str,
        init_params: dict[str, Any],
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """A5 / D17 stage 2 (P1.2.b): ``request_timeout_s`` is the
        per-call HTTP cap for *this* setup call. ``None`` falls back
        to the per-sandbox cap staged at create_sandbox time
        (stage 1). The control plane derives the per-call value from
        the manifest's ``setup_timeout_s + buffer`` so a hung setup
        surfaces in roughly that budget rather than the wider
        max-phase cap.
        """
        ...

    async def env_step(
        self,
        sb: SandboxHandle,
        action: Any,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """A5 / D17 stage 2: per-call HTTP cap derived from the
        manifest's ``step_timeout_s + buffer``."""
        ...

    async def env_teardown(
        self,
        sb: SandboxHandle,
        *,
        request_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """A5 / D17 stage 2: per-call HTTP cap derived from the
        manifest's ``teardown_timeout_s + buffer``."""
        ...

    async def run_in_sandbox(
        self,
        sb: SandboxHandle,
        cmd: list[str],
        *,
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult: ...

    async def put_archive(
        self,
        sb: SandboxHandle,
        target_dir: str,
        tarball: bytes,
        *,
        clean_target: bool = False,
    ) -> None:
        """D12 stage 1: extract a tar(.gz) archive into ``target_dir``.

        Used by :func:`xrlenv.control.reward.compute_in_sandbox_final_reward`
        to inject per-task grader assets into the sandbox at reward time —
        not at image build time. The agent's step() loop runs without
        ``/tests/`` (or whatever target the verifier uses) in the
        filesystem, closing the H1 grader-isolation hole.

        ``clean_target=True`` requests a root-backed ``rm -rf
        <target_dir>`` before extraction so agent-created residue
        cannot survive into the verifier phase. The backend MUST run
        the wipe with verifier authority and raise on non-zero exit.
        """
        ...

    async def stats(self, sb: SandboxHandle) -> ResourceUsage: ...

    async def list_sandbox_ids(self, *, backend: str | None = None) -> list[str]:
        """A3 / D15 (P1.1) — return the IDs of sandboxes the node-agent
        currently tracks.

        Used by :class:`~xrlenv.control.gc_reconciler.GCReconciler`
        as the node side of the spec-09 GC layer 3 reverse query.
        Both the in-process :class:`~xrlenv.node.NodeAgent` and the
        gRPC :class:`~xrlenv.control.grpc_endpoint.RemoteNodeTransport`
        satisfy this method.
        """
        ...

    async def query_image(self, image: str) -> ImageQueryResult:
        """A1 / D18+D19 (P1.2) — does this node have the given image?

        The reply carries presence + best-effort digest +
        per-node-monotonic last-used timestamp. Used by the
        scheduler for image-affinity scoring (D18) and by the
        coordinator for pre-flight checks (D19) before sending
        ``CreateSandboxCommand``.
        """
        ...

    async def report_images(
        self, *, include_shared_size: bool = False,
    ) -> NodeImageReport:
        """B7.6 (P1.2.c) — full per-node image cache snapshot.

        Asked on demand by the admin ``/images`` route to render the
        cluster-wide tier histogram + per-node free-disk view. Cheap on
        the node side (in-memory cache state + one ``free_disk_bytes``
        backend round-trip + one ``list_images`` for sizes).

        ``include_shared_size`` requests per-image layer-sharing data via
        Docker ``system df`` — slow on a large catalog, so it defaults to
        ``False`` (the /images view doesn't need it) and only ``xrlenv
        build calibrate`` sets it.
        """
        ...

    async def evict_image(
        self, *, image_ref: str, force: bool = False,
        timeout_s: float = 30.0,
    ) -> EvictOutcome:
        """Operator-driven node-cache eviction (``xrlenv images evict``).

        Sends one :class:`EvictImageCommand` to this node and waits for
        its reply. The node matches ``image_ref`` registry-agnostically
        against the tags it holds and removes the matching image(s),
        skipping in-use / pinned images unless ``force``. The escape
        hatch for the mutable-tag staleness problem (rebuild + re-push
        under the same tag → nodes never re-pull on their own).
        """
        ...

    async def fetch_trajectory(
        self,
        rollout_id: str,
        *,
        range_kind: FetchRangeKind = "whole",
        step_start: int = 0,
        step_end: int | None = None,
    ) -> Trajectory:
        """Spec 17 §"Fetch": pull a sealed trajectory from the owning node.

        Both InProcess (local NodeAgent) and Remote (RemoteNodeTransport
        over the bidi stream) satisfy this method. The control-plane
        :class:`xrlenv.control.trajectory_cache.TrajectoryCache` calls it
        on cache miss.
        """
        ...

    # P1.7.A.1 — raw container session for case 2/3 evaluation harnesses.
    # Both transports MUST implement these (NodeAgent already does via
    # the methods added in 2026-05-06; RemoteNodeTransport adds them as
    # spec-21 wire calls). The methods bypass the in-sandbox stub layer.

    async def acquire_container(
        self,
        *,
        rollout_id: str,
        backend: str,
        image: str,
        command: list[str] | None = None,
        entrypoint: list[str] | None = None,
        user: str | None = None,
        cap_add: list[str] | None = None,
        devices: list[str] | None = None,
        privileged: bool = False,
        network_mode: str | None = None,
        binds: list[str] | None = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        acquire_timeout_s: float | None = None,
        resources: ResourceSpec | None = None,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> Any:
        """Spawn a raw container scoped to ``rollout_id``.

        Returns a record-shaped object carrying ``container_id`` +
        ``container_name``. The full type is
        ``xrlenv.node.raw_container.RawContainerRecord`` for the
        in-process transport; the remote transport returns a
        compatible duck-typed shape (same field names).

        ``ensure_image_present`` (P1.7.B.2): default True — node
        runs ``ImageCacheManager.ensure_present(image)`` to pull /
        build / no-op as appropriate. False reverts to the strict
        legacy contract (raise if image absent locally), preserved
        as opt-in for deterministic-eval consumers.

        ``acquire_timeout_s`` (issue #12): wire-level deadline for
        the AcquireContainer round-trip. ``None`` uses the transport
        default (600 s, aligned with
        :py:attr:`ImageCacheConfig.default_pull_timeout_s` so a
        legitimate cold-pull doesn't race the wire). Pass a larger
        value (e.g. 1800 s) when acquiring a known-huge image
        (SWE-bench Pro 15 GB tags, multi-GB GPU images) on a slow
        link. The in-process transport ignores this kwarg (no wire
        timeout in same-process calls).
        """
        ...

    async def container_exec(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        backend: str = "docker",
    ) -> dict[str, Any]:
        """Direct docker exec; returns ``{exit_code, stdout, stderr,
        timed_out}``. Batched (full output buffered before reply) —
        P1.7.A.2 will add a streaming variant for swebench's 30+
        min test runs.
        """
        ...

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
        backend: str = "docker",
    ) -> None:
        """Restrict a running container's egress to ``allowlist`` (spec 07).
        The node compiles the allowlist and installs the iptables rules into
        the container's netns."""
        ...

    async def destroy_container(
        self,
        *,
        rollout_id: str,
        container_id: str,
        force: bool = True,
        backend: str = "docker",
    ) -> None:
        """``docker rm -f`` semantics. Idempotent on missing
        containers (the harness may have removed it itself before
        calling destroy)."""
        ...

    # P1.7.C.2 — multi-service compose projects.

    async def acquire_compose_project(
        self,
        *,
        rollout_id: str,
        project_name: str,
        compose_yaml: str,
        images: list[str] | None = None,
        main_service: str = "main",
        up_timeout_s: float | None = None,
        backend: str = "docker",
    ) -> Any:
        """Bring up a whole compose project on this node (``docker compose up
        --wait``) and register every member ↔ ``rollout_id``.

        Returns a record-shaped object with ``project_name`` / ``project_dir`` /
        ``main_container_id`` / ``main_container_name`` / ``service_container_ids``
        (+ a ``member_container_ids`` property) — the full type is
        :class:`xrlenv.node.raw_compose.ComposeProjectRecord` for the in-process
        transport; the remote transport returns a compatible duck-typed shape. The
        compose document is already CP-vetted + digest-pinned + reserved-label
        stamped (§2/§2.3); the node just runs it."""
        ...

    async def destroy_compose_project(
        self,
        *,
        rollout_id: str,
        project_name: str,
        force: bool = True,
        backend: str = "docker",
    ) -> None:
        """``docker compose down`` the whole project + deregister every member.
        Strict node-side (a failed down surfaces as an error, so the coordinator
        keeps the project reserved until a confirmed teardown — invariant 2)."""
        ...

    # P1.7.A.2 — raw container archives.

    async def container_put_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        target_dir: str,
        tarball: bytes,
        backend: str = "docker",
    ) -> None:
        """Extract ``tarball`` into ``target_dir`` inside the
        container — symmetric with ``docker container put_archive``.
        ``tarball`` may be plain or gzipped tar; docker auto-
        detects."""
        ...

    async def container_get_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
        backend: str = "docker",
    ) -> bytes:
        """Tar up ``source_path`` inside the container and return
        the bytes — symmetric with ``docker container get_archive``.
        Bounded by ``DEFAULT_MAX_MESSAGE_BYTES``; consumers
        expecting >100MB tarballs need their own chunking."""
        ...

    async def list_raw_container_ids(
        self, *, backend: str = "docker",
    ) -> list[str]:
        """Reverse query for the raw-GC reconciler. Returns the
        container_ids on this node carrying the
        ``xrlenv.session_kind=raw`` label."""
        ...

    async def list_managed_container_info(
        self, *, backend: str = "docker",
    ) -> list[tuple[str, str, str, str]]:
        """Audit H11 — EVERY xrlenv-managed container on this node WITH labels:
        ``(container_id, rollout_id, compose_project, session_kind)`` — including
        compose SIDECARS the raw-only listing omits. readopt-on-connect uses it to
        quarantine a node with a sidecar-only compose survivor."""
        ...

    async def force_destroy_raw_container(
        self, *, container_id: str,
    ) -> None:
        """Privileged ``docker rm -f`` that bypasses the
        per-rollout ownership check. Used **only** by the
        raw-GC reconciler for node-only orphan cleanup. NOT
        consumer-reachable: there's no matching
        ``rollout_control.proto`` RPC."""
        ...

    def container_exec_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        backend: str = "docker",
    ) -> Any:  # AsyncIterator[dict[str, Any]] — Any to keep the
        # Protocol's runtime-checkable surface light (real
        # asyncio.AsyncIterator typing pulls in TypeVar gymnastics
        # the existing transport methods don't carry).
        """Streaming exec — async iterator yielding per-chunk
        dicts. Each chunk dict has the ``ContainerExecChunk``
        shape (stdout / stderr / done / exit_code / timed_out).
        Consumer iterates with ``async for`` until ``done=True``."""
        ...


__all__ = ["NodeTransport"]
