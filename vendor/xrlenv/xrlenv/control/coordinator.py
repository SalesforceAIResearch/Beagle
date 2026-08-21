"""RolloutCoordinator — drives the spec-02 startup pipeline (Slice 1).

The coordinator is the only place RL semantics live on the control plane side.
It walks each rollout through the canonical startup sequence (resolve →
ensure-images → create + capacity → init → setup → first obs), mediates each
``step`` call between the consumer SDK and the in-sandbox EnvAdapter, and
seals the trajectory at finish.

Slice 1 implements:
- start_rollout (resolve → create → setup → first obs)
- step
- finish (teardown → destroy → seal)
- cancel (teardown best-effort → destroy → seal partial)

Slices 2+ add: deadlines, idle TTL + heartbeat, idempotency on retry, group
primitives, capacity reservation, scheduler bin-packing.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from contextlib import suppress
from typing import Any

from xrlenv.backends.base import NetworkPolicy, SandboxHandle
from xrlenv.control.admission import DEFAULT_QUEUE_TIMEOUT_S, AdmissionQueue
from xrlenv.control.deadlines import DeadlineWatcher
from xrlenv.control.defaults import DEFAULT_BACKEND, DEFAULT_NETWORK
from xrlenv.control.idle_ttl import IdleTtlWatcher
from xrlenv.control.instance_resolver import (
    InstanceResolverImportError,
    apply_to_manifest,
    load_resolver,
)
from xrlenv.control.node_transport import NodeTransport
from xrlenv.control.reward import (
    RewardComputationError,
    compute_in_sandbox_final_reward,
)
from xrlenv.control.scheduler import Placement, Scheduler
from xrlenv.control.state import (
    PendingRolloutRecord,  # noqa: F401 — re-exported transitively for tests
    RolloutRecord,
    SandboxRecord,
    StateStore,
    new_id,
)
from xrlenv.control.template_catalog import TemplateCatalog, TemplateManifest
from xrlenv.control.trajectory_sink import TrajectorySink
from xrlenv.errors import (
    ImageMissingOnNode,
    RolloutCancelled,
    RolloutFailed,
    RolloutTruncated,
    TemplateUnknown,
)
from xrlenv.observability.metrics import MetricsRegistry
from xrlenv.observability.tracing import get_tracer
from xrlenv.types import (
    Action,
    CancelGroupReport,
    Deadline,
    Observation,
    RolloutStatus,
    Step,
    StepResult,
    Trajectory,
)

LOGGER = logging.getLogger(__name__)


# Per-call bidi-RPC timeouts on the rollout terminate path. Without
# these, a wedged node (docker daemon busy, in-sandbox stub blocked)
# pinned rollouts in ``cancelling`` / ``finishing`` indefinitely; the
# only recovery was SQL surgery against ``state.db``. Defaults are
# deliberately generous so legitimately-slow operations on saturated
# nodes don't trip the timeout — only true unresponsive cases do.
# Override via ``RolloutCoordinator(..., teardown_timeout_s=...)``
# etc. when running against pathologically slow nodes.
_TEARDOWN_TIMEOUT_S: float = 90.0
_VERIFIER_FETCH_TIMEOUT_S: float = 60.0
_DESTROY_TIMEOUT_S: float = 180.0

# D17 stage 1 (P1.1) HTTP-cap derivation. The cap is passed to the
# node via ``CreateSandboxCommand.stub_request_timeout_s`` so the
# StubClient is built with a tighter ``aiohttp.ClientTimeout`` than
# the 1 h safety-net default before any stub-touching call. Audit
# response: the earlier ``init_params``-injection path got bypassed
# by manifests with ``init_cmd``.
_HTTP_TIMEOUT_BUFFER_S: float = 60.0
"""Headroom over the manifest's max inner timeout. Covers stub
serialization round-trip + transient queueing on a busy node so the
HTTP cap never bites a legitimate inner timeout. 60 s on top of the
max of (init, setup, step, teardown) was chosen to be wider than the
slowest realistic non-pathological round-trip (~10 s on a saturated
node) but tight enough to surface an absent-stub failure inside one
operator-noticeable cycle, not the 1 h default."""

# Restart sweep: any rollout that's still in a transient
# (``cancelling`` / ``finishing`` / ``starting``) state and hasn't
# been touched within this many seconds at coordinator startup
# is force-sealed. Covers process-crash recovery and "the previous
# ``xrlenv up`` was killed mid-cancel" cases. Long enough that a
# legitimately-slow in-flight terminate finishing on the new
# coordinator's clock isn't yanked out from under it.
_RESTART_SWEEP_GRACE_S: float = 300.0


class RolloutCoordinator:
    """Drives rollout lifecycle for the consumer-facing service."""

    def __init__(
        self,
        *,
        catalog: TemplateCatalog,
        scheduler: Scheduler,
        state: StateStore,
        trajectory_sink: TrajectorySink | None = None,
        admission: AdmissionQueue | None = None,
        deadline_watcher: DeadlineWatcher | None = None,
        idle_ttl_watcher: IdleTtlWatcher | None = None,
        default_idle_ttl_s: float = 120.0,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._catalog = catalog
        self._scheduler = scheduler
        self._state = state
        self._sink = trajectory_sink
        self._admission = admission
        self._metrics = metrics
        # The watcher needs a coroutine that drives truncation; we bind here
        # so DeadlineWatcher only knows about ``rollout_id`` and ``reason``.
        self._deadlines = deadline_watcher or DeadlineWatcher(self._truncate_callback)
        # Idle-TTL watcher reaps abandoned rollouts (spec 02 §"Idle TTL").
        # Reap path is the same _truncate_callback the deadline watcher uses;
        # the only difference is the reason string carried into seal.
        self._idle = idle_ttl_watcher or IdleTtlWatcher(self._truncate_callback)
        self._default_idle_ttl_s = default_idle_ttl_s
        # Per-rollout async lock so concurrent steps on the same rollout don't
        # interleave (per spec-02 lifecycle: at most one step in flight).
        self._rollout_locks: dict[str, asyncio.Lock] = {}
        # Per-rollout deadlines snapshot — used to compute the truncate clock
        # without re-reading the manifest each time.
        self._rollout_deadlines: dict[str, float] = {}
        # Spec 06 Pattern A: one resolver instance per outer manifest. The
        # coordinator pays the import + index-load cost once, then every
        # rollout re-uses the cached resolver.
        self._resolver_cache: dict[str, Any] = {}
        # D12 stage 1: per-rollout verifier-asset uploads. The resolver
        # builds these tarballs at start_rollout time; the reward path
        # extracts them into the sandbox immediately before running
        # reward.cmd. In-memory only (best-effort across coordinator
        # restarts; rollouts in flight at restart already lose their
        # in-memory state via other paths).
        self._verifier_uploads: dict[str, tuple[Any, ...]] = {}

    def sweep_stuck_transients(
        self, *, grace_s: float = _RESTART_SWEEP_GRACE_S,
    ) -> int:
        """Promote any rollout stuck in a transient state at startup.

        Call once from the runtime's startup path (after the
        coordinator + state-store are constructed but before nodes
        attach). Walks ``state.list_rollouts()`` for rows with status
        in ``{cancelling, finishing, starting}`` whose
        ``last_touched_at`` is older than ``grace_s``, and force-seals
        them:

        - ``cancelling`` → ``cancelled`` + reason ``/swept_at_startup``
        - ``finishing``  → ``failed``    + reason ``/swept_at_startup``
        - ``starting``   → ``failed``    + reason ``/swept_at_startup``

        Without this sweep, a process crash mid-``_terminate`` (or a
        wedged node that beat the per-RPC timeouts) would leave rows
        pinned in transient state across restarts — only SQL surgery
        could clear them. The grace window is generous so a
        legitimately-in-flight terminate that's still finishing on
        the new coordinator's clock isn't yanked out from under it.

        Returns the count of rows swept (for the audit log).
        """
        cutoff = time.time() - grace_s
        transient = (
            RolloutStatus.CANCELLING,
            RolloutStatus.FINISHING,
            RolloutStatus.STARTING,
        )
        terminal_for: dict[RolloutStatus, RolloutStatus] = {
            RolloutStatus.CANCELLING: RolloutStatus.CANCELLED,
            RolloutStatus.FINISHING:  RolloutStatus.FAILED,
            RolloutStatus.STARTING:   RolloutStatus.FAILED,
        }
        swept = 0
        for record in self._state.list_rollouts():
            if record.status not in transient:
                continue
            if record.last_touched_at >= cutoff:
                continue
            # Capture terminal mapping + values BEFORE update_rollout —
            # InMemoryStateStore returns live references, so the
            # subsequent ``record.status`` would already be the
            # terminal value and ``terminal_for[...]`` would KeyError.
            old_status = record.status
            new_status = terminal_for[old_status]
            old_touched_at = record.last_touched_at
            new_reason = (record.reason or "") + "/swept_at_startup"
            self._state.update_rollout(
                record.rollout_id,
                status=new_status,
                reason=new_reason,
            )
            self._state.append_event(
                f"rollout.{new_status.value}",
                rollout_id=record.rollout_id,
                payload={"reason": new_reason, "swept": True},
            )
            # Mirror the seal into the per-rollout coordinator.log so
            # the admin's lifecycle-bounds reader sees a terminal
            # event and stops rendering the rollout as "live".
            # Best-effort — the sink may not be open for this rollout
            # (re-opening at sweep time is unsafe; we only have the
            # state record). When the sink is available it falls
            # through to ``record_event``'s runs-root walk path so it
            # finds the persisted run dir from the previous process.
            if self._sink is not None:
                with suppress(Exception):
                    self._sink.record_event(
                        record.rollout_id,
                        f"rollout.{new_status.value}",
                        {
                            "status": new_status.value,
                            "reason": new_reason,
                            "swept": True,
                        },
                    )
            LOGGER.warning(
                "startup-sweep: rollout=%s force-sealed %s → %s "
                "(stuck %.0fs, last_touched_at=%s)",
                record.rollout_id,
                old_status.value,
                new_status.value,
                time.time() - old_touched_at,
                old_touched_at,
            )
            swept += 1
        if swept:
            LOGGER.warning(
                "startup-sweep: force-sealed %d stuck transient rollout(s)",
                swept,
            )
        return swept

    @property
    def deadline_watcher(self) -> DeadlineWatcher:
        return self._deadlines

    @property
    def idle_ttl_watcher(self) -> IdleTtlWatcher:
        return self._idle

    @property
    def catalog(self) -> TemplateCatalog:
        return self._catalog

    # ── Public API (called by RolloutService) ────────────────────────────────

    async def start_rollout(
        self,
        *,
        template_name: str,
        init: dict[str, Any],
        request_id: str | None = None,
        task_key: str | None = None,
        group_id: str | None = None,
        owner_id: str = "default",
        deadline: Deadline | None = None,
        backend: str | None = None,
        network: NetworkPolicy | None = None,
    ) -> tuple[str, Observation]:
        with get_tracer().start_as_current_span(
            "xrlenv.coordinator.dispatch_rollout",
            attributes={
                "template": template_name,
                "task_key": task_key or "",
                "group_id": group_id or "",
                "owner_id": owner_id,
                "deadline_s": deadline.hard_s if deadline else -1.0,
            },
        ):
            return await self._start_rollout_impl(
                template_name=template_name,
                init=init,
                request_id=request_id,
                task_key=task_key,
                group_id=group_id,
                owner_id=owner_id,
                deadline=deadline,
                backend=backend,
                network=network,
            )

    async def _start_rollout_impl(
        self,
        *,
        template_name: str,
        init: dict[str, Any],
        request_id: str | None,
        task_key: str | None,
        group_id: str | None,
        owner_id: str,
        deadline: Deadline | None,
        backend: str | None,
        network: NetworkPolicy | None,
    ) -> tuple[str, Observation]:
        # 1. Idempotency check (spec 02 §"Idempotency at every layer").
        # Audit M1: the idempotency key is namespaced by owner_id so a caller
        # cannot reuse / guess another tenant's request_id and be handed back
        # that tenant's rollout_id + observation. Each owner gets an isolated
        # request_id space; the server-stamped owner_id (never client-supplied)
        # is the namespace, so the owner boundary holds at the idempotency
        # layer too. NUL is a safe separator — request_ids are urlsafe text.
        idem_key = f"{owner_id}\x00{request_id}" if request_id is not None else None
        if idem_key is not None:
            existing = self._state.lookup_idempotent(idem_key)
            if existing is not None:
                record = self._state.get_rollout(existing)
                obs = (
                    record.steps[-1].obs
                    if record.steps
                    else record.metadata.get("init_obs")
                )
                return existing, obs

        # 2. Resolve template
        try:
            manifest = self._catalog.get(template_name)
        except KeyError as exc:
            raise TemplateUnknown(str(exc)) from exc

        # 2.5 Pattern A (spec 06): when the template carries an instance
        # resolver and the consumer named an instance_id in init, overlay
        # the per-instance image / resources / mounts / init_params onto
        # an effective manifest before placement so the scheduler sizes
        # against the right resources. The resolver may also supply a
        # per-task ``network`` policy — that wins over the request's
        # network value (the benchmark author owns the per-task
        # hermetic-vs-open call).
        manifest, resolved_network, resolved_verifier_uploads = (
            self._maybe_resolve_instance(manifest, init)
        )

        # 3. Acquire a placement — admission queue if wired, direct otherwise.
        # The admission queue raises CapacityExhausted on its own queue_timeout.
        # ``backend`` is per-rollout user policy (run-config or per-call kwarg);
        # the scheduler / capacity layer uses it to filter eligible nodes.
        # Falls back to the platform default when the request didn't carry one.
        effective_backend = backend or DEFAULT_BACKEND
        placement = await self._acquire_placement(
            manifest=manifest,
            task_key=task_key,
            request_id=request_id,
            group_id=group_id,
            owner_id=owner_id,
            init=init,
            deadline=deadline,
            backend=effective_backend,
        )
        node, backend = placement.node, placement.backend

        rollout_id = new_id()
        record = RolloutRecord(
            rollout_id=rollout_id,
            template=manifest.name,
            status=RolloutStatus.STARTING,
            request_id=request_id,
            task_key=task_key,
            group_id=group_id,
            owner_id=owner_id,
            node_id=node.node_id,
            init_params=dict(init),
            metadata={
                "manifest_digest": manifest.digest,
                "backend": backend,
            },
        )
        self._state.insert_rollout(record)
        if idem_key is not None:
            # Owner-namespaced key (audit M1) — see the lookup above.
            self._state.record_idempotent(idem_key, rollout_id)
        # D12 stage 1: stash verifier uploads keyed by rollout_id so
        # _compute_in_sandbox_final can retrieve them at reward time.
        if resolved_verifier_uploads:
            self._verifier_uploads[rollout_id] = resolved_verifier_uploads
        self._state.append_event(
            "rollout.starting",
            rollout_id=rollout_id,
            payload={"template": manifest.name, "node": node.node_id},
        )

        # 4. Create sandbox + run init script + call adapter.setup.
        # Network precedence: resolver per-task (Pattern A; benchmark
        # author's "this task needs hermetic/open" call) → request
        # (run-config / per-rollout kwarg, typed NetworkPolicy at the
        # boundary so typos can't reach here) → DEFAULT_NETWORK.
        effective_network: NetworkPolicy = (
            resolved_network
            if resolved_network is not None
            else network if network is not None
            else DEFAULT_NETWORK
        )
        bootstrap_started = time.monotonic()
        try:
            handle, first_obs, effective_phase_timeouts = await self._bootstrap_sandbox(
                rollout_id=rollout_id,
                node=node,
                backend=backend,
                manifest=manifest,
                init=init,
                placement=placement,
                network_policy=effective_network,
            )
        except Exception as exc:
            # Drop the scheduler reservation — idempotent, so
            # double-calling after _bootstrap_sandbox already
            # committed (i.e. failure happened *after* insert_sandbox)
            # is a safe no-op. The state record's destroy path is what
            # actually releases capacity in that case.
            self._scheduler.release_placement(placement)
            classified = _classify_startup_error(exc)
            self._state.update_rollout(
                rollout_id, status=RolloutStatus.FAILED, reason=classified
            )
            self._state.append_event(
                "rollout.failed",
                rollout_id=rollout_id,
                payload={"error": str(exc), "phase": "startup"},
            )
            if self._metrics is not None:
                self._metrics.observe_sandbox_create_failed(manifest.name, classified)
                self._metrics.observe_rollout_finished(manifest.name, RolloutStatus.FAILED)
            raise RolloutFailed(
                f"rollout {rollout_id} failed during startup",
                reason=classified,
            ) from exc
        bootstrap_seconds = time.monotonic() - bootstrap_started

        # 5. Open the trajectory sink (spec 08 platform-jsonl: writes meta.json
        # immediately so the viewer can locate the run before it terminates).
        sink_fields: dict[str, Any] = {}
        if self._sink is not None:
            locator = self._sink.open(
                rollout_id=rollout_id,
                manifest=manifest,
                init=init,
                node_id=node.node_id,
            )
            sink_fields = {
                "trajectory_sink": locator.sink,
                "trajectory_node_id": locator.node_id,
                "trajectory_uri": locator.uri,
                "trajectory_size_bytes": locator.size_bytes,
            }

        # 6. Mark RUNNING and stash the first observation
        self._state.update_rollout(
            rollout_id,
            status=RolloutStatus.RUNNING,
            sandbox_id=handle.id,
            metadata={
                **record.metadata,
                "init_obs": first_obs,
                "started_at": time.monotonic(),
                # H4 follow-up: persist the resolved per-phase budgets
                # so step() / _terminate() derive their per-call HTTP
                # caps from the *workload* budget, not the outer
                # manifest. Pattern-A benchmarks like terminal-bench-2
                # require this — the resolver supplies per-task
                # overrides via merged_init that the catalog manifest
                # alone cannot see.
                "effective_step_timeout_s": effective_phase_timeouts["step_timeout_s"],
                "effective_setup_timeout_s": effective_phase_timeouts["setup_timeout_s"],
                "effective_teardown_timeout_s": effective_phase_timeouts["teardown_timeout_s"],
            },
            **sink_fields,
        )
        self._state.append_event(
            "rollout.running",
            rollout_id=rollout_id,
            sandbox_id=handle.id,
            payload={},
        )
        if self._sink is not None:
            self._sink.record_event(
                rollout_id,
                "rollout.start",
                {
                    "template": manifest.name,
                    "node_id": node.node_id,
                    "sandbox_id": handle.id,
                    "backend": backend,
                    "bootstrap_seconds": round(bootstrap_seconds, 4),
                },
            )

        # 7. Arm the deadline watcher. Per spec 02 the hard-deadline clock
        # starts the moment the first observation is returned. When the
        # consumer didn't pass an explicit Deadline, fall back to
        # ``min(hard_s_default, ttl_default_s)`` so the spec-09 GC layer-1
        # safety net (default 1 h cap) always fires even when the
        # template's hard_s_default is generous (audit-driven manifest
        # author "expected" envelope vs operator "absolute cap").
        if deadline is not None:
            hard_s = deadline.hard_s
        else:
            hard_s = min(manifest.hard_s_default, manifest.ttl_default_s)
        self._rollout_deadlines[rollout_id] = hard_s
        self._deadlines.watch(rollout_id, hard_s)

        # Arm the idle-TTL reaper (spec 02 §"Idle TTL"). The clock resets
        # on every step and on explicit session.heartbeat() calls.
        idle_ttl_s = (
            deadline.idle_ttl_s
            if deadline and deadline.idle_ttl_s is not None
            else self._default_idle_ttl_s
        )
        self._idle.watch(rollout_id, idle_ttl_s)

        if self._metrics is not None:
            self._metrics.observe_rollout_started(manifest.name)
            self._metrics.observe_sandbox_create(
                manifest.name, backend, bootstrap_seconds
            )
            self._metrics.inc_sandbox_active(node.node_id, manifest.name)

        return rollout_id, first_obs

    def _maybe_resolve_instance(
        self,
        manifest: TemplateManifest,
        init: dict[str, Any],
    ) -> tuple[TemplateManifest, NetworkPolicy | None, tuple[Any, ...]]:
        """Spec 06 §"Pattern A": if the template declares an
        :class:`InstanceResolver` and the consumer named an
        ``instance_id`` in ``init``, overlay the resolver's per-instance
        fields onto an effective manifest.

        Returns ``(effective_manifest, resolved_network,
        verifier_uploads)``. The ``resolved_network`` is
        ``ResolvedInstance.network`` when the resolver supplied one
        (Pattern A's per-task hermetic / open signal); ``None``
        otherwise. ``verifier_uploads`` is the resolver's tuple of
        :class:`VerifierUpload` payloads for D12 timing-isolated
        grader-asset injection (empty tuple when the resolver supplies
        none).

        Resolver lookups are cached per-template (one resolver instance
        per outer manifest) so the import + index-load cost is paid
        once, not per rollout.
        """
        if manifest.instances is None:
            return manifest, None, ()
        instance_id = init.get("instance_id") or init.get("task_id")
        if not instance_id:
            # Pattern-A template invoked without an instance_id. The
            # outer template's defaults still apply; the EnvAdapter will
            # surface the missing field if it requires one.
            return manifest, None, ()
        try:
            resolver = self._resolver_for(manifest)
            resolved = resolver.resolve(str(instance_id))
        except InstanceResolverImportError as exc:
            # Fail fast with a clean reason — the operator's environment
            # is missing the resolver's dependencies.
            raise RolloutFailed(
                f"resolver_unavailable: {exc}",
                reason="resolver_unavailable",
            ) from exc
        overlay = apply_to_manifest(manifest, resolved)
        # Audit M2: re-run register-time security checks on the
        # post-overlay manifest. Pattern A's ``image is None``
        # short-circuits both ``_maybe_pin_image`` and the mount
        # allowlist at registration; the resolver populates them at
        # rollout time, so the same validators must fire here or we'd
        # quietly accept resolver-supplied unpinned images / denied
        # mount prefixes.
        validated = self._catalog.validate_overlay(overlay)
        return validated, resolved.network, tuple(resolved.verifier_uploads)

    def _resolver_for(self, manifest: TemplateManifest):  # type: ignore[no-untyped-def]
        if manifest.instances is None:
            raise RuntimeError(
                f"_resolver_for called for template {manifest.name} "
                "without an instances declaration"
            )
        cache = self._resolver_cache
        cached = cache.get(manifest.name)
        if cached is not None:
            return cached
        resolver = load_resolver(manifest.instances)
        cache[manifest.name] = resolver
        return resolver

    async def _acquire_placement(
        self,
        *,
        manifest: TemplateManifest,
        task_key: str | None,
        request_id: str | None,
        group_id: str | None,
        owner_id: str = "default",
        init: dict[str, Any],
        deadline: Deadline | None,
        backend: str,
    ) -> Placement:
        if self._admission is None:
            # No queue wired (tests / Slice 1 parity): fall through to the
            # scheduler directly. CapacityExhausted propagates as-is.
            return self._scheduler.place(
                manifest, task_key=task_key, backend=backend,
            )

        timeout_s = (
            deadline.queue_timeout_s
            if deadline and deadline.queue_timeout_s is not None
            else DEFAULT_QUEUE_TIMEOUT_S
        )
        return await self._admission.acquire(
            manifest=manifest,
            task_key=task_key,
            request_id=request_id,
            group_id=group_id,
            owner_id=owner_id,
            init_params=init,
            timeout_s=timeout_s,
            backend=backend,
        )

    async def step(self, rollout_id: str, action: Action) -> StepResult:
        record = self._state.get_rollout(rollout_id)
        if record.status != RolloutStatus.RUNNING:
            # Rollout was reaped asynchronously between steps (idle TTL,
            # hard deadline, group cancel, node loss, prior step crash).
            # Raise the matching carrier so batch_rollout / Session.__aexit__
            # bucket it correctly with its sealed partial trajectory — a
            # generic RolloutFailed(not_running) would mis-classify
            # truncations/cancellations as workload failures (audit M1
            # against commit 1c27026).
            partial = self._state.seal_trajectory(rollout_id)
            if record.status == RolloutStatus.TRUNCATED:
                raise RolloutTruncated(
                    f"rollout {rollout_id} truncated ({record.reason or 'unknown'})",
                    partial=partial,
                )
            if record.status == RolloutStatus.CANCELLED:
                raise RolloutCancelled(
                    f"rollout {rollout_id} cancelled ({record.reason or 'unknown'})",
                    partial=partial,
                )
            raise RolloutFailed(
                f"rollout {rollout_id} is not running (status={record.status})",
                reason=record.reason or "not_running",
                partial=partial,
            )

        if record.sandbox_id is None:
            raise RolloutFailed(
                f"rollout {rollout_id} has no sandbox bound",
                reason="sandbox_missing",
            )
        node = self._node_for(record)
        sandbox = self._state.get_sandbox(record.sandbox_id)
        handle = _handle_from_record(sandbox)

        # D17 stage 2 (audit H4 follow-up): per-call HTTP cap derived
        # from the *effective* per-rollout step budget (snapshot
        # written by _bootstrap_sandbox into record.metadata) with
        # fallback to the outer catalog manifest. Pattern-A benchmarks
        # like terminal-bench-2 carry per-task budgets that the outer
        # manifest cannot see; reading from the snapshot prevents the
        # HTTP cap from firing inside a legitimate long-running step.
        step_cap_s = _per_phase_http_cap(
            _effective_phase_timeout(
                record, self._catalog, "step", "step_timeout_s",
            )
        )

        # Race the env step against the per-rollout truncate event so a
        # hard-deadline expiry mid-step raises RolloutTruncated promptly
        # rather than waiting for the env to return.
        truncate_event = self._deadlines.event_for(rollout_id)
        backend_label = str(record.metadata.get("backend") or "unknown")
        step_started = time.monotonic()

        try:
            async with self._lock_for(rollout_id):
                if truncate_event is None:
                    raw = await node.env_step(
                        handle, action, request_timeout_s=step_cap_s,
                    )
                else:
                    step_task = asyncio.create_task(
                        node.env_step(
                            handle, action, request_timeout_s=step_cap_s,
                        )
                    )
                    trunc_task = asyncio.create_task(truncate_event.wait())
                    try:
                        done, _pending = await asyncio.wait(
                            {step_task, trunc_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        for t in (step_task, trunc_task):
                            if not t.done():
                                t.cancel()
                                with suppress(asyncio.CancelledError):
                                    await t
                    if trunc_task in done and step_task not in done:
                        raise RolloutTruncated(
                            f"rollout {rollout_id} truncated by hard deadline",
                            partial=self._state.seal_trajectory(rollout_id),
                        )
                    raw = step_task.result()
        except TimeoutError as exc:
            # H4 follow-up: per-call HTTP cap fired — the in-sandbox
            # stub is unresponsive (network drop, daemon crash, or a
            # legitimately-too-tight cap derived from an under-sized
            # manifest budget). Pre-fix this propagated raw and the
            # rollout stayed RUNNING until something else (hard
            # deadline, operator action) cleaned it up. Now we
            # terminalize cleanly: skip env_teardown (calling it
            # would just hang on the same wedged stub) but destroy
            # the sandbox via _terminate so the container goes away
            # and capacity releases. Raise RolloutFailed with the
            # partial trajectory so batch_rollout / Session.__aexit__
            # bucket the rollout correctly.
            LOGGER.warning(
                "rollout %s env_step exceeded per-call HTTP cap (%ss); "
                "sealing failed/transport_timeout",
                rollout_id, step_cap_s,
            )
            # Re-fetch the record so _terminate sees the latest sandbox /
            # node state (the lock above is the rollout-level lock, not
            # the state-store lock).
            record = self._state.get_rollout(rollout_id)
            partial = await self._terminate(
                record,
                # Transient = current state (RUNNING) — matches the
                # adapter-truncate path at L751 and the reward-failed
                # path at L879. The rollout is moved straight to FAILED
                # via the transient-then-terminal update sequence
                # _terminate runs internally.
                transient_status=RolloutStatus.RUNNING,
                terminal_status=RolloutStatus.FAILED,
                reason="transport_timeout",
                skip_env_teardown=True,
            )
            raise RolloutFailed(
                f"rollout {rollout_id} env_step exceeded per-call HTTP cap "
                f"({step_cap_s}s)",
                reason="transport_timeout",
                partial=partial,
            ) from exc

        result = _step_result_from_payload(raw)
        ts = time.monotonic() - float(record.metadata.get("started_at", time.monotonic()))
        step = Step(
            index=len(record.steps),
            action=action,
            obs=result.obs,
            reward=result.reward,
            done=result.done,
            truncated=result.truncated,
            info=result.info,
            ts=ts,
        )
        # Persist the step via the StateStore so both InMemory and Sqlite
        # backends accumulate identically; the rolling reward sum is updated
        # by the store under the same lock. Mirror to the trajectory sink so
        # the spec-20 platform-jsonl body grows alongside (the state-store
        # copy is the metadata, the sink is the durable body).
        self._state.append_step(rollout_id, step)
        if self._sink is not None:
            self._sink.record_step(rollout_id, step)
        # Each step is implicitly a touch (spec 02 §"Consumer-side heartbeat");
        # explicit Session.heartbeat() calls go through the touch() method.
        self._idle.touch(rollout_id)

        if self._metrics is not None:
            self._metrics.observe_step_latency(
                record.template, backend_label, time.monotonic() - step_started
            )

        # Audit M1 (2026-04-29 follow-up): when the EnvAdapter signals
        # ``result.truncated=True`` (step-level timeout — see
        # TerminalBench2EnvAdapter / ShellEnvAdapter), the rollout
        # cannot be a benchmark-valid completion. Seal as TRUNCATED
        # with reason="step_timeout", skip in_sandbox_final, and raise
        # RolloutTruncated through the SDK so the consumer's session
        # context-manager doesn't fall through to ``finish()`` and
        # mis-seal as FINISHED. Pre-fix, the SDK marked the session
        # done on result.truncated but the coordinator never
        # transitioned the rollout state, so a hung command surfaced
        # as a successful rollout with no final reward.
        if result.truncated:
            sealed = await self._terminate(
                record,
                transient_status=RolloutStatus.RUNNING,  # already running
                terminal_status=RolloutStatus.TRUNCATED,
                reason="step_timeout",
            )
            raise RolloutTruncated(
                f"rollout {rollout_id} truncated by step timeout",
                partial=sealed,
            )

        # Spec 02 RewardContract: in_sandbox_final runs reward.cmd inside
        # the sandbox at done. Per spec the final reward overrides any
        # per-step accumulation; per-step reward is always 0 in this mode
        # (templates that mix shaped + final rewards use env_step instead).
        if result.done:
            manifest = self._catalog.get(record.template)
            if manifest.reward.mode == "in_sandbox_final":
                await self._compute_in_sandbox_final(
                    rollout_id=rollout_id,
                    sandbox_handle=handle,
                    node=node,
                    manifest=manifest,
                )
        return result

    async def finish(self, rollout_id: str) -> Trajectory:
        record = self._state.get_rollout(rollout_id)
        if record.status.is_terminal:
            return self._state.seal_trajectory(rollout_id)
        return await self._terminate(
            record, RolloutStatus.FINISHING, RolloutStatus.FINISHED, reason=None
        )

    async def cancel(self, rollout_id: str, reason: str = "consumer_cancelled") -> Trajectory:
        record = self._state.get_rollout(rollout_id)
        if record.status.is_terminal:
            return self._state.seal_trajectory(rollout_id)
        return await self._terminate(
            record,
            RolloutStatus.CANCELLING,
            RolloutStatus.CANCELLED,
            reason=reason,
        )

    async def touch(self, rollout_id: str) -> None:
        """Reset the idle-TTL clock for ``rollout_id`` (spec 02 heartbeat).

        Called from ``Session.heartbeat()`` so a consumer doing a long
        ``policy.act`` between steps doesn't get reaped.
        """
        # Idempotent — touching an unwatched rollout is a no-op.
        self._idle.touch(rollout_id)

    def list_nodes(self) -> list[Any]:
        """Snapshot of the NodeRegistry's persistent mirror (D21).

        Returns the same ``NodeRecord`` rows the operator CLI's
        ``xrlenv nodes`` and the admin panel's ``/nodes`` view see.
        Sync because state-store access is a SQLite call.
        """
        return self._state.list_nodes()

    async def set_final_reward(self, rollout_id: str, final_reward: float) -> None:
        """Slice 4.5: SDK-side consumer_final back-fill.

        Updates the StateStore's ``rollouts.final_reward`` column and asks
        the trajectory sink to atomically rewrite ``meta.json`` so replay
        and the trajectory viewer see the canonical value. The
        ``trajectory.jsonl`` body is left untouched (spec 00 invariant 3 —
        steps are immutable after seal; only the locator envelope is
        rewritten).
        """
        record = self._state.get_rollout(rollout_id)
        self._state.update_rollout(rollout_id, final_reward=final_reward)
        if self._sink is not None:
            update = getattr(self._sink, "update_final_reward", None)
            if update is not None:
                with suppress(Exception):
                    update(
                        rollout_id=rollout_id,
                        final_reward=final_reward,
                        status=record.status,
                        reason=record.reason,
                        metadata=record.metadata,
                    )

    async def _compute_in_sandbox_final(
        self,
        *,
        rollout_id: str,
        sandbox_handle: SandboxHandle,
        node: NodeTransport,
        manifest: TemplateManifest,
    ) -> None:
        """Run the manifest's reward graders inside the sandbox and write
        the resulting per-grader scores + aggregated final reward into
        the StateStore. Honors the contract's ``on_error`` semantics
        (spec 02 RewardContract).
        """
        try:
            computation = await compute_in_sandbox_final_reward(
                node=node,
                sandbox=sandbox_handle,
                contract=manifest.reward,
                verifier_uploads=self._verifier_uploads.get(rollout_id, ()),
            )
        except RewardComputationError as exc:
            comp = exc.computation
            record = self._state.get_rollout(rollout_id)
            self._state.update_rollout(
                rollout_id,
                final_reward=0.0,
                metadata={
                    **record.metadata,
                    "rewards": _per_grader_dict(comp.per_grader),
                    "reward_error": comp.error_message,
                },
            )
            # Seal the rollout as FAILED/reward_failed *before* raising. Without
            # this, the RolloutFailed bubbles up to RolloutSession.__aexit__,
            # which catches the exception and issues a fresh cancel —
            # overwriting the terminal state to cancelled/aborted_with_exception
            # (audit H1 against commit af3b985). Calling _terminate here also
            # destroys the sandbox + seals the trajectory sink, so the session's
            # follow-up cancel hits the is_terminal short-circuit and returns
            # the already-sealed FAILED trajectory.
            record = self._state.get_rollout(rollout_id)
            partial = await self._terminate(
                record,
                RolloutStatus.RUNNING,
                RolloutStatus.FAILED,
                reason="reward_failed",
            )
            raise RolloutFailed(
                f"rollout {rollout_id} reward computation failed: {comp.error_message}",
                reason="reward_failed",
                partial=partial,
            ) from exc

        record = self._state.get_rollout(rollout_id)
        new_metadata = {
            **record.metadata,
            "rewards": _per_grader_dict(computation.per_grader),
        }
        if computation.error_message:
            new_metadata["reward_error"] = computation.error_message
        self._state.update_rollout(
            rollout_id,
            final_reward=computation.final_reward,
            metadata=new_metadata,
        )

        # Verifier-dir persistence is not done here — it now lives in
        # ``_terminate`` so it runs on EVERY terminal path (success,
        # truncation, failure, cancellation) before the sandbox is
        # destroyed. Truncated rollouts often have partial verifier
        # output (a half-written test.log or a reward.txt the agent
        # almost finished writing) that's diagnostically useful.

    async def _fetch_verifier_dir_to_run_dir(
        self,
        *,
        rollout_id: str,
        sandbox_handle: SandboxHandle,
        node: NodeTransport,
        container_path: str = "/logs/verifier",
    ) -> None:
        """Fetch a directory from inside the sandbox and write its
        contents under ``<run_dir>/verifier/`` host-side.

        Harbor's universal convention is that grader output lives at
        ``/logs/verifier/`` (test.log, reward.txt, reward.json,
        ctrf.json). Mirroring that on disk gives the consumer a
        familiar layout: agent's interaction → ``trajectory.jsonl``;
        verifier's output → ``verifier/``.

        Implementation: ``tar -czf - -C / logs/verifier | base64 -w0``
        through ``run_in_sandbox``. The base64 wrapper is needed
        because the bidi stub channel decodes stdout as UTF-8 with
        replacement; raw tarball bytes would be corrupted. Skip
        silently on any failure path — verifier dirs are an
        opt-in convention, not a contract.
        """
        if self._sink is None:
            return
        run_dir = self._sink.run_dir_for_rollout(rollout_id)
        if run_dir is None:
            return
        cmd = [
            "sh", "-c",
            f"tar -czf - -C / {container_path.lstrip('/')} 2>/dev/null | base64 -w0",
        ]
        try:
            result = await node.run_in_sandbox(
                sandbox_handle, cmd, timeout_s=30.0,
            )
        except Exception as exc:
            LOGGER.debug(
                "rollout=%s verifier-dir fetch transport failure: %s",
                rollout_id, exc,
            )
            return
        if result.exit_code != 0 or not result.stdout:
            return  # tar found nothing — not a benchmark with /logs/verifier.

        import base64 as _base64
        import io as _io
        import tarfile as _tarfile

        try:
            tarball = _base64.b64decode(result.stdout)
        except Exception:
            return
        # Extract under ``<run_dir>/verifier/`` so the admin panel's
        # ``_gather_verifier_artifacts`` walk (which keys on the
        # ``verifier/`` subdir) picks them up. The tar archive
        # entries themselves are ``logs/verifier/...`` (from
        # ``-C / logs/verifier``), so the on-disk layout becomes
        # ``<run_dir>/verifier/logs/verifier/<file>``. Stable across
        # benchmarks: any sandbox-side tree we add to the fetch in
        # the future shows up under the same ``verifier/`` umbrella.
        verifier_root = run_dir / "verifier"
        verifier_root.mkdir(parents=True, exist_ok=True)
        try:
            with _tarfile.open(fileobj=_io.BytesIO(tarball), mode="r:gz") as tar:
                # ``filter="data"`` rejects unsafe entries (absolute
                # paths, parent traversal, special files) — the
                # tarball came from inside an arguably-untrusted
                # sandbox so we don't blindly trust paths.
                tar.extractall(path=verifier_root, filter="data")
        except Exception as exc:
            LOGGER.warning(
                "rollout=%s verifier-dir extraction failed: %s",
                rollout_id, exc,
            )

    def get_rollout_owner(self, rollout_id: str) -> str | None:
        """Owner of a gym/step rollout by id, or ``None`` if unknown.

        Used by the gRPC servicer to enforce that follow-up RPCs (Step,
        Finish, Cancel, …) act only on the caller's own rollouts (audit M2)."""
        try:
            return self._state.get_rollout(rollout_id).owner_id
        except KeyError:
            return None

    async def cancel_group(
        self,
        group_id: str,
        reason: str = "group_cancelled",
        *,
        owner_id: str | None = None,
    ) -> CancelGroupReport:
        """Cancel every still-running rollout that carries ``group_id``.

        Idempotent: rollouts already in a terminal state are counted as
        ``already_terminal`` and not re-cancelled. The platform's contract
        (spec 02 §"Group / batch rollouts") is that this is best-effort
        and runs at the standard 5 s teardown budget per rollout.

        ``owner_id`` (audit M2): when set, only rollouts owned by that tenant
        are considered — so a consumer cannot cancel another tenant's group by
        guessing its ``group_id``. ``None`` (single-tenant / no-auth / admin)
        cancels every match, as before.
        """
        cancelled: list[str] = []
        already_terminal: list[str] = []
        for record in self._state.list_rollouts():
            if record.group_id != group_id:
                continue
            if owner_id is not None and record.owner_id != owner_id:
                continue
            if record.status.is_terminal:
                already_terminal.append(record.rollout_id)
                continue
            try:
                await self._terminate(
                    record,
                    RolloutStatus.CANCELLING,
                    RolloutStatus.CANCELLED,
                    reason=reason,
                )
                cancelled.append(record.rollout_id)
            except Exception:
                LOGGER.exception(
                    "cancel_group: rollout=%s teardown failed; sealing anyway",
                    record.rollout_id,
                )
                cancelled.append(record.rollout_id)
        return CancelGroupReport(
            group_id=group_id,
            cancelled=tuple(cancelled),
            already_terminal=tuple(already_terminal),
        )

    async def handle_sandbox_lost(
        self, node_id: str, sandbox_id: str, *, reason: str = "sandbox_lost",
    ) -> None:
        """A3 / D15 (P1.1) — seal the rollout owning ``sandbox_id`` as
        ``failed/sandbox_lost`` when the spec-09 GC layer 3 reconciler
        observes the node has lost the sandbox while the node itself
        is still attached. Mirrors :py:meth:`handle_node_lost` but for
        the single-sandbox case (one container died on a still-healthy
        node — Docker crash, OOM-killed, manual ``docker rm``).

        Idempotent: skips terminal rollouts and already-removed
        sandbox rows.
        """
        try:
            sandbox = self._state.get_sandbox(sandbox_id)
        except KeyError:
            return  # already cleaned up by another path
        rollout_id = sandbox.rollout_id
        if rollout_id is None:
            # Sandbox has no owning rollout (transient state during
            # create/destroy). Drop the sandbox row; nothing to seal.
            with suppress(KeyError):
                self._state.update_sandbox(sandbox_id, status="destroyed")
                self._state.remove_sandbox(sandbox_id)
            return
        try:
            record = self._state.get_rollout(rollout_id)
        except KeyError:
            with suppress(KeyError):
                self._state.update_sandbox(sandbox_id, status="destroyed")
                self._state.remove_sandbox(sandbox_id)
            return
        if record.status.is_terminal:
            with suppress(KeyError):
                self._state.update_sandbox(sandbox_id, status="destroyed")
                self._state.remove_sandbox(sandbox_id)
            return
        self._state.update_rollout(
            rollout_id, status=RolloutStatus.FAILED, reason=reason,
        )
        self._state.append_event(
            "rollout.failed",
            rollout_id=rollout_id,
            payload={"reason": reason, "node_id": node_id, "sandbox_id": sandbox_id},
        )
        with suppress(KeyError):
            self._state.update_sandbox(sandbox_id, status="destroyed")
            self._state.remove_sandbox(sandbox_id)
        if self._metrics is not None:
            self._metrics.observe_rollout_finished(
                record.template, RolloutStatus.FAILED,
            )
            self._metrics.dec_sandbox_active(node_id, record.template)
        if self._sink is not None:
            self._sink.record_event(
                rollout_id, "rollout.fail",
                {"reason": reason, "node_id": node_id, "sandbox_id": sandbox_id},
            )
            with suppress(Exception):
                locator = self._sink.seal(
                    rollout_id=rollout_id,
                    status=RolloutStatus.FAILED,
                    reason=reason,
                    final_reward=record.final_reward,
                    metadata=record.metadata,
                )
                self._state.update_rollout(
                    rollout_id,
                    trajectory_sink=locator.sink,
                    trajectory_node_id=locator.node_id,
                    trajectory_uri=locator.uri,
                    trajectory_size_bytes=locator.size_bytes,
                )
        self._post_terminate(rollout_id)
        LOGGER.warning(
            "rollout=%s sealed failed/%s (sandbox=%s on node=%s)",
            rollout_id, reason, sandbox_id, node_id,
        )

    async def handle_node_lost(self, node_id: str) -> None:
        """Seal every in-flight rollout on ``node_id`` as failed/node_lost.

        The node is unreachable by definition, so we cannot run env_teardown
        or container destroy through it. We mark the StateStore terminal,
        seal the trajectory sink, free the deadline watcher, and drop the
        sandbox row so the scheduler stops counting it against capacity.
        """
        try:
            records = list(self._state.list_rollouts())
        except sqlite3.ProgrammingError as exc:
            # The state store closed underneath us — typical during
            # ``DistributedRuntime.shutdown`` if a stream-cancel-driven
            # seal lost the race with ``state.close()``. Nothing useful
            # to do; the next process startup will reconcile.
            LOGGER.info(
                "handle_node_lost(%s): state store closed (%s); skipping seal",
                node_id, exc,
            )
            return
        for record in records:
            if record.node_id != node_id or record.status.is_terminal:
                continue
            self._state.update_rollout(
                record.rollout_id,
                status=RolloutStatus.FAILED,
                reason="node_lost",
            )
            self._state.append_event(
                "rollout.failed",
                rollout_id=record.rollout_id,
                payload={"reason": "node_lost", "node_id": node_id},
            )
            if record.sandbox_id is not None:
                with suppress(KeyError):
                    self._state.update_sandbox(record.sandbox_id, status="destroyed")
                    self._state.remove_sandbox(record.sandbox_id)
            if self._metrics is not None:
                self._metrics.observe_rollout_finished(
                    record.template, RolloutStatus.FAILED
                )
                self._metrics.dec_sandbox_active(node_id, record.template)
            if self._sink is not None:
                self._sink.record_event(
                    record.rollout_id,
                    "rollout.fail",
                    {"reason": "node_lost", "node_id": node_id},
                )
                with suppress(Exception):
                    locator = self._sink.seal(
                        rollout_id=record.rollout_id,
                        status=RolloutStatus.FAILED,
                        reason="node_lost",
                        final_reward=record.final_reward,
                        metadata=record.metadata,
                    )
                    self._state.update_rollout(
                        record.rollout_id,
                        trajectory_sink=locator.sink,
                        trajectory_node_id=locator.node_id,
                        trajectory_uri=locator.uri,
                        trajectory_size_bytes=locator.size_bytes,
                    )
            self._post_terminate(record.rollout_id)
            LOGGER.warning(
                "rollout=%s sealed failed/node_lost (node=%s went away)",
                record.rollout_id,
                node_id,
            )

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _bootstrap_sandbox(
        self,
        *,
        rollout_id: str,
        node: NodeTransport,
        backend: str,
        manifest: TemplateManifest,
        init: dict[str, Any],
        placement: Placement,
        network_policy: NetworkPolicy,
    ) -> tuple[SandboxHandle, Observation, dict[str, float]]:
        # Phase 3 of spec-02 startup: create sandbox.
        # ``network_policy`` is the resolved per-rollout value computed
        # by ``start_rollout`` from (run-config / per-rollout kwargs /
        # manifest fallback). For Pattern A, Slice 9b will additionally
        # let the resolver override per-task here.
        #
        # A1 / D19 (P1.2) pre-flight image check. Ask the chosen node
        # "do you have this image?" before sending CreateSandboxCommand.
        # The check serves two purposes:
        #
        #   1. Per-placement audit event (D20 audit-shape): record the
        #      QueryImage outcome + digest_source so spec 19 can
        #      account for what was verified where. Always emitted.
        #   2. Fail-fast for non-registry-digest modes: when the
        #      manifest declares ``per_node_local`` or
        #      ``shared_storage``, a missing image is a hard error
        #      because no registry authority exists for the node to
        #      pull from. ``ensure_present()`` would just retry a
        #      doomed pull. Surface the miss as
        #      ``reason="image_missing"`` so operators see a clear
        #      actionable error.
        #
        # For ``registry_digest`` (the default), a missing image is a
        # normal cold-cache state — the node's
        # ``ImageCacheManager.ensure_present()`` will pull from the
        # registry inside ``create_sandbox()`` and the rollout
        # proceeds. Don't fail-fast in that case.
        # Audit response (H3, 2026-05-02): early P1.2.a builds were
        # too eager — they turned every cold-cache miss into
        # ``image_missing``, breaking the registry-pull path that
        # spec 15 explicitly designed. The mode-gated fix preserves
        # both fail-fast for non-pullable modes AND the cold-cache
        # pull path for the default mode.
        if manifest.image:
            check = await node.query_image(manifest.image)
            # Always emit the per-placement audit event (M3 audit-
            # shape closure).
            digest_source = (
                "per_node" if manifest.image_pin_mode == "per_node_local"
                else "shared_storage" if manifest.image_pin_mode == "shared_storage"
                else "registry"
            )
            self._state.append_audit(
                "placement.image_check",
                payload={
                    "node_id": node.node_id,
                    "image": manifest.image,
                    "image_pin_mode": manifest.image_pin_mode,
                    "present": check.present,
                    "digest_source": digest_source,
                    "digest": check.digest,
                },
            )
            # Fail-fast only when the mode has no registry authority
            # behind it. ``registry_digest`` cold-cache misses fall
            # through to ``ensure_present()`` inside create_sandbox.
            if (
                not check.present
                and manifest.image_pin_mode != "registry_digest"
            ):
                raise ImageMissingOnNode(
                    f"node {node.node_id!r} does not have image "
                    f"{manifest.image!r} (image_pin_mode="
                    f"{manifest.image_pin_mode!r}; no registry "
                    f"authority — fail fast per D19 pre-flight)"
                )
        handle = await node.create_sandbox(
            rollout_id=rollout_id,
            backend=backend,
            template=manifest.template_ref(),
            resources=manifest.resources,
            network_policy=network_policy,
            # A5 / D17 stage 1 (audit response): stage the per-sandbox
            # HTTP cap before the node ever builds a StubClient. The
            # earlier merged_init injection got bypassed by manifests
            # with init_cmd because run_in_sandbox triggered _stub_for
            # before env_setup could stage the cap.
            stub_request_timeout_s=_http_cap_from_manifest(manifest),
        )
        sandbox_record = SandboxRecord(
            sandbox_id=handle.id,
            backend=handle.backend,
            backend_ref=handle.backend_ref,
            stub_endpoint=handle.stub_endpoint,
            template=manifest.name,
            image=manifest.image,
            node_id=node.node_id,
            rollout_id=rollout_id,
            # Pattern A: ``manifest.resources`` is the post-resolver
            # overlay. Snapshot it so the scheduler's per-node load
            # accounting charges the right per-task cost rather than
            # the outer manifest's defaults.
            effective_resources_json=manifest.resources.model_dump_json(),
        )
        self._state.insert_sandbox(sandbox_record)
        # State now has the sandbox; the scheduler can read it from
        # ``list_sandboxes`` and counting it in ``_pending`` would
        # double up. Drop the reservation immediately, before any
        # await — this minimises the (already-tiny) window in which a
        # concurrent ``place()`` would see *both* the state record and
        # the pending entry. After this point, the scheduler relies on
        # state for accounting.
        self._scheduler.commit_placement(placement)

        try:
            # Phase 5 of spec-02 startup: template init script (if any).
            # Slice 3.5: routes through NodeTransport.run_in_sandbox so it
            # works identically over the in-process NodeAgent and the gRPC
            # RemoteNodeTransport (closes audit M1).
            if manifest.init_cmd:
                init_result = await node.run_in_sandbox(
                    handle,
                    list(manifest.init_cmd),
                    timeout_s=manifest.init_timeout_s,
                )
                if init_result.exit_code != 0:
                    raise RuntimeError(
                        f"init.cmd failed: exit_code={init_result.exit_code} "
                        f"stderr={init_result.stderr.decode('utf-8', errors='replace')!r}"
                    )

            # Phase 7: adapter.setup. Merge layered init params with the
            # right precedence:
            #
            #   manifest.step_timeout_s / setup_timeout_s   (platform default)
            #     < manifest.env_adapter.init_params         (resolver-supplied; per-task)
            #       < user-supplied ``init`` kwarg            (per-rollout override)
            #
            # The previous order put manifest.{step,setup}_timeout_s LAST,
            # which silently overrode any per-task ``step_timeout_s`` the
            # resolver had written into ``manifest.env_adapter.init_params``
            # via ``apply_to_manifest``. That hit the tb2 tasks whose
            # ``[agent].timeout_sec`` in task.toml is >> the run-config's
            # deadlines.step_timeout_s — they truncated even with the
            # per-task value plumbed through, because the platform
            # default kept winning.
            merged_init = {
                "step_timeout_s": manifest.step_timeout_s,
                "setup_timeout_s": manifest.setup_timeout_s,
                **manifest.env_adapter.init_params,
                **init,
            }
            # D17 stage 2 (audit H4 follow-up): snapshot the resolved
            # per-phase budgets so later step() / _terminate() calls
            # derive their per-call HTTP caps from the *effective*
            # workload budget, not the outer manifest default. For
            # Pattern-A benchmarks like terminal-bench-2 the resolver
            # writes per-task budgets into ``manifest.env_adapter.
            # init_params`` (e.g. ``step_timeout_s = 1800`` for
            # ``crack-7z-hash``); without this snapshot, step()'s
            # HTTP cap would read the 30 s outer default and trip
            # ``aiohttp.ClientTimeout`` long before the adapter's own
            # subprocess budget fired. ``teardown_timeout_s`` isn't
            # in the merged_init seed but flows through if the
            # resolver / user supplies it; fall back to the outer
            # manifest's value otherwise.
            def _resolved(key: str, fallback: float) -> float:
                # Treat both "missing key" and "key present but None"
                # as "use the manifest default". The merge chain
                # (manifest seed ⊕ resolver init_params ⊕ user init)
                # can produce ``None`` if the user passes
                # ``init={"step_timeout_s": None}`` — bare ``.get(k,
                # default)`` would still return None in that case.
                v = merged_init.get(key)
                return float(v) if isinstance(v, (int, float)) else float(fallback)

            effective_phase_timeouts = {
                "step_timeout_s": _resolved("step_timeout_s", manifest.step_timeout_s),
                "setup_timeout_s": _resolved("setup_timeout_s", manifest.setup_timeout_s),
                "teardown_timeout_s": _resolved("teardown_timeout_s", manifest.teardown_timeout_s),
            }
            # A5 / D17 stage 1 audit response: the per-sandbox HTTP
            # cap is now staged at create_sandbox time (passed as a
            # kwarg through CreateSandboxCommand), not threaded
            # through env_setup's init_params. The earlier path got
            # bypassed by manifests with init_cmd.
            #
            # D17 stage 2 (P1.2.b): on top of that floor, env_setup
            # also carries its own per-call cap derived from the
            # manifest's setup_timeout_s + buffer. The HTTP cap is
            # the safety net; the EnvAdapter's own setup-phase
            # subprocess timeout is the workload's ground truth.
            setup_reply = await node.env_setup(
                handle,
                adapter_module=manifest.env_adapter.module,
                adapter_class=manifest.env_adapter.class_name,
                init_params=merged_init,
                request_timeout_s=_per_phase_http_cap(
                    effective_phase_timeouts["setup_timeout_s"]
                ),
            )
            first_obs = setup_reply.get("obs")
        except BaseException:
            # Spec-02 cleanup obligation: any phase that completed must be
            # unwound before sealing failed. The sandbox was created in
            # phase 3; tear it down so we don't leak the container.
            with suppress(Exception):
                await node.destroy_sandbox(handle)
            self._state.remove_sandbox(handle.id)
            raise

        return handle, first_obs, effective_phase_timeouts

    async def _terminate(
        self,
        record: RolloutRecord,
        transient_status: RolloutStatus,
        terminal_status: RolloutStatus,
        *,
        reason: str | None,
        skip_env_teardown: bool = False,
    ) -> Trajectory:
        self._state.update_rollout(record.rollout_id, status=transient_status, reason=reason)
        self._state.append_event(
            f"rollout.{transient_status.value}",
            rollout_id=record.rollout_id,
            payload={"reason": reason} if reason else {},
        )

        if record.sandbox_id is None:
            self._state.update_rollout(
                record.rollout_id, status=terminal_status, reason=reason
            )
            if self._metrics is not None:
                self._metrics.observe_rollout_finished(record.template, terminal_status)
            self._post_terminate(record.rollout_id)
            return self._state.seal_trajectory(record.rollout_id)

        sandbox = self._state.get_sandbox(record.sandbox_id)
        node = self._node_for(record)
        handle = _handle_from_record(sandbox)
        backend_label = handle.backend or "unknown"

        # Truncation kills the sandbox at the container layer; we never call
        # env_teardown because the EnvAdapter's pinned thread may be mid-step
        # and can't be safely preempted. The container dying takes the thread
        # with it.
        #
        # Each terminal-path bidi RPC is bounded by an explicit asyncio
        # timeout so a wedged node can't pin the rollout in
        # ``cancelling`` / ``finishing`` forever. Pre-fix, a hung
        # ``destroy_sandbox`` (e.g. node's docker daemon busy on a slow
        # build) blocked this whole function indefinitely; the rollout's
        # state column stayed at the transient value across operator
        # restarts, and the only recovery was direct SQL surgery against
        # ``state.db``. The values below are deliberately generous:
        # operations that need real time (e.g. ``docker rm -f`` on a
        # CPU-saturated node, or harbor's verifier ``tar`` on a large
        # ``/logs/verifier`` tree) get headroom; only genuine
        # node-unresponsive cases trip the timeout.
        if not skip_env_teardown:
            # D17 stage 2 (audit H4 follow-up): per-call HTTP cap from
            # the *effective* teardown budget snapshot, with fallback
            # to the catalog manifest. The outer asyncio.wait_for is
            # the operator-level safety net (stays at
            # _TEARDOWN_TIMEOUT_S so a wedged node can't pin us); the
            # per-call cap inside aiohttp is the workload-tuned net.
            try:
                teardown_cap_s: float | None = _per_phase_http_cap(
                    _effective_phase_timeout(
                        record, self._catalog,
                        "teardown", "teardown_timeout_s",
                    )
                )
            except Exception:
                # Catalog lookup miss (template was unregistered after
                # this rollout was scheduled) AND no snapshot present.
                # Fall back to None so the NodeAgent uses the
                # per-sandbox stage-1 cap.
                teardown_cap_s = None
            try:
                await asyncio.wait_for(
                    node.env_teardown(handle, request_timeout_s=teardown_cap_s),
                    timeout=_TEARDOWN_TIMEOUT_S,
                )
            except TimeoutError:
                LOGGER.warning(
                    "env_teardown timed out (>%ss) for rollout=%s; the "
                    "container destroy that follows will take the in-sandbox "
                    "stub down anyway",
                    _TEARDOWN_TIMEOUT_S, record.rollout_id,
                )
            except Exception:
                LOGGER.exception(
                    "env_teardown failed for rollout=%s", record.rollout_id
                )

        # Verifier-dir persistence (harbor trial-paths shape). Runs on
        # every terminal path before destroy so truncated/failed
        # rollouts also surface whatever partial verifier output the
        # agent created (a half-written test.log, an early reward.txt,
        # etc.). Best-effort: silent skip when the dir doesn't exist
        # (e.g. hello-shell) or fetch fails or the bidi takes too long.
        with suppress(Exception):
            await asyncio.wait_for(
                self._fetch_verifier_dir_to_run_dir(
                    rollout_id=record.rollout_id,
                    sandbox_handle=handle,
                    node=node,
                ),
                timeout=_VERIFIER_FETCH_TIMEOUT_S,
            )

        destroy_started = time.monotonic()
        destroy_failed = False
        try:
            await asyncio.wait_for(
                node.destroy_sandbox(handle),
                timeout=_DESTROY_TIMEOUT_S,
            )
        except TimeoutError:
            destroy_failed = True
            LOGGER.warning(
                "destroy_sandbox timed out (>%ss) for rollout=%s sandbox=%s "
                "on node=%s; force-sealing rollout as terminal and marking "
                "the sandbox row ``destroy_pending`` for the GC reconciler "
                "(D15) / node-startup sweep to clean up later",
                _DESTROY_TIMEOUT_S, record.rollout_id, record.sandbox_id,
                node.node_id,
            )
            with suppress(Exception):
                self._state.update_sandbox(record.sandbox_id, status="destroy_pending")
        except Exception:
            destroy_failed = True
            LOGGER.exception(
                "destroy_sandbox raised for rollout=%s sandbox=%s; "
                "force-sealing as terminal",
                record.rollout_id, record.sandbox_id,
            )
            with suppress(Exception):
                self._state.update_sandbox(record.sandbox_id, status="destroy_pending")
        finally:
            if not destroy_failed:
                self._state.update_sandbox(record.sandbox_id, status="destroyed")
                self._state.remove_sandbox(record.sandbox_id)
        if self._metrics is not None:
            self._metrics.observe_sandbox_destroy(
                record.template, backend_label, time.monotonic() - destroy_started
            )
            self._metrics.dec_sandbox_active(node.node_id, record.template)

        sealed_record = self._state.update_rollout(
            record.rollout_id, status=terminal_status, reason=reason
        )
        if self._metrics is not None:
            self._metrics.observe_rollout_finished(record.template, terminal_status)
        if self._sink is not None:
            event_name = (
                "rollout.truncate"
                if terminal_status == RolloutStatus.TRUNCATED
                else "rollout.cancel"
                if terminal_status == RolloutStatus.CANCELLED
                else "rollout.fail"
                if terminal_status == RolloutStatus.FAILED
                else "rollout.finish"
            )
            self._sink.record_event(
                record.rollout_id,
                event_name,
                {
                    "status": terminal_status.value,
                    "reason": reason,
                    "final_reward": sealed_record.final_reward,
                    "step_count": len(sealed_record.steps),
                },
            )
        self._state.append_event(
            f"rollout.{terminal_status.value}",
            rollout_id=record.rollout_id,
            payload={"reason": reason} if reason else {},
        )
        if self._sink is not None:
            try:
                locator = self._sink.seal(
                    rollout_id=record.rollout_id,
                    status=terminal_status,
                    reason=reason,
                    final_reward=sealed_record.final_reward,
                    metadata=sealed_record.metadata,
                )
                self._state.update_rollout(
                    record.rollout_id,
                    trajectory_sink=locator.sink,
                    trajectory_node_id=locator.node_id,
                    trajectory_uri=locator.uri,
                    trajectory_size_bytes=locator.size_bytes,
                )
            except KeyError:
                # Sink may have already been sealed (e.g. truncation race);
                # the on-disk meta.json has the canonical state already.
                pass
        self._post_terminate(record.rollout_id)
        return self._state.seal_trajectory(record.rollout_id)

    def _post_terminate(self, rollout_id: str) -> None:
        """Hooks that run after every terminal transition (idempotent)."""
        self._deadlines.cancel(rollout_id)
        self._idle.cancel(rollout_id)
        self._rollout_deadlines.pop(rollout_id, None)
        # D12 stage 1: drop the verifier-asset payloads — the sandbox
        # is gone, the bytes have served their purpose, no reason to
        # keep ~MB of tarballs in memory per rollout.
        self._verifier_uploads.pop(rollout_id, None)
        # Notify admission queue that capacity may have freed (invariant 2:
        # release on confirmed destroy, which we just did).
        if self._admission is not None:
            self._admission.kick()

    async def _truncate_callback(self, rollout_id: str, reason: str) -> None:
        """DeadlineWatcher -> coordinator hook on hard_s expiry."""
        try:
            record = self._state.get_rollout(rollout_id)
        except KeyError:
            return
        if record.status.is_terminal:
            return
        await self._terminate(
            record,
            RolloutStatus.RUNNING,  # transient status irrelevant for truncation
            RolloutStatus.TRUNCATED,
            reason=reason,
            skip_env_teardown=True,
        )

    # ── Replay ───────────────────────────────────────────────────────────────

    def replay(self, rollout_id: str) -> Trajectory:
        """Read back a sealed (or in-flight) trajectory.

        When a sink is wired, replay reads the on-disk body so trajectories
        survive control-plane restarts. When no sink is wired (tests), we fall
        back to the StateStore's in-memory copy.
        """
        if self._sink is None:
            return self._state.seal_trajectory(rollout_id)
        try:
            return self._sink.read(rollout_id)
        except FileNotFoundError:
            return self._state.seal_trajectory(rollout_id)

    def _node_for(self, record: RolloutRecord) -> NodeTransport:
        if record.node_id is None:
            raise RuntimeError(f"rollout {record.rollout_id} has no node assignment")
        for n in self._scheduler.nodes:
            if n.node_id == record.node_id:
                return n
        raise RuntimeError(f"node {record.node_id} disappeared from scheduler")

    def _lock_for(self, rollout_id: str) -> asyncio.Lock:
        if rollout_id not in self._rollout_locks:
            self._rollout_locks[rollout_id] = asyncio.Lock()
        return self._rollout_locks[rollout_id]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _http_cap_from_manifest(manifest: TemplateManifest) -> float:
    """Tightest HTTP cap that's still wider than every per-phase
    inner timeout the manifest declares.

    D17 stage 1 cap, staged on the per-sandbox record at create time
    via ``CreateSandboxCommand.stub_request_timeout_s``. Acts as the
    safety-net the StubClient's ``aiohttp.ClientTimeout`` is built
    with; wide enough that an env_step with the manifest's longest
    phase budget still completes without hitting the HTTP cap.

    Stage 2 (P1.2.b) makes individual env_setup / env_step /
    env_teardown calls carry their own per-phase cap on top of this
    safety-net via :func:`_per_phase_http_cap`. The stage-1 cap is
    still useful as the floor for stub-touching paths that don't
    have a phase-specific budget (notably ``run_in_sandbox`` /
    ``init_cmd`` early in lifecycle).
    """
    inner = max(
        manifest.init_timeout_s,
        manifest.setup_timeout_s,
        manifest.step_timeout_s,
        manifest.teardown_timeout_s,
    )
    return inner + _HTTP_TIMEOUT_BUFFER_S


def _effective_phase_timeout(
    record: RolloutRecord,
    catalog: TemplateCatalog,
    phase_key: str,
    manifest_default_attr: str,
) -> float:
    """Resolve a phase budget with the per-rollout snapshot taking
    precedence over the outer catalog manifest.

    H4 follow-up: when a rollout starts, ``_bootstrap_sandbox`` writes
    the post-resolver / post-init merged per-phase budgets into
    ``record.metadata['effective_{phase}_timeout_s']``. Reading from
    that snapshot lets per-call HTTP caps reflect Pattern-A per-task
    overrides (e.g. ``step_timeout_s = 1800`` for tb2's
    ``crack-7z-hash``) instead of the outer manifest's default. Falls
    back to the catalog manifest when the snapshot is absent — covers
    rollouts that started under an older code version OR were
    bootstrapped without going through ``_bootstrap_sandbox`` (some
    test fixtures).
    """
    snapshot = record.metadata.get(f"effective_{phase_key}_timeout_s")
    if isinstance(snapshot, (int, float)) and snapshot > 0:
        return float(snapshot)
    return float(getattr(catalog.get(record.template), manifest_default_attr))


def _per_phase_http_cap(phase_timeout_s: float) -> float:
    """D17 stage 2: per-call HTTP cap from one phase's manifest budget.

    Wider than the phase's inner timeout by
    :data:`_HTTP_TIMEOUT_BUFFER_S` so a legitimate-but-slow
    round-trip never hits the HTTP cap before the inner timeout
    fires. The inner timeout (the EnvAdapter's per-phase budget) is
    the workload's ground truth; the HTTP cap is the safety net for
    absent-stub scenarios. Used by
    :py:meth:`RolloutCoordinator.step` and by the env_setup /
    env_teardown call sites in :py:meth:`_bootstrap_sandbox` /
    :py:meth:`_terminate`.
    """
    return phase_timeout_s + _HTTP_TIMEOUT_BUFFER_S


def _handle_from_record(record: SandboxRecord) -> SandboxHandle:
    return SandboxHandle(
        id=record.sandbox_id,
        backend=record.backend,
        backend_ref=record.backend_ref,
        stub_endpoint=record.stub_endpoint,
    )


def _step_result_from_payload(raw: dict[str, Any]) -> StepResult:
    return StepResult(
        obs=raw.get("obs"),
        reward=float(raw.get("reward") or 0.0),
        done=bool(raw.get("done")),
        info=raw.get("info") or {},
        truncated=bool(raw.get("truncated")),
    )


def _per_grader_dict(per_grader: Any) -> dict[str, Any]:
    """Coerce a tuple[GraderResult, ...] into a JSON-serializable dict for
    trajectory.metadata.rewards. Failures keep ``score=null`` and an
    ``error`` field so the consumer / admin can see what went wrong.

    ``stdout`` / ``stderr`` (front-truncated to GRADER_OUTPUT_BYTES_CAP
    bytes by ``reward._capture_grader_output``) are surfaced so a
    ``score=0.0`` with ``error=null`` rollout — the
    "did the model fail or did the grader misfire?" case — carries
    its own diagnostic trail without a manual ``docker exec`` repro.
    """
    out: dict[str, Any] = {}
    for r in per_grader:
        entry: dict[str, Any] = {
            "score": r.score,
            "weight": r.weight,
            "error": r.error,
        }
        if r.stdout is not None:
            entry["stdout"] = r.stdout
        if r.stderr is not None:
            entry["stderr"] = r.stderr
        out[r.name] = entry
    return out


def _classify_startup_error(exc: BaseException) -> str:
    name = exc.__class__.__name__
    # A1 / D19 (P1.2) — pre-flight image-miss is a distinct
    # actionable failure from a generic image-pull error. Operators
    # see "image_missing" → run ``xrlenv warmup`` or
    # ``build-task-images.sh``; the generic image_pull_failed pointed
    # at a registry / network problem instead.
    if name == "ImageMissingOnNode":
        return "image_missing"
    if "Image" in name:
        return "image_pull_failed"
    if "Setup" in name:
        return "setup_failed"
    if "Init" in name:
        return "init_failed"
    if "Capacity" in name:
        return "over_capacity"
    return "sandbox_create_failed"


__all__ = ["RolloutCancelled", "RolloutCoordinator", "RolloutFailed"]
