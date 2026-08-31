"""Admission queue (spec 03 / spec 20 ``pending_rollouts``).

When the scheduler can't place a rollout right now (``CapacityExhausted``),
the request goes into the queue and the consumer-facing ``start_rollout`` call
blocks on a future. A background worker drains the queue:
- on every node-confirmed destroy (event-driven via :py:meth:`kick`)
- and as a safety net every ``poll_interval_s`` seconds

The queue rows live in :class:`StateStore` so they survive a control-plane
restart; in-memory waiter futures don't, which is fine — a restart's
recovery path re-issues admission for any rows whose request_id is still
valid (slice 2.5+ work).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any

from pydantic import BaseModel, ConfigDict

from xrlenv.backends.base import ResourceSpec
from xrlenv.control.scheduler import Placement, Scheduler
from xrlenv.control.state import (
    PendingRolloutRecord,
    StateStore,
    new_id,
)
from xrlenv.control.template_catalog import TemplateManifest
from xrlenv.errors import CapacityExhausted
from xrlenv.observability.metrics import MetricsRegistry

LOGGER = logging.getLogger(__name__)

# Stage-2 admission/capacity (notes/admission-stage-2-queue-clocks.md):
# the default admission-queue wait. 24 h — a backstop so a forgotten run
# can't leak a waiter forever, NOT a fail-fast deadline. Waiting in the
# queue consumes no cluster resources and is not a failure; a caller
# that wants fail-fast passes a small ``queue_timeout_s`` explicitly.
# Single-sourced here so every layer (admission, coordinators, the
# AcquireContainer endpoint) agrees.
DEFAULT_QUEUE_TIMEOUT_S: float = 86_400.0

# Fair-share throttle-warning dedup window. When an owner is at/over its per-owner cap its
# acquires are QUEUED (throttled, bounded by ``queue_timeout_s``) — correct, but silent, so it
# reads as a hang. We emit ONE warning per owner per this interval (not per parked acquire and
# not per drain scan) so the operator sees the throttle without the log flooding under a burst
# of over-cap acquires.
_OVER_CAP_WARN_INTERVAL_S: float = 60.0


class _Waiter(BaseModel):
    """One in-flight admission waiter.

    Held only in memory; the corresponding ``pending_rollouts`` row is the
    durable record. If the process crashes, the in-flight ``start_rollout``
    call dies with it — the row is recovered on next process start.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    manifest: TemplateManifest
    task_key: str | None
    future: asyncio.Future[Placement]
    backend: str | None = None
    """Per-rollout user-policy backend; ``None`` means fall back to
    :data:`xrlenv.control.defaults.DEFAULT_BACKEND` at scheduler time."""
    owner_id: str = "default"
    """Tenant this queued acquire belongs to (multi-user fair-share). The
    drain loop's per-owner cap gate reads it."""
    request_id: str | None = None
    """Stage-2 — the consumer's request id, so :py:meth:`queue_status`
    can answer a poll for this request's FIFO position."""
    container_runtime: str | None = None
    """§5.3 — the OCI runtime this queued acquire requested (e.g.
    ``sysbox-runc``). Threaded to ``Scheduler.place(container_runtime=…)``
    on both the fast path and EVERY drain retry, so a queued sysbox acquire
    is never re-placed without the per-node runtime filter. ``None`` (the
    default) is the ordinary runc placement, unchanged. In-memory only —
    raw-acquire pending recovery across a CP restart is out of scope for
    v1 (a queued sysbox acquire simply re-submits; see §5.3)."""
    reserve: ResourceSpec | None = None
    """Fleet reservation (phase 1) — when set, this queued acquire is a
    fleet-opening one and must be placed against this **footprint** (peak
    cpu+mem), not the container's own manifest resources. Threaded to
    ``Scheduler.place(reserve=…)`` on both the fast path and every drain
    retry, so a queued fleet opener admits + re-tries against the whole
    footprint. ``None`` (the default, every non-fleet acquire) is the
    legacy per-container placement, unchanged."""
    exclude_node_ids: frozenset[str] | None = None
    """D-AR-2026-07-07-B — node ids to steer *away* from on a re-admit after a
    create-time saturation 5xx / DeadlineExceeded. Stored on the waiter (not
    just applied on the fast path) so that if this acquire parks in the queue,
    EVERY drain retry re-passes the same exclusion — otherwise a queued waiter
    could drain straight back onto the still-hot node the coordinator just
    stepped off. Threaded to ``Scheduler.place(exclude_node_ids=…)``. ``None``
    (every ordinary acquire) leaves placement unchanged. The coordinator only
    ever sets this to a *proper subset* of the capable nodes — when the failed
    set would cover them all it relaxes to ``None`` first, so a queued waiter
    can still make progress on the shared pool."""


class AdmissionQueue:
    """Queue-and-await wrapper around :class:`Scheduler`.

    Coordinator calls :py:meth:`acquire` once per rollout; that returns either
    immediately (capacity available) or after the worker drains enough to
    place the rollout. ``queue_timeout_s`` (default
    :data:`DEFAULT_QUEUE_TIMEOUT_S` — 24 h) bounds how long the consumer
    is willing to wait; a small value is an explicit fail-fast opt-in.
    """

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        state: StateStore,
        poll_interval_s: float = 5.0,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._state = state
        self._poll = poll_interval_s
        self._waiters: dict[str, _Waiter] = {}
        # Per-rollout enqueue timestamp so the slow-path drain can record
        # xrlenv_queue_wait_seconds when the placement finally lands.
        self._enqueued_at: dict[str, float] = {}
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._accepting = True
        self._lock = asyncio.Lock()
        self._metrics = metrics
        # owner_id -> monotonic ts of its last over-cap throttle warning (dedup; see
        # _OVER_CAP_WARN_INTERVAL_S). Keeps a burst of over-cap acquires from flooding the log.
        self._over_cap_warned_at: dict[str, float] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="admission-worker")

    async def stop(self) -> None:
        """Cancel the worker; existing waiters are NOT cancelled here.

        Use :py:meth:`stop_accepting` first to stop new admissions while
        in-flight rollouts drain, then call :py:meth:`cancel_pending` to
        discard waiters, then call ``stop`` to tear down the worker task.
        """
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def stop_accepting(self) -> None:
        """Refuse new acquire() calls; existing waiters keep waiting."""
        self._accepting = False

    def cancel_pending(self) -> None:
        """Cancel every queued waiter and remove its sqlite row."""
        for pending_id, waiter in list(self._waiters.items()):
            if not waiter.future.done():
                waiter.future.set_exception(
                    CapacityExhausted("admission queue cancelled on shutdown")
                )
                if self._metrics is not None:
                    self._metrics.observe_admission("cancelled_in_queue")
                    self._metrics.set_queue_depth(waiter.manifest.name, 0.0)
            self._enqueued_at.pop(pending_id, None)
            with suppress(KeyError):
                self._state.remove_pending(pending_id)
        self._waiters.clear()

    # ── Consumer entry point ────────────────────────────────────────────────

    async def acquire(
        self,
        *,
        manifest: TemplateManifest,
        task_key: str | None = None,
        request_id: str | None = None,
        group_id: str | None = None,
        owner_id: str = "default",
        init_params: dict[str, Any] | None = None,
        timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
        backend: str | None = None,
        container_runtime: str | None = None,
        reserve: ResourceSpec | None = None,
        exclude_node_ids: frozenset[str] | None = None,
    ) -> Placement:
        if not self._accepting:
            if self._metrics is not None:
                self._metrics.observe_admission("rejected_full")
            raise CapacityExhausted("admission queue is no longer accepting requests")

        # A1 / D18 (P1.2): pre-fetch image presence per node so the
        # scheduler can apply the image-affinity bonus. Only when the
        # scheduler is configured for image-aware placement AND the
        # manifest carries a concrete image (Pattern A's resolver
        # already overlaid the per-instance image into ``manifest.image``
        # by the time admission sees it). Skip when we have nothing to
        # query against — Pattern A manifests where the resolver
        # didn't fire (no instance_id) keep their tag-only ``None``.
        image_present = await self._maybe_query_image_presence(
            manifest, backend, container_runtime,
        )
        # Audit P1.6.g-H2 (2026-05-05): also resolve the deferred-row
        # preferred_home so the scheduler can honor the planner's
        # spread plan on first-rollout placement. Same shape as in
        # the worker-loop drain path.
        preferred_home_node = self._lookup_preferred_home(manifest)

        # Fast path: try to place immediately. The scheduler is sync; if it
        # succeeds we never touch the queue.
        # Multi-user fair-share gate: when an owner is already at its
        # live per-owner cap, skip the immediate-place fast
        # path and park the request in the queue. The drain loop re-checks
        # the gate and admits it once the owner's running count drops below
        # the cap (e.g. one of their sandboxes is destroyed → kick). Off by
        # default — no policy configured → never blocks.
        _at_cap = self._owner_at_cap(owner_id)
        if _at_cap:
            # Throttled by the fair-share cap → park in the queue below. Warn (deduped) so this
            # reads as "throttled", not "hung".
            self._warn_over_cap(owner_id)
        if not _at_cap:
            # ``reserve`` (fleet opener) is passed only when set, so a non-fleet
            # placement is the byte-for-byte legacy call (no ``reserve`` kwarg).
            place_kwargs: dict[str, Any] = {
                "task_key": task_key, "backend": backend,
                "image_present": image_present,
                "preferred_home_node": preferred_home_node,
            }
            # §5.3 — runtime-aware placement; omitted when None so the
            # ordinary call shape is unchanged.
            if container_runtime:
                place_kwargs["container_runtime"] = container_runtime
            if reserve is not None:
                place_kwargs["reserve"] = reserve
            # D-AR-2026-07-07-B — steer the fast path away from just-failed nodes.
            if exclude_node_ids:
                place_kwargs["exclude_node_ids"] = exclude_node_ids
            try:
                placement = self._scheduler.place(manifest, **place_kwargs)
            except CapacityExhausted:
                pass
            else:
                if self._metrics is not None:
                    self._metrics.observe_admission("admitted")
                    self._metrics.observe_queue_wait(manifest.name, 0.0)
                return placement

        pending_id = new_id()
        future: asyncio.Future[Placement] = asyncio.get_running_loop().create_future()
        waiter = _Waiter(
            manifest=manifest, task_key=task_key, future=future,
            backend=backend, owner_id=owner_id, request_id=request_id,
            container_runtime=container_runtime,
            reserve=reserve,
            exclude_node_ids=exclude_node_ids,
        )
        async with self._lock:
            self._waiters[pending_id] = waiter
            self._enqueued_at[pending_id] = time.monotonic()
            self._state.enqueue_pending(
                PendingRolloutRecord(
                    pending_id=pending_id,
                    template=manifest.name,
                    init_params=dict(init_params or {}),
                    request_id=request_id,
                    task_key=task_key,
                    group_id=group_id,
                    owner_id=owner_id,
                )
            )
            depth_now = sum(
                1 for w in self._waiters.values() if w.manifest.name == manifest.name
            )
        if self._metrics is not None:
            self._metrics.observe_admission("queued")
            self._metrics.set_queue_depth(manifest.name, float(depth_now))
        self._wakeup.set()

        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except TimeoutError as exc:
            async with self._lock:
                self._waiters.pop(pending_id, None)
                self._enqueued_at.pop(pending_id, None)
                with suppress(KeyError):
                    self._state.remove_pending(pending_id)
                depth_now = sum(
                    1 for w in self._waiters.values() if w.manifest.name == manifest.name
                )
            if self._metrics is not None:
                self._metrics.observe_admission("queue_timeout")
                self._metrics.set_queue_depth(manifest.name, float(depth_now))
            raise CapacityExhausted(
                f"queue_timeout_s={timeout_s} expired waiting for capacity"
            ) from exc

    def kick(self) -> None:
        """Notify the worker that capacity may have freed (called from destroy)."""
        self._wakeup.set()

    def _owner_cap_state(self, owner_id: str) -> tuple[int, int] | None:
        """``(running, cap)`` for ``owner_id`` when a fair-share cap applies, else ``None``.

        Reads the live :class:`FairnessPolicy` and cluster-wide per-owner running counts from
        the StateStore on each call, so operator edits take effect on the next drain without a
        restart. Best-effort + off by default: any missing hook (test doubles), an empty/disabled
        policy, or an error resolves to ``None`` (no cap) — fairness never blocks unless an
        operator has explicitly configured a per-owner capacity.
        """
        get_policy = getattr(self._state, "get_fairness_policy", None)
        running_counts = getattr(self._state, "running_counts_by_owner", None)
        if get_policy is None or running_counts is None:
            return None
        try:
            policy = get_policy()
            if not policy.enabled:
                return None
            counts = running_counts()
            # Active owners are still passed for API compatibility; the current
            # policy uses a default cap plus optional owner state.
            active = set(counts) | {w.owner_id for w in self._waiters.values()}
            cap = policy.cap_for(owner_id, active)
            if cap is None:
                return None
            return (counts.get(owner_id, 0), cap)
        except Exception:
            LOGGER.exception(
                "admission: fair-share gate check failed for owner=%s; "
                "allowing (fail-open)", owner_id,
            )
            return None

    def _owner_at_cap(self, owner_id: str) -> bool:
        """Whether ``owner_id`` is at/over its fair-share cap right now (see
        :meth:`_owner_cap_state`; ``None`` state = no cap = not-at-cap)."""
        state = self._owner_cap_state(owner_id)
        return state is not None and state[0] >= state[1]

    def _warn_over_cap(self, owner_id: str) -> None:
        """Emit a DEDUPED warning that ``owner_id`` is throttled by its fair-share cap.

        A parked over-cap acquire is correct (queued, bounded by ``queue_timeout_s``) but silent
        — it reads as a hang. This makes the throttle visible while rate-limiting to at most one
        warning per owner per :data:`_OVER_CAP_WARN_INTERVAL_S`, so a burst of over-cap acquires
        (or the drain loop rescanning the same waiter every wakeup) can't flood the log."""
        now = time.monotonic()
        if now - self._over_cap_warned_at.get(owner_id, 0.0) < _OVER_CAP_WARN_INTERVAL_S:
            return
        self._over_cap_warned_at[owner_id] = now
        state = self._owner_cap_state(owner_id)
        running, cap = state if state is not None else (-1, -1)
        LOGGER.warning(
            "admission: owner=%s at its fair-share cap (running=%s >= cap=%s) — new acquires "
            "are QUEUED (throttled, bounded by queue_timeout_s), NOT hung; they admit as this "
            "owner's rollouts finish. Raise the owner's cap or lower its concurrency if this "
            "persists. (further over-cap warnings for this owner suppressed for %.0fs)",
            owner_id, running, cap, _OVER_CAP_WARN_INTERVAL_S,
        )

    def queue_status(
        self, request_id: str, owner_id: str | None = None,
    ) -> tuple[int, int, str]:
        """Stage-2 — a request's admission-queue rank.

        Returns ``(position, queue_depth, state)``: a 1-based position
        (1 = next to be admitted), the waiter count, and
        ``"queued"`` / ``"not_in_queue"``. ``_waiters`` is insertion-ordered,
        so a waiter's index is its rank. Snapshotted into a list first so a
        concurrent enqueue / drain can't mutate the dict mid-scan — this is a
        sync method and can't take the async ``_lock``.

        Multi-user (audit M1-residual): when ``owner_id`` is given, the view is
        **owner-scoped** — only that tenant's own waiters are considered, so
        ``position``/``depth`` are relative to their own queued requests and a
        request id owned by another tenant simply reads ``not_in_queue``. No
        cross-tenant existence/position is revealed. ``None`` (single-tenant /
        no-auth) keeps the original global view.
        """
        items = [
            w for w in self._waiters.values()
            if owner_id is None or w.owner_id == owner_id
        ]
        depth = len(items)
        for idx, w in enumerate(items):
            if w.request_id is not None and w.request_id == request_id:
                return (idx + 1, depth, "queued")
        return (0, depth, "not_in_queue")

    # ── Image-affinity helper (A1 / D18 — P1.2) ──────────────────────────────

    async def _maybe_query_image_presence(
        self, manifest: TemplateManifest, backend: str | None,
        container_runtime: str | None = None,
    ) -> dict[str, bool] | None:
        """Pre-fetch image presence per backend-capable node so the
        scheduler can apply the image-affinity bonus.

        Thin wrapper around the shared
        :func:`xrlenv.control.image_presence.query_image_presence`
        helper — same primitive serves the case-2/3 raw-container
        acquire path. Behaviour preserved.

        §5.3 — ``container_runtime`` narrows the presence fan-out to
        runtime-eligible nodes, so image-affinity can't steer a sysbox
        acquire toward a node that holds the image but can't run it.
        """
        from xrlenv.control.image_presence import query_image_presence
        return await query_image_presence(
            self._scheduler, manifest.image, backend=backend,
            container_runtime=container_runtime,
        )

    def _lookup_preferred_home(
        self, manifest: TemplateManifest,
    ) -> str | None:
        """Audit P1.6.g-H2 helper (2026-05-05). Returns the most-recent
        applied build plan's ``preferred_home`` node id for a row
        still in ``status="registered"`` matching ``manifest.image``,
        or ``None``.

        Cheap: a single indexed lookup against the build snapshot.
        Quietly returns ``None`` if the state store doesn't support
        the lookup (older test doubles), or the manifest has no
        concrete image — both cases collapse to "no preferred-home
        signal" and the scheduler scores normally.
        """
        if not manifest.image:
            return None
        finder = getattr(
            self._state, "find_registered_preferred_home", None,
        )
        if finder is None:
            return None
        try:
            result = finder(manifest.image)
        except Exception:
            LOGGER.exception(
                "preferred-home lookup failed for image=%s; "
                "falling through to non-affinity scoring",
                manifest.image,
            )
            return None
        return str(result) if result is not None else None

    # ── Worker loop ──────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while True:
            await self._drain_once()
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._poll)
            self._wakeup.clear()

    async def _drain_once(self) -> None:
        # Snapshot so we can iterate without holding the lock during scheduler
        # work (the scheduler itself reads StateStore — keep contention low).
        async with self._lock:
            snapshot = list(self._state.list_pending())

        for row in snapshot:
            waiter = self._waiters.get(row.pending_id)
            if waiter is None:
                # Stale row from a prior process; drop it.
                async with self._lock:
                    with suppress(KeyError):
                        self._state.remove_pending(row.pending_id)
                continue
            if waiter.future.done():
                async with self._lock:
                    self._waiters.pop(row.pending_id, None)
                    with suppress(KeyError):
                        self._state.remove_pending(row.pending_id)
                continue
            # Multi-user fair-share gate: leave this waiter queued while its
            # owner is at/over the (live) per-owner cap, so a hog can't keep
            # draining ahead of others. ``continue`` mirrors the
            # CapacityExhausted path — the loop keeps scanning, so a *different*
            # owner's waiter behind this one still gets placed (no
            # head-of-line block). Off by default (no policy → never blocks).
            if self._owner_at_cap(waiter.owner_id):
                self._warn_over_cap(waiter.owner_id)  # deduped — safe to call every scan
                continue
            image_present = await self._maybe_query_image_presence(
                waiter.manifest, waiter.backend, waiter.container_runtime,
            )
            # Audit P1.6.g-H2 (2026-05-05): when the manifest's image
            # is currently a deferred (registered) row in the build
            # snapshot, hand the row's preferred_home node to the
            # scheduler so first-rollout placement honors the
            # bin-packer's spread plan. Soft preference — the
            # scheduler scoring still picks a non-preferred node if
            # the preferred_home is meaningfully more loaded.
            preferred_home_node = self._lookup_preferred_home(
                waiter.manifest,
            )
            drain_place_kwargs: dict[str, Any] = {
                "task_key": waiter.task_key,
                "backend": waiter.backend,
                "image_present": image_present,
                "preferred_home_node": preferred_home_node,
            }
            # §5.3 — re-pass the queued acquire's runtime on EVERY drain
            # retry, so a queued sysbox acquire can't be re-placed without
            # the per-node runtime filter.
            if waiter.container_runtime:
                drain_place_kwargs["container_runtime"] = (
                    waiter.container_runtime
                )
            if waiter.reserve is not None:
                drain_place_kwargs["reserve"] = waiter.reserve
            # D-AR-2026-07-07-B — re-pass the re-admit exclusion on EVERY drain
            # retry, so a queued waiter can't drain back onto the hot node the
            # coordinator just stepped off of.
            if waiter.exclude_node_ids:
                drain_place_kwargs["exclude_node_ids"] = waiter.exclude_node_ids
            try:
                placement = self._scheduler.place(
                    waiter.manifest, **drain_place_kwargs,
                )
            except CapacityExhausted:
                continue  # leave in queue, retry on next wakeup
            except Exception as exc:
                waiter.future.set_exception(exc)
                async with self._lock:
                    self._waiters.pop(row.pending_id, None)
                    with suppress(KeyError):
                        self._state.remove_pending(row.pending_id)
                continue

            # The waiter may have given up between snapshot capture and
            # this set_result (e.g. ``acquire`` hit its queue_timeout
            # while we were inside the scheduler). If we just dropped
            # the placement without releasing the reservation, the
            # scheduler's pending-load count would leak forever and
            # block future placements with the same task_key. Release
            # before discarding.
            if waiter.future.done():
                self._scheduler.release_placement(placement)
                async with self._lock:
                    self._waiters.pop(row.pending_id, None)
                    with suppress(KeyError):
                        self._state.remove_pending(row.pending_id)
                continue
            waiter.future.set_result(placement)
            enqueued_at = self._enqueued_at.pop(row.pending_id, None)
            async with self._lock:
                self._waiters.pop(row.pending_id, None)
                with suppress(KeyError):
                    self._state.remove_pending(row.pending_id)
                depth_now = sum(
                    1
                    for w in self._waiters.values()
                    if w.manifest.name == waiter.manifest.name
                )
            if self._metrics is not None:
                self._metrics.observe_admission("admitted")
                if enqueued_at is not None:
                    self._metrics.observe_queue_wait(
                        waiter.manifest.name, time.monotonic() - enqueued_at
                    )
                self._metrics.set_queue_depth(waiter.manifest.name, float(depth_now))

    # ── Drain helper for shutdown ────────────────────────────────────────────

    async def wait_idle(self, timeout_s: float) -> bool:
        """Block until no waiters remain or the timeout fires.

        Returns ``True`` if the queue drained naturally; ``False`` on timeout.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._waiters:
                return True
            await asyncio.sleep(0.05)
        return not self._waiters


__all__ = ["AdmissionQueue"]
