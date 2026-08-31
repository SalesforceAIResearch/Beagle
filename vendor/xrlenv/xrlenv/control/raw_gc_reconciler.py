"""P1.7.A.2 — Raw-container GC reconciler (Raw-GC-M1 closure).

Parallel to :class:`xrlenv.control.gc_reconciler.GCReconciler` but
for raw containers. The case-1 reconciler diffs the StateStore
``sandbox`` rows against each node's ``list_sandbox_ids`` reply;
this one diffs ``RawContainerCoordinator.list_sessions()``
(coordinator's in-memory truth) against each node's
``list_raw_container_ids`` reply (docker's truth, label-filtered
on ``xrlenv.session_kind=raw``).

Two orphan classes:

- **Node-only orphan**: docker reports a raw container with no
  owning coordinator session. Confirmed across two consecutive
  sweeps before any action (issue #18 fix A) so a container
  created mid-acquire — node-only only until its acquire reply
  registers the session — is never reaped. A leftover from a
  CONFIRMED ``destroy`` whose node-side removal lingered has an
  already-terminal ``raw_rollouts`` row and is logged at INFO as
  routine deferred teardown; a genuinely unexplained container
  logs at WARNING (issue #18 fix B). (Note post-H8: a FAILED or
  TIMED-OUT destroy instead RETAINS the session — it stays
  coordinator-tracked, so it's a coordinator-only case below, not
  node-only.) Reaped via the privileged
  ``force_destroy_raw_container`` (matching spec-21
  ``ForceDestroyContainerCommand``) — the regular
  ``destroy_container`` path would refuse the destroy on the
  node-side ownership check, since the coordinator has no
  record of the orphan's rollout_id. The privileged path is
  internal infrastructure: not exposed on
  ``rollout_control.proto``, so consumers can't reach it.

- **Coordinator-only orphan**: coordinator has a session whose
  ``container_id`` isn't in the node's reply. Likely the harness
  ``docker rm``-ed the container directly, the docker daemon lost
  it, or the node autonomously reaped it (disk guard / OOM).
  Sealed via ``RawContainerCoordinator.seal_orphan`` — the durable
  ``raw_rollouts`` row goes ``reaped`` when the node reported a
  real reap cause, else ``released`` — and the in-memory session is
  dropped so its capacity charge is freed. If ``seal_orphan``
  raises, ``drop_orphan_session`` seals the row the same way before
  dropping (it must: ``SessionReaped`` is only reachable for a row
  that says ``reaped``).

Neither class emits a structured ``state.append_event`` record the
way :class:`GCReconciler` does for case-1 sandboxes; both are
surfaced through this module's WARNING/INFO logs and the
``reconcile_once`` report.

Lifecycle mirrors :class:`GCReconciler`: ``await
reconciler.start()`` from runtime startup, ``await
reconciler.shutdown()`` from runtime shutdown. Same ``interval_s``
default (60s); the per-node timeout defaults to 60s here, twice the
case-1 reconciler's 30s, because a raw node's inventory listing can
queue behind cold image pulls.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from xrlenv.errors import ManagedInventoryUnsupported, XRLEnvError

if TYPE_CHECKING:
    from xrlenv.control.node_registry import NodeRegistry
    from xrlenv.control.raw_container_service import RawContainerCoordinator
    from xrlenv.control.state import RawRolloutRecord, StateStore
    from xrlenv.observability.metrics import MetricsRegistry

from xrlenv.control.raw_container_service import (
    RAW_LOST_CP_MARKERS,
    RAW_LOST_CP_PREFIX,
)

LOGGER = logging.getLogger(__name__)

# Max consumer-liveness reaps per sweep, so a mass die-off (e.g. a single
# consumer holding 1k sessions crashing) doesn't fire 1k destroys at once —
# the rest reap on the next sweep. Env-tunable.
_LIVENESS_REAP_BATCH: int = int(
    __import__("os").environ.get("XRLENV_RAW_LIVENESS_REAP_BATCH", "50"),
)


class RawGCReconciler:
    """Spec 09 GC layer 3 driver, raw-container variant.

    Background loop fans out :py:meth:`list_raw_container_ids` to
    every connected node, diffs against the coordinator's
    in-memory session set, and dispatches per-orphan handlers.
    Single-node failures don't stop the sweep on others — each
    node is tried inside a ``try/except`` so a hung RPC can't
    pin the loop.
    """

    def __init__(
        self,
        *,
        registry: NodeRegistry,
        coordinator: RawContainerCoordinator,
        interval_s: float = 60.0,
        per_node_timeout_s: float = 60.0,
        state: StateStore | None = None,
        running_stale_s: float = 60.0,
        readopt_grace_s: float = 300.0,
        fleet_reservation_ttl_s: float | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._metrics = metrics
        self._coordinator = coordinator
        self._interval_s = interval_s
        self._per_node_timeout_s = per_node_timeout_s
        # Fleet reservation (phase 1) reclaim TTL — a persisted reservation row
        # with no live members older than this is reclaimed once we're past the
        # re-adoption grace (so a slow-reconnecting node's fleet isn't dropped).
        from xrlenv.control.raw_container_service import (
            FLEET_RESERVATION_TTL_DEFAULT_S,
        )
        self._fleet_reservation_ttl_s = (
            fleet_reservation_ttl_s
            if fleet_reservation_ttl_s is not None
            else FLEET_RESERVATION_TTL_DEFAULT_S
        )
        # WS3 — control-plane restart resilience. After a restart the
        # in-memory ``_sessions`` map is empty but the durable
        # ``raw_rollouts`` rows + node-side containers survive. The
        # sweep re-adopts those running containers (rebuilds the
        # session) instead of force-destroying live work. Until a row
        # has had a chance to be re-adopted — bounded by this grace,
        # measured from reconciler start, NOT row age (a long-running
        # rollout's row is legitimately old) — the SQLite seal protects
        # ``running`` rows that carry a ``container_id`` (the
        # re-adoptable ones) from being flipped ``lost-on-restart``.
        # After the grace, an un-re-adopted such row means its node
        # never came back; it's then sealed normally.
        self._readopt_grace_s = readopt_grace_s
        self._started_at: float = 0.0
        # Issue #18 fix #3: SQLite ``raw_rollouts`` reconciler. When a
        # StateStore is wired, the reconciler additionally scans for
        # non-terminal rows whose ``rollout_id`` isn't in the
        # coordinator's in-memory ``_sessions`` map and seals them as
        # ``failed (lost-…)``. Without this, a control-plane restart
        # (or any path that drops a session without sealing its row)
        # leaves ``acquiring`` / ``running`` rows in raw_rollouts forever,
        # surfacing as phantom rollouts on the admin panel. ``state=
        # None`` preserves the legacy behaviour for tests that don't
        # exercise this path.
        self._state = state
        # ``running_stale_s`` guards the sub-second race between a
        # session landing in ``_sessions`` and the ``status="running"``
        # SQLite write — a ``running`` row younger than this is given
        # the benefit of the doubt. ``acquiring`` rows are NOT
        # age-gated: liveness comes from the coordinator's
        # ``list_acquiring_ids`` set (audit M1, round 2), which tracks
        # the ``acquire`` coroutine directly and so needs no time
        # proxy — a request parked in the admission queue is protected
        # for as long as it's genuinely in flight, regardless of the
        # consumer's ``queue_timeout_s`` / ``acquire_timeout_s``.
        self._running_stale_s = running_stale_s
        # Issue #18 fix A — two-sweep confirmation: container_ids seen
        # node-only on the *previous* sweep, per node. A container is
        # only force-destroyed once observed node-only on two
        # consecutive sweeps, so a container created mid-acquire
        # (node-only only until its acquire reply registers the
        # session) is never reaped as an orphan.
        self._prev_node_only: dict[str, set[str]] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None:
            # WS3 — stamp start time BEFORE the startup sweep so the
            # re-adoption grace is measured from here, protecting
            # still-running rows until their nodes can reconnect.
            self._started_at = time.time()
            # Issue #18 fix #3: drain the SQLite ghost set BEFORE
            # the periodic loop starts. A coordinator restart wipes
            # ``_sessions`` but leaves any ``acquiring`` / ``running``
            # rows in raw_rollouts; the startup sweep flips them to
            # ``failed (lost-on-restart)`` so the admin panel doesn't
            # surface phantom in-flight rollouts. Best-effort —
            # failure logs but doesn't block lifecycle start.
            if self._state is not None:
                try:
                    self._reconcile_sqlite(reason=RAW_LOST_CP_MARKERS[0])
                except Exception:
                    LOGGER.exception(
                        "raw-gc-reconciler: startup SQLite sweep raised; "
                        "skipping (periodic sweep will retry)",
                    )
            self._task = asyncio.create_task(
                self._loop(), name="raw-gc-reconciler",
            )

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ── Loop / sweep ────────────────────────────────────────────────────────

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
                        "raw-gc-reconciler: sweep raised; "
                        "will retry next interval",
                    )
        except asyncio.CancelledError:
            raise

    async def reconcile_once(self) -> dict[str, dict[str, int]]:
        """One sweep across all connected nodes.

        Returns ``{node_id: {"node_only": N, "coordinator_only": M}}``
        per node for observability (admin panel + tests).
        """
        report: dict[str, dict[str, int]] = {}
        node_ids = list(self._registry.node_ids)
        # Coordinator-side truth: ``container_id → session`` for all
        # tracked sessions. We also need the per-node mapping to
        # detect coordinator-only orphans on the right node.
        sessions = self._coordinator.list_sessions()
        sessions_by_node: dict[str, dict[str, str]] = {}
        for session in sessions:
            sessions_by_node.setdefault(session.node_id, {})[
                session.container_id
            ] = session.rollout_id

        # Issue #18 fix B — index raw_rollouts rows by container_id so
        # a node-only container can be classified before it is logged:
        # a leftover from a ``destroy`` whose node-side removal timed
        # out has an already-terminal row, and should read as routine
        # cleanup rather than a scary "orphan". The row carries
        # container_id only once the acquire reply has registered it —
        # exactly the rows worth classifying.
        raw_rows_by_container: dict[str, RawRolloutRecord] = {}
        if self._state is not None:
            try:
                for _row in self._state.list_raw_rollouts():
                    if _row.container_id:
                        raw_rows_by_container[_row.container_id] = _row
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: list_raw_rollouts failed; node-"
                    "only containers this sweep can't be classified",
                )

        # P1.7.C.2 — persisted compose-project rows keyed by rollout_id, for
        # restart re-adoption (correlate a node compose-main container's
        # ``xrlenv.rollout_id`` label to its footprint/subnet-claim row).
        compose_rows_by_rollout: dict[str, Any] = {}
        readopt_compose = getattr(self._coordinator, "readopt_compose_project", None)
        if self._state is not None and readopt_compose is not None:
            list_fn = getattr(self._state, "list_compose_projects", None)
            if list_fn is not None:
                try:
                    compose_rows_by_rollout = {r.rollout_id: r for r in list_fn()}
                except Exception:
                    LOGGER.exception(
                        "raw-gc-reconciler: list_compose_projects failed",
                    )

        for node_id in node_ids:
            transport = self._registry.get(node_id)
            if transport is None:
                continue  # raced disconnect
            try:
                docker_ids = await asyncio.wait_for(
                    transport.list_raw_container_ids(),  # type: ignore[attr-defined]
                    timeout=self._per_node_timeout_s,
                )
            except TimeoutError:
                # Issue #18: a docker daemon swamped by concurrent
                # multi-GB cold pulls answers ``docker ps`` slowly.
                # This is an expected, self-healing condition — the
                # node is skipped and retried next sweep — not an
                # error, so it logs at WARNING without a traceback.
                # The per-node ``wait_for`` is set below the RPC's own
                # ceiling so the failure always surfaces here as a
                # clean ``TimeoutError`` rather than the RPC's error.
                LOGGER.warning(
                    "raw-gc-reconciler: list_raw_container_ids timed out "
                    "for node=%s after %.0fs; skipping it this sweep "
                    "(will retry next interval)",
                    node_id, self._per_node_timeout_s,
                )
                continue
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: list_raw_container_ids failed "
                    "for node=%s; skipping this node for this sweep",
                    node_id,
                )
                continue
            docker_set = set(docker_ids)
            coord_map = sessions_by_node.get(node_id, {})
            coord_set = set(coord_map.keys())

            node_only = docker_set - coord_set
            coordinator_only = coord_set - docker_set

            # WS3 — re-adopt restart survivors BEFORE orphan
            # classification. After a control-plane restart _sessions is
            # empty, so a still-running container looks node-only. If it
            # still has a non-terminal (running/acquiring) raw_rollouts
            # row, the CP restarted under a live rollout: rebuild the
            # session from the durable row + this reconnected node
            # instead of force-destroying live work and sealing the
            # rollout lost-on-restart. Re-adopted ids leave node_only so
            # they're neither force-destroyed nor counted as orphans.
            # P1.7.C.2 — per-container correlation labels the node reported this
            # sweep (container_id -> (rollout_id, compose_project)). Used both to
            # re-adopt compose-project restart survivors AND to route an orphan
            # compose-main to a whole-project teardown (never a bare single-
            # container force-destroy that would leak its session_kind=compose
            # sidecars — they're off the raw diff, so nothing else would reap them).
            container_info: dict[str, tuple[str, str]] = getattr(
                transport, "_last_container_info", {},
            )

            readopt = getattr(self._coordinator, "readopt", None)
            if self._state is not None and node_only and readopt is not None:
                readopted: set[str] = set()
                for cid in node_only:
                    # A compose PROJECT main (``compose_project`` label set) with a
                    # persisted project row re-adopts the WHOLE project (session +
                    # _compose_projects) so teardown downs the project, not just
                    # main. Fall through to the plain single-container readopt
                    # otherwise.
                    rid_lbl, proj_lbl = container_info.get(cid, ("", ""))
                    compose_row = (
                        compose_rows_by_rollout.get(rid_lbl) if proj_lbl else None
                    )
                    try:
                        if compose_row is not None and readopt_compose is not None:
                            if await readopt_compose(compose_row, cid, transport):
                                readopted.add(cid)
                            continue
                        row = raw_rows_by_container.get(cid)
                        if row is None or row.status not in (
                            "running", "acquiring",
                        ):
                            continue
                        if await readopt(row, transport):
                            readopted.add(cid)
                    except Exception:
                        LOGGER.exception(
                            "raw-gc-reconciler: readopt failed node=%s "
                            "container=%s; will retry next sweep",
                            node_id, cid[:12],
                        )
                if readopted:
                    LOGGER.info(
                        "raw-gc-reconciler: re-adopted %d restart-survivor "
                        "container(s) on node=%s (control-plane restart)",
                        len(readopted), node_id,
                    )
                    node_only = node_only - readopted

            # Issue #18 fix A — two-sweep confirmation. Only act on a
            # container observed node-only on the previous sweep too.
            # A container created mid-acquire is node-only for the gap
            # between docker-create and the acquire reply registering
            # its session; by the next sweep it has a session and is
            # no longer in ``node_only`` — so it is never reaped.
            prev_node_only = self._prev_node_only.get(node_id, set())
            confirmed_node_only = node_only & prev_node_only
            self._prev_node_only[node_id] = node_only

            # ``node_only`` here is the count *observed* this sweep;
            # only ``confirmed_node_only`` is acted on (force-destroyed).
            report[node_id] = {
                "node_only": len(node_only),
                "coordinator_only": len(coordinator_only),
            }

            for container_id in confirmed_node_only:
                rid_lbl, proj_lbl = container_info.get(container_id, ("", ""))
                await self._handle_node_only(
                    node_id, container_id, transport,
                    row=raw_rows_by_container.get(container_id),
                    compose_rollout_id=rid_lbl or None,
                    compose_project=proj_lbl or None,
                )
            # Audit P3 — the node reports WHY it autonomously reaped a
            # rollout (disk guard) alongside the container list; seal the
            # coordinator-only orphan with that real cause instead of a
            # generic teardown message.
            reaped_reasons = getattr(transport, "_last_reaped_reasons", {})
            for container_id in coordinator_only:
                rollout_id = coord_map[container_id]
                await self._handle_coordinator_only(
                    node_id, container_id, rollout_id,
                    reason=reaped_reasons.get(rollout_id),
                )

        # Fleet reservation (phase 1) — rebuild + reclaim. Runs AFTER the
        # per-node re-adoption above so the live fleet-member sessions exist:
        # reconstruct in-memory ``_fleets`` from the persisted footprint rows +
        # those live members, and reclaim any reservation row with no live
        # members past its TTL. ``allow_reclaim`` is gated on the re-adoption
        # grace so a still-reconnecting node's fleet isn't dropped prematurely.
        rebuild_fn = getattr(
            self._coordinator, "rebuild_fleets_from_state", None,
        )
        if rebuild_fn is not None:
            fleet_now = time.time()
            allow_reclaim = (
                fleet_now - self._started_at
            ) >= self._readopt_grace_s
            try:
                rebuilt, reclaimed = await rebuild_fn(
                    now=fleet_now,
                    reclaim_after_s=self._fleet_reservation_ttl_s,
                    allow_reclaim=allow_reclaim,
                )
                report["__fleets__"] = {
                    "rebuilt": rebuilt, "reclaimed": reclaimed,
                }
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: fleet rebuild/reclaim raised; "
                    "will retry next interval",
                )

        # P1.7.C.2 — reclaim stale compose-project rows (the reclaim backstop;
        # the rebuild half is the per-node ``readopt_compose_project`` above).
        # Shares the fleet re-adoption grace so a still-reconnecting node's
        # project isn't reclaimed before it can be re-adopted.
        reap_compose_fn = getattr(
            self._coordinator, "reap_stale_compose_projects", None,
        )
        if reap_compose_fn is not None:
            compose_now = time.time()
            allow_reclaim = (
                compose_now - self._started_at
            ) >= self._readopt_grace_s
            try:
                reclaimed = await reap_compose_fn(
                    now=compose_now,
                    reclaim_after_s=self._fleet_reservation_ttl_s,
                    allow_reclaim=allow_reclaim,
                )
                report["__compose__"] = {"reclaimed": reclaimed}
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: compose reclaim raised; "
                    "will retry next interval",
                )

        # Issue #18 fix #3: SQLite ghost sweep — independent of the
        # docker diff above. Catches rows whose status is non-
        # terminal but whose session is gone from in-memory state
        # (control-plane crash mid-acquire, an exec-timeout cascade that
        # dropped the session without sealing its row, etc.). Reported per
        # sweep as ``report["__sqlite__"]["ghosts"]`` so dashboards can
        # graph it.
        if self._state is not None:
            try:
                ghost_count = self._reconcile_sqlite(reason=RAW_LOST_CP_MARKERS[1])
                report["__sqlite__"] = {"ghosts": ghost_count}
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: SQLite sweep raised; "
                    "will retry next interval",
                )

        # Issue #18: deadline sweep — force-reap raw sessions past
        # their wall-clock deadline. This is the ONLY thing that
        # cleans up a session abandoned by a dead consumer: the
        # docker diff above sees the session in both _sessions AND
        # node-docker (they agree → "healthy"), and the SQLite sweep
        # sees the rollout_id in _sessions (→ not a ghost). Neither
        # detects abandonment — only the deadline does.
        try:
            reaped = await self._reconcile_deadlines()
            report["__deadlines__"] = {"reaped": reaped}
        except Exception:
            LOGGER.exception(
                "raw-gc-reconciler: deadline sweep raised; "
                "will retry next interval",
            )

        # Always present, so a caller reading the report can't KeyError on the
        # path where the sweep below raises.
        report["__liveness__"] = {"reaped": 0, "suspect": 0}
        try:
            # Phase 1 (mark) before phase 2 (destroy): a session that crossed the
            # quarantine this sweep without ever being marked still gets its
            # WARNING, so no reap is silent in the log.
            suspected = self._mark_liveness_suspects()
            live_reaped = await self._reconcile_liveness()
            report["__liveness__"] = {"reaped": live_reaped, "suspect": suspected}
        except Exception:
            LOGGER.exception(
                "raw-gc-reconciler: liveness sweep raised; "
                "will retry next interval",
            )
        finally:
            # In `finally`, NOT inside the `try`: a sweep that raises still
            # changed the suspect set (marking runs before the destroy that
            # failed), and if the failure repeats every interval the gauge would
            # stay wrong indefinitely rather than for one interval.
            self._sync_suspect_gauge()

        return report

    async def readopt_node_on_connect(self, transport: Any) -> bool:
        """H11 — re-adopt a just-(re)connected node's surviving raw + compose sessions from
        durable state BEFORE it is made schedulable.

        After a control-plane restart ``_sessions`` is empty, so admitting the node for placement
        immediately (``scheduler.add_node``) would let ``iter_load_entries`` report no surviving
        raw/compose load and admission could OVER-PLACE it — violating "capacity is released only
        on node-confirmed destroy". This re-adopts THIS node's live load synchronously on connect.

        LEAKED ORPHANS ARE REAPED, NOT QUARANTINED (audit H11, deploy-safety fix): a managed
        container with no LIVE re-adoptable row (a TERMINAL row, or no row) is an unambiguous
        orphan — a CP restart/deploy seals in-flight rollouts ``failed`` on shutdown-node-loss and
        their containers linger. We force-destroy such a raw orphan (or whole-project-down a
        compose orphan) and STILL admit, so the node carries no uncharged load. (Was: fail closed
        on any unmatched survivor → the node stayed UNSCHEDULABLE forever, because readopt only
        re-runs on reconnect and the periodic sweep's later reap never re-triggered it — every
        deploy that left terminal-row containers quarantined the whole fleet.)

        FAIL-CLOSED only on genuine ambiguity/error (returns ``False`` → caller does NOT admit,
        retries): an inventory/state-read failure; a persisted ``node_id`` that isn't this
        transport (corruption — don't route to the wrong node); an in-flight ACQUIRE for a
        surviving container (transient — reaping it would kill live work); a reap that can't be
        CONFIRMED (force-destroy / whole-project-down RPC error); a fleet-table read/rebuild
        failure; or (older agent) a missing broad managed-inventory capability. Returns ``True``
        once every survivor is either re-adopted or confirmed-reaped."""
        node_id = getattr(transport, "node_id", "?")
        if self._state is None:
            return True   # no durable state → nothing to account → safe to admit
        readopt = getattr(self._coordinator, "readopt", None)
        readopt_compose = getattr(self._coordinator, "readopt_compose_project", None)
        if readopt is None:
            return True

        def _is_current() -> bool:
            # audit H11 — this pass may only TRANSFER an existing session onto ``transport`` while
            # ``transport`` is STILL the node's current registered stream. A delayed OLD pass
            # (already superseded by a replacement) must not steal the session back and then
            # delete it via its failed-pass rollback.
            return self._registry.get(node_id) is transport
        try:
            docker_ids = await asyncio.wait_for(
                transport.list_raw_container_ids(),
                timeout=self._per_node_timeout_s,
            )
        except Exception:
            LOGGER.warning(
                "raw-gc-reconciler: readopt-on-connect list_raw_container_ids failed for "
                "node=%s — cannot determine surviving load; NOT admitting (fail closed, H11)",
                node_id,
            )
            return False
        container_info: dict[str, tuple[str, str]] = getattr(
            transport, "_last_container_info", {},
        )
        try:
            raw_rows_by_container = {
                r.container_id: r
                for r in self._state.list_raw_rollouts()
                if r.container_id
            }
        except Exception:
            LOGGER.warning(
                "raw-gc-reconciler: readopt-on-connect list_raw_rollouts failed for node=%s — "
                "NOT admitting (fail closed, H11)", node_id,
            )
            return False
        compose_rows_by_rollout: dict[str, Any] = {}
        if readopt_compose is not None:
            list_fn = getattr(self._state, "list_compose_projects", None)
            if list_fn is not None:
                try:
                    compose_rows_by_rollout = {r.rollout_id: r for r in list_fn()}
                except Exception:
                    LOGGER.warning(
                        "raw-gc-reconciler: readopt-on-connect list_compose_projects failed for "
                        "node=%s — NOT admitting (fail closed, H11)", node_id,
                    )
                    return False
        ok = True
        for cid in docker_ids:
            rid_lbl, proj_lbl = container_info.get(cid, ("", ""))
            compose_row = compose_rows_by_rollout.get(rid_lbl) if proj_lbl else None
            try:
                if compose_row is not None and readopt_compose is not None:
                    # Ownership: the persisted row must belong to THIS node (corrupt/stale
                    # inventory otherwise → don't route a session through the wrong transport).
                    if getattr(compose_row, "node_id", node_id) != node_id:
                        LOGGER.warning(
                            "raw-gc-reconciler: compose row rollout=%s node=%s != connecting "
                            "node=%s — skipping (fail closed, H11)",
                            rid_lbl, getattr(compose_row, "node_id", "?"), node_id,
                        )
                        ok = False
                        continue
                    if not await readopt_compose(
                        compose_row, cid, transport, is_current=_is_current,
                    ):
                        # readopt returned False → an in-flight acquire owns this rollout (or a
                        # row we can't route). The node reports a live main we could NOT transfer
                        # ownership of; admitting now would leave it routing through a stale/absent
                        # generation. Fail closed (audit H11) — retry once the acquire settles.
                        LOGGER.warning(
                            "raw-gc-reconciler: compose main container=%s (rollout=%s) could not "
                            "be re-adopted onto this transport — NOT admitting (fail closed, H11)",
                            cid[:12], rid_lbl,
                        )
                        ok = False
                    continue
                if proj_lbl:
                    # A compose-labeled MAIN whose project row is missing/terminal — a leaked
                    # compose orphan (deploy-safety fix, as for raw above). REAP the WHOLE project
                    # (main + sidecars) via the node's whole-project teardown (idempotent; handles
                    # an unregistered project by the project+rollout labels) and ADMIT — rather than
                    # quarantining the node forever. Fail closed only if the teardown can't be
                    # confirmed.
                    try:
                        await transport.destroy_compose_project(
                            rollout_id=rid_lbl, project_name=proj_lbl,
                        )
                        LOGGER.info(
                            "raw-gc-reconciler: reaped leaked compose-main orphan container=%s "
                            "(rollout=%s project=%s) on node=%s via whole-project down during "
                            "readopt — admitting (H11)", cid[:12], rid_lbl, proj_lbl, node_id,
                        )
                    except Exception:
                        LOGGER.warning(
                            "raw-gc-reconciler: could NOT reap leaked compose orphan project=%s "
                            "(rollout=%s) on node=%s — NOT admitting (fail closed, retry) (H11)",
                            proj_lbl, rid_lbl, node_id,
                        )
                        ok = False
                    continue
                row = raw_rows_by_container.get(cid)
                if row is None or row.status not in ("running", "acquiring"):
                    # LEAKED RAW ORPHAN (audit H11, deploy-safety fix): the node reports a managed
                    # raw container (session_kind=raw) with no live re-adoptable row — either a
                    # TERMINAL row (the rollout is already sealed failed/released/… — the classic
                    # deploy case: a CP restart's shutdown-node-loss seals in-flight rollouts
                    # ``failed``, then their containers linger on the node) or NO row at all. Both
                    # are UNAMBIGUOUS orphans (no one owns the rollout), so REAP the container
                    # (force-destroy — the same privileged path the periodic raw-GC sweep uses)
                    # and ADMIT: the node then carries no uncharged load. (Was: fail closed → the
                    # node stayed UNSCHEDULABLE *forever*, because readopt only re-runs on
                    # reconnect and the periodic sweep's later reap never re-triggered it — a
                    # deploy that left terminal-row containers quarantined every such node.) Only
                    # an in-flight ACQUIRE (running/acquiring row) is left to the re-adopt path
                    # below. If the reap can't be confirmed (RPC error) → fail closed + retry.
                    try:
                        await transport.force_destroy_raw_container(
                            container_id=cid,
                        )
                        LOGGER.info(
                            "raw-gc-reconciler: reaped leaked raw orphan container=%s (rollout=%s "
                            "row=%s) on node=%s during readopt — admitting (H11)",
                            cid[:12], rid_lbl or "?",
                            row.status if row is not None else "none", node_id,
                        )
                    except Exception:
                        LOGGER.warning(
                            "raw-gc-reconciler: could NOT reap leaked raw orphan container=%s on "
                            "node=%s — NOT admitting (fail closed, retry) (H11)", cid[:12], node_id,
                        )
                        ok = False
                    continue
                if row.node_id != node_id:
                    LOGGER.warning(
                        "raw-gc-reconciler: raw row rollout=%s node=%s != connecting node=%s — "
                        "skipping (fail closed, H11)", row.rollout_id, row.node_id, node_id,
                    )
                    ok = False
                    continue
                if not await readopt(row, transport, is_current=_is_current):
                    # readopt returned False → an in-flight acquire owns this rollout (or a row
                    # missing container/node identity). The node reports a live managed container
                    # we could NOT transfer ownership of — admitting would route it through a
                    # stale/absent generation. Fail closed (audit H11), retry when it settles.
                    LOGGER.warning(
                        "raw-gc-reconciler: raw container=%s (rollout=%s) could not be re-adopted "
                        "onto this transport — NOT admitting (fail closed, H11)",
                        cid[:12], rid_lbl or row.rollout_id,
                    )
                    ok = False
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: readopt-on-connect failed node=%s container=%s (H11)",
                    node_id, cid[:12],
                )
                ok = False
        # ── Sidecar-only compose survivor sweep (audit H11) ──────────────────────────────
        # The raw inventory above lists only ``session_kind=raw`` (single containers + compose
        # MAINS). A compose project whose main is GONE but whose SIDECARS
        # (``session_kind=compose``) are still alive is invisible to it — the node would look
        # clean and get admitted while those sidecars hold uncharged cpu/mem/disk. Query the
        # BROADER managed inventory (every ``xrlenv.rollout_id``-labelled container, incl
        # sidecars). Any managed container whose rollout has NO live session on this node after
        # re-adoption (its whole project couldn't be re-adopted) is REAPED — whole-project
        # teardown for a compose member, force-destroy for a plain raw container — and then the
        # node is ADMITTED, rather than quarantined forever. Fail closed only when the survivor
        # can't be dealt with: the reap isn't confirmed, or an in-flight acquire owns the rollout
        # (retry once it settles). An older node that can't serve the broader listing raises
        # ``ManagedInventoryUnsupported`` → fail closed (its raw-only view can't rule out
        # sidecar-only survivors). RPC failure → fail closed for the same reason.
        list_managed = getattr(transport, "list_managed_container_info", None)
        if list_managed is not None:
            try:
                managed = await asyncio.wait_for(
                    list_managed(), timeout=self._per_node_timeout_s,
                )
            except ManagedInventoryUnsupported:
                # audit H11 — an OLDER agent that can't return the broad inventory. Do NOT trust
                # its raw-only view as "clean" (it hides sidecar-only survivors) — fail closed.
                LOGGER.critical(
                    "raw-gc-reconciler: node=%s agent does NOT support the broad managed-container "
                    "inventory (older agent) — cannot rule out sidecar-only compose survivors; NOT "
                    "admitting (fail closed, H11). Upgrade the node agent to the CP version.",
                    node_id,
                )
                return False
            except Exception:
                LOGGER.warning(
                    "raw-gc-reconciler: readopt-on-connect list_managed_container_info failed "
                    "for node=%s — cannot rule out sidecar-only survivors; NOT admitting "
                    "(fail closed, H11)", node_id,
                )
                return False
            accounted = {
                s.rollout_id for s in self._coordinator.list_sessions()
                if getattr(s, "node_id", None) == node_id
            }
            acquiring = set()
            _acq = getattr(self._coordinator, "list_acquiring_ids", None)
            if _acq is not None:
                try:
                    acquiring = set(_acq())
                except Exception:
                    acquiring = set()
            for cid, rid_lbl, proj_lbl, kind in managed:
                if rid_lbl and rid_lbl in accounted:
                    continue  # its project re-adopted (main present) → sidecar covered
                if rid_lbl and rid_lbl in acquiring:
                    # An in-flight acquire owns this rollout (its container may be mid-creation) —
                    # do NOT reap it. Fail closed so the pass retries once the acquire settles.
                    LOGGER.warning(
                        "raw-gc-reconciler: managed container=%s (rollout=%s) has an in-flight "
                        "acquire — NOT admitting (fail closed, retry) (H11)", cid[:12], rid_lbl,
                    )
                    ok = False
                    continue
                # Unaccounted managed survivor after re-adoption + the main-loop reaps: a
                # sidecar-only compose project (main gone, sidecars alive) or a leaked container.
                # REAP it (deploy-safety fix): compose members via whole-project teardown by
                # label; a plain raw container via force-destroy. Then ADMIT — rather than
                # quarantining the node forever. Fail closed only if the reap can't be confirmed.
                try:
                    if proj_lbl:
                        # A compose member (main or sidecar) → whole-project teardown by the
                        # project+rollout labels (reaps every member, not just this container).
                        await transport.destroy_compose_project(
                            rollout_id=rid_lbl, project_name=proj_lbl,
                        )
                    else:
                        # A plain raw container (or a compose member with no project label) →
                        # force-destroy the single container.
                        await transport.force_destroy_raw_container(
                            container_id=cid,
                        )
                    LOGGER.info(
                        "raw-gc-reconciler: reaped unaccounted survivor container=%s (rollout=%s "
                        "project=%s kind=%s) on node=%s during readopt — admitting (H11)",
                        cid[:12], rid_lbl or "?", proj_lbl or "-", kind or "?", node_id,
                    )
                except Exception:
                    LOGGER.warning(
                        "raw-gc-reconciler: could NOT reap unaccounted survivor container=%s "
                        "(rollout=%s project=%s) on node=%s — NOT admitting (fail closed, retry) "
                        "(H11)", cid[:12], rid_lbl or "?", proj_lbl or "-", node_id,
                    )
                    ok = False
        # Rebuild fleet reservations from the re-adopted members (no reclaim on connect).
        rebuild_fn = getattr(self._coordinator, "rebuild_fleets_from_state", None)
        if rebuild_fn is not None:
            # Decide fail-closed capability by INSPECTING the signature, not by catching a
            # TypeError from the call (audit H11): a bare ``except TypeError`` would swallow a
            # genuine TypeError raised INSIDE the production rebuild and silently downgrade to a
            # suppressed best-effort retry — masking a real bug and letting reconnect return
            # success. Only a coordinator whose rebuild actually accepts ``raise_on_error`` is
            # driven fail-closed; an older double without it gets the best-effort call.
            kwargs: dict[str, Any] = {
                "now": time.time(),
                "reclaim_after_s": self._fleet_reservation_ttl_s,
                "allow_reclaim": False,
            }
            supports_raise = False
            try:
                supports_raise = "raise_on_error" in inspect.signature(rebuild_fn).parameters
            except (ValueError, TypeError):
                supports_raise = False
            if supports_raise:
                kwargs["raise_on_error"] = True   # fail closed if fleet rows are unreadable
            try:
                await rebuild_fn(**kwargs)
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: readopt-on-connect fleet rebuild failed node=%s (H11)",
                    node_id,
                )
                ok = False
        LOGGER.info(
            "raw-gc-reconciler: readopt-on-connect for node=%s complete=%s "
            "(surviving raw/compose load accounted before admitting) — H11", node_id, ok,
        )
        return ok

    # ── Per-orphan handlers ─────────────────────────────────────────────────

    async def _handle_node_only(
        self,
        node_id: str,
        container_id: str,
        transport: object,
        *,
        row: RawRolloutRecord | None = None,
        compose_rollout_id: str | None = None,
        compose_project: str | None = None,
    ) -> None:
        """Docker has a raw container with no owning coordinator
        session. Reap it via the privileged
        :meth:`NodeTransport.force_destroy_raw_container` path, which
        bypasses the per-rollout ownership check the regular destroy
        enforces. NOT consumer-reachable — only the reconciler can
        issue this command.

        Issue #18 fix B — the log level reflects *why* the container
        is unowned. A ``raw_rollouts`` row in a terminal status means
        ``destroy`` already ran for this rollout and only the
        node-side container removal didn't finish — routine deferred
        teardown, logged at INFO. No row, or a non-terminal row,
        means a genuinely unexplained container — logged at WARNING.

        P1.7.C.2 — a **compose-project main** orphan (``compose_project``
        label set, but re-adoption above couldn't rebuild a session — no
        persisted row, or the row was deleted) is NOT force-destroyed as a
        bare container: that would remove ``main`` and leak its
        ``session_kind=compose`` sidecars (off the raw diff → nothing else
        reaps them). It routes to the node's whole-project
        :meth:`destroy_compose_project` (down -v --remove-orphans), which
        stops+removes the entire stack — and, when the node has no in-memory
        registration (it restarted), the node's own confirmed-absence reap by
        the project+rollout labels (fail-closed).

        audit H10 — if that whole-project down FAILS, we RETAIN ``main`` and
        retry on the next sweep rather than falling back to a single-container
        force-destroy of ``main``. Removing ``main`` would hide the project's
        only periodic sentinel (its ``session_kind=compose`` sidecars are OFF
        the raw inventory, so they are NOT re-observed as their own orphans —
        the old fallback silently leaked them). Keeping ``main`` alive means the
        next sweep re-observes the same orphan and re-attempts the whole-project
        teardown until it is confirmed — never a partial main-only removal that
        strands the sidecars.

        Failures are logged + swallowed so one node's hang
        doesn't pin the sweep.
        """
        if compose_project and compose_rollout_id:
            LOGGER.warning(
                "raw-gc-reconciler: node-only compose-project orphan node=%s "
                "project=%s main=%s — downing the whole stack (rollout=%s)",
                node_id, compose_project, container_id[:12], compose_rollout_id,
            )
            try:
                await transport.destroy_compose_project(  # type: ignore[attr-defined]
                    rollout_id=compose_rollout_id, project_name=compose_project,
                )
            except Exception:
                # audit H10 — do NOT fall back to force-destroying only ``main``: that would
                # strand the ``session_kind=compose`` sidecars (off the raw diff → never
                # re-observed). RETAIN ``main`` as the sentinel so the next sweep re-attempts the
                # whole-project down until it is confirmed.
                LOGGER.exception(
                    "raw-gc-reconciler: whole-project down FAILED node=%s project=%s — RETAINING "
                    "main=%s as the sentinel; will retry whole-project teardown next sweep (NOT "
                    "force-destroying main, which would leak the sidecars — audit H10)",
                    node_id, compose_project, container_id[:12],
                )
            return
        if row is not None and row.status in (
            "released", "failed", "cancelled", "reaped", "capacity_rejected",
        ):
            LOGGER.info(
                "raw-gc-reconciler: completing deferred teardown "
                "node=%s container=%s — rollout %s already %s, "
                "node-side destroy had not finished",
                node_id, container_id[:12], row.rollout_id, row.status,
            )
        else:
            LOGGER.warning(
                "raw-gc-reconciler: node-only orphan node=%s "
                "container=%s — force-destroying (rollout=%s)",
                node_id, container_id[:12],
                row.rollout_id if row is not None else "unknown",
            )
        try:
            await transport.force_destroy_raw_container(  # type: ignore[attr-defined]
                container_id=container_id,
            )
        except Exception:
            LOGGER.exception(
                "raw-gc-reconciler: force_destroy_raw_container "
                "failed node=%s container=%s; will retry next sweep",
                node_id, container_id[:12],
            )

    async def _handle_coordinator_only(
        self, node_id: str, container_id: str, rollout_id: str,
        *, reason: str | None = None,
    ) -> None:
        """Coordinator session whose docker container is gone. Drop the
        in-memory session and seal the durable ``raw_rollouts`` row.

        ``reason`` (audit P3) — when the node reported that it reaped this
        rollout autonomously (disk-pressure guard), it flows through
        :meth:`RawContainerCoordinator.seal_orphan` so the row seals
        ``reaped`` with the real cause (e.g. "disk-guard: reaped runaway
        raw container …") instead of the generic ``released`` teardown a
        vanished container would otherwise get. ``None`` (container
        vanished for some other reason — OOM, external ``docker rm``)
        keeps the existing clean-teardown seal.

        Failures (XRLEnvError from the coordinator's container_id
        consistency check OR transient gRPC / network errors from
        the wire-level destroy) are logged-and-swallowed
        symmetric with :meth:`_handle_node_only`, so one orphan's
        cleanup hiccup never aborts the sweep before sibling
        orphans / sibling nodes get processed. Raw-GC-M2 closure.

        H10 — a coordinator-only orphan that is a COMPOSE PROJECT (its visible main
        container vanished) does NOT take the bare seal below: main-absence is not
        confirmation its off-diff ``session_kind=compose`` sidecars are gone, so a seal
        would release the whole project's AGGREGATE capacity while sidecars keep running
        (invariant 2). It routes to node-confirmed whole-project teardown instead, which
        retains + retries on a failed down.
        """
        # H10 — compose project: node-confirmed whole-project down, never a bare capacity seal.
        # ``getattr`` (like the readopt hooks above) so a minimal / non-compose coordinator
        # gracefully takes the single-container seal path.
        is_compose = getattr(self._coordinator, "is_compose_project", None)
        if is_compose is not None and is_compose(rollout_id):
            LOGGER.warning(
                "raw-gc-reconciler: coordinator-only COMPOSE orphan node=%s main=%s "
                "rollout=%s — node-confirmed whole-project down%s",
                node_id, container_id[:12], rollout_id,
                f" ({reason})" if reason else "",
            )
            try:
                await self._coordinator.destroy_compose_orphan(
                    rollout_id=rollout_id, reason=reason,
                )
            except Exception:
                # A failed down RETAINS the session + aggregate capacity (invariant 2) — do
                # NOT drop the session here; the next sweep re-attempts from a clean state.
                LOGGER.exception(
                    "raw-gc-reconciler: compose-orphan whole-project down FAILED node=%s "
                    "rollout=%s — session RETAINED (capacity held), will retry next sweep",
                    node_id, rollout_id,
                )
            return
        LOGGER.warning(
            "raw-gc-reconciler: coordinator-only orphan node=%s "
            "container=%s rollout=%s — dropping session%s",
            node_id, container_id[:12], rollout_id,
            f" ({reason})" if reason else "",
        )
        # Seal directly via ``seal_orphan`` — NOT ``destroy``. The
        # container is already gone on the node (that's the definition of
        # a coordinator-only orphan: present in the coordinator's
        # sessions but absent from the node's container list), so a
        # wire-level destroy would only race. Once the node has dropped
        # its own record that destroy RPC fails with a benign "not
        # registered on this node" error, and post-H8 ``destroy``'s
        # raised-teardown branch RETAINS the session + its capacity charge
        # and seals nothing — so the row would sit ``running`` and this
        # orphan would come back every sweep, forever, with the node's real
        # reap cause never recorded. ``seal_orphan`` skips the wire call and
        # seals ``reaped`` (``reason`` set → the node's real cause, e.g.
        # the disk-guard diagnostic) or ``released`` (``reason`` None).
        try:
            await self._coordinator.seal_orphan(
                rollout_id=rollout_id,
                container_id=container_id,
                reason=reason,
            )
        except XRLEnvError:
            # Mismatched container_id check (defensive) — coordinator state is internally
            # consistent so this shouldn't fire. Fall back to a generation-safe drop that ALSO
            # releases fleet / liveness state + kicks admission (audit Low) instead of a bare
            # _sessions pop; a newer generation reusing this rollout_id is left untouched.
            with suppress(Exception):
                await self._coordinator.drop_orphan_session(
                    rollout_id, container_id, reason=reason,
                )
        except Exception:
            # Defensive: a state-store hiccup inside ``seal_orphan``
            # (there's no wire call to fail anymore) — log + swallow so
            # the sweep proceeds. Next sweep retries from a clean state.
            # Without this branch a transient hiccup aborts
            # ``reconcile_once`` and skips every later node + orphan in
            # the same sweep cycle.
            LOGGER.exception(
                "raw-gc-reconciler: coordinator-only seal_orphan raised "
                "node=%s container=%s rollout=%s; will retry next sweep",
                node_id, container_id[:12], rollout_id,
            )
            # Generation-safe drop WITH proper fleet/liveness/admission cleanup (audit Low) so the
            # next sweep doesn't re-see the orphan and no fleet membership / admission kick leaks.
            # ``reason=reason`` (audit round 3): without it this fallback dropped the session
            # WITHOUT sealing the row ``reaped``, silently making ``SessionReaped`` unreachable
            # for this rollout forever — see ``drop_orphan_session``'s docstring.
            with suppress(Exception):
                await self._coordinator.drop_orphan_session(
                    rollout_id, container_id, reason=reason,
                )


    # ── Deadline sweep (Issue #18) ──────────────────────────────────────────

    async def _reconcile_deadlines(self) -> int:
        """Force-destroy raw sessions whose wall-clock deadline has
        passed. Returns the number reaped.

        A raw container otherwise lives until the consumer explicitly
        destroys it; a consumer process killed mid-rollout leaves the
        container, its capacity reservation, and its ``raw_rollouts``
        row charged forever. Every session carries a ``deadline_at``
        (consumer-supplied ``session_deadline_s`` or the coordinator's
        default cap); once it passes, this reaps it.

        Per-session failures are logged and swallowed so one stuck
        teardown can't abort the sweep for its siblings.
        """
        now = time.time()
        reaped = 0
        for session in self._coordinator.list_sessions():
            deadline_at = getattr(session, "deadline_at", 0.0) or 0.0
            if deadline_at <= 0.0 or now <= deadline_at:
                continue
            # Already mid-teardown (consumer destroy raced us, or a
            # prior sweep's reap is still in flight) — leave it.
            is_destroying = getattr(self._coordinator, "is_destroying", None)
            if is_destroying is not None and is_destroying(session.rollout_id):
                continue
            overdue_s = now - deadline_at
            LOGGER.warning(
                "raw-gc-reconciler: session deadline exceeded "
                "rollout=%s node=%s overdue=%.0fs — force-destroying",
                session.rollout_id, session.node_id, overdue_s,
            )
            try:
                await self._coordinator.destroy(
                    rollout_id=session.rollout_id,
                    container_id=session.container_id,
                    force=True,
                    reason=(
                        f"session deadline exceeded "
                        f"(overdue {overdue_s:.0f}s) — force-reaped by "
                        f"raw-gc-reconciler; the consumer never called "
                        f"destroy (process killed mid-rollout?)"
                    ),
                )
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: deadline force-destroy raised "
                    "rollout=%s; will retry next sweep",
                    session.rollout_id,
                )
                continue
            reaped += 1
        return reaped

    def _mark_liveness_suspects(self) -> int:
        """Phase 1 — flag sessions that have gone quiet past the TTL.

        Marking is **not** a teardown decision. Crossing the TTL only means we
        have stopped hearing from a consumer, and silence has two very different
        causes: a process that exited, and one whose host stalled and will come
        back. Only the quarantine horizon (phase 2) distinguishes them, by
        waiting. This pass exists so the wait is visible — an operator watching
        `xrlenv_raw_sessions_suspect` climb and then drain is watching stalls
        that the old destroy-on-TTL behaviour would have turned into lost work.

        Cheap and uncapped (no I/O — it only touches a dict), unlike the paced
        destroy pass.
        """
        candidates_fn = getattr(
            self._coordinator, "liveness_suspect_candidates", None,
        )
        mark = getattr(self._coordinator, "mark_suspect", None)
        if candidates_fn is None or mark is None:
            return 0        # coordinator without two-phase liveness support
        marked = 0
        for session in candidates_fn():
            LOGGER.warning(
                "raw-gc-reconciler: consumer liveness SUSPECT rollout=%s node=%s "
                "— no heartbeat within TTL; NOT destroying yet (quarantine). A "
                "stalled consumer that resumes keeps its session.",
                session.rollout_id, session.node_id,
            )
            try:
                mark(session.rollout_id)
            except Exception:
                # Match the destroy loops: log and skip this session rather than
                # abandoning the rest of the sweep's candidates.
                LOGGER.exception(
                    "raw-gc-reconciler: marking rollout=%s suspect raised; "
                    "skipping it this sweep", session.rollout_id,
                )
                continue
            marked += 1
        if marked and self._metrics is not None:
            with contextlib.suppress(Exception):
                self._metrics.raw_liveness_suspect_total.inc(marked)
        return marked

    def _sync_suspect_gauge(self) -> None:
        """Re-read the suspect count into the gauge.

        Called AFTER the destroy pass, not during marking: a reap pops
        ``_suspect_since``, so a gauge set before that runs reads high until the
        next sweep re-marks. Called from a ``finally`` so a raising sweep still
        resyncs — otherwise a failure that repeats every interval leaves this
        wrong indefinitely, and it is the metric the feature is judged by.
        """
        if self._metrics is None:
            return
        count_fn = getattr(self._coordinator, "suspect_count", None)
        if count_fn is None:
            return
        with contextlib.suppress(Exception):
            self._metrics.raw_sessions_suspect.set(count_fn())

    async def _reconcile_liveness(self) -> int:
        """Phase 2 — force-destroy sessions whose consumer stayed silent past
        the QUARANTINE horizon.

        Distinct from the deadline sweep: a session qualifies only when its
        consumer has heartbeated at least once, has **no session-scoped RPC in
        flight**, is not already being torn down, and its liveness clock is
        staler than the quarantine horizon (see
        ``RawContainerCoordinator.liveness_reap_candidates``). This reaps a
        hard-killed consumer's containers well before the deliberately-long
        wall-clock deadline, without touching healthy long-running jobs (they
        heartbeat, or hold an in-flight exec) — or, now, briefly stalled ones.

        Destroys are **paced** — at most ``_LIVENESS_REAP_BATCH`` per sweep —
        so a mass die-off doesn't fire thousands of destroys at once; the
        remainder reap on subsequent sweeps. Per-session failures are logged
        and swallowed so one stuck teardown can't abort the others.

        Round-3 audit fix — the candidate list is a SNAPSHOT taken once at the
        top of this sweep, but the destroy loop below is sequential and each
        iteration ``await``s a real wire teardown. A consumer whose session
        was captured in that snapshot can heartbeat (or issue any session RPC)
        WHILE an earlier sibling's destroy is still in flight — before its own
        turn in the loop ever arrives. Without re-checking, that session gets
        destroyed anyway, silently breaking the feature's central promise
        ("any liveness signal clears suspicion"). So immediately before firing
        each destroy we re-derive candidacy fresh (cheap — sync, no I/O) and
        skip anyone who has recovered since the snapshot was taken.

        **Known residual, deliberately not fixed** (round-4 audit). That re-check
        closes the cross-session window above; it does NOT close the window inside
        a session's OWN destroy. A consumer that signals after its re-check passes
        but while its wire call is in flight is still torn down. This is
        irreducible rather than unfixed — once ``destroy_container`` reaches the
        node the container is gone and there is nothing to roll back — so the
        promise is precisely "any liveness signal received BEFORE the destroy is
        issued clears suspicion", not "any signal at all". Exposure is unchanged
        from the single-phase reaper, and such a session had already been silent
        for the entire quarantine horizon, so a signal arriving mid-teardown does
        not retroactively make the decision wrong. What it must not do is get
        COUNTED as a rescue — see ``RawContainerCoordinator._clear_suspect``.
        """
        candidates_fn = getattr(
            self._coordinator, "liveness_reap_candidates", None,
        )
        if candidates_fn is None:  # coordinator without liveness support
            return 0
        candidates = candidates_fn()
        reaped = 0
        for session in candidates[:_LIVENESS_REAP_BATCH]:
            is_destroying = getattr(self._coordinator, "is_destroying", None)
            if is_destroying is not None and is_destroying(session.rollout_id):
                continue
            # Re-verify staleness right before firing — see the docstring note
            # above. A sibling's destroy earlier in THIS batch can take long
            # enough for this session's own liveness signal to land in the
            # meantime, and the snapshot above wouldn't reflect it.
            still_stale = {s.rollout_id for s in candidates_fn()}
            if session.rollout_id not in still_stale:
                LOGGER.info(
                    "raw-gc-reconciler: rollout=%s liveness signal landed "
                    "mid-sweep (queued for reap, then recovered before its "
                    "turn) — skipping the destroy",
                    session.rollout_id,
                )
                continue
            LOGGER.warning(
                "raw-gc-reconciler: consumer liveness lost rollout=%s "
                "node=%s — force-destroying (silent for the full QUARANTINE "
                "HORIZON, not merely the TTL; no RPC in flight)",
                session.rollout_id, session.node_id,
            )
            try:
                await self._coordinator.destroy(
                    rollout_id=session.rollout_id,
                    container_id=session.container_id,
                    force=True,
                    reason=(
                        "consumer liveness lost — no heartbeat or session RPC "
                        "for the full quarantine horizon; force-reaped by "
                        "raw-gc-reconciler. The consumer process likely died "
                        "without calling destroy (a merely stalled one would "
                        "have signalled before the horizon and been retained)"
                    ),
                )
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: liveness force-destroy raised "
                    "rollout=%s; will retry next sweep",
                    session.rollout_id,
                )
                continue
            reaped += 1
            if self._metrics is not None:
                with contextlib.suppress(Exception):
                    self._metrics.raw_liveness_reaped_total.inc()
        return reaped

    # ── SQLite ghost sweep (Issue #18 fix #3) ───────────────────────────────

    def _reconcile_sqlite(self, *, reason: str) -> int:
        """Mark non-terminal ``raw_rollouts`` rows that no longer have
        any in-flight or in-memory representation as ``failed`` with
        ``error=<reason>``. Returns the number of rows flipped.

        Liveness sources (audit M1, round 2):

        - ``acquiring`` rows — a row is a ghost iff its ``rollout_id``
          is in neither ``list_sessions`` NOR
          ``list_acquiring_ids``. The acquiring set tracks the
          ``acquire`` coroutine directly, so a request legitimately
          parked in the admission queue is protected no matter how
          large the consumer set ``queue_timeout_s`` — there's no
          time proxy to outgrow. After a control-plane restart the
          acquiring set is empty (the coroutines died with the
          process), so every leftover ``acquiring`` row correctly
          reads as a ghost.

        - ``running`` rows — a row is a ghost iff its ``rollout_id``
          isn't in ``list_sessions`` AND the row is older than
          ``running_stale_s``. The small age window only guards the
          sub-second race between the ``_sessions`` insert and the
          ``status="running"`` SQLite write.

        ``acquiring_ids`` is snapshotted BEFORE ``sessions`` so the
        atomic acquire→session handoff (which removes from the
        acquiring set and adds to sessions under one lock) can never
        be observed as "in neither": if the snapshot caught the row
        pre-handoff it's in ``acquiring_ids``; post-handoff (a later
        read) it's in ``sessions``.

        Caller catches exceptions — the periodic sweep keeps running
        even if SQLite is temporarily unreachable.
        """
        if self._state is None:
            return 0

        list_fn = getattr(self._state, "list_raw_rollouts", None)
        update_fn = getattr(self._state, "update_raw_rollout", None)
        if list_fn is None or update_fn is None:
            # Test double / older store missing the new surface — nothing
            # we can do; the operator's admin panel will surface stale
            # rows but the cluster keeps working.
            return 0

        # Order matters — see the docstring's handoff-race note.
        acquiring_probe = getattr(
            self._coordinator, "list_acquiring_ids", None,
        )
        in_flight_ids: set[str] = (
            set(acquiring_probe()) if acquiring_probe is not None else set()
        )
        sessions = self._coordinator.list_sessions()
        live_ids = {s.rollout_id for s in sessions}
        now = time.time()

        flipped = 0
        for status in ("acquiring", "running"):
            try:
                rows = list_fn(status=status)
            except Exception:
                LOGGER.exception(
                    "raw-gc-reconciler: list_raw_rollouts(status=%s) raised",
                    status,
                )
                continue
            for row in rows:
                if row.rollout_id in live_ids:
                    continue
                age_s = now - float(getattr(row, "created_at", now))
                if status == "acquiring":
                    # Set-based liveness — no age window.
                    if row.rollout_id in in_flight_ids:
                        continue
                else:  # running
                    if age_s < self._running_stale_s:
                        continue
                    # WS3 — protect a re-adoptable running row (one that
                    # carries a container_id) from being sealed
                    # lost-on-restart until it's had a chance to be
                    # re-adopted: the grace is measured from reconciler
                    # start, not row age, because a long-running
                    # rollout's row is legitimately old. After the grace
                    # an un-re-adopted row means its node never came back
                    # (or the container is gone) — sealed normally below.
                    if (
                        getattr(row, "container_id", None)
                        and (now - self._started_at) < self._readopt_grace_s
                    ):
                        continue
                try:
                    update_fn(
                        row.rollout_id,
                        status="failed",
                        error=(
                            f"{RAW_LOST_CP_PREFIX} {reason} "
                            f"(no in-memory session; row age "
                            f"{age_s:.0f}s, prior status {status!r})"
                        ),
                        finished_at=now,
                    )
                except Exception:
                    LOGGER.exception(
                        "raw-gc-reconciler: update_raw_rollout(%s) raised; "
                        "will retry next sweep",
                        row.rollout_id,
                    )
                    continue
                flipped += 1
                LOGGER.warning(
                    "raw-gc-reconciler: sqlite-ghost rollout=%s "
                    "prior_status=%s age=%.0fs reason=%s",
                    row.rollout_id, status, age_s, reason,
                )
        return flipped


__all__ = ["RawGCReconciler"]
