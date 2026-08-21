"""Trajectory sinks (spec 08, spec 20).

A :class:`TrajectorySink` owns the persisted body of a rollout: each step
appended to the on-disk store as it lands, then sealed at finish/cancel/
truncate. The state store carries only the :class:`TrajectoryLocator`
pointer per spec 00 invariant 8 (state store metadata, blobs on disk).

Phase 0 ships ``platform-jsonl``: one ``trajectory.jsonl`` line per step plus
a sibling ``meta.json`` carrying the locator + lifecycle envelope. Phase 1's
slime-sample / verl-dataproto sinks slot in behind the same Protocol; the
multi-sink wrapper that mirrors to multiple backends lives in spec 08.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from xrlenv.control.template_catalog import TemplateManifest
from xrlenv.types import RolloutStatus, Step, Trajectory

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Locator (spec 20 §"Trajectory locator")
# ──────────────────────────────────────────────────────────────────────────────


SinkKind = Literal["platform-jsonl", "slime-sample", "verl-dataproto", "multi", "none"]


class TrajectoryLocator(BaseModel):
    """Pointer the rollouts row stores; resolved by readers + the SDK."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sink: SinkKind
    node_id: str | None
    uri: str | None
    size_bytes: int | None = None
    children: tuple[TrajectoryLocator, ...] | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Sink Protocol
# ──────────────────────────────────────────────────────────────────────────────


class TrajectorySink(Protocol):
    """Receives step appends + seal calls; produces a :class:`TrajectoryLocator`.

    The coordinator drives every sink the same way: ``open`` on start_rollout,
    ``record_step`` on each step, ``record_event`` at lifecycle inflection
    points (sandbox.create, rollout.start/finish/truncate/fail), ``seal`` on
    finish/cancel/truncate. Readers consume :py:meth:`read` (resolved through
    the locator) to materialize a :class:`Trajectory`.
    """

    def open(
        self,
        *,
        rollout_id: str,
        manifest: TemplateManifest,
        init: dict[str, Any],
        node_id: str,
    ) -> TrajectoryLocator: ...

    def record_step(self, rollout_id: str, step: Step) -> None: ...

    def record_event(
        self,
        rollout_id: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> None: ...

    def seal(
        self,
        *,
        rollout_id: str,
        status: RolloutStatus,
        reason: str | None,
        final_reward: float,
        metadata: dict[str, Any],
    ) -> TrajectoryLocator: ...

    def read(self, rollout_id: str) -> Trajectory: ...

    def run_dir_for_rollout(self, rollout_id: str) -> Path | None:
        """Return the rollout's host-side run dir, or ``None`` when no
        artifacts were persisted (e.g. an in-memory-only sink)."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Run-dir layout (spec 20 §"Run-artifact layout")
# ──────────────────────────────────────────────────────────────────────────────


def run_dir_for(runs_root: Path, rollout_id: str, *, when: datetime | None = None) -> Path:
    """Compute ``<runs_root>/<YYYY-MM-DD>/<rollout_id>/``.

    Rollouts that span a UTC midnight stay anchored to their *start* date so
    the sink writes to one directory throughout. The seal updates ``meta.json``
    in the same directory.
    """
    when = when or datetime.now(UTC)
    return runs_root / when.strftime("%Y-%m-%d") / rollout_id


# ──────────────────────────────────────────────────────────────────────────────
# PlatformJsonlSink — phase-0 default (spec 08 durability matrix)
# ──────────────────────────────────────────────────────────────────────────────


class _OpenSink(BaseModel):
    """Per-rollout state held by the sink between open and seal."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_dir: Path
    jsonl_path: Path
    meta_path: Path
    coordinator_log_path: Path
    started_at_iso: str
    template_digest: str
    template_name: str
    node_id: str
    init: dict[str, Any]
    step_count: int = 0
    sealed: bool = False
    final_status: RolloutStatus | None = None
    final_reason: str | None = None
    final_reward: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlatformJsonlSink:
    """``platform-jsonl`` sink: writes meta.json + trajectory.jsonl on disk.

    The sink is process-wide (one instance shared across rollouts) and tracks
    per-rollout open state in an internal dict. Concurrent rollouts hit
    separate directories so a single :class:`threading.Lock` protects only the
    bookkeeping map; per-rollout file writes are append-only and serialized by
    the per-rollout async lock the coordinator already holds.
    """

    name: SinkKind = "platform-jsonl"

    def __init__(self, runs_root: Path) -> None:
        self._runs_root = runs_root
        self._open_rollouts: dict[str, _OpenSink] = {}
        self._lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def open(
        self,
        *,
        rollout_id: str,
        manifest: TemplateManifest,
        init: dict[str, Any],
        node_id: str,
    ) -> TrajectoryLocator:
        run_dir = run_dir_for(self._runs_root, rollout_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = run_dir / "trajectory.jsonl"
        meta_path = run_dir / "meta.json"
        coordinator_log_path = run_dir / "coordinator.log"

        # Truncate any prior body (safe: the rollout_id is a fresh UUID, so
        # this only fires on operator-driven re-runs of the same id).
        jsonl_path.write_bytes(b"")
        # coordinator.log is append-only across rollouts (one rollout per dir
        # so 'across rollouts' is just one), but truncate on re-open to match
        # trajectory.jsonl semantics.
        coordinator_log_path.write_bytes(b"")

        started_at_iso = datetime.now(UTC).isoformat()
        state = _OpenSink(
            run_dir=run_dir,
            jsonl_path=jsonl_path,
            meta_path=meta_path,
            coordinator_log_path=coordinator_log_path,
            started_at_iso=started_at_iso,
            template_digest=manifest.digest,
            template_name=manifest.name,
            node_id=node_id,
            init=dict(init),
        )
        with self._lock:
            self._open_rollouts[rollout_id] = state

        # Write an initial meta.json so the viewer / replay can locate the
        # body even before the rollout terminates (spec 20: meta.json is
        # ALWAYS present).
        self._write_meta(state, status=None, reason=None, final_reward=0.0)
        return self._locator(state)

    def record_step(self, rollout_id: str, step: Step) -> None:
        state = self._require_open(rollout_id)
        line = json.dumps(_step_to_record(step), separators=(",", ":")) + "\n"
        with state.jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(line)
        state.step_count += 1

    def record_event(
        self,
        rollout_id: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append a structured JSON line to the rollout's ``coordinator.log``.

        Spec 08 §"Per-rollout artifacts" calls for a per-rollout event log
        alongside ``trajectory.jsonl``. The coordinator emits at lifecycle
        inflection points (sandbox.create, rollout.start/finish/truncate/fail);
        callers stay defensive — events are best-effort signal, never block
        the rollout. Lookups happen by walking the runs root because rollouts
        already torn down may have been removed from the in-memory open map.
        """
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": dict(payload or {}),
        }
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        path = self._coordinator_log_path(rollout_id)
        if path is None:
            return
        try:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(line)
        except OSError:
            LOGGER.exception(
                "PlatformJsonlSink.record_event: failed to append to %s", path
            )

    def _coordinator_log_path(self, rollout_id: str) -> Path | None:
        """Return ``coordinator.log`` path for an open or already-sealed rollout."""
        with self._lock:
            state = self._open_rollouts.get(rollout_id)
        if state is not None:
            return state.coordinator_log_path
        # Rollout already sealed and dropped from the in-memory map; walk the
        # runs root the same way :py:meth:`_resolve_artifacts` does.
        if not self._runs_root.exists():
            return None
        for date_dir in sorted(self._runs_root.iterdir()):
            candidate = date_dir / rollout_id
            if candidate.is_dir():
                return candidate / "coordinator.log"
        return None

    def run_dir_for_rollout(self, rollout_id: str) -> Path | None:
        """Return the rollout's run dir if it exists (open or sealed).

        Mirrors ``_coordinator_log_path``'s lookup: prefer the in-memory
        open-rollout map, fall back to walking ``runs_root`` for
        already-sealed rollouts. Returns ``None`` when the rollout
        has no on-disk artifacts (e.g. the sink was never opened).

        Used by the coordinator's reward path to drop a verifier
        directory (test.log + reward.txt + reward.json + ...) under
        ``<run_dir>/verifier/`` — the harbor-shaped layout.
        """
        with self._lock:
            state = self._open_rollouts.get(rollout_id)
        if state is not None:
            return state.run_dir
        if not self._runs_root.exists():
            return None
        for date_dir in sorted(self._runs_root.iterdir()):
            candidate = date_dir / rollout_id
            if candidate.is_dir():
                return candidate
        return None
        return None

    def seal(
        self,
        *,
        rollout_id: str,
        status: RolloutStatus,
        reason: str | None,
        final_reward: float,
        metadata: dict[str, Any],
    ) -> TrajectoryLocator:
        state = self._require_open(rollout_id)
        state.sealed = True
        state.final_status = status
        state.final_reason = reason
        state.final_reward = final_reward
        state.metadata = dict(metadata)
        self._write_meta(
            state, status=status, reason=reason, final_reward=final_reward
        )
        # Drop the per-rollout entry so long-running processes don't hold
        # them indefinitely; the on-disk artifacts are the source of truth.
        with self._lock:
            self._open_rollouts.pop(rollout_id, None)
        return self._locator(state)

    # ── consumer_final back-fill (Slice 4.5) ─────────────────────────────────

    def update_final_reward(
        self,
        *,
        rollout_id: str,
        final_reward: float,
        status: RolloutStatus,
        reason: str | None,
        metadata: dict[str, Any],
    ) -> None:
        """Atomic post-seal update of ``meta.json`` for consumer_final.

        Spec 00 invariant 3 says steps are immutable after seal — we only
        rewrite the locator envelope (``meta.json``), never the
        ``trajectory.jsonl`` body. Atomic write-to-temp + rename keeps the
        file consistent under concurrent reads from the trajectory viewer
        / replay path.
        """
        meta_path, jsonl_path = self._meta_and_body_paths(rollout_id)
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        existing["final_reward"] = final_reward
        existing["status"] = status.value
        existing["reason"] = reason
        existing["metadata"] = metadata
        existing["ended_at"] = (
            existing.get("ended_at") or datetime.now(UTC).isoformat()
        )
        # Refresh locator size in case the body grew between original seal
        # and this update (e.g. when called mid-rollout — not the consumer_final
        # case but harmless to keep accurate).
        existing["locator"] = {
            "sink": self.name,
            "node_id": existing.get("node_id"),
            "uri": f"file://{jsonl_path}",
            "size_bytes": jsonl_path.stat().st_size if jsonl_path.exists() else 0,
            "children": None,
        }
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp.replace(meta_path)

    def _meta_and_body_paths(self, rollout_id: str) -> tuple[Path, Path]:
        for date_dir in sorted(self._runs_root.iterdir()):
            candidate = date_dir / rollout_id
            if candidate.is_dir():
                return candidate / "meta.json", candidate / "trajectory.jsonl"
        raise FileNotFoundError(
            f"no run dir found for rollout {rollout_id} under {self._runs_root}"
        )

    # ── Read-back ────────────────────────────────────────────────────────────

    def read(self, rollout_id: str) -> Trajectory:
        meta, jsonl_path = self._resolve_artifacts(rollout_id)
        steps = list(self._iter_steps(jsonl_path))
        status = RolloutStatus(meta.get("status") or RolloutStatus.RUNNING.value)
        # ``node_id`` lives at the top level of ``meta.json`` (per
        # :py:meth:`_write_meta`), so a naive ``dict(meta["metadata"])``
        # drops it. Fold it back in so callers can read which VM ran
        # the rollout via ``trajectory.metadata["node_id"]`` — symmetric
        # with the live :py:meth:`StateStore.seal_trajectory` paths.
        metadata = dict(meta.get("metadata") or {})
        if meta.get("node_id") is not None:
            metadata["node_id"] = meta["node_id"]
        return Trajectory(
            rollout_id=meta["rollout_id"],
            template=meta["template"],
            steps=steps,
            status=status,
            reason=meta.get("reason"),
            final_reward=float(meta.get("final_reward") or 0.0),
            metadata=metadata,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _require_open(self, rollout_id: str) -> _OpenSink:
        with self._lock:
            state = self._open_rollouts.get(rollout_id)
        if state is None:
            raise KeyError(
                f"PlatformJsonlSink: rollout {rollout_id} is not open; "
                "call open() before record_step / seal"
            )
        return state

    def _locator(self, state: _OpenSink) -> TrajectoryLocator:
        size = state.jsonl_path.stat().st_size if state.jsonl_path.exists() else 0
        return TrajectoryLocator(
            sink=self.name,
            node_id=state.node_id,
            uri=f"file://{state.jsonl_path}",
            size_bytes=size,
        )

    def _write_meta(
        self,
        state: _OpenSink,
        *,
        status: RolloutStatus | None,
        reason: str | None,
        final_reward: float,
    ) -> None:
        meta = {
            "rollout_id": state.run_dir.name,
            "template": state.template_name,
            "template_digest": state.template_digest,
            "node_id": state.node_id,
            "started_at": state.started_at_iso,
            "ended_at": datetime.now(UTC).isoformat() if status else None,
            "status": status.value if status else RolloutStatus.RUNNING.value,
            "reason": reason,
            "final_reward": final_reward,
            "step_count": state.step_count,
            "init": state.init,
            "metadata": state.metadata,
            "locator": self._locator(state).model_dump(),
        }
        # Atomic write so a half-flushed meta.json never poisons the viewer.
        tmp = state.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(state.meta_path)

    def _resolve_artifacts(self, rollout_id: str) -> tuple[dict[str, Any], Path]:
        # The runs root may have multiple date dirs; walk them.
        for date_dir in sorted(self._runs_root.iterdir()):
            candidate = date_dir / rollout_id
            if candidate.is_dir():
                meta_path = candidate / "meta.json"
                jsonl_path = candidate / "trajectory.jsonl"
                if not meta_path.exists():
                    raise FileNotFoundError(
                        f"meta.json missing for rollout {rollout_id} at {candidate}"
                    )
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return meta, jsonl_path
        raise FileNotFoundError(
            f"no run dir found for rollout {rollout_id} under {self._runs_root}"
        )

    def _iter_steps(self, jsonl_path: Path) -> Any:
        if not jsonl_path.exists():
            return []
        with jsonl_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield _step_from_record(json.loads(line))


# ──────────────────────────────────────────────────────────────────────────────
# Step ↔ jsonl record converters (kept verbose for forward-compat)
# ──────────────────────────────────────────────────────────────────────────────


def _step_to_record(step: Step) -> dict[str, Any]:
    return {
        "index": step.index,
        "action": step.action,
        "obs": step.obs,
        "reward": step.reward,
        "done": step.done,
        "truncated": step.truncated,
        "info": step.info,
        "ts": step.ts,
    }


def _step_from_record(rec: dict[str, Any]) -> Step:
    return Step(
        index=int(rec["index"]),
        action=rec.get("action"),
        obs=rec.get("obs"),
        reward=float(rec.get("reward") or 0.0),
        done=bool(rec.get("done")),
        truncated=bool(rec.get("truncated")),
        info=dict(rec.get("info") or {}),
        ts=float(rec.get("ts") or 0.0),
    )


__all__ = [
    "PlatformJsonlSink",
    "SinkKind",
    "TrajectoryLocator",
    "TrajectorySink",
    "run_dir_for",
]
