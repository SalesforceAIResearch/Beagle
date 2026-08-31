"""Cross-cutting data types used by the SDK, control plane, and sandbox stub.

The shapes here are the platform's *public contract* — code on either side of
the consumer/control or control/node seam serializes against them. Keeping them
in a single module makes the contract greppable and lets us evolve the wire
format without scattering struct definitions across packages.

All models are :class:`pydantic.BaseModel` so external inputs (template YAML,
JSON wire payloads, CLI flags) are validated at construction time. Frozen
models match the spec's invariants where mutation would corrupt downstream
state (e.g. :class:`Deadline` is pinned at admission per spec 02).

Sources of truth:
- spec 02 — Rollout API (`Deadline`, `StepResult`, `Trajectory`, lifecycle states)
- spec 14 — EnvAdapter (action/observation are opaque payloads)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ──────────────────────────────────────────────────────────────────────────────
# Action / observation are opaque to the platform (spec 14).
# We carry them as JSON-serializable payloads end-to-end.
# ──────────────────────────────────────────────────────────────────────────────

Action = Any
Observation = Any


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle states (spec 02 "Lifecycle states" table)
# ──────────────────────────────────────────────────────────────────────────────


class RolloutStatus(StrEnum):
    """Terminal + transient states of a rollout (spec 02).

    `destroying` is sandbox-side and intentionally not modelled here.
    """

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    FINISHING = "finishing"
    FINISHED = "finished"
    TRUNCATED = "truncated"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RolloutStatus.FINISHED,
            RolloutStatus.TRUNCATED,
            RolloutStatus.CANCELLED,
            RolloutStatus.FAILED,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Deadlines (spec 02 — single canonical view of every deadline)
# ──────────────────────────────────────────────────────────────────────────────


class Deadline(BaseModel):
    """Soft + hard deadlines plus per-phase overrides.

    ``hard_s`` is mandatory because the rollout's lifecycle clock starts the
    moment the first observation is returned (spec 02 per-rollout timeline).
    Per-phase overrides default to ``None`` and fall back to template values.

    Frozen because the deadline is pinned at admission and used to reason
    about per-rollout truncation (any later mutation would silently change
    truncation behaviour mid-rollout).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_s: float = Field(gt=0)
    soft_s: float | None = Field(default=None, gt=0)

    queue_timeout_s: float | None = Field(default=None, gt=0)
    image_pull_timeout_s: float | None = Field(default=None, gt=0)
    create_timeout_s: float | None = Field(default=None, gt=0)
    init_timeout_s: float | None = Field(default=None, gt=0)
    setup_timeout_s: float | None = Field(default=None, gt=0)
    step_timeout_s: float | None = Field(default=None, gt=0)
    reward_timeout_s: float | None = Field(default=None, gt=0)
    teardown_timeout_s: float | None = Field(default=None, gt=0)
    idle_ttl_s: float | None = Field(default=None, gt=0)


# ──────────────────────────────────────────────────────────────────────────────
# Step / Trajectory (spec 02)
# ──────────────────────────────────────────────────────────────────────────────


class StepResult(BaseModel):
    """Result of one ``EnvAdapter.step(action)`` call (spec 02 / spec 14)."""

    model_config = ConfigDict(extra="forbid")

    obs: Observation = None
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False


class Step(BaseModel):
    """One recorded entry in a :class:`Trajectory` (spec 02)."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    action: Action = None
    obs: Observation = None
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)
    ts: float
    """Monotonic-clock seconds since rollout start."""


class Trajectory(BaseModel):
    """The sealed record of a finished rollout (spec 02).

    ``final_reward`` is the scalar contract trainer adapters consume;
    populated per-mode by the coordinator at seal time.
    """

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    template: str
    steps: list[Step] = Field(default_factory=list)
    status: RolloutStatus
    reason: str | None = None
    final_reward: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CancelGroupReport(BaseModel):
    """Result of :py:meth:`Client.cancel_group` (spec 02 §"Cancellation")."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    cancelled: tuple[str, ...]
    """Rollout ids that were running and have now been sealed cancelled."""
    already_terminal: tuple[str, ...]
    """Rollout ids that were already in a terminal state when called."""


class TerminateRawGroupReport(BaseModel):
    """Result of :py:meth:`Client.terminate_raw_group` — the raw-container
    analogue of :class:`CancelGroupReport`. Destroys every still-running raw
    container carrying ``xrlenv.group_id`` (a consumer aborting a run tears its
    containers down actively instead of waiting for the raw-liveness reaper)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    terminated: tuple[str, ...]
    """Raw rollout ids whose containers were running and have now been destroyed."""
    already_terminal: tuple[str, ...]
    """Raw rollout ids already terminal (or never got a container) when called."""


class FailedRollout(BaseModel):
    """Bucketed entry in :class:`BatchRolloutResult.failed` (spec 05).

    Carries the raw exception text + the partial trajectory if one was
    produced before the failure. ``reason`` is the spec-02 reason label
    when available (``node_lost``, ``init_failed``, etc.); otherwise the
    exception class name.
    """

    model_config = ConfigDict(extra="forbid")

    rollout_id: str | None
    """``None`` when the rollout never advanced past admission."""
    reason: str
    error_kind: str
    error_message: str
    partial: Trajectory | None = None


class BatchRolloutResult(BaseModel):
    """Result of :py:meth:`Client.batch_rollout` (spec 05).

    Three buckets the consumer dispatches differently:
    - ``finished`` — sealed cleanly within the deadline.
    - ``truncated`` — hit the hard deadline; partial trajectory present.
    - ``failed`` — sandbox/init/setup/teardown error; partial may be present.
    """

    model_config = ConfigDict(extra="forbid")

    finished: list[Trajectory] = Field(default_factory=list)
    truncated: list[Trajectory] = Field(default_factory=list)
    failed: list[FailedRollout] = Field(default_factory=list)
