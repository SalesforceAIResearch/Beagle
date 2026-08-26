"""SDK exception hierarchy (spec 05 §"Error model").

Consumers branch on `category` and `retryable`. Mapping to gRPC status codes
and metric labels is documented in spec 05.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from xrlenv.types import Trajectory

ErrorCategory = Literal["user", "infra", "workload"]


class XRLEnvError(Exception):
    """Base class for every SDK-raised error."""

    retryable: ClassVar[bool] = False
    category: ClassVar[ErrorCategory] = "infra"


# ── User errors (caller misconfigured something) ──────────────────────────────


class TemplateUnknown(XRLEnvError):
    category: ClassVar[ErrorCategory] = "user"


class RewardFnRequired(XRLEnvError):
    """`mode: consumer_final` template was started without a reward_fn."""

    category: ClassVar[ErrorCategory] = "user"


class BackendCapabilityMissing(XRLEnvError):
    """Template requires a capability the chosen backend does not advertise."""

    category: ClassVar[ErrorCategory] = "user"


class MountDenied(XRLEnvError):
    category: ClassVar[ErrorCategory] = "user"


class AuthDenied(XRLEnvError):
    category: ClassVar[ErrorCategory] = "user"


class ManifestInvalid(XRLEnvError):
    """Template manifest failed validation at register time."""

    category: ClassVar[ErrorCategory] = "user"


class ArchiveTooLarge(XRLEnvError):
    """A ``container_get_archive`` transfer exceeded the control-plane
    relay cap (``XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES``).

    The control plane is the metadata/orchestration channel, not a bulk-
    data pipe (spec 00 invariant 6: blobs live on disk / object store,
    never in the control path). A caller trying to pull a whole container
    filesystem (e.g. EvoClaw's ``docker cp {c}:/testbed .``) through it is
    refused at the node *before* the transfer can starve the node or
    balloon control-plane memory. This fails **only that one transfer** —
    the rollout/eval is unaffected — so callers whose copy is optional
    (debug artifact export) degrade cleanly. For large artifacts, use the
    node-local / object-store artifact-export primitive
    (``notes/artifact-export-primitive-proposal.md``). Not retryable: the
    same over-cap copy would fail identically.
    """

    category: ClassVar[ErrorCategory] = "user"


class FleetOverBudget(XRLEnvError):
    """A fleet companion acquire would exceed the fleet's declared footprint.

    Fleet reservation (phase 1) reserves a whole task's peak ``cpu``/``mem``
    footprint on one node when the fleet opens (spec 03/10/21). A later
    companion container draws from that reservation; if the running members'
    resources plus this one would exceed the declared footprint, the acquire
    is refused **at the control plane, before any node command** — the fleet's
    other containers and its reservation are untouched. Not retryable: the
    same over-budget companion would fail identically. The fix is on the
    consumer side — declare a larger ``xrlenv.fleet_cpu_request`` /
    ``xrlenv.fleet_mem_request`` footprint (it under-declared its own peak).
    """

    category: ClassVar[ErrorCategory] = "user"


# ── Infra errors (platform issues; usually retryable) ─────────────────────────


class CapacityExhausted(XRLEnvError):
    retryable: ClassVar[bool] = True


class PinCapacityExhausted(CapacityExhausted):
    """P6 step-4c (§8.7) — a REQUIRED cpu-isolation acquire could not pin because
    the target node's core ledger was exhausted.

    A subclass of :class:`CapacityExhausted`, but semantically NODE-SPECIFIC: it
    means the scheduler's view of that node's ``pinned_cpus_free`` was stale (a
    heartbeat / ledger race), NOT that the shared admission pool is globally full.
    The control plane's re-admit path treats it as retriable — exclude the failed
    node and re-place on a sibling isolation-capable node — whereas a plain
    ``CapacityExhausted`` (from the admission/place step) is terminal. Raised
    node-side by ``RawContainerManager._allocate_cpuset``."""


class ControlPlaneLost(XRLEnvError):
    retryable: ClassVar[bool] = True


class NodeLost(XRLEnvError):
    retryable: ClassVar[bool] = True


class SessionReaped(XRLEnvError):
    """The control plane force-destroyed a raw-container session.

    Raised for ANY session whose row was sealed ``reaped`` — that is, any
    platform-initiated teardown that recorded a reason. The consumer-liveness
    quarantine is the usual cause, but the wall-clock ``session_deadline_s``
    sweep and node-side orphan seals reach the same state, so read ``reason``
    for what actually happened rather than assuming liveness.

    Distinct from "rollout not found", which means a stale/unknown handle. This
    says the platform tore the session down *on purpose* and names why, so a
    harness can classify it as infra-transient and re-run the trial instead of
    reporting a workload failure. ``reason`` is the teardown reason recorded on
    the ``raw_rollouts`` row; ``reaped_at`` is its ``finished_at`` epoch seconds
    (``None`` when the row didn't carry one, and always ``None`` on the client
    side — it has no error-metadata key, so it does not survive gRPC).

    Retryable: a fresh ``acquire`` succeeds — nothing about the workload failed.
    The consumer usually learns of this long after the fact, at its next session
    RPC, because a reap is silent until something touches the session.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        reason: str,
        reaped_at: float | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.reaped_at = reaped_at


class ManagedInventoryUnsupported(XRLEnvError):
    """A node agent could not return the BROAD managed-container inventory (it ignored
    ``include_all_managed`` — an older agent). audit H11: reconnect readoption fails CLOSED on
    this, because a raw-only inventory can't rule out sidecar-only compose survivors. Retryable —
    the node stays unschedulable until it's upgraded (or reconnects with the capability)."""

    retryable: ClassVar[bool] = True


class NodeCommandTimeout(XRLEnvError):
    """A control-plane → node command exceeded its reply ceiling.

    Distinct subclass (still an :class:`XRLEnvError`, so existing
    ``except XRLEnvError`` callers are unaffected) so the destroy path can
    tell a *timeout* (node briefly wedged / I/O-saturated — the teardown
    is likely still in progress and the raw-GC reconciler will finish it)
    apart from a hard failure. A consumer-initiated destroy that times out
    is sealed ``released`` with cleanup deferred, not ``failed``, so a
    slow teardown under disk-I/O saturation doesn't false-fail a rollout
    whose work already completed.
    """

    retryable: ClassVar[bool] = True


class ImagePullFailed(XRLEnvError):
    retryable: ClassVar[bool] = True


class ImageMissingOnNode(XRLEnvError):
    """A1 / D19 (P1.2) — the chosen node does not have the rollout's
    image cached locally. Raised by the coordinator's pre-flight check
    before ``CreateSandboxCommand`` is sent. Surfaces to the consumer
    as ``RolloutFailed`` with ``reason="image_missing"`` — a clear,
    actionable error instead of the downstream "pull access denied"
    mess from the Docker daemon.
    """

    retryable: ClassVar[bool] = True


class AssetFetchFailed(XRLEnvError):
    retryable: ClassVar[bool] = True


# ── Workload errors (sandbox/template-level) ─────────────────────────────────


class _RolloutCarrierError(XRLEnvError):
    """Base for the three errors that carry a partial trajectory."""

    category: ClassVar[ErrorCategory] = "workload"

    def __init__(self, message: str, partial: Trajectory | None = None) -> None:
        super().__init__(message)
        self.partial = partial


class RolloutTruncated(_RolloutCarrierError):
    """Hard deadline / step timeout; partial trajectory returned."""


class RolloutCancelled(_RolloutCarrierError):
    """Consumer-initiated cancel or group-cancel; partial trajectory returned."""


class RolloutFailed(_RolloutCarrierError):
    """Sandbox crash, init/setup/teardown failure, or reward failure."""

    def __init__(
        self,
        message: str,
        reason: str,
        partial: Trajectory | None = None,
    ) -> None:
        super().__init__(message, partial)
        self.reason = reason


class ReplayUnavailable(XRLEnvError):
    category: ClassVar[ErrorCategory] = "workload"


# ── Phase 3 (declared early so the SDK exception surface is stable) ───────────


class SessionExpired(XRLEnvError):
    pass


class SessionDegraded(XRLEnvError):
    pass


__all__ = [
    "ArchiveTooLarge",
    "AssetFetchFailed",
    "AuthDenied",
    "BackendCapabilityMissing",
    "CapacityExhausted",
    "ControlPlaneLost",
    "ErrorCategory",
    "ImageMissingOnNode",
    "ImagePullFailed",
    "ManifestInvalid",
    "MountDenied",
    "NodeCommandTimeout",
    "NodeLost",
    "ReplayUnavailable",
    "RewardFnRequired",
    "RolloutCancelled",
    "RolloutFailed",
    "RolloutTruncated",
    "SessionDegraded",
    "SessionExpired",
    "SessionReaped",
    "TemplateUnknown",
    "XRLEnvError",
]
