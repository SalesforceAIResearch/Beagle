"""Per-node image cache manager (spec 15).

Phase-0 surface: LRU eviction + operator pin list + ensure-present pull
with on-demand eviction. The phase-1 ``client.warmup(...)`` SDK API,
image-affinity scheduling, and lazy-load formats are deferred to later
slices; the manager already exposes the data structures they will hook
into so adding them stays additive.

Eviction order (D16, P1.2.b): cold images sort by *(eviction tier,
LRU)* — final task tags evict before stub-runtime layers, which evict
before base images. The rationale is rebuild cost: a final tag
rebuilds in seconds (a retag or a single ``RUN pip install`` layer);
a stub-runtime layer rebuilds in tens of seconds (one ``apt+pip``
install); a base image rebuilds in minutes (the upstream task's full
Dockerfile build). Evicting cheapest-to-recreate first preserves the
expensive layers across cache pressure. Within a tier the existing
LRU ordering still applies. The classifier defaults to the harbor /
tb2 tag conventions (``<bench>-base/<task>:V`` is base; everything
else is final); external plug-ins with different conventions can pass
a custom ``tier_classifier`` to :class:`ImageCacheManager`.

Lifecycle hooks the cache manager exposes:

- :py:meth:`acquire(image)` — the node-agent calls this just *after* a
  sandbox is created so the cache manager bumps the in-use refcount;
  in-use images are never evictable.
- :py:meth:`release(image)` — paired with destroy.
- :py:meth:`ensure_present(image, deadline_s)` — used by the agent's
  create path *before* it asks the backend to start a sandbox; this is
  the blocking wait + pull + evict-if-needed entry point.
- :py:meth:`pin(image)` / :py:meth:`unpin(image)` — runtime pin
  updates layered on top of the operator file (read at construction).
- :py:meth:`report` — snapshot of per-image tier + sizes used by the
  ``xrlenv images`` CLI and (later) the heartbeat report.

The manager is intentionally a per-node module — it lives next to
:class:`xrlenv.node.agent.NodeAgent` rather than on the control plane,
because disk pressure is a per-node phenomenon and the eviction
decision needs the local backend's view of free space.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from xrlenv.backends.base import ImageInUse, ImageRecord, SandboxBackend
from xrlenv.disk_policy import (
    CACHE_EVICT_START_MARGIN,
    CACHE_EVICT_TARGET_MARGIN,
    cache_evict_floor_bytes,
)
from xrlenv.image_refs import same_image
from xrlenv.node.adaptive_pull import AdjustableSemaphore, PullAimdController
from xrlenv.node.disk_io import DiskIoSampler
from xrlenv.observability.tracing import get_tracer

LOGGER = logging.getLogger(__name__)

ImageTier = Literal["in_use", "pinned", "recently_used", "cold"]

EvictionTier = Literal["final", "stub_runtime", "base"]

# Ownership classification (B7.6 follow-on, 2026-05).
#
# Distinct from :data:`EvictionTier`: tier is a disk-cost signal driving
# eviction order; ownership is "did xrlenv build this, and is it a final
# task image or a build byproduct?". The admin /images view filters on
# ownership by default so operators don't drown in operator-foreign
# images (``python:3.12-slim``, the daemon's pre-existing inventory)
# or in build intermediates (``<bench>-base/<task>:V``).
#
# Authoritative signal is two Docker labels baked at build time:
#
# - ``org.xrlenv.owned=true``  — applied to all xrlenv-built images.
# - ``org.xrlenv.role=final``  — final task image (sandbox-runnable).
# - ``org.xrlenv.role=intermediate`` — base layer the build pipeline
#   produces and the final image is layered on top of.
#
# Images without ``org.xrlenv.owned=true`` classify as ``external``.
# Operators must rebuild any pre-label xrlenv images for them to
# surface under the default-on filter.
OwnershipClass = Literal["xrlenv_final", "xrlenv_intermediate", "external"]

_LABEL_OWNED = "org.xrlenv.owned"
_LABEL_ROLE = "org.xrlenv.role"


def default_ownership_classifier(labels: dict[str, str]) -> OwnershipClass:
    """Classify an image's ownership from its Docker labels.

    Without ``org.xrlenv.owned=true``, an image is ``external`` even if
    the name happens to match a historical xrlenv pattern — phase-1
    decision: the label is the only signal so the contract stays
    operator-auditable. Plug-in authors must rebuild after the label
    rollout for their images to classify correctly.
    """
    if labels.get(_LABEL_OWNED) != "true":
        return "external"
    role = labels.get(_LABEL_ROLE, "final")
    if role == "intermediate":
        return "xrlenv_intermediate"
    # ``final``, missing, or any future role value defaults to final
    # so an unlabeled-role xrlenv image still surfaces under the
    # default-on filter rather than being suppressed as external.
    return "xrlenv_final"
"""D16 eviction-cost classification for a cached image.

Ordered cheapest-to-recreate first:

- ``"final"`` — task-image tag (``<bench>/<task>:V``). Rebuild is one
  ``RUN`` layer on top of a still-present stub-runtime / base, so cheap.
- ``"stub_runtime"`` — platform stub-runtime layer materialized as a
  separately-tagged image. Today's tb2 build path bakes the layer
  directly into final tags rather than tagging a separate image, so
  this tier is reserved for future build paths that materialize one
  (eStargz prefetch, multi-base builds, …).
- ``"base"`` — upstream task base (``<bench>-base/<task>:V``). Rebuild
  is the full upstream Dockerfile build, minutes per task. Evict last.
"""

TierClassifier = Callable[[str, dict[str, str]], EvictionTier]
"""Pluggable callable mapping ``(image name, image labels)`` → :data:`EvictionTier`.

The default :func:`default_tier_classifier` reads the harbor /
tb2 conventions. External plug-ins with different tag shapes pass
their own classifier to :class:`ImageCacheManager`.
"""

_TIER_PRIORITY: Final[dict[EvictionTier, int]] = {
    "final": 0,
    "stub_runtime": 1,
    "base": 2,
}


def default_tier_classifier(image: str, labels: dict[str, str]) -> EvictionTier:
    """Classify ``image`` for D16 eviction-cost ordering.

    Two-signal lookup, label-first:

    1. ``org.xrlenv.role`` label (set by the plug-in's Dockerfile —
       same label the admin /images ownership filter reads). This
       is the *authoritative* signal because the plug-in author is
       declaring intent, and it generalizes to any future plug-in
       without a core-side code change. Mapping:

       - ``role="intermediate"`` → ``"stub_runtime"`` (medium —
         expensive layer like a benchmark addon's pip install).
       - ``role="final"``        → ``"final"`` (cheap top layer).
       - ``role="base"``         → ``"base"`` (heavy bottom layer;
         lets plug-ins declare a base via label too).
       - any other value         → fall through to the name pattern.

    2. **Name pattern** (back-compat fallback for unlabeled images):
       ``[host[:port]/]<bench>-base/<task>[:V][@sha256:…]`` → ``"base"``.
       Default for everything else is ``"final"``.

       The pattern still matters because the upstream-base stage in
       most plug-in build scripts is a ``docker pull`` + ``docker tag``
       (no ``docker build``), and ``docker tag`` cannot attach labels.
       Adding labels would force a one-line ``FROM <upstream>`` build
       just to carry a ``LABEL`` directive, which costs an extra layer
       per instance for no semantic gain. The ``-base/`` repo
       convention lets us classify those retags correctly without that
       overhead.

    The structural marker is ``-base/`` in the *repo* portion of the
    reference. We strip an optional ``@sha256:…`` digest first, then
    strip the tag colon **only from the last path segment** so a
    ``host:port`` colon earlier in the reference (the common form for
    private / local registries — e.g.
    ``localhost:5000/foo-base/bar``) is preserved. The previous form,
    ``image.partition(":")[0]``, was too aggressive: for
    ``localhost:5000/foo-base/bar:0.1`` it left ``repo = "localhost"``
    and silently lost the ``-base/`` marker, so expensive base images
    at registries with host ports were classified as cheap final tags.
    Audit M4 follow-up.

    The ``"stub_runtime"`` tier value is the medium-cost middle slot.
    Originally reserved for a separate platform stub-runtime tag
    (never landed); now load-bearing for benchmark-addon intermediate
    images (the swebench-verified ``-bench/`` Stage-2 layer is the
    first in-tree consumer). Tier name kept rather than renamed
    because the cost-ordering semantics are identical and renaming
    would churn every plug-in's tier-classification expectation.
    """
    role = labels.get("org.xrlenv.role")
    if role == "intermediate":
        return "stub_runtime"
    if role == "final":
        return "final"
    if role == "base":
        return "base"

    no_digest = image.partition("@")[0]
    last_slash = no_digest.rfind("/")
    if last_slash >= 0:
        # Strip the tag colon from the last path segment only,
        # preserving any host:port colons earlier in the reference.
        repo = no_digest[: last_slash + 1] + no_digest[last_slash + 1 :].partition(":")[0]
    else:
        # Bare image with no path: "name[:tag]".
        repo = no_digest.partition(":")[0]
    if "-base/" in repo:
        return "base"
    return "final"


class ImageCacheConfig(BaseModel):
    """Tunables for the cache manager.

    Eviction high/low water marks are **workload-adaptive**, not a fixed
    fraction of disk. The headroom the cache keeps free scales with the
    size of the largest cached image, not with the disk:

        headroom = clamp(
            slots x largest_cached_image x disk_safety_factor,
            floor   = evict_*_bytes,
            ceiling = evict_*_cap_bytes,
        )

    This is deliberate. A fraction of disk (the pre-2026-06 model)
    reserved 20-30% of *every* node — 100-300 GiB idle on a 500 GiB-1 TiB
    box, for nothing — and under-reserved on a small laptop. Sizing the
    reserve to "a few image-pulls' worth of space" instead means a
    500 GiB and a 1 TiB node both keep the same modest buffer (enough to
    absorb a pull burst + running containers' overlay writes), and the
    buffer grows automatically when the workload's images get bigger.

    Until the cache has observed at least one image (empty on cold start)
    ``largest_cached_image`` is ``0`` and the **floor** applies, so a
    fresh node still has a sane absolute buffer.

    Defaults:

    - ``evict_threshold_bytes`` (15 GiB, floor) /
      ``evict_threshold_cap_bytes`` (50 GiB, ceiling) — start evicting
      cold images when free disk drops below the adaptive START headroom.
    - ``evict_target_bytes`` (25 GiB, floor) /
      ``evict_target_cap_bytes`` (75 GiB, ceiling) — stop evicting once
      free climbs back to the adaptive STOP headroom (one extra slot
      above START, for hysteresis).
    - ``evict_headroom_slots`` (4) / ``evict_disk_safety_factor`` (1.5) —
      the two knobs the adaptive headroom is derived from.
    - ``sweep_interval_s`` (60 s) — cadence of the periodic background
      sweep that runs eviction even when no ``ensure_present`` cache
      miss is firing. Closes the gap where a steady-state workload on
      already-cached images lets sandbox writable-overlay growth fill
      the disk without a trigger.
    - ``recent_window_s`` (30 min) — images touched within this window
      classify as ``recently_used`` (evictable, but only after ``cold``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evict_threshold_bytes: int = 15 * 1024**3
    """Absolute floor for the eviction-START headroom — applied when image
    sizes aren't known yet (empty cache)."""
    evict_target_bytes: int = 25 * 1024**3
    """Absolute floor for the eviction-STOP headroom."""

    evict_headroom_slots: int = 4
    """How many *largest-cached-image* worth of free disk eviction keeps as
    headroom. The reserve is ``slots x largest_image x disk_safety_factor``
    — it scales with the workload's image size, NOT the disk size, so a
    500 GiB and a 1 TiB node reserve the same (a pull-burst), instead of
    20-30% of the whole disk. Eviction stops one extra slot above the
    start point (hysteresis)."""

    evict_disk_safety_factor: float = 1.5
    """Dimensionless margin applied to BOTH the per-image eviction
    headroom and the disk-bounded pull ceiling. Higher → more free
    headroom + fewer concurrent pulls (safer); lower → denser cache. This
    is the one knob that trades safety for density; everything else is
    derived from observed free disk + image sizes."""

    evict_threshold_cap_bytes: int = 50 * 1024**3
    """Sanity ceiling on the adaptive eviction-START headroom — bounds a
    pathologically large image (e.g. a multi-GiB GPU base) from reserving
    an unreasonable buffer. Tuned per node via
    ``XRLENV_EVICT_THRESHOLD_CAP_GB``."""

    evict_target_cap_bytes: int = 75 * 1024**3
    """Sanity ceiling on the adaptive eviction-STOP headroom. Tuned per
    node via ``XRLENV_EVICT_TARGET_CAP_GB``."""

    sweep_interval_s: float = 60.0
    """Cadence of the periodic background eviction sweep. ``0`` or
    negative disables the sweep (manual / on-pull eviction only)."""

    recent_window_s: float = 30 * 60.0
    pull_concurrency: int = 2
    """AIMD **floor** — the minimum concurrent image pulls this node
    runs, i.e. the value the adaptive limiter decays to when the node is
    busy with live rollouts. Kept low (default ``2``) so cold pulls never
    starve time-sensitive agent containers.

    Concurrent pulls are governed by a single node-local AIMD limiter
    (see :class:`xrlenv.node.adaptive_pull.PullAimdController`): busy →
    multiplicative-decrease toward this floor; calm → additive-increase
    toward :attr:`pull_concurrency_ceiling`. Same-image pulls coalesce
    separately.

    Tuned per node via ``XRLENV_PULL_CONCURRENCY`` (read at startup,
    stamped into ``/etc/xrlenv/node.env`` by the deploy scripts)."""

    pull_concurrency_ceiling: int = 64
    """AIMD **ceiling** — the maximum concurrent pulls the node ramps up
    to when idle (e.g. during ``xrlenv build apply`` on an otherwise idle
    cluster, where saturating the registry / FSx pipe is the goal). The
    adaptive limiter never exceeds this. Tuned per node via
    ``XRLENV_PULL_CONCURRENCY_CEILING``."""

    pull_concurrency_initial: int = 16
    """AIMD **initial** limit at node start, clamped into
    ``[pull_concurrency, pull_concurrency_ceiling]``. The limiter then
    adapts up or down from here on each tick. Tuned per node via
    ``XRLENV_PULL_CONCURRENCY_INITIAL``."""

    pull_busy_threshold: int = 0
    """In-use container count at/below which the node counts as *idle*
    (AIMD additive-increase); above it the node is *busy* (AIMD
    multiplicative-decrease). Default ``0`` — any running rollout makes
    the node busy, maximally protecting time-sensitive agents."""

    pull_aimd_interval_s: float = 15.0
    """Cadence of the node-local AIMD tick that resizes the pull limiter.
    ``0`` or negative disables the loop (the limiter then stays fixed at
    :attr:`pull_concurrency_initial`). Mirrors the control-plane admission
    AIMD's 15 s cadence."""

    pull_aimd_additive_step: int = 2
    """How many slots the AIMD limiter gains per calm tick (additive
    increase). Decrease is always multiplicative (halving)."""

    pull_aimd_enabled: bool = True
    """When ``False`` the limiter stays fixed at
    :attr:`pull_concurrency_initial` and no AIMD loop runs — an escape
    hatch for deterministic behavior / debugging."""

    io_throttle_enabled: bool = True
    """When ``True`` (and a :class:`~xrlenv.node.disk_io.DiskIoSampler` is
    wired) the AIMD tick treats a *saturated data-root volume* as "busy"
    and decays the pull limit toward the floor — even when free disk is
    plentiful. This closes the gap where an IOPS/throughput-capped or
    near-full EBS volume is pegged at 100 % util (deep request queue,
    ``containerd`` in D-state, destroy commands timing out) while the
    free-disk controller still admits cold pulls. Set ``False`` to fall
    back to free-disk-only throttling."""

    io_util_high: float = 0.90
    """Utilization (``[0, 1]``) at/above which the data-root volume counts
    as *saturated* → AIMD multiplicative-decrease. This is a
    saturation-*detection* point on a physically bounded quantity, not a
    provisioned-IOPS assumption: the node discovers whatever ceiling the
    volume actually has by observing util pegged near 1.0. Tuned per node
    via ``XRLENV_IO_UTIL_HIGH_PCT``."""

    io_util_low: float = 0.70
    """Utilization at/below which the volume counts as *clear* again
    (hysteresis low-water mark) so the limiter can ramp back up. Must be
    ``<= io_util_high``. Tuned via ``XRLENV_IO_UTIL_LOW_PCT``."""

    io_sample_min_interval_s: float = 2.0
    """Minimum seconds between ``/sys`` io-stat samples; between samples
    the cached utilization is reused so the per-decision path stays
    syscall-free."""

    pull_max_attempts: int = 3
    """Issue #18 — how many times ``ensure_present`` attempts a
    registry pull before giving up. A registry (or its auth
    endpoint) can be transiently unreachable under heavy concurrent
    cold-pull load — a single connection timeout would otherwise
    lose the whole acquire. Retries are bounded by the per-call
    deadline, so a genuinely missing image still fails fast. ``1``
    disables retry (one attempt). Does not apply to builder-driven
    local builds — only registry pulls."""
    default_pull_timeout_s: float = 600.0
    """Default wall-clock deadline for ``ensure_present`` (registry pull
    or builder-driven local build). Cross-file invariant: keep in sync
    with :data:`xrlenv.backends.docker.DOCKER_CLIENT_HTTP_TIMEOUT_S` —
    the docker-py HTTP socket timeout below this layer has to be at
    least as generous or a cold pull aborts at the urllib3 layer
    before the cache deadline ever fires (PR #16)."""

    build_grace_window_s: float = 10 * 60.0
    """Sub-slice 2: a freshly-built image is treated as
    ``recently_used`` for this many seconds even before its first
    ``ensure_present`` cache hit. Without this, a build that
    finishes seconds before the eviction loop runs can be reaped
    immediately (it has no ``last_used`` touch yet), forcing a
    rebuild on the very next acquire. The default 10 min window
    is long enough to cover the typical "build → first rollout"
    lag on a busy cluster + leaves a bounded protection horizon
    so a stale never-used image still ages out within an hour."""


class ImageStateRecord(BaseModel):
    """Public per-image snapshot for the ``xrlenv images`` CLI."""

    model_config = ConfigDict(extra="forbid")

    name: str
    tier: ImageTier
    size_bytes: int
    shared_size_bytes: int | None = None
    """Bytes belonging to layers shared with other tagged images on this
    node (Docker ``SharedSize``). ``None`` when the backend doesn't
    surface layer-sharing — see :class:`ImageRecord.shared_size_bytes`.
    Used by ``xrlenv build calibrate`` to write the bin-packer-relevant
    *unique* size (``size_bytes - shared_size_bytes``) into the plan
    YAML's ``placement.size_hint_bytes`` field.
    """
    in_use_count: int
    last_used_at: float | None
    pinned: bool
    owner: OwnershipClass = "external"
    """Ownership class derived from the image's Docker labels — see
    :func:`default_ownership_classifier`. The admin ``/images`` view
    filters on this by default; off-by-default to keep
    ``xrlenv images`` CLI output unsurprising."""
    digest: str | None = None
    """The image's registry manifest digest (``repo@sha256:...``, from Docker
    ``RepoDigests``) — the same content-addressed identity the control-plane
    ``RegistryDigestResolver`` produces for a plan's ``:tag`` ref. ``None`` when
    the backend reports no RepoDigests (locally-built / never-pushed images).
    Used by ``xrlenv build calibrate`` to attribute an image a node pulled BY
    DIGEST (digest-pinned on acquire → held untagged) back to its plan entry
    when many tags share one repo and the tag/repo-path matchers can't — see
    :func:`~xrlenv.image_refs.repo_path` and the calibrate endpoint."""


class NodeImageReport(BaseModel):
    """Cluster-view-friendly snapshot of one node's cache state."""

    model_config = ConfigDict(extra="forbid")

    images: list[ImageStateRecord] = Field(default_factory=list)
    free_disk_bytes: int = 0
    pinned: tuple[str, ...] = ()


class ImageQueryResult(BaseModel):
    """A1 / D18+D19 (P1.2) — reply shape for ``ImageCacheManager.query``.

    Mirrors :class:`xrlenv.api._pb2.QueryImageReply` on the wire side
    and is the in-process return type for the same call against
    :class:`~xrlenv.node.NodeAgent` and the Protocol-level
    :py:meth:`NodeTransport.query_image`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    present: bool
    digest: str | None = None
    last_used_at: float = 0.0
    """Per-node monotonic-clock timestamp from
    :py:attr:`ImageCacheManager._last_used`. 0.0 when the image is
    absent or never been used. Comparable only on the same node —
    every node tracks its own monotonic clock, so the control plane
    must NOT subtract values across nodes."""


class EvictOutcome(BaseModel):
    """Result of :py:meth:`ImageCacheManager.evict_ref` (``xrlenv images evict``).

    Mirrors the per-node :class:`xrlenv.api._pb2.EvictImageReply`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["evicted", "absent", "in_use", "failed"]
    reclaimed_bytes: int = 0
    removed: tuple[str, ...] = ()
    """The exact local tags removed (the node may hold the ref under a
    registry-qualified tag the operator didn't type verbatim)."""
    detail: str = ""
    """Daemon/error message when ``status == 'failed'``; otherwise empty."""


ImageProducer = Callable[[str, float], Awaitable[None]]
"""Per-image producer callable (P1.6.g — H3 lazy lifecycle).

Signature: ``async def produce(image_ref: str, timeout_s: float) -> None``.
Raises on failure. Used by :py:meth:`ImageCacheManager.ensure_present`
when a benchmark-internal image ref isn't in any registry the
backend's ``pull_image`` knows about. The node-agent populates a
lookup callable that maps ``swebench-verified/<inst>:0.1`` (or
similar) to a closure over the right :class:`BenchmarkImageBuilder`.
Falls back to ``backend.pull_image`` when no producer is registered
for the ref.
"""

BuilderLookup = Callable[[str], "ImageProducer | None"]


class ImageCacheManager:
    """LRU + pin + refcount image cache for one node."""

    def __init__(
        self,
        *,
        backend: SandboxBackend,
        pins: set[str] | None = None,
        config: ImageCacheConfig | None = None,
        tier_classifier: TierClassifier | None = None,
        builder_lookup: BuilderLookup | None = None,
        disk_io_sampler: DiskIoSampler | None = None,
    ) -> None:
        self._backend: SkipValidation[SandboxBackend] = backend
        self._pins: set[str] = set(pins or set())
        self._cfg = config or ImageCacheConfig()
        # Optional I/O-saturation signal for the data-root volume. When
        # wired (and ``cfg.io_throttle_enabled``), the AIMD tick treats a
        # pegged volume as "busy" so cold pulls back off before they wedge
        # containerd — even with free disk to spare. ``None`` → free-disk
        # throttling only (prior behaviour).
        self._io_sampler = disk_io_sampler
        self._tier_classifier: TierClassifier = (
            tier_classifier or default_tier_classifier
        )
        # P1.6.g — when set, ensure_present consults this callable
        # before falling back to ``backend.pull_image``. Returns an
        # :data:`ImageProducer` that knows how to materialize the
        # ref locally (e.g. a benchmark builder's ``build()`` call),
        # or ``None`` to fall through to the registry-pull path.
        self._builder_lookup: BuilderLookup = builder_lookup or (
            lambda _ref: None
        )
        self._in_use: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        # Sub-slice 2: monotonic timestamp when ``ensure_present``
        # produced this image via the builder hook (build-on-acquire
        # OR a fresh BuildImageCommand). Within
        # ``cfg.build_grace_window_s`` of this timestamp, the
        # eviction tier sort treats the image as ``recently_used``
        # even without any subsequent ``acquire`` touch — protects
        # a freshly-built image from being reaped before its first
        # rollout. Cleared on first ``acquire``/``release`` so
        # ordinary LRU semantics resume after the grace window OR
        # after the first real touch, whichever comes first.
        self._built_at: dict[str, float] = {}
        # Serializes pull / evict decisions so two concurrent
        # ``ensure_present`` calls don't race past the disk-pressure check.
        self._lock = asyncio.Lock()
        # Per-image pull semaphore so multiple ``ensure_present`` calls for
        # the same image coalesce onto a single in-flight pull.
        self._pull_locks: dict[str, asyncio.Lock] = {}
        # Bounds total concurrent pulls across distinct images via a
        # single node-local AIMD limiter: the controller decays the limit
        # toward ``pull_concurrency`` (floor) when the node is busy and
        # ramps it toward ``pull_concurrency_ceiling`` when idle. The
        # ``run_pull_aimd_loop`` task resizes ``_pull_limiter`` each tick.
        _floor = self._cfg.pull_concurrency
        _ceiling = max(self._cfg.pull_concurrency_ceiling, _floor)
        self._pull_aimd = PullAimdController(
            floor=_floor,
            ceiling=_ceiling,
            initial=self._cfg.pull_concurrency_initial,
            additive_step=self._cfg.pull_aimd_additive_step,
        )
        self._pull_limiter = AdjustableSemaphore(self._pull_aimd.limit)
        self._aimd_loop_running = False
        # Issue #14 — last-sampled disk state, updated on every sweep
        # tick + every ``report()`` so the in-process NodeAgent
        # transport's ``disk_state()`` (sync) can return a fresh value
        # without re-entering the backend.
        self._last_free_disk_bytes: int = 0
        self._last_total_disk_bytes: int = 0
        # Cached image listing (cheap ``list_images`` — no ``docker system
        # df``), refreshed on the sweep tick. The adaptive eviction +
        # disk-bounded pull controllers read ``_largest_image_bytes`` and
        # ``_cached_images`` from here so their per-decision path is a
        # ``statvfs`` + in-memory read, never a per-tick daemon round-trip
        # (which is what wedged the node under heavy pulls).
        self._cached_images: list[ImageRecord] = []
        self._largest_image_bytes: int = 0
        # Monotonic stamp of the last ``_cached_images`` refresh. ``report()``
        # serves the admin /images fan-out from this cache when it is fresher
        # than one sweep interval, so a report RPC arriving mid-build never
        # contends with the build for the Docker daemon's ``images.list`` lock
        # (that contention was the 30 s+ admin-page stall under heavy builds).
        self._cached_images_at: float = 0.0

    # ── State queries ──────────────────────────────────────────────────────

    @property
    def pins(self) -> frozenset[str]:
        return frozenset(self._pins)

    @property
    def config(self) -> ImageCacheConfig:
        return self._cfg

    @property
    def last_free_disk_bytes(self) -> int:
        """Last-sampled free disk bytes (issue #14). ``0`` until the
        first sweep tick or ``report()`` populates it."""
        return self._last_free_disk_bytes

    @property
    def last_total_disk_bytes(self) -> int:
        """Last-sampled total disk bytes (issue #14)."""
        return self._last_total_disk_bytes

    def is_pinned(self, image: str) -> bool:
        return image in self._pins

    def in_use_count(self, image: str) -> int:
        return self._in_use.get(image, 0)

    def tier(self, image: str, *, now: float | None = None) -> ImageTier:
        """Classify ``image`` into the spec-15 priority tier.

        Spec 15 has five tiers; phase-0 collapses ``soon_needed`` (warmup
        directives, phase 1) into ``recently_used`` so callers only see
        the four phase-0 buckets.
        """
        if self._in_use.get(image, 0) > 0:
            return "in_use"
        if image in self._pins:
            return "pinned"
        ts = self._last_used.get(image)
        cur = now if now is not None else time.monotonic()
        if ts is not None and cur - ts <= self._cfg.recent_window_s:
            return "recently_used"
        # Sub-slice 2 — build-time grace window. A freshly-built
        # image has no ``last_used`` touch yet, but reaping it
        # immediately would force a rebuild on the next acquire.
        # Treat it as ``recently_used`` while inside the grace
        # window. Pre-existing ``last_used`` touches always win
        # (the grace check is only consulted after the standard
        # path returns ``cold``).
        built = self._built_at.get(image)
        if built is not None and cur - built <= self._cfg.build_grace_window_s:
            return "recently_used"
        return "cold"

    # ── Refcount hooks ─────────────────────────────────────────────────────

    def acquire(self, image: str) -> None:
        """Bump the in-use refcount (called from ``NodeAgent.create_sandbox``)."""
        self._in_use[image] = self._in_use.get(image, 0) + 1
        self._last_used[image] = time.monotonic()
        # First real touch resets the build grace window — standard
        # LRU semantics take over from here.
        self._built_at.pop(image, None)

    def release(self, image: str) -> None:
        """Decrement the in-use refcount; clamps at zero (idempotent destroy)."""
        cur = self._in_use.get(image, 0)
        if cur <= 1:
            self._in_use.pop(image, None)
        else:
            self._in_use[image] = cur - 1
        self._last_used[image] = time.monotonic()
        # Defensive: also clear here in case acquire was bypassed
        # (release-without-acquire is uncommon but legal).
        self._built_at.pop(image, None)

    def pin(self, image: str) -> None:
        self._pins.add(image)

    def unpin(self, image: str) -> None:
        self._pins.discard(image)

    # ── Pull + evict ───────────────────────────────────────────────────────

    async def evict_ref(
        self, image_ref: str, *, force: bool = False,
    ) -> EvictOutcome:
        """Operator-driven eviction of a specific ref (``xrlenv images evict``).

        Matches ``image_ref`` **registry-agnostically** against the tags
        this node actually holds — so a bare plan ref
        (``repo/name:tag``) matches the registry-qualified tag a node
        pulled (``host:5011/repo/name:tag``) — then removes the matching
        local image(s).

        This is the escape hatch for the mutable-tag staleness problem:
        after a rebuild + re-push under the *same* tag, a node never
        re-pulls on its own (:py:meth:`ensure_present` short-circuits on
        local presence), so it keeps serving the old bytes. Evicting the
        stale tag forces the next acquire to pull the current digest.

        In-use and pinned images are skipped unless ``force`` (so a live
        rollout's image is never yanked out from under it). Returns an
        :class:`EvictOutcome`; never raises for the ordinary
        absent/in-use/daemon-error cases.
        """
        async with self._lock:
            # Refresh against what's actually on disk now — an image
            # pulled since the last sweep would otherwise be invisible.
            # ``_refresh_image_stats`` does not take the lock itself (the
            # eviction sweep also calls it while holding ``self._lock``).
            await self._refresh_image_stats()
            matches = [
                rec for rec in self._cached_images
                if same_image(rec.name, image_ref)
            ]
            if not matches:
                return EvictOutcome(status="absent")

            removed: list[str] = []
            reclaimed = 0
            skipped = 0
            last_error = ""
            for rec in matches:
                name = rec.name
                blocked = self._in_use.get(name, 0) > 0 or name in self._pins
                if blocked and not force:
                    skipped += 1
                    continue
                try:
                    await self._backend.remove_image(name, force=force)
                except ImageInUse:
                    # Held by a container (force=False, or a still-running
                    # container even under force) — skip, never disrupt it.
                    skipped += 1
                    continue
                except Exception as exc:  # daemon unreachable, etc.
                    last_error = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning(
                        "image_cache: evict_ref remove_image(%s) failed: %s",
                        name, last_error,
                    )
                    continue
                removed.append(name)
                reclaimed += int(rec.size_bytes)
                # Drop local accounting so tier()/report() don't show a ghost.
                self._in_use.pop(name, None)
                self._last_used.pop(name, None)
                self._pins.discard(name)
                self._built_at.pop(name, None)

            if removed:
                gone = set(removed)
                self._cached_images = [
                    r for r in self._cached_images if r.name not in gone
                ]
                return EvictOutcome(
                    status="evicted",
                    reclaimed_bytes=reclaimed,
                    removed=tuple(removed),
                )
            if last_error:
                return EvictOutcome(status="failed", detail=last_error)
            # Matches existed but all were skipped (in-use / pinned) — or
            # raced to gone between the listing and the remove.
            return EvictOutcome(status="in_use" if skipped else "absent")

    async def ensure_present(
        self,
        image: str,
        *,
        deadline_s: float | None = None,
        prefetch: bool = False,
    ) -> None:
        with get_tracer().start_as_current_span(
            "xrlenv.node.ensure_present",
            attributes={
                "image": image,
                "deadline_s": deadline_s if deadline_s is not None else -1.0,
                "prefetch": prefetch,
            },
        ) as _span:
            if await self._is_present(image):
                _span.set_attribute("cache_hit", True)
                self._last_used[image] = time.monotonic()
                return
            _span.set_attribute("cache_hit", False)
            await self._ensure_present_impl(image, deadline_s=deadline_s)

    async def _ensure_present_impl(
        self,
        image: str,
        *,
        deadline_s: float | None = None,
    ) -> None:
        """Block until ``image`` is local; trigger eviction if needed.

        Behaviour:
        1. If the image is already cached, return immediately.
        2. Otherwise check free disk; evict cold images until either
           ``evict_target_bytes`` is reached or no more evictable images
           remain.
        3. Pull. The pull is serialized per-image (concurrent calls for
           the same image coalesce) and bounded by
           ``ImageCacheConfig.pull_concurrency`` across distinct images.
        4. Bound total wall-clock by ``deadline_s`` (default
           ``config.default_pull_timeout_s``).

        Returns silently on success; raises ``TimeoutError`` on deadline
        miss or :class:`OutOfDiskAfterEviction` when even after eviction
        the pinned set leaves no room (operator misconfiguration).
        """
        if await self._is_present(image):
            self._last_used[image] = time.monotonic()
            return

        timeout_s = deadline_s if deadline_s is not None else self._cfg.default_pull_timeout_s

        async with self._pull_lock_for(image):
            # Re-check after acquiring the per-image lock; another
            # concurrent ensure_present may have just pulled it.
            if await self._is_present(image):
                self._last_used[image] = time.monotonic()
                return
            await self._evict_if_needed()
            # Single node-local AIMD limiter for all pulls. Its bound is
            # resized by ``run_pull_aimd_loop``: it decays toward the
            # floor when the node is busy with live rollouts and ramps
            # toward the ceiling when idle, so cold pulls never starve
            # time-sensitive agents yet saturate the pipe on an idle
            # cluster. (The ``prefetch`` flag on ``ensure_present`` is
            # telemetry only now — limiting is unified and adaptive.)
            async with self._pull_limiter:
                # P1.6.g — H3 lazy lifecycle: if a benchmark builder
                # is registered for this ref (e.g.
                # ``swebench-verified/<inst>:0.1``, which isn't
                # registry-pullable), invoke the builder's
                # ``build()`` instead of ``backend.pull_image``.
                # Falls through to the registry-pull path for
                # ordinary images (``python:3.12-slim`` etc.).
                producer = self._builder_lookup(image)
                if producer is not None:
                    LOGGER.info(
                        "image_cache: producing %s via builder "
                        "(timeout=%gs)", image, timeout_s,
                    )
                    await producer(image, timeout_s)
                    # Sub-slice 2: stamp the build-time grace window
                    # so a later eviction loop sees this image as
                    # ``recently_used`` even before its first
                    # ``acquire`` touch.
                    self._built_at[image] = time.monotonic()
                else:
                    LOGGER.info(
                        "image_cache: pulling %s (timeout=%gs)",
                        image, timeout_s,
                    )
                    await self._pull_with_retry(image, timeout_s)
        self._last_used[image] = time.monotonic()

    async def _pull_with_retry(self, image: str, timeout_s: float) -> None:
        """Pull ``image`` via the backend, retrying transient failures
        within the ``timeout_s`` deadline budget (issue #18).

        A registry — or its auth endpoint — is flaky under heavy
        concurrent cold-pull load; a single transient connection
        timeout would otherwise lose the whole acquire. Retries up
        to :py:attr:`ImageCacheConfig.pull_max_attempts` times,
        with exponential backoff, but never past the deadline. A
        genuinely missing image fails the same way every attempt
        and surfaces the last error once the attempt budget is
        spent — the few seconds of wasted backoff is an acceptable
        price for salvaging the far more common transient case.
        """
        deadline = time.monotonic() + timeout_s
        attempts = max(1, self._cfg.pull_max_attempts)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            remaining = deadline - time.monotonic()
            # The first attempt always runs; a later one only while
            # the deadline still has room. ``remaining`` is the wire
            # budget handed to this attempt.
            per_attempt = remaining if remaining > 0 else max(timeout_s, 1.0)
            try:
                await self._backend.pull_image(image, timeout_s=per_attempt)
                return
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                backoff = min(2.0 ** (attempt - 1), 15.0)
                if deadline - time.monotonic() <= backoff:
                    break  # no budget left to retry
                LOGGER.warning(
                    "image_cache: pull %s failed (attempt %d/%d): %r; "
                    "retrying in %.0fs",
                    image, attempt, attempts, exc, backoff,
                )
                await asyncio.sleep(backoff)
        assert last_exc is not None  # the loop body ran at least once
        raise last_exc

    async def _evict_if_needed(self) -> None:
        """Evict cold images until free disk reaches the effective target.

        Stops early when no more evictable images are left. Caller's
        ``ensure_present`` decides whether to proceed or surface
        :class:`OutOfDiskAfterEviction`.
        """
        free = await self._backend.free_disk_bytes()  # statvfs — cheap, no daemon
        total = await self._backend.total_disk_bytes()
        # Issue #14 — refresh the cached disk state every sweep tick so
        # the in-process disk_state() probe + the heartbeat sampler
        # both see fresh-enough numbers between actual eviction work.
        self._last_free_disk_bytes = max(0, int(free))
        self._last_total_disk_bytes = max(0, int(total))
        # Cheap path: threshold reads the cached largest image size — no
        # daemon call — so an idle/healthy node pays only one ``statvfs``.
        if free >= self._effective_threshold():
            return

        async with self._lock:
            # Fallback for the sweep-disabled path: ensure we have a
            # listing to evict from even if the periodic refresh never ran.
            if not self._cached_images:
                await self._refresh_image_stats()
            target = self._effective_target()
            # Evict from the cached listing (refreshed on the sweep tick /
            # by report()), so eviction never blocks on a live
            # ``list_images`` under load. Stale-but-safe: ``remove_image``
            # on an already-gone image is caught below.
            cold = self._cold_lru_order(self._cached_images)
            for record in cold:
                free = await self._backend.free_disk_bytes()
                if free >= target:
                    break
                # Log AFTER a successful removal, not before the attempt.
                # On a full disk the cold list is re-walked every sweep;
                # images that can't actually be removed (held by a
                # non-xrlenv container — e.g. the HyperPod otel/efa
                # sidecars) would otherwise emit an INFO line each tick
                # forever (the prod log-spam during the disk-full wedge).
                try:
                    await self._backend.remove_image(record.name)
                    LOGGER.info(
                        "image_cache: evicted cold image %s "
                        "(size=%dB free=%dB target=%dB)",
                        record.name, record.size_bytes, free, target,
                    )
                except ImageInUse:
                    # Held by a non-xrlenv container (e.g. a node sidecar).
                    # Expected and safe — skip quietly rather than spam
                    # every sweep; the image simply isn't reclaimable here.
                    LOGGER.debug(
                        "image_cache: skip eviction of %s — in use by a "
                        "non-xrlenv container (held externally)", record.name,
                    )
                except Exception:
                    LOGGER.exception(
                        "image_cache: remove_image(%s) failed; continuing",
                        record.name,
                    )

    def _adaptive_headroom(self, slots: int, floor: int, cap: int) -> int:
        """Free-disk headroom sized to the live pull burst, not the disk.

        ``slots x largest_cached_image x disk_safety_factor`` — so a
        big-image workload reserves more and a small-image one less, on
        ANY disk size — clamped to ``[floor, cap]``. Falls back to the
        absolute ``floor`` when no image size is known yet (empty cache).
        Reads the cached ``_largest_image_bytes`` (no daemon call).
        """
        s = self._largest_image_bytes
        if s <= 0:
            return floor
        reserve = int(slots * s * self._cfg.evict_disk_safety_factor)
        return max(floor, min(reserve, cap))

    def _effective_threshold(self) -> int:
        """Eviction-START headroom (keep free ≥ this).

        Floored at ``CACHE_EVICT_START_MARGIN x scheduler-admit-floor``
        so eviction begins BEFORE free disk drops into the scheduler's
        placement-excluded range. Without this floor the cache's band
        could sit at/below the scheduler's admit threshold and pin a
        node permanently excluded (the P1 disk-exclusion deadlock; see
        ``xrlenv.disk_policy`` + notes/audit.md)."""
        return max(
            self._adaptive_headroom(
                self._cfg.evict_headroom_slots,
                self._cfg.evict_threshold_bytes,
                self._cfg.evict_threshold_cap_bytes,
            ),
            cache_evict_floor_bytes(
                self._last_total_disk_bytes, CACHE_EVICT_START_MARGIN,
            ),
        )

    def _effective_target(self) -> int:
        """Eviction-STOP headroom — one slot above the start, for
        hysteresis so eviction doesn't immediately re-trigger.

        Floored at ``CACHE_EVICT_TARGET_MARGIN x scheduler-admit-floor``
        (> the start margin) so a node the cache maintains at its target
        is always comfortably ABOVE the scheduler's disk-admit threshold
        and therefore never deadlocks excluded."""
        return max(
            self._adaptive_headroom(
                self._cfg.evict_headroom_slots + 1,
                self._cfg.evict_target_bytes,
                self._cfg.evict_target_cap_bytes,
            ),
            cache_evict_floor_bytes(
                self._last_total_disk_bytes, CACHE_EVICT_TARGET_MARGIN,
            ),
        )

    # ── Disk-guard hooks (WS2) ───────────────────────────────────────────────
    #
    # The node-side DiskPressureGuard (xrlenv.node.disk_guard) reuses the
    # cache's *adaptive* free-disk headroom as its pressure threshold (no
    # fixed disk-fraction) and asks the cache how much space image eviction
    # could still reclaim — so the guard only kills a runaway *container*
    # when freeing images can't relieve the pressure (the prod
    # RECLAIMABLE-0 / writable-layer-fills-disk failure). All three read
    # cached state — no daemon round-trip — so the guard's hot path stays
    # one ``statvfs``.

    async def sample_disk(self) -> tuple[int, int]:
        """Fresh ``(free_bytes, total_bytes)`` for the data-root via one
        ``statvfs`` (cheap, no docker daemon round-trip). The disk guard
        polls this every tick — a runaway writable layer fills fast, so
        the guard needs a live reading, not the sweep-cadence cache."""
        free = await self._backend.free_disk_bytes()
        total = await self._backend.total_disk_bytes()
        return (max(0, int(free)), max(0, int(total)))

    def effective_evict_threshold(self) -> int:
        """Adaptive free-disk level at/below which the node is pressured
        (the same level image eviction starts at)."""
        return self._effective_threshold()

    def effective_evict_target(self) -> int:
        """Adaptive free-disk level the node should recover to."""
        return self._effective_target()

    def evictable_image_bytes(self) -> int:
        """Upper bound on free space image eviction can still reclaim:
        the summed size of cold (not in-use, not pinned) cached images.
        ``0`` means freeing images cannot relieve disk pressure — the
        signal the disk guard uses to decide a runaway container, not an
        image, is the culprit. Reads the cached listing (no daemon call);
        overcounts shared layers, which is conservative here (it can only
        make the guard defer to image eviction, never over-kill)."""
        return sum(
            r.size_bytes for r in self._cold_lru_order(self._cached_images)
        )

    def _store_image_stats(self, images: list[ImageRecord]) -> None:
        """Replace the cached listing + derived stats from a fresh listing.

        Single assignment site for ``_cached_images`` / ``_largest_image_bytes``
        / ``_cached_images_at`` so the freshness stamp can never drift from the
        data it stamps. Callers pass whatever ``list_images`` returned."""
        self._cached_images = images
        self._largest_image_bytes = max(
            (int(img.size_bytes) for img in images), default=0,
        )
        self._cached_images_at = time.monotonic()

    async def _refresh_image_stats(self) -> None:
        """Refresh the cached image listing + largest-image size with a
        cheap ``list_images`` (no ``docker system df``). Best-effort: on
        failure keep the last-known values so the control loops degrade to
        stale data rather than blocking/erroring."""
        try:
            images = await self._backend.list_images(include_shared_size=False)
        except Exception:
            LOGGER.debug(
                "image_cache: list_images refresh failed; keeping last stats",
                exc_info=True,
            )
            return
        self._store_image_stats(images)

    async def run_sweep_loop(
        self, *, interval_s: float | None = None,
    ) -> None:
        """Periodic background-eviction loop (issue #13).

        Wakes every ``interval_s`` seconds (default
        :py:attr:`ImageCacheConfig.sweep_interval_s`) and runs
        :py:meth:`_evict_if_needed`. The eviction call is a no-op
        when free disk is comfortably above the effective high-
        water mark, so an idle cluster with healthy disk pays only
        the cost of one ``statvfs`` per tick.

        Closes the trigger gap where the on-pull eviction path
        (``ensure_present`` cache miss) never fires during a
        steady-state workload that reuses cached images while
        sandbox writable-overlay growth quietly fills the disk.

        Cancels cleanly on ``CancelledError``; the caller owns the
        task's lifetime (see ``xrlenv/node/grpc_link.py`` for the
        production wiring next to the heartbeat loop).
        """
        period = interval_s if interval_s is not None else self._cfg.sweep_interval_s
        if period <= 0:
            return
        # Warm the cached image stats immediately so the adaptive eviction
        # + pull controllers have a largest-image size before the first
        # tick (otherwise they fall back to the absolute floor / static
        # ceiling for one period).
        await self._refresh_image_stats()
        try:
            while True:
                await asyncio.sleep(period)
                try:
                    # Cheap ``list_images`` (no df) — the only periodic
                    # daemon call; keeps the controllers' inputs fresh.
                    await self._refresh_image_stats()
                    await self._evict_if_needed()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception(
                        "image_cache: background eviction sweep failed; "
                        "will retry on next tick",
                    )
        except asyncio.CancelledError:
            return

    @property
    def pull_concurrency_limit(self) -> int:
        """Current adaptive pull-concurrency bound (for telemetry/tests)."""
        return self._pull_limiter.limit

    async def run_pull_aimd_loop(
        self, *, interval_s: float | None = None,
    ) -> None:
        """Node-local AIMD loop: resize the pull limiter to track load.

        Each tick reads the node's in-use container count: **busy**
        (above :attr:`ImageCacheConfig.pull_busy_threshold`) →
        multiplicative-decrease the pull limit toward the floor; **calm**
        → additive-increase toward the ceiling. This keeps cold pulls
        from starving live, time-sensitive agents while still saturating
        the registry / FSx pipe on an idle cluster — the node finds its
        own safe pull rate, no restart or manual tuning required.

        Idempotent: a second concurrent call returns immediately, so it
        is safe to launch from both the runtime and the gRPC reconnect
        path. Cancels cleanly on ``CancelledError``; the caller owns the
        task's lifetime (mirrors :meth:`run_sweep_loop`).
        """
        if not self._cfg.pull_aimd_enabled or self._aimd_loop_running:
            return
        period = (
            interval_s if interval_s is not None else self._cfg.pull_aimd_interval_s
        )
        if period <= 0:
            return
        self._aimd_loop_running = True
        try:
            while True:
                await asyncio.sleep(period)
                try:
                    prev = self._pull_aimd.limit
                    # Disk-bounded ceiling: never allow more concurrent
                    # pulls than the current free disk can buffer
                    # (free / (largest_image x safety)). This is what makes
                    # the small adaptive eviction reserve safe — pulls
                    # auto-throttle as the disk fills, so they can't
                    # overshoot. ``free`` is a cheap statvfs; the image
                    # size is the cached value (no per-tick daemon call).
                    free = await self._backend.free_disk_bytes()
                    s = self._largest_image_bytes
                    if s > 0:
                        disk_ceiling = max(
                            self._pull_aimd.floor,
                            int(free / (s * self._cfg.evict_disk_safety_factor)),
                        )
                        self._pull_aimd.set_ceiling(
                            min(self._cfg.pull_concurrency_ceiling, disk_ceiling),
                        )
                    in_use_total = sum(self._in_use.values())
                    # I/O-saturation backoff: a pegged data-root volume
                    # (IOPS/throughput-capped EBS) counts as busy even with
                    # free disk to spare, so pulls decay toward the floor
                    # instead of overrunning the device and wedging
                    # containerd's teardown path. Fail-open: no sampler /
                    # unreadable /sys → io_saturated False (free-disk
                    # throttling only).
                    io_saturated = False
                    io_util: float | None = None
                    if (
                        self._io_sampler is not None
                        and self._cfg.io_throttle_enabled
                    ):
                        io_saturated = self._io_sampler.saturated()
                        io_util = self._io_sampler.last_utilization
                    busy = (
                        in_use_total > self._cfg.pull_busy_threshold
                        or io_saturated
                    )
                    new = self._pull_aimd.observe(busy=busy)
                    if new != prev:
                        await self._pull_limiter.set_limit(new)
                        LOGGER.info(
                            "image_cache: pull concurrency %d → %d "
                            "(in_use=%d, busy=%s, io_util=%s, "
                            "io_saturated=%s, free=%dB, ceiling=%d)",
                            prev, new, in_use_total, busy,
                            f"{io_util:.2f}" if io_util is not None else "n/a",
                            io_saturated, free, self._pull_aimd.ceiling,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception(
                        "image_cache: pull AIMD tick failed; "
                        "will retry on next tick",
                    )
        except asyncio.CancelledError:
            return
        finally:
            self._aimd_loop_running = False

    def _cold_lru_order(self, images: list[ImageRecord]) -> list[ImageRecord]:
        """Return the eviction-eligible images sorted ``(tier, LRU)``.

        Eligibility: not in-use, not pinned. Sort keys, in order:

        1. **Eviction tier** via ``self._tier_classifier`` —
           ``final`` (cheapest to recreate) goes before
           ``stub_runtime`` goes before ``base``. See :data:`EvictionTier`.
        2. **LRU timestamp** within the tier — oldest first; images we
           never observed in-use this process lifetime sort as
           "infinitely old" (timestamp = 0.0) so they evict first
           within their tier.

        The tier-first ordering is the D16 contract: under disk pressure
        we should burn rebuild-cheap final tags before we burn the
        expensive base layers, even if a cold base happens to be older
        than a cold final.
        """
        candidates = [
            img for img in images
            if self._in_use.get(img.name, 0) == 0 and img.name not in self._pins
        ]

        def _key(img: ImageRecord) -> tuple[int, float]:
            ts = self._last_used.get(img.name, 0.0)
            tier = self._tier_classifier(img.name, img.labels)
            return (_TIER_PRIORITY[tier], ts)

        return sorted(candidates, key=_key)

    # ── Reporting ──────────────────────────────────────────────────────────

    async def query(self, image: str) -> ImageQueryResult:
        """A1 / D18+D19 (P1.2) — answer "do you have this image?" with
        as much per-image metadata as the cache can cheaply produce.

        Used by:

        - The scheduler's image-affinity scoring (D18) — prefers nodes
          whose cache has the image.
        - The coordinator's pre-flight check (D19) — fails the rollout
          fast with ``reason="image_missing"`` when the chosen node
          doesn't have it.

        Cheap path: ``image_exists`` is the only mandatory backend
        round-trip. Digest lookup is best-effort (``list_images`` is
        cheap on Docker but the result still walks the daemon's
        index); on backends where that's expensive we'd add a
        per-image digest cache, but Docker handles it fine today.
        """
        present = await self._is_present(image)
        digest: str | None = None
        if present:
            try:
                for record in await self._backend.list_images():
                    if record.name == image:
                        digest = record.digest
                        break
            except Exception:
                LOGGER.debug(
                    "image_cache.query: list_images failed for digest "
                    "lookup of %s; reporting present=True with no digest",
                    image,
                    exc_info=True,
                )
        last_used = self._last_used.get(image, 0.0) if present else 0.0
        return ImageQueryResult(
            present=present, digest=digest, last_used_at=last_used,
        )

    async def report(
        self, *, include_shared_size: bool = False,
    ) -> NodeImageReport:
        images = await self._images_for_report(
            include_shared_size=include_shared_size,
        )
        now = time.monotonic()
        records: list[ImageStateRecord] = []
        for img in images:
            tier = self.tier(img.name, now=now)
            owner = default_ownership_classifier(img.labels)
            records.append(
                ImageStateRecord(
                    name=img.name,
                    tier=tier,
                    size_bytes=img.size_bytes,
                    shared_size_bytes=img.shared_size_bytes,
                    in_use_count=self._in_use.get(img.name, 0),
                    last_used_at=self._last_used.get(img.name),
                    pinned=img.name in self._pins,
                    owner=owner,
                    digest=img.digest,
                )
            )
        # Disk figures are ``statvfs`` (an OS call, no Docker round-trip), so
        # they stay live even on the cache-served path — they're cheap and
        # never contend with a running build.
        free = await self._backend.free_disk_bytes()
        total = await self._backend.total_disk_bytes()
        self._last_free_disk_bytes = max(0, int(free))
        self._last_total_disk_bytes = max(0, int(total))
        return NodeImageReport(
            images=records,
            free_disk_bytes=free,
            pinned=tuple(sorted(self._pins)),
        )

    async def _images_for_report(
        self, *, include_shared_size: bool,
    ) -> list[ImageRecord]:
        """The image listing ``report()`` renders, served daemon-free when
        possible.

        - ``include_shared_size=True`` (``xrlenv build calibrate`` / the
          budget provider) always does a live listing — the SharedSize walk
          is the whole point and the cache never carries it.
        - Otherwise serve the sweep-maintained ``_cached_images`` when it is
          fresher than one sweep interval. This is the admin /images hot path:
          a report RPC arriving during a heavy build returns from memory
          instead of competing with the build for the Docker ``images.list``
          lock. Staleness is bounded by ``sweep_interval_s`` (≈60 s), which is
          well within the page's own 60 s auto-refresh cadence.
        - On a cold/stale cache, do one live listing (propagating errors as
          before so an unreachable daemon still surfaces) and store it.
        """
        if not include_shared_size and self._cached_images and (
            time.monotonic() - self._cached_images_at
            <= self._cfg.sweep_interval_s
        ):
            return self._cached_images
        images = await self._backend.list_images(
            include_shared_size=include_shared_size,
        )
        # Refresh the adaptive controllers' cached stats from this listing —
        # free since we already have it. (When include_shared_size=True the
        # records still carry size_bytes, so the largest-image stat is valid;
        # only shared_size differs, which the eviction path doesn't read.)
        self._store_image_stats(images)
        return images

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _is_present(self, image: str) -> bool:
        # Backend-native existence check: handles both tag-form and
        # digest-form refs natively (Docker's ``images.get(ref)``).
        # The list_images()-and-iterate fallback misses the case where
        # the catalog pinned an image to its local content-addressed
        # ``Id`` ("xrlenv/hello-shell@sha256:...") but list_images
        # only reports the tag form ("xrlenv/hello-shell:0.1") — the
        # mismatch would trigger a doomed registry pull for a
        # locally-built image.
        return await self._backend.image_exists(image)

    def _pull_lock_for(self, image: str) -> asyncio.Lock:
        if image not in self._pull_locks:
            self._pull_locks[image] = asyncio.Lock()
        return self._pull_locks[image]


__all__ = [
    "EvictionTier",
    "ImageCacheConfig",
    "ImageCacheManager",
    "ImageQueryResult",
    "ImageStateRecord",
    "ImageTier",
    "NodeImageReport",
    "TierClassifier",
    "default_tier_classifier",
]
