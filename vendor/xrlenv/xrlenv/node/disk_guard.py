"""WS2 — node-autonomous disk-pressure guard.

A runaway rollout that writes into its container's overlay upper-dir can
fill the node data-root: Docker imposes no per-container writable-layer
cap, and the prod XFS data-root is mounted ``noquota`` so
``--storage-opt size=`` isn't available. When the disk hits 100% the
node wedges — ``container_get_archive`` replies fail to serialize, the
heartbeat stream breaks, and the control plane marks the node lost (the
failure that took ``aws-node-host`` offline for 3.6h).

The image-cache eviction sweep cannot help here: it frees *image*
layers, but a runaway *writable layer* is not an image, and once every
cached image backs a running container nothing is reclaimable (the
futile per-tick eviction log-spam observed in prod). This guard closes
that gap.

Hot path is one ``statvfs`` per tick. Only when free disk falls below
the image cache's *adaptive* pressure threshold **and** image eviction
can't relieve it (``evictable_image_bytes`` < the shortfall) does the
guard pay for the expensive offender scan (``docker ps -s``) and kill
the largest-writable-layer raw container — failing that one rollout
cleanly instead of losing the whole node.

There are no fixed disk-fraction thresholds: the pressure/recovery
levels come from the image cache's adaptive ``slots x largest-image x
safety`` reserve, so they scale with the workload and the disk rather
than a hardcoded percentage (see ``notes`` / the user's adaptive-infra
preference).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from xrlenv.node.raw_container import RawContainerDiskUsage

LOGGER = logging.getLogger("xrlenv.node.disk_guard")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("disk-guard: ignoring non-numeric %s=%r", name, raw)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class DiskGuardConfig(BaseModel):
    """Operator-tunable guard knobs. Defaults are deliberately the only
    constants here — the *thresholds* are adaptive (sourced from the
    image cache), so what's left is just cadence + an on/off switch."""

    #: ``statvfs`` poll cadence. Tighter than the 60 s image sweep
    #: because a runaway writable layer can fill a disk fast.
    interval_s: float = 15.0
    #: Master switch — set False to disable autonomous container kills.
    enabled: bool = True

    @classmethod
    def from_env(cls) -> DiskGuardConfig:
        return cls(
            interval_s=_env_float("XRLENV_DISK_GUARD_INTERVAL_S", 15.0),
            enabled=_env_bool("XRLENV_DISK_GUARD_ENABLED", True),
        )


# Dependency callables (injected for testability — no live docker / cache
# needed to unit-test the policy).
SampleDisk = Callable[[], Awaitable[tuple[int, int]]]
ThresholdFn = Callable[[], int]
EvictableFn = Callable[[], int]
ListOffenders = Callable[[], Awaitable[list[RawContainerDiskUsage]]]
KillFn = Callable[[RawContainerDiskUsage], Awaitable[None]]


class DiskPressureGuard:
    """Polls free disk and kills runaway raw containers under pressure
    that image eviction can't relieve.

    All collaborators are injected so the policy is unit-testable with
    plain callables. In production they're wired from the node's image
    cache (thresholds + disk sample + evictable bytes) and raw-container
    manager (offender scan + force-destroy) by :func:`build_disk_guard`.
    """

    def __init__(
        self,
        *,
        sample_disk: SampleDisk,
        critical_threshold: ThresholdFn,
        recovery_target: ThresholdFn,
        evictable_image_bytes: EvictableFn,
        list_offenders: ListOffenders,
        kill: KillFn,
        cfg: DiskGuardConfig | None = None,
    ) -> None:
        self._sample_disk = sample_disk
        self._critical_threshold = critical_threshold
        self._recovery_target = recovery_target
        self._evictable_image_bytes = evictable_image_bytes
        self._list_offenders = list_offenders
        self._kill = kill
        self._cfg = cfg or DiskGuardConfig()

    async def check_once(self) -> list[RawContainerDiskUsage]:
        """One pressure assessment. Returns the containers it killed
        (empty when healthy or when image eviction can handle it)."""
        free, total = await self._sample_disk()
        critical = int(self._critical_threshold())
        # ``(0, 0)`` is the backend's "couldn't sample" sentinel — treat
        # as healthy (don't kill on a blind reading), matching the
        # heartbeat disk-state convention.
        if total <= 0 or free >= critical:
            return []

        evictable = int(self._evictable_image_bytes())
        recovery = int(self._recovery_target())
        shortfall = max(0, recovery - free)
        if evictable >= shortfall:
            # Freeing cold images can recover the shortfall — let the
            # image-eviction sweep handle it; don't kill a rollout.
            LOGGER.info(
                "disk-guard: free=%dB < critical=%dB but %dB of images are "
                "evictable (>= %dB shortfall); deferring to image eviction",
                free, critical, evictable, shortfall,
            )
            return []

        offenders = sorted(
            await self._list_offenders(),
            key=lambda o: o.size_rw_bytes,
            reverse=True,
        )
        killed: list[RawContainerDiskUsage] = []
        projected_free = free
        for off in offenders:
            if projected_free >= recovery:
                break
            if off.size_rw_bytes <= 0:
                continue
            LOGGER.warning(
                "disk-guard: killing runaway raw container container=%s "
                "rollout=%s image=%s writable=%dB — node free=%dB below "
                "critical=%dB and only %dB reclaimable from images",
                off.container_id[:12], off.rollout_id or "?", off.image,
                off.size_rw_bytes, free, critical, evictable,
            )
            try:
                await self._kill(off)
            except Exception:
                LOGGER.exception(
                    "disk-guard: failed to kill container=%s; continuing",
                    off.container_id[:12],
                )
                continue
            killed.append(off)
            projected_free += off.size_rw_bytes

        if not killed:
            LOGGER.error(
                "disk-guard: node under disk pressure (free=%dB < critical="
                "%dB, only %dB image-reclaimable) but found no killable "
                "raw-container offender — manual intervention may be required",
                free, critical, evictable,
            )
        elif projected_free < recovery:
            LOGGER.warning(
                "disk-guard: killed %d container(s) (~%dB) but projected "
                "free=%dB still below recovery=%dB; will reassess next tick",
                len(killed), projected_free - free, projected_free, recovery,
            )
        return killed

    async def run_loop(self) -> None:
        """Periodic poll. Idempotent to relaunch (per reconnect), like
        the image-cache sweep loop; cancellation ends it cleanly."""
        if not self._cfg.enabled:
            LOGGER.info("disk-guard: disabled via config; not polling")
            return
        try:
            while True:
                await asyncio.sleep(self._cfg.interval_s)
                try:
                    await self.check_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception("disk-guard: check_once failed")
        except asyncio.CancelledError:
            return


def build_disk_guard(agent: object) -> DiskPressureGuard | None:
    """Wire a guard from a node ``agent``'s image cache + raw manager.

    Returns ``None`` when the node has no image cache or no docker-backed
    raw-container manager (e.g. a fake-backend test node) — the caller
    simply skips launching the guard loop.
    """
    cache = getattr(agent, "image_cache", None)
    raw_manager = None
    getter = getattr(agent, "raw_container_manager", None)
    if callable(getter):
        raw_manager = getter("docker")
    if cache is None or raw_manager is None:
        return None

    async def _kill(off: RawContainerDiskUsage) -> None:
        # Audit P3 — record WHY before the kill so the control-plane
        # reconciler can seal this rollout with the real disk-pressure
        # cause (surfaced via ListRawContainersReply.reaped_reasons)
        # instead of a generic "container vanished" teardown message.
        note = getattr(raw_manager, "note_disk_reaped", None)
        if callable(note) and off.rollout_id:
            note(
                off.rollout_id,
                f"disk-guard: reaped runaway raw container "
                f"(writable {off.size_rw_bytes} bytes) to relieve node "
                f"disk pressure",
            )
        await raw_manager.force_destroy(container_id=off.container_id)

    return DiskPressureGuard(
        sample_disk=cache.sample_disk,
        critical_threshold=cache.effective_evict_threshold,
        recovery_target=cache.effective_evict_target,
        evictable_image_bytes=cache.evictable_image_bytes,
        list_offenders=raw_manager.list_disk_usage,
        kill=_kill,
        cfg=DiskGuardConfig.from_env(),
    )


__all__ = [
    "DiskGuardConfig",
    "DiskPressureGuard",
    "build_disk_guard",
]
