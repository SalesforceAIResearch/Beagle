"""Adaptive disk-I/O saturation signal (node-local).

The image-cache pull controller already throttles on *free disk*
(``free / (largest_image x safety)``), but a near-full or IOPS-capped
volume can be fully saturated while it still has free space. That was the
symptom that wedged ``containerd`` under heavy cold-pull + in-container
``git reset`` load on the HyperPod EBS volumes: ``%util`` pegged at 100 %
with a ~140-deep request queue, so ``docker rm`` (destroy) commands
parked in D-state and timed out — while the *free-disk* controller saw
plenty of space and kept admitting pulls.

:class:`DiskIoSampler` reads the kernel's per-device ``io_ticks`` (the
milliseconds the device had at least one I/O in flight) from ``/sys`` and
computes ``%util`` over the interval between calls — exactly what
``iostat`` reports. It exposes a hysteretic :meth:`saturated` that the
pull controller ORs into its "busy" signal, so pulls back off toward the
floor while the data-root volume is pegged and ramp back up once it
clears.

Design notes:

* **Adapts to the hardware — no provisioned-IOPS guess.** We never encode
  "3000 IOPS"; we observe saturation (``util`` ≈ 100 %) and discover
  whatever ceiling the underlying EBS/gp3 volume actually has. The
  high/low watermarks are *saturation-detection* points on a physically
  bounded ``[0, 1]`` quantity (utilization), not a capacity assumption —
  and both are operator-tunable. This is the same spirit as the eviction
  headroom (sized to observed image size, not a fixed disk fraction).
* **Sub-second hot path.** A sample is one small ``/sys`` file read,
  rate-limited to ``min_interval_s``; callers between samples read the
  cached value. No daemon round-trip.
* **Fail-open.** Any failure (no ``/sys``, parse error, unresolved
  device, non-Linux, counter wrap) → :meth:`utilization` returns ``None``
  → :meth:`saturated` returns ``False``: I/O throttling silently disables
  and the node falls back to free-disk-only behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)

# Field index of ``io_ticks`` in /sys/block/<dev>/stat (and the symlinked
# /sys/dev/block/<major>:<minor>/stat): the cumulative milliseconds the
# device had at least one I/O in flight. See Linux
# Documentation/block/stat.rst. Index is stable across kernels (newer
# kernels append discard/flush fields after the historical 11).
_IO_TICKS_FIELD: int = 9


class DiskIoSampler:
    """Hysteretic disk-I/O saturation signal for one block device.

    Construct with a ``path_provider`` returning a filesystem path on the
    volume to watch (the docker data-root); the device is resolved lazily
    on first sample so construction never blocks on the daemon. For tests,
    inject ``io_ticks_reader`` (and ``clock``) to bypass ``/sys`` entirely.
    """

    def __init__(
        self,
        *,
        path_provider: Callable[[], str | None],
        high: float = 0.90,
        low: float = 0.70,
        min_interval_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        io_ticks_reader: Callable[[], int | None] | None = None,
    ) -> None:
        if not 0.0 < low <= high <= 1.0:
            raise ValueError("require 0 < low <= high <= 1")
        self._path_provider = path_provider
        self._high = high
        self._low = low
        self._min_interval_s = max(0.0, min_interval_s)
        self._clock = clock
        self._io_ticks_reader = io_ticks_reader
        # Lazily-resolved /sys stat path. ``_resolved`` sticks True only
        # once the provider returned a real path (success or not), so we
        # keep retrying while it returns None (daemon still starting).
        self._resolved = False
        self._stat_path: str | None = None
        self._last_sample_t: float | None = None
        self._last_io_ticks: int | None = None
        self._last_util: float | None = None
        self._saturated = False

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def last_utilization(self) -> float | None:
        """The last computed ``%util`` in ``[0, 1]`` (cached, no sample).

        ``None`` before two samples exist or when the device can't be
        measured. Use for logging alongside :meth:`saturated`."""
        return self._last_util

    def utilization(self) -> float | None:
        """Return ``%util`` (``[0, 1]``) over the interval since the last
        sample, or ``None`` when it can't be measured.

        Rate-limited to ``min_interval_s``: within that window the cached
        value is returned, so calling this per scheduling decision costs
        nothing. The first call (no prior sample) returns ``None``.
        """
        now = self._clock()
        if (
            self._last_sample_t is not None
            and now - self._last_sample_t < self._min_interval_s
        ):
            return self._last_util
        ticks = self._read_io_ticks()
        prev_t = self._last_sample_t
        prev_ticks = self._last_io_ticks
        self._last_sample_t = now
        self._last_io_ticks = ticks
        if ticks is None:
            self._last_util = None
            return None
        if prev_t is None or prev_ticks is None:
            # First readable sample — no delta yet.
            return self._last_util
        dt = now - prev_t
        if dt <= 0:
            return self._last_util
        dticks = ticks - prev_ticks
        if dticks < 0:
            # Counter reset / wrap — discard this interval.
            self._last_util = None
            return None
        util = dticks / (dt * 1000.0)
        util = 0.0 if util < 0.0 else (1.0 if util > 1.0 else util)
        self._last_util = util
        return util

    def saturated(self) -> bool:
        """Hysteretic saturation verdict: flips ``True`` once ``util``
        crosses ``high``, stays ``True`` until it falls to ``low``, then
        flips back. Fail-open: ``None`` utilization → ``False`` (never
        throttle on a signal we can't read)."""
        u = self.utilization()
        if u is None:
            # Fail-open AND clear the latch: an unreadable sample must not
            # leave the sampler stuck "saturated", or a transient /sys
            # hiccup would keep pulls throttled until util later dips below
            # the low watermark — even though we can no longer measure it
            # (audit M1). Honors the documented "None utilization -> False".
            self._saturated = False
            return False
        if u >= self._high:
            self._saturated = True
        elif u <= self._low:
            self._saturated = False
        return self._saturated

    # ── internals ─────────────────────────────────────────────────────────────

    def _read_io_ticks(self) -> int | None:
        if self._io_ticks_reader is not None:
            try:
                return self._io_ticks_reader()
            except Exception:
                return None
        stat_path = self._resolve_stat_path()
        if stat_path is None:
            return None
        try:
            with open(stat_path) as fh:
                fields = fh.read().split()
            return int(fields[_IO_TICKS_FIELD])
        except Exception:
            LOGGER.debug("disk_io: read %s failed", stat_path, exc_info=True)
            return None

    def _resolve_stat_path(self) -> str | None:
        if self._resolved:
            return self._stat_path
        try:
            fs_path = self._path_provider()
        except Exception:
            fs_path = None
        if not fs_path:
            # Provider not ready (e.g. docker info not up yet) — retry on
            # a later sample rather than sticking to "unavailable".
            return None
        self._resolved = True
        try:
            st = os.stat(fs_path)
            major, minor = os.major(st.st_dev), os.minor(st.st_dev)
            cand = f"/sys/dev/block/{major}:{minor}/stat"
            if os.path.exists(cand):
                self._stat_path = cand
                LOGGER.info(
                    "disk_io: monitoring %s for %s (dev %d:%d)",
                    cand, fs_path, major, minor,
                )
            else:
                self._stat_path = None
                LOGGER.info(
                    "disk_io: no /sys stat for %s (dev %d:%d); "
                    "I/O-aware pull throttle disabled on this node",
                    fs_path, major, minor,
                )
        except Exception:
            self._stat_path = None
            LOGGER.debug(
                "disk_io: device resolution failed for %s", fs_path,
                exc_info=True,
            )
        return self._stat_path


__all__ = ["DiskIoSampler"]
