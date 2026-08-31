"""Hardware probe (spec 10).

Phase-0 surface intentionally minimal: vcpus, mem, disk, kvm, gpu, kernel
version. The capacity estimator consumes ``HardwareInfo`` together with each
template's resource profile to compute per-(node, template) max concurrency.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class HardwareInfo(BaseModel):
    """Snapshot of one node's hardware (spec 10 inputs)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vcpus: int = Field(ge=1)
    mem_bytes: int = Field(ge=0)
    disk_bytes: int = Field(ge=0)
    has_kvm: bool
    has_gpu: bool
    gpu_model: str | None
    kernel_version: str
    platform: str  # "linux" | "darwin" | "windows" | ...


def probe_hardware(disk_root: str = "/") -> HardwareInfo:
    """Snapshot the host's hardware profile.

    Designed to be called at node-agent startup and on a slow re-probe loop;
    cheap (a few syscalls + maybe a sysctl). Cross-platform — Linux uses
    ``/proc/meminfo``; Darwin uses ``sysctl hw.memsize``.

    ``disk_root`` MUST name a path on the volume sandboxes actually write
    into — the container runtime's data-root — because ``disk_bytes`` is
    what :class:`~xrlenv.control.capacity.StaticCapacityEstimator` sizes the
    sandbox-writable pool against. The ``"/"`` default is a last resort for
    single-volume hosts; on a node whose data-root sits on a separate volume
    (e.g. an EBS/NVMe mount at ``/opt/sagemaker``) it measures the wrong
    filesystem and the estimator caps the node far below its real capacity.
    :py:meth:`xrlenv.node.agent.NodeAgent.hardware` passes the backend's
    resolved data-root; see ``DockerBackend.disk_monitor_path``.
    """
    return HardwareInfo(
        vcpus=os.cpu_count() or 1,
        mem_bytes=_probe_mem_bytes(),
        disk_bytes=shutil.disk_usage(disk_root).total,
        has_kvm=Path("/dev/kvm").exists(),
        has_gpu=False,
        gpu_model=None,
        kernel_version=platform.release(),
        platform=platform.system().lower(),
    )


def _probe_mem_bytes() -> int:
    sys = platform.system().lower()
    if sys == "linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    # Format: "MemTotal:       16308868 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024
        except OSError:
            pass
    elif sys == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return int(out)
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    # Last resort — return a conservative 4 GiB so the capacity estimator does
    # not divide by zero; operators can override the probe if it matters.
    return 4 * 1024 * 1024 * 1024
