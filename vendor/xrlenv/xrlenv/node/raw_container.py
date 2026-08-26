"""P1.7.A.1 — Node-side raw container manager.

Case 2/3 evaluation harnesses (swebench, harbor, OSWorld) issue
raw ``docker exec`` against containers they own. They don't know
about xrlenv's in-sandbox stub layer. This module provides the
node-side handler that talks directly to the docker daemon,
bypassing the stub entirely.

Maintains an in-memory ``container_id → rollout_id`` ownership
map. ``acquire_container`` registers; ``exec`` / ``destroy``
reject calls whose ``rollout_id`` doesn't match the registered
owner. Same enforcement model as the stub-bound spec-21
commands, just without the stub.

Phase-1 is docker-only — this module assumes a
``docker.DockerClient`` instance. Phase-2's CubeSandbox backend
will need a parallel manager (or a Protocol abstraction lifted
to ``BackendAdapter``) when that lands.

NOT used by the case-1 (RL training) path — that path uses
``NodeAgent.create_sandbox`` which mounts the stub-runtime layer
and starts the in-sandbox stub HTTP server. Raw containers and
sandboxes coexist on the same node; their identities and
lifecycles are independent (consistent with spec-00 invariant 1:
sandbox identity ≠ rollout identity).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import math
import os
import random
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager, suppress
from typing import Any, Protocol

import docker.errors
import requests.exceptions
import yaml
from pydantic import BaseModel, ConfigDict

from xrlenv.api.constants import DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES
from xrlenv.backends.base import (
    CpuIsolation,
    ResourceSpec,
    RuntimeLimits,
    effective_cpu_isolation,
)
from xrlenv.backends.egress import (
    DockerNsenterEnforcer,
    EgressAllowlist,
    EgressEnforcer,
    compile_egress_rules,
    container_can_escape_egress,
    is_shared_netns,
)
from xrlenv.errors import ArchiveTooLarge, PinCapacityExhausted, XRLEnvError
from xrlenv.node.health import NodeHealthCollector, NodeHealthSnapshot
from xrlenv.node.raw_compose import (
    DEFAULT_UP_TIMEOUT_S,
    ComposeProjectRecord,
    ComposeProjectRunner,
)

if False:  # TYPE_CHECKING — avoid circular import
    from xrlenv.node.image_cache import ImageCacheManager  # noqa: F401

LOGGER = logging.getLogger(__name__)

# P1 (cluster-resource-isolation-plan) — node-side fallback cgroup
# limits. The control plane always stamps AcquireContainerCommand.
# resources with the effective ResourceSpec, so these are a defensive
# guard: a missing / all-zero / non-positive resources field must
# resolve to a real cap, never to an unbounded container. Values mirror
# the control plane's _DEFAULT_RAW_RESOURCES *limit* fields.
_DEFAULT_RAW_CGROUP_CPU_LIMIT = 2.0
_DEFAULT_RAW_CGROUP_MEM_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
# Docker's CFS scheduler period (100 ms) — the standard base for
# expressing a fractional CPU cap as cpu_quota / cpu_period.
_CFS_PERIOD_US = 100_000

# docker-py's demultiplexer raises ``ValueError("N is not a valid stream")`` when an
# exec attach stream misaligns (a short read on the daemon socket under concurrent exec —
# the frame header lands mid-payload so a data byte is read as the stream-type). It is
# transient: a fresh exec opens a new stream. Left unhandled it is FATAL to the caller —
# a single occurrence killed EvoClaw's tag-watcher thread and hung a whole rollout for
# hours. The batched ``exec`` resync-retries a corrupted read this many times.
_EXEC_DEMUX_RETRIES = 2


def _effective_cpu_limit(resources: ResourceSpec | None) -> float:
    """P1/P2 — the CPU limit (cores) to enforce: the effective
    ResourceSpec's ``cpu_limit``, or the node default when missing /
    non-positive. Shared by the cgroup-quota and cpuset-pinning paths
    so both agree on the cap."""
    if resources is not None and resources.cpu_limit > 0:
        return resources.cpu_limit
    return _DEFAULT_RAW_CGROUP_CPU_LIMIT


def _effective_cgroup_run_kwargs(
    resources: ResourceSpec | None,
) -> dict[str, Any]:
    """P1 — translate the effective ``ResourceSpec`` into docker-py
    ``containers.run`` cgroup kwargs (CPU + memory).

    Regression guard (cluster-resource-isolation-plan P1): a missing or
    non-positive limit falls back to the node default so a raw container
    is **never** spawned with no cgroup limits. "No spec" means "default
    spec", not "no limits".
    """
    cpu_limit = _effective_cpu_limit(resources)
    mem_limit = (
        resources.mem_limit_bytes
        if resources is not None and resources.mem_limit_bytes > 0
        else _DEFAULT_RAW_CGROUP_MEM_LIMIT_BYTES
    )
    return {
        "cpu_period": _CFS_PERIOD_US,
        "cpu_quota": int(cpu_limit * _CFS_PERIOD_US),
        "mem_limit": int(mem_limit),
    }


def _runtime_limits_run_kwargs(
    runtime_limits: RuntimeLimits | None,
) -> dict[str, Any]:
    """P0b — translate the harness's ``RuntimeLimits`` into docker-py
    ``containers.run`` kwargs (pids / shm / tmpfs / read-only rootfs).

    Only the fields the harness explicitly set are applied; cluster
    mode injects no pids/shm defaults of its own, so an unspecified
    limit behaves exactly as it would against a local Docker daemon.
    """
    if runtime_limits is None:
        return {}
    kw: dict[str, Any] = {}
    if runtime_limits.pids_limit is not None:
        kw["pids_limit"] = runtime_limits.pids_limit
    if runtime_limits.shm_size_bytes is not None:
        kw["shm_size"] = runtime_limits.shm_size_bytes
    if runtime_limits.tmpfs:
        kw["tmpfs"] = dict(runtime_limits.tmpfs)
    if runtime_limits.readonly_rootfs:
        kw["read_only"] = True
    return kw


# ── P2 — cpuset pinning (cluster-resource-isolation-plan) ───────────────────
#
# CFS quota (cpu_quota) caps *average* CPU time but not wall-clock
# latency — the scheduler can still interleave a container off-core
# mid-test. For deterministic grading of timing-sensitive suites each
# raw container gets a disjoint set of physical cores. The node owns
# this allocation; harness cpuset_* kwargs are Level-3 rejected.


def _cpuset_str(cores: Iterable[int]) -> str:
    """Render core indices as a docker ``cpuset_cpus`` string."""
    return ",".join(str(c) for c in sorted(cores))


def _parse_cpuset(spec: str) -> tuple[int, ...]:
    """Parse a docker ``cpuset_cpus`` string (``"0,2-4"``) back into
    core indices. Used for crash recovery and orphan reclamation."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            with suppress(ValueError):
                out.update(range(int(lo), int(hi) + 1))
        else:
            with suppress(ValueError):
                out.add(int(part))
    return tuple(sorted(out))


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-2b — cgroup-v2 cpuset-inheritance self-test (isolation_capable)
#
# See notes/cluster-resource-isolation-plan.md §8.10. A node advertises
# ``isolation_capable=true`` only if a real probe proves this node's docker +
# cgroup driver honor ``cgroup_parent`` cpuset propagation to an unpinned child
# container. Gated on the ``cgroupfs`` docker cgroup driver (v1 decision); a
# ``systemd``-driver node, a cgroup-v1 node, or ANY probe failure ⇒ ``false``
# (fall back to today's per-container pinning / CFS quota). Fail-safe: a node
# never claims isolation it hasn't proven.
# ──────────────────────────────────────────────────────────────────────────────

_CGROUP_ROOT = "/sys/fs/cgroup"
# Bounded poll to absorb the kernel's cpuset-propagation latency after the
# parent shrink (effective-cpuset updates are near-instant but not synchronous).
_SELFTEST_PROPAGATION_POLL_S = 2.0
_SELFTEST_POLL_INTERVAL_S = 0.05


def _cgroup_v2_cpuset_available(cgroup_root: str = _CGROUP_ROOT) -> bool:
    """True iff ``cgroup_root`` is a cgroup-v2 unified hierarchy exposing the
    ``cpuset`` controller. cgroup v2 has a root ``cgroup.controllers`` file
    (absent on v1); the controller must be listed to be delegable to children."""
    try:
        with open(os.path.join(cgroup_root, "cgroup.controllers")) as f:
            return "cpuset" in f.read().split()
    except OSError:
        return False  # cgroup v1 (no such file) or unreadable


def _docker_cgroup_driver(docker_client: Any) -> str:
    """The docker daemon's cgroup driver (``docker info`` ``CgroupDriver``),
    e.g. ``"cgroupfs"`` or ``"systemd"``; ``""`` when it can't be determined."""
    try:
        return str(docker_client.info().get("CgroupDriver") or "")
    except Exception:
        return ""


def _delegated_shared_parent_writable(
    cgroup_root: str = _CGROUP_ROOT, name: str = "xrlenv-shared",
) -> bool:
    """P6 §8.13 — True iff the shared-parent cgroup ``xrlenv-shared`` exists and
    its ``cpuset.cpus`` is WRITABLE by the current process.

    The node agent runs as a non-root user (spec 19) and cannot ``mkdir`` /
    write under ``/sys/fs/cgroup`` (a DAC check ``CAP_SYS_ADMIN`` does not
    override). So the root enable step (``deploy/node/enable_cpu_isolation.sh``)
    creates ``xrlenv-shared`` and ``chown``\\s it to the agent user
    (delegation). This is how the non-root agent detects P6 capability +
    manages the complement WITHOUT any privileged cgroup op of its own:
    ``os.access`` here is ``False`` when the file is absent (node never
    enabled) or not writable (delegation torn down)."""
    cpuset = os.path.join(cgroup_root, name, "cpuset.cpus")
    return os.access(cpuset, os.W_OK)


def _read_cpus_allowed_list(pid: int) -> set[int]:
    """The set of logical CPUs a process is *effectively* allowed to run on,
    from ``/proc/<pid>/status`` ``Cpus_allowed_list`` (the field §8.5 checks).
    Empty set when the file / field can't be read (treated as a probe failure
    by the caller)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    return set(_parse_cpuset(line.split(":", 1)[1].strip()))
    except OSError:
        pass
    return set()


def _run_cgroup_isolation_selftest(
    docker_client: Any,
    *,
    image: str,
    cgroup_root: str = _CGROUP_ROOT,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """The real §8.10 self-test (cgroupfs driver). Returns whether a shrink of a
    shared parent cpuset propagated to an unpinned child container's effective
    CPUs. Any failure / unmet gate ⇒ ``False`` (never raises — the caller's
    fail-safe also traps, but keeping it total here keeps the seam clean).

    **Runtime scope (important).** The probe container runs under the docker
    daemon's DEFAULT runtime (``runc``) — no ``runtime=`` override. A ``true``
    verdict therefore proves only that *runc* honors ``cgroup_parent`` cpuset
    propagation on this node; it does NOT prove the same for **sysbox** (which
    manages cgroups differently and carries much of this fleet's DinD workload).
    Step 3 must not place sysbox-runtime containers under the shared parent on
    the strength of this probe alone — either extend the probe per-runtime or
    scope the shared-parent scheme to runc containers (§8.6 / §8.10, tracked as
    a step-3 gate).

    Requires root (writes under ``/sys/fs/cgroup``) + docker; only ever run on a
    real node. Because the node agent is NON-ROOT (spec 19), this probe no
    longer runs in the agent — it runs at ENABLE time
    (``deploy/node/enable_cpu_isolation.sh``, as root) as the gate BEFORE that
    script delegates ``xrlenv-shared`` to the agent user; the agent then
    verifies the delegation instead (§8.13,
    :meth:`RawContainerManager._default_isolation_selftest`). Unit tests drive
    ``isolation_capable`` through the injected seam; a root+cgroup2+cgroupfs
    integration test exercises this probe end to end.
    """
    # ── Gates (cheap, no side effects) ────────────────────────────────────────
    if not _cgroup_v2_cpuset_available(cgroup_root):
        return False
    if _docker_cgroup_driver(docker_client) != "cgroupfs":
        return False  # v1 decision: systemd-driver nodes stay non-capable
    if not image:
        return False  # no probe image configured (XRLENV_SELFTEST_IMAGE unset)
    try:
        docker_client.images.get(image)  # present? — NO pull at node init
    except Exception:
        return False
    online = sorted(
        os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity")
        else set(range(os.cpu_count() or 1)),
    )
    if len(online) < 2:
        return False  # need two CPUs so the shrink drops one and can be observed
    cpu_keep, cpu_drop = online[0], online[1]

    # Unique per call (pid + random) so two concurrent first-callers of
    # ``isolation_capable()`` can't collide on the same throwaway cgroup: a bare
    # pid-only name would let the second ``mkdir`` fail EEXIST and cache a
    # spurious ``False`` on a genuinely capable node. Each probe now gets its own
    # cgroup + container and computes an independent (identical) verdict.
    parent_name = f"xrlenv-selftest-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    parent_dir = os.path.join(cgroup_root, parent_name)
    container: Any = None
    try:
        os.mkdir(parent_dir)
        with open(os.path.join(parent_dir, "cgroup.subtree_control"), "w") as f:
            f.write("+cpuset")  # delegate cpuset to children (the container)
        _write_cpuset(parent_dir, f"{cpu_keep},{cpu_drop}")
        # No command override: the probe image must be long-lived by default
        # (a pause / infra image — the configured XRLENV_SELFTEST_IMAGE). We
        # can't assume a shell/`sleep` exists in an arbitrary image, and a
        # pause image's default entrypoint already keeps PID 1 alive. If the
        # configured image's default exits immediately, the PID read below
        # fails and the probe returns False (fail-safe).
        container = docker_client.containers.run(
            image, detach=True, remove=False,
            cgroup_parent=f"/{parent_name}", network_disabled=True,
            labels={"xrlenv.selftest": "1"},
        )
        container.reload()
        pid = int(container.attrs["State"]["Pid"])
        if pid <= 0:
            return False
        # Baseline: the child must see both CPUs before the shrink, else the
        # cgroup_parent placement itself didn't take (a docker-layer failure).
        if not _poll_cpus_allowed(
            pid, {cpu_keep, cpu_drop}, sleep=sleep, monotonic=monotonic,
        ):
            return False
        # Shrink the PARENT and require the child's effective CPUs to follow.
        _write_cpuset(parent_dir, str(cpu_keep))
        return _poll_cpus_allowed(
            pid, {cpu_keep}, sleep=sleep, monotonic=monotonic,
        )
    except Exception:
        LOGGER.warning(
            "isolation self-test errored; advertising isolation_capable=false",
            exc_info=True,
        )
        return False
    finally:
        if container is not None:
            with suppress(Exception):
                container.remove(force=True)
        with suppress(OSError):
            os.rmdir(parent_dir)


def _write_cpuset(cgroup_dir: str, value: str) -> None:
    with open(os.path.join(cgroup_dir, "cpuset.cpus"), "w") as f:
        f.write(value)


def _poll_cpus_allowed(
    pid: int,
    expected: set[int],
    *,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> bool:
    """Poll ``/proc/<pid>/status`` until the process's effective CPUs equal
    ``expected`` (cpuset propagation is near-instant but not synchronous), up to
    :data:`_SELFTEST_PROPAGATION_POLL_S`."""
    deadline = monotonic() + _SELFTEST_PROPAGATION_POLL_S
    while True:
        if _read_cpus_allowed_list(pid) == expected:
            return True
        if monotonic() >= deadline:
            return False
        sleep(_SELFTEST_POLL_INTERVAL_S)


# ──────────────────────────────────────────────────────────────────────────────
# P6 step-3 (§8.11) — shared-parent cpuset: confine unpinned runc containers to
# the complement of the pinned cores. Landed in 3a as the behavior-neutral
# foundation (the classes + ledger wiring) — gated OFF until the manager passes
# a shared parent (3b). With ``shared_parent=None`` + ``min_shared_cores=0``
# (the 3a defaults) the ledger behaves EXACTLY as before this step.
# ──────────────────────────────────────────────────────────────────────────────


# §8.4 — the shared-pool floor default: keep at least this fraction of the
# node's logical CPUs pinnable-but-shared (a cap on how much may be pinned, NOT
# an idle carve-out). Overridable per node (manager ``min_shared_cores``).
_DEFAULT_SHARED_FLOOR_FRACTION = 0.25


class _CgroupWriter(Protocol):
    """The injectable seam for the shared-parent cgroup writes (§8.11 testability).

    :class:`_RealCgroupWriter` does the real ``/sys/fs/cgroup`` writes (needs root
    + cgroup v2 + the cgroupfs docker driver — i.e. a capable node); tests inject
    a fake to assert the exact ``cpuset.cpus`` strings written on ensure / pin /
    unpin / reconcile without root."""

    def ensure_group(self, path: str) -> None: ...
    def write_cpuset_cpus(self, path: str, value: str) -> None: ...
    def remove_group(self, path: str) -> None: ...


class _RealCgroupWriter:
    """Production :class:`_CgroupWriter` — real ``/sys/fs/cgroup`` writes.

    On a capable node the shared parent ``xrlenv-shared`` is pre-created and
    ``chown``\\ed to the (non-root) agent user by the root enable step
    (``deploy/node/enable_cpu_isolation.sh``, §8.13). So every write below operates
    on files this agent now OWNS — ``ensure_group``'s ``mkdir`` hits
    ``FileExistsError`` (the dir already exists) and the ``cpuset.cpus`` /
    ``cgroup.subtree_control`` writes succeed via delegation, without the agent
    needing root."""

    def ensure_group(self, path: str) -> None:
        """Ensure the cgroup dir exists (created by the enable step under
        delegation; ``mkdir`` here is a no-op that traps ``FileExistsError``)
        and (re-)delegate the cpuset controller to its children so a container
        placed under it inherits the cpuset. Idempotent."""
        with suppress(FileExistsError):
            os.mkdir(path)
        with open(os.path.join(path, "cgroup.subtree_control"), "w") as f:
            f.write("+cpuset")

    def write_cpuset_cpus(self, path: str, value: str) -> None:
        with open(os.path.join(path, "cpuset.cpus"), "w") as f:
            f.write(value)

    def remove_group(self, path: str) -> None:
        with suppress(OSError):
            os.rmdir(path)


class _SharedCpusetParent:
    """P6 step-3 (§8.11) — the node-level shared parent cgroup (``xrlenv-shared``).

    Every *unpinned runc* container on a capable node is created under this
    parent (``cgroup_parent``) with no own ``cpuset_cpus``; its effective cpuset
    is the parent's, so ONE write to the parent's ``cpuset.cpus`` reconfines all
    unpinned children at once (no per-container ``docker update``). The parent's
    cpuset is kept at ``all - in_use`` (the dynamic complement) by the
    :class:`_CoreLedger` on each pin/unpin.

    §8.13: the parent cgroup itself is pre-created and delegated (``chown``\\ed)
    to the non-root agent user by the root enable step, so the writes here need
    no root — see :class:`_RealCgroupWriter`. :meth:`ensure` therefore only
    re-affirms + seeds the cpuset; it does not depend on being able to ``mkdir``
    at the cgroup root (which the non-root agent cannot).
    """

    def __init__(
        self,
        *,
        total_cores: int,
        cgroup_root: str = _CGROUP_ROOT,
        name: str = "xrlenv-shared",
        writer: _CgroupWriter | None = None,
    ) -> None:
        self._all: frozenset[int] = frozenset(range(max(1, total_cores)))
        self._name = name
        self._path = os.path.join(cgroup_root, name)
        self._writer: _CgroupWriter = (
            writer if writer is not None else _RealCgroupWriter()
        )
        self._ensured = False

    @property
    def cgroup_parent(self) -> str:
        """The ``cgroup_parent`` kwarg value for unpinned runc containers."""
        return f"/{self._name}"

    def ensure(self) -> None:
        """Create the parent (idempotent) with ``cpuset.cpus = all`` — the
        no-pinning state where the shared pool is every logical CPU (§8.4)."""
        if self._ensured:
            return
        self._writer.ensure_group(self._path)
        self._writer.write_cpuset_cpus(self._path, _cpuset_str(sorted(self._all)))
        self._ensured = True

    def set_complement(self, in_use: Iterable[int]) -> None:
        """Write ``cpuset.cpus = all - in_use`` (the dynamic complement).

        The floor (§8.4) keeps the complement non-empty on the ``allocate`` path.
        But ``mark_in_use`` / ``enable_shared_parent`` drive ``in_use`` from
        pre-existing pinned containers with no floor check, so an over-pinned
        node (every logical CPU already pinned by legacy containers at wiring)
        could make the complement empty. We MUST NOT write ``""`` there: on
        cgroup v2 an empty ``cpuset.cpus`` means "inherit the parent" — which
        would hand every unpinned child ALL cores (the exact trampling P6
        prevents). Enforce the non-empty invariant by confining to a single core
        (bounded 1-core contention beats whole-node trampling) + warn."""
        self.ensure()
        complement = sorted(self._all - set(in_use))
        if not complement:
            fallback = max(self._all)
            LOGGER.warning(
                "shared-parent cpuset: every logical CPU is pinned (over-pinned "
                "node — not reachable via the floor). Confining unpinned tasks "
                "to a single core %d rather than writing an empty set (which "
                "would hand them all cores). Isolation is degraded until pins "
                "release.", fallback,
            )
            complement = [fallback]
        self._writer.write_cpuset_cpus(self._path, _cpuset_str(complement))


def _inject_shared_cgroup_parent(compose_yaml: str, cgroup_parent: str) -> str:
    """P6 step-3c (§8.11) — inject ``cgroup_parent`` into every RUNC service of a
    (vetted, image-only) compose document so its sidecars join the shared pool
    (the complement of the pinned cores) instead of running full-host — a sidecar
    with full-host affinity would reopen the trampling bug.

    Runtime scope (as elsewhere in step 3): a **sysbox**-runtime service
    (``runtime`` set to anything but ``runc``) is left untouched — the self-test
    only proved runc. The control plane already rejects an operator-supplied
    ``cgroup_parent`` (KwargsPolicy Level 3), so there's never one to overwrite.

    Best-effort + total: on any YAML parse/shape surprise it returns the original
    document unchanged — a transform bug must never block a compose ``up``."""
    try:
        doc = yaml.safe_load(compose_yaml)
    except Exception:
        return compose_yaml
    if not isinstance(doc, dict):
        return compose_yaml
    services = doc.get("services")
    if not isinstance(services, dict):
        return compose_yaml
    changed = False
    for svc in services.values():
        if not isinstance(svc, dict):
            continue
        runtime = svc.get("runtime")
        if runtime is not None and runtime != "runc":
            continue  # sysbox etc. — left on today's path (runtime scope)
        svc["cgroup_parent"] = cgroup_parent
        changed = True
    if not changed:
        return compose_yaml
    return yaml.safe_dump(doc, sort_keys=False)


class _CoreLedger:
    """Tracks which host CPU cores are pinned to raw containers (P2).

    The node's source of truth for free cores. Rebuilt from live
    containers on first use (:meth:`mark_in_use`) so a node-process
    restart between acquire and destroy can't leak cores.

    P6 step-3 (§8.11): on a capable node the manager wires a
    :class:`_SharedCpusetParent` + a ``min_shared_cores`` floor. Then every
    allocate/release/mark_in_use keeps the shared parent's cpuset at
    ``all - in_use`` (under this lock, so the complement is serialised against
    concurrent pins), and ``allocate`` refuses once granting would drop the
    shared pool below the floor. With the defaults (``shared_parent=None``,
    ``min_shared_cores=0``) both additions are inert — identical to pre-P6-step-3
    behavior — so 3a lands behavior-neutral.
    """

    def __init__(
        self,
        total_cores: int,
        *,
        shared_parent: _SharedCpusetParent | None = None,
        min_shared_cores: int = 0,
    ) -> None:
        self._total = max(1, total_cores)
        self._free: set[int] = set(range(self._total))
        self._lock = threading.Lock()
        self._shared_parent = shared_parent
        # Never let the floor exceed the node's core count (would refuse all
        # pinning); clamp to [0, total].
        self._min_shared_cores = max(0, min(min_shared_cores, self._total))

    @property
    def total(self) -> int:
        return self._total

    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    def _sync_complement_locked(self) -> None:
        """Best-effort: push ``all - in_use`` to the shared parent. Caller holds
        the lock. A write failure degrades isolation (unpinned tasks may reach a
        pinned core) but never breaks the allocate/release bookkeeping, so it's
        logged, not raised — the pinned container still holds its cores via its
        own ``cpuset_cpus``."""
        if self._shared_parent is None:
            return
        in_use = set(range(self._total)) - self._free
        try:
            self._shared_parent.set_complement(in_use)
        except Exception:
            LOGGER.warning(
                "shared-parent complement write failed (isolation degraded "
                "until the next successful update)", exc_info=True,
            )

    def allocate(self, n: int) -> tuple[int, ...] | None:
        """Reserve ``n`` cores (clamped to ``[1, total]``). Returns the core
        indices, or ``None`` when granting would leave fewer than
        ``min_shared_cores`` in the shared pool (with the default floor 0 this
        reduces to "fewer than ``n`` free")."""
        n = max(1, min(n, self._total))
        with self._lock:
            # Floor (§8.4): free-after-grant must stay ≥ the shared-pool floor.
            if len(self._free) - n < self._min_shared_cores:
                return None
            picked = sorted(self._free)[:n]
            self._free.difference_update(picked)
            self._sync_complement_locked()
            return tuple(picked)

    def release(self, cores: Iterable[int]) -> None:
        with self._lock:
            self._free.update(
                c for c in cores if 0 <= c < self._total
            )
            self._sync_complement_locked()

    def mark_in_use(self, cores: Iterable[int]) -> None:
        """Crash recovery — mark cores already held by a pre-existing
        container as allocated so a restart doesn't double-assign."""
        with self._lock:
            self._free.difference_update(
                c for c in cores if 0 <= c < self._total
            )
            self._sync_complement_locked()

    def enable_shared_parent(
        self, parent: _SharedCpusetParent, min_shared_cores: int,
    ) -> None:
        """P6 step-3b — attach the shared parent + floor (a capable node opts in
        here, after reconcile). Idempotent; writes the initial complement
        (``all - current in_use``) so the shared pool matches the ledger the
        instant it's wired. Under the lock, consistent with concurrent pins."""
        with self._lock:
            self._shared_parent = parent
            self._min_shared_cores = max(0, min(min_shared_cores, self._total))
            self._sync_complement_locked()

    @property
    def min_shared_cores(self) -> int:
        return self._min_shared_cores


# Node-saturation create-retry (see notes/design-node-saturation-recovery.md).
# A transient create failure that xrlenv already recognizes as node saturation
# (5xx / timeout — ``_is_node_health_error``; e.g. sysbox-fs ``pre-register …
# DeadlineExceeded`` under a burst of concurrent sysbox creates) is retried a
# bounded number of times with exponential backoff instead of failing the
# acquire — the condition clears in seconds. Only a *transient busy-daemon*
# fault is retried (``_is_retryable_create_error`` — 5xx / timeout); a clean
# dead-daemon ConnectionError, 4xx request faults (404/400), and 409
# name-conflicts (handled in ``_run_with_name_reclaim``) are NOT retried here.
#
# The total retry wait is bounded FOUR ways: the attempt count
# (``_HEALTH_RETRY_MAX``), the per-retry ceiling (``_HEALTH_RETRY_CAP_S``), a hard
# wall-clock cap (``_HEALTH_RETRY_TOTAL_CAP_S``), AND — audit P2 — the caller's
# remaining acquire wire budget (``acquire_timeout_s`` reaches the node as the
# whole AcquireContainer deadline, so a fail-fast caller fails fast on the node
# too, never retrying past the control plane's ``_send_and_wait`` timeout).
# Worst-case ≈ 31 s (1+2+4+8+16, jittered down), well under the 600 s default.
_HEALTH_RETRY_MAX = 5
_HEALTH_RETRY_BASE_S = 1.0
_HEALTH_RETRY_CAP_S = 30.0
_HEALTH_RETRY_TOTAL_CAP_S = 45.0  # hard wall-clock ceiling across all retries


def _is_timeout_exc(exc: BaseException) -> bool:
    """True if ``exc`` is a docker / ``requests`` timeout-class error."""
    return isinstance(exc, requests.exceptions.Timeout) or (
        "timed out" in str(exc).lower()
    )


def _docker_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status of a docker-py ``APIError`` (or ``None``).

    ``APIError`` carries ``.status_code``; some wrapped errors only have
    it on ``.response``. Returns ``None`` for transport errors that never
    reached an HTTP response (timeouts, connection refused)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    return status if isinstance(status, int) else None


# docker-py ships no stubs, so DockerException is Any and mypy rejects
# subclassing it (hence the inline ignore). Subclassing is still the right shape:
# it makes the fault flow through the create path's existing
# ``except (DockerException, RequestException)`` handlers unchanged.
class ContainerNotStartedError(docker.errors.DockerException):  # type: ignore[misc]
    """``containers.run`` returned 2xx but the container is not actually running.

    Docker's create+start can report success and still hand back an unusable
    container when the daemon's storage layer is under strain. Observed on a
    cold-cache node pulling 25 images at once into the containerd snapshotter
    (``driver=overlayfs``): the container was created, ``run`` returned normally,
    and 187 ms later every operation on it failed with
    ``500 … RWLayer of container <id> is unexpectedly nil`` — dockerd never wired
    up the read-write layer. Because ``run`` did not raise, nothing downstream
    retried; the acquire returned a corpse and the consumer died on its first
    ``exec`` with a ``409 … is not running``.

    Raising this from the create path (instead of returning the container) folds
    the case into the existing transient-create retry: the broken container is
    reaped by ``_reap_rollout_orphans`` on the next attempt, and a recovered
    retry returns a healthy container. Treated as BOTH retryable and a node
    health signal, since it is a symptom of daemon saturation.
    """


_NEVER_STARTED_STATUSES = frozenset({"created", "dead"})


def _container_start_fault(container: Any) -> str | None:
    """Describe why ``container`` is unusable right after start, else ``None``.

    Called once per create, so it must not mistake a legitimately short-lived
    container for a fault: a container that ran and exited cleanly returns
    ``None`` (``acquire`` has never required a long-lived process, and callers
    that run a one-shot command rely on that). Three shapes are faults — inspect
    itself raising right after a successful create, dockerd saying the container
    never left ``created``/``dead``, or a recorded start ``Error``. All three
    mean start did not actually take.

    FAIL-OPEN by design: every path that cannot *positively* establish a fault
    returns ``None``. A false positive here would retry — and after
    ``_HEALTH_RETRY_MAX`` attempts fail — a perfectly good acquire, which is
    strictly worse than missing a rare daemon fault that the consumer would
    surface anyway. So an un-inspectable container (a client object without
    ``reload``, an inspect payload with no ``State``) is treated as healthy.
    """
    reload = getattr(container, "reload", None)
    if not callable(reload):
        # Nothing to inspect with — cannot establish a fault, so don't invent one.
        return None
    try:
        reload()
    except (
        docker.errors.DockerException,
        requests.exceptions.RequestException,
    ) as exc:
        # Inspect failing immediately after a successful create is itself the
        # symptom (the RWLayer-nil case 500s on inspect-adjacent calls too).
        return f"inspect failed right after create: {exc}"
    state = (getattr(container, "attrs", None) or {}).get("State") or {}
    if state.get("Running"):
        return None
    status = str(state.get("Status") or "unknown")
    error = str(state.get("Error") or "")
    if status in _NEVER_STARTED_STATUSES or error:
        return (
            f"container not running after create "
            f"(status={status!r} exit_code={state.get('ExitCode')!r} "
            f"docker_error={error!r})"
        )
    # Ran and exited on its own — not a fault. Preserves the one-shot-command
    # contract; only a container that never started is retried.
    return None


def _is_node_health_error(exc: BaseException) -> bool:
    """True if ``exc`` is a docker-daemon SATURATION/health signal that
    should drive the AIMD admission limiter down — a timeout, a transport
    failure, or a 5xx from dockerd.

    A 4xx client error (409 name-in-use, 404 image-not-found, 400 bad
    request) is a *request* fault, not evidence the node is overloaded.
    Counting those as health let one duplicate container name (or a
    missing image) multiplicatively collapse the whole node's admission
    limit — the prod failure this guards against. When the fault never
    produced an HTTP status (a bare transport error that isn't a clean
    timeout/connection error) we default to treating it as health:
    over-throttling on an unknown transport fault is safer than ignoring
    a real daemon problem."""
    if isinstance(exc, ContainerNotStartedError):
        # A create that "succeeded" into an unusable container is a saturation
        # symptom, not a request fault — throttle future admits.
        return True
    if _is_timeout_exc(exc):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    status = _docker_status_code(exc)
    if status is not None:
        return status >= 500
    return True


def _is_retryable_create_error(exc: BaseException) -> bool:
    """True if ``exc`` is a *transient busy-daemon* create fault worth retrying
    in place — a timeout or a 5xx from dockerd (e.g. sysbox-fs ``pre-register …
    DeadlineExceeded`` under a create burst; the daemon is momentarily
    overloaded, not down, so a retry a second or two later succeeds).

    Deliberately NARROWER than :func:`_is_node_health_error`: a clean transport
    ``ConnectionError`` (connection refused / reset — dockerd down or restarting)
    is NOT retried. A 31 s backoff can't revive a down daemon, and failing fast
    lets the control plane mark the node unhealthy sooner. A down daemon still
    feeds the AIMD limiter (via ``_is_node_health_error``) so future admits
    throttle — it just isn't hammered in place. 4xx request faults (409 handled
    by name-reclaim, 404 missing image) are terminal and never retried."""
    if isinstance(exc, ContainerNotStartedError):
        # Retryable for the same reason a 5xx is: the daemon's storage layer was
        # momentarily wedged, and the next attempt (after the orphan reap)
        # usually lands a healthy container.
        return True
    if _is_timeout_exc(exc):
        return True
    status = _docker_status_code(exc)
    if status is not None:
        return status >= 500
    # No HTTP status and not a timeout → a bare transport error (e.g. a clean
    # ConnectionError from a down daemon). Don't retry in place; the AIMD signal
    # is enough. Conservative on purpose — an unknown transport fault we can't
    # attribute to daemon busyness is safer failed-fast than retried for 31 s.
    return False


def _is_name_conflict(exc: BaseException) -> bool:
    """True if ``exc`` is docker's 409 'container name already in use'."""
    return _docker_status_code(exc) == 409 and "already in use" in str(exc).lower()


def _translate_docker_error(
    *, operation: str, target: str, client: Any, exc: BaseException,
) -> XRLEnvError:
    """Issue #18 (Ask #3) — turn a raw docker-py / ``requests``
    exception into an :class:`XRLEnvError` carrying node-side
    diagnostic context.

    Without this the node-agent dispatcher packs the bare exception
    onto the wire and the consumer sees, opaquely,
    ``ReadTimeout: UnixHTTPConnectionPool(... read timeout=60)`` —
    docker-py internals with no hint about which timeout layer fired
    or what to do. The translated message names the operation, the
    target, and (for timeout-class errors) the node's configured
    docker HTTP-client timeout, so a stale-node-binary diagnosis is
    one read instead of archaeology.
    """
    http_timeout: float | None = None
    try:
        http_timeout = float(client.api.timeout)
    except Exception:
        http_timeout = None

    msg = (
        f"node-side docker {operation} failed for {target}: "
        f"{type(exc).__name__}: {exc}"
    )
    is_timeout = _is_timeout_exc(exc)
    if is_timeout and http_timeout is not None:
        msg += (
            f". This is the node-side docker HTTP client timeout "
            f"({http_timeout:.0f}s) — the node↔dockerd call, "
            f"distinct from the control↔node wire timeout. "
        )
        if http_timeout <= 60.0:
            msg += (
                "60s is docker-py's default: this node-agent predates "
                "the 600s pin (DOCKER_CLIENT_HTTP_TIMEOUT_S) — redeploy "
                "the node-agent (deploy/refresh.sh)."
            )
        else:
            msg += (
                "The docker daemon did not finish within that ceiling "
                "— most likely overloaded under concurrent load."
            )
    return XRLEnvError(msg)


@asynccontextmanager
async def _null_async_cm() -> AsyncIterator[None]:
    """No-op async context manager — used when the destroy semaphore
    is disabled (``destroy_concurrency=0``)."""
    yield


# Unique end-of-stream sentinel for the archive chunk pump. A real
# tar chunk is ``bytes``; ``None``/``b""`` could in principle appear
# in a stream, so we use an identity-comparable object instead.
_ARCHIVE_STREAM_END = object()


def _next_archive_chunk(it: Any) -> Any:
    """Advance a docker-py tar-stream iterator by one chunk.

    Runs inside ``asyncio.to_thread`` — this is the ONLY place the
    blocking docker-socket read happens, so the node event loop is
    free between chunks (the heartbeat + outbound pump keep running
    even during a multi-hundred-MB ``/testbed`` copy). Returns
    :data:`_ARCHIVE_STREAM_END` at end-of-stream."""
    return next(it, _ARCHIVE_STREAM_END)


class RawContainerRecord(BaseModel):
    """In-memory state per acquired raw container.

    ``rollout_id`` carries the ownership claim — it's checked on
    every subsequent exec / destroy. ``container_id`` is docker's
    long-form id (the handle); ``container_name`` is whatever
    docker assigned (or the operator's override). Stored together
    so the audit log + admin panel can show both.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rollout_id: str
    container_id: str
    container_name: str
    image: str
    created_at: _dt.datetime
    cpuset: tuple[int, ...] = ()
    """P2 — host CPU cores pinned to this container, reserved from the
    node core ledger at acquire and released at destroy. Empty when
    cpuset pinning is disabled or the ledger had no free cores."""

    container_runtime: str | None = None
    """OCI runtime the container was created under (e.g. ``sysbox-runc``);
    ``None``/``runc`` = the default runtime. Persisted so :meth:`destroy` can
    route a sysbox teardown through the tighter sysbox destroy gate — concurrent
    sysbox-fs FUSE *unmounts* wedge the daemon just like concurrent creates
    overwhelm its register step (2026-07-08: conc-32 sweep leaked 4 sysbox
    containers whose ``docker rm`` wedged on ``fusermount3`` in D-state)."""


class _ComposeProjectState(BaseModel):
    """Node-side state for one brought-up compose project (P1.7.C.2).

    Ties the docker-compose project to its owning rollout + the runner's
    :class:`ComposeProjectRecord` + the image refs ensured at ``up`` (released on
    ``down``). Keyed by ``project_name`` in ``RawContainerManager._compose_projects``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rollout_id: str
    record: ComposeProjectRecord
    images: tuple[str, ...] = ()


class RawContainerDiskUsage(BaseModel):
    """A running raw container's writable-layer footprint (WS2).

    ``size_rw_bytes`` is docker's ``SizeRw`` — the bytes the container
    has written into its overlay upper-dir on top of the read-only
    image layers. This is the figure that grows unbounded when a
    workload writes to its own rootfs and fills the node data-root.
    ``rollout_id`` (from the ``xrlenv.rollout_id`` label) lets the disk
    guard name the rollout it kills. Produced only on demand by
    :meth:`RawContainerManager.list_disk_usage` — populating SizeRw is
    a layer-graph walk (``docker ps -s``), never on a hot path.
    """

    container_id: str
    rollout_id: str
    image: str
    size_rw_bytes: int


class RawContainerManager:
    """Thread-safe (async) manager for raw containers on a node.

    Public surface mirrors the spec-21 command set:

    - :meth:`acquire` → spawn + register (returns ``RawContainerRecord``)
    - :meth:`exec` → docker exec (returns ``ExecResult``-shaped dict)
    - :meth:`destroy` → ``docker rm -f`` + deregister
    - :meth:`list_owned` → introspection helper for GC + admin

    All mutations go through ``self._lock`` so concurrent acquire/
    destroy on the same container don't race the ownership map.
    """

    def __init__(
        self,
        *,
        docker_client: Any,
        image_cache: Any | None = None,
        destroy_concurrency: int = 4,
        create_concurrency: int = 4,
        sysbox_create_concurrency: int = 1,
        sysbox_destroy_concurrency: int = 1,
        archive_concurrency: int = 4,
        max_get_archive_relay_bytes: int = DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES,
        total_cores: int | None = None,
        egress_enforcer: EgressEnforcer | None = None,
        compose_runner: ComposeProjectRunner | None = None,
        isolation_selftest: Callable[[], bool] | None = None,
        min_shared_cores: int | None = None,
        cgroup_writer: _CgroupWriter | None = None,
    ) -> None:
        # ``docker_client`` is duck-typed (a ``docker.DockerClient`` in
        # production, a fake in tests). Held by reference; we don't
        # own its lifetime.
        self._client = docker_client
        # §5.3/§5.5 — cached (Runtimes, DefaultRuntime) from ``docker
        # info``. The registered-runtime set only changes on a daemon
        # restart (which restarts this agent too), so caching is safe.
        # Populated lazily by ``registered_runtimes()``.
        self._runtime_info_cache: tuple[frozenset[str], str] | None = None
        # P1.7.B.2: optional ImageCacheManager. When present and
        # ``ensure_image_present=True`` (the new default), routes
        # missing-image acquires through ``image_cache.ensure_present``
        # which pulls (registry images), invokes the registered
        # builder (lazy_registrations refs from ``xrlenv build apply``),
        # or no-ops as appropriate. The legacy "strict, raise on
        # missing" path is preserved as the explicit opt-out.
        self._image_cache = image_cache
        self._records: dict[str, RawContainerRecord] = {}
        self._lock = asyncio.Lock()
        # Audit P3 — bounded log of rollouts the node autonomously reaped
        # (currently the disk-pressure guard) + WHY, keyed by rollout_id.
        # Surfaced to the control plane in ListRawContainersReply so the
        # raw-GC reconciler seals the row with the real cause instead of a
        # generic teardown message. Bounded (LRU) so it can't grow without
        # limit; a reason lingers long enough for the ~60s reconcile sweep
        # to observe the vanished container and pick it up.
        self._disk_reaped: OrderedDict[str, str] = OrderedDict()
        self._disk_reaped_max = 256
        # Issue #18 fix #4: cap concurrent ``docker rm -f`` calls per
        # node. Under heavy parallel load (SWE-bench Pro at
        # ``--num-workers=64`` on a 2-node cluster) the docker daemon
        # serialised 30+ simultaneous overlay-fs teardowns and
        # individual ``container.remove`` calls stretched to 35-90 s.
        # That blew the control-plane's 30 s destroy ceiling on its
        # own AND amplified disk-pressure (layers held alive while
        # teardown queued). Cap defaults to 4 — measured to keep
        # daemon p99 latency bounded without collapsing destroy
        # throughput. ``destroy_concurrency=0`` opts out (unbounded —
        # legacy behaviour, exposed for tests that exercise the
        # pre-cap path).
        self._destroy_concurrency = destroy_concurrency
        self._destroy_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(destroy_concurrency)
            if destroy_concurrency > 0
            else None
        )
        # Issue #18 — symmetric cap on concurrent ``docker run``
        # (container create). A burst of acquires under heavy load
        # (SWE-bench Pro at ``--num-workers=64`` on a 2-node cluster)
        # fired dozens of simultaneous ``containers.run`` calls at one
        # docker daemon already saturated extracting multi-GB image
        # layers; each create then stretched past the node's 600 s
        # docker HTTP-client ceiling and failed the acquire outright
        # (surfacing consumer-side as ``containers.run failed ...
        # ReadTimeout``). Capping creates bounds the daemon's
        # create-time pressure the way ``destroy_concurrency`` bounds
        # teardown. ``0`` disables the cap (unbounded — legacy / tests).
        self._create_concurrency = create_concurrency
        self._create_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(create_concurrency)
            if create_concurrency > 0
            else None
        )
        # A separate, tighter cap for sysbox (non-runc) creates. sysbox-runc's
        # pre-register with sysbox-fs is far slower than a plain runc create, so
        # the general cap (4) still lets concurrent sysbox creates overwhelm
        # sysbox-fs — surfacing as ``pre-register with sysbox-fs: DeadlineExceeded``
        # (a transient the retry loop then recovers, but cheaper to not trigger).
        # Serialising sysbox creates (default 1) stops it at the source.
        # ``0`` disables the sysbox-specific gate (falls back to the general one).
        self._sysbox_create_concurrency = sysbox_create_concurrency
        self._sysbox_create_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(sysbox_create_concurrency)
            if sysbox_create_concurrency > 0
            else None
        )
        # Symmetric tighter cap for sysbox DESTROYS. A sysbox teardown unmounts
        # the container's sysbox-fs FUSE layers (``fusermount3``); concurrent
        # unmounts under high create/destroy churn wedge sysbox-fs the same way
        # concurrent creates overwhelm its register step — the wedged
        # ``docker rm`` then hangs in D-state and LEAKS the container, holding a
        # cap slot and dragging the whole sysbox layer (2026-07-08 conc-32 sweep:
        # 4 sysbox containers leaked, teardowns stuck on ``fusermount3``). The
        # general destroy cap (4) is too loose; serialise sysbox destroys
        # (default 1). ``0`` disables the sysbox gate (falls back to the general).
        self._sysbox_destroy_concurrency = sysbox_destroy_concurrency
        self._sysbox_destroy_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(sysbox_destroy_concurrency)
            if sysbox_destroy_concurrency > 0
            else None
        )
        # Node-lost guardrail — cap on concurrent bulk container⇄node
        # byte transfers (``get_archive`` / ``put_archive``). This is
        # the multi-tenant blast-radius bound: with 10+ users each able
        # to submit the heaviest job (EvoClaw copies the WHOLE
        # ``/testbed`` — hundreds of MB of many small files — out of
        # every eval container), an unbounded fan-out of large tar
        # streams would (a) pin the default ThreadPoolExecutor's workers
        # away from create/exec, (b) balloon node RAM, and (c) saturate
        # the docker daemon's tar IO. Capping concurrent transfers keeps
        # every tenant's copy paced instead of letting one wave starve
        # the node. Paired with ``get_archive_stream`` reading the tar in
        # ``ARCHIVE_CHUNK_BYTES`` hops off the event loop (a single copy
        # can no longer freeze the heartbeat regardless of size), this is
        # what lets the node survive the workload that took it ``lost``.
        # ``4`` mirrors the create/destroy caps. ``0`` disables the cap
        # (unbounded — kept for tests / the legacy path).
        self._archive_concurrency = archive_concurrency
        self._archive_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(archive_concurrency)
            if archive_concurrency > 0
            else None
        )
        # Plane-split guardrail — the max bytes a single
        # ``get_archive`` may relay through the control plane. The
        # control plane is a metadata channel, not a bulk-data pipe
        # (spec 00 invariant 6). A transfer whose streamed size exceeds
        # this is refused (``ArchiveTooLarge``) so no tenant can push a
        # whole container filesystem through the CP. ``0`` disables the
        # cap (unbounded — legacy / tests). See
        # ``DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES``.
        self._max_get_archive_relay_bytes = max_get_archive_relay_bytes
        # Stage-1 admission/capacity observability — see
        # notes/admission-stage-1-observability.md. Records docker-run
        # latency + docker-error counts so the heartbeat can carry
        # per-node health signals to the control plane.
        self._health = NodeHealthCollector()
        # P2 — cpuset pinning is per-container and OPT-IN: applied only when
        # an acquire's RuntimeLimits sets ``cpu_pinning=True`` (a genuinely
        # timing-sensitive task). Default is quota-only (CFS ``--cpus``),
        # faithful to how harbor runs the container — no node-wide flag, no
        # node restart to change it. The core ledger backs the opt-in path
        # and is reconciled from live containers on first pinning-acquire so
        # a node-process restart never leaks or double-assigns cores.
        self._core_ledger = _CoreLedger(total_cores or os.cpu_count() or 1)
        self._ledger_reconciled = False
        # The single privileged surface for egress restriction (spec 07):
        # tightens a running container's OUTPUT chain to an allowlist via
        # nsenter+iptables. Defaults to the real applier; tests inject a fake.
        self._egress_enforcer = egress_enforcer or DockerNsenterEnforcer()
        # P1.7.C.2 — multi-service compose projects. ``_compose_projects`` maps a
        # docker-compose project name to its owning rollout + brought-up record +
        # ensured image refs, so ``destroy_compose_project`` can down the WHOLE
        # stack and release its images. Each member container is ALSO registered
        # in ``_records`` (below) so the existing container-scoped exec / archive /
        # ownership path addresses ``main`` (and any member) unchanged. The runner
        # is lazily built (it shells out to ``docker compose``); tests inject one.
        self._compose_runner: ComposeProjectRunner | None = compose_runner
        self._compose_projects: dict[str, _ComposeProjectState] = {}
        # audit H10: project names reserved for an in-flight ``acquire_compose_project`` (before
        # its ``up`` registers into ``_compose_projects``). Guards two concurrent acquires that
        # pass the SAME (explicit / collision-sanitized) project name from both running
        # ``docker compose -p <name> up`` and the second clobbering the first's ownership.
        self._reserving_projects: set[str] = set()
        # audit H10 — project names whose durable-DISK cleanup (named volumes / network) could
        # not be confirmed pruned during a teardown (a transient docker hiccup). NO capacity
        # impact (the containers are already gone), but a leaked volume consumes disk + a reused
        # name risks stale data. Retried best-effort on the next compose teardown
        # (``_retry_pending_resource_prunes``) — a bounded GC/retry that needs no periodic task
        # and never touches the hot compose-up path. Lost on node restart → the operator cleanup
        # contract (the actionable teardown warning) covers that residual.
        self._pending_resource_prune: set[str] = set()
        # P6 step-2b — the cgroup-v2 cpuset-inheritance self-test backing
        # ``isolation_capable()`` (§8.10). Injectable seam: tests pass a
        # callable returning the simulated outcome; production runs the real
        # container probe against ``XRLENV_SELFTEST_IMAGE``. Result is cached
        # on first call (the capability doesn't change without a daemon/kernel
        # change, which restarts this agent).
        self._isolation_selftest: Callable[[], bool] = (
            isolation_selftest
            if isolation_selftest is not None
            else self._default_isolation_selftest
        )
        self._isolation_capable_cache: bool | None = None
        # P6 step-3b — the shared-parent cpuset scheme (§8.11). Wired lazily on
        # the first acquire IF this node is capable (see
        # ``_ensure_isolation_wiring``); a non-capable node leaves
        # ``_shared_parent`` None and behaves exactly as today. ``min_shared_cores``
        # is the §8.4 floor — the manager param overrides; ``None`` derives 25% of
        # the node's cores at wiring time. ``cgroup_writer`` is the injectable
        # /sys writer seam (tests pass a fake; production uses real writes).
        self._min_shared_cores_cfg = min_shared_cores
        self._cgroup_writer = cgroup_writer
        self._shared_parent: _SharedCpusetParent | None = None
        self._isolation_wired = False
        # P6 step-4a — cached count of live legacy unpinned-runc containers (the
        # R5 transition gap: started before the shared parent was wired, so NOT
        # under it). Set at wiring, refreshed on the heartbeat cadence
        # (``refresh_legacy_gap``). While > 0, ``pinned_cpu_capacity`` folds it
        # into a 0 free count so the step-4 scheduler predicate won't treat this
        # node as a hard ``required`` target until the gap drains (§8.12).
        self._legacy_unpinned_runc: int = 0

    @staticmethod
    def _semaphore_depth(
        sem: asyncio.Semaphore | None, concurrency: int,
    ) -> tuple[int, int]:
        """(inflight, queued) for a create semaphore, or (0, 0) if uncapped.
        No public API for semaphore depth — read the internals."""
        if sem is None:
            return 0, 0
        inflight = max(0, concurrency - sem._value)
        waiters = sem._waiters
        return inflight, (len(waiters) if waiters is not None else 0)

    def health_snapshot(self) -> NodeHealthSnapshot:
        """Current node-health signals, for the heartbeat (Stage 1).

        Folds the rolling docker-op samples with the live create-gate depth.
        Sums BOTH the general and the sysbox-specific create gates — sysbox
        creates queue behind their own tighter semaphore, so omitting it would
        report the node idle while sysbox create pressure is the actual
        bottleneck. Cheap + in-memory — safe to call on the heartbeat path.
        """
        gen_inflight, gen_queued = self._semaphore_depth(
            self._create_semaphore, self._create_concurrency,
        )
        sys_inflight, sys_queued = self._semaphore_depth(
            self._sysbox_create_semaphore, self._sysbox_create_concurrency,
        )
        return self._health.snapshot(
            create_inflight=gen_inflight + sys_inflight,
            create_queued=gen_queued + sys_queued,
        )

    def _destroy_gate(self, *, sysbox: bool = False) -> Any:
        """Async context manager bounding concurrent ``docker rm -f`` destroys.

        ``sysbox=True`` (a non-runc teardown) uses the tighter
        ``_sysbox_destroy_semaphore`` — sysbox-fs FUSE unmount is far heavier
        than a plain runc remove, so sysbox destroys need a lower concurrency
        than plain runc destroys (mirrors ``_create_gate``). Falls back to the
        general ``_destroy_semaphore``; a no-op when the relevant cap is 0
        (unbounded — legacy / tests)."""
        sem = (
            self._sysbox_destroy_semaphore
            if sysbox and self._sysbox_destroy_semaphore is not None
            else self._destroy_semaphore
        )
        if sem is None:
            return _null_async_cm()
        return sem

    def _create_gate(self, *, sysbox: bool = False) -> Any:
        """Async context manager bounding concurrent ``docker run`` creates.

        ``sysbox=True`` (a non-runc create) uses the tighter
        ``_sysbox_create_semaphore`` — sysbox-fs pre-register is much slower, so
        sysbox creates need a lower concurrency than plain runc creates. Falls
        back to the general ``_create_semaphore``; a no-op when the relevant cap
        is 0 (unbounded — legacy / tests)."""
        sem = (
            self._sysbox_create_semaphore
            if sysbox and self._sysbox_create_semaphore is not None
            else self._create_semaphore
        )
        if sem is None:
            return _null_async_cm()
        return sem

    def _archive_gate(self) -> Any:
        """Async context manager holding ``_archive_semaphore`` if a
        concurrency cap is configured, or a no-op when
        ``archive_concurrency=0`` (unbounded — legacy / tests). Bounds
        concurrent bulk container⇄node transfers so no burst of large
        ``get_archive`` / ``put_archive`` copies can starve the node."""
        if self._archive_semaphore is None:
            return _null_async_cm()
        return self._archive_semaphore

    async def _ensure_ledger_reconciled(self) -> None:
        """P2 — one-time rebuild of the core ledger from live raw
        containers. A node-process restart between acquire and destroy
        would otherwise leave the ledger believing cores are free that
        a surviving container still pins; docker is the source of
        truth. Best-effort: a docker failure leaves the ledger fully
        free rather than blocking acquires."""
        if self._ledger_reconciled:
            return
        async with self._lock:
            if self._ledger_reconciled:
                return
            try:
                containers = await asyncio.to_thread(
                    self._client.containers.list,
                    filters={"label": "xrlenv.session_kind=raw"},
                    all=True,
                )
            except Exception as exc:
                LOGGER.warning(
                    "core-ledger reconcile: docker list failed (%r); "
                    "starting with a fully-free ledger", exc,
                )
                self._ledger_reconciled = True
                return
            reclaimed = 0
            for c in containers:
                attrs = getattr(c, "attrs", None) or {}
                spec = attrs.get("HostConfig", {}).get("CpusetCpus") or ""
                cores = _parse_cpuset(spec)
                if cores:
                    self._core_ledger.mark_in_use(cores)
                    reclaimed += len(cores)
            self._ledger_reconciled = True
            if reclaimed:
                LOGGER.info(
                    "core-ledger reconcile: %d core(s) marked in-use "
                    "from %d live raw container(s)",
                    reclaimed, len(containers),
                )

    async def _ensure_isolation_wiring(self) -> None:
        """P6 step-3b (§8.11) — on a CAPABLE node, wire the shared parent + §8.4
        floor into the ledger ONCE (lazily, at first acquire, once docker +
        capability are known). A non-capable node leaves the ledger untouched
        and keeps today's per-container path — behavior-neutral for it.

        After this, every allocate/release keeps ``xrlenv-shared.cpuset.cpus`` at
        ``all - in_use`` and the create path confines unpinned-runc containers
        under the shared parent (see the acquire path)."""
        if self._isolation_wired:
            return
        # Use the capability computed ONCE at NodeHello (which runs the self-test
        # before any command is processed). Do NOT trigger the self-test from the
        # acquire path — that would add a docker-info / cgroup probe to every
        # first acquire (and a node's hello already resolved it). If it hasn't run
        # yet (cache is None — e.g. an in-process runtime that never sends hello),
        # defer wiring: leave the node on today's path until capability is known.
        capable = self._isolation_capable_cache
        if capable is None:
            return
        # Reconcile first (it takes ``self._lock`` itself) so the INITIAL
        # complement reflects live pinned cores. Done outside the wiring lock
        # below to avoid re-entering the non-reentrant lock.
        if capable:
            await self._ensure_ledger_reconciled()
        # Set the flag + wire the shared parent atomically under the manager lock
        # (double-checked), so a concurrent first-acquire can't observe
        # ``_isolation_wired`` set while ``_shared_parent`` is still None.
        async with self._lock:
            if self._isolation_wired:
                return
            self._isolation_wired = True
            if not capable:
                return
            total = self._core_ledger.total
            floor = (
                self._min_shared_cores_cfg
                if self._min_shared_cores_cfg is not None
                else math.ceil(total * _DEFAULT_SHARED_FLOOR_FRACTION)
            )
            parent = _SharedCpusetParent(
                total_cores=total, writer=self._cgroup_writer,
            )
            self._shared_parent = parent
            self._core_ledger.enable_shared_parent(parent, floor)
        legacy = await self._count_legacy_unpinned_runc()
        self._legacy_unpinned_runc = legacy
        LOGGER.info(
            "CPU-isolation shared-parent cpuset ENABLED on this node (floor=%d/%d logical "
            "CPUs kept in the shared pool; floor == total intentionally means "
            "nothing is pinnable). %d pre-existing unpinned-runc container(s) "
            "are NOT under the shared parent (R5 transition gap — no live "
            "migration; drains as they exit; step 4 won't treat `required` as a "
            "hard guarantee here until it reaches 0).",
            self._core_ledger.min_shared_cores, total, legacy,
        )

    async def _count_legacy_unpinned_runc(self) -> int:
        """P6 step-3b R5 (§8.11) — count live raw containers that are unpinned
        (no cpuset) + runc + NOT under the shared parent, i.e. started before
        this node wired ``xrlenv-shared``. Best-effort (docker failure → 0).
        This is the transition gap the step-4 ``required`` placement gate must
        see drain to zero on a node before it treats the complement as
        authoritative there."""
        parent = "" if self._shared_parent is None else self._shared_parent.cgroup_parent
        try:
            containers = await asyncio.to_thread(
                self._client.containers.list,
                filters={"label": "xrlenv.session_kind=raw"},
                all=True,
            )
        except Exception:
            return 0
        count = 0
        for c in containers:
            hc = (getattr(c, "attrs", None) or {}).get("HostConfig", {})
            if hc.get("CpusetCpus"):
                continue  # pinned — has its own exclusive cores, not shared-pool
            if str(hc.get("Runtime") or "runc") != "runc":
                continue  # sysbox etc. — left on today's path by design
            if (hc.get("CgroupParent") or "") == parent:
                continue  # already under the shared parent (created post-wiring)
            count += 1
        return count

    async def _allocate_cpuset(
        self, resources: ResourceSpec | None, *, cpu_isolation: CpuIsolation,
    ) -> tuple[int, ...]:
        """P2/P6 — reserve ``ceil(cpu_limit)`` whole cores for a raw container
        when the effective ``cpu_isolation`` pins (``BEST_EFFORT`` or
        ``REQUIRED`` — ``.pins``). Returns the core indices, ``()`` when
        isolation is ``OFF``, or — for ``BEST_EFFORT`` on an exhausted ledger —
        ``()`` (graceful degradation to CFS quota, better than failing an
        acquire the scheduler admitted).

        P6 step-4c — ``REQUIRED`` is **pin-or-fail**: on ledger exhaustion it
        raises :class:`PinCapacityExhausted` instead of degrading, so a
        hard-isolated rollout never silently runs on shared CFS quota (which would
        reopen the very trampling the caller required isolation to prevent). The
        step-4b scheduler predicate steers ``required`` only to isolation-capable
        nodes with free pinnable cores (subtracting in-flight reservations), so
        reaching exhaustion here is a stale-heartbeat / ledger race — the
        NODE-SPECIFIC ``PinCapacityExhausted`` lets the control plane re-admit on
        a sibling capable node (vs a plain ``CapacityExhausted``, which is
        terminal). The raise happens BEFORE any container is created and holds no
        ledger reservation (``allocate`` returned ``None``), so nothing leaks."""
        if not cpu_isolation.pins:
            return ()
        await self._ensure_ledger_reconciled()
        want = math.ceil(_effective_cpu_limit(resources))
        cores = self._core_ledger.allocate(want)
        if cores is None:
            if cpu_isolation is CpuIsolation.REQUIRED:
                raise PinCapacityExhausted(
                    f"cpu_isolation=required but the node core ledger is "
                    f"exhausted ({self._core_ledger.free_count()}/"
                    f"{self._core_ledger.total} free, wanted {want}) — refusing "
                    f"to degrade a REQUIRED pin to CFS quota (pin-or-fail)"
                )
            LOGGER.warning(
                "cpuset pinning: node core ledger exhausted (%d/%d "
                "free, wanted %d) — spawning with CFS quota only (best_effort)",
                self._core_ledger.free_count(),
                self._core_ledger.total, want,
            )
            return ()
        return cores

    def registered_runtimes(self) -> tuple[frozenset[str], str]:
        """``(runtimes, default_runtime)`` from ``docker info``, cached.

        Used to (a) verify a requested ``container_runtime`` is registered
        before ``containers.run`` (§5.5), and (b) advertise
        ``supported_runtimes`` / ``default_runtime`` on ``NodeHello``
        (§5.3). Cached because docker's runtime set only changes on a
        daemon restart, which restarts this agent too.

        On a transient ``docker info`` failure we return a conservative
        ``({"runc"}, "runc")`` WITHOUT caching it — a sysbox acquire then
        fails loud at verification (the safe outcome), and the next call
        retries the probe.
        """
        if self._runtime_info_cache is not None:
            return self._runtime_info_cache
        try:
            info = self._client.info()
        except Exception:
            LOGGER.warning(
                "raw-container: `docker info` failed while probing runtimes; "
                "advertising conservative {'runc'} until it recovers",
                exc_info=True,
            )
            return (frozenset({"runc"}), "runc")
        runtimes = frozenset(
            (info.get("Runtimes") or {}).keys(),
        ) or frozenset({"runc"})
        default = str(info.get("DefaultRuntime") or "runc")
        self._runtime_info_cache = (runtimes, default)
        return self._runtime_info_cache

    def runtimes_probed(self) -> bool:
        """True once a successful ``docker info`` runtime probe has populated the
        cache — i.e. the advertised ``supported_runtimes`` are authoritative
        rather than the conservative ``{'runc'}`` fallback ``registered_runtimes``
        returns (uncached) when docker hasn't answered yet.

        The node link gates its first ``NodeHello`` on this: a node whose agent
        starts seconds after a docker restart (the redeploy race) would otherwise
        enumerate runtimes against a not-yet-ready daemon and advertise a
        conservative set for the whole connection — invisible to the scheduler's
        runtime filter until a manual restart."""
        return self._runtime_info_cache is not None

    def isolation_capable(self) -> bool:
        """P6 (§8.6, §8.10, §8.13) — whether this node can enforce the
        shared-parent cpuset isolation scheme, advertised on ``NodeHello``.

        Runs the capability check once (cached). Because the agent is non-root
        (spec 19), the heavy container probe runs at ENABLE time (as root) and
        this check verifies the resulting **delegation** is intact — the docker
        cgroup driver is ``cgroupfs`` (v1 gate), cgroup v2 exposes ``cpuset``,
        and the enable step created + ``chown``\\ed ``xrlenv-shared`` to this
        agent (see :meth:`_default_isolation_selftest`). A ``systemd``-driver
        node, a cgroup-v1 node, a node that was never enabled, or ANY error ⇒
        ``False`` — the node falls back to today's per-container pinning / CFS
        quota. Fail-safe: a node never advertises isolation it hasn't
        demonstrated (worst case is a capable node under-reporting, never a
        false ``true``)."""
        if self._isolation_capable_cache is None:
            try:
                self._isolation_capable_cache = bool(self._isolation_selftest())
            except Exception:
                LOGGER.warning(
                    "isolation self-test raised; advertising "
                    "isolation_capable=false",
                    exc_info=True,
                )
                self._isolation_capable_cache = False
        return self._isolation_capable_cache

    def _default_isolation_selftest(self) -> bool:
        """The production capability check (§8.13).

        The node agent runs as a NON-ROOT user (spec 19), which cannot
        ``mkdir`` / write under ``/sys/fs/cgroup`` — so it can't run the real
        container probe (:func:`_run_cgroup_isolation_selftest`, which needs
        root). That probe instead runs at ENABLE time
        (``deploy/node/enable_cpu_isolation.sh``, as root) as the gate before the
        enable step DELEGATES the shared parent ``xrlenv-shared`` to this agent
        user. Here we verify that delegation is intact — the cheap, root-free
        signal that the node was enabled and this agent can manage the shared
        cpuset:

        * the docker cgroup driver is ``cgroupfs`` (the P6-v1 gate); AND
        * cgroup v2 exposes the ``cpuset`` controller; AND
        * ``/sys/fs/cgroup/xrlenv-shared/cpuset.cpus`` exists and is writable by
          this (non-root) agent — i.e. the enable step created + ``chown``\\ed it.

        Absent / not-writable ⇒ the node was never enabled (or the delegation
        was torn down, e.g. a driver flip or reboot) ⇒ non-capable. Fail-safe:
        never a false ``true`` (the worst case is a capable node under-reporting
        until the enable step re-runs)."""
        if _docker_cgroup_driver(self._client) != "cgroupfs":
            return False  # P6-v1 gate: systemd-driver nodes stay non-capable
        if not _cgroup_v2_cpuset_available():
            return False
        return _delegated_shared_parent_writable()

    def pinned_cpu_capacity(self) -> tuple[int, int]:
        """P6 (§8.6, R6) — ``(pinned_cpus_free, pinned_cpus_total)`` for the
        heartbeat's live pinned-CPU accounting.

        ``free`` = cores not currently reserved by a pinned container, ``total``
        = the node's pinnable core count — straight off the core ledger.

        P6 step-4a (§8.12): on a capable node that still has **legacy
        unpinned-runc** containers (the R5 transition gap — not under the shared
        parent, so the complement isn't yet authoritative), ``free`` is folded to
        **0** so the step-4 scheduler predicate refuses ``required`` here until the
        gap drains. ``total`` still reports the real ledger total (so operator
        views read ``0/total`` = "no pinnable capacity right now", not "no
        cores"). Behavior-neutral until the scheduler consumes ``free`` (step 4b);
        the gap is zero on a freshly-enabled node (the enablement docker restart
        doesn't leave legacy containers)."""
        total = self._core_ledger.total
        if self._shared_parent is not None and self._legacy_unpinned_runc > 0:
            return (0, total)
        return (self._core_ledger.free_count(), total)

    async def refresh_legacy_gap(self) -> int:
        """P6 step-4a — recompute + cache the live legacy-unpinned-runc count (the
        heartbeat calls this on its cadence so the gap drains as those containers
        exit). No-op / 0 on a non-capable node (no shared parent wired). Returns
        the current count.

        P6 step-4b pre-req (audit follow-up): PROACTIVELY wire isolation on a
        capable node here (the heartbeat cadence) rather than waiting for the
        first acquire. Otherwise a capable-but-not-yet-wired node with
        pre-existing legacy unpinned-runc containers (e.g. an agent restart that
        did NOT drain them) reports full free capacity via ``pinned_cpu_capacity``
        — and step-4b's ``required`` predicate would place hard-pinned work on a
        node whose complement isn't yet enforced. After wiring, ``_shared_parent``
        is set and the count is real; ``_ensure_isolation_wiring`` is idempotent
        (returns immediately once wired) so only the first heartbeat pays the
        reconcile cost. A non-capable node stays ``_shared_parent is None`` ⇒ 0."""
        await self._ensure_isolation_wiring()
        if self._shared_parent is None:
            self._legacy_unpinned_runc = 0
        else:
            self._legacy_unpinned_runc = await self._count_legacy_unpinned_runc()
        return self._legacy_unpinned_runc

    async def acquire(
        self,
        *,
        rollout_id: str,
        image: str,
        command: list[str] | None = None,
        entrypoint: list[str] | None = None,
        user: str | None = None,
        cap_add: list[str] | None = None,
        devices: list[str] | None = None,
        privileged: bool = False,
        network_mode: str | None = None,
        binds: list[str] | None = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        ensure_image_present: bool = True,
        userns_mode: str = "host",
        ensure_image_deadline_s: float | None = None,
        resources: ResourceSpec | None = None,
        runtime_limits: RuntimeLimits | None = None,
        container_runtime: str | None = None,
    ) -> RawContainerRecord:
        """Spawn a raw container scoped to ``rollout_id``.

        Always merges the platform's reserved labels on top of any
        operator-provided labels:
        ``xrlenv.rollout_id``, ``xrlenv.session_kind=raw``.

        ``ensure_image_present`` (P1.7.B.2):

        - **True** (default — the new UX): the manager runs
          :meth:`_ensure_image_present` which delegates to
          :class:`xrlenv.node.image_cache.ImageCacheManager`'s
          ``ensure_present`` when one is wired (P1.7.B.2 fully
          ships this); pulls / builds / no-ops as appropriate.
          When no ImageCacheManager is wired (legacy fixtures),
          falls back to the strict path below.
        - **False** (deterministic-eval opt-out): refuses acquire
          if the image is missing locally — preserves the
          P1.7.A.1 contract for harnesses that need "no surprise
          pulls during evaluation."

        ``ensure_image_deadline_s`` (issue #12): overrides the
        ``ImageCacheManager.ensure_present`` pull / build deadline
        for this acquire only. ``None`` keeps the manager's default
        (600 s). The control plane derives this from the consumer's
        ``acquire_timeout_s`` so a 10 GB image pulled under a 1800 s
        wire wait actually gets 1800 s at the node, instead of
        failing at 600 s while the wire patiently waited (audit
        M1 against commit `86afca1`).

        ``resources`` (P1, cluster-resource-isolation-plan): the
        effective ``ResourceSpec`` the control plane placed this
        container with. The CPU/memory limits become docker cgroup
        kwargs (``cpu_period`` / ``cpu_quota`` / ``mem_limit``) so the
        container is actually capped at runtime — without this a raw
        container bursts to every host core and oversubscribes the
        node. ``None`` (or a non-positive limit) falls back to the
        node default; a raw container is never spawned uncapped.

        ``runtime_limits`` (P0b): container-shape limits (pids / shm /
        tmpfs / read-only rootfs) the harness requested. Only the
        fields the harness set are applied — cluster mode injects no
        pids/shm defaults, so an unspecified limit behaves exactly as
        against a local Docker daemon. ``None`` applies none.

        cpuset pinning (P2): when enabled, ``ceil(cpu_limit)`` whole
        host cores are reserved from the node core ledger and the
        container is pinned to them (``cpuset_cpus``) so timing-
        sensitive tests aren't perturbed by CFS interleaving. The cores
        are released on destroy.
        """
        # Audit P2 — the caller's ``ensure_image_deadline_s`` is the whole
        # AcquireContainer wire budget (the CP's ``_send_and_wait`` timeout, which
        # also caps the image pull). The create retry-with-backoff must not sleep
        # past it: a fail-fast caller (small ``acquire_timeout_s``) would otherwise
        # time out at the control plane while this node keeps retrying and
        # possibly spawns an untracked container replying to an already-failed
        # command. Anchor an absolute deadline at acquire entry (before the pull,
        # which eats into it) so create retries only use the budget that remains.
        _acquire_started = time.monotonic()
        _retry_deadline: float | None = (
            _acquire_started + ensure_image_deadline_s
            if ensure_image_deadline_s is not None
            else None
        )
        await self._ensure_image_present(
            image, strict=not ensure_image_present,
            deadline_s=ensure_image_deadline_s,
        )

        merged_labels = dict(labels or {})
        merged_labels["xrlenv.rollout_id"] = rollout_id
        merged_labels["xrlenv.session_kind"] = "raw"

        run_kwargs: dict[str, Any] = {
            "image": image,
            "detach": True,
            "labels": merged_labels,
            # P1 — CPU/memory cgroup limits. Always present (the helper
            # falls back to the node default), so a raw container is
            # never spawned unbounded.
            **_effective_cgroup_run_kwargs(resources),
            # P0b — container-shape RuntimeLimits (pids / shm / tmpfs /
            # read-only). Only harness-specified limits are applied.
            **_runtime_limits_run_kwargs(runtime_limits),
        }
        # §5.1 — OCI runtime selector. Omit the kwarg for the default (None
        # or the daemon default "runc") so the normal raw path is
        # byte-for-byte unchanged; docker-py maps ``runtime=`` onto
        # ``HostConfig.Runtime``. The runtime is verified as registered on
        # this node before ``containers.run`` (§5.5, below).
        _uses_sysbox = bool(container_runtime) and container_runtime != "runc"
        if _uses_sysbox:
            run_kwargs["runtime"] = container_runtime
        # Docker's built-in tini (``--init``) as PID 1 reaps orphaned/zombie child
        # processes and forwards signals — the correct default for agent/eval
        # containers whose workloads spawn helper subprocesses. Without it, a
        # non-reaping PID 1 (e.g. ``tail -f /dev/null``) leaves orphans defunct,
        # which breaks tests that assert a child exits with its parent (e.g.
        # EvoClaw nushell ``plugin_process_exits_when_nushell_exits``). Harmless
        # for other workloads; ``XRLENV_RAW_INIT=0`` opts a node out.
        #
        # §5.6 — but NOT in front of a Sysbox system container: sysbox-runc
        # workloads run systemd (or an inner init/dockerd) as PID 1, and
        # injecting tini as PID 1 breaks that. For a sysbox acquire we skip
        # the tini injection entirely (per-acquire, not node-wide — the
        # ``XRLENV_RAW_INIT`` escape hatch still governs the normal path).
        _init_on = os.environ.get(
            "XRLENV_RAW_INIT", "1",
        ).strip().lower() not in ("0", "false", "no", "off")
        if _init_on and not _uses_sysbox:
            run_kwargs["init"] = True
        if command:
            run_kwargs["command"] = command
        if entrypoint is not None:
            # docker-py's ``containers.run(entrypoint=...)`` accepts
            # a string or a list. Pass through verbatim; the
            # ``[""]`` "clear ENTRYPOINT" idiom comes from the
            # operator-facing surface and we don't try to translate
            # — docker handles it.
            run_kwargs["entrypoint"] = entrypoint
        if user:
            run_kwargs["user"] = user
        if cap_add:
            run_kwargs["cap_add"] = list(cap_add)
        if devices:
            run_kwargs["devices"] = list(devices)
        if privileged:
            # docker-py: ``containers.run(privileged=True)`` requests
            # full host capabilities. Policy decision made by the
            # control plane against ``allow_privileged``; by the time
            # we get here, the operator has already opted in.
            run_kwargs["privileged"] = True
        if network_mode:
            # docker-py: ``containers.run(network_mode="bridge"|"host"|
            # "none"|...)``. ``"host"`` permitted only when the cluster
            # policy's ``allow_host_network`` opt-in is set; gating done
            # at the control plane.
            run_kwargs["network_mode"] = network_mode
        if binds:
            # docker-py prefers ``volumes`` over the legacy
            # ``host_config.Binds`` field. Each entry is a
            # ``"/host/path:/container/path[:mode]"`` spec — translate
            # to the dict form docker-py's high-level manager expects.
            volumes: dict[str, dict[str, str]] = {}
            for spec in binds:
                parts = spec.split(":")
                host = parts[0]
                container = parts[1] if len(parts) > 1 else host
                mode = parts[2] if len(parts) > 2 else "rw"
                volumes[host] = {"bind": container, "mode": mode}
            run_kwargs["volumes"] = volumes
        if name:
            run_kwargs["name"] = name
        if environment:
            run_kwargs["environment"] = environment
        # B5.4 — userns-remap opt-in. ``"host"`` (default) explicitly
        # opts out of the daemon's userns-remap config (matches
        # ``docker run --userns=host``); ``"remap"`` translates to
        # docker-py's ``userns_mode=""`` which means "use the
        # daemon's userns-remap if configured." Reject anything else
        # outright so a remap-enabled daemon can't silently use its
        # default for an unknown wire value.
        if userns_mode == "host":
            run_kwargs["userns_mode"] = "host"
        elif userns_mode == "remap":
            # Empty string ⇒ docker honors daemon's userns-remap.
            run_kwargs["userns_mode"] = ""
        else:
            raise ValueError(
                f"unsupported userns_mode={userns_mode!r}; "
                "expected 'host' or 'remap'",
            )

        # §5.5 — verify a requested non-default runtime is actually
        # registered in docker on THIS node before ``containers.run``.
        # Fail loud (don't silently fall back to runc — that would run a
        # systemd/DinD workload under the wrong runtime and fail
        # confusingly). Placement should already have steered sysbox
        # acquires to sysbox-capable nodes (§5.3); this is the last-line
        # node-local guard for a stale snapshot or a direct/test call.
        if _uses_sysbox:
            _registered, _ = self.registered_runtimes()
            if container_runtime not in _registered:
                raise XRLEnvError(
                    f"raw-container acquire: requested container_runtime "
                    f"{container_runtime!r} is not registered in docker on "
                    f"this node (registered runtimes: {sorted(_registered)}). "
                    f"Install/enable it (e.g. sysbox-ce) and restart docker, "
                    f"or acquire without container_runtime to use the default.",
                )

        # P2 — reserve dedicated cores. Done *after* all local
        # run-kwargs validation (the userns_mode check above can raise)
        # and immediately before the guarded ``containers.run`` call, so
        # a pre-run ValueError can't leak a ledger reservation. Any
        # failure from here on releases the cores (see the except below).
        # P6: the effective isolation mode is read from the ResourceSpec the
        # control plane derived (resources.cpu_isolation), falling back to the
        # legacy runtime_limits.cpu_pinning alias when the CP left it OFF — so
        # this honors an explicit cpu_isolation without changing today's
        # cpu_pinning behavior.
        # P6 step-3b — on a capable node, wire the shared parent + floor once
        # (before the first allocate, so the floor governs it). No-op on a
        # non-capable node.
        await self._ensure_isolation_wiring()
        cpuset_cores = await self._allocate_cpuset(
            resources,
            cpu_isolation=effective_cpu_isolation(resources, runtime_limits),
        )
        if cpuset_cores:
            # Pinned: exclusive whole cores via the container's own cpuset. The
            # ledger already shrank the shared parent to exclude these (§8.2).
            run_kwargs["cpuset_cpus"] = _cpuset_str(cpuset_cores)
        elif self._shared_parent is not None and not _uses_sysbox:
            # P6 step-3b — an UNPINNED runc container on a capable node is placed
            # under the shared parent (no own cpuset), so its effective cpuset is
            # the complement of the pinned cores and it can't trample them. Sysbox
            # containers are left on today's path (the self-test only proved runc);
            # a non-capable node has no shared parent → unchanged.
            run_kwargs["cgroup_parent"] = self._shared_parent.cgroup_parent

        # Issue #18 (Ask #3): translate raw docker-py / requests
        # exceptions into an XRLEnvError with node-side context
        # (operation, image, the docker HTTP-client timeout) so the
        # consumer doesn't get an opaque ``ReadTimeout:
        # UnixHTTPConnectionPool(...)`` with no hint about which
        # layer fired. ``containers.run`` is the call the SWE-bench
        # Pro run blew the 60s default on.
        try:
            # Bounded retry-with-backoff on a transient node-saturation create
            # failure (5xx / timeout — e.g. sysbox-fs pre-register DeadlineExceeded
            # under a create burst). Recovered transients return a container
            # instead of failing the acquire; the create gate + AIMD recording
            # live inside. See ``_create_with_retry``.
            container = await self._create_with_retry(
                run_kwargs, rollout_id=rollout_id, sysbox=_uses_sysbox,
                retry_deadline=_retry_deadline,
            )
        except (
            docker.errors.DockerException,
            requests.exceptions.RequestException,
        ) as exc:
            # Give-up (retries exhausted) or a non-retryable request fault (404 /
            # 409-after-reclaim). Health was already recorded once inside the
            # retry helper for a saturation fault; a request fault records
            # nothing (a duplicate name / missing image must not collapse the
            # node's admission limit — the prod regression this guards against).
            # P2 — the container never spawned; return its reserved cores so a
            # failed acquire can't leak them.
            if cpuset_cores:
                self._core_ledger.release(cpuset_cores)
            raise _translate_docker_error(
                operation="containers.run",
                target=f"image={image!r}",
                client=self._client,
                exc=exc,
            ) from exc
        record = RawContainerRecord(
            rollout_id=rollout_id,
            container_id=container.id,
            container_name=container.name,
            image=image,
            created_at=_dt.datetime.now(_dt.UTC),
            cpuset=cpuset_cores,
            container_runtime=container_runtime,
        )
        async with self._lock:
            self._records[container.id] = record
        # Mark the image in-use in the cache for the container's lifetime
        # (released on destroy). Without this a running raw container's
        # image is misclassified "cold" — the image-eviction sweep then
        # spins futilely trying to remove an in-use image every tick (the
        # prod log-spam), and the disk guard's ``evictable_image_bytes``
        # over-counts it. Mirrors the case-1 sandbox acquire/release.
        if self._image_cache is not None:
            self._image_cache.acquire(image)
        LOGGER.info(
            "raw-container.acquire rollout=%s container=%s image=%s",
            rollout_id, container.id[:12], image,
        )
        return record

    # ── P1.7.C.2 — multi-service compose projects ─────────────────────────────

    async def _ensure_compose_runner(self) -> ComposeProjectRunner:
        """Lazily build the compose runner, wiring image-ensure through the node's
        image cache so ``docker compose up`` never triggers an uncached pull."""
        if self._compose_runner is None:
            image_cache = self._image_cache

            async def _ensure(ref: str) -> None:
                if image_cache is not None:
                    await image_cache.ensure_present(ref)

            self._compose_runner = ComposeProjectRunner(ensure_image=_ensure)
        return self._compose_runner

    async def acquire_compose_project(
        self,
        *,
        rollout_id: str,
        project_name: str,
        compose_yaml: str,
        images: Iterable[str] = (),
        main_service: str = "main",
        up_timeout_s: float | None = None,
    ) -> ComposeProjectRecord:
        """Bring up a whole compose project and register every member container.

        The runner (``docker compose up -d --wait``) runs under the create gate —
        a compose ``up`` is N container creates, so it inherits the same per-node
        create-pressure pacing single acquires do. Each resulting member container
        is registered in ``_records`` ↔ ``rollout_id`` so the existing
        container-scoped exec / archive / ownership path addresses ``main`` (the
        harbor exec target) and any sidecar unchanged. On any failure the runner
        has already torn the partial project down, so nothing is registered."""
        main_service = main_service or "main"
        image_refs = tuple(images)
        # P6 step-3c — on a capable node, confine every RUNC service to the shared
        # pool (complement of the pinned cores) by injecting cgroup_parent into the
        # compose document before `docker compose up`. A sidecar with full-host
        # affinity would reopen the trampling bug. Non-capable node → no shared
        # parent → unchanged; sysbox services are skipped inside the injector.
        await self._ensure_isolation_wiring()
        if self._shared_parent is not None:
            compose_yaml = _inject_shared_cgroup_parent(
                compose_yaml, self._shared_parent.cgroup_parent,
            )
        # audit H10: RESERVE the project name atomically before `up`, rejecting a collision with
        # a live project OR another in-flight acquire — otherwise two acquires with the same name
        # both `docker compose -p <name> up` and the second overwrites the first's ownership
        # (the first can then never destroy its project, and `up` can mutate its resources).
        async with self._lock:
            if project_name in self._compose_projects:
                raise XRLEnvError(
                    f"compose acquire: project name {project_name!r} is already active on this "
                    f"node (owner rollout {self._compose_projects[project_name].rollout_id!r}).",
                )
            if project_name in self._reserving_projects:
                raise XRLEnvError(
                    f"compose acquire: project name {project_name!r} is being acquired "
                    f"concurrently on this node — collision rejected.",
                )
            self._reserving_projects.add(project_name)
        runner = await self._ensure_compose_runner()
        record: ComposeProjectRecord | None = None
        try:
            async with self._create_gate():
                record = await runner.up(
                    project_name=project_name,
                    compose_yaml=compose_yaml,
                    images=image_refs,
                    main_service=main_service,
                    up_timeout_s=up_timeout_s or DEFAULT_UP_TIMEOUT_S,
                )
            now = _dt.datetime.now(_dt.UTC)
            # Registration is a run of in-memory dict writes under the lock with no await
            # BETWEEN them, so it is atomic w.r.t. cancellation once the lock is acquired —
            # a cancel can only land at the ``async with self._lock`` acquire (nothing
            # registered yet), never mid-registration (no partial state).
            async with self._lock:
                for service, cid in record.service_container_ids.items():
                    self._records[cid] = RawContainerRecord(
                        rollout_id=rollout_id,
                        container_id=cid,
                        container_name=(
                            record.main_container_name if service == main_service else cid
                        ),
                        image=f"compose:{project_name}/{service}",
                        created_at=now,
                    )
                self._compose_projects[project_name] = _ComposeProjectState(
                    rollout_id=rollout_id, record=record, images=image_refs,
                )
        except BaseException:
            # audit H10 — cancellation-safe rollback. ``up`` may have brought a LIVE stack up
            # before we were cancelled (or before registration ran); if it is NOT registered it
            # would leak as an unowned project (invisible to destroy, holding cpu/mem/disk).
            # Tear the just-created stack down best-effort. If registration DID complete, leave it
            # for the caller's normal destroy path (the stack is owned + accounted).
            async with self._lock:
                registered = self._compose_projects.get(project_name) is not None and (
                    self._compose_projects[project_name].rollout_id == rollout_id
                )
            if record is not None and not registered:
                with suppress(Exception):
                    await runner.down(
                        project_name=project_name,
                        project_dir=record.project_dir,
                    )
            raise
        finally:
            # The reservation is ALWAYS released — success, failure, or cancellation — so a
            # crashed/cancelled acquire never wedges the name against a retry (audit H10).
            async with self._lock:
                self._reserving_projects.discard(project_name)
        # Refcount the images in-use for the project's lifetime (released on
        # destroy) so the eviction sweep doesn't misclassify them cold — the
        # same reason single-container ``acquire`` marks its image in-use.
        if self._image_cache is not None:
            for ref in set(image_refs):
                self._image_cache.acquire(ref)
        LOGGER.info(
            "compose.acquire rollout=%s project=%s services=%d main=%s",
            rollout_id, project_name, len(record.service_container_ids),
            record.main_container_id[:12],
        )
        return record

    async def destroy_compose_project(
        self, *, rollout_id: str, project_name: str, force: bool = True,
    ) -> None:
        """``docker compose down`` the whole project + deregister every member.

        Ownership-enforced like :meth:`destroy`: refuses an unregistered project
        or a rollout that doesn't own it. Idempotent teardown (the runner's
        ``down`` swallows an already-gone project). Reaps the WHOLE stack — never
        leaves a sidecar behind."""
        # audit H10 — piggyback the bounded GC/retry for any earlier teardown's unconfirmed
        # disk-resource prune (leaked named volumes / networks). Best-effort, never raises.
        await self._retry_pending_resource_prunes()
        async with self._lock:
            state = self._compose_projects.get(project_name)
        if state is None:
            # H10 — idempotent CONFIRMED-ABSENCE teardown. The project isn't in this node's
            # in-memory map in two recovery cases: (a) a prior `down` COMPLETED but its reply
            # timed out (state already popped), or (b) the node-agent RESTARTED (memory-only
            # ownership wiped) while Docker members may survive. Raising "not registered" here
            # would strand the coordinator's aggregate capacity forever — every retry would
            # re-raise and re-retain. Instead reap any surviving members by the project+rollout
            # labels and VERIFY none remain: on success the coordinator finalizes + frees; on
            # Docker uncertainty / a surviving member the helper RAISES (fail closed) so capacity
            # stays charged and the next attempt retries — never a false confirmed-absence that
            # frees capacity while members are still running.
            #
            # audit H10 — fully SERIALIZE the recovery reap against a concurrent same-name ACQUIRE
            # via the project-name reservation. RESERVE the name for the whole reap so acquire +
            # recovery never run `docker`/`docker compose` on the same project concurrently (a
            # concurrent acquire's `up` reconciles by project name and could otherwise adopt/recreate
            # this rollout's leftover members, and its named volumes/network share the label). If a
            # NEW acquire already holds the name, DEFER (raise a retryable error) rather than reap
            # concurrently: the coordinator retains capacity and retries, by which time the acquire
            # has released the name and the reap runs cleanly + confirmed. (Was: reap concurrently
            # while skipping the name-scoped prune — left the `docker compose up` reconciliation
            # race open, audit H10.)
            async with self._lock:
                name_taken = (
                    project_name in self._reserving_projects
                    or project_name in self._compose_projects
                )
                if not name_taken:
                    self._reserving_projects.add(project_name)
            if name_taken:
                raise XRLEnvError(
                    f"compose recovery: project name {project_name!r} is being acquired "
                    f"concurrently on this node — DEFERRING the reap to avoid racing `docker "
                    f"compose up` reconciliation; capacity retained, will retry (audit H10).",
                )
            try:
                # We hold the reservation → no concurrent acquire → full prune is safe.
                await self._reap_compose_project_by_label(project_name, rollout_id)
            finally:
                async with self._lock:
                    self._reserving_projects.discard(project_name)
            return
        if state.rollout_id != rollout_id:
            raise XRLEnvError(
                f"compose destroy: rollout {rollout_id!r} does not own project "
                f"{project_name!r} (owner: {state.rollout_id!r}).",
            )
        del force  # down is always a full stop+remove (down -v --remove-orphans)
        runner = await self._ensure_compose_runner()
        async with self._destroy_gate():
            await runner.down(
                project_name=project_name,
                project_dir=state.record.project_dir,
            )
        async with self._lock:
            for cid in state.record.service_container_ids.values():
                self._records.pop(cid, None)
            self._compose_projects.pop(project_name, None)
        if self._image_cache is not None:
            for ref in set(state.images):
                self._image_cache.release(ref)
        LOGGER.info(
            "compose.destroy rollout=%s project=%s", rollout_id, project_name,
        )

    async def _reap_compose_project_by_label(
        self, project_name: str, rollout_id: str,
        *, prune_project_resources: bool = True,
    ) -> None:
        """H10 — FAIL-CLOSED confirmed-absence teardown of a compose project not in
        ``_compose_projects`` (a completed-down whose reply timed out, or a node-agent restart
        that wiped memory-only ownership).

        Selects members by BOTH docker compose's ``com.docker.compose.project`` label AND
        ``xrlenv.rollout_id`` (``compose_prepare`` stamps the latter on every service), so a
        project-NAME collision from a *different* rollout can never be reaped by mistake
        (ownership-safe). Removes each member **with its volumes**, then RE-LISTS to VERIFY none
        remain. Any Docker list/removal failure — or a surviving member on the recheck — RAISES,
        so the coordinator RETAINS aggregate capacity and retries rather than a false "confirmed
        absence" freeing capacity while members are still running (node-confirmed release
        invariant). Network cleanup is best-effort (a leaked network holds no capacity)."""
        label_filter = [
            f"com.docker.compose.project={project_name}",
            f"xrlenv.rollout_id={rollout_id}",
        ]

        def _list_members() -> list[Any]:
            # A raised DockerException here PROPAGATES (fail closed) — we cannot confirm absence.
            return list(self._client.containers.list(all=True, filters={"label": label_filter}))

        members = await asyncio.to_thread(_list_members)
        for m in members:
            # NotFound = already gone (raced away) → that member is confirmed absent; any OTHER
            # DockerException propagates → fail closed (the member may still be alive).
            with suppress(docker.errors.NotFound):
                await asyncio.to_thread(m.remove, force=True, v=True)  # remove volumes too
        # VERIFY teardown: re-list under the same ownership filter. A survivor means we could not
        # confirm absence — RAISE so capacity stays charged and the next attempt retries.
        remaining = await asyncio.to_thread(_list_members)
        if remaining:
            raise XRLEnvError(
                f"compose destroy: {len(remaining)} member(s) of project {project_name!r} "
                f"(rollout {rollout_id!r}) still present after removal — cannot confirm "
                f"teardown; capacity retained for retry.",
            )
        async with self._lock:
            for m in members:
                self._records.pop(str(getattr(m, "id", "")), None)
        # ── Durable DISK cleanup — EXPLICITLY SEPARATE from the capacity release above ───────
        # The container reap (above) is the capacity-relevant, verified, fail-closed step: once
        # it confirms no member remains, the coordinator's node-confirmed release is sound. NAMED
        # volumes + the project network carry the project label (``docker rm -v`` only drops
        # ANONYMOUS volumes); they hold NO capacity, so a failure to prune them must NOT fail-close
        # the teardown (audit H10). But we do RE-VERIFY and LOG survivors so the disk / project-
        # reuse uncertainty is SURFACED, not silently swallowed. SKIP entirely when
        # ``prune_project_resources`` is False — a NEW same-name acquire now owns the project name,
        # and its resources carry the SAME project label, so pruning by name would delete the new
        # owner's. The member reap stays rollout-scoped, so this skip leaks only the OLD rollout's
        # disk, never the new owner's.
        if prune_project_resources:
            proj_filter = {"label": f"com.docker.compose.project={project_name}"}
            await self._prune_project_resources_by_label(
                lambda: self._client.volumes.list(filters=proj_filter),
                lambda r: r.remove(force=True),
                kind="volume", project_name=project_name, rollout_id=rollout_id,
            )
            await self._prune_project_resources_by_label(
                lambda: self._client.networks.list(filters=proj_filter),
                lambda r: r.remove(),
                kind="network", project_name=project_name, rollout_id=rollout_id,
            )
        LOGGER.info(
            "compose.destroy rollout=%s project=%s — confirmed-absence teardown "
            "(verified %d member(s) removed by project+rollout label)",
            rollout_id, project_name, len(members),
        )

    async def _prune_project_resources_by_label(
        self, list_fn: Callable[[], Any], remove_fn: Callable[[Any], Any],
        *, kind: str, project_name: str, rollout_id: str,
    ) -> None:
        """Best-effort remove of the resources ``list_fn`` returns, then RE-LIST to VERIFY and
        LOG any survivor / list-error (audit H10). NEVER raises — this is durable-DISK cleanup,
        deliberately DECOUPLED from the capacity release (containers, already verified + fail-
        closed above), so a docker hiccup here can't fail-close a teardown whose containers are
        gone. The warning surfaces the disk / same-name-reuse uncertainty instead of swallowing
        it silently."""
        try:
            for res in await asyncio.to_thread(list_fn):
                with suppress(docker.errors.DockerException):
                    await asyncio.to_thread(remove_fn, res)
            remaining = len(await asyncio.to_thread(list_fn))   # RE-VERIFY absence
        except docker.errors.DockerException as exc:
            self._pending_resource_prune.add(project_name)   # H10 — queue for retry
            LOGGER.warning(
                "compose.destroy rollout=%s project=%s — could not confirm %s prune (%s); "
                "best-effort DISK cleanup (no capacity impact). Queued for retry on the next "
                "compose teardown; if it persists, operator cleanup: "
                "`docker %s ls --filter label=com.docker.compose.project=%s` then rm (H10)",
                rollout_id, project_name, kind, exc, kind, project_name,
            )
            return
        if remaining:
            self._pending_resource_prune.add(project_name)   # H10 — queue for retry
            LOGGER.warning(
                "compose.destroy rollout=%s project=%s — %d %s(s) still present after prune; "
                "best-effort DISK cleanup (no capacity impact). Queued for retry on the next "
                "compose teardown; if it persists, operator cleanup: "
                "`docker %s ls --filter label=com.docker.compose.project=%s` then rm (H10)",
                rollout_id, project_name, remaining, kind, kind, project_name,
            )

    async def _retry_pending_resource_prunes(self) -> None:
        """audit H10 — bounded GC/retry for durable-DISK cleanup a prior teardown could not
        confirm. For each queued project name that is NOT currently live (no ``_compose_projects``
        entry — so a same-name acquire hasn't reclaimed it), re-attempt the named-volume + network
        prune; drop it from the queue only once BOTH are confirmed gone. Best-effort + never
        raises (disk cleanup is decoupled from capacity). Piggybacks on the next compose teardown
        so no periodic task is needed and the hot compose-up path is untouched."""
        for name in list(self._pending_resource_prune):
            async with self._lock:
                if name in self._compose_projects:
                    # A live (possibly reused-name) project owns it now — its resources are in use;
                    # don't prune. Drop from the queue: this name is no longer an orphan.
                    self._pending_resource_prune.discard(name)
                    continue
            proj_filter = {"label": f"com.docker.compose.project={name}"}
            cleaned = True
            for list_fn, remove_fn in (
                (lambda pf=proj_filter: self._client.volumes.list(filters=pf),
                 lambda r: r.remove(force=True)),
                (lambda pf=proj_filter: self._client.networks.list(filters=pf),
                 lambda r: r.remove()),
            ):
                try:
                    for res in await asyncio.to_thread(list_fn):
                        with suppress(docker.errors.DockerException):
                            await asyncio.to_thread(remove_fn, res)
                    if await asyncio.to_thread(list_fn):
                        cleaned = False
                except docker.errors.DockerException:
                    cleaned = False
            if cleaned:
                self._pending_resource_prune.discard(name)
                LOGGER.info(
                    "compose.destroy — retry pruned leaked disk resources for project=%s (H10)",
                    name,
                )

    async def _create_with_retry(
        self, run_kwargs: dict[str, Any], *, rollout_id: str, sysbox: bool,
        retry_deadline: float | None = None,
    ) -> Any:
        """Create the container under the create gate, retrying a bounded number
        of times on a *transient busy-daemon* fault.

        The motivating case is sysbox: under a create burst, ``sysbox-fs``
        pre-register momentarily overloads and the ``docker run`` returns a
        ``500 … DeadlineExceeded``. That's a transient (``_is_retryable_create_error``
        — 5xx / timeout), not a hard ceiling; a retry a second or two later
        succeeds. This wraps the ``_create_gate`` block (Issue #18) with an
        exponential backoff-with-jitter retry so a recovered transient returns a
        container instead of killing the acquire. A clean dead-daemon
        ``ConnectionError`` and 4xx request faults are NOT retried (they re-raise).

        Four correctness points this method is responsible for:

        * **AIMD accounting.** Record at most ONE health error per acquire — for
          any acquire that hit *any* ``_is_node_health_error`` fault, not one per
          attempt and not only on final give-up. ``HealthAimdController``
          edge-triggers on *new* errors, so N records = N halvings = the node
          collapses to the floor from a single bad burst. A create that retries
          then succeeds still records one signal (the node WAS saturated → future
          admits should throttle). A down-daemon ConnectionError records (feeds
          AIMD) even though it isn't retried in place.
        * **No duplicate-container leak (fail closed).** A timed-out / 5xx'd create
          may have actually spawned the container without the client seeing the
          response. Before every retry we reap any container carrying this
          acquire's unique ``xrlenv.rollout_id`` label (``_reap_rollout_orphans``).
          If that reap cannot be *confirmed* clean (the list failed, or a found
          orphan would not remove) we **do not retry** — recreating on top of a
          possibly-live orphan would leak a duplicate with no control-plane
          session. We fail closed and surface the last transient; the caller
          re-submits with a fresh rollout_id and raw-GC reaps the orphan.
        * **Bounded total wait.** Retries are capped three ways: the attempt count,
          the per-retry ceiling, and a hard wall-clock total — the smaller of
          ``_HEALTH_RETRY_TOTAL_CAP_S`` and the caller's remaining acquire wire
          budget (``retry_deadline``, anchored at acquire entry). A fail-fast
          caller therefore fails fast on the node too, never replying to a command
          the control plane has already timed out.

        The create gate is *released during backoff* (each attempt re-enters it),
        so other acquires proceed while this one waits. ``sysbox=True`` routes
        through the lower sysbox create semaphore (approach C — prevention).
        """
        recorded_health = False
        last_exc: Exception | None = None
        started = time.monotonic()
        for attempt in range(_HEALTH_RETRY_MAX + 1):
            # A prior attempt (attempt > 0) failed with a transient (timeout /
            # 5xx). Docker may have created the container even though the client
            # never got the response — reap any orphan wearing THIS acquire's
            # unique rollout_id label before recreating. If the reap can't be
            # confirmed clean, fail CLOSED (don't stack a possible duplicate).
            # Short-circuit: the reap runs only for attempt > 0.
            if attempt > 0 and not await self._reap_rollout_orphans(rollout_id):
                LOGGER.error(
                    "raw-container.acquire rollout=%s: could not confirm the "
                    "prior create attempt left no orphan; failing closed "
                    "instead of risking a duplicate container",
                    rollout_id,
                )
                raise last_exc or RuntimeError(
                    "create-retry aborted: unconfirmed orphan reap",
                )
            try:
                # Issue #18: the create gate bounds how many ``docker run`` calls
                # hit this node's daemon at once (a lower cap for slow sysbox
                # creates). The image pull already completed above, so the gate
                # serialises only the create burst.
                async with self._create_gate(sysbox=sysbox):
                    _create_t0 = time.monotonic()
                    container = await self._run_with_name_reclaim(
                        run_kwargs, rollout_id=rollout_id,
                    )
                    # Stage 1: time the create call itself (inside the gate,
                    # excluding gate-wait) — the smooth node-saturation signal.
                    self._health.record_create(time.monotonic() - _create_t0)
                    # A 2xx from ``containers.run`` is NOT proof the container
                    # runs: under snapshotter strain dockerd hands back a
                    # container whose RW layer is nil. Verify before returning,
                    # so the fault is retried here instead of surfacing as the
                    # consumer's first ``exec`` failing 409 "is not running".
                    fault = await asyncio.to_thread(
                        _container_start_fault, container,
                    )
                    if fault is not None:
                        raise ContainerNotStartedError(
                            f"{fault}; image={run_kwargs.get('image')!r}",
                        )
                return container
            except (
                docker.errors.DockerException,
                requests.exceptions.RequestException,
            ) as exc:
                last_exc = exc
                # Feed the AIMD limiter once for any acquire that hit a saturation
                # signal (5xx / timeout / transport) — even a non-retryable
                # dead-daemon ConnectionError should throttle future admits. A 4xx
                # request fault is NOT health (counting it would let one duplicate
                # name collapse the whole admission limit — the prod regression
                # this guards against).
                if _is_node_health_error(exc) and not recorded_health:
                    self._health.record_docker_error(
                        is_timeout=_is_timeout_exc(exc),
                    )
                    recorded_health = True
                # Retry ONLY a transient busy-daemon fault. A 409 / 404 request
                # fault and a clean dead-daemon ConnectionError re-raise now.
                if not _is_retryable_create_error(exc):
                    raise
                if attempt >= _HEALTH_RETRY_MAX:
                    raise  # retries exhausted — surface the last transient error
                # Exponential backoff, per-retry ceiling, jittered *down* (0.75x
                # to 1.0x) so a thundering herd of concurrent retries
                # desynchronises and no single wait exceeds the cap.
                backoff = min(
                    _HEALTH_RETRY_BASE_S * (2 ** attempt), _HEALTH_RETRY_CAP_S,
                ) * random.uniform(0.75, 1.0)
                # Total-wait ceiling: the smaller of the hard wall-clock cap and
                # the caller's remaining acquire wire budget. Give up now
                # (surfacing the last transient) rather than sleep past either —
                # never reply to a command the control plane has already timed out.
                now = time.monotonic()
                budget_ok = (now - started) + backoff <= _HEALTH_RETRY_TOTAL_CAP_S
                wire_ok = retry_deadline is None or (now + backoff) <= retry_deadline
                if not (budget_ok and wire_ok):
                    raise
                LOGGER.warning(
                    "raw-container.acquire rollout=%s: transient node-saturation "
                    "create fault (%s: %s); retry %d/%d in %.1fs",
                    rollout_id, type(exc).__name__, exc,
                    attempt + 1, _HEALTH_RETRY_MAX, backoff,
                )
                await asyncio.sleep(backoff)
        # The loop always returns on success or raises on give-up; this is
        # unreachable and only satisfies the type checker's exhaustiveness.
        raise last_exc or RuntimeError("create-retry loop exited without result")

    async def _reap_rollout_orphans(self, rollout_id: str) -> bool:
        """Force-remove any container carrying this acquire's unique
        ``xrlenv.rollout_id`` label, returning whether the node is *confirmed*
        clean of orphans for this rollout.

        Called before a create retry: a create that timed out (or 5xx'd) may have
        actually spawned the container without the client seeing the response, so
        recreating would leak a duplicate. The label is 1:1 with this acquire, so
        anything wearing it is an orphan from a failed attempt and is safe to
        remove.

        Returns ``True`` only when we can *prove* no orphan remains: the list
        succeeded and every found container was removed (or none existed / raced
        away). Returns ``False`` on any uncertainty — the list call failed (unknown
        state), or a found orphan would not remove — so the caller can fail closed
        rather than stack a duplicate on top of a possibly-live container. The
        coordinator's raw-GC by-label sweep is the durable backstop for whatever we
        could not remove here."""
        try:
            orphans = await asyncio.to_thread(
                self._client.containers.list,
                all=True,
                filters={"label": f"xrlenv.rollout_id={rollout_id}"},
            )
        except docker.errors.DockerException:
            LOGGER.warning(
                "raw-container.acquire rollout=%s: could not list orphans before "
                "create retry — cannot confirm clean, failing closed",
                rollout_id, exc_info=True,
            )
            return False
        clean = True
        for orphan in orphans:
            oid = str(getattr(orphan, "id", ""))[:12]
            try:
                await asyncio.to_thread(orphan.remove, force=True)
                LOGGER.warning(
                    "raw-container.acquire rollout=%s: reaped orphan container %s "
                    "from a timed-out create attempt before retry",
                    rollout_id, oid,
                )
            except docker.errors.NotFound:
                pass  # already gone — nothing to reap
            except docker.errors.DockerException:
                # An orphan we KNOW exists but could not remove — recreating now
                # would leak a duplicate. Mark unclean so the caller fails closed.
                LOGGER.warning(
                    "raw-container.acquire rollout=%s: failed to reap orphan %s "
                    "before create retry — failing closed", rollout_id, oid,
                    exc_info=True,
                )
                clean = False
        return clean

    async def _run_with_name_reclaim(
        self, run_kwargs: dict[str, Any], *, rollout_id: str,
    ) -> Any:
        """``containers.run`` with one-shot orphaned-name reclaim.

        A 409 "name already in use" almost always means an xrlenv orphan
        is holding the requested name — a prior rollout's container whose
        destroy never propagated (control-plane restart, or a destroy RPC
        that timed out under node I/O saturation), or a duplicate
        submission of the same harbor ``session_id`` (the container name
        is derived from it, so it is NOT globally unique). The control
        plane passes the name straight through, so without this the
        create just 409s and the rollout fails even though the node could
        serve it. We reclaim the name and retry once — but ONLY when the
        holder is an xrlenv-managed raw container (carries our
        ``xrlenv.session_kind=raw`` label); a foreign container is never
        touched, and a second conflict propagates unretried."""
        try:
            return await asyncio.to_thread(
                self._client.containers.run, **run_kwargs,
            )
        except docker.errors.APIError as exc:
            name = run_kwargs.get("name")
            if not name or not _is_name_conflict(exc):
                raise
            reclaimed = await asyncio.to_thread(
                self._reclaim_orphaned_name, str(name),
            )
            if not reclaimed:
                raise
            LOGGER.warning(
                "raw-container.acquire rollout=%s reclaimed orphaned "
                "container name %r (removed a stale xrlenv container) and "
                "retrying create once",
                rollout_id, name,
            )
            return await asyncio.to_thread(
                self._client.containers.run, **run_kwargs,
            )

    def _reclaim_orphaned_name(self, name: str) -> bool:
        """Remove the container currently holding ``name`` iff it is an
        xrlenv-managed raw container. Returns ``True`` when the name is
        free to reuse (removed, or already gone), ``False`` when it must
        not be reclaimed (foreign container, or removal failed) so the
        caller surfaces the original 409. Synchronous — runs in a thread.
        """
        try:
            existing = self._client.containers.get(name)
        except docker.errors.NotFound:
            # Raced away between the 409 and now — the name is free.
            return True
        except docker.errors.DockerException:
            return False
        labels = getattr(existing, "labels", None) or {}
        if labels.get("xrlenv.session_kind") != "raw":
            # NOT ours — never remove a foreign container. Surface the 409.
            LOGGER.warning(
                "raw-container.acquire: name %r is held by a non-xrlenv "
                "container %s; refusing to reclaim",
                name, str(getattr(existing, "id", ""))[:12],
            )
            return False
        try:
            existing.remove(force=True)
        except docker.errors.NotFound:
            return True
        except docker.errors.DockerException:
            LOGGER.warning(
                "raw-container.acquire: failed to remove stale xrlenv "
                "container %s holding name %r",
                str(getattr(existing, "id", ""))[:12], name, exc_info=True,
            )
            return False
        return True

    async def _ensure_image_present(
        self, image: str, *, strict: bool = True,
        deadline_s: float | None = None,
    ) -> None:
        """Verify image presence on the local docker daemon.

        ``strict=True`` (legacy P1.7.A.1 contract): consumer /
        harness pulls outside this RPC. Surfaces the missing-image
        case as a fast failure with a clear message.

        ``strict=False`` (P1.7.B.2 default UX): when an
        :class:`xrlenv.node.image_cache.ImageCacheManager` is
        wired (case-1 already constructs one for sandboxes; case
        2/3 inherits the same instance), delegate to
        ``image_cache.ensure_present(image, deadline_s=...)`` —
        builder-lookup-then-pull, eviction-aware, the same
        primitive case-1 sandboxes use. When no ImageCacheManager
        is wired (legacy fixtures, hand-built test doubles), fall
        back to the strict path so behaviour is preserved.
        """
        if not strict and self._image_cache is not None:
            # Delegate to the cache layer. ensure_present handles
            # registry pulls, builder-driven local builds, eviction
            # under disk pressure, and idempotent no-ops when
            # already cached. Surfaces TimeoutError /
            # OutOfDiskAfterEviction with clear messages.
            # ``deadline_s=None`` lets ensure_present apply the
            # cache manager's configured default (currently 600 s);
            # a numeric override flows through from the consumer's
            # ``acquire_timeout_s`` (issue #12).
            try:
                await self._image_cache.ensure_present(
                    image, deadline_s=deadline_s,
                )
            except XRLEnvError:
                raise
            except Exception as exc:
                raise XRLEnvError(
                    f"raw-container acquire: ImageCacheManager."
                    f"ensure_present({image!r}) failed: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            return

        # Strict path (default for opt-out, fallback for
        # no-cache fixtures).
        try:
            await asyncio.to_thread(self._client.images.get, image)
        except Exception as exc:
            # docker-py raises ImageNotFound, but we don't want to
            # import docker here (the manager is duck-typed). The
            # message is what the operator sees.
            raise XRLEnvError(
                f"raw-container acquire: image {image!r} not "
                f"present on this node. ensure_image_present=False "
                f"(strict mode) — pull or build the image before "
                f"acquiring, or pass ensure_image_present=True "
                f"(default) to route through the cluster's image-"
                f"distribution layer.",
            ) from exc

    # ── Exec ────────────────────────────────────────────────────────────────

    async def exec(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Run ``cmd`` inside the container via docker exec.

        Batched (returns full output after wait). P1.7.A.2 will add
        a streaming variant for swebench's 30+ min test runs.

        Ownership: rejects if the ``container_id`` is not registered
        OR if the registered owner's ``rollout_id`` doesn't match
        the caller's. Returned dict shape mirrors the
        ``ExecReply`` proto field set: ``{exit_code, stdout, stderr,
        timed_out}``.
        """
        self._assert_owner(rollout_id, container_id)
        container = await asyncio.to_thread(
            self._client.containers.get, container_id,
        )

        kwargs: dict[str, Any] = {"demux": True}
        if cwd:
            kwargs["workdir"] = cwd
        if env:
            kwargs["environment"] = env
        if user:
            kwargs["user"] = user

        # docker-py's ``exec_run`` is blocking; offload to a thread.
        # ``demux=True`` returns ``(stdout, stderr)`` separately.
        # asyncio.wait_for handles the timeout — we don't get a
        # native docker-side timeout, so we race against the wall
        # clock and surface ``timed_out=True`` on cancel.
        #
        # A misaligned attach stream makes docker-py's demuxer raise
        # ``ValueError("N is not a valid stream")`` — transient (a fresh exec
        # opens a new stream), so resync-retry rather than letting it propagate
        # as a fatal RPC error (see ``_EXEC_DEMUX_RETRIES``). The command is a
        # one-shot batched exec, so re-running it is safe.
        result = None
        for _attempt in range(_EXEC_DEMUX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(container.exec_run, cmd, **kwargs),
                    timeout=timeout_s,
                )
                break
            except TimeoutError:
                LOGGER.warning(
                    "raw-container.exec timed out rollout=%s container=%s "
                    "cmd=%r timeout_s=%.1f",
                    rollout_id, container_id[:12], cmd, timeout_s,
                )
                return {
                    "exit_code": -1,
                    "stdout": b"",
                    "stderr": b"",
                    "timed_out": True,
                }
            except ValueError as exc:
                if "not a valid stream" not in str(exc) or _attempt >= _EXEC_DEMUX_RETRIES:
                    raise
                LOGGER.warning(
                    "raw-container.exec demux stream corruption (%s) rollout=%s "
                    "container=%s cmd=%r; resync-retry %d/%d",
                    exc, rollout_id, container_id[:12], cmd,
                    _attempt + 1, _EXEC_DEMUX_RETRIES,
                )

        exit_code = int(getattr(result, "exit_code", 0) or 0)
        # ``demux=True`` returns a 2-tuple; ``demux=False`` returns
        # combined bytes. We always set demux=True above.
        output = getattr(result, "output", (b"", b""))
        stdout = output[0] or b""
        stderr = output[1] or b""
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
        }

    # ── Egress restriction (spec 07) ────────────────────────────────────────

    async def apply_egress(
        self,
        *,
        rollout_id: str,
        container_id: str,
        allowlist: EgressAllowlist,
        dns_resolver: str | None = None,
    ) -> None:
        """Restrict a running container's egress to ``allowlist``.

        Generic mechanism — the caller decides when/what to allow. An empty
        ``allowlist`` blocks all external egress (loopback stays up). The
        allowlist is compiled here (node-side: the rule set is the single
        source of truth) and installed into the container's netns by the
        enforcer.

        Safety:

        - Ownership: rejected unless ``container_id`` is registered to the
          caller's ``rollout_id`` (same model as :meth:`exec`).
        - **Private netns only**: refuses ``network_mode=host`` /
          ``container:<id>`` — entering such a netns and flushing OUTPUT
          would rewrite the node host's (or a sibling's) chain.
        - **Fail-closed**: if the enforcer fails partway, the
          half-restricted (= under-restricted) container is destroyed, so a
          restricted task can never run on open/partial egress.

        Raises :class:`~xrlenv.errors.XRLEnvError` on a shared netns / no live
        pid; the enforcer error propagates (after teardown) on apply failure.
        """
        self._assert_owner(rollout_id, container_id)
        # Compile first (pure) — a bad allowlist fails before touching docker.
        program = compile_egress_rules(allowlist, dns_resolver=dns_resolver)
        container = await asyncio.to_thread(
            self._client.containers.get, container_id,
        )
        await asyncio.to_thread(container.reload)
        host_config = container.attrs.get("HostConfig") or {}
        network_mode = str(host_config.get("NetworkMode") or "")
        if is_shared_netns(network_mode):
            raise XRLEnvError(
                f"apply_egress refuses container {container_id[:12]!r} with "
                f"network_mode={network_mode!r}: egress rules are installed via "
                "nsenter into the target's netns, and a host / shared "
                "(container:<id>) netns would rewrite the node host's (or a "
                "sibling container's) OUTPUT chain, not the sandbox's.",
            )
        if container_can_escape_egress(host_config):
            # The boundary only holds if the workload can't flush the rules.
            raise XRLEnvError(
                f"apply_egress refuses container {container_id[:12]!r}: it is "
                "privileged or holds CAP_NET_ADMIN, so the agent inside could "
                "flush the OUTPUT chain — egress restriction would be a false "
                "anti-cheat signal. Acquire restricted tasks without "
                "privileged / NET_ADMIN.",
            )
        pid = int((container.attrs.get("State") or {}).get("Pid") or 0)
        if pid <= 0:
            raise XRLEnvError(
                f"apply_egress: container {container_id[:12]!r} has no live pid "
                "(it may have exited); cannot enter its netns",
            )
        try:
            await self._egress_enforcer.apply(container_pid=pid, program=program)
        except Exception:
            # Fail-closed: a partial program is an under-restricted container.
            LOGGER.error(
                "raw-container.apply_egress enforcer FAILED rollout=%s "
                "container=%s; destroying (fail-closed)",
                rollout_id, container_id[:12],
            )
            with suppress(Exception):
                await self.destroy(
                    rollout_id=rollout_id, container_id=container_id, force=True,
                )
            raise
        LOGGER.info(
            "raw-container.apply_egress rollout=%s container=%s rules=%d",
            rollout_id, container_id[:12], len(allowlist.rules),
        )

    # ── Archives (P1.7.A.2) ─────────────────────────────────────────────────

    async def put_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        target_dir: str,
        tarball: bytes,
    ) -> None:
        """Extract ``tarball`` into ``target_dir`` inside the
        container via ``docker.containers.put_archive``.

        Direct docker SDK call — no stub involvement, no
        ``mkdir`` / wipe pre-step. The case-1 ``PutArchiveCommand``
        does mkdir + optional ``rm -rf`` as root for the verifier-
        asset injection path (audit H1). The raw path is case-2/3
        where the harness manages its own target-dir layout —
        swebench ``exec_run("mkdir -p /tmp")`` first; harbor's
        ``upload_dir`` relies on the image's existing layout. We
        do NOT mkdir on the harness's behalf.

        **Contract**: ``target_dir`` MUST already exist in the
        container. If it doesn't, docker's ``put_archive``
        returns False (or the daemon raises) and we surface
        XRLEnvError. Caller's responsible for the tar bytes;
        docker auto-detects gzip.
        """
        self._assert_owner(rollout_id, container_id)
        container = await asyncio.to_thread(
            self._client.containers.get, container_id,
        )
        # Hold the same bulk-transfer gate as ``get_archive``: a big
        # golden/source ``put_archive`` is symmetric daemon+IO pressure,
        # and bounding both directions together keeps the node's total
        # concurrent bulk-copy load capped under a multi-tenant burst.
        async with self._archive_gate():
            ok = await asyncio.to_thread(
                container.put_archive, target_dir, tarball,
            )
        if not ok:
            raise XRLEnvError(
                f"raw-container put_archive: docker returned False "
                f"for container={container_id[:12]!r} target={target_dir!r}",
            )

    async def get_archive_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
    ) -> AsyncIterator[bytes]:
        """Tar ``source_path`` inside the container and yield the
        archive as a sequence of ``bytes`` chunks.

        **This is the node-lost fix.** docker-py's ``get_archive``
        returns ``(stream, stat)`` where ``stream`` is a *lazy*
        generator (``response.iter_content`` → blocking socket reads):
        the tar body is pulled only as the iterator advances. The old
        implementation did ``b"".join(bits)`` in the coroutine body —
        i.e. it drained the whole (up to hundreds-of-MB) tar on the
        single asyncio event-loop thread with no ``await``. While that
        ran, the node could send NOTHING — the heartbeat task couldn't
        wake and the outbound pump couldn't drain — so the control
        plane saw >60 s of silence and marked the node ``lost``, sealing
        every in-flight rollout there. EvoClaw copies the whole
        ``/testbed`` out of every eval container, so it hit this every
        wave; tb2.1/SWE grade from a few-KB file and never did.

        Here every blocking read is a one-chunk ``asyncio.to_thread``
        hop (:func:`_next_archive_chunk`), so the event loop runs
        between chunks and the heartbeat keeps flowing regardless of
        archive size. The whole transfer holds ``_archive_gate`` so a
        burst of concurrent copies is bounded, not a thundering herd.
        Peak in-flight bytes per transfer is one docker chunk rather
        than the entire tarball; the wire dispatch re-slices at
        ``ARCHIVE_CHUNK_BYTES`` for its NodeMsg size bound.
        """
        self._assert_owner(rollout_id, container_id)
        container = await asyncio.to_thread(
            self._client.containers.get, container_id,
        )
        async with self._archive_gate():
            # ``container.get_archive`` returns fast — it only opens the
            # ``stream=True`` response + reads the stat header; the body
            # is streamed lazily below. We take docker-py's default
            # ~2 MiB chunking (no ``chunk_size`` arg) so the call matches
            # both real docker-py and the test fakes' signature; the
            # per-chunk size only sets the to_thread hop granularity.
            bits, _stat = await asyncio.to_thread(
                container.get_archive, source_path,
            )
            it = iter(bits)
            cap = self._max_get_archive_relay_bytes
            sent = 0
            while True:
                chunk = await asyncio.to_thread(_next_archive_chunk, it)
                if chunk is _ARCHIVE_STREAM_END:
                    break
                if not chunk:
                    continue
                sent += len(chunk)
                # Plane-split guardrail: refuse to relay a whole
                # container filesystem through the control plane. We
                # count as we stream (a directory's tar size isn't
                # knowable up front) and fail THIS transfer the moment
                # it crosses the cap — the partial bytes already sent
                # are discarded by the control-plane collector on the
                # FAILED reply, and the rollout is untouched.
                if cap and sent > cap:
                    raise ArchiveTooLarge(
                        f"get_archive of {source_path!r} exceeded the "
                        f"{cap}-byte control-plane relay cap (streamed "
                        f"{sent} bytes). Large artifacts must not transit "
                        f"the control plane; this single transfer was "
                        f"refused (the rollout is unaffected). Use the "
                        f"artifact-export primitive for bulk capture, or "
                        f"raise XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES.",
                    )
                yield chunk

    async def get_archive(
        self,
        *,
        rollout_id: str,
        container_id: str,
        source_path: str,
    ) -> bytes:
        """Bytes-returning convenience wrapper over
        :meth:`get_archive_stream`.

        Retained for callers/tests that want the whole tarball in one
        object. Unlike the pre-fix version this still drains the socket
        off the event loop (one ``to_thread`` hop per chunk inside the
        stream), so even this buffered path can't freeze the heartbeat.
        The wire dispatch (``grpc_link``) uses the streaming method
        directly and never buffers the whole tarball."""
        chunks: list[bytes] = []
        async for chunk in self.get_archive_stream(
            rollout_id=rollout_id,
            container_id=container_id,
            source_path=source_path,
        ):
            chunks.append(chunk)
        return b"".join(chunks)

    # ── Streaming exec (P1.7.A.2) ────────────────────────────────────────────

    async def exec_stream(
        self,
        *,
        rollout_id: str,
        container_id: str,
        cmd: list[str],
        timeout_s: float = 1800.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        heartbeat_interval_s: float = 10.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator yielding per-chunk dicts as
        ``container.exec_run(stream=True)`` produces output.

        Each yielded dict has the ``ContainerExecChunk`` shape:
        ``{stdout: bytes, stderr: bytes, done: bool, exit_code:
        int, timed_out: bool}``. Exactly one terminator chunk
        with ``done=True`` is yielded last; the chunk's
        ``exit_code`` reports the final exit (or ``-1`` on
        timeout). Consumers iterate until they see ``done=True``.

        Why a generator + chunks rather than batched: long-
        running runs (swebench's 30+ min eval scripts; tb2's
        1-2 hour tasks) need real-time output to keep idle
        timeouts from dropping the path AND to avoid buffering
        the full stdout/stderr in node memory before sending.
        """
        self._assert_owner(rollout_id, container_id)
        container = await asyncio.to_thread(
            self._client.containers.get, container_id,
        )

        # docker-py's ``exec_run(stream=True, demux=True)`` returns
        # ``(exit_code, generator)`` where exit_code is None until
        # exhausted and the generator yields ``(stdout_chunk,
        # stderr_chunk)`` tuples. Either chunk may be None per
        # iteration. Exit code becomes available after the
        # generator finishes; we read it via the lower-level
        # ``exec_inspect`` because docker-py doesn't expose it
        # cleanly otherwise on the streaming path.
        kwargs: dict[str, Any] = {"stream": True, "demux": True}
        if cwd:
            kwargs["workdir"] = cwd
        if env:
            kwargs["environment"] = env
        if user:
            kwargs["user"] = user

        # Build the exec via the low-level API so we can read
        # ``exit_code`` after streaming finishes (the high-level
        # ``container.exec_run`` is convenient but loses the
        # exec_id we need for inspect).
        api = self._client.api
        exec_create_kwargs: dict[str, Any] = {}
        if cwd:
            exec_create_kwargs["workdir"] = cwd
        if env:
            exec_create_kwargs["environment"] = env
        if user:
            exec_create_kwargs["user"] = user
        exec_info = await asyncio.to_thread(
            api.exec_create, container.id, cmd, **exec_create_kwargs,
        )
        exec_id = exec_info["Id"]

        # ``exec_start(stream=True, demux=True)`` returns an
        # iterator of ``(stdout_chunk, stderr_chunk)`` tuples.
        # We wrap consumption in ``asyncio.to_thread`` per chunk
        # to avoid blocking the event loop on long polls — but
        # iteration itself is sync, so the cleanest pattern is:
        # spin up a thread that drains the iterator into an
        # ``asyncio.Queue``; the async generator yields off the
        # queue. Per-chunk thread-hop avoids starvation.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        _SENTINEL = object()

        def _drain() -> None:
            try:
                stream_iter = api.exec_start(
                    exec_id, stream=True, demux=True,
                )
                for chunk in stream_iter:
                    # ``chunk`` is a 2-tuple (stdout, stderr)
                    # with either side possibly None.
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:  # pragma: no cover — defensive
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        drain_task = asyncio.create_task(asyncio.to_thread(_drain))

        # Audit Raw-Stream-M1 closure: emit empty-payload
        # heartbeat chunks every ``heartbeat_interval_s`` of
        # silence (default 10s — well below the consumer-side
        # ``_send_and_stream`` 30s per-chunk timeout). Without
        # this, a quiet long-running exec (a compile / test that
        # takes >30s without printing) would trip the consumer-
        # side timer and surface as a false-positive wedge. The
        # heartbeats keep the stream alive at every hop (control
        # plane queue, gRPC keepalive, intermediate NAT/LB) and
        # reset chunk-side idle timers.
        timed_out = False
        try:
            deadline = loop.time() + timeout_s
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    timed_out = True
                    break
                wait = min(remaining, heartbeat_interval_s)
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=wait,
                    )
                except TimeoutError:
                    # Distinguish stream-deadline timeout from
                    # chunk-silence heartbeat. If we still have
                    # whole-stream budget left, this is a
                    # heartbeat-tick: emit an empty chunk and
                    # keep going.
                    if loop.time() < deadline:
                        yield {
                            "stdout": b"",
                            "stderr": b"",
                            "done": False,
                            "exit_code": 0,
                            "timed_out": False,
                        }
                        continue
                    timed_out = True
                    break
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    LOGGER.warning(
                        "raw-container.exec_stream drain raised: %r",
                        item,
                    )
                    break
                stdout_chunk, stderr_chunk = item
                yield {
                    "stdout": stdout_chunk or b"",
                    "stderr": stderr_chunk or b"",
                    "done": False,
                    "exit_code": 0,
                    "timed_out": False,
                }
        finally:
            # Always wait for the drain task so we don't leak it.
            # Cancellation is safe — exec_start blocks in C, but
            # cancelling the asyncio.to_thread wrapper just
            # detaches from us; the docker socket eventually
            # closes when the exec completes.
            if not drain_task.done():
                # On timeout, try to kill the exec so the docker
                # daemon stops piping bytes and the drain thread
                # exits. Best-effort.
                with suppress(Exception):
                    await asyncio.to_thread(
                        api.exec_resize, exec_id, height=1, width=80,
                    )
                drain_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await drain_task

        # Pull final exit code from inspect — only meaningful
        # post-stream. On timeout we mark exit_code=-1.
        if timed_out:
            exit_code = -1
        else:
            try:
                inspect = await asyncio.to_thread(
                    api.exec_inspect, exec_id,
                )
                exit_code = int(inspect.get("ExitCode") or 0)
            except Exception:
                exit_code = 0

        # Terminator chunk.
        yield {
            "stdout": b"",
            "stderr": b"",
            "done": True,
            "exit_code": exit_code,
            "timed_out": timed_out,
        }

    # ── Destroy ─────────────────────────────────────────────────────────────

    async def destroy(
        self, *, rollout_id: str, container_id: str, force: bool = True,
    ) -> None:
        """Remove the container and deregister.

        Symmetric ownership enforcement with :meth:`exec` — refuses
        unregistered ``container_id`` (caller can't destroy a
        container they don't own by guessing its id) AND refuses
        when the registered owner is a different rollout.
        Idempotent only on the *docker-side-already-gone* case for
        registered records (the harness may have ``docker rm``'d
        the container itself between our check and the docker
        round-trip — that's a benign race, not a security issue).
        """
        async with self._lock:
            record = self._records.get(container_id)
        if record is None:
            raise XRLEnvError(
                f"raw-container destroy: container "
                f"{container_id[:12]!r} not registered on this "
                f"node. Caller cannot destroy a container they "
                f"don't own (was it acquired? was it already "
                f"destroyed?).",
            )
        if record.rollout_id != rollout_id:
            raise XRLEnvError(
                f"raw-container destroy: rollout {rollout_id!r} "
                f"does not own container {container_id[:12]!r} "
                f"(owner: {record.rollout_id!r}).",
            )

        # Issue #18 fix #4: serialise the docker-daemon work behind
        # the per-node destroy semaphore. The ownership check above runs
        # unbounded so a queued caller still gets ``XRLEnvError``
        # immediately on a bad rollout_id. A sysbox teardown routes through
        # the tighter sysbox destroy gate (concurrent FUSE unmounts wedge
        # sysbox-fs — see _sysbox_destroy_semaphore).
        _sysbox = bool(record.container_runtime) and (
            record.container_runtime != "runc"
        )
        async with self._destroy_gate(sysbox=_sysbox):
            try:
                container = await asyncio.to_thread(
                    self._client.containers.get, container_id,
                )
            except Exception:
                # Docker side is gone but we still tracked it — benign
                # race (harness ``docker rm``-ed between our ownership
                # check and now). Deregister and exit cleanly.
                async with self._lock:
                    self._records.pop(container_id, None)
                # P2 — the container is gone; reclaim its cores.
                if record.cpuset:
                    self._core_ledger.release(record.cpuset)
                # Release the image refcount acquired at spawn (symmetric
                # with acquire) so the cache can evict it once cold.
                if self._image_cache is not None:
                    self._image_cache.release(record.image)
                return

            try:
                await asyncio.to_thread(container.remove, force=force)
            except asyncio.CancelledError:
                # The destroy was abandoned mid-flight — the control-plane
                # wire ceiling fired (the daemon couldn't finish the
                # remove in time) or the node is shutting down. A daemon
                # that can't complete a remove is saturated; feed that to
                # the health signal so AIMD throttles new admissions here.
                # Pre-fix, destroy-path stalls fed nothing, so a node
                # whose destroys hung kept (even grew) its admission limit
                # while the create path looked momentarily fine — the
                # health signal pointed the wrong way (the destroy-stall
                # inversion seen in the 2026-06-09 analysis).
                self._health.record_docker_error(is_timeout=True)
                raise
            except Exception as exc:  # pragma: no cover — defensive
                # A genuine daemon fault (timeout / 5xx / transport) is a
                # saturation signal; a 4xx (e.g. already-gone) is not.
                if _is_node_health_error(exc):
                    self._health.record_docker_error(
                        is_timeout=_is_timeout_exc(exc),
                    )
                LOGGER.warning(
                    "raw-container.destroy remove failed rollout=%s "
                    "container=%s err=%r",
                    rollout_id, container_id[:12], exc,
                )
            finally:
                async with self._lock:
                    self._records.pop(container_id, None)
                # P2 — return the pinned cores to the node ledger.
                if record.cpuset:
                    self._core_ledger.release(record.cpuset)
                # Release the image refcount acquired at spawn.
                if self._image_cache is not None:
                    self._image_cache.release(record.image)

    # ── Privileged destroy (reconciler-only) ────────────────────────────────

    async def force_destroy(self, *, container_id: str) -> None:
        """Privileged ``docker rm -f`` that bypasses the
        ownership check :meth:`destroy` enforces.

        Used **only** by the raw-GC reconciler to clean up
        node-only orphans (containers labeled
        ``xrlenv.session_kind=raw`` that the coordinator doesn't
        know about — typically CP-restart leftovers). NOT
        consumer-reachable: the matching spec-21 command isn't
        exposed via ``rollout_control.proto``.

        Idempotent: missing container is a no-op (the orphan's
        already gone). The in-memory ``_records`` map is also
        cleaned up if the id happens to be there.
        """
        # Issue #18 fix #4: shares the destroy concurrency gate with
        # :meth:`destroy` so a reconciler sweep can't independently
        # thrash the docker daemon while consumer destroys are also
        # queued.
        async with self._destroy_gate():
            try:
                container = await asyncio.to_thread(
                    self._client.containers.get, container_id,
                )
            except Exception:
                # Already gone — clean up any registry entry that
                # happens to exist and exit.
                async with self._lock:
                    self._records.pop(container_id, None)
                return

            # P2 — recover the container's pinned cores so the orphan
            # path reclaims them too. Prefer the in-memory record;
            # fall back to docker's HostConfig for a true orphan (the
            # reconciler's CP-restart-leftover case).
            record = self._records.get(container_id)
            held_image = record.image if record is not None else None
            if record is not None and record.cpuset:
                cpuset_cores: tuple[int, ...] = record.cpuset
            else:
                attrs = getattr(container, "attrs", None) or {}
                cpuset_cores = _parse_cpuset(
                    attrs.get("HostConfig", {}).get("CpusetCpus") or "",
                )

            try:
                await asyncio.to_thread(container.remove, force=True)
            except asyncio.CancelledError:
                # Reconciler-path destroy abandoned mid-flight (wire
                # ceiling / shutdown) — same saturation signal as the
                # consumer ``destroy`` path above.
                self._health.record_docker_error(is_timeout=True)
                raise
            except Exception as exc:  # pragma: no cover — defensive
                if _is_node_health_error(exc):
                    self._health.record_docker_error(
                        is_timeout=_is_timeout_exc(exc),
                    )
                LOGGER.warning(
                    "raw-container.force_destroy remove failed "
                    "container=%s err=%r",
                    container_id[:12], exc,
                )
            finally:
                async with self._lock:
                    self._records.pop(container_id, None)
                if cpuset_cores:
                    self._core_ledger.release(cpuset_cores)
                # Release the image refcount if we were tracking this
                # container (a true CP-restart orphan has no record, so
                # nothing to release — we never acquired it).
                if held_image is not None and self._image_cache is not None:
                    self._image_cache.release(held_image)
        LOGGER.info(
            "raw-container.force_destroy container=%s",
            container_id[:12],
        )

    # ── Introspection ───────────────────────────────────────────────────────

    async def list_owned(
        self, *, rollout_id: str | None = None,
    ) -> list[RawContainerRecord]:
        """Return a snapshot of registered records (manager's
        in-memory truth).

        Filtered by ``rollout_id`` when supplied. The reconciler
        diffs this against :meth:`list_on_docker` to detect
        coordinator-side orphans.
        """
        async with self._lock:
            records = list(self._records.values())
        if rollout_id is not None:
            records = [r for r in records if r.rollout_id == rollout_id]
        return records

    async def list_on_docker(self) -> list[str]:
        """Return docker's truth: container_ids of all containers
        on this node carrying the ``xrlenv.session_kind=raw``
        label. Independent of the in-memory ``_records`` map —
        the reconciler diffs the two to find orphans on either
        side.

        ``all=True`` includes stopped containers; raw containers
        in case-2/3 are typically long-running ``sleep infinity``,
        but a stopped-and-not-yet-removed container is still an
        orphan if the coordinator doesn't know about it.
        """
        # ``sparse=True`` is load-bearing, not an optimisation. The
        # default ``containers.list`` inspects every container
        # (``GET /containers/<id>/json``) to populate full attrs, so a
        # container destroyed between the ``/containers/json`` listing
        # and its per-container inspect raises ``NotFound`` and aborts
        # the *whole* listing (issue #18). Raw containers churn
        # constantly during a run, so that race fired routinely and the
        # GC reconciler skipped the node for the sweep. ``sparse=True``
        # builds the Container objects from the list response alone —
        # no per-container inspect, no race — and ``.id`` is still
        # populated from the summary.
        containers = await asyncio.to_thread(
            self._client.containers.list,
            filters={"label": "xrlenv.session_kind=raw"},
            all=True,
            sparse=True,
        )
        return [c.id for c in containers]

    async def list_on_docker_info(self) -> list[tuple[str, str, str]]:
        """Docker's truth WITH the correlation labels — ``(container_id,
        rollout_id, compose_project)`` for each ``xrlenv.session_kind=raw``
        container. Uses the low-level API (like :meth:`list_disk_usage`) so the
        top-level ``Labels`` are present without a per-container inspect (the same
        race-free path ``list_on_docker`` relies on). ``compose_project`` is empty
        for an ordinary single raw container; the raw-GC reconciler uses it to
        route a compose-main's rebuild / whole-project teardown (P1.7.C.2)."""
        containers = await asyncio.to_thread(
            self._client.api.containers,
            filters={"label": "xrlenv.session_kind=raw"},
            all=True,
        )
        out: list[tuple[str, str, str]] = []
        for c in containers:
            labels = c.get("Labels") or {}
            out.append((
                c.get("Id", ""),
                labels.get("xrlenv.rollout_id", ""),
                labels.get("xrlenv.compose_project", ""),
            ))
        return out

    async def list_all_managed_on_docker_info(
        self,
    ) -> list[tuple[str, str, str, str]]:
        """EVERY xrlenv-managed container WITH correlation labels — ``(container_id,
        rollout_id, compose_project, session_kind)`` — filtered on the presence of the
        ``xrlenv.rollout_id`` label (a docker "label key exists" match), so it catches BOTH
        ``session_kind=raw`` (single container / compose main) AND ``session_kind=compose``
        (compose SIDECARS) that :meth:`list_on_docker_info` deliberately omits.

        readopt-on-connect (audit H11) uses this to detect a sidecar-only compose survivor — a
        project whose main is gone but whose sidecars are still alive — which the raw-only sweep
        can't see, so the node can be QUARANTINED rather than admitted with uncharged work. Uses
        the same race-free low-level ``api.containers`` path as :meth:`list_on_docker_info`."""
        containers = await asyncio.to_thread(
            self._client.api.containers,
            filters={"label": "xrlenv.rollout_id"},
            all=True,
        )
        out: list[tuple[str, str, str, str]] = []
        for c in containers:
            labels = c.get("Labels") or {}
            out.append((
                c.get("Id", ""),
                labels.get("xrlenv.rollout_id", ""),
                labels.get("xrlenv.compose_project", ""),
                labels.get("xrlenv.session_kind", ""),
            ))
        return out

    async def list_disk_usage(self) -> list[RawContainerDiskUsage]:
        """Running raw containers with their writable-layer (``SizeRw``)
        footprint — the disk guard's offender list.

        EXPENSIVE: ``size=True`` makes docker walk the layer graph for
        every container (the ``docker ps -s`` cost). The disk guard
        calls this ONLY after a cheap ``statvfs`` has already flagged
        pressure — never on its polling hot path. ``all=False`` (running
        only): a stopped container's writable layer still occupies disk,
        but killing a stopped container frees the same bytes a normal GC
        pass would, so the guard targets live offenders that GC/destroy
        can't yet reach. Per-container failures are skipped, not fatal.

        Uses the LOW-LEVEL ``client.api.containers(size=True)`` — the
        high-level ``ContainerCollection.list()`` does NOT accept
        ``size=`` (raises ``TypeError`` on real docker-py; a live smoke
        caught this after a fake accepted it). The low-level call returns
        the ``GET /containers/json?size=1`` dicts directly.
        """
        containers = await asyncio.to_thread(
            self._client.api.containers,
            filters={"label": "xrlenv.session_kind=raw"},
            size=True,
        )
        out: list[RawContainerDiskUsage] = []
        for c in containers:
            try:
                size_rw = int(c.get("SizeRw") or 0)
            except (TypeError, ValueError):
                size_rw = 0
            labels = c.get("Labels") or {}
            rollout_id = labels.get("xrlenv.rollout_id", "")
            cid = c.get("Id", "")
            # Prefer the in-memory record's image (exact acquired ref);
            # fall back to docker's summary Image for an untracked orphan.
            record = self._records.get(cid)
            image = record.image if record is not None else c.get("Image", "")
            out.append(
                RawContainerDiskUsage(
                    container_id=cid,
                    rollout_id=rollout_id,
                    image=image,
                    size_rw_bytes=size_rw,
                ),
            )
        return out

    def note_disk_reaped(self, rollout_id: str, reason: str) -> None:
        """Record that this node autonomously reaped ``rollout_id`` and
        why (audit P3 — the disk-pressure guard calls this before it
        force-destroys a runaway container). Bounded LRU so it can't grow
        without limit; surfaced to the control plane by
        :meth:`disk_reaped_reasons`."""
        if not rollout_id:
            return
        self._disk_reaped[rollout_id] = reason
        self._disk_reaped.move_to_end(rollout_id)
        while len(self._disk_reaped) > self._disk_reaped_max:
            self._disk_reaped.popitem(last=False)

    def disk_reaped_reasons(self) -> dict[str, str]:
        """Snapshot of recently node-reaped ``rollout_id -> reason`` for
        the ListRawContainers reply. Not cleared on read — the reconcile
        sweep may need more than one pass to observe the vanished
        container; the LRU bound handles cleanup."""
        return dict(self._disk_reaped)

    # ── Internal ────────────────────────────────────────────────────────────

    def _assert_owner(self, rollout_id: str, container_id: str) -> None:
        """Raise if ``rollout_id`` doesn't own ``container_id``.

        Synchronous read — the records dict is owned by the lock,
        but a stale read here is safe: the worst case is a
        concurrent destroy clearing the record between our check
        and the ensuing exec, in which case docker-py's
        ``containers.get`` raises NotFound and the exec returns
        an error.
        """
        record = self._records.get(container_id)
        if record is None:
            raise XRLEnvError(
                f"raw-container exec: container "
                f"{container_id[:12]!r} not registered on this "
                f"node (was it acquired? was it already destroyed?).",
            )
        if record.rollout_id != rollout_id:
            raise XRLEnvError(
                f"raw-container exec: rollout {rollout_id!r} does "
                f"not own container {container_id[:12]!r} "
                f"(owner: {record.rollout_id!r}).",
            )
