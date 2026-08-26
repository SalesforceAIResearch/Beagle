"""docker-py drop-in for xrlenv.

Goal: a consumer's existing benchmark code that starts with
``client = docker.from_env()`` (SWE-bench's harness, terminal-bench's
DockerComposeManager, OSWorld's provider, etc.) can swap one line —

.. code-block:: python

    -import docker
    -client = docker.from_env()
    +import xrlenv
    +client = xrlenv.from_env()                              # local
    +client = xrlenv.from_env(control=cluster_control)      # cluster

— and have every container created through that ``client`` actually
run on a fleet node the xrlenv scheduler picked, with the platform's
capacity / image-distribution / cancellation features applied
transparently.

Architecture: docker-py's high-level managers (``client.containers``,
``client.images``) delegate every operation to **low-level**
methods on ``docker.APIClient`` (``api.create_container``,
``api.start``, ``api.put_archive``, ``api.exec_create``,
``api.images``, etc.). Overriding those low-level methods means
the high-level managers Just Work for free — the consumer's
``client.containers.run(...)``, ``container.exec_run(...)``,
``client.images.pull(...)`` all route through xrlenv without
touching the manager classes.

How it works
============

``docker.DockerClient.__init__`` is literally::

    def __init__(self, *args, **kwargs):
        self.api = APIClient(*args, **kwargs)

So we subclass ``DockerClient`` and assign our own ``self.api``. The
manager objects (``client.containers``, ``client.images``, etc.) are
inherited from docker-py unchanged — they all delegate to
``self.client.api.<wire-method>()``.

Two modes:

LocalDocker (spike, single-host)
--------------------------------

``XrlenvAPIClient`` calls ``super().__init__()`` with the same
env-based config ``docker.from_env()`` uses, so every method
docker-py's manager classes call works against the local Docker
daemon. There are no method overrides — the subclass exists to
*reserve the seam* for cluster mode without reimplementing
docker-py's API surface.

In LocalDocker mode there's nothing to route — one host, one
daemon. The point of this mode is consumer ergonomics: they don't
have to think about docker vs. xrlenv at all on a laptop, and
benchmark harnesses (SWE-bench, terminal-bench, OSWorld) drive the
client unmodified through every docker-py code path.

Cluster (next slice)
--------------------

``XrlenvAPIClient`` skips ``super().__init__()`` (no daemon to dial)
and intercepts every wire-level method, routing through xrlenv's
existing gRPC stack to a scheduler-chosen node-agent. That's where
the real value-add (capacity-aware placement, image distribution,
cancellation primitives) lives.

xrlenv-specific metadata (task_key, group_id, resources) will ride
on **Docker labels**, not custom kwargs — labels are the standard
extensibility hook docker-py forwards end-to-end without us having
to override the high-level manager classes. Consumers will pass
e.g. ``labels={"xrlenv.task_key": "bench/1"}`` to
``client.containers.run(...)``, and the cluster-mode APIClient will
read those at create time to drive the scheduler.

The ``ContainerControl`` Protocol is the seam — concrete
implementations are :class:`LocalDockerContainerControl` (this file)
and ``ClusterContainerControl`` (next slice).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import re
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

import docker
import docker.utils

from xrlenv.backends.base import RuntimeLimits
from xrlenv.errors import XRLEnvError

if TYPE_CHECKING:
    from xrlenv.client.client import Client
    from xrlenv.client.container_session import ClusterContainerSession
    from xrlenv.types import TerminateRawGroupReport

LOGGER = logging.getLogger(__name__)

# Issue #18 (Ask #1, follow-up B) — an acquire whose admission-queue
# wait is at/above this many seconds counts as "queued" for the
# drop-in's telemetry + one-shot heads-up. 1 s mirrors the control-
# plane's own server-side WARN threshold so the two signals agree.
_QUEUE_WAIT_NOTIFY_THRESHOLD_S = 1.0

# Reserved docker label a consumer sets to bound the control-plane
# admission-queue wait per acquire (cluster mode). Hoisted into the
# first-class ``queue_timeout_s`` acquire kwarg in ``create_container``
# — the label path survives docker-py's high-level managers, which
# reject unknown kwargs. Value is seconds, parsed as a float.
_LABEL_QUEUE_TIMEOUT_S = "xrlenv.queue_timeout_s"

# Reserved docker label for the raw-session lifetime cap (issue #18).
# Hoisted into the ``session_deadline_s`` acquire kwarg. Same
# label-vs-kwarg rationale as ``_LABEL_QUEUE_TIMEOUT_S``. Value is
# seconds, parsed as a float.
_LABEL_SESSION_DEADLINE_S = "xrlenv.session_deadline_s"

# Reserved docker label for the per-node image-pull / acquire wire
# deadline (issue #12 / #18). Hoisted into the ``acquire_timeout_s``
# acquire kwarg. SWE-bench-Pro-scale images (a unique multi-GB image
# per instance, every acquire a cold pull) can exceed the 600s
# server default on a contended cluster; this label lets a drop-in
# consumer widen it without an xrlenv-shaped API. Value is seconds.
_LABEL_ACQUIRE_TIMEOUT_S = "xrlenv.acquire_timeout_s"


# API version we claim in cluster mode. docker-py's managers
# read this from ``client.api._version`` to build host-config
# dicts (kwargs translation, capability gating). 1.41 is broadly
# compatible with the docker daemon versions our nodes run
# (Docker Engine 20.10+); newer claim breaks legacy harnesses,
# older claim breaks features harnesses use.
_CLUSTER_API_VERSION = "1.41"

# The EXACT staged cmd of swebench's timeout-watchdog kill — the only detached exec cluster
# mode tolerates as a contained no-op (audit Low, exec_start). Upstream calls
# ``container.exec_run("kill -TERM <pid>", detach=True)``; exec_create wraps a string cmd as
# ``["sh", "-c", cmd]`` and we always report Pid 0, so the staged list is exactly this.
_WATCHDOG_KILL_CMD = ["sh", "-c", "kill -TERM 0"]

# Upper bound on retained infra-failure records / container→rollout associations (audit Low):
# a generous cap so a well-behaved consumer never hits it, but growth is bounded even if a
# consumer supplies rollout metadata and never pops the record. Oldest is evicted at the cap.
_MAX_INFRA_RECORDS = 4096

# A node-side failure that crosses the CP→node hop is re-raised by the control plane as a bare
# ``XRLEnvError`` whose message carries the concrete kind in a structured prefix. The CP emits
# one shape per operation (``xrlenv/control/grpc_endpoint.py``):
#   "node <id>: remote command <Kind>: <msg>"       (acquire / batched command)
#   "remote stream <Kind>: <msg>"                    (streaming exec — swebench's /eval.sh)
#   "node <id>: remote get_archive <Kind>: <msg>"    (archive)
# When the client rehydrates a kind it doesn't map (transport._KIND_TO_EXC), it likewise
# surfaces as bare ``XRLEnvError``, and its message IS exactly one of these CP shapes (the
# reason metadata the CP set). ``_infra_kind`` recovers the concrete kind by matching the CP
# shape ANCHORED AT THE START of the message: an optional ``node <id>: `` prefix, then
# ``remote <op> <Kind>:`` with an enumerated op. Anchoring at ``^`` (via ``match``) means a
# ``node …: remote command NodeLost:`` clause embedded MID-message — e.g. inside a
# ``gRPC error UNKNOWN: …`` string — can NOT spoof a kind (audit Low; a single deterministic
# wire field, NOT log scraping).
# The node-id slot is a SINGLE token (``[^:\s]+`` — hostnames / IPs / ``worker-N`` never carry
# spaces), so a multi-token ``node <arbitrary prose>: remote command NodeLost:`` can't sneak the
# whole clause into the node slot and spoof a kind (audit Low). Combined with ``match`` (^-
# anchored) this needs a genuine ``node <id>:`` / ``remote <op>`` prefix at the very start.
_WIRE_KIND_RE = re.compile(
    r"(?:node [^:\s]+: )?remote (?:command|stream|get_archive|put_archive) (\w+):",
)


def _infra_kind(exc: BaseException) -> str:
    """The concrete failure kind for a cluster exception: the exception type name, or — when
    the type was flattened to bare ``XRLEnvError`` across the node-control wire — the kind
    named in the CP's structured ``remote <op> <Kind>:`` message prefix, matched ANCHORED at
    the start of the message (command / stream / get_archive / put_archive) (audit M8/Low)."""
    name = type(exc).__name__
    if name == "XRLEnvError":
        m = _WIRE_KIND_RE.match(str(exc))   # anchored at start — no mid-message spoofing
        if m:
            return m.group(1)
    return name


# De-dupe the "ignoring docker-py kwargs" warnings. A harness (e.g. swebench) creates
# hundreds of containers passing the SAME unsupported kwargs, so warning per-create
# floods the sweep log with identical lines. Warn ONCE per unique signature per
# process; the lock makes check-then-add atomic under the ThreadPoolExecutor callers.
_KWARG_WARN_LOCK = threading.Lock()
_KWARG_WARN_SEEN: set[tuple[str, ...]] = set()


def _warn_kwargs_once(signature: tuple[str, ...]) -> bool:
    """True the FIRST time ``signature`` is seen this process (caller should warn),
    False thereafter. Thread-safe so a burst of concurrent creates warns once, not N."""
    with _KWARG_WARN_LOCK:
        if signature in _KWARG_WARN_SEEN:
            return False
        _KWARG_WARN_SEEN.add(signature)
        return True


def _warn_unused_kwargs(method: str, unused: dict[str, Any]) -> None:
    """Loudly warn when the cluster path receives docker-py kwargs
    it can't propagate downstream to ``acquire_container``.

    Previously these were silently swallowed via ``**_unused``,
    which made integration regressions invisible (a caller's
    ``entrypoint=`` / ``volumes=`` / ``working_dir=`` would simply
    not take effect on the cluster, with no signal to the
    operator). Logging at WARNING level surfaces the gap in normal
    operator workflows; tests can capture and assert on it via
    ``caplog``.
    """
    if not unused:
        return
    # Drop kwargs explicitly known to be no-ops in cluster mode.
    # ``detach`` is forced by acquire_container's contract;
    # docker-py callers routinely pass it without expecting
    # cluster-side action. Suppressing it cuts noise without
    # hiding real surprises.
    quiet = {"detach"}
    keys = sorted(k for k in unused if k not in quiet)
    if not keys:
        return
    if not _warn_kwargs_once((method, *keys)):   # once per (method, kwargs) — not per create
        return
    LOGGER.warning(
        "xrlenv cluster %s: ignoring docker-py kwargs %s that the "
        "raw-container path does not forward today. If your harness "
        "depends on them, either request the option upstream "
        "(xrlenv/compat/docker_client.py + acquire_container surface) "
        "or use a local-mode runtime. Silently dropping these in the "
        "past masked integration breakage; this warning is intentional.",
        method,
        keys,
    )


def _parse_devices_field(value: Any) -> list[str]:
    """Convert docker-py's normalized ``Devices`` host_config entry back
    to docker CLI-style spec strings for validation + wire forwarding.

    docker-py's high-level ``containers.create(devices=["/dev/kvm"])``
    bundles into ``host_config["Devices"]`` as a list of dicts shaped
    ``{"PathOnHost": "/dev/x", "PathInContainer": "/dev/y",
    "CgroupPermissions": "rwm"}``. The cluster-side policy validator
    and the AcquireContainer wire field both want the CLI-style
    ``"/dev/x"`` / ``"/dev/x:/dev/y"`` / ``"/dev/x:/dev/y:rwm"`` strings,
    so we round-trip back to that representation here. Raw strings
    (some callers bypass the high-level managers) pass through verbatim.
    """
    if not value:
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        host = item.get("PathOnHost") or ""
        if not host:
            continue
        container = item.get("PathInContainer") or ""
        perms = item.get("CgroupPermissions") or ""
        spec = f"{host}:{container}" if container and container != host else host
        if perms:
            spec = f"{spec}:{perms}"
        out.append(spec)
    return out


def _reject_platform_kwarg(value: Any) -> None:
    """Emit a targeted warning when a caller passes ``platform=...``
    to ``create_container`` in cluster mode. Distinct from the
    generic ``_warn_unused_kwargs`` because the rejection is
    intentional, not a "we'll wire this someday" gap:

    The per-node architecture in a cluster is operator-controlled
    at deploy time — each node's docker daemon serves a single
    arch, and the scheduler routes the container to whichever
    available node matches the image. A consumer-side
    ``platform="linux/x86_64"`` hint can't change that; honoring
    it would either be a no-op (already matches) or a contradiction
    (would force a pull of a manifest the node can't run).

    Mirror in issue #6: the allowlist→blocklist design call
    explicitly classifies platform as ``reject with operator-
    rationale``, not ``forward through``.
    """
    if not _warn_kwargs_once(("platform", repr(value))):   # once per value — not per create
        return
    LOGGER.warning(
        "xrlenv cluster create_container: ignoring docker-py kwarg "
        "platform=%r — node architecture is operator-controlled at "
        "deploy time (each node's docker daemon serves a single "
        "arch). Cluster mode routes the container to a node whose "
        "arch matches the image; a consumer-side platform hint "
        "cannot change that. This is intentional rejection (not a "
        "silent drop, not a TODO) — see xrlenv issue #6 for the "
        "broader allowlist/blocklist design call.",
        value,
    )


# ── P0a — resource host_config handling (cluster-resource-isolation) ────────
#
# docker-py bundles cpu / memory / runtime limits into ``host_config``.
# In cluster mode these used to be silently warn-and-dropped, so a harness
# that capped cpu/mem in local Docker ran *uncapped* on the cluster — the
# root cause of the timing-test jitter in the plan's motivating incident.
# P0a replaces the silent drop with explicit handling:
#   - CPU / memory          -> an effective request the scheduler honors
#   - soft knobs (shares,   -> hard error (no deterministic isolation)
#     memory reservation)
#   - RuntimeLimits keys    -> hard error until P0b wires them
#   - cpuset_* / cgroup_parent -> Level-3 policy rejection (via kwargs_policy)

_RESOURCE_HOST_CONFIG_KEYS = frozenset({
    "NanoCpus", "CpuQuota", "CpuPeriod", "CpuShares",
    "Memory", "MemorySwap", "MemoryReservation",
    "PidsLimit", "ShmSize", "Tmpfs", "ReadonlyRootfs",
    "CpusetCpus", "CpusetMems",
})


class ClusterResourceKwargError(ValueError):
    """A docker ``host_config`` resource kwarg cannot be honored in
    cluster mode. Replaces the old silent warn-and-drop so a harness
    learns immediately rather than getting a wrong (uncapped) run."""


def _raise_unsupported_resource(
    *, kwarg: str, requested: Any, reason: str, action: str,
) -> None:
    """Raise a four-part hard error (requested / cap / reason / action) —
    the message standard from cluster-resource-isolation-plan P0."""
    raise ClusterResourceKwargError(
        f"xrlenv cluster create_container: docker host_config "
        f"`{kwarg}` cannot be honored.\n"
        f"  requested: {requested!r}\n"
        f"  reason:    {reason}\n"
        f"  action:    {action}",
    )


def _resolve_effective_cpu_mem(
    host_config: dict[str, Any],
) -> tuple[float | None, int | None]:
    """Translate the CPU/memory ``host_config`` keys into an effective
    ``(cpu_limit cores, mem_limit bytes)`` pair (P0a).

    Hard-errors on keys that cannot be honored in cluster mode v1: soft
    controls (no hard isolation) and RuntimeLimits keys (not wired until
    P0b). Returns ``(None, None)`` when the harness set no CPU/memory.
    """
    # Soft / relative controls — not a hard cap; reject (P0a).
    if host_config.get("CpuShares"):
        _raise_unsupported_resource(
            kwarg="CpuShares", requested=host_config["CpuShares"],
            reason=(
                "CpuShares is a relative scheduler weight, not a hard cap; "
                "it cannot give the deterministic CPU isolation cluster "
                "grading needs."
            ),
            action=(
                "express a hard CPU limit via nano_cpus, or "
                "cpu_quota + cpu_period, instead."
            ),
        )
    if host_config.get("MemoryReservation"):
        _raise_unsupported_resource(
            kwarg="MemoryReservation",
            requested=host_config["MemoryReservation"],
            reason=(
                "MemoryReservation is a soft limit; it cannot stand in for "
                "hard memory enforcement."
            ),
            action="express a hard memory cap via mem_limit (Memory) instead.",
        )
    # CPU — NanoCpus, or CpuQuota (+ CpuPeriod). Reject conflicting specs.
    nano = host_config.get("NanoCpus")
    quota = host_config.get("CpuQuota")
    cpu_limit: float | None = None
    if nano and quota:
        _raise_unsupported_resource(
            kwarg="NanoCpus+CpuQuota",
            requested={"NanoCpus": nano, "CpuQuota": quota},
            reason="NanoCpus and CpuQuota are both set — conflicting CPU specs.",
            action="pass exactly one of nano_cpus or cpu_quota/cpu_period.",
        )
    if nano:
        cpu_limit = float(nano) / 1e9
    elif quota:
        # docker's default CFS period is 100ms when only a quota is given.
        period = host_config.get("CpuPeriod") or 100_000
        cpu_limit = float(quota) / float(period)
    # Memory — honor Memory; MemorySwap must be unambiguous (plan Risk 2).
    memory = host_config.get("Memory") or None
    swap = host_config.get("MemorySwap")
    # MemorySwap == Memory means swap disabled (unambiguous, nothing to
    # forward). Anything else — including -1 "unlimited swap" — has
    # daemon/cgroup-dependent semantics; reject rather than guess.
    if swap is not None and swap != 0 and (memory is None or swap != memory):
        _raise_unsupported_resource(
            kwarg="MemorySwap", requested=swap,
            reason=(
                "MemorySwap semantics vary by daemon / cgroup config "
                "(-1 means unlimited swap); only MemorySwap == Memory "
                "(swap disabled) is unambiguous."
            ),
            action=(
                "set memswap_limit equal to mem_limit to disable swap, "
                "or omit it."
            ),
        )
    mem_limit_bytes = int(memory) if memory else None
    return cpu_limit, mem_limit_bytes


def _resolve_runtime_limits(
    host_config: dict[str, Any],
) -> RuntimeLimits | None:
    """P0b — extract the container-shape RuntimeLimits keys
    (``PidsLimit`` / ``ShmSize`` / ``Tmpfs`` / ``ReadonlyRootfs``) the
    harness set in ``host_config``. Returns ``None`` when the harness
    set none — the node then applies no constraint (docker default).

    Only harness-specified limits are forwarded; cluster mode does not
    inject pids/shm defaults the harness did not request, so behaviour
    matches local Docker.
    """
    pids = host_config.get("PidsLimit") or None
    shm = host_config.get("ShmSize") or None
    tmpfs = dict(host_config.get("Tmpfs") or {})
    read_only = bool(host_config.get("ReadonlyRootfs"))
    limits = RuntimeLimits(
        pids_limit=int(pids) if pids else None,
        shm_size_bytes=int(shm) if shm else None,
        tmpfs=tmpfs,
        readonly_rootfs=read_only,
    )
    return None if limits.is_empty() else limits


class _DropInRunner:
    """Dispatches the drop-in's async work to a single loop.

    Why this exists: the docker-py drop-in is **synchronous** by
    contract (docker-py is sync), but the xrlenv ``Client`` is
    async + holds loop-bound state (gRPC channels, asyncio.Queues,
    Futures). Naive ``asyncio.run(coro)`` in each sync override
    creates a fresh loop per call, which is incompatible with a
    Client whose state was bound to a different loop — gRPC's
    async stubs in particular crash with
    ``Future <...> attached to a different loop`` when awaited
    on a loop that didn't create them.

    Two construction modes:

    - **Owned** (``__init__(loop=None)``): spins up a dedicated
      background loop on a daemon thread. Used by the
      ``from_env(grpc_host=..., grpc_port=..., ...)`` connect-mode
      factory: the gRPC ``Client`` is built **on that loop** so
      all of its internal state binds to it. Every drop-in call's
      coroutine dispatches via ``run_coroutine_threadsafe`` to
      the same loop — no cross-loop hazards.

    - **Attached** (``__init__(loop=<existing>)``): reuses a loop
      the caller already runs (e.g. an embedded
      ``build_distributed_runtime`` whose Client was built on the
      caller's running loop). Useful when the sync drop-in code
      runs in ``asyncio.to_thread`` from inside that loop —
      ``run_coroutine_threadsafe`` posts back to the original
      loop. The caller owns the loop; ``close()`` is a no-op.

    Closing an owned runner (via ``XrlenvDockerClient.close()``)
    stops the loop + joins the thread. Daemon-true so a leaked
    drop-in doesn't keep the process alive.
    """

    def __init__(
        self, *, loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        if loop is not None:
            # Attached mode: caller owns the loop; we just
            # dispatch to it. Caller is responsible for keeping
            # the loop alive while this drop-in is in use.
            self._loop = loop
            self._owned = False
            self._thread: threading.Thread | None = None
        else:
            # Owned mode: spin up a fresh background loop.
            self._loop = asyncio.new_event_loop()
            self._owned = True
            self._thread = threading.Thread(
                target=self._loop.run_forever,
                name="xrlenv-dropin-runner",
                daemon=True,
            )
            self._thread.start()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def run(self, coro: Any) -> Any:
        """Schedule ``coro`` on the runner's loop, block the
        calling (sync) thread until it resolves. Exceptions raised
        inside the coro propagate to the caller."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        # Attached mode: caller owns the loop, nothing for us
        # to shut down.
        if not self._owned:
            return
        if self._thread is None or not self._thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        # Best-effort: if the thread refused to stop in time, the
        # loop's still alive but daemon=True means it won't block
        # process exit.


def _run_sync(coro: Any, *, runner: _DropInRunner | None = None) -> Any:
    """Run an async coroutine from a sync docker-py callsite.

    Two modes:

    - **runner-backed** (preferred for cluster mode): dispatch
      via the drop-in's owned background loop. Eliminates cross-
      loop hazards when the Client was built on the runner's
      loop. Used when ``ClusterContainerControl`` carries a
      runner (``from_env(grpc_host=..., grpc_port=..., ...)``
      sets one up automatically).

    - **fresh-loop** (legacy / power-user): when no runner is
      bound, fall back to ``asyncio.run`` — works ONLY when no
      event loop is active in the calling thread AND when the
      coro doesn't await any Client/loop-bound state. Raises a
      clear error if called from inside an active loop.
    """
    if runner is not None:
        return runner.run(coro)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop active — typical sync-harness case (and the
        # coro must not touch loop-bound state).
        return asyncio.run(coro)
    raise RuntimeError(
        "xrlenv docker drop-in (cluster mode) called from within "
        "an asyncio event loop with no runner-backed dispatch. "
        "Either build the drop-in via "
        "``xrlenv.from_env(grpc_host=..., grpc_port=..., "
        "consumer_token=...)`` (drop-in owns the loop) or call "
        "from a thread with no active loop.",
    )


class ContainerControl(Protocol):
    """Marker interface for the routing seam.

    Implementations control how :class:`XrlenvAPIClient` resolves
    docker-py wire calls. The current LocalDocker variant is a tag
    (no methods) since the underlying real ``docker.APIClient``
    already implements every method we need; cluster-mode routing
    is on the same ``XrlenvAPIClient`` subclass via different
    overrides.
    """

    mode: str


class LocalDockerContainerControl:
    """Single-host control: route every docker-py method to the
    local Docker daemon via the inherited ``docker.APIClient``.

    Use cases:

    - Laptop dev iteration — consumer runs ``xrlenv.from_env()``
      against Docker Desktop. Functionally identical to
      ``docker.from_env()``; the subclass exists to reserve the seam
      for cluster mode without changing call semantics today.
    - SWE-bench / terminal-bench / OSWorld validation — their
      harnesses' docker-py usage runs unmodified across the entire
      manager surface (images.list/get/pull, streaming exec,
      container.create+start patterns, ...). Nothing is overridden;
      every method delegates to the real ``docker.APIClient``
      ``super().__init__()`` set up.

    The actual orchestration value-add (fleet placement, capacity,
    cancellation) lives in ``ClusterContainerControl``.
    """

    mode = "local"


class ClusterContainerControl:
    """Cluster-mode control: route every docker-py method through
    the xrlenv ``Client`` to whatever node the scheduler picked.

    Holds (a) the live ``Client`` we dispatch through and (b) a
    ``container_id → ClusterContainerSession`` map so subsequent
    calls (exec, put_archive, destroy) on a container created
    earlier in the session find their session.

    Each ``api.create_container`` (or, more accurately, the
    high-level ``client.containers.run`` / ``.create``) on the
    drop-in ends up acquiring a fresh raw container from the
    cluster — one rollout per container. The drop-in tracks the
    rollout_id internally so the consumer never sees it; from the
    consumer's POV it's just a container id like any other docker
    container.

    Pre-req: the chosen node must already have the requested
    image (no implicit pull on the raw-container path; same
    contract as ``Client.acquire_container``). Operators pre-pull
    or build before consumers hit the drop-in.
    """

    mode = "cluster"

    def __init__(
        self, *, client: Client, runner: _DropInRunner | None = None,
    ) -> None:
        self._client = client
        # Optional runner — present when ``from_env`` built the
        # Client on a dedicated background loop. None means the
        # caller managed loop / threading themselves; the api
        # overrides fall back to ``asyncio.run`` per call (and
        # will fail loudly on cross-loop violations).
        self._runner = runner
        # Per-drop-in-instance session map. Two consumers' drop-ins
        # don't share state — each constructs its own
        # ClusterContainerControl with its own ``Client``.
        # Stores ``(session, image)`` per container_id so
        # ``inspect_container`` can return the original image even
        # though the SDK ``ClusterContainerSession`` doesn't carry
        # it (the SDK's only caller that needs image is the
        # drop-in itself).
        self._sessions: dict[str, tuple[ClusterContainerSession, str]] = {}
        # Issue #18 (Ask #1, follow-up B) — admission-queue telemetry.
        # The control plane now queues acquires past cluster capacity
        # instead of erroring; without surfacing that, a consumer
        # over-requesting concurrency sees only a slower run and no
        # signal to lower their worker count next time. We accumulate
        # per-acquire queue waits + the peak concurrent-container
        # count, log a one-shot heads-up on the first queued acquire,
        # and an end-of-run summary (atexit) with a measured
        # sustainable-concurrency suggestion.
        self._acquire_total = 0
        self._acquire_queued = 0
        self._queue_wait_sum_s = 0.0
        self._queue_wait_max_s = 0.0
        self._live_peak = 0
        self._first_queue_warning_emitted = False
        atexit.register(self._log_admission_summary)

    def register_session(
        self, session: ClusterContainerSession, *, image: str,
    ) -> None:
        self._sessions[session.container_id] = (session, image)
        self._record_admission(session)

    # ── Issue #18 (Ask #1, follow-up B) — admission telemetry ───────────────

    def _record_admission(self, session: ClusterContainerSession) -> None:
        """Fold one acquire's admission-queue wait into the running
        telemetry + emit a one-shot heads-up the first time an
        acquire actually queues."""
        self._acquire_total += 1
        # ``len(_sessions)`` was just bumped by ``register_session``;
        # this is the live concurrency right now — its running max is
        # the concurrency the cluster actually sustained.
        self._live_peak = max(self._live_peak, len(self._sessions))
        wait_s = getattr(session, "queue_wait_s", 0.0) or 0.0
        if wait_s < _QUEUE_WAIT_NOTIFY_THRESHOLD_S:
            return
        self._acquire_queued += 1
        self._queue_wait_sum_s += wait_s
        self._queue_wait_max_s = max(self._queue_wait_max_s, wait_s)
        if not self._first_queue_warning_emitted:
            self._first_queue_warning_emitted = True
            LOGGER.warning(
                "xrlenv cluster: an acquire queued %.1fs for capacity — the "
                "cluster is at its admission limit right now (heavy concurrency, "
                "other consumers, or a cold image pull can each cause this). The "
                "run still completes, just slower; an end-of-run summary prints "
                "on exit.",
                wait_s,
            )

    def _log_admission_summary(self) -> None:
        """atexit hook — print the admission summary once the run is
        done. Silent when nothing was acquired (drop-in built but
        unused) or when no acquire ever queued (cluster kept up)."""
        if self._acquire_total == 0:
            return
        if self._acquire_queued == 0:
            LOGGER.info(
                "xrlenv cluster admission summary: %d acquires, none "
                "queued — the cluster kept up with your concurrency.",
                self._acquire_total,
            )
            return
        mean_wait = self._queue_wait_sum_s / self._acquire_queued
        pct_queued = 100.0 * self._acquire_queued / self._acquire_total
        LOGGER.warning(
            "xrlenv cluster admission summary:\n"
            "  acquires:  %d total, %d queued (%.0f%%)\n"
            "  queue wait: mean %.1fs, max %.1fs\n"
            "  peak concurrent containers observed: %d\n"
            "  note: queueing means the cluster hit its admission limit during "
            "this run; ~%d containers ran at once at peak. If that peak is well "
            "below your worker count, the cluster is the bottleneck — lower "
            "workers toward it. (A sequential caller naturally shows peak 1, so "
            "this number is 'how many you ran at once', not a capacity verdict.)",
            self._acquire_total, self._acquire_queued, pct_queued,
            mean_wait, self._queue_wait_max_s,
            self._live_peak, self._live_peak,
        )

    def get_session(self, container_id: str) -> ClusterContainerSession:
        entry = self._sessions.get(container_id)
        if entry is None:
            raise docker.errors.NotFound(
                f"xrlenv cluster drop-in: container "
                f"{container_id[:12]!r} not registered. Containers "
                f"created outside this drop-in instance (e.g. by a "
                f"different ``xrlenv.from_env`` call or directly via "
                f"``Client.acquire_container``) aren't visible here.",
            )
        return entry[0]

    def get_image(self, container_id: str) -> str:
        entry = self._sessions.get(container_id)
        if entry is None:
            raise docker.errors.NotFound(
                f"xrlenv cluster drop-in: container "
                f"{container_id[:12]!r} not registered.",
            )
        return entry[1]

    def drop_session(self, container_id: str) -> None:
        self._sessions.pop(container_id, None)

    @property
    def client(self) -> Client:
        return self._client

    @property
    def runner(self) -> _DropInRunner | None:
        return self._runner


class XrlenvAPIClient(docker.APIClient):
    """docker-py APIClient subclass.

    Two modes, dispatched on ``control.mode``:

    - **LocalDocker mode**: calls ``super().__init__()`` with the
      same env-based config ``docker.from_env()`` uses. Every
      inherited method works against the local Docker daemon
      unchanged. The subclass exists to reserve the routing seam
      without changing call semantics on a single host.

    - **Cluster mode**: skips ``super().__init__()`` (no daemon
      to dial — we route through the xrlenv ``Client``) and
      overrides the low-level api methods that docker-py's
      high-level managers (``client.containers``, ``client.images``)
      delegate to. Each override translates the docker-API call
      into a ``Client``-side RPC (acquire / exec / archive /
      destroy / image-cache query). Methods we haven't
      explicitly overridden raise ``NotImplementedError`` rather
      than failing strangely on uninitialized parent state.
      As more harnesses get tested, more overrides land.

    Cluster-mode metadata (task_key, group_id, etc.) rides on
    Docker labels, not custom kwargs — the standard extensibility
    hook docker-py already forwards end-to-end without manager-
    class override gymnastics.
    """

    def __init__(self, *, control: ContainerControl) -> None:
        self._control = control
        # Structured failure side channel (audit M8) — present in BOTH modes so the accessor
        # methods are always safe (local mode simply never records). Each container operation
        # (acquire / exec / archive) stashes the failure KIND here, keyed by the rollout
        # displayed_name, before the harness wraps/swallows it. ``_container_rollout`` maps a
        # live container_id -> its displayed_name so POST-acquire ops (which run without the
        # consumer's contextvar, e.g. on swebench's exec watchdog thread) can still correlate.
        self._infra_failures: dict[str, str] = {}
        self._container_rollout: dict[str, str] = {}
        self._infra_failures_lock = threading.Lock()
        if control.mode == "local":
            # Same env-based init docker.from_env() does.
            super().__init__(**docker.utils.kwargs_from_env())
        elif control.mode == "cluster":
            # Skip super().__init__() — no daemon socket to dial.
            # The base class's internal session / transport state
            # stays uninitialized; in this mode every callable
            # we don't explicitly want to expose must be guarded.
            self._cluster_control: ClusterContainerControl = control  # type: ignore[assignment]
            # Audit Cluster-Dropin-M3 closure. docker-py's
            # high-level managers (``client.containers.run``,
            # ``client.containers.create``, ...) read internal
            # APIClient state — particularly ``_version`` — that
            # ``super().__init__()`` would normally populate from
            # an API-version-negotiation round-trip with the
            # daemon. Without it, the managers AttributeError
            # before reaching our ``api.*`` overrides.
            #
            # Set the minimum state the managers actually consult
            # so the high-level surface works end-to-end:
            self._version = _CLUSTER_API_VERSION
            self._auth_configs = {"auths": {}}
            self._general_configs: dict[str, Any] = {}
            self.base_url = "xrlenv://cluster"
            # ``timeout`` is read by some helpers; default to a
            # generous value (the actual timeout enforcement is
            # at the spec-21 wire level).
            self.timeout = 60
            # Per-client exec registry: docker-py's exec triple
            # is stateful (exec_create returns an id; exec_start
            # uses it; exec_inspect reads its result). Track the
            # request + result keyed on synthetic exec_id so the
            # three calls compose. Counter-based id assignment
            # so tests can assert on a stable shape.
            self._exec_pending: dict[str, dict[str, Any]] = {}
            self._exec_results: dict[str, dict[str, Any]] = {}
            # exec_ids currently mid-stream (popped from _exec_pending, not yet in
            # _exec_results) — lets exec_inspect report Running:True for an in-flight exec
            # instead of raising NotFound at a timeout watchdog's inspect (audit M2).
            self._exec_streaming: set[str] = set()
            self._exec_counter: int = 0
            self._install_cluster_safety_net()
        else:
            raise NotImplementedError(
                f"ContainerControl mode={control.mode!r} not supported. "
                f"Use 'local' or 'cluster'.",
            )

    # ── Structured infra-failure side channel (audit M8) ────────────────────────
    #
    # Upstream harnesses (swebench) catch a container operation's exception, LOG it, and either
    # wrap it (``BuildImageError``) or swallow it into ``completed=false`` — erasing the real
    # exception type from anything a caller can inspect programmatically. Recovering the
    # infra-vs-content distinction by re-parsing the human-readable ``run_instance.log`` proved
    # unreliable (spoofable by traceback-shaped text in an exception message; blind to a
    # logger-prefixed wrapped chain). Instead, each operation (acquire in ``create_container``;
    # post-acquire exec/archive via ``_run_op``) records the failure KIND HERE, at the mechanism
    # boundary, before the harness can obscure it — and the consuming ADAPTER owns the policy of
    # which kinds warrant an infra retry.
    #
    # Correlation contract: records are keyed by the rollout ``displayed_name``. Concurrent
    # operations for DIFFERENT rollouts use different keys and don't collide. A consumer that
    # runs multiple attempts for the SAME name must serialize them (the swebench sweep does:
    # each instance's infra + content retries run strictly sequentially, and it clears the key
    # before an attempt + pops it after), so one attempt's evidence never crosses into another.

    def _record_infra_failure(self, key: str, kind: str) -> None:
        with self._infra_failures_lock:
            # Bound the dict so a consumer that supplies rollout metadata but never pops the
            # record can't grow it without limit (audit Low). Dicts preserve insertion order;
            # evict the oldest when at the cap.
            if (key not in self._infra_failures
                    and len(self._infra_failures) >= _MAX_INFRA_RECORDS):
                self._infra_failures.pop(next(iter(self._infra_failures)), None)
            self._infra_failures[key] = kind

    def _remember_container_rollout(self, container_id: str, displayed_name: str) -> None:
        """Associate a freshly acquired container with its rollout displayed_name so a
        post-acquire op failure can be correlated back to it (audit M8). NOT capped: this map
        is ACTIVE lifecycle state, bounded by the live-container count and dropped on destroy
        (``_forget_container``) — a hard cap could silently evict a LIVE container's mapping and
        lose its failure correlation (audit Low). If it ever grew unbounded, the leak would be
        missing destroys, which is the bug to fix, not a mapping to discard."""
        with self._infra_failures_lock:
            self._container_rollout[container_id] = displayed_name

    def _forget_container(self, container_id: str) -> None:
        """Drop the container→rollout association on destroy so the map is lifecycle-bounded
        (audit Low: no unbounded growth of associations)."""
        with self._infra_failures_lock:
            self._container_rollout.pop(container_id, None)

    def _record_op_failure(self, container_id: str, exc: BaseException) -> None:
        """Record a POST-acquire operation failure (exec / archive) against the container's
        rollout displayed_name, if known. The kind is recovered structurally via ``_infra_kind``
        (handles a wire-flattened bare ``XRLEnvError``). Recording is unconditional on the kind
        — the consuming ADAPTER owns the policy of which kinds warrant an infra retry (audit
        M8: mechanism records evidence, policy lives in the benchmark adapter)."""
        with self._infra_failures_lock:
            corr = self._container_rollout.get(container_id)
        # Route through the bounded recorder (audit Low: _record_op_failure must not bypass
        # _MAX_INFRA_RECORDS). The corr lookup + the record are two separate lock acquisitions
        # (the lock is non-reentrant), which is fine — a concurrent destroy can only drop the
        # mapping, in which case we simply don't record for a gone container.
        if corr is not None:
            self._record_infra_failure(corr, _infra_kind(exc))

    def _run_op(self, coro: Any, *, container_id: str) -> Any:
        """``_run_sync`` for a post-acquire container op, recording any ``XRLEnvError`` as
        structured failure evidence before it propagates to the harness (which may swallow it
        into ``completed=false``) (audit M8)."""
        try:
            return _run_sync(coro, runner=self._cluster_control.runner)
        except XRLEnvError as exc:
            self._record_op_failure(container_id, exc)
            raise

    def take_infra_failure(self, key: str) -> str | None:
        """Pop (read + clear) the last failure KIND recorded for ``key`` (the rollout
        ``displayed_name``), or None. A consumer reads this AFTER an upstream call returned
        ``completed=false`` to recover a swallowed/wrapped failure's kind — WITHOUT scraping
        logs (audit M8). The consumer decides whether the kind warrants an infra retry."""
        with self._infra_failures_lock:
            return self._infra_failures.pop(key, None)

    def clear_infra_failure(self, key: str) -> None:
        """Drop any stale record for ``key`` before a fresh attempt, so evidence never crosses
        retries/reruns (audit M8)."""
        with self._infra_failures_lock:
            self._infra_failures.pop(key, None)

    # ── Cluster-mode safety net ─────────────────────────────────────────────
    #
    # In cluster mode, ``super().__init__()`` was skipped — the
    # parent's ``_session`` / ``_url`` / etc. don't exist. Most
    # docker.APIClient methods (and the methods on the mixins
    # ContainerApiMixin / ImageApiMixin / ExecApiMixin / …) would
    # blow up on uninitialized internal state if called.
    #
    # ``__getattr__`` only fires for *missing* attributes —
    # inherited methods are FOUND on the class hierarchy so it
    # doesn't see them. To guard inherited methods we shadow them
    # at __init__ time on the instance: setattr replaces the
    # class-level method with a NotImplemented stub on this
    # particular client. (Audit Cluster-Dropin-M1 closure —
    # previously inherited methods like ``exec_create`` /
    # ``put_archive`` could run against uninitialized parent
    # state.)

    # Methods explicitly wired in cluster mode. Anything in this
    # set is a real method on this subclass; the safety net
    # leaves them alone.
    _CLUSTER_OVERRIDES: frozenset[str] = frozenset({
        "create_container",
        "start",
        "stop",
        "remove_container",
        "inspect_container",
        # P1.7.B exec triple (batched + streaming).
        "exec_create",
        "exec_start",
        "exec_inspect",
        # P1.7.B archives.
        "put_archive",
        "get_archive",
        # P1.7.B image ops — adapt onto operator-pre-pulled
        # contract; see docstrings on each method.
        "pull",
        "inspect_image",
        "images",
        "remove_image",
        "history",
        # Container listing — cluster-mode no-op returning [].
        # swebench's grader calls api.containers via
        # client.containers.list(all=True) in
        # make_run_report's leak-counting step; without this entry
        # the safety net shadows the real override and grading
        # truncates with NotImplementedError mid-report.
        "containers",
    })

    def _install_cluster_safety_net(self) -> None:
        """Walk every public method on docker.APIClient (and its
        mixins) and shadow each non-override with a
        NotImplementedError stub on this instance. Instance attrs
        take precedence over class attrs, so the inherited
        method becomes unreachable per-instance without modifying
        the class.

        Called from ``__init__`` only when ``mode == "cluster"``.
        Local mode is unaffected (the parent's real methods stay
        live).
        """
        seen: set[str] = set()
        # Walk MRO so methods defined on docker.APIClient itself
        # AND on each mixin get covered. Skip ``object`` so we
        # don't shadow built-in dunders.
        for cls in type(self).__mro__:
            if cls is object:
                continue
            if cls is type(self):
                # Our own subclass: don't shadow our own
                # overrides (they're already real).
                continue
            for attr_name in vars(cls):
                if attr_name.startswith("_"):
                    continue
                if attr_name in self._CLUSTER_OVERRIDES:
                    continue
                if attr_name in seen:
                    continue
                attr = getattr(cls, attr_name, None)
                if not callable(attr):
                    continue
                seen.add(attr_name)
                # Bind a closure that captures the name for the
                # error message.
                self.__dict__[attr_name] = _make_not_implemented_stub(
                    attr_name, sorted(self._CLUSTER_OVERRIDES),
                )

    # ── Cluster-mode lifecycle (P1.7.B foundation) ──────────────────────────

    def create_container(
        self,
        image: str,
        command: Any = None,
        *,
        detach: bool = True,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        entrypoint: Any = None,
        user: str | None = None,
        host_config: dict[str, Any] | None = None,
        platform: Any = None,
        queue_timeout_s: float | None = None,
        session_deadline_s: float | None = None,
        acquire_timeout_s: float | None = None,
        runtime: str | None = None,
        **_unused: Any,
    ) -> dict[str, str]:
        """In cluster mode: acquire a raw container via the
        ``Client`` and stash the resulting session on the
        ``ClusterContainerControl`` keyed by container_id. Returns
        the docker-API-shaped dict ``{"Id": <container_id>}`` —
        the high-level ``client.containers`` manager wraps that
        into a ``Container`` object the harness uses.

        ``detach=True`` is the only mode raw acquire supports
        (containers are spawned with their CMD running in the
        background; the harness ``exec_run``s into them after).
        Non-detach container.run is a follow-up.

        ``queue_timeout_s`` (issue #18, cluster-mode only): how long
        the control-plane admission queue will hold this acquire
        waiting for cluster capacity before failing. ``None`` uses
        the server default (3600 s). This is a drop-in *extension*
        kwarg — not part of docker-py's ``create_container`` — so it
        is silently ignored in local-Docker mode (no admission
        queue there). Raise it when deliberately over-requesting
        concurrency against a small cluster.

        ``session_deadline_s`` (issue #18, cluster-mode only):
        wall-clock cap on the acquired container's lifetime. The
        control plane force-destroys it once the cap passes — a
        safety net against a consumer that dies without calling
        ``remove``. ``None`` uses the server default (4 h). Also a
        drop-in extension kwarg, silently ignored in local mode.

        ``acquire_timeout_s`` (issue #12 / #18, cluster-mode only):
        wire deadline for the acquire round trip, which covers the
        node-side image pull. ``None`` uses the server default
        (600 s). Raise it for large images on a contended cluster
        where a cold pull can exceed 600 s. Also a drop-in extension
        kwarg, silently ignored in local mode.

        All three extension kwargs can equivalently be passed as the
        reserved labels ``xrlenv.queue_timeout_s`` /
        ``xrlenv.session_deadline_s`` / ``xrlenv.acquire_timeout_s``
        — the label path survives docker-py's high-level
        ``containers.create`` / ``.run``, which reject unknown
        kwargs.

        ``**_unused``: docker-py's ``create_container`` takes
        many more kwargs (volumes, ports, network_mode, ...).
        Most don't make sense on a remote-daemon path; we
        silently swallow them in cluster mode rather than fail
        on each caller's full kwarg menu. P1.7.B follow-ups can
        wire specific kwargs (e.g. cap_add for swebench).
        """
        if self._control.mode != "cluster":
            # Local mode: pass everything straight through to the
            # parent docker-py implementation; no kwarg validation or
            # rejection. The host docker daemon respects all docker-py
            # kwargs natively.
            return super().create_container(
                image, command, detach=detach, name=name, labels=labels,
                environment=environment, entrypoint=entrypoint, user=user,
                host_config=host_config, platform=platform, **_unused,
            )

        # Cluster mode: forward the docker-py kwargs that
        # ``acquire_container`` supports. The drop-in is intentionally
        # not authoritative for Level 1 / Level 2 policy — the control
        # plane sees the cluster's loaded ``nodes.yaml`` policy and is
        # the sole validator. The drop-in only fast-fails Level 3
        # (always-unsafe, no policy override) so operators can't
        # accidentally trip those locally; everything else flows to
        # the wire for the control plane to decide.
        #
        # Audit M1 (2026-05-13) walked back the previous "drop-in
        # validates against DEFAULT_POLICY" design: that pre-check
        # made operator opt-ins (``allow_host_network``,
        # ``allow_privileged``, ``allowed_host_paths``) and operator
        # extensions (``allowed_devices: [..., /dev/sda]``) unreachable
        # — the drop-in rejected before the cluster policy was ever
        # consulted. Single authoritative point keeps the operator's
        # ``nodes.yaml`` as the source of truth.
        if platform is not None:
            # Level 4: architectural mismatch, warn-and-drop (kept on
            # the legacy path for backward-compat — Level 4 isn't a
            # security boundary, just an advisory).
            _reject_platform_kwarg(platform)
        # Extract policy-relevant host_config fields. docker-py's
        # high-level ``containers.create(cap_add=..., devices=..., ...)``
        # bundles these into the ``HostConfig`` dict that lands here.
        cap_add: list[str] = []
        devices_list: list[str] = []
        privileged: bool = False
        network_mode: str | None = None
        pid_mode: str | None = None
        ipc_mode: str | None = None
        cgroup_parent: str | None = None
        cpuset_cpus: str | None = None
        cpuset_mems: str | None = None
        binds: list[str] = []
        # P0a — effective CPU/memory request the harness expressed via
        # host_config; threaded to the control plane as a scheduling input.
        cpu_limit: float | None = None
        mem_limit_bytes: int | None = None
        # P0b — container-shape RuntimeLimits the harness expressed via
        # host_config (pids / shm / tmpfs / read-only); scheduling-neutral.
        runtime_limits: RuntimeLimits | None = None
        # §5.4 — OCI runtime selector. docker-py maps both
        # ``containers.run(runtime="sysbox-runc")`` and
        # ``HostConfig(runtime=...)`` onto ``HostConfig.Runtime``, so that's
        # the primary path; the top-level ``runtime=`` kwarg (from the
        # signature above) is a fallback for callers/docker-py versions that
        # pass it through — HostConfig.Runtime wins if both are present.
        # host_config keys consumed by validation + wire forwarding;
        # anything outside this set falls through to _warn_unused_kwargs.
        # P0a — resource keys are now consumed (extracted or hard-errored
        # by _resolve_effective_cpu_mem), never silently warn-dropped.
        _consumed_host_config = {
            "CapAdd", "Devices", "Privileged", "NetworkMode",
            "PidMode", "IpcMode", "CgroupParent", "Binds", "Runtime",
        } | set(_RESOURCE_HOST_CONFIG_KEYS)
        if host_config:
            cap_add = list(host_config.get("CapAdd") or [])
            devices_list = _parse_devices_field(host_config.get("Devices"))
            privileged = bool(host_config.get("Privileged"))
            network_mode = host_config.get("NetworkMode") or None
            pid_mode = host_config.get("PidMode") or None
            ipc_mode = host_config.get("IpcMode") or None
            cgroup_parent = host_config.get("CgroupParent") or None
            cpuset_cpus = host_config.get("CpusetCpus") or None
            cpuset_mems = host_config.get("CpusetMems") or None
            binds = list(host_config.get("Binds") or [])
            # §5.4 — HostConfig.Runtime is where docker-py lands the runtime;
            # it wins over the top-level ``runtime=`` kwarg fallback.
            runtime = host_config.get("Runtime") or runtime
            # P0a — extract the harness's CPU/memory request (or
            # hard-error on a resource kwarg cluster mode can't honor)
            # instead of silently dropping it.
            cpu_limit, mem_limit_bytes = _resolve_effective_cpu_mem(
                host_config,
            )
            # P0b — extract the container-shape RuntimeLimits.
            runtime_limits = _resolve_runtime_limits(host_config)
            # Warn only on host_config entries the cluster path
            # genuinely doesn't honor.
            unsupported_host_config = {
                k: v for k, v in host_config.items()
                if k not in _consumed_host_config and v is not None and v != []
            }
            if unsupported_host_config:
                _warn_unused_kwargs(
                    "create_container[host_config]",
                    unsupported_host_config,
                )

        # Level-3 fast-fail (security boundary; no policy override
        # possible). Drop-in validates locally to give harness authors
        # an immediate error without an RPC round-trip — the control
        # plane would reject anyway. Level 1 / Level 2 are NOT checked
        # here (per audit M1) — the cluster policy is authoritative
        # and may permit them via operator config.
        from xrlenv.control.kwargs_policy import (
            DEFAULT_POLICY,
            KwargsPolicyViolation,
            validate_kwargs,
        )
        all_rejections = validate_kwargs(
            pid_mode=pid_mode,
            ipc_mode=ipc_mode,
            cgroup_parent=cgroup_parent,
            # P0a — cpuset_* are Level-3: CPU/memory placement is
            # cluster-owned (the node pins cores from its own ledger).
            cpuset_cpus=cpuset_cpus,
            cpuset_mems=cpuset_mems,
            # ``network_mode=container:...`` is Level 3; ``=host`` is
            # Level 2 (deferred to control plane). Pass it in and
            # filter the result.
            network_mode=network_mode,
            policy=DEFAULT_POLICY,
        )
        level_3 = [r for r in all_rejections if r.level == 3]
        if level_3:
            raise KwargsPolicyViolation(level_3)

        _warn_unused_kwargs("create_container", _unused)
        # Normalize command: docker-py accepts string, list, or None.
        if isinstance(command, str):
            command_list = command.split()
        elif command is None:
            command_list = None
        else:
            command_list = list(command)
        # Normalize entrypoint to ``list[str] | None`` for the
        # acquire_container kwarg. docker-py accepts a string (split
        # on whitespace, matching CMD-string semantics) or a list;
        # ``""`` is the docker CLI's "clear the entrypoint" idiom,
        # which we preserve as a single-element ``[""]`` so the wire
        # carries the operator's intent unambiguously (proto3
        # ``repeated`` collapses unset and empty-list, so we never
        # send a literal empty list).
        entrypoint_list: list[str] | None
        if entrypoint is None:
            entrypoint_list = None
        elif isinstance(entrypoint, str):
            entrypoint_list = [""] if entrypoint == "" else entrypoint.split()
        else:
            ep = list(entrypoint)
            entrypoint_list = ep if ep else [""]
        # P1.7.B.3: merge contextvar-scoped metadata into the
        # outgoing labels dict. ``xrlenv.rollout_metadata(...)``
        # in the smoke driver's per-instance scope sets the
        # contextvar; here we read it and emit
        # ``xrlenv.rollout.artifact_path`` /
        # ``xrlenv.rollout.displayed_name`` docker labels. The
        # control plane's RawContainerCoordinator parses those
        # keys off ``AcquireContainerCommand.labels`` and writes
        # them to typed columns on ``RawRolloutRecord``.
        # Operator-passed labels via ``containers.create(labels=...)``
        # are preserved; xrlenv-reserved keys take precedence on
        # conflict (operator can't override the cluster's view).
        from xrlenv.compat.metadata import (
            LABEL_TASK_KEY,
            current_rollout_metadata,
            metadata_to_labels,
        )
        # Read the rollout metadata ONCE on THIS (calling) thread — the contextvar the
        # consumer set with ``xrlenv.rollout_metadata(...)`` is visible here, but not inside
        # ``_go()`` which runs on the runner thread. ``displayed_name`` is the correlation key
        # for the infra-failure record below (audit M8).
        _rollout_meta = current_rollout_metadata()
        meta_labels = metadata_to_labels(_rollout_meta)
        if meta_labels:
            merged_labels = dict(labels or {})
            merged_labels.update(meta_labels)
        elif labels:
            merged_labels = dict(labels)
        else:
            merged_labels = None  # type: ignore[assignment]

        # Promote the reserved ``xrlenv.task_key`` label (operator-
        # passed, never xrlenv-emitted) into the first-class scheduler
        # kwarg. The scheduler uses ``task_key`` for anti-affinity —
        # it has to be on the AcquireContainerRequest field, not just
        # a docker label, for ``max_runs_per_task`` to actually fire.
        # Operators write the conventional label key in coding-bench
        # / swebench harnesses; we hoist it here so they don't need
        # an xrlenv-shaped API to pass it.
        task_key: str | None = None
        if merged_labels is not None and LABEL_TASK_KEY in merged_labels:
            tk = merged_labels.pop(LABEL_TASK_KEY)
            if isinstance(tk, str) and tk:
                task_key = tk
            if not merged_labels:
                merged_labels = None  # don't send an empty labels dict

        # Issue #18 (Ask #1) — hoist the reserved
        # ``xrlenv.queue_timeout_s`` label into the first-class
        # acquire kwarg, mirroring the ``task_key`` hoist above.
        # Harnesses drive the drop-in through docker-py's high-level
        # ``client.containers.create(...)`` / ``.run(...)``, which
        # rejects unknown kwargs before they reach this override —
        # so a label is the robust way for a consumer to set the
        # admission-queue deadline without an xrlenv-shaped API.
        # An explicit ``queue_timeout_s=`` kwarg (works when calling
        # the low-level ``api.create_container`` directly) takes
        # precedence over the label.
        effective_queue_timeout_s = queue_timeout_s
        if (
            effective_queue_timeout_s is None
            and merged_labels is not None
            and _LABEL_QUEUE_TIMEOUT_S in merged_labels
        ):
            raw = merged_labels.pop(_LABEL_QUEUE_TIMEOUT_S)
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "xrlenv cluster: ignoring non-numeric %s label "
                    "%r — using the server default queue timeout.",
                    _LABEL_QUEUE_TIMEOUT_S, raw,
                )
            else:
                if parsed > 0:
                    effective_queue_timeout_s = parsed
            if not merged_labels:
                merged_labels = None  # don't send an empty labels dict

        # Issue #18 — same hoist for the ``xrlenv.session_deadline_s``
        # label → ``session_deadline_s`` acquire kwarg (the raw-session
        # lifetime cap). Explicit kwarg wins over the label.
        effective_session_deadline_s = session_deadline_s
        if (
            effective_session_deadline_s is None
            and merged_labels is not None
            and _LABEL_SESSION_DEADLINE_S in merged_labels
        ):
            raw = merged_labels.pop(_LABEL_SESSION_DEADLINE_S)
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "xrlenv cluster: ignoring non-numeric %s label "
                    "%r — using the server default session deadline.",
                    _LABEL_SESSION_DEADLINE_S, raw,
                )
            else:
                if parsed > 0:
                    effective_session_deadline_s = parsed
            if not merged_labels:
                merged_labels = None  # don't send an empty labels dict

        # Issue #12 / #18 — same hoist for ``xrlenv.acquire_timeout_s``
        # → ``acquire_timeout_s`` acquire kwarg (the pull / acquire
        # wire deadline). Explicit kwarg wins over the label.
        effective_acquire_timeout_s = acquire_timeout_s
        if (
            effective_acquire_timeout_s is None
            and merged_labels is not None
            and _LABEL_ACQUIRE_TIMEOUT_S in merged_labels
        ):
            raw = merged_labels.pop(_LABEL_ACQUIRE_TIMEOUT_S)
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "xrlenv cluster: ignoring non-numeric %s label "
                    "%r — using the server default acquire timeout.",
                    _LABEL_ACQUIRE_TIMEOUT_S, raw,
                )
            else:
                if parsed > 0:
                    effective_acquire_timeout_s = parsed
            if not merged_labels:
                merged_labels = None  # don't send an empty labels dict

        ctrl = self._cluster_control

        async def _go() -> str:
            session = await ctrl.client.acquire_container(
                image=image,
                command=command_list,
                entrypoint=entrypoint_list,
                user=user or None,
                cap_add=cap_add or None,
                devices=devices_list or None,
                privileged=privileged,
                network_mode=network_mode,
                binds=binds or None,
                name=name,
                labels=merged_labels,
                environment=environment,
                task_key=task_key,
                queue_timeout_s=effective_queue_timeout_s,
                session_deadline_s=effective_session_deadline_s,
                acquire_timeout_s=effective_acquire_timeout_s,
                # P2/P6 — per-rollout cpuset-pinning hint from the
                # ``xrlenv.rollout_metadata(cpu_isolation=...)`` contextvar
                # (read once above as ``_rollout_meta``). Lets a drop-in
                # harness pin a SPECIFIC task's container to whole cores so
                # OpenMP/BLAS don't oversubscribe the CFS quota. ``OFF``
                # (default) is unchanged behavior.
                cpu_isolation=_rollout_meta.cpu_isolation,
                # P0a — harness CPU/memory request as a scheduling input.
                cpu_limit=cpu_limit,
                mem_limit_bytes=mem_limit_bytes,
                # P0b — container-shape RuntimeLimits.
                runtime_limits=runtime_limits,
                # §5.4 — OCI runtime selector (HostConfig.Runtime or the
                # top-level runtime= kwarg). None = docker default runtime.
                container_runtime=runtime,
            )
            ctrl.register_session(session, image=image)
            return session.container_id

        corr = _rollout_meta.displayed_name
        try:
            container_id = _run_sync(_go(), runner=self._cluster_control.runner)
        except XRLEnvError as exc:
            # Record the acquire failure KIND at the mechanism boundary BEFORE it propagates
            # into the harness, which wraps it (swebench's BuildImageError) or swallows it
            # (``completed=false``). ``_infra_kind`` recovers the concrete kind even from a
            # wire-flattened bare ``XRLEnvError`` (audit M8). Keyed by the rollout
            # displayed_name; the consuming adapter owns the retry policy.
            if corr:
                self._record_infra_failure(corr, _infra_kind(exc))
            raise
        # Correlate this live container with its rollout so a POST-acquire op failure (exec /
        # archive, which runs without the consumer's contextvar) can be recorded too (audit M8).
        if corr:
            self._remember_container_rollout(container_id, corr)
        # docker-py's create_container returns a dict with at
        # least ``Id``; the high-level manager wraps it.
        return {"Id": container_id, "Warnings": []}

    def start(
        self, container: Any, *_args: Any, **_kwargs: Any,
    ) -> None:
        """In cluster mode: no-op. ``acquire_container`` already
        spawned the container detached + running. docker-py's
        flow is ``create_container`` returns an unstarted dict,
        then ``container.start()`` activates it; we fold both
        into the acquire call so ``.start()`` has nothing to do.
        """
        if self._control.mode != "cluster":
            return super().start(container, *_args, **_kwargs)
        # Container is already running post-acquire. Nothing to do.
        return None

    def stop(
        self, container: Any, *_args: Any, **_kwargs: Any,
    ) -> None:
        """In cluster mode: no-op (the harness usually calls
        ``stop`` followed by ``remove`` — we let ``remove`` do
        the actual ``docker rm -f`` since raw containers are
        ephemeral per evaluation). The harness sees no error;
        the container stays up until ``remove_container`` fires.
        """
        if self._control.mode != "cluster":
            return super().stop(container, *_args, **_kwargs)
        return None

    def remove_container(
        self, container: Any, *_args: Any,
        force: bool = False, **_kwargs: Any,
    ) -> None:
        """In cluster mode: destroy the session via the
        ``Client``. Idempotent — if the session was already
        destroyed (or was never registered with this drop-in),
        we silently return rather than raise.

        **Cluster-mode deviation from docker-py**: ``force`` is
        always treated as True regardless of the caller's flag.
        Cluster ``stop()`` is a no-op (raw containers are
        ephemeral per evaluation; we let ``remove`` do the real
        ``docker rm -f``); a non-force remove against a still-
        running raw container would fail otherwise. swebench's
        typical ``container.stop(); container.remove(force=True)``
        flow is unaffected; a less-common ``container.stop();
        container.remove()`` (no force) Just Works under cluster
        mode where it would error against a real docker daemon.
        Audit Cluster-Dropin-M2 closure.
        """
        if self._control.mode != "cluster":
            return super().remove_container(
                container, *_args, force=force, **_kwargs,
            )
        container_id = _container_id_arg(container)
        ctrl = self._cluster_control
        try:
            session = ctrl.get_session(container_id)
        except docker.errors.NotFound:
            return None  # idempotent — session unknown / already gone

        async def _go() -> None:
            # Always force in cluster mode — see docstring.
            await session.destroy(force=True)
        try:
            # Plain _run_sync — do NOT record a teardown failure into the infra-evidence
            # channel (audit M8 last-write-wins): teardown runs AFTER the eval and upstream
            # swallows its exceptions too, so recording here would let a teardown NodeLost
            # OVERWRITE the eval's outcome — masking a retryable eval failure, or turning a
            # deterministic content failure into a spurious infra retry. The eval-stage record
            # (acquire/exec/stream/archive) is the only evidence the adapter should act on.
            _run_sync(_go(), runner=self._cluster_control.runner)
        finally:
            # Always clean up local state — a NodeLost mid-destroy must not leak the local
            # session or the correlation map (audit Low). Capacity release stays
            # control-plane-driven (invariant 2: freed only on node-confirmed destroy), so
            # dropping our local refs here is safe.
            ctrl.drop_session(container_id)
            self._forget_container(container_id)
        return None

    # ── Exec triple (P1.7.B batched; streaming follow-up) ──────────────────

    def exec_create(
        self,
        container: Any,
        cmd: Any,
        *,
        stdout: bool = True,
        stderr: bool = True,
        stdin: bool = False,
        tty: bool = False,
        privileged: bool = False,
        user: str = "",
        environment: dict[str, str] | None = None,
        workdir: str | None = None,
        **_unused: Any,
    ) -> dict[str, str]:
        """Stage an exec for ``exec_start``. docker-py's contract
        is two-step: ``exec_create`` returns ``{"Id": exec_id}``;
        ``exec_start(exec_id)`` actually runs it. We mirror that
        by stashing the call info on this client and returning a
        synthetic id.

        Cluster mode honors: ``cmd``, ``user``, ``environment``,
        ``workdir``. ``stdout`` / ``stderr`` / ``stdin`` / ``tty``
        / ``privileged`` are docker-API flags the raw-container
        path doesn't expose; we silently swallow them. Streaming
        flags belong to ``exec_start`` per docker-py's design.
        """
        if self._control.mode != "cluster":
            return super().exec_create(
                container, cmd, stdout=stdout, stderr=stderr,
                stdin=stdin, tty=tty, privileged=privileged,
                user=user, environment=environment,
                workdir=workdir, **_unused,
            )
        _warn_unused_kwargs("exec_create", _unused)
        container_id = _container_id_arg(container)
        # Normalize cmd: docker-py accepts string or list.
        if isinstance(cmd, str):
            cmd_list: list[str] = ["sh", "-c", cmd]
        else:
            cmd_list = list(cmd)
        self._exec_counter += 1
        exec_id = f"xrlenv-exec-{self._exec_counter:08d}"
        self._exec_pending[exec_id] = {
            "container_id": container_id,
            "cmd": cmd_list,
            "user": user or None,
            "environment": environment,
            "workdir": workdir,
        }
        return {"Id": exec_id}

    def exec_start(
        self,
        exec_id: str,
        detach: bool = False,
        tty: bool = False,
        stream: bool = False,
        socket: bool = False,
        demux: bool = False,
    ) -> Any:
        """Run the exec staged by ``exec_create``. Cluster mode
        currently supports only the **batched** path
        (``stream=False``); ``stream=True`` is the swebench
        ``exec_run_with_timeout`` path which lands as a follow-up
        once the sync-iterator-from-async-generator bridge is
        in place. ``socket=True`` (raw socket attach) is not
        supported in cluster mode and unlikely to ever be
        (raw-container path doesn't expose docker streams
        directly).

        Return shape mirrors docker-py: when ``demux=True``
        returns ``(stdout_bytes, stderr_bytes)``; otherwise
        returns combined ``stdout+stderr`` bytes.
        """
        if self._control.mode != "cluster":
            return super().exec_start(
                exec_id, detach=detach, tty=tty,
                stream=stream, socket=socket, demux=demux,
            )
        if stream:
            return self._exec_start_streaming(exec_id, demux=demux)
        if socket:
            raise NotImplementedError(
                "xrlenv docker drop-in (cluster mode): "
                "``api.exec_start(socket=True)`` is not "
                "supported (raw-container path doesn't expose "
                "docker streams).",
            )
        if detach:
            # Validate the exec was STAGED first — an unknown id is a caller bug, not a
            # success (audit M2): don't fake a result for an id we never saw.
            detach_pending = self._exec_pending.get(exec_id)
            if detach_pending is None:
                raise docker.errors.NotFound(
                    f"xrlenv docker drop-in: exec {exec_id!r} not staged via exec_create",
                )
            # The ONLY detached exec we tolerate is swebench's timeout-watchdog kill. Upstream
            # (docker_utils.exec_run_with_timeout) does ``container.exec_run("kill -TERM
            # <pid>", detach=True)``; our exec_create wraps a string cmd as ``["sh", "-c",
            # cmd]``, and we always report Pid 0, so the STAGED cmd is EXACTLY this list.
            # Match it exactly (representation-exact, audit Low) — NOT any command that merely
            # contains a ``kill`` token (``echo kill``), a different pid (``kill -TERM 1`` is
            # not our Pid-0 watchdog), or a whitespace variant. Cluster execs have no host PID
            # and we issue no in-container kill; real termination is the cluster exec timeout +
            # teardown. Any other detached command is unsupported — fail loudly, not a fake
            # success.
            _cmd = detach_pending.get("cmd") or []
            if _cmd != _WATCHDOG_KILL_CMD:
                raise NotImplementedError(
                    "xrlenv docker drop-in (cluster mode): api.exec_start(detach=True) is "
                    "only supported for the timeout-watchdog kill; got cmd "
                    f"{detach_pending.get('cmd')!r}. Use the batched (stream=False) path.",
                )
            self._exec_pending.pop(exec_id, None)
            # Synthetic exit_code so docker-py's Container.exec_run -> exec_inspect resolves.
            self._exec_results[exec_id] = {"exit_code": 0}
            LOGGER.debug("cluster exec_start(detach=True) watchdog-kill no-op for %s", exec_id)
            return (b"", b"") if demux else b""
        pending = self._exec_pending.get(exec_id)
        if pending is None:
            raise docker.errors.NotFound(
                f"xrlenv docker drop-in: exec {exec_id!r} not "
                f"staged via exec_create",
            )
        ctrl = self._cluster_control
        session = ctrl.get_session(pending["container_id"])

        async def _go() -> Any:
            return await session._transport.container_exec(  # type: ignore[attr-defined]
                rollout_id=session.rollout_id,
                container_id=session.container_id,
                cmd=pending["cmd"],
                # ``timeout_s`` is per-exec, but docker-py's
                # exec_start has no timeout arg — we pass a
                # generous 30 min. Watchdog-driven harnesses
                # (swebench's exec_run_with_timeout) need the
                # streaming path for real-time cancellation;
                # batched callers eat the wall-clock as an
                # implementation detail.
                timeout_s=1800.0,
                cwd=pending["workdir"],
                env=pending["environment"],
                user=pending["user"],
            )
        result = self._run_op(_go(), container_id=session.container_id)
        # Normalise to a dict so ``_exec_results`` carries the
        # same shape on both code paths (the streaming terminator
        # also stores a chunk dict). ``container_exec`` returns
        # a ``RawExecResult`` dataclass on the gRPC + in-process
        # transports; older fakes return a dict — accept either.
        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            }
        # Stash so exec_inspect can read ExitCode.
        self._exec_results[exec_id] = result_dict
        # Pop from pending — docker semantics is one-shot.
        self._exec_pending.pop(exec_id, None)
        stdout = result_dict.get("stdout") or b""
        stderr = result_dict.get("stderr") or b""
        if demux:
            return (stdout, stderr)
        return stdout + stderr

    def _exec_start_streaming(
        self, exec_id: str, *, demux: bool,
    ) -> Iterator[Any]:
        """Sync iterator over an async ``ClusterContainerSession.exec_stream``.

        Bridges the docker-py contract (``api.exec_start(stream=True)``
        returns a synchronous iterator) onto our async SDK by
        running the async iteration in a daemon thread that
        feeds a ``queue.Queue``; the returned sync iterator
        pulls off the queue.

        Bridges out:

        - **Heartbeat chunks** (empty stdout AND stderr,
          ``done=False`` — emitted by ``RawContainerManager.exec_stream``
          to keep idle TCP alive on quiet runs) are **filtered**.
          docker-py wouldn't yield empty chunks; harnesses
          might be confused by them.
        - **Terminator chunk** (``done=True``) ends iteration.
          Its ``exit_code`` is stashed on ``_exec_results`` so a
          subsequent ``api.exec_inspect(exec_id)`` reads the
          right value (matching swebench's
          ``exec_run_with_timeout`` flow which calls
          ``exec_inspect`` after the iterator exhausts).
        - **Drain-side errors** (gRPC blip, node disconnect,
          chunk-timeout from the consumer-side
          ``_send_and_stream``) propagate as the next
          ``__next__`` call's exception.
        """
        pending = self._exec_pending.pop(exec_id, None)
        if pending is None:
            raise docker.errors.NotFound(
                f"xrlenv docker drop-in: exec {exec_id!r} not "
                f"staged via exec_create",
            )
        ctrl = self._cluster_control
        session = ctrl.get_session(pending["container_id"])

        # Mark in-flight so exec_inspect reports Running:True (with a benign Pid) if a
        # caller's timeout watchdog inspects mid-stream — swebench's exec_run_with_timeout
        # does exactly this before trying to kill the exec (audit M2). Cleared on the
        # terminator chunk, when the exit_code lands in _exec_results.
        self._exec_streaming.add(exec_id)

        def _on_terminator(chunk_dict: dict[str, Any]) -> None:
            self._exec_results[exec_id] = chunk_dict

        def _on_error(exc: BaseException) -> None:
            # Record a STREAMING-exec infra failure at the mechanism boundary (audit M8) — this
            # is swebench's main /eval.sh path. Only XRLEnvErrors are cluster failures; the
            # adapter filters the kind for retry policy.
            if isinstance(exc, XRLEnvError):
                self._record_op_failure(session.container_id, exc)

        return _SyncStreamIterator(
            session=session,
            cmd=pending["cmd"],
            cwd=pending["workdir"],
            env=pending["environment"],
            user=pending["user"],
            demux=demux,
            runner=self._cluster_control.runner,
            on_terminator=_on_terminator,
            on_error=_on_error,
            # Clear the active marker on ANY terminal path (terminator OR error OR short
            # stream) so a stream that raises before its terminator can't leave the exec
            # reported Running forever (audit M9).
            on_close=lambda: self._exec_streaming.discard(exec_id),
        )

    def exec_inspect(self, exec_id: str) -> dict[str, Any]:
        """Return the stored result for ``exec_id``. docker-py's contract is a rich dict;
        we return the minimum shape callers actually read: ``ExitCode`` + ``Running`` +
        ``Pid``.

        ``Pid`` is always 0 in cluster mode (audit M2): a real docker exec_inspect returns
        a host PID while running and 0 when finished, but a cluster exec runs remotely with
        no host PID. swebench's ``exec_run_with_timeout`` reads ``["Pid"]`` on a test
        timeout to ``kill -TERM`` the exec — omitting the key crashed that watchdog cleanup.
        With 0 the upstream kill is a contained no-op; real termination is handled by the
        cluster exec's own timeout + container teardown. A mid-stream exec reports
        ``Running: True`` (rather than raising NotFound) so the same watchdog can inspect it
        before the terminator lands."""
        if self._control.mode != "cluster":
            return super().exec_inspect(exec_id)
        result = self._exec_results.get(exec_id)
        if result is not None:
            return {
                "ID": exec_id,
                "ExitCode": int(result.get("exit_code") or 0),
                "Running": False,
                "Pid": 0,
            }
        if exec_id in self._exec_streaming:
            return {"ID": exec_id, "ExitCode": 0, "Running": True, "Pid": 0}
        raise docker.errors.NotFound(
            f"xrlenv docker drop-in: exec {exec_id!r} not found (was it started?)",
        )

    # ── Archives (P1.7.B) ──────────────────────────────────────────────────

    def put_archive(
        self, container: Any, path: str, data: bytes,
    ) -> bool:
        """Extract ``data`` (tar bytes) into ``path`` inside the
        container. docker-py's contract returns bool; cluster
        mode raises XRLEnvError on failure (which docker-py never
        does — it returns False). Caller should treat ``True``
        as success and rely on the exception path for diagnosis."""
        if self._control.mode != "cluster":
            return super().put_archive(container, path, data)
        container_id = _container_id_arg(container)
        session = self._cluster_control.get_session(container_id)

        async def _go() -> None:
            await session.put_archive(path, data)
        self._run_op(_go(), container_id=container_id)
        return True

    def get_archive(
        self,
        container: Any,
        path: str,
        chunk_size: int = 2097152,
        encode_stream: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """Tar up ``path`` inside the container and return
        ``(iter_of_chunks, stat_dict)`` per docker-py's contract.

        Cluster mode collapses the tarball to a single chunk
        (the SDK's ``get_archive`` returns full bytes; chunking
        is a docker-py affordance our wire doesn't propagate).
        ``stat_dict`` carries just the path for callers who
        only check ``stat["name"]``.
        """
        if self._control.mode != "cluster":
            return super().get_archive(
                container, path, chunk_size=chunk_size,
                encode_stream=encode_stream,
            )
        container_id = _container_id_arg(container)
        session = self._cluster_control.get_session(container_id)

        async def _go() -> bytes:
            return await session.get_archive(path)
        tarball = self._run_op(_go(), container_id=container_id)
        return (iter([tarball]), {"name": path})

    # ── Image ops (P1.7.B operator-pre-pulled contract) ────────────────────
    #
    # In cluster mode the consumer's host is NOT where the
    # container runs — image presence is a per-node property.
    # The raw-container path's contract is "operator pre-pulls
    # images on every node before consumers acquire" (see
    # ``RawContainerManager._ensure_image_present`` — acquire
    # fast-fails with XRLEnvError if the chosen node lacks the
    # image).
    #
    # Given that contract, the docker-py image methods behave as
    # operator-trust no-ops:
    #
    # - ``inspect_image`` — return synthetic Image dict (the
    #   harness's "is this image local" probe always says yes;
    #   if the chosen node actually lacks it, ``acquire_container``
    #   surfaces the error fast).
    # - ``pull`` — log + return success. The image cache layer
    #   (P1.2 / P1.6) handles distribution at the operator's
    #   request, not here.
    # - ``images`` — return ``[]``. The consumer-side host has
    #   no docker images to enumerate (the daemon lives on
    #   nodes); harnesses use this for cleanup loops which
    #   become no-ops, harmless.
    # - ``remove_image`` — log + no-op. Image lifetime is owned
    #   by the per-node LRU cache; the consumer can't unilaterally
    #   evict a shared image.
    # - ``history`` — return ``[]``. swebench uses this for
    #   ``find_dependent_images`` cleanup; empty history yields
    #   no dependents to clean, also harmless.
    #
    # Callers wanting REAL per-node image management should use
    # the operator surface (``xrlenv build apply`` for builds,
    # ``deploy/ship-images.sh`` for distribution, the admin
    # ``/images`` view for visibility).

    def pull(
        self,
        repository: str,
        tag: str | None = None,
        stream: bool = False,
        **_unused: Any,
    ) -> Any:
        """In cluster mode: log + return success. Per the
        operator-pre-pulled contract, image distribution happens
        outside this RPC; the harness's pull is a probe-style
        call we satisfy without actually fetching anything."""
        if self._control.mode != "cluster":
            return super().pull(
                repository, tag=tag, stream=stream, **_unused,
            )
        _warn_unused_kwargs("pull", _unused)
        ref = repository if tag is None else f"{repository}:{tag}"
        LOGGER.info(
            "xrlenv docker drop-in (cluster): api.pull(%r) is a "
            "no-op — operator pre-pull contract. Will fast-fail "
            "at acquire_container if the chosen node actually "
            "lacks the image.",
            ref,
        )
        # docker-py's stream-mode pull yields progress dicts;
        # non-stream returns a string. Match shape minimally.
        if stream:
            return iter([
                {"status": f"pull no-op (cluster mode): {ref}"},
            ])
        return f"pull no-op (cluster mode): {ref}"

    def inspect_image(self, image: str) -> dict[str, Any]:
        """In cluster mode: return synthetic Image dict so the
        harness's "is this image local" probe always succeeds.
        The real check happens at ``acquire_container`` time
        (per-node ImageNotFound surfaces as a clean
        XRLEnvError)."""
        if self._control.mode != "cluster":
            return super().inspect_image(image)
        # Minimum fields docker-py's Image wrapper + harnesses
        # actually read.
        return {
            "Id": f"sha256:{'0' * 64}",  # synthetic
            "RepoTags": [image],
            "RepoDigests": [],
            "Config": {},
            "Architecture": "amd64",
        }

    def images(
        self,
        name: str | None = None,
        all: bool = False,
        filters: Any = None,
        **_unused: Any,
    ) -> list[dict[str, Any]]:
        """In cluster mode: return ``[]``. The consumer-side
        host has no docker images of its own to list; harnesses
        iterate this for cleanup loops which become no-ops.
        Operators wanting visibility into per-node image caches
        should use the admin panel's ``/images`` view (B7.6,
        P1.2.c)."""
        if self._control.mode != "cluster":
            return super().images(
                name=name, all=all, filters=filters, **_unused,
            )
        _warn_unused_kwargs("images", _unused)
        return []

    def containers(
        self,
        quiet: bool = False,
        all: bool = False,
        trunc: bool = False,
        latest: bool = False,
        since: str | None = None,
        before: str | None = None,
        limit: int = -1,
        size: bool = False,
        filters: Any = None,
        **_unused: Any,
    ) -> list[dict[str, Any]]:
        """In cluster mode: enumerate the still-alive containers this
        drop-in instance created (the registry the
        ``ClusterContainerControl`` already maintains for
        exec/destroy lookups). Returned as docker-API-shaped dicts;
        the high-level ``client.containers.list()`` wraps them into
        ``Container`` model instances the harness reads.

        Why this is a meaningful answer rather than ``[]``: swebench's
        grader calls ``client.containers.list(all=True)`` from
        ``swebench.harness.reporting.make_run_report`` to count
        leftover containers whose name contains the ``run_id`` —
        a real leak-detection signal. The drop-in already drops a
        session from the registry on ``destroy``, so every entry
        here is genuinely "still alive from this client's
        perspective." swebench's filter by ``run_id in name`` picks
        the relevant subset.

        Limitations:
        - Scope is per-client-instance. Containers another
          ``xrlenv.from_env(...)`` instance or a direct
          ``Client.acquire_container`` created aren't visible here.
        - All entries report ``State: running`` since we don't carry
          phase info on sessions; this is fine for swebench's
          run_id-in-name filter.
        - The ``filters`` / ``since`` / ``before`` etc. docker-API
          kwargs are not honored — we warn via _warn_unused_kwargs
          if non-trivial values are passed."""
        if self._control.mode != "cluster":
            return super().containers(
                quiet=quiet, all=all, trunc=trunc, latest=latest,
                since=since, before=before, limit=limit, size=size,
                filters=filters, **_unused,
            )
        _warn_unused_kwargs("containers", _unused)
        out: list[dict[str, Any]] = []
        for container_id, (session, image) in (
            self._cluster_control._sessions.items()
        ):
            container_name = getattr(session, "container_name", None) or ""
            names = [f"/{container_name}"] if container_name else []
            out.append({
                "Id": container_id,
                "Names": names,
                "Image": image,
                "State": "running",
                "Status": "Up",
            })
        return out

    def remove_image(
        self,
        image: str,
        force: bool = False,
        noprune: bool = False,
    ) -> None:
        """In cluster mode: log + no-op. Image lifetime is owned
        by the per-node LRU cache (P1.2.b); a consumer-side
        ``api.remove_image`` can't unilaterally evict a shared
        image without breaking other rollouts that depend on
        it."""
        if self._control.mode != "cluster":
            return super().remove_image(
                image, force=force, noprune=noprune,
            )
        LOGGER.info(
            "xrlenv docker drop-in (cluster): api.remove_image"
            "(%r) is a no-op — image lifetime owned by the "
            "per-node LRU cache.",
            image,
        )
        return None

    def history(self, image: str) -> list[dict[str, Any]]:
        """In cluster mode: return ``[]``. swebench uses
        ``image.history()`` to find dependent images for
        cleanup; empty history yields no dependents which means
        no-op cleanup — consistent with our remove_image no-op."""
        if self._control.mode != "cluster":
            return super().history(image)
        return []

    def inspect_container(self, container: Any) -> dict[str, Any]:
        """In cluster mode: return the minimum docker-API-shaped
        inspect dict that docker-py's high-level manager needs
        to construct a ``Container`` object. The harness
        sometimes calls inspect to read state; we return the
        labels we set at acquire + the container id + name.
        Full docker inspect is a follow-up if needed."""
        if self._control.mode != "cluster":
            return super().inspect_container(container)
        container_id = _container_id_arg(container)
        session = self._cluster_control.get_session(container_id)
        image = self._cluster_control.get_image(container_id)
        return {
            "Id": session.container_id,
            "Name": f"/{session.container_name}",
            "Config": {
                "Image": image,
                "Labels": {
                    "xrlenv.rollout_id": session.rollout_id,
                    "xrlenv.session_kind": "raw",
                },
            },
            "State": {
                "Status": "exited" if session.destroyed else "running",
                "Running": not session.destroyed,
                "ExitCode": 0,
            },
        }


_STREAM_SENTINEL = object()


class _SyncStreamIterator:
    """Sync iterator that drains an async ``ClusterContainerSession.exec_stream``
    via a daemon thread + ``queue.Queue`` bridge.

    Lifetime: the daemon thread starts on construction (so
    iteration latency reflects the async stream's natural
    throughput, not a per-``__next__`` round-trip). The thread
    closes when the async generator exhausts (terminator chunk
    OR exception).
    """

    def __init__(
        self,
        *,
        session: ClusterContainerSession,
        cmd: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | None,
        demux: bool,
        on_terminator: Any,
        on_close: Any = None,
        on_error: Any = None,
        runner: _DropInRunner | None = None,
    ) -> None:
        self._demux = demux
        self._on_terminator = on_terminator
        # Fires with the exception if the drain raises — the mechanism boundary for a
        # STREAMING-exec infra failure (audit M8). swebench runs the main /eval.sh through the
        # streaming path, so a stream-side NodeLost must be recorded here, not just on the
        # batched/archive paths.
        self._on_error = on_error
        # Fires exactly once on ANY terminal path (terminator, sentinel, or error) — used
        # to clear the drop-in's active-exec marker so a stream that raises before its
        # terminator doesn't leave the exec reported Running forever (audit M9).
        self._on_close = on_close
        self._closed = False
        self._close_lock = threading.Lock()
        # ``queue.Queue`` is thread-safe + sync (the docker-py
        # contract is sync). The drain thread puts items on the
        # queue; ``__next__`` pulls them off.
        self._queue: queue.Queue[Any] = queue.Queue()

        async def _drain() -> None:
            try:
                async for chunk in session.exec_stream(
                    cmd, cwd=cwd, env=env, user=user,
                ):
                    item = {
                        "stdout": chunk.stdout,
                        "stderr": chunk.stderr,
                        "done": chunk.done,
                        "exit_code": chunk.exit_code,
                        "timed_out": chunk.timed_out,
                    }
                    # PRODUCER-side terminator handling (audit M9): stash the result now so
                    # a caller that ABANDONS the stream (never drains the terminator) still
                    # gets a completed exec_inspect, not Running-forever.
                    if chunk.done:
                        self._on_terminator(item)
                    self._queue.put(item)
            except BaseException as exc:
                # Catch BaseException, not Exception (audit Low): ``asyncio.CancelledError`` is
                # a BaseException, so an Exception-only handler would let a CANCELLED stream skip
                # on_error, reach ``finally``, and queue the ordinary sentinel — the consumer
                # would see a clean ``StopIteration`` with no failure evidence.
                #
                # Record the failure at the mechanism boundary BEFORE it reaches the consumer
                # (swebench's watchdog thread), which swallows it into completed=false (audit
                # M8). The callback must never mask the original exception, so guard it against
                # ANY raise. Then surface the exception to the consumer via the queue, and
                # re-raise a CancelledError so cancellation is honored (the task actually stops).
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except BaseException:
                        LOGGER.debug("stream on_error callback failed", exc_info=True)
                self._queue.put(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return
            finally:
                # Producer finished (terminator / exhaustion / error) — clear the active
                # marker independently of any further consumer iteration (audit M9).
                self._close()
                self._queue.put(_STREAM_SENTINEL)

        if runner is not None:
            # Dispatch the drain coroutine onto the runner's
            # loop. ``run_coroutine_threadsafe`` returns a
            # ``concurrent.futures.Future`` we don't otherwise
            # need — the queue serves as the back-channel; the
            # drain is fire-and-forget from this thread's POV.
            asyncio.run_coroutine_threadsafe(_drain(), runner.loop)
            self._thread: threading.Thread | None = None
        else:
            # No runner — fall back to a fresh-loop drain on a
            # daemon thread. Only safe when the session's
            # transport / Client doesn't carry loop-bound state
            # (LocalRuntime in-process happens to be OK; gRPC
            # is not).
            def _thread_target() -> None:
                # The drain re-raises a CancelledError to honor task cancellation on the runner
                # loop; in this no-runner thread fallback nothing cancels it, and it has already
                # been surfaced to the consumer via the queue — so swallow it here rather than
                # leak an unhandled thread exception (audit Low).
                import contextlib
                with contextlib.suppress(asyncio.CancelledError):
                    asyncio.run(_drain())

            self._thread = threading.Thread(
                target=_thread_target,
                name=f"xrlenv-exec-stream-{session.container_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    def __iter__(self) -> Iterator[Any]:
        return self

    def _close(self) -> None:
        """Idempotently fire on_close (clears the active-exec marker). Called on every
        terminal path — by the PRODUCER when the drain finishes AND by the consumer's
        ``__next__`` — so an abandoned/errored/short stream never leaks Running state
        (audit M9). Locked because producer + consumer threads race here."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if self._on_close is not None:
            self._on_close()

    def __next__(self) -> Any:
        while True:
            item = self._queue.get()
            if item is _STREAM_SENTINEL:
                self._close()
                raise StopIteration
            if isinstance(item, BaseException):
                # BaseException, not Exception (audit Low): re-raise a queued
                # ``asyncio.CancelledError`` too, so a cancelled stream surfaces as an error to
                # the consumer rather than reading as clean exhaustion (StopIteration).
                self._close()
                raise item
            chunk = item
            if chunk["done"]:
                # Terminator: stash result for exec_inspect, end
                # iteration. Don't yield the terminator itself —
                # docker-py's stream iterator doesn't have one.
                self._on_terminator(chunk)
                self._close()
                raise StopIteration
            # Filter heartbeat chunks (kept the wire alive but
            # don't carry meaningful bytes).
            if not chunk["stdout"] and not chunk["stderr"]:
                continue
            if self._demux:
                return (chunk["stdout"], chunk["stderr"])
            return chunk["stdout"] + chunk["stderr"]


def _make_not_implemented_stub(
    attr_name: str, supported: list[str],
) -> Any:
    """Build the closure used by ``_install_cluster_safety_net``.

    Each shadowed method becomes this lambda — calling it raises
    ``NotImplementedError`` with a message naming what IS wired.
    Factored out so the closure binding is unambiguous (in-loop
    lambdas closing over loop variables are a classic footgun).
    """

    def _stub(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"xrlenv docker drop-in (cluster mode) does not yet "
            f"implement ``api.{attr_name}``. Wired so far: "
            f"{supported}. File a follow-up if your harness "
            f"needs this method.",
        )
    _stub.__name__ = f"_cluster_stub_{attr_name}"
    return _stub


def _container_id_arg(container: Any) -> str:
    """docker-py methods accept either a container_id string or a
    Container-object-with-.id; normalise to the string."""
    if isinstance(container, str):
        return container
    cid = getattr(container, "id", None) or getattr(container, "Id", None)
    if cid is None:
        raise ValueError(
            f"xrlenv docker drop-in: cannot extract container id "
            f"from {container!r}",
        )
    return str(cid)


class XrlenvDockerClient(docker.DockerClient):
    """Drop-in replacement for ``docker.DockerClient``.

    Subclasses ``docker.DockerClient`` so ``isinstance`` checks in
    consumer code keep passing. ``__init__`` skips the parent's
    daemon-dial and assigns ``self.api`` to an :class:`XrlenvAPIClient`
    — which, in LocalDocker mode, calls its own ``super().__init__()``
    and ends up functionally identical to a real ``docker.APIClient``
    against the local daemon. Cluster-mode routing plugs in via
    the ``ContainerControl`` seam without changing this class.

    The drop-in optionally owns a ``_DropInRunner`` (the connect-
    mode ``from_env(grpc_host=...)`` factory wires one up). When
    so, ``close()`` shuts the runner down + closes the underlying
    Client.
    """

    def __init__(
        self,
        *,
        control: ContainerControl | None = None,
        runner: _DropInRunner | None = None,
        owned_client: Client | None = None,
    ) -> None:
        # Don't call super().__init__() — it constructs a real
        # docker.APIClient. We assign self.api ourselves; the inherited
        # manager properties (containers, images, ...) read from it
        # lazily.
        if control is None:
            control = LocalDockerContainerControl()
        self.api = XrlenvAPIClient(control=control)
        # ``_owned_runner`` / ``_owned_client`` are present only on
        # the ``from_env(grpc_host=...)`` self-contained factory's
        # output — ``close()`` tears them down. When the caller
        # passed in their own client+runner we leave both alone.
        self._owned_runner = runner
        self._owned_client = owned_client

    def close(self) -> None:
        """Tear down resources we own.

        - LocalDocker mode: forward to ``docker.APIClient.close()``
          (closes the underlying docker daemon HTTP session).
        - Cluster mode with self-contained ``from_env(grpc_host=...)``:
          close the gRPC Client (on the runner's loop) and shut the
          runner down.
        - Cluster mode with caller-supplied ``client`` / ``runner``:
          no-op — caller owns lifetime.
        """
        if self._owned_client is not None and self._owned_runner is not None:
            try:
                self._owned_runner.run(self._owned_client.close())
            except Exception:
                LOGGER.warning(
                    "xrlenv docker drop-in: error closing owned "
                    "Client; tearing runner down anyway",
                    exc_info=True,
                )
            self._owned_runner.close()
            self._owned_runner = None
            self._owned_client = None
            return
        # LocalDocker / caller-managed cluster path: defer to
        # docker.APIClient.close() (no-op in cluster mode since
        # the underlying session was never initialised).
        try:
            close = getattr(self.api, "close", None)
            if callable(close):
                close()
        except Exception:
            LOGGER.warning(
                "xrlenv docker drop-in: error in api.close()",
                exc_info=True,
            )

    def terminate_raw_group(
        self, group_id: str, reason: str = "group_terminated",
    ) -> TerminateRawGroupReport | None:
        """Destroy every still-running raw container carrying ``group_id`` — the SYNC drop-in
        over :meth:`Client.terminate_raw_group`.

        A consumer driving raw containers through this drop-in (``containers.run`` /
        ``acquire_container``) that aborts a run — e.g. Ctrl-C — calls this to tear its
        containers down actively: a node-confirmed destroy frees capacity immediately instead
        of leaving them for the raw-liveness reaper (which destroys only at the ~900 s
        quarantine horizon, not the 120 s TTL) or the 4 h wall-clock deadline. Idempotent
        (already-terminal rows are reported, not re-destroyed).

        Cluster mode only — the ``from_env(grpc_host=...)`` factory that owns a Client + runner.
        Returns ``None`` in LocalDocker / caller-managed mode (no cluster raw-group registry to
        sweep; local containers are torn down with ``container.stop()`` / ``remove()``)."""
        if self._owned_client is None or self._owned_runner is None:
            return None
        return _run_sync(
            self._owned_client.terminate_raw_group(group_id, reason),
            runner=self._owned_runner,
        )

    def __enter__(self) -> XrlenvDockerClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def from_env(
    *,
    control: ContainerControl | None = None,
    client: Client | None = None,
    runner: _DropInRunner | None = None,
    grpc_host: str | None = None,
    grpc_port: int | None = None,
    consumer_token: str | None = None,
    grpc_secure: bool | None = None,
    grpc_channel_options: list[tuple[str, Any]] | None = None,
) -> XrlenvDockerClient:
    """Mirrors ``docker.from_env()``. Maximum-drop-in form.

    **Maximum drop-in** (no kwargs at all): the audience's existing
    one-line ``client = docker.from_env()`` becomes ``client =
    xrlenv.from_env()``. Connection config is read from environment:

    - ``XRLENV_GRPC_HOST`` (string) — control-plane host. **If
      unset, returns a LocalDocker-mode client** (same daemon
      ``docker.from_env()`` would dial), so a fresh checkout +
      laptop dev keeps working with zero env vars.
    - ``XRLENV_GRPC_PORT`` (int, default 50051).
    - ``XRLENV_CONSUMER_TOKEN`` (string) — bearer token from
      ``xrlenv tokens issue consumer``. Required when
      ``XRLENV_GRPC_HOST`` is set.
    - ``XRLENV_GRPC_SECURE`` (``"true"`` / ``"1"`` / ``"yes"``;
      default false) — TLS channel.

    Mirrors ``docker.from_env()``'s reading of ``DOCKER_HOST`` etc.
    The audience never touches the factory call site; the operator
    sets env vars at deploy time.

    **Connect mode (explicit kwargs)**: ``grpc_host=...`` etc. take
    precedence over env vars. Use this when wrapping logic that
    needs different config per-instance, but the typical
    audience-facing pattern is the no-arg form.

    **LocalDocker mode** (no env vars, no kwargs): functionally
    identical to ``docker.from_env()`` against the local daemon.

    **Caller-supplied Client / runner / control**: SDK-level
    escape hatches for tests + advanced consumers; not part of the
    drop-in contract.

    Connection config precedence (highest first):
        kwargs > environment variables > LocalDocker fallback

    Cluster-mode rollout metadata (task_key, group_id, resources)
    rides on Docker labels — the standard extensibility hook
    docker-py forwards end-to-end.
    """
    # P1.7.B.2 W5b: env-var fallback. Kwargs take precedence; when
    # unset, read from env. ``XRLENV_GRPC_HOST`` is the gate — its
    # presence flips us into cluster mode. Operator sets these in
    # the consumer's deploy environment so the audience's harness
    # reads ``xrlenv.from_env()`` literally.
    import os

    if grpc_host is None:
        env_host = os.environ.get("XRLENV_GRPC_HOST")
        if env_host:
            grpc_host = env_host
    if grpc_port is None:
        env_port = os.environ.get("XRLENV_GRPC_PORT")
        if env_port:
            try:
                grpc_port = int(env_port)
            except ValueError:
                LOGGER.warning(
                    "xrlenv.from_env(): XRLENV_GRPC_PORT=%r is not "
                    "an int; falling back to default 50051",
                    env_port,
                )
                grpc_port = 50051
        else:
            grpc_port = 50051
    if consumer_token is None:
        consumer_token = os.environ.get("XRLENV_CONSUMER_TOKEN")
    if grpc_secure is None:
        env_secure = os.environ.get("XRLENV_GRPC_SECURE", "")
        grpc_secure = env_secure.strip().lower() in ("true", "1", "yes", "on")

    owned_runner: _DropInRunner | None = None
    owned_client: Client | None = None

    # Honor the documented precedence: kwargs > environment variables.
    # Build an owned Client from grpc_host (whether kwarg or env-derived)
    # only when the caller hasn't supplied an explicit ``client=`` /
    # ``control=`` / ``runner=`` — those bypass the connect path entirely.
    if (
        grpc_host is not None
        and control is None
        and client is None
        and runner is None
    ):
        # Self-contained connect-mode factory. Build runner +
        # Client up front on the same loop.
        from xrlenv.client.client import Client as _Client  # avoid cycle

        owned_runner = _DropInRunner()

        async def _build() -> Client:
            # ``Client.grpc()`` is sync but constructs the gRPC
            # aio channel via ``grpc.aio.insecure_channel(...)`` /
            # ``secure_channel(...)``, which binds to the running
            # loop. Run it inside an async helper dispatched on
            # the runner's loop so the channel binds there.
            return _Client.grpc(
                host=grpc_host, port=grpc_port,
                token=consumer_token,
                secure=grpc_secure,
                channel_options=grpc_channel_options,
            )

        try:
            owned_client = owned_runner.run(_build())
        except Exception:
            owned_runner.close()
            raise
        client = owned_client
        runner = owned_runner

    if control is None and client is not None:
        control = ClusterContainerControl(client=client, runner=runner)
    return XrlenvDockerClient(
        control=control, runner=owned_runner, owned_client=owned_client,
    )
