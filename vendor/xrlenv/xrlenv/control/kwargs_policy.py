"""Cluster-wide policy for docker-py kwargs that flow through the
xrlenv-cluster drop-in to the node-side ``containers.run``.

Background (issue #6 on the public tracker): the drop-in's allowlist
``image, command, entrypoint, name, labels, environment, task_key, user,
cap_add, ensure_image_present, userns_mode`` is too narrow. Every new
benchmark that needs a different docker-py kwarg (``devices`` for
SCUBA-style KVM benchmarks, ``cap_add=[NET_ADMIN]`` for nested-network
agents, ``shm_size`` for memory-heavy test suites, ...) bottlenecks on a
fresh proto-extension slice.

The fix is a four-tier policy with sensible defaults so the operator
doesn't have to author a config file to onboard the current benchmark
set, plus a single optional ``policy:`` section in ``nodes.yaml`` for
tuning. The tiers:

  Level 0 — Always allowed. Standard image-shape kwargs that can't
    compromise the host (user, entrypoint, environment, labels, the
    docker default capability set, mem_limit, shm_size, tmpfs,
    read_only, ...). No policy override available; nothing to tune.

  Level 1 — Allowed by default, operator can restrict via ``denied_caps``
    or ``allowed_devices``. The defaults unblock the benchmarks we care
    about (NET_ADMIN for nested networking, /dev/kvm for SCUBA-style
    nested VMs).

  Level 2 — Rejected by default. Operator opts in per cluster
    (``allow_host_network``, ``allow_privileged``, ``allowed_host_paths``).
    These break the sandbox boundary and need an explicit per-deployment
    decision.

  Level 3 — Never allowed; no policy override. Always-fatal escapes
    of namespace / cgroup / network isolation (``pid_mode=host``,
    ``ipc_mode=host``, ``cgroup_parent``, ``network_mode=container:...``),
    plus the CPU/memory placement knobs the cluster owns
    (``cpuset_cpus``, ``cpuset_mems``).

  Level 4 — Rejected with architectural rationale (not a security
    boundary): ``platform`` (per-node arch is cluster-managed),
    ``userns_mode`` (operator-set at daemon level via spec-19).

The control plane is the authoritative enforcement point. The drop-in
fast-fails against ``DEFAULT_POLICY`` so end users get an immediate
error without an extra RPC round-trip; the control plane re-validates
against the cluster-wide policy loaded from ``nodes.yaml`` because the
end user's drop-in defaults can't see operator-specific tweaks (a
``denied_caps: [SYS_MODULE]`` the operator set, or an
``allow_host_network: true`` opt-in). Defense-in-depth at the node is
NOT wired — per spec-21 the node has no inbound listener, so the
control plane is the only path to it; one enforcement point is enough.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

# ─── Code-level constants (Level 0 + Level 3, no policy override) ────────────

# Level 0 — standard caps Docker grants by default plus a few benign extras
# the test-harness ecosystem leans on. Source: Docker's runtime/cap default
# set; SYS_PTRACE / SYS_NICE / IPC_LOCK are common test-suite needs and not
# meaningful escalations on a sandboxed container.
LEVEL_0_CAPS: frozenset[str] = frozenset({
    "AUDIT_WRITE",
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "FSETID",
    "KILL",
    "MKNOD",
    "NET_BIND_SERVICE",
    "NET_RAW",
    "SETFCAP",
    "SETGID",
    "SETPCAP",
    "SETUID",
    "SYS_CHROOT",
    "SYS_PTRACE",
    "SYS_NICE",
    "IPC_LOCK",
})

# Level 1 — elevated caps allowed by default; operator may add to
# ``policy.denied_caps`` to clamp. NET_ADMIN unblocks nested iptables /
# tc / userland VPN setups; SYS_ADMIN unblocks nested-mount-namespace
# tricks some browser-in-container harnesses need.
LEVEL_1_CAPS: frozenset[str] = frozenset({
    "NET_ADMIN",
    "SYS_ADMIN",
    "SYS_MODULE",
    "SYS_RAWIO",
    "LINUX_IMMUTABLE",
    "AUDIT_CONTROL",
    "BLOCK_SUSPEND",
    "WAKE_ALARM",
    "SYS_TIME",
    "SYS_PACCT",
    "SYS_BOOT",
    "MAC_ADMIN",
    "MAC_OVERRIDE",
    "LEASE",
    "NET_BROADCAST",
    "SYSLOG",
})


def _normalize_cap(cap: str) -> str:
    """Strip leading ``CAP_`` and upper-case. ``cap_add=["NET_ADMIN"]`` and
    ``["CAP_NET_ADMIN"]`` should be the same kwarg."""
    return cap.upper().removeprefix("CAP_")


# ─── Operator-tunable policy (Level 1 + Level 2) ─────────────────────────────


class KwargsPolicy(BaseModel):
    """Operator-tunable cluster docker-kwarg policy.

    Loaded from the ``policy:`` section of ``nodes.yaml``. Defaults work
    for swebench / terminal-bench / coding-bench / SCUBA-style KVM
    benchmarks out of the box — operators only edit this when they
    want to restrict or extend.

    All fields are frozen + ``extra="forbid"`` to fail-fast on typos in
    the yaml.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Level 1 — defaults allowed; operator can tune.
    allowed_devices: tuple[str, ...] = Field(
        default=("/dev/kvm", "/dev/net/tun", "/dev/fuse"),
        description=(
            "Host devices the harness may pass via docker-py "
            "``devices=[...]``. Unblocks SCUBA-style nested-VM "
            "benchmarks (/dev/kvm), userland VPN tooling (/dev/net/tun), "
            "and userspace filesystems (/dev/fuse)."
        ),
    )
    denied_caps: tuple[str, ...] = Field(
        default=(),
        description=(
            "Linux capabilities the harness MUST NOT add via "
            "``cap_add=[...]``. Default empty — the Level-1 cap set "
            "(NET_ADMIN, SYS_ADMIN, ...) is allowed unless listed here."
        ),
    )

    # Level 2 — defaults reject; operator opts in.
    allow_host_network: bool = Field(
        default=False,
        description=(
            "Allow ``network_mode=host`` (bypasses xrlenv's egress "
            "allowlist; exposes the host's network stack to the "
            "container). Default false."
        ),
    )
    allow_privileged: bool = Field(
        default=False,
        description=(
            "Allow ``privileged=True`` (full host capabilities; can "
            "escape the sandbox). Default false."
        ),
    )
    allowed_host_paths: tuple[str, ...] = Field(
        default=(),
        description=(
            "Host filesystem paths the harness may bind into the "
            "container via ``volumes={host: container}`` / "
            "``host_config.Binds``. Default empty — remote nodes can't "
            "reach the trainer-host fs anyway, so this is only useful "
            "for paths that exist on the node VMs themselves."
        ),
    )
    allowed_runtimes: tuple[str, ...] = Field(
        default=(),
        description=(
            "OCI runtimes the harness may select via ``container_runtime`` "
            "(e.g. ``sysbox-runc``). Default empty → EVERY non-default "
            "runtime override is rejected; docker's default runtime "
            "(``runc``, asserted at bootstrap) needs no entry. This is an "
            "explicit operator opt-in because an alternate runtime changes "
            "isolation/init/placement (§5.2). ``runc`` is always allowed "
            "even when not listed (it's the default, not an override)."
        ),
    )


DEFAULT_POLICY: KwargsPolicy = KwargsPolicy()


# ─── Rejection record + exception ────────────────────────────────────────────


class KwargsRejection(BaseModel):
    """One rejected kwarg + actionable rationale."""

    model_config = ConfigDict(frozen=True)

    kwarg: str
    level: int
    reason: str
    hint: str | None = None

    def format(self) -> str:
        head = (
            f"xrlenv: rejected docker kwarg `{self.kwarg}` "
            f"(level {self.level}): {self.reason}"
        )
        if self.hint:
            return f"{head}\n  fix: {self.hint}"
        return head


class KwargsPolicyViolation(Exception):
    """Raised by the drop-in or control plane when kwargs are rejected.

    Carries the full rejection list (not just the first hit) so the
    operator can fix everything in one pass instead of whack-a-mole.
    """

    def __init__(self, rejections: list[KwargsRejection]) -> None:
        self.rejections = list(rejections)
        super().__init__("\n".join(r.format() for r in self.rejections))


# ─── Per-kwarg validators (each returns 0 or 1 rejection) ────────────────────


def _validate_devices(
    devices: Iterable[str] | None, policy: KwargsPolicy,
) -> KwargsRejection | None:
    if not devices:
        return None
    allowed = set(policy.allowed_devices)
    for spec in devices:
        # docker-py accepts "/dev/x", "/dev/x:/dev/y", or
        # "/dev/x:/dev/y:rwm". The host path is the first colon-piece.
        host = spec.split(":", 1)[0]
        if host not in allowed:
            return KwargsRejection(
                kwarg="devices",
                level=1,
                reason=(
                    f"device {host!r} not in cluster's allowed_devices "
                    f"{sorted(allowed)!r}."
                ),
                hint=(
                    f"operator: add {host!r} to nodes.yaml "
                    "policy.allowed_devices and restart the control plane."
                ),
            )
    return None


def _validate_cap_add(
    caps: Iterable[str] | None, policy: KwargsPolicy,
) -> KwargsRejection | None:
    if not caps:
        return None
    denied = {_normalize_cap(c) for c in policy.denied_caps}
    if not denied:
        return None
    for cap in caps:
        if _normalize_cap(cap) in denied:
            return KwargsRejection(
                kwarg="cap_add",
                level=1,
                reason=(
                    f"capability {cap!r} is in cluster's denied_caps "
                    f"{sorted(policy.denied_caps)!r}."
                ),
                hint=(
                    f"operator: remove {cap!r} from nodes.yaml "
                    "policy.denied_caps if you want to allow this."
                ),
            )
    return None


def _validate_privileged(
    privileged: bool | None, policy: KwargsPolicy,
) -> KwargsRejection | None:
    if not privileged:
        return None
    if policy.allow_privileged:
        return None
    return KwargsRejection(
        kwarg="privileged",
        level=2,
        reason=(
            "privileged containers can mount /dev, write host kernel "
            "modules, and access host cgroups — sandbox escape surface."
        ),
        hint=(
            "operator: set ``allow_privileged: true`` in nodes.yaml "
            "policy section if you accept this risk for your cluster."
        ),
    )


def _validate_runtime(
    runtime: str | None, policy: KwargsPolicy,
) -> KwargsRejection | None:
    # None / empty / "runc" is the daemon default, not an override — always OK.
    if not runtime or runtime == "runc":
        return None
    if runtime in policy.allowed_runtimes:
        return None
    return KwargsRejection(
        kwarg="container_runtime",
        level=2,
        reason=(
            f"runtime {runtime!r} changes container isolation, init and "
            "placement (e.g. sysbox-runc runs nested docker/systemd under a "
            "user namespace); it is an explicit operator opt-in, not a "
            "default-allowed knob."
        ),
        hint=(
            f"operator: add {runtime!r} to ``allowed_runtimes`` in the "
            "nodes.yaml policy section, and make sure the runtime is "
            "installed + advertised on the target node pool (§5.3), if you "
            "accept this for your cluster."
        ),
    )


def _validate_network_mode(
    network_mode: str | None, policy: KwargsPolicy,
) -> KwargsRejection | None:
    if not network_mode:
        return None
    nm = network_mode.lower()
    if nm.startswith("container:") or nm == "container":
        return KwargsRejection(
            kwarg="network_mode",
            level=3,
            reason=(
                f"network_mode={network_mode!r} joins another container's "
                "namespace; not supported on a shared cluster (no policy override)."
            ),
            hint=None,
        )
    if nm == "host":
        if policy.allow_host_network:
            return None
        return KwargsRejection(
            kwarg="network_mode",
            level=2,
            reason=(
                "network_mode=host bypasses xrlenv's egress allowlist "
                "(spec-19) and exposes the host's network stack."
            ),
            hint=(
                "operator: set ``allow_host_network: true`` in nodes.yaml "
                "policy section if you accept this risk."
            ),
        )
    return None


def _validate_pid_mode(pid_mode: str | None) -> KwargsRejection | None:
    if not pid_mode:
        return None
    if pid_mode.lower() == "host":
        return KwargsRejection(
            kwarg="pid_mode",
            level=3,
            reason=(
                "pid_mode=host escapes the PID namespace; the container "
                "can see and signal host processes (no policy override)."
            ),
            hint=None,
        )
    return None


def _validate_ipc_mode(ipc_mode: str | None) -> KwargsRejection | None:
    if not ipc_mode:
        return None
    if ipc_mode.lower() == "host":
        return KwargsRejection(
            kwarg="ipc_mode",
            level=3,
            reason=(
                "ipc_mode=host escapes the IPC namespace; allows shared "
                "memory access with host processes (no policy override)."
            ),
            hint=None,
        )
    return None


def _validate_cgroup_parent(parent: str | None) -> KwargsRejection | None:
    if not parent:
        return None
    return KwargsRejection(
        kwarg="cgroup_parent",
        level=3,
        reason=(
            "cgroup_parent overrides the cluster's resource accounting "
            "hierarchy; operator-controlled at cluster level (no policy override)."
        ),
        hint=None,
    )


def _validate_cpuset_cpus(cpuset_cpus: str | None) -> KwargsRejection | None:
    # cpuset pinning is cluster-owned: the node assigns a disjoint core
    # set per grading container from its own core ledger. A harness that
    # pins to specific cpus would collide on every co-located container
    # and defeat xrlenv's placement / capacity accounting — same class of
    # escape as cgroup_parent, hence Level 3, no policy override.
    if not cpuset_cpus:
        return None
    return KwargsRejection(
        kwarg="cpuset_cpus",
        level=3,
        reason=(
            "cpuset_cpus overrides cluster-owned CPU placement; the node "
            "assigns the cpuset per container (no policy override)."
        ),
        hint=(
            "remove the cpuset_cpus kwarg — the cluster pins cores itself; "
            "use cpu_quota / nano_cpus to express a CPU limit instead."
        ),
    )


def _validate_cpuset_mems(cpuset_mems: str | None) -> KwargsRejection | None:
    # Same rationale as cpuset_cpus: NUMA-memory-node placement is
    # cluster-owned and breaks accounting if a harness sets it.
    if not cpuset_mems:
        return None
    return KwargsRejection(
        kwarg="cpuset_mems",
        level=3,
        reason=(
            "cpuset_mems overrides cluster-owned memory-node placement "
            "(no policy override)."
        ),
        hint="remove the cpuset_mems kwarg — placement is cluster-managed.",
    )


def _host_path_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """A bind host path is allowed if it exactly equals an ``allowed_host_paths``
    entry OR is nested under one (each entry is a directory prefix — allowing a
    directory allows its subtree). The ``+ "/"`` guard stops ``/mnt/data`` from
    matching a sibling like ``/mnt/database``. Needed for dynamic per-run bind
    paths (e.g. a Sysbox pool node real-binding ``/path/to/data<hash>``
    under an allowed ``/path/to/data`` prefix)."""
    for entry in allowed:
        base = entry.rstrip("/")
        if host == base or host.startswith(base + "/"):
            return True
    return False


def _validate_binds(
    binds: Iterable[str] | None, policy: KwargsPolicy,
) -> KwargsRejection | None:
    if not binds:
        return None
    allowed = tuple(policy.allowed_host_paths)
    for spec in binds:
        host = spec.split(":", 1)[0]
        if not _host_path_allowed(host, allowed):
            return KwargsRejection(
                kwarg="binds",
                level=2,
                reason=(
                    f"bind {spec!r}: host path {host!r} not in cluster's "
                    "allowed_host_paths. host filesystem isn't directly "
                    "reachable from a remote node anyway — use "
                    "``container.put_archive(...)`` to copy files in."
                ),
                hint=(
                    f"operator: if the node VM itself has {host!r} and "
                    "you really need it mounted, add it to nodes.yaml "
                    "policy.allowed_host_paths."
                ),
            )
    return None


def _validate_userns_mode(mode: str | None) -> KwargsRejection | None:
    if not mode or mode.lower() == "host":
        return None
    return KwargsRejection(
        kwarg="userns_mode",
        level=4,
        reason=(
            "userns_mode is operator-controlled at the cluster level via "
            "spec-19; harness-side overrides are not honored."
        ),
        hint=(
            "operator: configure daemon-level userns-remap in "
            "/etc/docker/daemon.json on each node."
        ),
    )


def _validate_platform(platform: str | None) -> KwargsRejection | None:
    if not platform:
        return None
    return KwargsRejection(
        kwarg="platform",
        level=4,
        reason=(
            "platform is set per-node at cluster bootstrap; the cluster "
            "already knows each node's architecture and ignores harness hints."
        ),
        hint=(
            "remove the ``platform=`` kwarg, or talk to your operator if "
            "you need a different arch."
        ),
    )


# ─── Public entry point ──────────────────────────────────────────────────────


def validate_kwargs(
    *,
    devices: Iterable[str] | None = None,
    cap_add: Iterable[str] | None = None,
    privileged: bool | None = None,
    network_mode: str | None = None,
    pid_mode: str | None = None,
    ipc_mode: str | None = None,
    cgroup_parent: str | None = None,
    cpuset_cpus: str | None = None,
    cpuset_mems: str | None = None,
    binds: Iterable[str] | None = None,
    userns_mode: str | None = None,
    platform: str | None = None,
    runtime: str | None = None,
    policy: KwargsPolicy = DEFAULT_POLICY,
) -> list[KwargsRejection]:
    """Validate the full kwarg set against ``policy``.

    Returns the full rejection list (empty = all clean) so the caller
    can surface every problem in one go. Use
    :class:`KwargsPolicyViolation` to raise when the list is non-empty:

        >>> rejections = validate_kwargs(privileged=True, pid_mode="host")
        >>> if rejections:
        ...     raise KwargsPolicyViolation(rejections)

    Validators each handle a single kwarg and return either ``None``
    (clean) or one :class:`KwargsRejection`. Multiple bad kwargs on the
    same call collect into one violation; the operator gets a single
    error log instead of N round-trips of fix-and-retry.
    """
    rejections: list[KwargsRejection] = []
    for r in (
        _validate_devices(devices, policy),
        _validate_cap_add(cap_add, policy),
        _validate_privileged(privileged, policy),
        _validate_network_mode(network_mode, policy),
        _validate_pid_mode(pid_mode),
        _validate_ipc_mode(ipc_mode),
        _validate_cgroup_parent(cgroup_parent),
        _validate_cpuset_cpus(cpuset_cpus),
        _validate_cpuset_mems(cpuset_mems),
        _validate_binds(binds, policy),
        _validate_userns_mode(userns_mode),
        _validate_platform(platform),
        _validate_runtime(runtime, policy),
    ):
        if r is not None:
            rejections.append(r)
    return rejections


__all__ = [
    "DEFAULT_POLICY",
    "LEVEL_0_CAPS",
    "LEVEL_1_CAPS",
    "KwargsPolicy",
    "KwargsPolicyViolation",
    "KwargsRejection",
    "validate_kwargs",
]
