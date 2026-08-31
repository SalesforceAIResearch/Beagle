"""Egress restriction for a *running* container (spec 07, IP/CIDR slice).

One generic mechanism: tighten an already-running container's egress to an
IP/CIDR allowlist, enforced by an iptables ``OUTPUT`` chain installed into
the container's network namespace. No knowledge of tasks, agents, or LLM
endpoints lives here — that policy is the caller's (the harness decides
*when* and *what* to allow; this module only enforces a CIDR set).

Pieces:

- :class:`EgressRule` / :class:`EgressAllowlist` — the allowlist (CIDRs +
  optional ports). An **empty** allowlist is valid and means "block all
  external egress" (loopback still works); it is NOT an error.
- :func:`compile_egress_rules` — pure: allowlist → the exact ordered
  iptables (v4) program + an ip6tables (v6) lockdown. No I/O, fully
  unit-testable.
- :func:`is_shared_netns` — pure guard: refuse host / shared netns.
- :class:`EgressEnforcer` / :class:`DockerNsenterEnforcer` — the one
  privileged surface (``nsenter -t <pid> -n iptables …``); tests inject a
  recording fake.

This is the post-install "restrict now" path, so the program is **strict**:
it does NOT broadly accept ESTABLISHED OUTPUT, so any flow opened before
the restriction (e.g. an agent's ``npm``/``git`` bootstrap) is dropped on
its next packet. Allowed-CIDR flows still match the per-CIDR ACCEPT and
their return path is via the (untouched) INPUT chain. DNS-name allowlisting
and IPv6 allowlisting are out of scope (phase 2); v6 is locked down so a v4
allowlist can't be bypassed over IPv6.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from ipaddress import ip_network
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, model_validator

# Cloud-metadata endpoints blocked under every restriction (spec 07 §"Cloud
# metadata block"): 169.254.169.254 = AWS IMDS + the GCE metadata IP;
# 169.254.170.2 = the ECS task-metadata endpoint. DROP'd before any ACCEPT
# so an overlapping allowlist entry can't reopen them.
DEFAULT_METADATA_IPS: tuple[str, ...] = ("169.254.169.254/32", "169.254.170.2/32")

# iptables ``multiport`` caps one rule at 15 ports.
_MULTIPORT_MAX = 15


class EgressRule(BaseModel):
    """One allowed destination: an IPv4 ``cidr`` and optional ``ports``.

    ``ports`` ``None`` = all ports; a tuple restricts to those destination
    ports (tcp + udp). DNS-name (``host:``) and IPv6 entries are phase 2.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cidr: str
    ports: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _validate(self) -> EgressRule:
        try:
            net = ip_network(self.cidr, strict=False)  # tolerate host bits
        except ValueError as exc:
            raise ValueError(f"invalid cidr {self.cidr!r}: {exc}") from exc
        if net.version != 4:
            raise ValueError(
                f"egress rule cidr={self.cidr!r} is IPv6; v6 allowlisting is "
                "phase 2 (v6 egress is locked down). Pin a v4 cidr.",
            )
        if self.ports is not None:
            if len(self.ports) > _MULTIPORT_MAX:
                raise ValueError(
                    f"cidr {self.cidr!r} lists {len(self.ports)} ports; iptables "
                    f"multiport allows at most {_MULTIPORT_MAX}",
                )
            for p in self.ports:
                if not (1 <= p <= 65535):
                    raise ValueError(f"port {p} out of range 1..65535")
        return self


class EgressAllowlist(BaseModel):
    """An ordered set of :class:`EgressRule`. **Empty is valid** and means
    "block all external egress" (loopback + the mandatory metadata DROP
    remain) — for restricting a task whose agent needs no external endpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[EgressRule, ...] = ()


class EgressProgram(NamedTuple):
    """Compiled policy: the v4 iptables program + the v6 ip6tables lockdown,
    each an ordered list of argv (the args after the ``iptables`` binary).
    Applied in order; the leading ``-F OUTPUT`` makes re-application
    idempotent.
    """

    v4: list[list[str]]
    v6: list[list[str]]


def is_shared_netns(network_mode: str | None) -> bool:
    """True if ``network_mode`` shares a netns with the host or another
    container (``host`` / ``container:<id>``).

    Egress rules are installed by entering the target's netns
    (``nsenter -t <pid> -n``) and the program begins with ``-F OUTPUT``. In a
    shared/host netns that flush + rules would rewrite the node host's (or a
    sibling's) chain, not the sandbox's — so callers MUST refuse it.
    ``bridge`` / ``default`` / ``none`` / a user-defined network all get
    their own netns and are safe.
    """
    nm = (network_mode or "").strip().lower()
    return nm == "host" or nm.startswith("container:")


# §6 — runtimes whose inner root can rewrite the netns iptables the
# ``nsenter``-based allowlist installs, defeating egress restriction. Sysbox
# system containers give their inner root a real (virtualized) netns +
# NET_ADMIN *inside* the container's user namespace, so the workload can
# ``iptables -F OUTPUT`` on the very chain we rely on — exactly the escape the
# privileged/NET_ADMIN guard already refuses. Kept as a set so a future
# escape-capable runtime is one entry, not a new branch.
_EGRESS_UNSAFE_RUNTIMES = frozenset({"sysbox-runc"})


def container_can_escape_egress(host_config: dict[str, Any]) -> bool:
    """True if the container could modify its own iptables rules from inside —
    i.e. it is ``Privileged``, holds ``CAP_NET_ADMIN``, or runs under a runtime
    whose inner root controls its own netns (``sysbox-runc``).

    Egress restriction is only a trusted boundary if the workload cannot flush
    the OUTPUT chain. Docker's default cap set excludes ``NET_ADMIN``, so a
    plain container is safe; but the raw-container path can forward
    ``cap_add`` / ``privileged`` / ``container_runtime``. Callers MUST refuse
    :func:`apply_egress` for such a container — otherwise a successful
    ApplyEgress is a false anti-cheat signal (the agent inside can
    ``iptables -F OUTPUT``). Takes docker-inspect ``HostConfig`` (``Privileged``
    bool + ``CapAdd`` list, names with or without the ``CAP_`` prefix, +
    ``Runtime`` string).
    """
    if host_config.get("Privileged"):
        return True
    if str(host_config.get("Runtime") or "").strip() in _EGRESS_UNSAFE_RUNTIMES:
        return True
    cap_add = host_config.get("CapAdd") or []
    return any(
        str(c).strip().upper().removeprefix("CAP_") == "NET_ADMIN"
        for c in cap_add
    )


def _accept_cidr_rules(cidr: str, ports: tuple[int, ...] | None) -> list[list[str]]:
    if not ports:
        return [["-A", "OUTPUT", "-d", cidr, "-j", "ACCEPT"]]
    dports = ",".join(str(p) for p in ports)
    return [
        ["-A", "OUTPUT", "-p", proto, "-d", cidr,
         "-m", "multiport", "--dports", dports, "-j", "ACCEPT"]
        for proto in ("tcp", "udp")
    ]


def compile_egress_rules(
    allowlist: EgressAllowlist,
    *,
    dns_resolver: str | None = None,
    metadata_ips: tuple[str, ...] = DEFAULT_METADATA_IPS,
) -> EgressProgram:
    """Compile ``allowlist`` to its v4 iptables program + v6 lockdown.

    v4 OUTPUT order (load-bearing):
    1. flush OUTPUT (idempotent)
    2. ACCEPT loopback (``-o lo``)
    3. **DROP** each metadata IP — before any ACCEPT
    4. ACCEPT DNS (udp/tcp :53) to ``dns_resolver`` if given
    5. ACCEPT each declared cidr (optionally port-restricted)
    6. catch-all REJECT (``icmp-admin-prohibited`` — fail fast, not hang)

    No broad ESTABLISHED accept (strict — see module docstring). An empty
    allowlist compiles to "block all external" (steps 1-3 + 6).

    Raises ``ValueError`` for a malformed / IPv6 ``dns_resolver`` (validated
    here, in the pure path, so a typo fails before any container is touched
    rather than mid-``nsenter`` after a partial apply).
    """
    if dns_resolver is not None:
        try:
            resolver_net = ip_network(dns_resolver, strict=False)
        except ValueError as exc:
            raise ValueError(
                f"invalid dns_resolver {dns_resolver!r}: {exc}",
            ) from exc
        if resolver_net.version != 4:
            raise ValueError(
                f"dns_resolver {dns_resolver!r} is IPv6; only v4 is supported",
            )
    v4: list[list[str]] = [
        ["-F", "OUTPUT"],
        ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"],
    ]
    for mip in metadata_ips:
        v4.append(["-A", "OUTPUT", "-d", mip, "-j", "DROP"])
    if dns_resolver is not None:
        for proto in ("udp", "tcp"):
            v4.append(
                ["-A", "OUTPUT", "-p", proto, "-d", dns_resolver,
                 "--dport", "53", "-j", "ACCEPT"],
            )
    for rule in allowlist.rules:
        v4.extend(_accept_cidr_rules(rule.cidr, rule.ports))
    v4.append(
        ["-A", "OUTPUT", "-j", "REJECT", "--reject-with", "icmp-admin-prohibited"],
    )

    v6: list[list[str]] = [
        ["-F", "OUTPUT"],
        ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-j", "REJECT", "--reject-with", "icmp6-adm-prohibited"],
    ]
    return EgressProgram(v4=v4, v6=v6)


class EgressApplyError(RuntimeError):
    """An iptables/ip6tables invocation failed while applying a policy."""


class EgressEnforcer(ABC):
    """Applies a compiled :class:`EgressProgram` to a running container's
    netns. The single privileged surface — callers/tests depend on this ABC,
    not on ``nsenter``.
    """

    @abstractmethod
    async def apply(self, *, container_pid: int, program: EgressProgram) -> None:
        """Install ``program`` in the netns of ``container_pid``. Idempotent
        (the program flushes OUTPUT first). Raises :class:`EgressApplyError`
        on the first failing invocation (never leaves a partial = under-
        enforced policy silently in place).
        """


class DockerNsenterEnforcer(EgressEnforcer):
    """Applies via ``nsenter -t <pid> -n {iptables,ip6tables}``.

    Enters only the container's *network* namespace, so the iptables binary +
    modules come from the node host (which has them) and operate on the
    container's isolated netns. Requires the node process to hold
    ``CAP_NET_ADMIN``.
    """

    def __init__(
        self,
        *,
        nsenter_bin: str = "nsenter",
        iptables_bin: str = "iptables",
        ip6tables_bin: str = "ip6tables",
    ) -> None:
        self._nsenter = nsenter_bin
        self._iptables = iptables_bin
        self._ip6tables = ip6tables_bin

    async def apply(self, *, container_pid: int, program: EgressProgram) -> None:
        await self._run_program(container_pid, self._iptables, program.v4)
        await self._run_program(container_pid, self._ip6tables, program.v6)

    async def _run_program(
        self, pid: int, tables_bin: str, rules: list[list[str]],
    ) -> None:
        for args in rules:
            argv = [self._nsenter, "-t", str(pid), "-n", tables_bin, *args]
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise EgressApplyError(
                    f"egress rule {tables_bin} {' '.join(args)} failed "
                    f"(rc={proc.returncode}): "
                    f"{stderr.decode(errors='replace').strip()}",
                )
