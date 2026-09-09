"""``BeaglePierAgent`` — the ONE pier shim that runs any beagle agent.

The pier analog of :mod:`beagle.benchmarks.harness._harbor_agent`. Pier (``datacurve-pier``) is a
harbor fork with the same installed-agent interface, so this is that shim retargeted at pier's
``BaseInstalledAgent`` / ``BaseEnvironment`` / ``AgentContext``. It reconstructs the beagle agent
from a small serializable *identity*, wraps pier's environment as a :class:`HarborEnvRuntime`, and
splits the agent across pier's two phases so a **network-phased** benchmark (DeepSWE:
``allow_internet=false`` for the agent) works:

* :meth:`install` — pier's INSTALL phase (network open): git bootstrap + the agent's own
  ``install`` (clone + build). This is why an installed agent that clones itself works on DeepSWE.
* :meth:`network_allowlist` — the RUN phase is restricted to the agent's
  :meth:`~beagle.agents.core.base.Runnable.network_hosts` (the LLM gateway), via pier's native
  allowlist.
* :meth:`run` — pier's RUN phase: the agent's ``run_in`` (already installed; only the LLM endpoint).

Only ever imported *by pier* (via the shim import path), so importing pier at module top is fine —
beagle core never imports it, keeping pier an optional dependency (``beagle[deep-swe]``).
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import shlex
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pier.agents.installed.base import BaseInstalledAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext

from beagle.agents.core.base import AgentInstallError
from beagle.benchmarks.harness._common import (
    _GIT_BOOTSTRAP,
    _rebuild_agent,
    declare_task_budget,
)
from beagle.rollout.runtime.harbor_env import HarborEnvRuntime
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult


def _run_egress_cidrs(hosts: list[str]) -> list[str]:
    """The v4 ``/32`` CIDRs to seal the agent's RUN phase to, from its run-host URLs
    (``http://192.0.2.20:18088`` -> ``192.0.2.20/32``). A run host that is a *hostname* (not a
    bare IPv4) yields no CIDR — such a trial stays on pier's Squid domain-filter path rather than the
    iptables open-install seal. See ``notes/pier-open-install-egress.md`` in xrlenv."""
    cidrs: list[str] = []
    for h in hosts:
        host = urlparse(h).hostname or (h or "")
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            continue
        cidrs.append(f"{host}/32")
    return cidrs


def _all_ipv4(hosts: list[str]) -> bool:
    """True iff every non-empty run host is a bare IPv4 → the trial takes pier's **open-install**
    path (single-container OPEN acquire + post-install iptables ``apply_egress``). Any hostname keeps
    it on the Squid domain-filter path. Kept in lockstep with xrlenv's ``_egress_domains`` predicate."""
    present = [h for h in hosts if h]
    return bool(present) and len(_run_egress_cidrs(present)) == len(present)


def _run_egress_targets(hosts: list[str]) -> list[tuple[str, int]]:
    """Validated ``(hostname, port)`` targets from agent RUN URLs."""
    targets: list[tuple[str, int]] = []
    for raw in hosts:
        if not raw:
            continue
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"invalid RUN egress URL scheme in {raw!r}")
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host or (
            not _is_ipv4(host) and re.fullmatch(r"[a-z0-9.-]+", host) is None
        ):
            raise ValueError(f"invalid RUN egress host {raw!r}")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid RUN egress port in {raw!r}: {exc}") from exc
        if port is None:
            port = 80 if parsed.scheme == "http" else 443
        targets.append((host, port))
    return list(dict.fromkeys(targets))


def _is_ipv4(host: str) -> bool:
    try:
        ipaddress.IPv4Address(host)
    except ValueError:
        return False
    return True


async def _resolve_and_pin_run_egress(
    environment: BaseEnvironment, hosts: list[str],
) -> tuple[list[str], tuple[int, ...] | None]:
    """Resolve hostname targets before RUN, pin them in ``/etc/hosts``, and return CIDRs.

    Resolution and pinning happen after trusted installation but before the task instruction is
    given to the agent. TLS still uses the original hostname/SNI; iptables permits only the resolved
    IPv4 addresses. Resolution or pinning failure aborts before untrusted execution.
    """
    targets = _run_egress_targets(hosts)
    distinct_ports = {port for _host, port in targets}
    if len(distinct_ports) > 1:
        raise ValueError(
            "phase-split Pier egress does not support heterogeneous endpoint ports; "
            f"got {sorted(distinct_ports)}"
        )
    resolved: dict[str, list[str]] = {}
    for host, _port in targets:
        if host in resolved:
            continue
        if _is_ipv4(host):
            resolved[host] = [host]
            continue
        result = await environment.exec(
            f"getent ahostsv4 {shlex.quote(host)}",
            timeout_sec=15,
        )
        addresses = sorted({
            token
            for line in (result.stdout or "").splitlines()
            if line.split()
            for token in [line.split()[0]]
            if _is_ipv4(token)
        })
        if result.return_code != 0 or not addresses:
            raise RuntimeError(
                f"could not resolve RUN egress hostname {host!r} to IPv4 before sealing"
            )
        resolved[host] = addresses

    host_lines = [
        f"{address} {host}"
        for host, addresses in resolved.items()
        if not _is_ipv4(host)
        for address in addresses
    ]
    if host_lines:
        args = " ".join(shlex.quote(line) for line in host_lines)
        pinned = await environment.exec(
            f"printf '%s\\n' {args} >> /etc/hosts",
            user="root",
            timeout_sec=15,
        )
        if pinned.return_code != 0:
            raise RuntimeError(
                f"could not pin RUN egress hostnames in /etc/hosts: {pinned.stderr or ''}"
            )

    cidrs = sorted({
        f"{address}/32"
        for addresses in resolved.values()
        for address in addresses
    })
    ports = tuple(distinct_ports) or None
    return cidrs, ports


class BeaglePierAgent(BaseInstalledAgent):
    """Adapts any beagle ``Runnable`` to pier's installed-agent interface (see the harbor shim)."""

    SUPPORTS_ATIF: bool = False
    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "beagle"

    def __init__(
        self,
        logs_dir: Path,
        *,
        identity: dict[str, Any],
        phase_split_egress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, **kwargs)
        self._identity = identity
        self._agent = _rebuild_agent(identity)
        self._phase_split_egress = phase_split_egress
        # pier's ``<trial>/agent`` dir — the handle to the trial's config.json, hence to the task's
        # declared agent budget (pier is a harbor fork; same trial layout, same timeout fields).
        self._logs_dir = Path(logs_dir)
        self._runtime: HarborEnvRuntime | None = None
        self._handle: Any | None = None
        self._task_ctx: TaskContext | None = None
        self._install_error: str | None = None
        self._result: Any | None = None

    def install_spec(self):
        """The declarative INSTALL step pier's default ``install()`` runs (which :meth:`install`
        chains via ``super()``): ensure git is present as root. The agent's own clone/build happens
        imperatively in :meth:`install` right after."""
        from pier.models.agent.install import AgentInstallSpec, InstallStep

        return AgentInstallSpec(
            agent_name=self.name(),
            steps=[InstallStep(run=_GIT_BOOTSTRAP, user="root")],
            metadata={})

    def network_allowlist(self):
        """The trial's legacy allowlist, or empty for Beagle-owned phase splitting.

        Fresh cluster jobs set ``phase_split_egress`` and deliberately give Pier no startup
        allowlist, selecting xrlenv's raw/open container for trusted INSTALL. :meth:`run` then pins
        and seals the RUN endpoints before exposing the instruction. Jobs saved before this feature
        have no flag and preserve the historical IP/Squid behavior verbatim:

        * **Open-install path** (all-IPv4 run hosts — e.g. deep-swe's LLM-gateway IP): return only
          the RUN hosts. xrlenv routes such a trial onto the single-container **OPEN** acquire, so the
          trusted install phase runs with a DIRECT route (its clone/build hosts need no allowlist —
          the ~9x Squid install tax is gone) and :meth:`run` iptables-seals the run phase to these IPs.
        * **Squid path** (a run host is a hostname → needs DNS-aware domain filtering): include the
          INSTALL hosts too, because pier then applies one allowlist for the whole trial.

        Uses pier's native ``allowlist_from_urls``. Kept in lockstep with xrlenv's ``_egress_domains``."""
        from pier.agents.network import allowlist_from_urls

        run_hosts = list(self._agent.network_hosts())
        if self._phase_split_egress:
            return allowlist_from_urls([])
        if _all_ipv4(run_hosts):
            return allowlist_from_urls(run_hosts)
        return allowlist_from_urls(run_hosts + list(self._agent.install_hosts()))

    async def install(self, environment: BaseEnvironment) -> None:
        """INSTALL phase (network open): run the declarative step (git bootstrap) then the agent's
        own ``install`` (clone + build) in the trial container. An :class:`AgentInstallError` is
        captured and surfaced from :meth:`run` rather than crashing pier's install phase.

        Fresh Beagle cluster jobs acquire an open raw container for this trusted phase. Legacy/local
        jobs retain Pier's existing proxy behavior."""
        await super().install(environment)  # runs install_spec steps (git bootstrap, root)
        loop = asyncio.get_running_loop()
        # Route the agent's commands (install clone + run) through pier's own ``agent_process_env``:
        # on a filtered-egress trial (DeepSWE, ``allow_internet=False``) that injects the Squid
        # egress-proxy vars, so the clone reaches the allowlisted git/package hosts and the run
        # reaches the allowlisted LLM gateway — the container's only route out. Identity when the
        # trial isn't filtered-egress. This reuses pier's feature, exactly as pier's native
        # installed-agent ``_exec`` does; the generic runtime stays egress-agnostic.
        self._runtime = HarborEnvRuntime(environment, loop, env_hook=environment.agent_process_env)
        self._handle = self._runtime.acquire()  # the trial container (no new container)
        # Pin git to pier's authenticated egress proxy at the highest git-config level, OVERRIDING
        # any ``http.proxy`` the task's base image baked in (git config wins over the ``https_proxy``
        # env var, so a baked cred-less proxy would send git to a 407). Guarded on the proxy env, so
        # it's a no-op off filtered-egress. Use the env's ASYNC exec directly (we're on the loop
        # thread — the sync runtime.exec would deadlock), with agent_process_env supplying the Squid
        # proxy + its ``agent:<token>`` creds so ``$HTTPS_PROXY`` is set for the git config.
        await environment.exec(
            'if [ -n "$HTTPS_PROXY" ]; then git config --global http.proxy "$HTTPS_PROXY"; '
            'git config --global https.proxy "$HTTPS_PROXY"; fi',
            env=environment.agent_process_env(None))
        pwd = await environment.exec("pwd")
        self._task_ctx = TaskContext(image=None, repo_path=(pwd.stdout or "/").strip())
        try:
            await asyncio.to_thread(
                self._agent.install, self._handle, self._task_ctx, runtime=self._runtime)
        except AgentInstallError as e:
            self._install_error = str(e)

    def _declare_budget(self, *, spent_s: float = 0.0) -> None:
        """Put the TASK's clock on the context the agent receives, before it runs.

        REPLACES the context — ``TaskContext`` is frozen, so assigning the field would raise
        ``FrozenInstanceError`` on every trial that got this far. Unlike harbor, pier runs the
        agent's own install in its SETUP phase (this shim's :meth:`install`), so the whole budget
        is available to the run phase — but pier's own pre-agent work (the egress seal) still
        spends it, so ``spent_s`` carries that.
        """
        if self._task_ctx is not None:
            self._task_ctx = declare_task_budget(
                self._task_ctx, self._logs_dir.parent / "config.json", spent_s=spent_s)

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """RUN phase (network restricted to :meth:`network_allowlist`): run the already-installed
        agent via its sync ``run_in`` in a worker thread. The result is stashed on ``self``; pier
        calls :meth:`populate_context_post_run` after this (even on failure)."""
        if self._install_error is not None:
            self._result = TaskResult(
                task_id="trial", status=RolloutStatus.FAILED, error=self._install_error)
            return
        phase_started = time.monotonic()
        run_hosts = list(self._agent.network_hosts())
        apply_egress = getattr(environment, "apply_egress", None)
        if self._phase_split_egress:
            disabled = getattr(environment, "task_internet_disabled", None)
            if disabled is None or apply_egress is None:
                raise RuntimeError(
                    "phase-split Pier egress requires task_internet_disabled() and apply_egress()"
                )
            if disabled():
                cidrs, ports = await _resolve_and_pin_run_egress(environment, run_hosts)
                await apply_egress(cidrs, ports=ports)
        # Open-install seal (open-setup -> tighten). On the open-install path the container acquired
        # OPEN, so install ran with a DIRECT route; now — before the agent starts — restrict egress to
        # ONLY the agent's run hosts (the LLM gateway IP) via pier's spec-07 iptables ``apply_egress``,
        # a hard allowlist the unprivileged agent can't undo. Same effective restriction as the Squid
        # path, without its per-request tax. Gated on ``_all_ipv4`` (matches :meth:`network_allowlist`
        # + xrlenv's ``_egress_domains``): a no-op on the Squid path (hostname run host → no cidrs) and
        # off-cluster (``apply_egress`` absent, e.g. local mode / an online task).
        elif _all_ipv4(run_hosts) and apply_egress is not None:
            await apply_egress(_run_egress_cidrs(run_hosts))
        self._declare_budget(spent_s=time.monotonic() - phase_started)
        task = Task(task_id="trial", problem_statement=instruction, benchmark="")
        self._result = await asyncio.to_thread(
            self._agent.run_in, self._handle, task, self._task_ctx, runtime=self._runtime)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Pier's post-run hook (called after :meth:`run`, even on failure): feed the agent's tokens
        + metadata into the trial context. agent/trajectory.json (ATIF) is emitted by the harness
        POST-JOB (PierHarness inherits HarborHarness._emit_trajectories), not here."""
        result = self._result
        if result is None:
            return
        tokens = result.tokens or {}
        # harbor's n_input_tokens is total input INCLUDING cache (== beagle ``prompt``); n_cache_tokens
        # is the cached subset. Set BOTH so the cache split survives the harbor round-trip into
        # result.json / run.json — omitting n_cache_tokens (the old bug) zeroed run.json's cache buckets.
        context.n_input_tokens = tokens.get("prompt")
        context.n_output_tokens = tokens.get("completion")
        context.n_cache_tokens = (tokens.get("cache_read") or 0) + (tokens.get("cache_write") or 0)
        context.metadata = {
            "agent": self._identity.get("agent"),
            "error": result.error,
            "patch": result.patch,
        }


__all__ = ["BeaglePierAgent"]
