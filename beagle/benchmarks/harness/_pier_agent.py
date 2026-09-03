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
import time
from pathlib import Path
from typing import Any

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
    import ipaddress
    from urllib.parse import urlparse

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


class BeaglePierAgent(BaseInstalledAgent):
    """Adapts any beagle ``Runnable`` to pier's installed-agent interface (see the harbor shim)."""

    SUPPORTS_ATIF: bool = False
    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "beagle"

    def __init__(self, logs_dir: Path, *, identity: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(logs_dir, **kwargs)
        self._identity = identity
        self._agent = _rebuild_agent(identity)
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
        """The trial's egress allowlist. Two shapes, keyed on whether every RUN host is a bare IPv4:

        * **Open-install path** (all-IPv4 run hosts — e.g. deep-swe's LLM-gateway IP): return only
          the RUN hosts. xrlenv routes such a trial onto the single-container **OPEN** acquire, so the
          trusted install phase runs with a DIRECT route (its clone/build hosts need no allowlist —
          the ~9x Squid install tax is gone) and :meth:`run` iptables-seals the run phase to these IPs.
        * **Squid path** (a run host is a hostname → needs DNS-aware domain filtering): include the
          INSTALL hosts too, because pier then applies one allowlist for the whole trial.

        Uses pier's native ``allowlist_from_urls``. Kept in lockstep with xrlenv's ``_egress_domains``."""
        from pier.agents.network import allowlist_from_urls

        run_hosts = list(self._agent.network_hosts())
        if _all_ipv4(run_hosts):
            return allowlist_from_urls(run_hosts)
        return allowlist_from_urls(run_hosts + list(self._agent.install_hosts()))

    async def install(self, environment: BaseEnvironment) -> None:
        """INSTALL phase (network open): run the declarative step (git bootstrap) then the agent's
        own ``install`` (clone + build) in the trial container. An :class:`AgentInstallError` is
        captured and surfaced from :meth:`run` rather than crashing pier's install phase.

        NOTE (open-install is not possible here): pier's container has NO direct egress — the Squid
        proxy is its only route out, DNS included (confirmed: dropping the proxy env makes the clone
        fail ``Could not resolve host``). And pier's ``BaseEnvironment`` exposes no ``apply_egress``
        primitive (harbor's ``PUBLIC → ALLOWLIST`` per-phase policy has no pier equivalent), so beagle
        can't open a direct route for install. Everything — install AND run — must go through Squid;
        a heavy install (opencode's ~1.5k-package ``bun install``) that out-waits pier's ~360s setup
        window is accommodated by raising the setup-timeout, not by bypassing the proxy."""
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
        # Open-install seal (open-setup -> tighten). On the open-install path the container acquired
        # OPEN, so install ran with a DIRECT route; now — before the agent starts — restrict egress to
        # ONLY the agent's run hosts (the LLM gateway IP) via pier's spec-07 iptables ``apply_egress``,
        # a hard allowlist the unprivileged agent can't undo. Same effective restriction as the Squid
        # path, without its per-request tax. Gated on ``_all_ipv4`` (matches :meth:`network_allowlist`
        # + xrlenv's ``_egress_domains``): a no-op on the Squid path (hostname run host → no cidrs) and
        # off-cluster (``apply_egress`` absent, e.g. local mode / an online task).
        run_hosts = list(self._agent.network_hosts())
        apply_egress = getattr(environment, "apply_egress", None)
        if _all_ipv4(run_hosts) and apply_egress is not None:
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
