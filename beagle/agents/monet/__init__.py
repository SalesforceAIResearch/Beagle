"""Monet — reference adapter for an internally-developed Node agent.

This whole file is what you drop to onboard monet (a Node package with entry
``bin/monet.js``). It runs **in-container**: the exact version (``repo @ ref``) is
cloned into the task container and built, so an un-evolved run (baseline ref) and an
evolved run (a candidate branch) use the identical path — only the ref differs.
Monet is claude-code-shaped but ours, so it can be evolvee or evolver.

``run`` is the single, harness-agnostic integration point: it drives the given
:class:`ContainerRuntime` (a real Docker container on the docker path; harbor's
trial environment, wrapped, on the harbor path) — so there is **no** per-harness
monet class. The pure bash-building / stream-parsing helpers live in
:mod:`beagle.agents.monet._helpers`.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import replace
from pathlib import Path

from beagle.agents.core.base import (
    Agent,
    AgentInstallError,
    AgentSource,
    Editor,
    EditResult,
    Evolvable,
    Runnable,
    Topology,
    resolve_agent_timeout,
)
from beagle.agents.core.registry import register
from beagle.agents.monet._helpers import (
    DEFAULT_CONTAINER_PATH,
    DEFAULT_INSTALL_CMD,
    DEFAULT_MAX_TURNS,
    DEFAULT_MONET_ARGS,
    DEFAULT_OUTPUT_DIR,
    MONET_BIN_PATH,
    MonetConfig,
    build_inner_script,
    build_install_script,
    count_monet_turns,
    hit_max_turns,
    last_stream_error,
    parse_combined_output,
    parse_monet_usage,
    summarize_monet_failure,
)
from beagle.rollout.runtime import ContainerRuntime
from beagle.rollout.runtime.transport import GitClone, clone_with_retry
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult, TrajectoryRef, Transparency


@register("monet")
class MonetAgent(Agent, Runnable, Evolvable, Editor):
    """The monet harness — white-box, usable as evolvee or evolver.

    Config keys (``spec.config``): ``container_path``, ``install_cmd``,
    ``monet_args``, ``provider`` (LLM gateway name → prepends ``--provider``; without
    it monet uses a direct API key), ``effort`` (reasoning level → prepends
    ``--effort``; survives the harbor shim, unlike the model's ``reasoning_effort``),
    ``forward_env``, ``max_turns``, ``timeout``, ``max_tokens``, ``output_dir``,
    ``token_env`` (env var holding a clone credential for a private experiment copy).
    """

    transparency = Transparency.WHITE_BOX
    topology = Topology.IN_CONTAINER

    def _default_source(self) -> AgentSource:
        """Baseline source. The repo is **user-supplied** run config, never hardcoded.

        Every evolvable agent points at an *experiment copy you own* (the evolver
        pushes candidate branches there). beagle ships only monet-intrinsic
        defaults (the ``bin/monet.js`` entrypoint) — the repo/ref come from
        ``evolvee.source`` in your run config.
        """
        src = self.spec.source
        if src is None or not src.repo:
            raise ValueError(
                "monet requires an experiment-copy repo: set `evolvee.source.repo` "
                "(and `ref`) in your run config. beagle never hardcodes the repo — "
                "you own the copy and the evolver pushes candidate branches to it "
                "(experiment-copy rule)."
            )
        return src if src.entrypoint else replace(src, entrypoint=MONET_BIN_PATH)

    def _config(self, model: str, entrypoint: str) -> MonetConfig:
        c = self.config
        from beagle.agents.core.forward_env import normalize_forward_env

        base_args = tuple(c["monet_args"]) if "monet_args" in c else DEFAULT_MONET_ARGS
        # The provider (LLM gateway) is a distinct knob from monet's behavior flags: prepend
        # ``--provider <name>`` when configured, so a caller selects the gateway WITHOUT having
        # to restate the default behavior flags (``monet_args`` replaces them wholesale). Without
        # a provider monet uses its built-in default (a direct API key), which fails in a sealed
        # benchmark container ("No Anthropic API key configured").
        # monet's reasoning effort → ``--effort <level>``: a distinct knob, prepended like the
        # provider. Sourced from the ``effort`` config key (which SURVIVES the harbor shim — the
        # shim serializes agent.config but DROPS the model block, so a model's typed
        # ``reasoning_effort`` is lost on the harbor path) or, as a fallback for the docker path
        # (full spec preserved), the model's ``reasoning_effort``. Without it monet runs at its
        # default ``none`` (minimal reasoning) — a large quality drop on hard tasks. Skipped if the
        # caller already spelled ``--effort`` into ``monet_args``.
        effort = c.get("effort") or getattr(self.spec.model, "reasoning_effort", None)
        if effort and "--effort" not in base_args:
            base_args = ("--effort", str(effort)) + base_args

        provider = c.get("provider")
        monet_args = (("--provider", str(provider)) + base_args) if provider else base_args
        forward_env = tuple(normalize_forward_env(c.get("forward_env")))  # (container, host) pairs
        return MonetConfig(
            model=model,
            container_path=c.get("container_path", DEFAULT_CONTAINER_PATH),
            entrypoint=entrypoint or MONET_BIN_PATH,
            install_cmd=c.get("install_cmd", DEFAULT_INSTALL_CMD),
            monet_args=monet_args,
            forward_env=forward_env,
            max_turns=int(c.get("max_turns", DEFAULT_MAX_TURNS)),
            max_tokens=c.get("max_tokens"),
            output_dir=c.get("output_dir", DEFAULT_OUTPUT_DIR),
        )

    def _cfg(self) -> MonetConfig:
        """The resolved :class:`MonetConfig` for this rollout (source ref + model + config knobs)."""
        src = self.source()  # baseline or bound candidate ref; raises if no repo
        model = self.spec.model.name if self.spec.model else "claude-opus-4-8"
        return self._config(model, src.entrypoint)

    def _acquire_run_args(self) -> list[str]:
        # Clear the image ENTRYPOINT so ``sleep infinity`` runs on the docker path (ignored by harbor).
        # The shared run() seam (base.Runnable.run) does acquire→install→run_in→destroy + phase timing.
        return ["--entrypoint", ""]

    def _install_error_result(self, task: Task, message: str) -> TaskResult:
        # Attach monet's trajectory ref to a broken clone/build so the failure carries the stream path.
        return self._error(task, self._cfg(), message)

    def install(self, handle: object, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> None:
        """INSTALL phase (network-open on a phased harness): clone the exact monet ``repo@ref`` into
        ``container_path`` and build it (node bootstrap + ``npm ci``). Raises
        :class:`AgentInstallError` so a broken install surfaces as a FAILED rollout, not a silent
        empty patch. The experiment copy is typically private, so the clone is token-authenticated
        via the shared GitClone helper."""
        src = self.source()
        cfg = self._config(self.spec.model.name if self.spec.model else "claude-opus-4-8",
                           src.entrypoint)
        token_env = self.config.get("token_env")
        clone_env: dict[str, str] = {}
        if token_env:
            val = os.environ.get(token_env)
            if not val:
                raise AgentInstallError(
                    f"token_env {token_env!r} is named in config but not set in the "
                    f"environment (needed to clone the private experiment copy)")
            clone_env[token_env] = val
        clone = GitClone(repo_url=src.repo, ref=src.ref or "",
                         container_path=cfg.container_path, token_env=token_env)
        # Retry a transient clone failure (concurrent trials cloning the same private repo trip GitHub's
        # auth throttle → a spurious 401) with backoff + jitter; definitive errors still fail fast.
        r = clone_with_retry(runtime, handle, clone, env=clone_env or None, timeout=300)
        if not r.ok:
            raise AgentInstallError(f"git clone failed (rc={r.returncode}): {r.stderr.strip()!r}")
        r = runtime.exec(handle, ["bash", "-lc", build_install_script(cfg)], timeout=600)
        if not r.ok:
            raise AgentInstallError(f"install failed (rc={r.returncode}): {r.stderr.strip()!r}")
        # Filtered-egress proxy support for monet's Node native fetch (RUN phase reaches the gateway
        # only through the harness's Squid proxy, but native fetch ignores HTTP_PROXY). Install undici
        # + a preload that points the process-wide dispatcher (a symbol native fetch also reads) at
        # the proxy env. Done in INSTALL — RUN has no network to install then. Best-effort (guarded);
        # a no-op off filtered-egress (no proxy env → EnvHttpProxyAgent tunnels nothing).
        cp = shlex.quote(cfg.container_path)
        # Pin undici to the version Node BUNDLES (process.versions.undici) — a mismatched npm undici
        # sets an incompatible global dispatcher ("invalid onRequestStart method") that breaks the
        # very native fetch we're trying to proxy. Fall back to latest if the version can't be read.
        runtime.exec(handle, ["bash", "-lc", (
            f'cd {cp} && UV=$(node -e "process.stdout.write(process.versions.undici||\'\')" 2>/dev/null) && '
            'npm install "undici${UV:+@$UV}" --no-save --no-audit --no-fund >/dev/null 2>&1 || '
            'npm install undici --no-save --no-audit --no-fund >/dev/null 2>&1 || true')],
            timeout=180)
        # Two things, both in beagle's OWN preload (never monet's code): (1) route native fetch at the
        # filtered-egress proxy; (2) **diagnostics only** — subscribe to undici's connection channels so
        # the RAW transport error (ECONNRESET / UND_ERR_SOCKET / connect-timeout / TLS) is logged before
        # monet reduces it to a bare "fetch failed". Subscribing does NOT alter control flow — monet still
        # fails identically; we just surface the cause the infra drop leaves invisible. Guarded no-op if
        # undici/diagnostics_channel differ.
        preload = (
            'try{const u=require("undici");u.setGlobalDispatcher(new u.EnvHttpProxyAgent())}catch(e){}'
            'try{const dc=require("diagnostics_channel");'
            'const L=(t,m)=>{const e=(m&&m.error)||m;const c=e&&(e.code||(e.cause&&e.cause.code));'
            'console.error("[beagle-net] "+t+" code="+c+" msg="+(e&&e.message)'
            '+" cause="+(e&&e.cause&&(e.cause.message||e.cause)));};'
            'dc.subscribe("undici:client:connectError",m=>L("connectError",m));'
            'dc.subscribe("undici:request:error",m=>L("requestError",m));}catch(e){}'
        )
        runtime.exec(handle, ["bash", "-lc",
            f"printf '%s' {shlex.quote(preload)} > {cp}/beagle-proxy.cjs"])

    def run_in(
        self, handle: object, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime
    ) -> TaskResult:
        """RUN phase (network restricted to :meth:`network_hosts` — the LLM gateway): invoke monet
        once on the task (prompt as DATA via env to dodge argv limits; monet frames it — no
        per-benchmark templating; gateway creds ride ``forward_env``), then COMMIT the worktree so a
        ``git diff base..HEAD`` grader (DeepSWE/pier) sees monet's edits. The commit is a no-op for a
        working-tree grader (swe-bench/tb) — it changes no file contents. Maps output → TaskResult."""
        cfg = self._config(self.spec.model.name if self.spec.model else "claude-opus-4-8",
                           self.source().entrypoint)
        # Make monet's Node native fetch honor the harness's Squid proxy on a filtered-egress trial
        # (DeepSWE/pier) — its LLM call to the gateway is the container's only route out. The proxy
        # env is injected into agent commands by agent_process_env; two belts: NODE_USE_ENV_PROXY
        # (Node 24+ built-in) and the install-time undici preload (older Node). Both no-op off
        # filtered-egress (no proxy env → direct fetch, unchanged).
        env = {
            "MONET_PROMPT": task.prompt(),
            "NODE_USE_ENV_PROXY": "1",
            "NODE_OPTIONS": f"--require {cfg.container_path}/beagle-proxy.cjs",
        }
        for container_name, host_name in cfg.forward_env:
            v = os.environ.get(host_name)
            if v is not None:
                env[container_name] = v
        # Record the pre-run commit so we can recover monet's edits as `git diff base..HEAD`
        # below — monet self-commits inside the worktree, so its stream carries an empty patch
        # (the diff is already sealed in commits); base..HEAD is the authoritative submission.
        repo = shlex.quote(task_ctx.repo_path)
        base = runtime.exec(handle, ["bash", "-lc", f"cd {repo} && git rev-parse HEAD 2>/dev/null || true"])
        base_ref = base.stdout.strip()
        script = build_inner_script(
            cfg, repo_path=task_ctx.repo_path, shell_preamble=task_ctx.shell_preamble)
        # The rollout's wall clock: what the task/benchmark DECLARED, lowered by an explicit
        # agent.timeout if the run config sets one. No house default — see resolve_agent_timeout.
        cfg = replace(cfg, timeout=resolve_agent_timeout(self.config, task_ctx))
        invoke = runtime.exec(handle, ["bash", "-lc", script], env=env, timeout=cfg.timeout)
        # monet's inner script may leave edits staged/uncommitted; commit them so pier's
        # `git diff base..HEAD` submission reflects the work (DeepSWE agents must commit).
        runtime.exec(handle, ["bash", "-lc",
            f'cd {repo} && git add -A && git -c user.email=agent@beagle.local '
            f'-c user.name=beagle commit -q -m "beagle agent changes" || true'])
        base_patch = ""
        if base_ref:
            diff = runtime.exec(handle, ["bash", "-lc",
                f"cd {repo} && git diff {shlex.quote(base_ref)}..HEAD 2>/dev/null || true"])
            base_patch = diff.stdout
        return self._result(
            task, cfg, invoke.returncode, invoke.stdout, invoke.stderr, base_patch=base_patch)

    def network_hosts(self) -> list[str]:
        """The LLM gateway monet reaches during :meth:`run_in` — allowlisted on a restricted run
        phase (DeepSWE/pier). Same gateway env every agent uses; empty when none is configured."""
        from beagle.agents.core.litellm_gateway import gateway_litellm_kwargs

        kw = gateway_litellm_kwargs()
        return [kw["api_base"]] if kw and kw.get("api_base") else []

    def install_hosts(self) -> list[str]:
        """Hosts :meth:`install` reaches: monet's git host + the node/npm indexes its bootstrap pulls
        (a task image lacking node >= 20.5 fetches node via nodesource/apt or apk; deps via npm).
        Allowlisted so INSTALL can clone + build behind a filtered-egress benchmark's proxy.
        Best-effort across base-image package managers."""
        from urllib.parse import urlparse

        hosts = [
            "github.com", "codeload.github.com", "objects.githubusercontent.com",  # clone
            "raw.githubusercontent.com",
            "registry.npmjs.org",                                                  # npm ci
            "deb.nodesource.com",                                                  # node (nodesource)
            "deb.debian.org", "security.debian.org",                               # apt (debian)
            "archive.ubuntu.com", "security.ubuntu.com", "ports.ubuntu.com",       # apt (ubuntu)
            "dl-cdn.alpinelinux.org",                                              # apk (alpine)
            "nodejs.org",                                                          # direct node dl
        ]
        src_host = urlparse(self.source().repo).hostname
        if src_host and src_host not in hosts:
            hosts.append(src_host)
        return hosts

    # -- result mapping ------------------------------------------------------

    def _result(
        self, task: Task, cfg: MonetConfig, rc: int, stdout: str, stderr: str,
        *, base_patch: str = "",
    ) -> TaskResult:
        # rc 124 = exec timeout, rc 125 = runtime/transport error (both from the
        # runtime layer, distinct from monet's own exit code parsed below).
        error: str | None = None
        if rc == 124:
            error = f"timeout after {cfg.timeout}s"
        elif rc == 125:
            error = f"runtime error during exec: {stderr.strip()!r}"
        monet_rc, patch, monet_stdout = parse_combined_output(stdout)
        # Prefer the committed diff (base..HEAD) when present — monet self-commits, so its stream
        # patch is empty even on a solved task; base..HEAD is the authoritative submission.
        if base_patch.strip():
            patch = base_patch
        max_turns = hit_max_turns(monet_stdout)
        if error is None and monet_rc not in (None, 0) and not max_turns:
            error = summarize_monet_failure(monet_rc, stderr)  # type: ignore[arg-type]
        if error is None and not patch:
            se = last_stream_error(monet_stdout)
            if se is not None:
                error = f"stream_error: {se}"
        usage = parse_monet_usage(monet_stdout)
        # Resolved = finished a normal run and left a scorable patch, OR hit the turn
        # cap but still produced one (budget exhaustion with a usable diff).
        resolved = (monet_rc == 0 and error is None) or (max_turns and bool(patch))
        return TaskResult(
            task_id=task.task_id,
            status=RolloutStatus.FAILED if error else RolloutStatus.COMPLETED,
            resolved=resolved,
            patch=patch or None,
            num_turns=count_monet_turns(monet_stdout),
            tokens=usage.to_token_counts(),
            error=error,
            trajectory=TrajectoryRef(path=Path(cfg.stream_path), format="monet-stream-json"),
            # Hand back the raw stream so the docker/swe-bench path persists agent/monet.stream.jsonl +
            # ATIF (its container is torn down before the host can sync /logs/agent). No-op on harbor/pier,
            # which bind-mount /logs/agent and never call the artifact-persist path. Mirrors opencode.
            trajectory_text=monet_stdout or None,
            # On failure, keep the FULL stderr — the one-line `error` is only its first meaningful line,
            # so the real cause (a gateway throw, a crash) is otherwise lost once the container is gone.
            stderr_text=(stderr or None) if error else None,
        )

    def _error(self, task: Task, cfg: MonetConfig, message: str) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            status=RolloutStatus.FAILED,
            error=message,
            trajectory=TrajectoryRef(path=Path(cfg.stream_path), format="monet-stream-json"),
        )

    def edit(
        self,
        instruction: str,
        workspace: Path,
        *,
        plan_mode: bool = False,
        model: str | None = None,
        timeout_s: int | None = None,
        extra_args: list[str] | None = None,
        log_path: str | Path | None = None,
    ) -> EditResult:
        # monet as a coding agent (evolver): run its binary host-side against the
        # workspace. Edits are left in workspace git state.
        raise NotImplementedError("monet edit() not yet implemented")


__all__ = ["MonetAgent"]
