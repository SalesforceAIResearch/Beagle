"""opencode — adapter for the open-source Bun coding agent.

This whole file is what you drop to onboard opencode (a Bun monorepo whose headless
entry is ``bun packages/opencode/src/index.ts run``). It runs **in-container**: the exact
version (``repo @ ref``) is cloned into the task container and built from source with Bun,
so an un-evolved run (baseline ref) and an evolved run (a candidate branch) use the
identical path — only the ref differs.

``run`` is the single, harness-agnostic integration point on the always-open path
(docker / harbor); a network-phased harness (pier/DeepSWE) drives :meth:`install` +
:meth:`run_in` across its own phases instead. Either way it's the same clone→build→invoke
path against the given :class:`ContainerRuntime` — no per-harness opencode class. The pure
bash-building / stream-parsing helpers live in :mod:`beagle.agents.opencode._helpers`.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import replace
from pathlib import Path

from beagle.agents.opencode._helpers import (
    DEFAULT_BUN_VERSION,
    DEFAULT_CONTAINER_PATH,
    DEFAULT_INSTALL_CMD,
    DEFAULT_MAX_TURNS,
    DEFAULT_OPENCODE_ARGS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROVIDER_ID,
    DEFAULT_TIMEOUT_SEC,
    OPENCODE_ENTRYPOINT,
    OpenCodeConfig,
    build_install_script,
    build_inner_script,
    build_provider_config,
    count_opencode_turns,
    last_stream_error,
    parse_combined_output,
    parse_opencode_usage,
    summarize_opencode_failure,
)
from beagle.agents.core.base import (
    Agent,
    AgentInstallError,
    AgentSource,
    EditResult,
    Editor,
    Evolvable,
    Runnable,
    Topology,
)
from beagle.agents.core.registry import register
from beagle.rollout.runtime import ContainerRuntime
from beagle.rollout.runtime.transport import GitClone, clone_with_retry
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult, Transparency, TrajectoryRef


@register("opencode")
class OpenCodeAgent(Agent, Runnable, Evolvable, Editor):
    """The opencode harness — white-box, usable as evolvee or evolver.

    Config keys (``spec.config``): ``container_path``, ``bun_version``, ``install_cmd``,
    ``opencode_args``, ``provider`` (LLM gateway → the opencode provider id the gateway is
    registered under and the ``--model`` prefix; without it opencode uses its own provider
    defaults, which fail in a sealed container), ``effort`` (reasoning level → ``--variant``;
    survives the harbor shim, unlike the model's ``reasoning_effort``), ``forward_env``,
    ``max_turns`` (accepted for a uniform vocabulary but a no-op — opencode has no turn-cap
    flag), ``timeout``, ``output_dir``, ``token_env`` (env var holding a clone credential for
    a private experiment copy).
    """

    transparency = Transparency.WHITE_BOX
    topology = Topology.IN_CONTAINER

    def _default_source(self) -> AgentSource:
        """Baseline source. The repo is **user-supplied** run config, never hardcoded.

        Every evolvable agent points at an *experiment copy you own* (the evolver pushes
        candidate branches there). beagle ships only opencode-intrinsic defaults (the Bun
        ``src/index.ts`` entrypoint) — the repo/ref come from ``evolvee.source`` in your run
        config.
        """
        src = self.spec.source
        if src is None or not src.repo:
            raise ValueError(
                "opencode requires an experiment-copy repo: set `evolvee.source.repo` "
                "(and `ref`) in your run config. beagle never hardcodes the repo — you own "
                "the copy and the evolver pushes candidate branches to it (experiment-copy rule)."
            )
        return src if src.entrypoint else replace(src, entrypoint=OPENCODE_ENTRYPOINT)

    def _config(self, model: str, entrypoint: str) -> OpenCodeConfig:
        c = self.config
        from beagle.agents.core.forward_env import normalize_forward_env

        opencode_args = tuple(c["opencode_args"]) if "opencode_args" in c else DEFAULT_OPENCODE_ARGS
        # opencode's reasoning effort → ``--variant <level>``. Sourced from the ``effort`` config
        # key (which SURVIVES the harbor shim — the shim serializes agent.config but DROPS the model
        # block, so a model's typed ``reasoning_effort`` is lost on the harbor path) or, as a fallback
        # for the docker path (full spec preserved), the model's ``reasoning_effort``.
        variant = c.get("effort") or getattr(self.spec.model, "reasoning_effort", None) or ""
        # The provider (LLM gateway) is the opencode provider id we register the gateway under and
        # prefix ``--model`` with, so a caller selects the gateway without restating behavior flags.
        provider_id = str(c.get("provider") or DEFAULT_PROVIDER_ID)
        forward_env = tuple(normalize_forward_env(c.get("forward_env")))  # (container, host) pairs
        return OpenCodeConfig(
            model=model,
            container_path=c.get("container_path", DEFAULT_CONTAINER_PATH),
            entrypoint=entrypoint or OPENCODE_ENTRYPOINT,
            bun_version=str(c.get("bun_version", DEFAULT_BUN_VERSION)),
            install_cmd=c.get("install_cmd", DEFAULT_INSTALL_CMD),
            opencode_args=opencode_args,
            provider_id=provider_id,
            variant=str(variant),
            forward_env=forward_env,
            max_turns=int(c.get("max_turns", DEFAULT_MAX_TURNS)),
            timeout=float(c.get("timeout", DEFAULT_TIMEOUT_SEC)),
            output_dir=c.get("output_dir", DEFAULT_OUTPUT_DIR),
        )

    def _cfg(self) -> OpenCodeConfig:
        """The resolved :class:`OpenCodeConfig` for this rollout (source ref + model + config knobs)."""
        src = self.source()  # baseline or bound candidate ref; raises if no repo
        model = self.spec.model.name if self.spec.model else "gpt-5.5"
        return self._config(model, src.entrypoint)

    def _acquire_run_args(self) -> list[str]:
        # Clear the image ENTRYPOINT so ``sleep infinity`` runs on the docker path (ignored by harbor).
        # The shared run() seam (base.Runnable.run) does acquire→install→run_in→destroy + phase timing.
        return ["--entrypoint", ""]

    def _install_error_result(self, task: Task, message: str) -> TaskResult:
        # Attach opencode's trajectory ref to a broken clone/build so the failure carries the stream path.
        return self._error(task, self._cfg(), message)

    def install(self, handle: object, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> None:
        """INSTALL phase (network-open on a phased harness): clone the exact opencode ``repo@ref`` into
        ``container_path`` and build it (toolchain + Bun bootstrap + ``bun install``). Raises
        :class:`AgentInstallError` so a broken install surfaces as a FAILED rollout, not a silent empty
        patch. The experiment copy is typically private, so the clone is token-authenticated via the
        shared GitClone helper."""
        src = self.source()
        cfg = self._config(self.spec.model.name if self.spec.model else "gpt-5.5", src.entrypoint)
        token_env = self.config.get("token_env")
        clone_env: dict[str, str] = {}
        if token_env:
            val = os.environ.get(token_env)
            if not val:
                raise AgentInstallError(
                    f"token_env {token_env!r} is named in config but not set in the environment "
                    f"(needed to clone the private experiment copy)")
            clone_env[token_env] = val
        clone = GitClone(repo_url=src.repo, ref=src.ref or "",
                         container_path=cfg.container_path, token_env=token_env)
        # Retry a transient clone failure (GitHub auth-throttles many trials cloning the same private
        # repo at once → a spurious 401 mid ref-negotiation) with backoff + jitter; definitive errors
        # still fail fast.
        r = clone_with_retry(runtime, handle, clone, env=clone_env or None, timeout=300)
        if not r.ok:
            raise AgentInstallError(f"git clone failed (rc={r.returncode}): {r.stderr.strip()!r}")
        # bun install on a 30+-package monorepo is heavy; give it a generous ceiling.
        r = runtime.exec(handle, ["bash", "-lc", build_install_script(cfg)], timeout=1200)
        if not r.ok:
            raise AgentInstallError(f"install failed (rc={r.returncode}): {r.stderr.strip()!r}")

    def run_in(
        self, handle: object, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime
    ) -> TaskResult:
        """RUN phase (network restricted to :meth:`network_hosts` — the LLM gateway): invoke opencode
        once on the task (prompt piped via stdin as DATA to dodge argv limits; opencode frames it — no
        per-benchmark templating; the gateway is declared via opencode's native ``OPENCODE_CONFIG_CONTENT``
        and gateway creds ride ``forward_env``), then COMMIT the worktree so a ``git diff base..HEAD``
        grader (DeepSWE/pier) sees opencode's edits. The commit is a no-op for a working-tree grader
        (swe-bench/tb). Maps output → TaskResult."""
        cfg = self._config(self.spec.model.name if self.spec.model else "gpt-5.5",
                           self.source().entrypoint)
        env = {"OPENCODE_PROMPT": task.prompt()}
        # Point opencode at the LLM gateway via its native config-content env — an OpenAI-compatible
        # provider block (no file, no workspace pollution). Bun's fetch honors HTTPS_PROXY natively, so
        # on a filtered-egress trial (DeepSWE/pier) the harness's injected proxy env is enough — no
        # undici/preload shim. When no gateway is configured, opencode falls back to its own provider
        # resolution (a direct key), which fails in a sealed container — the same contract as monet.
        from beagle.agents.core.litellm_gateway import gateway_litellm_kwargs

        gateway = gateway_litellm_kwargs()
        if gateway and gateway.get("api_base"):
            env["OPENCODE_CONFIG_CONTENT"] = build_provider_config(cfg, gateway)
            # Our provider block declares the model inline, so opencode never needs its remote model
            # catalog. Disable that fetch (opencode's native flag): from *source* the build-time model
            # snapshot is absent, so opencode would otherwise GET https://models.opencode.ai/api.json at
            # startup and — because that populate path is `Effect.orDie` — a restricted-egress trial
            # (DeepSWE/pier seals the run phase to the gateway IP) turns the blocked fetch into a fatal
            # "Unexpected server error" *before the first LLM call*. The prebuilt binary bakes the
            # snapshot in and never hits this; running the source (required for an evolved ref) does.
            env["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
        for container_name, host_name in cfg.forward_env:
            v = os.environ.get(host_name)
            if v is not None:
                env[container_name] = v
        # Record the pre-run commit so we can recover opencode's edits as `git diff base..HEAD`
        # (opencode edits the worktree directly; committing seals them for pier's submission).
        repo = shlex.quote(task_ctx.repo_path)
        base = runtime.exec(handle, ["bash", "-lc", f"cd {repo} && git rev-parse HEAD 2>/dev/null || true"])
        base_ref = base.stdout.strip()
        script = build_inner_script(
            cfg, repo_path=task_ctx.repo_path, shell_preamble=task_ctx.shell_preamble)
        invoke = runtime.exec(handle, ["bash", "-lc", script], env=env, timeout=cfg.timeout)
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
        """The LLM gateway opencode reaches during :meth:`run_in` — allowlisted on a restricted run
        phase (DeepSWE/pier). Same gateway env every agent uses; empty when none is configured."""
        from beagle.agents.core.litellm_gateway import gateway_litellm_kwargs

        kw = gateway_litellm_kwargs()
        return [kw["api_base"]] if kw and kw.get("api_base") else []

    def install_hosts(self) -> list[str]:
        """Hosts :meth:`install` reaches: opencode's git host + the Bun/npm indexes its bootstrap pulls
        (a task image lacking Bun fetches it from bun.sh; workspace deps + the ai-sdk packages via the
        npm registry). ``bun install`` also builds native modules (tree-sitter grammars, node-pty) via
        ``node-gyp rebuild``, which downloads Node headers from ``nodejs.org`` — without it, a
        filtered-egress trial's Squid returns 403 and the native build (and thus install) fails.
        Allowlisted so INSTALL can clone + build behind a filtered-egress benchmark's proxy. Best-effort
        across base-image package managers."""
        from urllib.parse import urlparse

        hosts = [
            "github.com", "codeload.github.com", "objects.githubusercontent.com",  # clone
            "raw.githubusercontent.com",
            "bun.sh", "bun.com",                                                   # bun installer
            "registry.npmjs.org",                                                  # bun install deps
            "nodejs.org",                                                          # node-gyp headers (native builds)
            "deb.nodesource.com",                                                  # node (nodesource apt repo, if base image pins it)
            "deb.debian.org", "security.debian.org",                               # apt (debian)
            "archive.ubuntu.com", "security.ubuntu.com", "ports.ubuntu.com",       # apt (ubuntu)
            "dl-cdn.alpinelinux.org",                                              # apk (alpine)
        ]
        src_host = urlparse(self.source().repo).hostname
        if src_host and src_host not in hosts:
            hosts.append(src_host)
        return hosts

    # -- result mapping ------------------------------------------------------

    def _result(
        self, task: Task, cfg: OpenCodeConfig, rc: int, stdout: str, stderr: str,
        *, base_patch: str = "",
    ) -> TaskResult:
        # rc 124 = exec timeout, rc 125 = runtime/transport error (both from the runtime
        # layer, distinct from opencode's own exit code parsed below).
        error: str | None = None
        if rc == 124:
            error = f"timeout after {cfg.timeout}s"
        elif rc == 125:
            error = f"runtime error during exec: {stderr.strip()!r}"
        opencode_rc, stream = parse_combined_output(stdout)
        patch = base_patch if base_patch.strip() else ""
        if error is None and opencode_rc not in (None, 0):
            error = summarize_opencode_failure(opencode_rc, stderr)  # type: ignore[arg-type]
        if error is None:
            se = last_stream_error(stream)
            if se is not None:
                error = f"stream_error: {se}"
        usage = parse_opencode_usage(stream)
        # Resolved = opencode finished a normal run with no surfaced error. The scorable patch
        # (base..HEAD) is graded by the benchmark; opencode itself reports success via rc + stream.
        resolved = opencode_rc == 0 and error is None
        return TaskResult(
            task_id=task.task_id,
            status=RolloutStatus.FAILED if error else RolloutStatus.COMPLETED,
            resolved=resolved,
            patch=patch or None,
            num_turns=count_opencode_turns(stream),
            tokens=usage.to_token_counts(),
            error=error,
            trajectory=TrajectoryRef(path=Path(cfg.stream_path), format="opencode-json"),
            # Surface the native stream so the docker-drop-in path (DockerHarness/swe-bench, which
            # tears the container down before the host can sync /logs/agent) persists it as
            # agent/opencode.stream.jsonl + converts it to agent/trajectory.json (ATIF). The
            # harbor/pier paths instead sync the on-disk stream file post-step; both land the same
            # canonical artifact.
            trajectory_text=stream or None,
        )

    def _error(self, task: Task, cfg: OpenCodeConfig, message: str) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            status=RolloutStatus.FAILED,
            error=message,
            trajectory=TrajectoryRef(path=Path(cfg.stream_path), format="opencode-json"),
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
        # opencode as a coding agent (evolver): run its binary host-side against the
        # workspace. Edits are left in workspace git state.
        raise NotImplementedError("opencode edit() not yet implemented")


__all__ = ["OpenCodeAgent"]
