"""mini-swe — reference adapter for the open-source mini-swe-agent.

Repo: https://github.com/swe-agent/mini-swe-agent. mini-swe is bash-only and
**config-driven** — its prompts and step template live in YAML under
``src/minisweagent/config/`` (e.g. ``benchmarks/swebench.yaml``). So its *evolvable
surface* is that YAML: an evolver edits it on a branch, and the run uses that ref.

We drive upstream's **own ``mini`` CLI** (the faithful single-task entrypoint —
https://mini-swe-agent.com/latest/usage/mini/), installed *inside* the beagle-provisioned
container and run with ``--environment-class local`` so it acts on that container's fs. This
keeps behavior identical to how upstream runs a single task and never reaches into mini-swe's
library internals. (Upstream's ``mini-extra swebench`` *batch* mode owns its own per-instance
containers + ``preds.json`` — that's a native-runner shape, not how beagle drives an agent per
task.)

mini-swe requires **Python ≥3.10**, but a task container pins its own (SWE-bench images ship 3.9).
So we do NOT install into the container's system Python — :data:`_MINI_INSTALL` uses ``uv`` to stand
up a managed Python 3.11 venv (``/agent/.venv``) and installs the evolved ``repo@ref`` there. Only
the mini *orchestrator* runs on 3.11; the agent's bash tool-calls still execute in the container's
own environment (mini's ``LocalEnvironment`` subprocess), so the repo's testbed Python is untouched.
"""

from __future__ import annotations

import os
import shlex
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
)
from beagle.agents.core.forward_env import normalize_forward_env
from beagle.agents.core.litellm_gateway import gateway_litellm_kwargs
from beagle.agents.core.registry import register
from beagle.agents.core.usage import Usage
from beagle.agents.core.usage import add as usage_add
from beagle.rollout.runtime import ContainerRuntime
from beagle.rollout.runtime.transport import GitClone, clone_with_retry
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult, TrajectoryRef, Transparency

#: In-container install of mini-swe on a **managed Python 3.11** (task containers pin an older one,
#: often 3.9, with a pip too old to PEP 660-install a pyproject-only project). ``uv`` is fetched by
#: its standalone installer into ``/agent/bin`` (curl→wget→pip fallback), then creates a 3.11 venv
#: (uv auto-downloads the interpreter) and editable-installs the already-cloned ``/agent``. The
#: `mini` entrypoint lands at ``/agent/.venv/bin/mini``. Fails loud (``set -e``) so a broken install
#: surfaces as a FAILED rollout, not a silent empty patch.
_MINI_INSTALL = r"""
set -e
mkdir -p /agent/bin
export UV_INSTALL_DIR=/agent/bin
if ! /agent/bin/uv --version >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then wget -qO- https://astral.sh/uv/install.sh | sh
  else pip install -q uv && cp "$(python -c 'import shutil;print(shutil.which("uv"))')" /agent/bin/uv
  fi
fi
/agent/bin/uv venv --python 3.11 /agent/.venv
/agent/bin/uv pip install -q -e /agent --python /agent/.venv/bin/python
"""


@register("mini-swe")
class MiniSweAgent(Agent, Runnable, Evolvable, Editor):
    """mini-swe-agent — white-box, usable as evolvee or evolver.

    Config keys (``spec.config``): the shared first-level vocabulary — ``provider`` (gates gateway
    routing), ``effort`` (→ ``model.model_kwargs.reasoning_effort``), ``max_turns`` (→
    ``agent.step_limit``) — plus ``config_path`` (mini's ``-c`` preset = the evolvable surface),
    ``timeout``, ``forward_env`` (``[container_var, host_var]`` gateway-creds pairs).
    """

    transparency = Transparency.WHITE_BOX
    topology = Topology.IN_CONTAINER  # native shape is HOST_DRIVER; see module docstring
    #: Point at your fork (so the evolver can push branches); upstream is the default.
    REPO = "https://github.com/swe-agent/mini-swe-agent"

    def _default_source(self) -> AgentSource:
        # The evolvable surface is the config YAML — that's the entrypoint.
        return self.spec.source or AgentSource(
            repo=self.REPO, entrypoint="src/minisweagent/config/benchmarks/swebench.yaml"
        )

    def install(self, handle: object, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> None:
        """INSTALL phase (network-open on a phased harness): fetch the exact repo@ref (carries the
        evolved config YAML) and install mini-swe on a managed Python 3.11 INSIDE ``handle``. The
        experiment copy is typically PRIVATE, so the clone is token-authenticated via the shared
        GitClone helper (URL rewrite with ``token_env`` + SHA fetch-by-commit). ``runtime.exec`` is
        check=False, so CHECK each step — a silent failure would read as a benign "scored 0"."""
        src = self.source()
        cfg = self.config
        token_env = cfg.get("token_env")
        clone_env: dict[str, str] = {}
        if token_env:
            val = os.environ.get(token_env)
            if not val:
                raise AgentInstallError(
                    f"token_env {token_env!r} is named in config but not set in the "
                    f"environment (needed to clone the private experiment copy)")
            clone_env[token_env] = val
        clone = GitClone(repo_url=src.repo, ref=src.ref or "", container_path="/agent",
                         token_env=token_env)
        # Retry a transient clone failure (concurrent trials cloning the same private repo trip GitHub's
        # auth throttle → a spurious 401) with backoff + jitter; definitive errors still fail fast.
        cloned = clone_with_retry(runtime, handle, clone, env=clone_env or None, timeout=600)
        if not cloned.ok:
            raise AgentInstallError(
                f"mini-swe clone failed (rc={cloned.returncode}): "
                f"{(cloned.stderr or cloned.stdout).strip()[:800]}")
        # Managed Python 3.11 (see _MINI_INSTALL) — the container's own Python is too old. Allow
        # time for uv to fetch the interpreter + deps.
        install = runtime.exec(handle, ["bash", "-lc", _MINI_INSTALL], timeout=900)
        if not install.ok:
            raise AgentInstallError(
                f"mini-swe install failed (rc={install.returncode}): "
                f"{(install.stderr or install.stdout).strip()[:800]}")

    def run_in(
        self, handle: object, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime
    ) -> TaskResult:
        """RUN phase (network restricted to :meth:`network_hosts` — the gateway): drive upstream's
        own ``mini`` CLI (already installed) against the container, capture the patch, and COMMIT it
        so BOTH a working-tree grader (swe-bench reads ``TaskResult.patch``) and a ``base..HEAD``
        grader (deep-swe/pier's ``verifier.collect``) see the work."""
        src = self.source()
        model = self.spec.model.name if self.spec.model else "gpt-5"
        cfg = self.config
        config_path = f"/agent/{cfg.get('config_path', src.entrypoint)}"
        # Upstream's OWN `mini` CLI, non-interactive single run (https://mini-swe-agent.com): -t task,
        # -m model, -y (yolo), --exit-immediately, --agent-class default (non-interactive), -c the
        # config preset (the evolvable surface), -l 0 (no cost cap). The first-level vocabulary
        # (effort/max_turns) rides `vocab`, gateway routing rides `gw` (gated on `provider`), all as
        # `-c` overrides that layer over the preset; MSWEA_CONFIGURED skips mini's TTY-requiring
        # first-time wizard; creds ride forward_env.
        gw = _mini_gateway_c_args() if cfg.get("provider") else ""
        vocab = _mini_vocab_c_args(cfg)
        ov = _mini_prompt_override_c_args(self.prompt_override())
        run_env = {"MSWEA_CONFIGURED": "1"}
        run_env.update({c: os.environ[h] for c, h in
                        normalize_forward_env(cfg.get("forward_env")) if os.environ.get(h) is not None})
        repo = shlex.quote(task_ctx.repo_path)
        # Record the base commit BEFORE the agent runs. The submission is `git diff <base>..HEAD`
        # (deep-swe/pier), and deep-swe agents COMMIT their own work — so a post-run working-tree
        # diff would be empty. Capturing base..HEAD reflects the work whether the agent committed it
        # or left it uncommitted (swe-bench). Empty (best-effort) if the workspace isn't a git repo.
        base = runtime.exec(
            handle, ["bash", "-lc", f"cd {repo} && git rev-parse HEAD 2>/dev/null || true"]).stdout.strip()
        # Write the trajectory into the harness's SYNCED agent-logs dir (``/logs/agent``, like
        # monet's stream) — pier/harbor sync it to the trial's ``agent/`` and the harness converts
        # it to ATIF post-job. (On the docker path ``/logs/agent`` is a throwaway; the harness reads
        # the captured text below and writes it into the run dir instead.)
        run_res = runtime.exec(
            handle,
            ["bash", "-lc", (
                f"mkdir -p /logs/agent && cd {repo} && "
                f"/agent/.venv/bin/mini -t {shlex.quote(task.prompt())} -m {shlex.quote(model)} "
                f"-y --exit-immediately --environment-class local --agent-class default "
                f"-c {shlex.quote(config_path)}{gw}{vocab}{ov} -o /logs/agent/mini.traj.json -l 0")],
            env=run_env,
            timeout=cfg.get("timeout", 1800),
        )
        # Capture the full submission = everything since `base`. Commit any uncommitted edits first,
        # then `git diff base..HEAD` — so the patch reflects BOTH the agent's own commits (deep-swe)
        # and uncommitted working-tree edits (swe-bench). Without a base commit (non-git workspace),
        # fall back to the working-tree diff.
        commit = (f'cd {repo} && git add -A && git -c user.email=agent@beagle.local '
                  f'-c user.name=beagle commit -q -m "beagle agent changes" || true')
        if base:
            runtime.exec(handle, ["bash", "-lc", commit])
            diff = runtime.exec(
                handle, ["bash", "-lc", f"cd {repo} && git diff {shlex.quote(base)}..HEAD"]).stdout
        else:
            diff = runtime.exec(
                handle, ["bash", "-lc", f"cd {repo} && git add -A && git diff --cached"]).stdout
            runtime.exec(handle, ["bash", "-lc", commit])
        # Read back the trajectory the CLI wrote (`-o`) → tokens + turns. Best-effort.
        traj_raw = runtime.exec(
            handle, ["bash", "-lc", "cat /logs/agent/mini.traj.json 2>/dev/null || true"]).stdout
        tokens, n_turns = _parse_trajectory(traj_raw)
        ref = TrajectoryRef(path=Path("mini.traj.json"), format="mini-swe")
        # A failed `mini` that ALSO produced no patch is a generation failure — surface it.
        if not run_res.ok and not (diff or "").strip():
            return TaskResult(
                task_id=task.task_id, status=RolloutStatus.FAILED, tokens=tokens, num_turns=n_turns,
                error=f"mini run failed (rc={run_res.returncode}): "
                      f"{(run_res.stderr or run_res.stdout).strip()[:800]}",
                trajectory=ref, trajectory_text=traj_raw or None)
        return TaskResult(
            task_id=task.task_id, status=RolloutStatus.COMPLETED, patch=diff or None,
            tokens=tokens, num_turns=n_turns, trajectory=ref, trajectory_text=traj_raw or None)

    def network_hosts(self) -> list[str]:
        """Only the LLM gateway is contacted during :meth:`run_in` — allowlisted on a restricted
        run phase (deep-swe/pier). Empty when no gateway is configured (litellm's own defaults)."""
        kw = gateway_litellm_kwargs()
        return [kw["api_base"]] if kw and kw.get("api_base") else []

    def install_hosts(self) -> list[str]:
        """Hosts :meth:`install` reaches: the fork's git host + the package indexes uv pulls from
        (its standalone installer, a managed CPython, and the editable deps). Allowlisted so the
        INSTALL phase can clone + build behind a filtered-egress benchmark's proxy."""
        from urllib.parse import urlparse

        hosts = [
            "github.com", "codeload.github.com", "objects.githubusercontent.com",  # clone + uv python
            "astral.sh", "releases.astral.sh", "raw.githubusercontent.com",        # uv installer
            "pypi.org", "files.pythonhosted.org",                                  # uv pip deps
        ]
        src_host = urlparse(self.source().repo).hostname          # a self-hosted fork host, if any
        if src_host:
            hosts.append(src_host)
        return hosts

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
        # mini-swe as a coding agent (evolver): run `mini` against the workspace
        # with the instruction as the task. Edits land in workspace git state.
        raise NotImplementedError("mini-swe edit() not yet implemented")


def _mini_gateway_c_args() -> str:
    """The shared litellm→gateway settings (:func:`gateway_litellm_kwargs`), formatted as mini's
    ``-c model.model_kwargs.…`` CLI overrides — ``""`` when no gateway is configured. The routing
    itself is reusable infra; only this CLI *formatting* is mini-swe-specific."""
    kw = gateway_litellm_kwargs()
    if not kw:
        return ""
    return "".join(f" -c model.model_kwargs.{k}={shlex.quote(str(v))}" for k, v in kw.items())


def _mini_vocab_c_args(cfg: dict) -> str:
    """The shared **first-level vocabulary** knobs formatted as mini's ``-c`` overrides, so a caller
    configures mini-swe with the same fields as any other agent (they layer over the ``-c`` preset):

    * ``effort``    → the **Responses API** model class + ``reasoning_effort``. A reasoning model
      returns its reasoning and the tool call as separate items on the Responses API
      (``litellm.responses`` → ``response.output``); via Chat Completions a gateway may instead
      split them into a second ``choices`` entry that mini's default chat class (which reads
      ``choices[0]``) never sees → "no tool calls found". So requesting reasoning also selects
      ``model.model_class=litellm_response`` (mini ships it upstream — we only *select* it). Left on
      the chat default when no effort is set.
    * ``max_turns`` → ``agent.step_limit`` (mini's per-run step cap; :class:`AgentConfig.step_limit`)

    ``""`` for any knob left unset. (``provider`` gates the gateway routing in :meth:`run_in`, not
    here — mini-swe reaches its LLM through the gateway's litellm ``api_base``, not a provider name.)
    mini json-decodes each value, so ``150`` types as ``int`` and ``high`` stays a string."""
    parts = []
    if cfg.get("effort"):
        parts.append(" -c model.model_class=litellm_response")
        parts.append(f" -c model.model_kwargs.reasoning_effort={shlex.quote(str(cfg['effort']))}")
    if cfg.get("max_turns") is not None:
        parts.append(f" -c agent.step_limit={shlex.quote(str(int(cfg['max_turns'])))}")
    return "".join(parts)


def _mini_prompt_override_c_args(override: dict[str, str]) -> str:
    """The optional layer-1/2 framing override (:meth:`Agent.prompt_override` →
    ``{system?, instruction?}``) formatted as mini's ``-c`` overrides — ``""`` when unset:

    * ``system``      → ``-c agent.system_template=…``   (layer 1)
    * ``instruction`` → ``-c agent.instance_template=…`` (layer 2 — this REPLACES mini's
      instance_template, so the value MUST keep the ``{{task}}`` placeholder or the task text
      is lost).

    The whole ``key=value`` is shlex-quoted as one token so multi-line prompt bodies survive; mini
    splits each ``-c`` spec on the first ``=`` (:func:`get_config_from_spec`)."""
    slots = {"system": "agent.system_template", "instruction": "agent.instance_template"}
    return "".join(
        f" -c {shlex.quote(f'{key}={override[name]}')}"
        for name, key in slots.items() if override.get(name))


def _parse_trajectory(raw: str) -> tuple[dict[str, int], int]:
    """Best-effort ``(tokens, num_turns)`` from a mini-swe trajectory JSON (the CLI's ``-o``
    output). mini stashes each LiteLLM response's usage under ``message["extra"]["response"]
    ["usage"]``; sum those and count assistant turns. Returns ``({}, 0)`` on any parse miss —
    the git-diff patch is authoritative, so a trajectory-parse miss never fails the run.

    LiteLLM is OpenAI-shaped: ``prompt_tokens`` INCLUDES ``prompt_tokens_details.cached_tokens``
    (a subset), so the fresh bucket is ``prompt_tokens - cached_tokens`` and cache-write isn't
    billed separately (0)."""
    import json

    try:
        doc = json.loads(raw)
    except (ValueError, TypeError):
        return {}, 0
    messages = doc.get("messages") if isinstance(doc, dict) else doc
    if not isinstance(messages, list):
        return {}, 0
    total = Usage()
    turns = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            turns += 1
        usage = ((m.get("extra") or {}).get("response") or {}).get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        total = usage_add(total, Usage(
            input_uncached=max(0, pt - cached),   # cached ⊂ prompt_tokens (OpenAI semantics)
            cache_read=cached,
            output=int(usage.get("completion_tokens") or 0),
        ))
    tokens = total.to_token_counts() if total != Usage() else {}
    return tokens, turns


__all__ = ["MiniSweAgent"]
