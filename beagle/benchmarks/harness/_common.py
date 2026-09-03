"""Framework-free helpers shared by the installed-agent shims (harbor + pier).

These carry **no** harbor/pier import, so either shim can use them without pulling in the other
framework — ``beagle[terminal-bench]`` (harbor) and ``beagle[deep-swe]`` (pier) install
independently. The shims themselves (:mod:`beagle.benchmarks._harbor_agent`,
:mod:`beagle.benchmarks._pier_agent`) only add the framework-specific base class.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any

from beagle.types import TaskContext

LOGGER = logging.getLogger(__name__)

#: Seconds reserved from the task's budget for the agent's POST-RUN CAPTURE, so it finishes before
#: the framework cancels the trial.
#:
#: What it pays for: after the agent process exits, every adapter runs a small fixed sequence of
#: container commands — ``git add -A`` + commit, ``git diff base..HEAD``, read back the trajectory
#: file. That is a handful of exec round-trips whose cost is a property of the *bookkeeping*, not of
#: the budget, which is why this is a flat reservation and NOT a fraction: 10% of a 6000 s task
#: would sacrifice ten minutes of agent time to buy seconds of work.
#:
#: Why 60: the sequence is 3-4 execs over an already-warm session, and a `git diff` on a SWE repo is
#: seconds. It is an estimate, not a measurement — harbor times the agent phase as a whole
#: (``agent_execution.started_at/finished_at``) and nothing on disk isolates the capture, so there
#: was nothing to fit it to. Both failure directions are mild and one is observable: too small
#: loses ``patch.diff`` (reward is the verifier's and survives; tokens are recovered post-job from
#: the native stream), too large costs the agent a minute. Replace it with a measured value if the
#: adapters ever time their own capture.
_POST_RUN_RESERVE_S = 60.0

# Distro-agnostic git bootstrap: harbor/pier task images are minimal and often lack git (and
# ca-certificates), which the agent needs to clone its own source. Runs as root in install().
# Retries the network-touching steps — a single CDN blip shouldn't kill a long trial before the
# agent ever runs.
_GIT_BOOTSTRAP = r"""
set -e
if ! command -v git >/dev/null 2>&1; then
  attempt=1
  while :; do
    if command -v apk >/dev/null 2>&1; then
      apk add --no-cache git ca-certificates && break
    elif command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates && break
    elif command -v microdnf >/dev/null 2>&1; then
      microdnf install -y git ca-certificates && break
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y git ca-certificates && break
    elif command -v yum >/dev/null 2>&1; then
      yum install -y git ca-certificates && break
    elif command -v zypper >/dev/null 2>&1; then
      zypper --non-interactive install git ca-certificates && break
    elif command -v pacman >/dev/null 2>&1; then
      pacman -Sy --noconfirm git ca-certificates && break
    else
      echo "no supported package manager (apk/apt/dnf/yum/zypper/pacman)" >&2; exit 1
    fi
    if [ "$attempt" -ge 3 ]; then echo "git bootstrap failed after 3 attempts" >&2; exit 1; fi
    sleep $((attempt * 5)); attempt=$((attempt + 1))
  done
fi
"""


#: Marker the workspace check echoes when the resolved dir really is a git worktree.
_REPO_OK = "BEAGLE_REPO_OK"


class WorkspaceSetupError(RuntimeError):
    """The benchmark's declared agent-phase workspace could not be established.

    Raised BEFORE the agent runs, so the trial is recorded as *errored* rather than as a
    reward-0 capability failure. Deliberately not in ``INFRA_RETRY_EXCEPTIONS``: a retry
    re-runs the same image and would only hide the same bad workspace.
    """


def workspace_probe_command(task_env: dict[str, str]) -> str:
    """The command whose stdout is the agent's working directory.

    A benchmark that declares nothing gets the framework's own ``pwd`` — unchanged behaviour for
    images that already set a WORKDIR.
    """
    return task_env.get("repo_path_cmd") or "pwd"


def workspace_check_command(repo_path: str, preamble: str) -> str:
    """One command that reports both halves of the contract, run UNDER the agent's own preamble so
    what is checked is what the agent will get.

    Echoes :data:`_REPO_OK` iff the resolved dir is a git worktree, then prints the interpreter
    that ``python`` resolves to. ``git rev-parse`` rather than ``[ -d .git ]``: a git WORKTREE's
    ``.git`` is a FILE, and the directory test would reject a perfectly good checkout.
    """
    return (f"{preamble}\n"
            f"cd {shlex.quote(repo_path)} 2>/dev/null && "
            f"git rev-parse --is-inside-work-tree >/dev/null 2>&1 && echo {_REPO_OK}\n"
            "command -v python 2>/dev/null || command -v python3 2>/dev/null || true\n")


def interpret_workspace_check(repo_path: str, stdout: str) -> dict[str, Any]:
    """Turn the check's output into trial metadata, or refuse to start.

    Two tiers, deliberately different:

    * **no git worktree → raise** :class:`WorkspaceSetupError`. Unambiguous: an agent with no repo
      cannot do the task, and staying quiet books it as a reward-0 capability failure — the
      measurement lie this whole seam exists to prevent.
    * **interpreter outside a virtual/conda env → record.** Staged: until the false-positive rate
      is measured across a corpus, arming this would turn red any task that legitimately installs
      into the base interpreter. It lands in ``result.json``, so the population is countable after
      one run.

    Framework-free on purpose — this is the decision logic both shims share, and gating it behind
    an optional extra would leave it unverified wherever that extra isn't installed.
    """
    out = (stdout or "").strip()
    python_path = next(
        (ln.strip() for ln in reversed(out.splitlines()) if ln.strip().startswith("/")), "")
    if _REPO_OK not in out:
        raise WorkspaceSetupError(
            f"no git worktree at the resolved workspace {repo_path!r}: the benchmark's "
            f"repo_path_cmd did not find the task repo, so the agent would work in the wrong "
            f"directory and score 0 for a reason that has nothing to do with the agent. "
            f"probe stdout={out!r}")
    # ``/envs/`` (conda) or ``/venv``/``/.venv`` (virtualenv) — an interpreter still on the base
    # image PATH means the preamble matched nothing.
    env_ok = any(m in python_path for m in ("/envs/", "/venv/", "/.venv/"))
    if not env_ok:
        LOGGER.warning(
            "workspace check: python resolved to %r, which is not inside a task environment — "
            "the agent may run a different interpreter than the verifier tests with",
            python_path or "<none>")
    return {"workspace": repo_path, "task_python": python_path, "task_env_active": env_ok}


def effective_agent_budget_s(trial_config_path: Path | str) -> float | None:
    """The agent-phase deadline the framework will actually enforce for this trial, or ``None``.

    MIRRORS harbor's ``Trial._compute_agent_timeout_sec``::

        base = agent.override_timeout_sec or <task.toml> [agent].timeout_sec
        min(base, agent.max_timeout_sec or inf) * (agent_timeout_multiplier or timeout_multiplier)

    Both inputs are on the HOST beside the shim: the trial's ``config.json`` (written by
    ``Trial._init_result``, the first statement of ``Trial.run``, so it exists before any agent
    runs) names the task dir, and the task dir holds ``task.toml``. ``None`` means the task
    declares no agent budget — the framework then imposes no deadline either, so there is nothing
    to derive from and the caller falls back to its own default.

    Deliberately a mirror, not a call: harbor never exposes the computed budget to an agent. The
    drift risk is one-directional — a stale mirror can only compute a budget that is *smaller*
    than the real one, which costs an agent some headroom but can never let it overrun the
    framework's own ``wait_for``. Any read/parse problem returns ``None`` rather than raising: an
    undiscoverable budget must fall back, never fail a trial.

    (``config.json`` may be dumped with ``exclude_defaults=True``, so an ABSENT multiplier means
    1.0 — not "unknown".)
    """
    try:
        cfg = json.loads(Path(trial_config_path).read_text(encoding="utf-8"))
        agent = cfg.get("agent") or {}
        base = agent.get("override_timeout_sec")
        if not isinstance(base, (int, float)) or base <= 0:
            task_path = (cfg.get("task") or {}).get("path")
            if not task_path:
                return None
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - <3.11 dev boxes
                import tomli as tomllib  # type: ignore[no-redef]
            task_toml = tomllib.loads(
                (Path(task_path) / "task.toml").read_text(encoding="utf-8"))
            base = (task_toml.get("agent") or {}).get("timeout_sec")
        if not isinstance(base, (int, float)) or base <= 0:
            return None
        cap = agent.get("max_timeout_sec")
        if isinstance(cap, (int, float)) and cap > 0:
            base = min(float(base), float(cap))
        mult = cfg.get("agent_timeout_multiplier")
        if not isinstance(mult, (int, float)) or mult <= 0:
            mult = cfg.get("timeout_multiplier")
        if not isinstance(mult, (int, float)) or mult <= 0:
            mult = 1.0
        return float(base) * float(mult)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as e:
        LOGGER.debug("could not resolve the trial's agent budget from %s: %r", trial_config_path, e)
        return None


def graceful_agent_timeout_s(budget_s: float | None, configured: float | None) -> float | None:
    """The agent's own clock: the task's budget less the post-run capture reserve.

    ``None`` budget (the task declares none, or it couldn't be read) → ``configured`` unchanged, so
    whatever the run config states remains the bound.

    The reserve is capped at half the budget — not a tuning knob, just the invariant that
    bookkeeping headroom can never cost more time than it protects. It only binds for budgets under
    two minutes, which no corpus here has (terminal-bench's smallest is 900 s, SWE-rebench's 3000 s).
    """
    if budget_s is None or budget_s <= 0:
        return configured
    return budget_s - min(_POST_RUN_RESERVE_S, budget_s / 2)


def declare_task_budget(
    task_ctx: TaskContext, trial_config_path: Path | str, *, spent_s: float = 0.0
) -> TaskContext:
    """Return ``task_ctx`` with :attr:`~beagle.types.TaskContext.agent_timeout_s` set from the
    trial's own task budget (less the post-run capture reserve, less ``spent_s``).

    ``spent_s`` is whatever the caller's phase has already consumed before handing the agent
    control — the framework's deadline started earlier than the agent does.

    Returns a NEW context: ``TaskContext`` is a frozen dataclass, so assigning to the field raises
    ``FrozenInstanceError`` — which would fail every trial that reached it.
    """
    budget = graceful_agent_timeout_s(effective_agent_budget_s(trial_config_path), None)
    return replace(task_ctx, agent_timeout_s=(
        None if budget is None else max(1.0, float(round(budget - spent_s)))))


def _rebuild_agent(identity: dict[str, Any]):
    """Reconstruct the beagle agent from its serializable identity descriptor."""
    from beagle.agents.core.registry import build
    from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec

    src = identity.get("source")
    spec = AgentSpec(
        name=identity["agent"],
        model=ModelSpec(name=identity["model"]) if identity.get("model") else None,
        config=dict(identity.get("config") or {}),
        source=AgentSource(**src) if src else None,
    )
    return build(spec)


__all__ = ["_GIT_BOOTSTRAP", "WorkspaceSetupError", "_rebuild_agent", "declare_task_budget",
           "effective_agent_budget_s", "graceful_agent_timeout_s",
           "interpret_workspace_check", "workspace_check_command", "workspace_probe_command"]
