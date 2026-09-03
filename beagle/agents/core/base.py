"""Agent base class and composable capability mixins.

An agent's **role** in a run (evolvee vs evolver) is the *user's choice*, not an
intrinsic property — monet, terminus, and mini-swe can each serve as either. So
role is never baked into the class. Instead, an agent declares the **capabilities**
it intrinsically has by mixing them in, and the trainer assigns a role that its
capabilities support.

Three capabilities, composed freely:

* :class:`Runnable` — implements :meth:`run`; the agent can attempt benchmark
  tasks. Declares a :class:`Topology` (where the agent process runs).
* :class:`Editor` — implements :meth:`edit`; the agent is a coding agent that runs
  an *instruction* against a *workspace* and edits it. This is the thin primitive
  an **evolver** exposes. It is deliberately minimal: one instruction, one
  workspace, one result. The *evolution algorithm* owns the recipe — it decides
  the prompts (analyze / implement / review / …), how many times to call
  :meth:`edit`, and what to interleave (mini-evals, guards, verdicts). Nothing
  about that recipe is baked into the agent contract.
* :class:`Evolvable` — is parameterized by an :class:`AgentSource` (a git repo @ a
  ref) it exposes and can rebind; the source is θ, the thing being evolved.

**AgentSource is the key concept.** An evolvable agent's code lives in an external
git repo (its own repo — for open-source agents, a fork we control so evolved
branches have a home). A *run* pins a specific ``ref``: a baseline ref is
un-evolved; a candidate ref/branch produced by an evolver is evolved. Same repo,
different ref. beagle stores only the thin adapter plus a default repo URL.

Note the unification: :meth:`Runnable.run` (a coding agent on a benchmark task in a
container) and :meth:`Editor.edit` (a coding agent on an instruction in a
workspace) are two uses of the same thing. White-box agents have both; external
CLIs have only :meth:`edit`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from beagle.agents.core.spec import AgentSource, AgentSpec
from beagle.types import AgentRole, RolloutStatus, Task, TaskContext, TaskResult, Transparency

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from beagle.rollout.binding import RolloutBinding
    from beagle.rollout.runtime import ContainerRuntime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    """ISO-8601 with a trailing ``Z`` — the shape harbor uses for its trial timing timestamps."""
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


# --- the coding-agent primitive's result ------------------------------------


@dataclass
class EditResult:
    """What one :meth:`Editor.edit` call returns — mirrors a real meta-agent run.

    Deliberately minimal: the final message, resource accounting, and success. The
    *edits themselves* are left in the workspace (git state); the algorithm reads
    the diff/commits from the worktree after the call, exactly as a real proposer
    loop does. No mutation/feedback types are imposed on the agent.
    """

    text: str = ""
    exit_code: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0
    log_path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.error is None


# --- capabilities ------------------------------------------------------------


class Capability(str, Enum):
    """What an agent can intrinsically do (mixed in, not assigned)."""

    ROLLOUT = "rollout"      # Runnable: can attempt benchmark tasks
    EDIT = "edit"            # Editor: can run a coding instruction against a workspace
    EVOLVABLE = "evolvable"  # Evolvable: parameterized by a mutable AgentSource


class Topology(str, Enum):
    """Where an agent's process runs relative to the task environment."""

    #: Agent is installed and runs *inside* the task container (e.g. monet).
    #: Implemented first.
    IN_CONTAINER = "in_container"
    #: Agent runs on the host/orchestrator and drives the environment via
    #: ``runtime.exec`` (e.g. mini-swe's native docker environment). Designed for.
    HOST_DRIVER = "host_driver"
    #: Agent runs in a remote/cloud service. Designed for.
    REMOTE = "remote"


class AgentBudgetUndeclared(RuntimeError):
    """No one stated how long the agent may run — raised instead of inventing a number.

    There is deliberately **no default wall clock** in beagle. A house constant is invisible in
    the run config, outranks whatever the benchmark actually declared, and truncates every task
    worth more than the guess — which is exactly what happened when four adapters each hardcoded
    1800 s while SWE-rebench tasks shipped budgets of 3000 s and 6000 s. An unstated budget is a
    configuration error, and it says so at the first rollout rather than silently capping a
    campaign.
    """


def resolve_agent_timeout(config: dict[str, Any], task_ctx: TaskContext) -> float:
    """The wall clock for one rollout. **The benchmark's own budget wins; the config is the guard.**

    1. ``task_ctx.agent_timeout_s`` — what the task/benchmark declared (see
       :attr:`~beagle.types.TaskContext.agent_timeout_s`). On the harbor/pier path the shim
       resolves it per trial from ``task.toml``, so a corpus with heterogeneous budgets
       (SWE-rebench: 3000 s, four tasks at 6000 s) is honoured task by task.
    2. ``config["timeout"]`` — the run config's ``agent.timeout``. It applies **only when nothing
       was declared** (the docker-drop-in path, whose harness has no deadline of its own). It is
       NOT a ceiling on a declared budget: the generated configs carry an explicit fallback value,
       so treating it as a ceiling would truncate every harbor task right back to that number —
       the bug this precedence exists to prevent. To shorten a declared budget, scale it with
       ``run.timeout_multiplier``, which the harness applies to the task's own value.
    3. Neither → :class:`AgentBudgetUndeclared`.

    Editor/evolver work (``Editor.edit``) is not a rollout and doesn't come through here — it
    carries its own operation timeout.
    """
    declared = task_ctx.agent_timeout_s if task_ctx is not None else None
    configured = (config or {}).get("timeout")
    if declared is not None:
        if configured is not None and float(configured) != float(declared):
            # DEBUG, not INFO: the generated configs always carry the fallback and harbor tasks
            # always declare, so this is the normal case — at INFO it would be one line per trial
            # (~860 on a full SWE-rebench run) saying nothing went wrong.
            LOGGER.debug(
                "agent clock: using the task's declared %.0fs budget; the run config's "
                "agent.timeout=%s applies only to benchmarks that declare none (scale a declared "
                "budget with run.timeout_multiplier instead)", float(declared), configured)
        return float(declared)
    if configured is not None:
        return float(configured)
    raise AgentBudgetUndeclared(
        f"no agent time budget for benchmark {task_ctx.benchmark_name or '<unknown>'!r}: the "
        "task/harness declared none (TaskContext.agent_timeout_s) and the run config sets no "
        "`agent.timeout`. On the harbor/pier path this means the task's task.toml has no "
        "[agent] timeout_sec; on a docker-drop-in benchmark the benchmark must declare one. "
        "beagle does not substitute a hidden default — set `agent.timeout` (the generated configs "
        "carry it explicitly) or declare the budget on the benchmark.")


class AgentInstallError(RuntimeError):
    """Raised by :meth:`Runnable.install` when the agent can't be materialized in the container
    (clone/build failure, missing credential). The default :meth:`Runnable.run` turns it into a
    ``FAILED`` :class:`TaskResult`; a network-phased harness surfaces it as an install-phase error."""


class Runnable(ABC):
    """Capability: attempt benchmark tasks (be rolled out and scored).

    An installed agent has a two-phase lifecycle so a **network-phased** harness (pier/harbor) can
    open the network for the trusted *install* and lock it to the LLM endpoint for the *run*:

    * :meth:`install` — clone + build the agent into the container (internet-allowed phase);
    * :meth:`run_in` — run the (installed) agent (network restricted to :meth:`network_hosts`).

    Implement those two (+ optionally :meth:`network_hosts`) and inherit the default :meth:`run`,
    which composes them for an always-open harness (docker drop-in). A monolithic agent may instead
    override :meth:`run` directly.
    """

    #: Where this agent runs. The runner materializes the source accordingly.
    topology: ClassVar[Topology] = Topology.IN_CONTAINER

    def run(self, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> TaskResult:
        """Default combined lifecycle: acquire a container, :meth:`install` the agent, :meth:`run_in`
        it, tear down. Used by always-open harnesses (docker drop-in) where install + run share one
        network. A network-phased harness (pier/harbor) does NOT call this — it calls :meth:`install`
        and :meth:`run_in` across its own phases.

        This is the **single run() seam**: it records per-phase wall-clock (``environment_setup`` =
        acquire, ``agent_setup`` = install, ``agent_execution`` = run_in) onto the result so a per-task
        harness (docker drop-in / swe-bench) gets the harbor-shaped timing breakdown + ``duration_sec``
        that harbor writes natively. An agent tweaks the acquire (e.g. clear the image ENTRYPOINT) via
        :meth:`_acquire_run_args` rather than overriding this whole method.
        """
        timing: dict[str, dict[str, str]] = {}

        def _span(name: str, start: datetime, end: datetime) -> None:
            timing[name] = {"started_at": _iso(start), "finished_at": _iso(end)}

        run_args = self._acquire_run_args()
        t0 = _utcnow()
        handle = runtime.acquire(image=task_ctx.image or "", command=["sleep", "infinity"],
                                 run_args=run_args or None)
        t1 = _utcnow()
        _span("environment_setup", t0, t1)
        try:
            try:
                self.install(handle, task_ctx, runtime=runtime)
            except AgentInstallError as e:
                t2 = _utcnow()
                _span("agent_setup", t1, t2)
                res = self._install_error_result(task, str(e))
                res.timing, res.duration_sec = timing, (t2 - t0).total_seconds()
                return res
            t2 = _utcnow()
            _span("agent_setup", t1, t2)
            # A declared budget covers this WHOLE call (harbor's agent deadline starts when the
            # shim hands us control, and the agent's own clone/build happens in install() above),
            # so run_in only gets what acquire+install left. Without this an agent that installs
            # slowly then runs to its full clock overruns the framework and is cancelled mid-
            # capture — losing the patch. Floor at 1 s so a budget already spent fails fast and
            # visibly rather than running unbounded.
            if task_ctx.agent_timeout_s is not None:
                spent = (t2 - t0).total_seconds()
                # Whole seconds: a timeout is second-granularity anyway, and an unrounded remainder
                # puts six decimals of wall-clock noise into every artifact and error message.
                task_ctx = replace(task_ctx, agent_timeout_s=max(
                    1.0, float(round(task_ctx.agent_timeout_s - spent))))
            res = self.run_in(handle, task, task_ctx, runtime=runtime)
            t3 = _utcnow()
            _span("agent_execution", t2, t3)
            res.timing, res.duration_sec = timing, (t3 - t0).total_seconds()
            return res
        finally:
            runtime.destroy(handle)

    def _acquire_run_args(self) -> list[str]:
        """Extra docker ``run`` args for the container acquire (default: none). monet/opencode clear
        the image ENTRYPOINT (``["--entrypoint", ""]``) so ``sleep infinity`` runs; ignored by harbor."""
        return []

    def _install_error_result(self, task: Task, message: str) -> TaskResult:
        """The TaskResult for an :class:`AgentInstallError` in :meth:`run` (a broken clone/build).
        Default: a bare FAILED result. An agent overrides to attach its native trajectory ref."""
        return TaskResult(task_id=task.task_id, status=RolloutStatus.FAILED, error=message)

    def install(self, handle: Any, task_ctx: TaskContext, *, runtime: ContainerRuntime) -> None:
        """Install the agent into an already-acquired ``handle`` (clone + build) — the INSTALL phase
        (network open on a phased harness). Default: nothing to install. Raise
        :class:`AgentInstallError` to fail the rollout with a clear message."""
        return None

    def run_in(
        self, handle: Any, task: Task, task_ctx: TaskContext, *, runtime: ContainerRuntime
    ) -> TaskResult:
        """Run the already-installed agent in ``handle`` and return its result — the RUN phase
        (network restricted to :meth:`network_hosts`). The **caller** owns the container lifecycle
        (no acquire/destroy here). Required unless :meth:`run` is overridden. Leave native artifacts
        on disk and set :attr:`TaskResult.artifact_dir` / ``trajectory``."""
        raise NotImplementedError("implement run_in() (or override run())")

    def network_hosts(self) -> list[str]:
        """URLs/hosts the agent must reach during :meth:`run_in` (e.g. the LLM gateway) — a
        network-restricted run phase allowlists exactly these. Default: none."""
        return []

    def install_hosts(self) -> list[str]:
        """URLs/hosts the agent must reach during :meth:`install` (its source host + package
        indexes). On a filtered-egress benchmark (pier's Squid allowlist), these are added to the
        allowlist so the INSTALL phase can clone + build. Pier applies ONE allowlist for the trial,
        so these stay reachable during run too — true per-phase narrowing (cut install hosts at run)
        is a pier-adapter follow-up (reload Squid post-install, like harbor's open-setup→tighten).
        Default: none."""
        return []

    def rollout_binding(self, task_ctx: TaskContext) -> RolloutBinding:
        """How the harness should run this agent. Defaults to wrapping :meth:`run`.

        Override only to return a native binding (e.g. a harbor ``import_path``).
        """
        from beagle.rollout.binding import GenericBinding

        return GenericBinding(run=self.run)


class Editor(ABC):
    """Capability: run a coding instruction against a workspace and edit it.

    The thin primitive an evolver exposes. The evolution algorithm calls this as
    many times as it likes, with its own prompts, interleaving anything it wants
    between calls. The agent does not know about "analyze/implement/review" — that
    is the algorithm's recipe.
    """

    @abstractmethod
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
        """Run ``instruction`` in ``workspace`` and return the result.

        ``plan_mode=True`` runs read-only (for analysis/planning). Edits are left in
        ``workspace`` (the caller reads git state). This mirrors a real meta-agent CLI
        invocation (``cursor-agent --workspace … [--mode plan]``). ``log_path``, when given,
        is where the backend's raw stream/log is tee'd (a proposer loop asks for a per-stage
        log there).
        """
        raise NotImplementedError


class Evolvable(ABC):
    """Capability: be parameterized by a mutable :class:`AgentSource` (white-box only).

    ``source()`` returns the current version (reflecting any bound candidate);
    ``with_source()`` rebinds to a specific ref so the runner installs/invokes
    *that* version. Concrete agents implement :meth:`_default_source`.
    """

    @abstractmethod
    def _default_source(self) -> AgentSource:
        """The agent's baseline source (repo + default ref + entrypoint)."""
        raise NotImplementedError

    def source(self) -> AgentSource:
        """The current source version — a bound candidate if set, else the baseline."""
        return getattr(self, "_bound_source", None) or self._default_source()

    def with_source(self, source: AgentSource) -> Evolvable:
        """Return a copy of this agent pinned to a specific source version."""
        import copy

        clone = copy.copy(self)
        clone._bound_source = source  # type: ignore[attr-defined]
        return clone


# --- base agent -------------------------------------------------------------


class Agent(ABC):
    """Identity + configuration for one agent.

    Compose with the capability mixins the agent supports, e.g.
    ``class MyAgent(Agent, Runnable, Evolvable, Editor)``.
    """

    #: Registry name; stamped by the ``@register`` decorator, overridable by spec.
    NAME: ClassVar[str] = ""
    #: Default source repo (git URL) for evolvable agents — the one config datum
    #: beagle keeps about an external agent; a run overrides the ref, not the repo.
    REPO: ClassVar[str] = ""
    #: Transparency axis. WHITE_BOX means the source is ours to mutate; only
    #: white-box agents can be :class:`Evolvable`.
    transparency: ClassVar[Transparency] = Transparency.WHITE_BOX

    def __init__(self, spec: AgentSpec | None = None) -> None:
        self.spec = spec or AgentSpec(name=self.NAME)

    @property
    def name(self) -> str:
        return self.spec.name or self.NAME

    @property
    def role(self) -> AgentRole | None:
        """The role assigned for this run (usage), if any — not intrinsic."""
        return self.spec.role

    @property
    def config(self) -> dict[str, Any]:
        """Agent-specific config (the free-form dict from the spec)."""
        return self.spec.config

    def prompt_override(self) -> dict[str, str]:
        """Optional config-level replacement for the agent's OWN framing — its layer-1 system
        prompt and/or layer-2 generic instruction — as ``{"system": ..., "instruction": ...}``
        (either key optional; empty by default). An **escape hatch for eval/ablation**, set via the
        role block's ``prompt_override``: it replaces framing that normally lives in the agent's
        source, so using it during evolution decouples what runs from what the evolver edits — keep
        it to evaluation. **Best-effort and uniform**: this reads the knob for every agent, but only
        agents whose adapter applies it (a config-driven harness like mini-swe) honor it; others
        ignore it. See ``notes/task-prompt-injection.md``."""
        return {k: str(v) for k, v in (self.config.get("prompt_override") or {}).items() if v}

    def installed_version(self) -> str | None:
        """The agent's **installed** version, for a config version gate — or ``None`` if the agent
        has no checkable installed version (the default).

        Black-box agents backed by an external binary (e.g. cursor) override this to return the
        binary's reported version, so a run can pin an expected version and fail loud on drift.
        Source-versioned agents (e.g. monet, pinned by ``source.ref``) return ``None`` and are
        exempt — their "version" is the git ref, not an installed binary."""
        return None

    @property
    def capabilities(self) -> set[Capability]:
        caps: set[Capability] = set()
        if isinstance(self, Runnable):
            caps.add(Capability.ROLLOUT)
        if isinstance(self, Editor):
            caps.add(Capability.EDIT)
        if isinstance(self, Evolvable):
            caps.add(Capability.EVOLVABLE)
        return caps

    def can_be_evolvee(self) -> bool:
        """Usable as an evolvee: rollout-able and its source can be mutated."""
        return isinstance(self, Runnable) and isinstance(self, Evolvable)

    def can_be_evolver(self) -> bool:
        """Usable as an evolver: can run coding instructions against a workspace."""
        return isinstance(self, Editor)

    def describe(self) -> str:
        caps = "+".join(sorted(c.value for c in self.capabilities)) or "none"
        return f"{self.name} [{self.transparency.value}; {caps}]"


__all__ = [
    "AgentSource",
    "EditResult",
    "Capability",
    "Topology",
    "AgentBudgetUndeclared",
    "resolve_agent_timeout",
    "Runnable",
    "AgentInstallError",
    "Editor",
    "Evolvable",
    "Agent",
]
