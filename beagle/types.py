"""Core domain vocabulary shared across every beagle module.

These are the small, stable data types that flow between the dataloader, the
rollout runner, the agents, and the evolution algorithm. They are intentionally
benchmark-agnostic: a ``Task`` from SWE-bench and a ``Task`` from Terminal-Bench
look the same to the rest of the system, with benchmark-specific overflow tucked
into ``extras``.

Rich rollout traces (``Trajectory`` / ``Step``) are owned by **xrlenv** and left
in the benchmark's native format on disk. beagle references them by path
(:class:`TrajectoryRef`) rather than re-parsing, which is what keeps artifacts
drop-in compatible with the upstream harness. Parsing/normalization, when needed,
is a separate concern handled under :mod:`beagle.analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# --- lightweight aliases -----------------------------------------------------

TaskId = str
Reward = float
#: Token accounting with a cache-status breakdown, e.g. ``{"prompt": 1234, "completion": 567,
#: "total": 1801, "input_uncached": 800, "cache_read": 400, "cache_write": 34}``. ``prompt`` is the
#: total billable input and ``completion`` the output (both legacy, always present); the cache split
#: — ``input_uncached`` (fresh) + ``cache_read`` (hit) + ``cache_write`` (creation) = ``prompt`` — lets
#: a downstream cost estimate price each bucket separately. Built by ``agents/core/usage.py:Usage``.
TokenCounts = dict[str, int]


class RolloutStatus(str, Enum):
    """Terminal disposition of a single rollout.

    Mirrors ``xrlenv.RolloutStatus`` so beagle results can be reconciled with
    the substrate's own bookkeeping without a translation table.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    """The role an agent plays in an evolution run.

    Orthogonal to :class:`Transparency` — a white-box agent can be an evolvee and
    a black-box agent can be an evolver; these are just the common pairings.
    """

    #: The harness being improved; it is rolled out on benchmark tasks and scored.
    EVOLVEE = "evolvee"
    #: The coding agent that proposes edits to the evolvee's harness.
    EVOLVER = "evolver"


class Transparency(str, Enum):
    """How much of an agent's implementation beagle controls."""

    #: We own the harness source and can mutate it file-by-file.
    WHITE_BOX = "white_box"
    #: External CLI / opaque internals; only prompt, flags, and config are tunable.
    BLACK_BOX = "black_box"


# --- tasks -------------------------------------------------------------------


@dataclass
class Task:
    """A single benchmark task, normalized across benchmarks.

    Attributes
    ----------
    task_id:
        Unique identifier, e.g. ``"astropy__astropy-12345"`` or a harbor task
        name. Must be stable — it keys resume, dedup, and scoring.
    problem_statement:
        The **raw** core task text, straight from the benchmark. Never framed or
        overwritten — a benchmark's optional pre/post info hooks wrap *around* it
        into :attr:`instruction`; this field stays the bare task.
    instruction:
        The assembled **data payload** handed to the agent — the benchmark's
        ``additional_info_pre`` + :attr:`problem_statement` + ``additional_info_post``
        (see :meth:`beagle.benchmarks.base.Benchmark.load_tasks`). It carries **data
        only** (task + benchmark-supplied facts), never framing/how-to-work prose —
        that is the agent's own (its system prompt + generic instruction). Empty for a
        directly-constructed task; use :meth:`prompt` to read it with a raw fallback.
    repo_url, base_commit:
        Populated for repo-shaped benchmarks (SWE-bench family); empty strings
        for self-contained container benchmarks (Terminal-Bench, WebArena).
    benchmark:
        Canonical benchmark name this task came from (set by the loader). Lets a
        mixed dataset remember each task's origin for grading and weighting.
    extras:
        Benchmark-specific overflow (hints, requirements, harbor task dir, ...).
        Reserved keys are documented per-loader in :mod:`beagle.benchmarks`.
    """

    task_id: TaskId
    problem_statement: str = ""
    instruction: str = ""
    repo_url: str = ""
    base_commit: str = ""
    benchmark: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    def prompt(self) -> str:
        """The task text to hand the agent: the assembled :attr:`instruction`
        payload when a benchmark built one via ``load_tasks``, else the raw
        :attr:`problem_statement`. DATA only — the agent supplies its own framing."""
        return self.instruction or self.problem_statement


@dataclass(frozen=True)
class TaskContext:
    """Everything the runtime needs to stand up a task's container.

    This is the bridge between a benchmark and the rollout substrate: the loader
    produces it, and the runner hands it to xrlenv to acquire a container that
    matches the benchmark's native environment.
    """

    image: str | None
    repo_path: str = ""
    shell_preamble: str = ""
    benchmark_name: str = ""


@dataclass
class TrajectoryRef:
    """A pointer to a rollout trace left on disk in the benchmark's native format.

    beagle deliberately does **not** re-serialize trajectories into a house
    format; it records where the upstream harness wrote them so downstream
    analysis can read them in place.
    """

    path: Path
    #: e.g. ``"harbor-trial"``, ``"monet-stream-json"``, ``"swebench-run"``.
    format: str = ""


@dataclass
class TaskResult:
    """Outcome of running one agent on one task.

    ``artifact_dir`` points at the benchmark's native output directory so results
    reference native artifacts rather than a re-serialized copy.
    """

    task_id: TaskId
    #: Benchmark this task came from (set by the harness from ``Task.benchmark``); lets a
    #: grader/persistence derive the task's artifact subtree without a separate lookup.
    benchmark: str = ""
    status: RolloutStatus = RolloutStatus.PENDING
    #: Passed the benchmark's grader (or in-band reward >= threshold).
    resolved: bool = False
    #: Patch applied cleanly to base_commit (SWE-bench family); set by grader.
    applied: bool = False
    #: Numeric reward for in-band-graded benchmarks; ``None`` for patch graders.
    reward: Reward | None = None
    #: Unified diff produced by the agent, when the benchmark is patch-graded.
    patch: str | None = None
    num_turns: int = 0
    #: Total agent wall-clock (acquire + install + run), in seconds. 0.0 when untimed.
    duration_sec: float = 0.0
    #: Per-phase wall-clock spans for a per-task harness (docker drop-in), mirroring harbor's native
    #: trial timing so swe-bench result.json is comparable: ``{phase: {started_at, finished_at}}`` with
    #: ISO-8601/Z timestamps. Phases: ``environment_setup`` (container acquire), ``agent_setup``
    #: (install/clone+build), ``agent_execution`` (run_in). Empty on the harbor/pier path (harbor
    #: writes its own native timing) or when untimed.
    timing: dict[str, dict[str, str]] = field(default_factory=dict)
    tokens: TokenCounts = field(default_factory=dict)
    #: Non-``None`` describes a timeout / crash / infra failure.
    error: str | None = None
    #: Native artifact directory for this task (harbor trial dir, or the swebench
    #: ``run_dir/<benchmark>/<task_id>/`` subtree the per-task harness writes).
    artifact_dir: Path | None = None
    raw_log_path: Path | None = None
    trajectory: TrajectoryRef | None = None
    #: Raw native trajectory CONTENT captured in-rollout (e.g. mini-swe's ``mini.traj.json``),
    #: for harnesses whose container is torn down before the host can sync it. The per-task
    #: harness writes it to ``<artifact_dir>/agent/`` and converts it to ATIF. ``None`` when the
    #: native stream is already on disk (harbor syncs it) or the agent produced none.
    trajectory_text: str | None = None
    #: Full agent-process stderr, captured on FAILURE so the real cause survives (the one-line
    #: ``error`` above is only its first meaningful line). Persisted to ``<artifact_dir>/agent/
    #: stderr.log`` by the per-task harness. ``None`` on success / when there's nothing to keep.
    stderr_text: str | None = None


__all__ = [
    "TaskId",
    "Reward",
    "TokenCounts",
    "RolloutStatus",
    "AgentRole",
    "Transparency",
    "Task",
    "TaskContext",
    "TrajectoryRef",
    "TaskResult",
]
